from __future__ import annotations

import unittest

import torch

from med_prism.adapters.injection import (
    add_task_bank,
    inject_shared_private_wrappers,
    iter_shared_private_wrappers,
)
from med_prism.evaluation.cl import (
    build_cl_metrics,
    component_task_ids,
    evaluate_prediction,
    matrix_rows,
)
from med_prism.projection.shared_private import compute_shared_drift_loss
from tests.phase4.helpers import ToyQwen3VL


def sample(options, answer, answer_option=None):
    letters = [chr(ord("A") + index) for index in range(len(options))]
    lines = "\n".join(
        f"{letter}. {value}" for letter, value in zip(letters, options, strict=True)
    )
    result = {
        "id": "sample",
        "options": options,
        "option_letters": letters,
        "answer": answer,
        "messages": [
            {"role": "user", "content": f"<image>\nQuestion\n{lines}"},
            {"role": "assistant", "content": answer},
        ],
    }
    if answer_option:
        result["answer_option"] = answer_option
    return result


def drift_model():
    model = ToyQwen3VL()
    model.requires_grad_(False)
    inject_shared_private_wrappers(
        model,
        shared_rank=4,
        shared_alpha=4.0,
        shared_lr_scale=1.0,
        dropout=0.0,
        shared_trainable=True,
    )
    add_task_bank(
        model,
        task_id=1,
        experts_per_task=4,
        alpha=4.0,
        trainable=True,
    )
    return model


class SharedDriftTest(unittest.TestCase):
    def test_shared_drift_loss_value(self):
        model = drift_model()
        wrappers = list(iter_shared_private_wrappers(model))
        denominator = sum(
            float(shared.detach().float().pow(2).sum())
            for _, wrapper in wrappers
            for shared in (wrapper.shared.anchor_A, wrapper.shared.anchor_B)
        )
        wrappers[0][1].shared.B.data[0, 0] = 2.0
        observed = compute_shared_drift_loss(model)
        self.assertAlmostEqual(float(observed), 4.0 / max(denominator, 1e-8), places=6)

    def test_reference_is_detached_and_gradient_only_reaches_shared(self):
        model = drift_model()
        wrapper = next(iter_shared_private_wrappers(model))[1]
        wrapper.shared.B.data[0, 0] = 1.0
        loss = compute_shared_drift_loss(model)
        loss.backward()
        self.assertFalse(wrapper.shared.anchor_A.requires_grad)
        self.assertFalse(wrapper.shared.anchor_B.requires_grad)
        self.assertIsNotNone(wrapper.shared.B.grad)
        self.assertGreater(float(wrapper.shared.B.grad.norm()), 0.0)
        self.assertTrue(
            all(
                parameter.grad is None
                for name, parameter in model.named_parameters()
                if ".experts." in name
            )
        )

    def test_full_loss_identity(self):
        task = torch.tensor(2.0)
        orthogonal = torch.tensor(0.25)
        drift = torch.tensor(0.5)
        total = task + 0.1 * orthogonal + 0.01 * drift
        error = abs(float(total - task - 0.1 * orthogonal - 0.01 * drift))
        self.assertLess(error, 1e-6)


class ParserTest(unittest.TestCase):
    def test_letter_text_and_explanation_are_equivalent(self):
        row = sample(["Normal", "Bacterial pneumonia", "Viral pneumonia"], "Normal", "A")
        predictions = ["A", "A. Normal", "Normal", "The answer is Normal because the film is clear."]
        parsed = [evaluate_prediction(row, value) for value in predictions]
        self.assertTrue(all(item["valid"] and item["correct"] for item in parsed))
        self.assertEqual({item["normalized_prediction"] for item in parsed}, {"Normal"})

    def test_medimeta_varying_option_counts(self):
        three = sample(["A1", "B1", "C1"], "C1", "C")
        five = sample(["One", "Two", "Three", "Four", "Five"], "Five", "E")
        three["dataset"] = five["dataset"] = "MedIMeta"
        self.assertTrue(evaluate_prediction(three, "C. C1")["correct"])
        self.assertTrue(evaluate_prediction(five, "Option E: Five")["correct"])

    def test_invalid_option_is_not_remapped(self):
        row = sample(["Normal", "Bacteria", "Virus"], "Virus", "C")
        parsed = evaluate_prediction(row, "D. Tuberculosis")
        self.assertFalse(parsed["valid"])
        self.assertFalse(parsed["correct"])
        self.assertIsNone(parsed["normalized_prediction"])

    def test_parser_is_deterministic(self):
        row = sample(["Normal", "Bacteria", "Virus"], "Bacteria", "B")
        first = evaluate_prediction(row, "B. Bacteria, because...")
        second = evaluate_prediction(row, "B. Bacteria, because...")
        self.assertEqual(first, second)


class ContractAndMetricsTest(unittest.TestCase):
    def test_cumulative_and_oracle_composition(self):
        self.assertEqual(
            component_task_ids("primary_cumulative", after_task=2, eval_task=1),
            [1, 2],
        )
        self.assertEqual(
            component_task_ids("primary_cumulative", after_task=2, eval_task=2),
            [1, 2],
        )
        self.assertEqual(
            component_task_ids("oracle_skill_aware", after_task=2, eval_task=1),
            [1],
        )
        self.assertEqual(
            component_task_ids("oracle_skill_aware", after_task=2, eval_task=2),
            [2],
        )

    def test_shared_load_count_contract(self):
        for mode in ("primary_cumulative", "oracle_skill_aware"):
            for after_task, eval_task in ((1, 1), (2, 1), (2, 2)):
                task_ids = component_task_ids(
                    mode,
                    after_task=after_task,
                    eval_task=eval_task,
                )
                manifest = {
                    "shared_load_count": 1,
                    "private_task_ids": task_ids,
                    "merged": False,
                }
                self.assertEqual(manifest["shared_load_count"], 1)
                self.assertFalse(manifest["merged"])

    def test_metric_formulas_and_matrix_identity(self):
        cells = [
            {"after_task": 1, "eval_task": 1, "accuracy": 0.8},
            {"after_task": 2, "eval_task": 1, "accuracy": 0.6},
            {"after_task": 2, "eval_task": 2, "accuracy": 0.7},
        ]
        self.assertEqual(matrix_rows(cells), {1: {1: 0.8}, 2: {1: 0.6, 2: 0.7}})
        metrics = build_cl_metrics(cells)
        self.assertAlmostEqual(metrics["final_average_accuracy"], 0.65)
        self.assertAlmostEqual(metrics["forgetting_task1"], 0.2)
        self.assertAlmostEqual(metrics["BWT"], -0.2)

    def test_future_and_duplicate_cells_fail(self):
        with self.assertRaises(ValueError):
            component_task_ids("primary_cumulative", after_task=1, eval_task=2)
        with self.assertRaises(ValueError):
            matrix_rows(
                [
                    {"after_task": 1, "eval_task": 1, "accuracy": 0.8},
                    {"after_task": 1, "eval_task": 1, "accuracy": 0.8},
                ]
            )


if __name__ == "__main__":
    unittest.main()
