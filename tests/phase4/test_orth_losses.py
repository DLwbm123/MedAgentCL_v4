import math
import unittest

import torch

from med_prism.adapters.injection import freeze_except_task, iter_rank1_wrappers
from med_prism.projection.orth_losses import (
    compute_rank1_orth_loss,
    rank1_overlap_squared,
)
from tests.phase4.helpers import make_bank, set_expert


class OrthLossTest(unittest.TestCase):
    def test_fixed_tensor_squared_and_rms_values(self):
        current_a = torch.tensor([[1.0, 0.0]], requires_grad=True)
        current_b = torch.tensor([[1.0, 0.0]], requires_grad=True)
        old_a = torch.tensor([[1.0, 0.0]], requires_grad=True)
        old_b = torch.tensor([[1.0, 0.0]], requires_grad=True)
        overlap = rank1_overlap_squared(current_a, current_b, old_a, old_b)
        torch.testing.assert_close(overlap, torch.ones(1, 1))
        squared = overlap.mean()
        rms = torch.sqrt(squared + 1e-8)
        self.assertAlmostEqual(float(squared), 1.0)
        self.assertAlmostEqual(float(rms), math.sqrt(1.0 + 1e-8), places=6)

    def test_task1_has_exact_zero_orth_loss_with_gradient_path(self):
        model = make_bank(((1, 4, 4.0, True),))
        result = compute_rank1_orth_loss(
            model, current_task_id=1, loss_type="rms", eps=1e-8
        )
        self.assertEqual(float(result.loss), 0.0)
        self.assertEqual(result.pair_count, 0)
        result.loss.backward()

    def test_task2_rms_loss_backpropagates_only_to_current_experts(self):
        model = make_bank(
            ((1, 1, 1.0, False), (2, 1, 1.0, True)), width=2
        )
        freeze_except_task(model, 2)
        for _, wrapper in iter_rank1_wrappers(model):
            set_expert(wrapper.task_experts(1)[0], [1.0, 0.0], [1.0, 0.0])
            set_expert(wrapper.task_experts(2)[0], [1.0, 1.0], [1.0, 1.0])
        result = compute_rank1_orth_loss(
            model, current_task_id=2, loss_type="rms", eps=1e-8
        )
        self.assertEqual(result.layer_count, 72)
        self.assertEqual(result.pair_count, 72)
        self.assertGreater(float(result.loss), 0.0)
        result.loss.backward()
        current_grads = []
        old_grads = []
        for _, wrapper in iter_rank1_wrappers(model):
            current = wrapper.task_experts(2)[0]
            old = wrapper.task_experts(1)[0]
            current_grads.extend([current.A.grad, current.B.grad])
            old_grads.extend([old.A.grad, old.B.grad])
        self.assertTrue(all(grad is not None for grad in current_grads))
        self.assertTrue(any(float(grad.norm()) > 0 for grad in current_grads))
        self.assertTrue(all(grad is None for grad in old_grads))

    def test_squared_and_rms_relationship(self):
        model = make_bank(
            ((1, 1, 1.0, False), (2, 1, 1.0, True)), width=2
        )
        for _, wrapper in iter_rank1_wrappers(model):
            set_expert(wrapper.task_experts(1)[0], [1.0, 0.0], [1.0, 0.0])
            set_expert(wrapper.task_experts(2)[0], [1.0, 1.0], [1.0, 1.0])
        squared = compute_rank1_orth_loss(
            model, current_task_id=2, loss_type="squared", eps=1e-8
        )
        rms = compute_rank1_orth_loss(
            model, current_task_id=2, loss_type="rms", eps=1e-8
        )
        self.assertAlmostEqual(
            float(rms.loss), math.sqrt(float(squared.loss) + 1e-8), places=6
        )


if __name__ == "__main__":
    unittest.main()
