import unittest

import torch
import torch.nn as nn

from med_prism.adapters.rank1_lora import Rank1ExpertLinear


def make_wrapper(experts, alpha):
    base = nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        base.weight.zero_()
    wrapper = Rank1ExpertLinear(
        base, module_identity="model.language_model.q_proj", dropout=0.0
    )
    wrapper.add_task(
        1, experts_per_task=experts, alpha=alpha, trainable=True
    )
    return wrapper


class Rank1AdapterTest(unittest.TestCase):
    def test_rank1_forward_matches_explicit_sum(self):
        wrapper = make_wrapper(4, 4.0)
        for index, expert in enumerate(wrapper.task_experts(1), start=1):
            with torch.no_grad():
                expert.A.fill_(index)
                expert.B.fill_(0.25 * index)
        x = torch.tensor([[1.0, 2.0, 3.0]])
        expected = sum(
            torch.nn.functional.linear(
                torch.nn.functional.linear(x, expert.A), expert.B
            )
            * expert.scaling
            for expert in wrapper.task_experts(1)
        )
        torch.testing.assert_close(wrapper(x), expected)

    def test_cumulative_forward_is_base_plus_each_task(self):
        wrapper = make_wrapper(1, 1.0)
        wrapper.add_task(2, experts_per_task=1, alpha=1.0, trainable=True)
        for task_id, value in ((1, 0.2), (2, 0.3)):
            expert = wrapper.task_experts(task_id)[0]
            with torch.no_grad():
                expert.A.fill_(1.0)
                expert.B.fill_(value)
        x = torch.ones(2, 3)
        expected = (
            wrapper.base_layer(x)
            + wrapper.task_contribution(x, 1)
            + wrapper.task_contribution(x, 2)
        )
        torch.testing.assert_close(wrapper(x), expected)

    def test_scaling_is_alpha_over_per_task_rank_for_e_1_4_16(self):
        x = torch.tensor([[1.0, 1.0, 1.0]])
        for experts in (1, 4, 16):
            wrapper = make_wrapper(experts, float(experts))
            for expert in wrapper.task_experts(1):
                with torch.no_grad():
                    expert.A.fill_(1.0)
                    expert.B.fill_(1.0 / experts)
                self.assertEqual(float(expert.scaling), 1.0)
            torch.testing.assert_close(
                wrapper.task_contribution(x, 1), torch.full((1, 2), 3.0)
            )

    def test_task1_scaling_does_not_change_when_new_tasks_are_added(self):
        wrapper = make_wrapper(4, 4.0)
        for expert in wrapper.task_experts(1):
            with torch.no_grad():
                expert.A.fill_(0.5)
                expert.B.fill_(0.25)
        x = torch.ones(1, 3)
        before = wrapper.task_contribution(x, 1).clone()
        wrapper.add_task(2, experts_per_task=16, alpha=16.0, trainable=True)
        wrapper.add_task(3, experts_per_task=1, alpha=1.0, trainable=True)
        after = wrapper.task_contribution(x, 1)
        torch.testing.assert_close(before, after)
        self.assertTrue(all(float(e.scaling) == 1.0 for e in wrapper.task_experts(1)))

    def test_duplicate_task_is_rejected(self):
        wrapper = make_wrapper(1, 1.0)
        with self.assertRaisesRegex(ValueError, "Duplicate task"):
            wrapper.add_task(1, experts_per_task=1, alpha=1.0, trainable=True)


if __name__ == "__main__":
    unittest.main()
