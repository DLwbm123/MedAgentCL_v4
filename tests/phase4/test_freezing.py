import unittest

import torch

from med_prism.adapters.injection import freeze_except_task
from tests.phase4.helpers import make_bank


class FreezingTest(unittest.TestCase):
    def test_only_current_task_experts_are_trainable(self):
        model = make_bank(
            ((1, 4, 4.0, False), (2, 4, 4.0, True))
        )
        freeze_except_task(model, 2)
        trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable)
        self.assertTrue(all(".experts.task_0002__" in name for name in trainable))
        self.assertFalse(
            any(
                parameter.requires_grad
                for name, parameter in model.named_parameters()
                if ".experts.task_0001__" in name
            )
        )

    def test_optimizer_step_changes_current_and_preserves_old(self):
        model = make_bank(
            ((1, 1, 1.0, False), (2, 1, 1.0, True))
        )
        freeze_except_task(model, 2)
        old_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if ".experts.task_0001__" in name
        }
        current_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if ".experts.task_0002__" in name
        }
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.SGD(trainable, lr=0.1)
        loss = model(torch.ones(1, 4)).sum()
        loss.backward()
        optimizer.step()
        named = dict(model.named_parameters())
        self.assertTrue(
            any(not torch.equal(value, named[name]) for name, value in current_before.items())
        )
        self.assertTrue(
            all(torch.equal(value, named[name]) for name, value in old_before.items())
        )
        self.assertTrue(
            all(named[name].grad is None for name in old_before)
        )


if __name__ == "__main__":
    unittest.main()
