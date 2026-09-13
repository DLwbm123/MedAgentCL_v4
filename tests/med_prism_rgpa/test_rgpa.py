import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from med_prism.adapters.rank1_lora import Rank1ExpertLinear
from med_prism.projection.orth_losses import compute_rank1_key_isolation_loss
from med_prism.rgpa.core import Protection, allocation, mapping, digest, transform, group_off
from med_prism.rgpa.state import save_resume, load_resume


def model():
    torch.manual_seed(7)
    m = nn.Module()
    m.language = nn.Module()
    m.language.layers = nn.ModuleList([nn.Module() for _ in range(36)])
    for k in (0, 10, 20, 30):
        for name in ('q_proj', 'v_proj'):
            w = Rank1ExpertLinear(nn.Linear(8, 7), module_identity=f'language.layers.{k}.{name}')
            w.add_task(1, experts_per_task=16, alpha=16., trainable=False)
            w.add_task(2, experts_per_task=16, alpha=16., trainable=True)
            for e in w.experts.values():
                with torch.no_grad():
                    e.B.normal_()
            w.set_active_tasks([1, 2])
            setattr(m.language.layers[k], name, w)
    return m


def histories(m, s, rho=.25, slack=True):
    s, z, w = allocation(s, rho)
    return {'1': dict(method_version='2.4', rho=rho, slack_enabled=slack,
                     group_map_hash=digest(mapping(m, 2)), salience=s.tolist(), z=z.tolist(), w=w.tolist())}


def loss(m, protection=None):
    if protection is None:
        return compute_rank1_key_isolation_loss(m, current_task_id=2).loss
    with protection.use():
        return compute_rank1_key_isolation_loss(m, current_task_id=2).loss


class Tests(unittest.TestCase):
    def recovery(self, s, rho):
        m = model()
        params = [p for n, p in m.named_parameters() if p.requires_grad and n.endswith('.A')]
        baseline = loss(m)
        g0 = torch.autograd.grad(baseline, params)
        p = Protection(m, 2, histories(m, s, rho), rho=rho)
        actual = loss(m, p)
        g1 = torch.autograd.grad(actual, params)
        self.assertTrue(torch.equal(baseline, actual))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(g0, g1)))

    def test_rho_zero_real_project_loss_gradient(self):
        self.recovery(list(range(16)), 0.)

    def test_uniform_real_project_loss_gradient(self):
        self.recovery([3.] * 16, .25)

    def test_weights_and_squared_not_w_squared(self):
        _, z, w = allocation(list(range(16)))
        self.assertAlmostEqual(float(w.mean()), 1., places=7)
        self.assertTrue(bool(((w >= .75) & (w <= 1.25)).all()))
        c = torch.arange(32.).reshape(2, 16) / 32
        torch.testing.assert_close(transform(c, w, torch.zeros(16)).square(), w * c.square())
        c = torch.tensor([0., .01, .03, -.03], requires_grad=True)
        t = transform(c, torch.ones(4), torch.tensor([0., .02, .02, .02])).square().sum()
        torch.testing.assert_close(torch.autograd.grad(t, c)[0], torch.tensor([0., 0., .02, -.02]))

    def test_group_off_full_recovery_and_scaling(self):
        m = model().eval()
        before = {k: v.clone() for k, v in m.state_dict().items()}
        w = m.language.layers[0].q_proj
        x = torch.randn(1, 3, 8)
        full = w(x)
        expected = full - sum(e(x, w.dropout) for e in w.task_experts(2)[:4])
        with group_off(m, 2, 0):
            off = w(x)
            torch.testing.assert_close(off, expected)
            # Different depth band is untouched.
            other = m.language.layers[10].q_proj(x)
        torch.testing.assert_close(full, w(x), rtol=0, atol=0)
        torch.testing.assert_close(other, m.language.layers[10].q_proj(x), rtol=0, atol=0)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in m.state_dict().items()))
        labels = torch.tensor([1, 2, 3])
        torch.testing.assert_close(nn.functional.cross_entropy(full[0], labels),
                                   nn.functional.cross_entropy(w(x)[0], labels), rtol=0, atol=0)

    def test_short_training_checkpoint_resume_next_step(self):
        m = model()
        p = Protection(m, 2, histories(m, range(16)))
        opt = torch.optim.AdamW([v for v in m.parameters() if v.requires_grad], lr=.001)
        def step(m, p, opt):
            opt.zero_grad()
            result = loss(m, p) + m.language.layers[0].q_proj(torch.ones(2, 8)).square().mean()
            result.backward()
            opt.step()
            return result.detach()
        step(m, p, opt)
        with tempfile.TemporaryDirectory() as d:
            save_resume(m, p, d)
            optimizer_state = copy.deepcopy(opt.state_dict())
            expected = step(m, p, opt)
            restored = model()
            state = load_resume(restored, d)
            p2 = Protection(restored, 2, histories(restored, range(16)), saved=state)
            opt2 = torch.optim.AdamW([v for v in restored.parameters() if v.requires_grad], lr=.001)
            opt2.load_state_dict(optimizer_state)
            actual = step(restored, p2, opt2)
            self.assertEqual(p.state(), p2.state())
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for a, b in zip(m.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_inference_invariance_sidecar(self):
        m = model().eval()
        x = torch.randn(2, 8)
        before = m.language.layers[0].q_proj(x)
        p = Protection(m, 2, histories(m, range(16)))
        loss(m, p)
        torch.testing.assert_close(before, m.language.layers[0].q_proj(x), rtol=0, atol=0)

    def test_missing_state_and_changed_sigma_rejected(self):
        m = model()
        with self.assertRaises(ValueError):
            Protection(m, 2, {})
        p = Protection(m, 2, histories(m, range(16)))
        saved = p.state()
        # Resume takes saved sigma, NOT changed initialization geometry.
        for n, parameter in m.named_parameters():
            if parameter.requires_grad and n.endswith('.A'):
                parameter.data.zero_()
        p2 = Protection(m, 2, histories(m, range(16)), saved=saved)
        self.assertEqual(p.sigmas, p2.sigmas)

    def test_positive_part_before_average(self):
        delta = torch.tensor([[2., -2.], [-2., 2.]])
        self.assertEqual(delta.clamp_min(0).mean(0).tolist(), [1., 1.])


if __name__ == '__main__':
    unittest.main()
