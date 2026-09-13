"""Opt-in Gram transform; the baseline's normalization and reduction are reused."""
from contextlib import contextmanager
import hashlib
import json
import re
from unittest.mock import patch

import torch
import torch.nn.functional as F

from med_prism.adapters.injection import iter_rank1_wrappers
from med_prism.ms_crc.composition import group_id
from med_prism.projection import orth_losses


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def allocation(salience, rho=.25):
    s = torch.as_tensor(salience, dtype=torch.float32).detach().cpu()
    if s.shape != (16,) or not torch.isfinite(s).all() or (s < 0).any():
        raise ValueError('Expected sixteen finite nonnegative salience scores')
    if not 0 <= rho <= .25:
        raise ValueError('rho outside the fixed design bounds')
    z = (s - s.mean()) / s.max() if s.max() > 0 else torch.zeros_like(s)
    w = 1 + rho * z
    return s, z, w


def mapping(model, task):
    result = []
    for name, wrapper in iter_rank1_wrappers(model):
        match = re.search(r'\.layers\.(\d+)\.', name)
        if not match or name.rsplit('.', 1)[-1] not in ('q_proj', 'v_proj'):
            raise ValueError('Unexpected wrapper: ' + name)
        experts = wrapper.task_experts(task)
        if len(experts) != 16 or any(e.A.shape[0] != 1 for e in experts):
            raise ValueError('RGPA requires sixteen local rank-1 experts')
        result.extend((name, k, group_id(int(match[1]), k)) for k in range(16))
    return result


@contextmanager
def group_off(model, task, group):
    """Zero the actual expert forward output, preserving every tensor/scaling."""
    if model.training or group not in range(16):
        raise ValueError('Group removal is frozen evaluation only')
    modules = dict(iter_rank1_wrappers(model))
    handles = []
    try:
        for name, k, gid in mapping(model, task):
            if gid == group:
                handles.append(modules[name].task_experts(task)[k].register_forward_hook(
                    lambda module, args, output: torch.zeros_like(output)))
        yield
    finally:
        for handle in handles:
            handle.remove()


def transform(c, w, tau):
    # Preserve the identity path exactly, including its derivative at c == 0.
    residual = torch.where(tau == 0, c, c.sign() * (c.abs() - tau).clamp_min(0))
    return w.sqrt() * residual


class Protection:
    def __init__(self, model, task, histories, *, rho=.25, slack=True, eps=1e-8, saved=None):
        self.task, self.rho, self.slack, self.eps = task, rho, slack, eps
        self.histories = histories
        self.group_map = mapping(model, task)
        self.group_hash = digest(self.group_map)
        self.sigmas = {}
        self.columns = []
        groups = {(name, k): gid for name, k, gid in self.group_map}
        if set(histories) != {str(i) for i in range(1, task)}:
            raise ValueError('Missing historical RGPA sidecar; no uniform fallback')
        for i, h in histories.items():
            s, z, w = allocation(h['salience'], rho)
            if (h['method_version'] != '2.4' or h['group_map_hash'] != self.group_hash
                    or h['rho'] != rho or h['slack_enabled'] != slack
                    or h['z'] != z.tolist() or h['w'] != w.tolist()):
                raise ValueError('Historical RGPA contract mismatch: ' + i)
        for name, wrapper in iter_rank1_wrappers(model):
            current = wrapper.task_experts(task)
            previous = [i for i in wrapper.task_ids if i != task]
            if not current or not previous:
                continue
            weights, taus = [], []
            for i in previous:
                key = f'{name}/task{i}'
                if saved is None:
                    a = F.normalize(torch.cat([e.A for e in current]).detach().float(), dim=-1, eps=eps)
                    b = F.normalize(torch.cat([e.A for e in wrapper.task_experts(i)]).detach().float(), dim=-1, eps=eps)
                    sigma = float((a.square().sum(-1).mean() * b.square().sum(-1).mean() / a.shape[-1]).sqrt())
                else:
                    sigma = saved['sigmas'][key]  # Never recompute on resume.
                self.sigmas[key] = sigma
                h = histories[str(i)]
                for k, _ in enumerate(wrapper.task_experts(i)):
                    gid = groups[name, k]
                    weights.append(h['w'][gid])
                    taus.append(rho * max(-h['z'][gid], 0) * sigma if slack else 0.)
            self.columns.append((weights, taus))
        if saved is not None and self.state() != saved:
            raise ValueError('Resume protection state mismatch')

    def state(self):
        return dict(method_version='2.4', task=self.task, rho=self.rho,
                    slack_enabled=self.slack, eps=self.eps, group_map_hash=self.group_hash,
                    histories=self.histories, sigmas=self.sigmas)

    @contextmanager
    def use(self):
        """Scoped interception of only the existing pre-reduction key Gram.

        Each DDP process has its own Python module. No baseline source or
        inference path is patched on disk. Calls and column counts fail closed.
        """
        original = orth_losses.rank1_key_cosines
        count = 0
        def apply(a, b, *, eps=1e-8):
            nonlocal count
            c = original(a, b, eps=eps)
            weights, taus = self.columns[count]
            count += 1
            if len(weights) != c.shape[1] or eps != self.eps:
                raise ValueError('Baseline Gram contract changed')
            return transform(c, c.new_tensor(weights), c.new_tensor(taus))
        with patch.object(orth_losses, 'rank1_key_cosines', apply):
            yield
        if count != len(self.columns):
            raise RuntimeError(f'Expected {len(self.columns)} key Grams, saw {count}')
