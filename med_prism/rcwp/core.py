"""FP32 sufficient statistics and signed net-write loss; no model mutations."""
import hashlib
import torch


def digest(t):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(t.shape), t.dtype)).encode() +
                          t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def factors(wrapper, task, detach=True):
    experts = wrapper.task_experts(task)
    if not experts or wrapper.private_adapter_type != "rank1_expert_bank":
        raise ValueError("RCWP requires an existing rank-1 expert bank")
    convert = (lambda x: x.detach().float()) if detach else (lambda x: x.float())
    A = torch.cat([convert(e.A) for e in experts], 0)
    # Each expert has its own actual alpha/rank scale. Never divide by bank rank.
    B = torch.cat([convert(e.B) * e.scaling.detach().float() for e in experts], 1)
    return A, B


def bank_binding(wrapper, task):
    return [{"A": digest(e.A), "B": digest(e.B), "scaling": digest(e.scaling),
             "rank": e.A.shape[0], "dtype": str(e.A.dtype)}
            for e in wrapper.task_experts(task)]


def output_basis(B):
    U, s, Vh = torch.linalg.svd(B.detach().float(), full_matrices=False)
    keep = s > (max(B.shape) * torch.finfo(torch.float32).eps * s.max())
    if not keep.any():
        raise ValueError("Zero effective B rank: no sensitivity field can be fitted")
    # B @ lift == Q, also for rank-deficient B. Original A/B are never refactored.
    return U[:, keep], Vh[keep].T / s[keep], int(keep.sum())


class Moments:
    """Only O(r^2) FP32 aggregates, never persistent token/sample features."""
    def __init__(self, rank, output_rank):
        self.n = 0
        self.z = torch.zeros(rank)
        self.c = torch.zeros(output_rank)
        self.zz = torch.zeros(rank, rank)
        self.cz = torch.zeros(output_rank, rank)
        self.cc = torch.zeros(())

    def add(self, z, c):
        z, c = z.detach().float(), c.detach().float()
        if not torch.isfinite(z).all() or not torch.isfinite(c).all():
            raise ValueError("Nonfinite boundary features/sensitivity")
        self.n += len(z)
        self.z += z.sum(0).cpu()
        self.c += c.sum(0).cpu()
        self.zz += (z.T @ z).cpu()
        self.cz += (c.T @ z).cpu()
        self.cc += c.square().sum().cpu()

    def design(self, mu, scale):
        if not self.n:
            raise ValueError("Empty boundary partition")
        ez, ec = self.z / self.n, self.c / self.n
        center = (self.zz / self.n - ez[:, None] * mu[None, :] -
                  mu[:, None] * ez[None, :] + mu[:, None] * mu[None, :])
        G = torch.ones(len(mu)+1, len(mu)+1)
        G[:-1, :-1] = center / scale[:, None] / scale[None, :]
        G[:-1, -1] = G[-1, :-1] = (ez-mu)/scale
        C = torch.cat([(self.cz/self.n - ec[:, None]*mu)/scale, ec[:, None]], 1)
        return G, C

    def error(self, K, mu, scale):
        G, C = self.design(mu, scale)
        return float((self.cc/self.n - 2*(K*C).sum() + ((K@G)*K).sum()).clamp_min(0))


def fit_field(fit, holdout, ridge=1e-3):
    if ridge != 1e-3:
        raise ValueError("v2.1 fixes ridge=1e-3; no search")
    mu = fit.z/fit.n
    scale = (fit.zz.diag()/fit.n - mu.square()).clamp_min(0).sqrt().clamp_min(1e-6)
    G, C = fit.design(mu, scale)
    penalty = torch.eye(len(mu)+1)*ridge
    penalty[-1, -1] = 0  # Intercept is not regularized.
    system = G+penalty
    try:
        K = torch.linalg.solve(system, C.T).T
    except RuntimeError:
        K = (torch.linalg.pinv(system) @ C.T).T
    if not torch.isfinite(K).all():
        raise ValueError("Nonfinite fitted field")
    constant = torch.zeros_like(K)
    constant[:, -1] = fit.c/fit.n
    condition = float(torch.linalg.cond(system))
    if not torch.isfinite(torch.tensor(condition)):
        raise ValueError("Nonfinite field condition number")
    return K, mu, scale, {"condition_number": condition,
        "fit_error": fit.error(K,mu,scale), "holdout_error": holdout.error(K,mu,scale),
        "constant_fit_error": fit.error(constant,mu,scale),
        "constant_holdout_error": holdout.error(constant,mu,scale), "finite": True}, constant


def phi(z, mu, scale):
    return torch.cat([(z-mu)/scale, torch.ones_like(z[..., :1])], -1)


def signed_score(h, current_A, current_B, historical_A, historical_B, M, mu, scale):
    """FP32 auxiliary math; the ONLY live leaves are current A/B.

    Low-rank contraction equals (B_i M phi)^T (B_t A_t h), including all scales.
    Deterministic B A stopgrad(h) as defined by RCWP; no auxiliary dropout draw.
    The original task-loss forward keeps its existing dropout unchanged.
    No rank-pair clipping.
    """
    h = h.detach().float()
    ai, bi, M, mu, scale = [x.detach().float() for x in (historical_A,historical_B,M,mu,scale)]
    zt = h @ current_A.float().T
    coordinates = phi(h @ ai.T, mu, scale) @ M.T
    return ((coordinates @ (bi.T @ current_B.float())) * zt).sum(-1)


def net_penalty(scores, nu):
    """scores: history x batch, ALREADY summed over every wrapper/token/rank."""
    if scores.shape[0] == 0:
        return scores.new_zeros(())
    if not torch.isfinite(scores).all() or not torch.isfinite(nu).all() or (nu < 1).any():
        raise ValueError("Invalid RCWP score/scale")
    return torch.relu(-scores/nu.detach()[:, None]).square().mean()


def gold_margin(logits, labels):
    # Causal shift: logits[p-1] predict labels[p]. Image/prompt labels stay -100.
    logits, labels = logits[:, :-1].float(), labels[:, 1:]
    valid = labels != -100
    if not valid.any() or (valid.sum(-1) == 0).any():
        raise ValueError("Every boundary example needs a gold answer token")
    gold = labels.clamp_min(0)
    target = logits.gather(-1, gold[..., None]).squeeze(-1)
    others = logits.scatter(-1, gold[..., None], -torch.inf).logsumexp(-1)
    return (((target-others)*valid).sum(-1)/valid.sum(-1)).mean()
