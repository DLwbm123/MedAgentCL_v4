"""Transportable parameter-memory primitives for a Med-PRISM research pilot.

This is NOT a Qwen integration or an empirically validated continual-learning method.
No model weights, patient data, or training checkpoints are included.

Conventions
-----------
A: (rank, input_dim), B: (output_dim, rank).
X: (n_tokens, input_dim), Z_ref: (n_tokens, rank).
Use identical examples, token positions, prefixes, and a SINGLE FIXED reference
RMS scale for X and Z_ref. Do not normalize each token separately.
The effective bank is B @ A; absorb any fixed LoRA scaling into B for diagnostics.

For eta > 0, minimize over D:
  ||(X @ (A + D).T - Z_ref) @ B.T||_F^2 / n
  + eta * ||B @ D||_F^2.
Without a repair-rank budget or edit cap, the solution minimizes this objective
even if B is rank deficient; the effective correction B @ D is then unique.
With a repair-rank budget, the optimum can be non-unique at spectral ties.
With an edit cap, the returned damped correction is feasible but is NOT claimed
to be the exact joint rank-and-norm constrained optimum.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
from typing import Optional
import torch
from torch import Tensor


@dataclass
class TransportResult:
    # Kept in the solve dtype. The caller casts, remeasures, and commits explicitly.
    A_new: Tensor
    delta: Tensor
    diagnostics: dict[str, float]


def _validate(A: Tensor, B: Tensor, X: Tensor, Z: Tensor, eta: float) -> None:
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be finite and strictly positive")
    if any(t.ndim != 2 for t in (A, B, X, Z)):
        raise ValueError("A, B, X and Z_ref must be matrices")
    r, d = A.shape
    if r == 0 or d == 0 or X.shape[0] == 0 or B.shape[0] == 0:
        raise ValueError("Empty dimensions are unsupported")
    if B.shape[1] != r or X.shape[1] != d or Z.shape != (X.shape[0], r):
        raise ValueError("Inconsistent matrix shapes")
    if any(t.device != A.device for t in (B, X, Z)):
        raise ValueError("All inputs must be on the same device")
    if any(not t.is_floating_point() for t in (A, B, X, Z)):
        raise TypeError("Inputs must be floating-point tensors")
    if any(not bool(torch.isfinite(t).all()) for t in (A, B, X, Z)):
        raise ValueError("Inputs contain NaN or infinity")


def _dtype(*tensors: Tensor) -> torch.dtype:
    return torch.float64 if any(t.dtype == torch.float64 for t in tensors) else torch.float32


def _solve_delta(A: Tensor, X: Tensor, Z: Tensor, eta: float) -> Tensor:
    """No automatic jitter: changing eta silently would change the claimed optimum."""
    n, d = X.shape
    E = Z - X @ A.T
    ridge = n * eta
    if n <= d:
        gram = X @ X.T
        gram = (gram + gram.T) / 2
        gram = gram + ridge * torch.eye(n, dtype=X.dtype, device=X.device)
        chol, info = torch.linalg.cholesky_ex(gram)
        if bool((info != 0).any()):
            raise RuntimeError("Cholesky failed. Retry explicitly in float64 or increase eta and log it.")
        return E.T @ torch.cholesky_solve(X, chol)
    gram = X.T @ X
    gram = (gram + gram.T) / 2
    gram = gram + ridge * torch.eye(d, dtype=X.dtype, device=X.device)
    chol, info = torch.linalg.cholesky_ex(gram)
    if bool((info != 0).any()):
        raise RuntimeError("Cholesky failed. Retry explicitly in float64 or increase eta and log it.")
    return torch.cholesky_solve(X.T @ E, chol).T



def _rank_budget_delta(B: Tensor, D_full: Tensor, X: Tensor, eta: float,
                       repair_rank: Optional[int]) -> tuple[Tensor, dict[str, float]]:
    """Best rank-budgeted EFFECTIVE correction for the ridge objective.

    Uses only an output-factor SVD and a rank x rank eigenproblem. In exact
    arithmetic the correction minimizes the objective subject to rank(B D)<=s.
    Numerically negligible B singular modes are discarded and explicitly logged.
    At eigenvalue ties, the optimal effective correction need not be unique.
    """
    if repair_rank is None:
        return D_full, {"repair_rank_requested": -1.0, "discarded_B_energy_fraction": 0.0}
    if not isinstance(repair_rank, int) or isinstance(repair_rank, bool) or repair_rank < 0:
        raise ValueError("repair_rank must be a nonnegative integer or None")
    if repair_rank == 0:
        return torch.zeros_like(D_full), {"repair_rank_requested": 0.0, "discarded_B_energy_fraction": 0.0}
    _, singular, vh = torch.linalg.svd(B, full_matrices=False)
    cutoff = 64 * torch.finfo(B.dtype).eps * singular.max()
    keep = singular > cutoff
    total = singular.square().sum()
    discarded = float(singular[~keep].square().sum() / total) if float(total) > 0 else 0.0
    q = int(keep.sum())
    if q == 0:
        return torch.zeros_like(D_full), {"repair_rank_requested": float(repair_rank), "effective_B_rank": 0.0, "discarded_B_energy_fraction": discarded}
    sig = singular[keep]
    vh = vh[keep]
    R = sig[:, None] * vh  # B = U R on retained output support
    D_bar = R @ D_full
    DX = D_bar @ X.T
    F = DX @ DX.T + (X.shape[0] * eta) * (D_bar @ D_bar.T)
    F = (F + F.T) / 2
    eigenvalues, V = torch.linalg.eigh(F)
    k = min(repair_rank, q)
    Vtop = V[:, -k:]
    D_bar_s = Vtop @ (Vtop.T @ D_bar)
    # R^dagger, applied without constructing an input-dimensional projector.
    D = (vh.T / sig[None, :]) @ D_bar_s
    pos = eigenvalues.clamp_min(0)
    energy = float(pos[-k:].sum() / pos.sum()) if float(pos.sum()) > 0 else 1.0
    gap = float(eigenvalues[-k] - eigenvalues[-k-1]) if k < q else float(eigenvalues[0])
    return D, {"repair_rank_requested": float(repair_rank), "effective_B_rank": float(q),
               "captured_repair_spectral_energy": energy, "repair_eigengap": gap,
               "discarded_B_energy_fraction": discarded}


def effective_squared_norm(B: Tensor, A: Tensor) -> Tensor:
    """Compute ||B A||_F^2 without forming an output_dim x input_dim matrix."""
    G = B.T @ B
    return ((G @ A) * A).sum()


def _read_error(A: Tensor, B: Tensor, X: Tensor, Z: Tensor) -> Tensor:
    err = X @ A.T - Z
    return ((err @ (B.T @ B)) * err).sum() / X.shape[0]


@torch.no_grad()
def transport_keys(
    A_ref: Tensor,
    B_fixed: Tensor,
    X_current: Tensor,
    Z_ref: Tensor,
    *,
    eta: float = 0.1,
    max_relative_edit: Optional[float] = 0.05,
    repair_rank: Optional[int] = 2,
) -> TransportResult:
    """Fit and optionally damp a historical-key correction; no in-place mutation.

    max_relative_edit caps ||B D||_F / ||B A_ref||_F. It is a proposed
    engineering safeguard, NOT an old-task risk guarantee. Set None to obtain
    the optimum before damping (which is rank constrained unless repair_rank=None). Damping preserves local error nonincrease
    in exact arithmetic. Re-evaluate this after casting to the deployed dtype.
    """
    _validate(A_ref, B_fixed, X_current, Z_ref, eta)
    if max_relative_edit is not None and (
        not math.isfinite(max_relative_edit) or max_relative_edit < 0
    ):
        raise ValueError("max_relative_edit must be nonnegative and finite, or None")
    dt = _dtype(A_ref, B_fixed, X_current, Z_ref)
    A, B, X, Z = [t.detach().to(dt) for t in (A_ref, B_fixed, X_current, Z_ref)]
    D_full = _solve_delta(A, X, Z, eta)
    D, rank_stats = _rank_budget_delta(B, D_full, X, eta, repair_rank)
    base_sq = effective_squared_norm(B, A).clamp_min(0)
    raw_edit_sq = effective_squared_norm(B, D).clamp_min(0)
    gamma = 1.0
    if max_relative_edit is not None:
        if float(base_sq) == 0:
            gamma = 0.0
        elif float(raw_edit_sq) > 0:
            gamma = min(1.0, max_relative_edit * math.sqrt(float(base_sq / raw_edit_sq)))
    D = D * gamma
    new_A = A + D
    before = _read_error(A, B, X, Z)
    after = _read_error(new_A, B, X, Z)
    edit_sq = effective_squared_norm(B, D).clamp_min(0)
    eps = torch.finfo(dt).tiny
    diagnostics = {
        **rank_stats, "eta": float(eta), "n_tokens": float(X.shape[0]),
        "damping": gamma, "read_error_before": float(before),
        "read_error_after": float(after),
        "regularized_objective_before": float(before),
        "regularized_objective_after": float(after + eta * edit_sq),
        "effective_relative_edit": math.sqrt(float(edit_sq / base_sq.clamp_min(eps))),
    }
    if not bool(torch.isfinite(new_A).all()):
        raise RuntimeError("Non-finite result; do not commit the correction")
    return TransportResult(new_A, D, diagnostics)


def profiled_read_loss(
    A_ref: Tensor, B_fixed: Tensor, X_current: Tensor, Z_ref: Tensor, *, eta: float = 0.1,
    repair_rank: Optional[int] = None
) -> Tensor:
    """Optional stronger research variant: exact profiled local objective.

    Only X_current receives gradients. The analytic optimum is detached, using
    the envelope theorem (away from active spectral ties for a rank budget).
    This yields the FIRST derivative of the exact profile
    objective, not its full higher-order derivatives. Do not apply the edit cap
    inside this function: a clipped unconstrained solution is not generally the
    optimum of a constrained problem. Apply a fixed, detached reference scale
    to X_current and Z_ref before calling. Normalize the returned loss externally
    using a fixed teacher-response energy if required.
    """
    _validate(A_ref, B_fixed, X_current, Z_ref, eta)
    dt = _dtype(A_ref, B_fixed, X_current, Z_ref)
    A, B, Z = [t.detach().to(dt) for t in (A_ref, B_fixed, Z_ref)]
    X = X_current.to(dt)
    with torch.no_grad():
        D_full = _solve_delta(A, X.detach(), Z, eta)
        D, _ = _rank_budget_delta(B, D_full, X.detach(), eta, repair_rank)
    return _read_error(A + D, B, X, Z) + eta * effective_squared_norm(B, D)


def self_test() -> dict[str, object]:
    """Synthetic numerical checks only; no Med-PRISM empirical claims."""
    torch.manual_seed(20260906)
    torch.set_num_threads(1)
    dt = torch.float64
    d, r, m, n = 24, 4, 11, 80
    A = torch.randn(r, d, dtype=dt)
    B = torch.randn(m, r, dtype=dt)
    X_old = torch.randn(n, d, dtype=dt)
    T = torch.eye(d, dtype=dt) + 0.06 * torch.randn(d, d, dtype=dt)
    X = X_old @ T.T
    Z = X_old @ A.T
    eta = 0.1
    result = transport_keys(A, B, X, Z, eta=eta, max_relative_edit=None, repair_rank=None)
    assert result.diagnostics["regularized_objective_after"] <= result.diagnostics["regularized_objective_before"] + 1e-9
    H = X.T @ X + n * eta * torch.eye(d, dtype=dt)
    E = Z - X @ A.T
    normal_residual = B.T @ B @ (result.delta @ H - E.T @ X)
    assert float(normal_residual.norm()) < 1e-8
    # Gauge-equivalent factorization should produce the same effective bank.
    Q = torch.eye(r, dtype=dt) + 0.15 * torch.randn(r, r, dtype=dt)
    AQ = torch.linalg.solve(Q, A)
    ZQ = torch.linalg.solve(Q, Z.T).T
    rq = transport_keys(AQ, B @ Q, X, ZQ, eta=eta, max_relative_edit=None, repair_rank=None)
    gauge_error = float((B @ result.A_new - B @ Q @ rq.A_new).abs().max())
    assert gauge_error < 1e-10
    # The exact coordinate-inverse limit is recovered on full-rank data.
    exact = transport_keys(A, B, X, Z, eta=1e-10, max_relative_edit=None, repair_rank=None)
    oracle_A = torch.linalg.solve(T.T, A.T).T
    coordinate_error = float((B @ exact.A_new - B @ oracle_A).abs().max())
    assert coordinate_error < 1e-7
    # Rank-deficient output factor still has the correct action optimum.
    Bs = B.clone(); Bs[:, -1] = Bs[:, 0]
    rs = transport_keys(A, Bs, X, Z, eta=eta, max_relative_edit=None, repair_rank=None)
    assert rs.diagnostics["regularized_objective_after"] <= rs.diagnostics["regularized_objective_before"] + 1e-9
    # Damping preserves the claimed bound.
    capped = transport_keys(A, B, X, Z, eta=eta, max_relative_edit=0.02, repair_rank=None)
    assert capped.diagnostics["effective_relative_edit"] <= 0.0200000001
    assert capped.diagnostics["read_error_after"] <= capped.diagnostics["read_error_before"] + 1e-9
    # Dual solve and untouched directions outside the sampled input span.
    Xu = X[:8]; Zu = Z[:8]
    ru = transport_keys(A, B, Xu, Zu, eta=eta, max_relative_edit=None, repair_rank=None)
    primal_delta = torch.linalg.solve(Xu.T @ Xu + len(Xu) * eta * torch.eye(d, dtype=dt), Xu.T @ (Zu - Xu @ A.T)).T
    assert torch.allclose(ru.delta, primal_delta, atol=1e-10, rtol=1e-10)
    U = torch.linalg.svd(Xu, full_matrices=True).Vh
    unseen = U[-1]
    unseen_error = float((ru.delta @ unseen).norm())
    assert unseen_error < 1e-10
    # Envelope-gradient implementation versus differentiating through the solve.
    Xe = X.clone().requires_grad_(True)
    loss_env = profiled_read_loss(A, B, Xe, Z, eta=eta)
    grad_env, = torch.autograd.grad(loss_env, Xe)
    Xf = X.clone().requires_grad_(True)
    Df = _solve_delta(A, Xf, Z, eta)
    loss_full = _read_error(A + Df, B, Xf, Z) + eta * effective_squared_norm(B, Df)
    grad_full, = torch.autograd.grad(loss_full, Xf)
    gradient_error = float((grad_env - grad_full).abs().max())
    assert gradient_error < 1e-9
    # Closed-form minimized objective.
    K = X @ X.T + n * eta * torch.eye(n, dtype=dt)
    action_E = E @ B.T
    closed_value = eta * (action_E * torch.linalg.solve(K, action_E)).sum()
    assert torch.allclose(loss_env, closed_value, atol=1e-9, rtol=1e-9)
    # Rank-budgeted correction: feasibility, local optimum, and gauge invariance.
    ranked = transport_keys(A, B, X, Z, eta=eta, max_relative_edit=None, repair_rank=2)
    effective_rank = int(torch.linalg.matrix_rank(B @ ranked.delta, tol=1e-9))
    assert effective_rank <= 2
    assert ranked.diagnostics["regularized_objective_after"] <= ranked.diagnostics["regularized_objective_before"] + 1e-9
    ranked_q = transport_keys(AQ, B @ Q, X, ZQ, eta=eta, max_relative_edit=None, repair_rank=2)
    ranked_gauge_error = float((B @ ranked.A_new - B @ Q @ ranked_q.A_new).abs().max())
    assert ranked_gauge_error < 1e-9
    # Compare against direct Eckart-Young truncation of the dense completed square.
    h_eig, h_vec = torch.linalg.eigh(H)
    hsqrt = (h_vec * h_eig.sqrt()[None, :]) @ h_vec.T
    hinvsqrt = (h_vec * h_eig.rsqrt()[None, :]) @ h_vec.T
    Mfull = B @ result.delta @ hsqrt
    u, sv, vh = torch.linalg.svd(Mfull, full_matrices=False)
    oracle_delta_weight = ((u[:, :2] * sv[:2][None, :]) @ vh[:2]) @ hinvsqrt
    ranked_optimality_error = float((B @ ranked.delta - oracle_delta_weight).abs().max())
    assert ranked_optimality_error < 1e-9
    # Envelope derivative for a rank-limited optimum, away from spectral ties.
    Xr = X.clone().requires_grad_(True)
    lr = profiled_read_loss(A, B, Xr, Z, eta=eta, repair_rank=2)
    gr, = torch.autograd.grad(lr, Xr)
    direction = torch.randn_like(X); direction = direction / direction.norm()
    h = 1e-5
    finite_difference = (profiled_read_loss(A, B, X + h*direction, Z, eta=eta, repair_rank=2) - profiled_read_loss(A, B, X - h*direction, Z, eta=eta, repair_rank=2)) / (2*h)
    ranked_gradient_error = abs(float(finite_difference - (gr * direction).sum()))
    assert ranked_gradient_error < 1e-6
    # FP32 smoke test (reported separately from precision-sensitive FP64 tests).
    fp32 = transport_keys(A.float(), B.float(), X.float(), Z.float(), eta=eta)
    assert fp32.diagnostics["read_error_after"] <= fp32.diagnostics["read_error_before"] + 1e-3
    return {
        "scope": "Synthetic linear-algebra tests only; no Qwen or MedicalSkill-CL run.",
        "torch_version": torch.__version__, "device": "cpu", "status": "PASS",
        "checks": ["objective_nonincrease", "normal_equation", "gauge_invariance", "linear_coordinate_limit", "rank_deficient_B", "edit_cap", "primal_dual_equivalence", "unobserved_span_unchanged", "envelope_gradient", "profile_value", "float32_smoke", "rank_budget_feasibility", "rank_budget_gauge_invariance", "rank_budget_optimality", "rank_budget_envelope_gradient"],
        "max_gauge_error_float64": gauge_error,
        "max_coordinate_error_float64": coordinate_error,
        "max_envelope_gradient_error_float64": gradient_error,
        "unobserved_span_error_float64": unseen_error,
        "rank_budget_gauge_error_float64": ranked_gauge_error,
        "rank_budget_optimality_error_float64": ranked_optimality_error,
        "rank_budget_gradient_direction_error_float64": ranked_gradient_error,
        "rank_budget_example": ranked.diagnostics,
        "uncapped_example": result.diagnostics,
        "capped_example": capped.diagnostics,
    }


if __name__ == "__main__":
    print(json.dumps(self_test(), ensure_ascii=False, indent=2))
