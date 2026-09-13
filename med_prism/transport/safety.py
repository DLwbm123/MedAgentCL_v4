"""Solver-dtype and actual deployed-factor checks. No automatic jitter."""
import time
import torch
from .analytic_transport import transport_keys, effective_squared_norm, _read_error


def measure(A0, A1, B, X, Z, eta):
    # FP64 diagnostics reduce cancellation; A1 has ALREADY been rounded to deployment dtype.
    A0, A1, B, X, Z = [x.double() for x in (A0, A1, B, X, Z)]
    delta = A1 - A0
    before = _read_error(A0, B, X, Z).clamp_min(0)
    after = _read_error(A1, B, X, Z).clamp_min(0)
    edit = effective_squared_norm(B, delta).clamp_min(0)
    base = effective_squared_norm(B, A0).clamp_min(0)
    # Singular values of R D equal those of B D; never construct dense d_out*d_in.
    _, R = torch.linalg.qr(B, mode="reduced")
    singular = torch.linalg.svdvals(R @ delta)
    cutoff = max(float(singular.max()) * 1e-5, 1e-10)
    return {"read_error_before": float(before), "read_error_after": float(after),
            "regularized_objective_before": float(before),
            "regularized_objective_after": float(after + eta * edit),
            "effective_relative_edit": float((edit / base.clamp_min(torch.finfo(torch.float64).tiny)).sqrt()),
            "effective_rank_estimate": int((singular > cutoff).sum()), "rank_estimate_cutoff": cutoff,
            "finite": all(bool(torch.isfinite(x).all()) for x in (A1, after, edit, singular))}


@torch.no_grad()
def propose(A, B, X, Z, Xh, Zh, model_dtype, config):
    started = time.perf_counter()
    dtype = getattr(torch, config.solve_dtype)
    tensors = [v.to(dtype) for v in (A, B, X, Z)]
    retry_reason = None
    try:
        result = transport_keys(*tensors, eta=config.eta, repair_rank=config.repair_rank,
                                max_relative_edit=config.max_relative_edit)
    except RuntimeError as exc:
        if not config.fallback_fp64 or dtype == torch.float64 or "Cholesky failed" not in str(exc):
            raise
        retry_reason = str(exc)
        dtype = torch.float64
        result = transport_keys(*[v.double() for v in tensors], eta=config.eta,
                                repair_rank=config.repair_rank, max_relative_edit=config.max_relative_edit)
    candidate = result.A_new.to(model_dtype)
    solved = measure(A, result.A_new, B, X, Z, config.eta)
    # Budget is on the solver's D, not on subtraction of two rounded A matrices.
    # The latter can contain addition-roundoff even before deployment casting.
    _, R_solve = torch.linalg.qr(B.to(dtype), mode="reduced")
    solve_singular = torch.linalg.svdvals(R_solve @ result.delta)
    solve_cutoff = max(float(solve_singular.max()) * 1e-5, 1e-10)
    solve_delta_rank = int((solve_singular > solve_cutoff).sum())
    post = measure(A, candidate, B, X, Z, config.eta)
    hold = measure(A, candidate, B, Xh, Zh, config.eta)
    def nonincrease(after, before):
        return after <= before + config.numerical_atol + config.numerical_rtol * abs(before)
    def gate(post, hold):
        reasons = []
        for label, stats in (("solve", solved), ("post_cast", post)):
            if not stats["finite"]:
                reasons.append(label + ":nonfinite")
            if stats["effective_relative_edit"] > config.max_relative_edit + config.edit_tolerance:
                reasons.append(label + ":edit_cap")
            if not nonincrease(stats["read_error_after"], stats["read_error_before"]):
                reasons.append(label + ":fit_error_increase")
            if not nonincrease(stats["regularized_objective_after"], stats["regularized_objective_before"]):
                reasons.append(label + ":objective_increase")
        if config.repair_rank is not None and solve_delta_rank > config.repair_rank:
            reasons.append("solve:rank_budget")
        if not hold["finite"]:
            reasons.append("holdout:nonfinite")
        if config.require_holdout_nonincrease and not nonincrease(hold["read_error_after"], hold["read_error_before"]):
            reasons.append("holdout:error_increase")
        return reasons

    initial_gamma = float(result.diagnostics["damping"])
    initial_post_edit = post["effective_relative_edit"]
    gamma_scale = 1.0
    backtracking_steps = 0
    reasons = gate(post, hold)
    # Fixed deployment safeguard, not a solver change or hyperparameter search.
    # result.delta already contains initial_gamma * D_s. Scale that ORIGINAL
    # direction, never a cast/subtracted delta or an accumulated candidate.
    while reasons == ["post_cast:edit_cap"] and backtracking_steps < 16 and initial_gamma > 0:
        gamma_scale *= config.max_relative_edit / post["effective_relative_edit"] * 0.999
        candidate = (A.to(dtype) + result.delta * gamma_scale).to(model_dtype)
        post = measure(A, candidate, B, X, Z, config.eta)
        hold = measure(A, candidate, B, Xh, Zh, config.eta)
        backtracking_steps += 1
        reasons = gate(post, hold)
    diagnostics = {**result.diagnostics, "solve_check": solved, "post_cast": post, "holdout": hold,
                   "initial_gamma": initial_gamma, "final_gamma": initial_gamma * gamma_scale,
                   "backtracking_steps": backtracking_steps,
                   "initial_post_cast_relative_edit": initial_post_edit,
                   "final_post_cast_relative_edit": post["effective_relative_edit"],
                   "solve_delta_effective_rank": solve_delta_rank, "solve_delta_rank_cutoff": solve_cutoff,
                   "solve_dtype": str(dtype), "model_dtype": str(model_dtype), "fp64_retry_reason": retry_reason,
                   "commit": not reasons, "reject_reason": ";".join(reasons) or None,
                   "elapsed_time": time.perf_counter() - started}
    return candidate, diagnostics
