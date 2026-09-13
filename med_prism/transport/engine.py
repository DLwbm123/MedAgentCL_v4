"""Sequential task-boundary orchestration; no training-time registration."""
import hashlib
import resource
import time
import torch
from .banks import fingerprint, check_invariants, geometry, tensor_hash
from .calibration import shuffled_target
from .checkpoint import apply_tensors, validate_transition
from .diagnostics import Diagnostics, write_json
from .hooks import QwenBlockBridge
from .safety import propose, measure


@torch.no_grad()
def run_transport(model, teacher, student, records, config, directory, *, bridge=None):
    started = time.perf_counter()
    diagnostics = Diagnostics(directory, config)
    task = student.task_id
    model.eval()
    model.requires_grad_(False)
    before = fingerprint(model)
    student_tensors = {name: value.detach().cpu().clone() for name, value in model.named_parameters()
                       if ".shared." in name or ".experts." in name}
    if config.mode == "off":
        invariant = check_invariants(before, fingerprint(model), set())
        write_json(diagnostics.directory / "invariants.json", invariant)
        return diagnostics.finish({"status": "PASS", "mode": "off", "teacher_pass_seconds": 0,
                                   "student_calibration_seconds": 0, "wall_seconds": time.perf_counter()-started,
                                   "invariants": invariant})
    validate_transition(teacher, student)
    bridge = bridge or QwenBlockBridge(model)
    task_sets = {tuple(w.task_ids) for w in bridge.targets.values()}
    if task_sets != {tuple(range(1, task + 1))}:
        raise ValueError("Loaded model does not match complete student task set")
    fit = [r for r in records if r["split"] == "fit"]
    hold = [r for r in records if r["split"] == "holdout"]
    if len(fit) != config.calibration_samples or len(hold) != config.holdout_samples:
        raise ValueError("Calibration counts differ from frozen configuration")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Calibration fit/holdout ID overlap")
    original_active = {name: tuple(w.active_tasks) for name, w in bridge.targets.items()}
    if bridge.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(bridge.device)
    allowed = set()
    try:
        teacher_started = time.perf_counter()
        apply_tensors(model, teacher.tensors())
        for w in bridge.targets.values():
            w.set_active_tasks(range(1, task))
        for i, record in enumerate(records):
            bridge.collect_teacher(record, task, config)
            if (i + 1) % 8 == 0:
                print(f"TPM teacher {i+1}/{len(records)}", flush=True)
        teacher_time = time.perf_counter() - teacher_started
        teacher_equivalence = bridge.check_native_equivalence(records[0])
        apply_tensors(model, student_tensors)
        for name, w in bridge.targets.items():
            w.set_active_tasks(original_active[name])
        # Includes the same visual encoder/embedding state as teacher. Exact native
        # check tests this assumption after switching to current student factors.
        student_equivalence = bridge.check_native_equivalence(records[0])
        write_json(diagnostics.directory / "native_equivalence.json", {
            "teacher": teacher_equivalence, "student_pre_tpm": student_equivalence, "architecture": bridge.audit})
        write_json(diagnostics.directory / "token_selection.json", {
            "seed": config.seed, "rule": "attention-valid stratified category round-robin",
            "fit_holdout_disjoint": True, "teacher_student_share_positions": True,
            "holdout_semantics": "current-training-derived fit-holdout, not unseen model validation",
            "records": [{"id": r["id"], "split": r["split"], **r["selection"]} for r in records]})
        # Inputs cannot be regenerated or accidentally re-tokenized from this point.
        for record in records:
            del record["batch"]
        calibration_started = time.perf_counter()
        for block, names in enumerate(bridge.block_targets):
            inputs = {name: {"fit": [], "holdout": []} for name in names}
            for record in records:
                _, captured = bridge.run_block(block, record, capture=True)
                if set(captured) != set(names):
                    raise RuntimeError("Student projection capture incomplete")
                for name, x in captured.items():
                    if len(x) != len(record["positions"]):
                        raise ValueError("Student/teacher token count mismatch")
                    inputs[name][record["split"]].append(x)
            for name in names:
                banks = bridge.banks[name]
                geometry_before = geometry(banks, task)
                sumsq = sum(r["teacher"][name]["sumsq"] for r in fit)
                numel = sum(r["teacher"][name]["numel"] for r in fit)
                rms = (sumsq / numel) ** .5
                if not (rms > 0):
                    raise RuntimeError(f"Zero/nonfinite teacher reference RMS: {name}")
                X = torch.cat(inputs[name]["fit"]).to(bridge.device) / rms
                Xh = torch.cat(inputs[name]["holdout"]).to(bridge.device) / rms
                for bank in banks:
                    if bank.task_id >= task:
                        continue
                    A, B = bank.matrices(device=bridge.device)
                    Z = torch.cat([r["teacher"][name]["targets"][bank.task_id] for r in fit]).to(bridge.device) / rms
                    Zh = torch.cat([r["teacher"][name]["targets"][bank.task_id] for r in hold]).to(bridge.device) / rms
                    matched_Z = Z
                    permutation_hash = None
                    if config.shuffle_teacher_pairing:
                        Z, permutation = shuffled_target(Z, config.seed)
                        permutation_hash = hashlib.sha256(permutation.numpy().tobytes()).hexdigest()
                    row = {"task_id": task, "historical_bank_id": bank.task_id, "block": block,
                           "module_name": name, "projection": name.rsplit(".", 1)[-1],
                           "depth": ("shallow", "middle", "deep")[min(2, block*3//len(bridge.blocks))],
                           "A_shape": list(A.shape), "B_shape": list(B.shape), "reference_rms": rms,
                           "reference_rms_source": "teacher_fit_selected_tokens_only",
                           "n_fit_tokens": len(X), "n_holdout_tokens": len(Xh),
                           "student_fit_input_sha256": tensor_hash(X),
                           "student_holdout_input_sha256": tensor_hash(Xh),
                           "pairing_permutation_sha256": permutation_hash,
                           "shuffle_scope": "fit only; holdout always true correspondence",
                           "eta": config.eta, "repair_rank_requested": config.repair_rank,
                           "geometry_before": geometry_before}
                    try:
                        candidate, stats = propose(A, B, X, Z, Xh, Zh, bank.factors[0].A.dtype, config)
                        row.update(stats)
                        if "effective_B_rank" not in row:
                            singular = torch.linalg.svdvals(B)
                            row["effective_B_rank"] = int((singular > 64*torch.finfo(B.dtype).eps*singular.max()).sum())
                        for key in ("captured_repair_spectral_energy", "repair_eigengap"):
                            row.setdefault(key, None)
                        if config.shuffle_teacher_pairing:
                            row["matched_fit_diagnostic"] = measure(A, candidate, B, X, matched_Z, config.eta)
                        if row["commit"]:
                            bank.commit(candidate)
                            # Verify actual per-expert deployment write, not just proposed concatenation.
                            actual, _ = bank.matrices(device=bridge.device)
                            if not torch.equal(actual, candidate.float()):
                                raise RuntimeError("Committed expert rows differ from remeasured candidate")
                            allowed.update(bank.names)
                    except Exception as exc:
                        bank.commit(A)
                        allowed.difference_update(bank.names)
                        row.update(commit=False, reject_reason=f"{type(exc).__name__}: {exc}")
                    row["geometry_after_this_bank"] = geometry(banks, task)
                    diagnostics.append(row)
                    if config.strict and not row["commit"]:
                        raise RuntimeError(f"Strict TPM rejection: {name}, task {bank.task_id}: {row['reject_reason']}")
                write_json(diagnostics.directory / f"geometry_block_{block:02d}_{row['projection']}.json",
                           {"before": geometry_before, "after": geometry(banks, task)})
            # Commit all same-block q/v banks BEFORE producing downstream state.
            for record in records:
                record["hidden"], _ = bridge.run_block(block, record)
                for name in names:
                    del record["teacher"][name]
            print(f"TPM block {block+1}/{len(bridge.blocks)} committed/rejected; downstream inputs refreshed", flush=True)
        student_time = time.perf_counter() - calibration_started
        invariant = check_invariants(before, fingerprint(model), allowed)
        write_json(diagnostics.directory / "invariants.json", invariant)
        return diagnostics.finish({"status": "PASS", "task_id": task, "mode": config.mode,
            "wall_seconds": time.perf_counter()-started, "teacher_pass_seconds": teacher_time,
            "student_calibration_seconds": student_time, "invariants": invariant,
            "cpu_process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(bridge.device) if bridge.device.type == "cuda" else None,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(bridge.device) if bridge.device.type == "cuda" else None,
            "native_equivalence_extra_passes": "one direct + one streamed batch per teacher/student",
            "precision_note": "Solve-rank checked numerically; BF16 edit rank diagnostic only, not an exact rank claim"})
    except Exception as exc:
        apply_tensors(model, student_tensors)
        write_json(diagnostics.directory / "failure.json", {"status": "FAIL", "exception": str(exc),
                   "accepted_checkpoint_allowed": False, "adapter_state_rolled_back": True})
        raise
    finally:
        for name, w in bridge.targets.items():
            w.set_active_tasks(original_active[name])
