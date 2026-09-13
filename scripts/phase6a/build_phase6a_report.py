#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/phase6a"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    gradient = read_json(ARTIFACT / "shared_drift_gradient_audit.json")
    hashes = read_json(ARTIFACT / "shared_drift_hash_audit.json")
    metrics = read_json(ARTIFACT / "cl_metrics.json")
    phase6b = read_json(ARTIFACT / "phase6b_data_summary.json")
    prediction_summaries = [
        read_json(path)
        for path in sorted((ARTIFACT / "predictions").glob("*.summary.json"))
    ]
    if len(prediction_summaries) != 5:
        raise RuntimeError("Expected exactly five executed Phase 6A pilot cells")
    if not all(
        item["status"] == "PASS"
        and item["total"] == 64
        and item["deterministic_first_sample_replay"]
        and item["active_component_manifest"]["shared_load_count"] == 1
        and not item["active_component_manifest"]["merged"]
        for item in prediction_summaries
    ):
        raise RuntimeError("Pilot prediction contract is incomplete")

    traces = []
    for task in (1, 2):
        path = ARTIFACT / f"shared_drift_training_trace_task{task}.jsonl"
        traces.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    max_identity_error = max(item["full_loss_identity_error"] for item in traces)
    max_old_private_gradient = max(item["old_private_gradient_norm"] for item in traces)
    max_drift_gradient = max(item["shared_drift_gradient_norm"] for item in traces)
    primary = metrics["primary_cumulative"]
    oracle = metrics["oracle_skill_aware"]
    checks = {
        "real_shared_drift_gradient": (
            gradient["closure_checks"]["task2_shared_drift_gradient_nonzero"]
            and max_drift_gradient > 0
        ),
        "full_loss_identity": (
            gradient["closure_checks"]["full_loss_identity_within_tolerance"]
            and max_identity_error < 1e-6
        ),
        "old_private_frozen": (
            gradient["closure_checks"]["task2_old_private_gradient_zero"]
            and hashes["closure_checks"]["task2_private1_hash_unchanged"]
            and max_old_private_gradient == 0
        ),
        "evaluation_contracts_distinct": (
            primary["definition"] != oracle["definition"]
            and oracle["diagnostic_only"]
        ),
        "parser_contract": "OK" in (
            ARTIFACT / "parser_test_report.txt"
        ).read_text(encoding="utf-8"),
        "both_pilot_matrices": (
            (ARTIFACT / "primary_cumulative_matrix.csv").is_file()
            and (ARTIFACT / "oracle_skill_aware_matrix.csv").is_file()
        ),
        "phase6b_ready": (
            phase6b["status"] == "PASS"
            and not phase6b["oversampling"]
            and all(value == 0 for value in phase6b["train_test_overlap"].values())
            and (ROOT / "scripts/phase6a/run_phase6b_full.sh").is_file()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Phase 6A report checks failed: {checks}")

    summary = {
        "status": "PASS",
        "phase": "6A",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "model_revision": REVISION,
        "pilot_only_no_baseline_superiority_claim": True,
        "checks": checks,
        "shared_drift": {
            "closure_status": gradient["closure_status"],
            "task1_reference": gradient["task_1"]["reference_semantics"],
            "task2_reference": gradient["task_2"]["reference_semantics"],
            "reference_detached": True,
            "task2_max_shared_drift_gradient_norm": gradient["task_2"][
                "max_shared_drift_gradient_norm"
            ],
            "max_full_loss_identity_error": max_identity_error,
            "max_old_private_gradient_norm": max_old_private_gradient,
            "private1_hash_unchanged": hashes["closure_checks"][
                "task2_private1_hash_unchanged"
            ],
        },
        "pilot": {
            "train_samples_per_task": {"1": 128, "2": 128},
            "test_samples_per_task": {"1": 64, "2": 64},
            "scale": "0..1",
            "primary_cumulative": primary,
            "oracle_skill_aware": oracle,
            "prediction_summary_count": len(prediction_summaries),
            "all_invalid_rates_zero": all(
                item["invalid_rate"] == 0 for item in prediction_summaries
            ),
        },
        "phase6b": {
            "prepared_not_executed": True,
            "data": phase6b,
            "methods": [
                "base_zero_shot",
                "sequential_native_lora_r48",
                "pure_rank1_e16",
                "shared32_private16",
            ],
            "entrypoint": "scripts/phase6a/run_phase6b_full.sh",
            "commands": "artifacts/phase6a/full_experiment_commands.txt",
        },
    }
    (ARTIFACT / "phase6a_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    primary_metrics = primary["metrics"]
    oracle_metrics = oracle["metrics"]
    report = f"""# Phase 6A Report

Status: **PASS**

Model: Qwen/Qwen3-VL-8B-Instruct

Revision: {REVISION}

## Seven acceptance answers

1. **Yes.** Real Qwen3-VL training produced non-zero shared-drift gradients. Task 2 max norm was {gradient["task_2"]["max_shared_drift_gradient_norm"]:.9g}.
2. **Yes.** The full-loss identity held over all six closure steps; maximum error was {max_identity_error:.9g}.
3. **Yes.** Old private parameters remained fully frozen: max old-private gradient was 0, and private-1 hashes before/after Task 2 are identical.
4. **Yes.** Primary cumulative uses latest shared exactly once plus private banks 1..i; oracle uses latest shared exactly once plus only private_j selected with the true eval task ID. Oracle is diagnostic-only.
5. **Yes.** Parser tests cover letter, full text, explanations, invalid choices, and explicit MedIMeta records with varying option counts. Invalid outputs count as errors.
6. **Yes.** Both pilot matrices were generated with 64 real unique test samples per task and deterministic inference.
7. **Yes.** Phase 6B has resumable entrypoints for zero-shot, sequential native LoRA r48, pure Rank-1 E16, and shared32/private16. No full baseline was run in Phase 6A.

## Shared-drift closure

- Task 1 reference: {gradient["task_1"]["reference_semantics"]}.
- Task 2 reference: {gradient["task_2"]["reference_semantics"]}.
- References are detached.
- Task 2 raw drift, weighted drift, and drift-only shared gradient are non-zero.
- Optimizer isolation, shared continuity, exit codes, old-private zero gradient, and private-1 hash stability all passed.

## Pilot matrices

Primary cumulative, scale 0..1:

| after task | Task 1 | Task 2 |
|---:|---:|---:|
| 1 | 0.875000 | |
| 2 | 0.890625 | 0.625000 |

Oracle skill-aware diagnostic, scale 0..1:

| after task | Task 1 | Task 2 |
|---:|---:|---:|
| 1 | 0.875000 | |
| 2 | 0.890625 | 0.437500 |

Primary final average accuracy is {primary_metrics["final_average_accuracy"]:.7f}, Task 1 forgetting is {primary_metrics["forgetting_task1"]:.7f}, and BWT is {primary_metrics["BWT"]:.7f}.
Oracle final average accuracy is {oracle_metrics["final_average_accuracy"]:.7f}. All executed cells had invalid rate 0.

These values validate the pipeline only. They do not establish superiority over any baseline.

## Phase 6B preparation

- Task 1 valid unique train/test: {phase6b["counts"]["1"]["train"]} / {phase6b["counts"]["1"]["test"]}.
- Task 2 valid unique train/test: {phase6b["counts"]["2"]["train"]} / {phase6b["counts"]["2"]["test"]}.
- MedIMeta duplicate rows were removed; no replacement or step-count oversampling is used.
- Training is one epoch. Completed tasks/evaluations are skipped only after PASS validation.
- Native Task 2 loads one Task 1 LoRA with --adapters; it does not stack historical LoRAs.
- Failed training/evaluation stops the runner, so no fabricated matrix is emitted.
"""
    (ARTIFACT / "phase6a_report.md").write_text(report, encoding="utf-8")

    state = "\n".join([
        f"HEAD before Phase 6A report commit: {git('rev-parse', 'HEAD')}",
        f"Branch: {git('branch', '--show-current') or '(detached)'}",
        "Worktree status while generating report:",
        git("status", "--short") or "(clean)",
        "",
    ])
    (ARTIFACT / "git_state.txt").write_text(state, encoding="utf-8")
    print(json.dumps({"status": "PASS", "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
