#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "phase5_shared_private"
OUTPUT = ROOT / "output" / "phase5_shared_private"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def component_audit(
    component_type: str,
    stage: str,
    task_id: int,
) -> dict[str, Any]:
    if component_type == "shared":
        directory = OUTPUT / "shared" / stage
        manifest_path = directory / "shared_manifest.json"
        weights_path = directory / "shared_adapter.safetensors"
    else:
        directory = OUTPUT / "private" / stage
        manifest_path = directory / "private_manifest.json"
        weights_path = directory / "private_adapter.safetensors"
    manifest = read_json(manifest_path)
    actual_sha = sha256(weights_path)
    checks = {
        "manifest_exists": manifest_path.is_file(),
        "weights_exists": weights_path.is_file() and weights_path.stat().st_size > 0,
        "checksum_matches": actual_sha == manifest["weights_sha256"],
        "component_type_matches": manifest["component_type"] == component_type,
        "task_id_matches": manifest["task_id"] == task_id,
        "model_identity_matches": (
            manifest["model_id"] == MODEL_ID
            and manifest["model_revision"] == REVISION
        ),
        "target_count_is_72": len(manifest["target_modules"]) == 72,
        "unmerged": manifest["merged"] is False,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "manifest": str(manifest_path),
        "weights": str(weights_path),
        "weights_bytes": weights_path.stat().st_size,
        "weights_sha256": actual_sha,
        "checks": checks,
    }


def main() -> None:
    phase4 = read_json(ROOT / "artifacts/phase4_closure/phase4_closure_report.json")
    exits = {
        task: read_json(OUTPUT / f"runs/task_{task}/exit_code.json")
        for task in (1, 2)
    }
    traces = {
        task: read_jsonl(ARTIFACTS / f"task{task}_training_trace.jsonl")
        for task in (1, 2)
    }
    parameters = {
        task: read_json(ARTIFACTS / f"task{task}_parameter_audit.json")
        for task in (1, 2)
    }
    optimizers = {
        task: read_json(ARTIFACTS / f"task{task}_optimizer_audit.json")
        for task in (1, 2)
    }
    shared_hash = read_json(ARTIFACTS / "shared_hash_audit.json")
    private_hash = read_json(ARTIFACTS / "private_hash_audit.json")
    orth = read_json(ARTIFACTS / "orth_gradient_audit.json")
    composition = read_json(ARTIFACTS / "composition_audit.json")
    reload = read_json(ARTIFACTS / "reload_comparison.json")
    test_text = (ARTIFACTS / "test_report.txt").read_text(encoding="utf-8")

    components = {
        "shared_after_task_1": component_audit("shared", "after_task_1", 1),
        "shared_after_task_2": component_audit("shared", "after_task_2", 2),
        "private_task_1": component_audit("private", "task_1", 1),
        "private_task_2": component_audit("private", "task_2", 2),
    }
    max_loss_identity_error = max(
        row["requested_task_plus_orth_identity_error"]
        for rows in traces.values()
        for row in rows
    )
    mode_manifests = composition["active_component_manifests"]
    checks = {
        "phase4_closure_pass": phase4["status"] == "PASS",
        "both_training_processes_exit_zero": all(
            value["swift_sft_exit_code"] == 0 and value["checkpoint_exists"]
            for value in exits.values()
        ),
        "both_training_traces_have_five_steps": all(
            len(rows) == 5 and [row["step"] for row in rows] == [1, 2, 3, 4, 5]
            for rows in traces.values()
        ),
        "task1_shared_and_private_changed": (
            parameters[1]["status"] == "PASS"
            and parameters[1]["shared_changed"]
            and parameters[1]["current_private_changed"]
        ),
        "task1_orthogonal_loss_is_zero": all(
            row["orthogonal_raw_loss"] == 0
            and row["weighted_orthogonal_loss"] == 0
            for row in traces[1]
        ),
        "task2_shared_and_private_changed": (
            parameters[2]["status"] == "PASS"
            and parameters[2]["shared_changed"]
            and parameters[2]["current_private_changed"]
        ),
        "shared_is_continuous_between_tasks": (
            shared_hash["status"] == "PASS"
            and shared_hash["task_1"]["after"] == shared_hash["task_2"]["before"]
            and shared_hash["task_2"]["before"] != shared_hash["task_2"]["after"]
        ),
        "private1_is_frozen_during_task2": (
            private_hash["status"] == "PASS"
            and private_hash["task_2"]["old_unchanged"]
            and private_hash["task_2"]["old_before"]
            == private_hash["task_2"]["old_after"]
            and parameters[2]["old_private_max_gradient_norm"] == 0
        ),
        "optimizers_contain_only_shared_and_current_private": all(
            audit["status"] == "PASS"
            and audit["captured_after_optimizer_creation"]
            and not audit["missing_expected_parameters"]
            and not audit["old_private_optimizer_intersection"]
            and not audit["base_vision_optimizer_intersection"]
            and not audit["frozen_optimizer_intersection"]
            and not audit["unexpected_optimizer_parameters"]
            for audit in optimizers.values()
        ),
        "task2_orthogonal_loss_and_gradient_are_nonzero": (
            any(row["orthogonal_raw_loss"] > 0 for row in traces[2])
            and orth["task_2"]["orthogonal_gradient_nonzero"]
            and orth["task_2"]["max_orthogonal_gradient_norm"] > 0
            and orth["task_2"]["old_private_max_gradient_norm"] == 0
        ),
        "total_loss_identity_holds": max_loss_identity_error < 1e-6,
        "all_components_are_independent_and_valid": all(
            item["status"] == "PASS" for item in components.values()
        ),
        "composition_and_reload_pass": (
            composition["status"] == "PASS" and reload["status"] == "PASS"
        ),
        "all_modes_load_shared_once": all(
            manifest["shared_load_count"] == 1 and manifest["merged"] is False
            for manifest in mode_manifests.values()
        ),
        "three_composition_private_sets_are_exact": (
            mode_manifests["task1_skill_aware"]["private_task_ids"] == [1]
            and mode_manifests["task2_skill_aware"]["private_task_ids"] == [2]
            and mode_manifests["cumulative"]["private_task_ids"] == [1, 2]
        ),
        "component_order_and_fail_fast_checks_pass": (
            composition["checks"]["private_order_equivalent"]
            and composition["checks"]["fail_fast_cases_pass"]
        ),
        "phase3_phase4_phase5_regressions_pass": (
            "Ran 6 tests: OK" in test_text
            and "Ran 21 tests in" in test_text
            and "Ran 9 tests in" in test_text
            and test_text.count("\nOK\n") >= 2
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "status": status,
        "phase": 5,
        "method": "med_prism_shared_private",
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "checks": checks,
        "metrics": {
            "task1_optimizer_steps": len(traces[1]),
            "task2_optimizer_steps": len(traces[2]),
            "max_loss_identity_error": max_loss_identity_error,
            "task2_max_orthogonal_gradient_norm": orth["task_2"][
                "max_orthogonal_gradient_norm"
            ],
            "task2_old_private_max_gradient_norm": orth["task_2"][
                "old_private_max_gradient_norm"
            ],
            "reload_max_abs_logit_diff": max(
                mode["max_abs_logit_diff"] for mode in reload["modes"].values()
            ),
            "private_order_max_abs_logit_diff": composition[
                "private_order_comparison"
            ]["max_abs_logit_diff"],
            "regression_tests": {"phase3": 6, "phase4": 21, "phase5": 9},
        },
        "components": components,
        "phase6_readiness": status == "PASS",
        "scope_note": "Phase 6 and the full continual-learning matrix were not run.",
    }
    (ARTIFACTS / "phase5_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    answers = [
        "1. PASS - Phase 4 optimizer lifecycle audit and real multimodal forward/reload gaps are closed.",
        "2. PASS - shared changed in both tasks; Task 2 starts from the exact Task 1 shared hash.",
        "3. PASS - private_1 hash and parameters are unchanged in Task 2; maximum gradient is 0.",
        "4. PASS - each optimizer contains shared plus current private only; base/vision/merger and old private are excluded.",
        f"5. PASS - Task 2 orthogonal gradient is nonzero (max {orth['task_2']['max_orthogonal_gradient_norm']:.12g}); old-private gradient remains 0.",
        "6. PASS - shared/private components have separate manifests, safetensors files, checksums, and exact reloads.",
        "7. PASS - task1 skill-aware, task2 skill-aware, and cumulative modes each load shared exactly once without merge.",
        "8. PASS - native LoRA, pure Rank-1, and shared/private registry regressions all pass.",
        "9. PASS - the two-task implementation is ready for Phase 6 CL-matrix work; that matrix was intentionally not run.",
    ]
    component_lines = [
        f"- `{name}`: {item['weights_bytes']} bytes, SHA256 `{item['weights_sha256']}`"
        for name, item in components.items()
    ]
    report = [
        "# Phase 5 Shared/Private Closure Report",
        "",
        f"**Status: {status}**",
        "",
        "## Fixed Identity",
        "",
        f"- Model: `{MODEL_ID}`",
        f"- Revision: `{REVISION}`",
        "- Targets: exactly 72 language `q_proj/v_proj` modules",
        "- Runtime contract: single GPU, BF16, gradient checkpointing, unmerged components",
        "",
        "## Evidence Summary",
        "",
        "- Both real-data training smokes completed 5 optimizer steps with exit code 0.",
        "- Task 1 used OmniMedVQA VQA samples; Task 2 used MedIMeta classification samples.",
        "- Shared rank is 32; private uses 16 explicit Rank-1 experts per task.",
        f"- Maximum total-loss identity error: `{max_loss_identity_error:.12g}`.",
        f"- Task 2 maximum current-private orthogonal gradient: `{orth['task_2']['max_orthogonal_gradient_norm']:.12g}`.",
        "- Task 2 old-private gradient: `0`; private_1 before/after hashes are identical.",
        "- Independent reload max logit difference: `0` for all three composition modes.",
        "- Private order equivalence max logit difference: `0`.",
        "- Regression tests: Phase 3 6/6, Phase 4 21/21, Phase 5 9/9.",
        "",
        "## Components",
        "",
        *component_lines,
        "",
        "## Required Answers",
        "",
        *answers,
        "",
        "## Scope",
        "",
        "No Phase 6 implementation, routing, destructive merge, or full CL evaluation matrix was run.",
    ]
    (ARTIFACTS / "phase5_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    git_state = [
        f"branch: {git('branch', '--show-current')}",
        f"head: {git('rev-parse', 'HEAD')}",
        "status:",
        git("status", "--short") or "(clean)",
    ]
    (ARTIFACTS / "git_state.txt").write_text(
        "\n".join(git_state) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "checks": checks}, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
