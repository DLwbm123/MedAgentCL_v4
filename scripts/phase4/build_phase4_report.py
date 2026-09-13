#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def task_summary(root: Path, task_id: int) -> dict:
    task = root / f"task_{task_id:02d}"
    checkpoint = task / "train" / "checkpoint-5"
    exit_code = read_json(task / "exit_code.json")
    target = read_json(task / "target_module_audit.json")
    trainer = read_json(task / "trainer_activation_audit.json")
    gradient = read_json(task / "orth_gradient_audit.json")
    manifest = read_json(checkpoint / "rank1_manifest.json")
    trace = read_jsonl(task / "med_prism_loss_trace.jsonl")
    return {
        "task_id": task_id,
        "exit_code": exit_code,
        "target_audit": target,
        "trainer_audit": trainer,
        "gradient_audit": gradient,
        "manifest": manifest,
        "loss_trace_count": len(trace),
        "max_loss_identity_error": max(
            (row["total_loss_identity_error"] for row in trace), default=None
        ),
        "max_old_expert_grad_norm": max(
            (row["old_expert_grad_norm"] for row in trace), default=0.0
        ),
        "max_orth_gradient_norm": max(
            (row["orth_gradient_norm"] for row in trace), default=0.0
        ),
        "orth_loss_raw": [row["orth_loss_raw"] for row in trace],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    task1 = task_summary(args.output_root, 1)
    task2 = task_summary(args.output_root, 2)
    reload_audit = read_json(
        args.output_root / "task_02" / "real_reload_audit.json"
    )
    checks = {
        "cpu_tests_21_pass": "Ran 21 tests" in (
            args.output_root / "cpu_tests.log"
        ).read_text(encoding="utf-8"),
        "both_swift_runs_exit_zero": all(
            item["exit_code"]["swift_sft_exit_code"] == 0
            for item in (task1, task2)
        ),
        "both_runs_use_project_trainer": all(
            item["trainer_audit"]["is_med_prism_trainer"]
            for item in (task1, task2)
        ),
        "targets_exactly_72_language_qv": all(
            item["target_audit"]["wrapper_count"] == 72
            and item["target_audit"]["q_proj"] == 36
            and item["target_audit"]["v_proj"] == 36
            and item["target_audit"]["vision"] == 0
            and item["target_audit"]["merger"] == 0
            for item in (task1, task2)
        ),
        "task1_orth_is_zero": all(value == 0 for value in task1["orth_loss_raw"]),
        "task2_orth_gradient_nonzero": task2["gradient_audit"]["orth_gradient_nonzero"],
        "old_experts_unchanged_and_zero_grad": (
            task2["gradient_audit"]["old_parameters_unchanged"]
            and task2["max_old_expert_grad_norm"] == 0
        ),
        "loss_identity_holds": all(
            item["max_loss_identity_error"] is not None
            and item["max_loss_identity_error"] < 1e-5
            for item in (task1, task2)
        ),
        "real_qwen_reload_pass": reload_audit["status"] == "PASS",
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checks": checks,
        "model_revision": task2["manifest"]["immutable_revision"],
        "ms_swift_version": task2["manifest"]["ms_swift_version"],
        "task_01": task1,
        "task_02": task2,
        "real_reload_audit": reload_audit,
    }
    args.json_output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Phase 4 Med-PRISM Rank-1 Smoke Report",
        "",
        f"- Status: **{payload['status']}**",
        f"- Qwen3-VL revision: {payload['model_revision']}",
        f"- ms-swift: {payload['ms_swift_version']}",
        "- Task 1 / Task 2: 5 optimizer steps each, one GPU, no DDP.",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if value else 'BLOCKED'}: {name}"
        for name, value in checks.items()
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if payload["status"] != "PASS":
        raise SystemExit("Phase 4 report contains blocked checks")


if __name__ == "__main__":
    main()
