#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--rerun-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.artifact_root
    optimizer = {
        str(task_id): read_json(
            root / f"phase4_rank1_task{task_id}_optimizer_audit.json"
        )
        for task_id in (1, 2)
    }
    forward = read_json(root / "phase4_rank1_forward_reload_comparison.json")
    scaling = read_json(root / "phase4_rank1_scaling_regression.json")
    cpu_text = (root / "phase4_cpu_test_report.txt").read_text(encoding="utf-8")
    exits = {
        str(task_id): read_json(
            args.rerun_root / f"task_{task_id:02d}" / "exit_code.json"
        )
        for task_id in (1, 2)
    }
    checks = {
        "optimizer_task_1_pass": optimizer["1"].get("status") == "PASS",
        "optimizer_task_2_pass": optimizer["2"].get("status") == "PASS",
        "optimizer_only_current_task": all(
            not audit.get("missing_current_parameters")
            and not audit.get("old_optimizer_intersection")
            and not audit.get("frozen_optimizer_intersection")
            and not audit.get("unexpected_optimizer_parameters")
            for audit in optimizer.values()
        ),
        "real_multimodal_forward_reload_pass": forward.get("status") == "PASS",
        "adapter_changes_real_logits": all(
            value > 1e-6
            for value in forward["base_vs_adapter_max_abs_logit_diff"].values()
        ),
        "two_independent_reloads_exact": all(
            value == 0.0
            for value in forward["reload_1_vs_reload_2_max_abs_logit_diff"].values()
        ),
        "scaling_e_1_4_16_tasks_1_2_3_pass": scaling.get("status") == "PASS",
        "cpu_tests_21_of_21_pass": "Ran 21 tests" in cpu_text and "OK" in cpu_text,
        "both_five_step_runs_exit_zero": all(
            item.get("swift_sft_exit_code") == 0 for item in exits.values()
        ),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checks": checks,
        "optimizer_audits": optimizer,
        "forward_reload_comparison": forward,
        "scaling_regression": scaling,
        "training_exit_codes": exits,
    }
    json_path = root / "phase4_closure_report.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Phase 4 Closure Report",
        "",
        f"- Status: **{payload['status']}**",
        "- Scope: optimizer ownership, explicit scaling, and real Qwen3-VL multimodal reload equivalence.",
        "- Training: Task 1 and Task 2 closure reruns used five optimizer steps each.",
        "",
        "## Checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if value else 'BLOCKED'}: {name}"
        for name, value in checks.items()
    )
    (root / "phase4_closure_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "checks": checks}))
    if payload["status"] != "PASS":
        raise SystemExit("Phase 4 closure remains blocked")


if __name__ == "__main__":
    main()
