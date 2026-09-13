#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/phase6a"
RUNTIME = ARTIFACT / "closure_runtime"
OUTPUT = Path("/remote-home/wangbomin/medagentcl_v4_phase6a/closure")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    traces = {
        task: read_jsonl(RUNTIME / f"task{task}_training_trace.jsonl")
        for task in (1, 2)
    }
    for task, rows in traces.items():
        destination = ARTIFACT / f"shared_drift_training_trace_task{task}.jsonl"
        destination.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    gradient = read_json(RUNTIME / "shared_drift_gradient_audit.json")
    hashes = read_json(RUNTIME / "shared_drift_hash_audit.json")
    optimizers = {
        task: read_json(RUNTIME / f"task{task}_optimizer_audit.json")
        for task in (1, 2)
    }
    exits = {
        task: read_json(OUTPUT / f"runs/task_{task}/exit_code.json")
        for task in (1, 2)
    }
    checks = {
        "both_processes_exit_zero": all(
            value["swift_sft_exit_code"] == 0
            and value["shared_manifest_exists"]
            and value["private_manifest_exists"]
            for value in exits.values()
        ),
        "both_tasks_have_at_least_three_steps": all(
            len(rows) >= 3 for rows in traces.values()
        ),
        "full_loss_identity_within_tolerance": max(
            row["full_loss_identity_error"]
            for rows in traces.values()
            for row in rows
        )
        < 1e-6,
        "task2_shared_drift_raw_nonzero": any(
            row["shared_drift_raw_loss"] > 0 for row in traces[2]
        ),
        "task2_weighted_shared_drift_nonzero": any(
            row["weighted_shared_drift_loss"] > 0 for row in traces[2]
        ),
        "task2_shared_drift_gradient_nonzero": gradient["task_2"][
            "shared_drift_gradient_nonzero"
        ],
        "task2_old_private_gradient_zero": all(
            row["old_private_gradient_norm"] == 0 for row in traces[2]
        ),
        "task2_private1_hash_unchanged": (
            hashes["task_2"]["old_private_unchanged"]
            and hashes["task_2"]["old_private_before"]
            == hashes["task_2"]["old_private_after"]
        ),
        "shared_continuity": (
            hashes["task_1"]["shared_after"]
            == hashes["task_2"]["shared_before"]
        ),
        "optimizer_isolation": all(
            audit["status"] == "PASS"
            and not audit["old_private_optimizer_intersection"]
            and not audit["base_vision_optimizer_intersection"]
            and not audit["unexpected_optimizer_parameters"]
            for audit in optimizers.values()
        ),
        "reference_is_detached": all(
            gradient[f"task_{task}"]["reference_detached"] for task in (1, 2)
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    gradient["closure_status"] = status
    gradient["closure_checks"] = checks
    hashes["closure_status"] = status
    hashes["closure_checks"] = {
        key: checks[key]
        for key in (
            "task2_private1_hash_unchanged",
            "shared_continuity",
        )
    }
    (ARTIFACT / "shared_drift_gradient_audit.json").write_text(
        json.dumps(gradient, indent=2) + "\n",
        encoding="utf-8",
    )
    (ARTIFACT / "shared_drift_hash_audit.json").write_text(
        json.dumps(hashes, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Phase 6A Shared-Drift Closure",
        "",
        f"**Status: {status}**",
        "",
        "## Reference Semantics",
        "",
        "- Task 1 reference: detached shared initialization state.",
        "- Task 2 reference: detached frozen shared_after_task_1 loaded from its explicit manifest.",
        "- Drift is normalized squared distance over all 72 shared q_proj/v_proj adapters.",
        "- shared gradient hook scale remains 0.1; drift weight is 0.01.",
        "",
        "## Checks",
        "",
        *[
            f"- {name}: {'PASS' if passed else 'FAIL'}"
            for name, passed in checks.items()
        ],
    ]
    (ARTIFACT / "shared_drift_closure.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "checks": checks}, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
