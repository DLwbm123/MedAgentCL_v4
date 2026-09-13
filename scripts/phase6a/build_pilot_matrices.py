#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from med_prism.evaluation.cl import build_cl_metrics, component_task_ids


ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/phase6a"
PREDICTIONS = ARTIFACT / "predictions"


def read_summary(name: str) -> dict:
    path = PREDICTIONS / f"{name}.summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["status"] != "PASS":
        raise RuntimeError(f"Evaluation did not pass: {path}")
    if (
        payload["active_component_manifest"]["shared_load_count"] != 1
        or payload["active_component_manifest"]["merged"]
        or not payload["deterministic_first_sample_replay"]
    ):
        raise RuntimeError(f"Evaluation contract failed: {path}")
    return payload


def cell(payload: dict, *, mode: str) -> dict:
    expected_tasks = component_task_ids(
        mode,
        after_task=payload["after_task"],
        eval_task=payload["eval_task"],
    )
    active = payload["active_component_manifest"]
    if active["private_task_ids"] != expected_tasks:
        raise RuntimeError(
            f"{mode} component mismatch: {active['private_task_ids']} != {expected_tasks}"
        )
    return {
        "mode": mode,
        "after_task": payload["after_task"],
        "eval_task": payload["eval_task"],
        "accuracy": payload["accuracy"],
        "invalid_rate": payload["invalid_rate"],
        "total": payload["total"],
        "prediction_path": payload["prediction_path"],
        "active_component_manifest": active,
    }


def write_matrix(path: Path, cells: list[dict]) -> None:
    indexed = {(item["after_task"], item["eval_task"]): item for item in cells}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "after_task",
                "R_task1_accuracy",
                "R_task1_invalid_rate",
                "R_task2_accuracy",
                "R_task2_invalid_rate",
                "scale",
            ]
        )
        for after_task in (1, 2):
            task1 = indexed[(after_task, 1)]
            task2 = indexed.get((after_task, 2))
            writer.writerow(
                [
                    after_task,
                    task1["accuracy"],
                    task1["invalid_rate"],
                    "" if task2 is None else task2["accuracy"],
                    "" if task2 is None else task2["invalid_rate"],
                    "0..1",
                ]
            )


def main() -> None:
    common_r11 = read_summary("primary_cumulative_after_task1_eval_task1")
    primary = [
        cell(common_r11, mode="primary_cumulative"),
        cell(
            read_summary("primary_cumulative_after_task2_eval_task1"),
            mode="primary_cumulative",
        ),
        cell(
            read_summary("primary_cumulative_after_task2_eval_task2"),
            mode="primary_cumulative",
        ),
    ]
    oracle_r11 = dict(common_r11)
    oracle_r11["mode"] = "oracle_skill_aware"
    oracle = [
        cell(oracle_r11, mode="oracle_skill_aware"),
        cell(
            read_summary("oracle_skill_aware_after_task2_eval_task1"),
            mode="oracle_skill_aware",
        ),
        cell(
            read_summary("oracle_skill_aware_after_task2_eval_task2"),
            mode="oracle_skill_aware",
        ),
    ]
    write_matrix(ARTIFACT / "primary_cumulative_matrix.csv", primary)
    write_matrix(ARTIFACT / "oracle_skill_aware_matrix.csv", oracle)
    metrics = {
        "status": "PASS",
        "scale": "0..1",
        "primary_cumulative": {
            "definition": "latest shared exactly once plus all private banks 1..i; no eval task ID selection",
            "cells": primary,
            "metrics": build_cl_metrics(primary),
        },
        "oracle_skill_aware": {
            "definition": "latest shared exactly once plus private_j selected using true eval task ID; diagnostic only",
            "diagnostic_only": True,
            "cells": oracle,
            "metrics": build_cl_metrics(oracle),
        },
    }
    (ARTIFACT / "cl_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "primary": metrics["primary_cumulative"]["metrics"],
                "oracle": metrics["oracle_skill_aware"]["metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
