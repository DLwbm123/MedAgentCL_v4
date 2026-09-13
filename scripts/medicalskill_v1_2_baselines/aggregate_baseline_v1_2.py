#!/usr/bin/env python3
"""Aggregate a formal 15-cell matrix for any harness-compatible baseline."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from scripts.medicalskill_v1_2_baselines.baseline_harness import (
    EVALUATION_PROTOCOL,
    atomic_json,
    load_json,
    validate_completion,
    validate_dataset_lock,
)

TASKS = ["VQA", "Diagnosis", "Concept", "Grounding", "Reasoning"]


def metric_path(root: Path, stage: int, task: int) -> Path:
    return (
        root
        / "evaluation"
        / f"stage_{stage:02d}"
        / f"primary_cumulative_task_{task:02d}.summary.json"
    )


def build_matrix(root: Path) -> list[list[float | None]]:
    return [
        [
            float(load_json(metric_path(root, stage, task))["primary_score"])
            if task <= stage
            else None
            for task in range(1, 6)
        ]
        for stage in range(1, 6)
    ]


def write_matrix(path: Path, values: list[list[float | None]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["after_stage", *TASKS, "average"])
        for stage, row in enumerate(values, 1):
            seen = [value for value in row if value is not None]
            if len(seen) != stage:
                raise RuntimeError(f"Stage {stage} does not contain {stage} cells")
            writer.writerow(
                [
                    stage,
                    *[
                        "" if value is None else f"{value:.8f}"
                        for value in row
                    ],
                    f"{sum(seen) / len(seen):.8f}",
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--stage-root-name", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-limit", type=int, default=0)
    args = parser.parse_args()
    root = args.output_root.resolve()
    data = args.data_root.resolve()
    dataset = validate_dataset_lock(data)
    run_manifest = load_json(root / "run_manifest.json")
    if run_manifest.get("status") != "PASS":
        raise RuntimeError("Run manifest is not PASS")
    if run_manifest.get("method") != args.method:
        raise RuntimeError("Method differs from run manifest")
    immutable_hash = run_manifest["immutable_config_sha256"]

    values = build_matrix(root)
    write_matrix(root / "primary_cumulative_matrix.csv", values)
    safe_method = re.sub(r"[^a-zA-Z0-9_.-]+", "_", args.method)
    write_matrix(root / f"{safe_method}_lower_triangular_matrix.csv", values)
    final = [float(value) for value in values[-1] if value is not None]
    diagonal = [float(values[index][index]) for index in range(5)]
    best = [
        max(float(values[stage][task]) for stage in range(task, 5))
        for task in range(5)
    ]
    forgetting = [best[index] - final[index] for index in range(5)]
    bwt_terms = [final[index] - diagonal[index] for index in range(4)]
    metrics = {
        "status": "PASS",
        "method": args.method,
        "scale": "0..1",
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "matrix_cells": 15,
        "final_average_score": sum(final) / 5,
        "per_task_best_score": best,
        "per_task_final_score": final,
        "per_task_forgetting": forgetting,
        "average_forgetting": sum(forgetting) / 5,
        "BWT": sum(bwt_terms) / len(bwt_terms),
        "BWT_terms": bwt_terms,
        "FWT": None,
        "FWT_status": "NOT_COMPUTED_NO_FUTURE_TASK_EVALUATION",
    }
    atomic_json(root / "cl_metrics.json", metrics)

    stages: dict[str, Any] = {}
    total_cells = 0
    total_main = 0.0
    total_aux = 0.0
    total_gpu_hours = 0.0
    total_eval = 0.0
    for stage in range(1, 6):
        completion_path = (
            root
            / args.stage_root_name
            / f"stage_{stage:02d}"
            / "completion.json"
        )
        completion = validate_completion(
            completion_path,
            method=args.method,
            stage=stage,
            immutable_config_sha256=immutable_hash,
        )
        evaluation = load_json(
            root
            / "evaluation"
            / f"stage_{stage:02d}"
            / "stage_evaluation_summary.json"
        )
        active = evaluation.get("active_adapter", {})
        ok = (
            evaluation.get("status") == "PASS"
            and evaluation.get("method") == args.method
            and evaluation.get("evaluation_protocol") == EVALUATION_PROTOCOL
            and evaluation.get("evaluated_task_ids") == list(range(1, stage + 1))
            and evaluation.get("cell_count") == stage
            and active.get("adapter_load_count") == stage
            and active.get("stage1_adapters_active") is False
            and active.get("oracle_task_id") is False
        )
        if not ok:
            raise RuntimeError(f"Evaluation contract failed at stage {stage}")
        runtime = completion["runtime_accounting"]
        total_main += float(runtime["main_train_seconds"])
        total_aux += float(runtime["auxiliary_seconds"])
        total_gpu_hours += float(runtime["training_gpu_hours"])
        total_eval += float(evaluation.get("elapsed_seconds", 0.0))
        total_cells += int(evaluation["cell_count"])
        stages[str(stage)] = {
            "status": "PASS",
            "completion": str(completion_path),
            "stage2_checkpoint": completion["stage2_checkpoint"],
            "adapter_weights_sha256": completion["stage2_adapter"][
                "adapter_weights_sha256"
            ],
            "runtime_accounting": runtime,
            "parameter_accounting": completion["parameter_accounting"],
        }
    gates = {
        "dataset_acceptance": dataset["status"] == "PASS",
        "run_manifest": run_manifest["status"] == "PASS",
        "all_stage_contracts": len(stages) == 5,
        "exactly_15_lower_triangular_cells": total_cells == 15,
        "stage2_only_primary_inference": True,
        "oracle_task_id": False,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    summary = {
        "status": status,
        "format_version": "medicalskill_cl_baseline_summary_v1",
        "method": args.method,
        "data_root": str(data),
        "output_root": str(root),
        "eval_limit_per_task": args.eval_limit,
        "gates": gates,
        "stages": stages,
        "metrics": metrics,
        "runtime": {
            "main_train_seconds_t1_t5": total_main,
            "auxiliary_seconds_t1_t5": total_aux,
            "continual_training_seconds_t1_t5": total_main + total_aux,
            "training_gpu_hours_t1_t5": total_gpu_hours,
            "evaluation_seconds_t1_t5": total_eval,
            "evaluation_reported_separately": True,
        },
        "historical_memory": run_manifest["historical_memory"],
    }
    atomic_json(root / "baseline_experiment_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
