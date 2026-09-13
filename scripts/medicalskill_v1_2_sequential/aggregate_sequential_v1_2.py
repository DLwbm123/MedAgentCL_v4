#!/usr/bin/env python3
"""Aggregate the 15-cell MedicalSkill sequential-LoRA matrix."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


TASKS = ["VQA", "Diagnosis", "Concept", "Grounding", "Reasoning"]
METHOD = "sequential_native_lora_r48"
PROTOCOL = "lower_triangular_seen_tasks_v1"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metric_path(root: Path, stage: int, task: int) -> Path:
    return root / "evaluation" / f"stage_{stage:02d}" / f"primary_cumulative_task_{task:02d}.summary.json"


def matrix(root: Path) -> list[list[float | None]]:
    return [
        [float(load(metric_path(root, stage, task))["primary_score"]) if task <= stage else None for task in range(1, 6)]
        for stage in range(1, 6)
    ]


def write_matrix(path: Path, values: list[list[float | None]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["after_stage", *TASKS, "average"])
        for stage, row in enumerate(values, 1):
            seen = [value for value in row if value is not None]
            if len(seen) != stage:
                raise RuntimeError(f"Stage {stage} does not have exactly {stage} cells")
            writer.writerow(
                [stage, *["" if value is None else f"{value:.8f}" for value in row], f"{sum(seen) / len(seen):.8f}"]
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-limit", type=int, default=0)
    args = parser.parse_args()
    root, data = args.output_root.resolve(), args.data_root.resolve()
    values = matrix(root)
    write_matrix(root / "primary_cumulative_matrix.csv", values)
    write_matrix(root / "sequential_lower_triangular_matrix.csv", values)
    final = [float(value) for value in values[-1] if value is not None]
    diagonal = [float(values[index][index]) for index in range(5)]
    best = [max(float(values[stage][task]) for stage in range(task, 5)) for task in range(5)]
    forgetting = [best[index] - final[index] for index in range(5)]
    bwt_terms = [final[index] - diagonal[index] for index in range(4)]
    metrics = {
        "status": "PASS",
        "method": METHOD,
        "ordinary_lora": True,
        "scale": "0..1",
        "evaluation_protocol": PROTOCOL,
        "matrix_cells": 15,
        "final_average_score": sum(final) / 5,
        "per_task_best_score": best,
        "per_task_final_score": final,
        "per_task_forgetting": forgetting,
        "average_forgetting": sum(forgetting) / 5,
        "BWT": sum(bwt_terms) / len(bwt_terms),
        "BWT_terms": bwt_terms,
        "FWT": None,
        "FWT_status": "NOT_COMPUTED_NO_BASE_OR_FUTURE_TASK_EVALUATION",
    }
    write_json(root / "cl_metrics.json", metrics)

    stages = {}
    gates = {
        "dataset_acceptance": load(data / "audits/acceptance_matrix.json").get("status") == "PASS",
        "run_manifest": load(root / "run_manifest.json").get("status") == "PASS",
    }
    total_cells = 0
    for stage in range(1, 6):
        completion = load(root / "sequential" / f"stage_{stage:02d}" / "completion.json")
        evaluation = load(root / "evaluation" / f"stage_{stage:02d}" / "stage_evaluation_summary.json")
        ok = (
            completion.get("status") == "PASS"
            and completion.get("method") == METHOD
            and completion.get("ordinary_lora") is True
            and evaluation.get("status") == "PASS"
            and evaluation.get("evaluation_protocol") == PROTOCOL
            and evaluation.get("evaluated_task_ids") == list(range(1, stage + 1))
            and evaluation.get("cell_count") == stage
            and evaluation.get("active_adapter", {}).get("historical_adapter_stack") is False
        )
        total_cells += int(evaluation.get("cell_count", 0))
        stages[str(stage)] = {
            "status": "PASS" if ok else "FAIL",
            "checkpoint": completion.get("checkpoint"),
            "initialization": completion.get("initialization"),
            "adapter_weights_sha256": completion.get("adapter_weights_sha256"),
            "evaluated_task_ids": evaluation.get("evaluated_task_ids"),
        }
    gates["all_stage_contracts"] = all(value["status"] == "PASS" for value in stages.values())
    gates["exactly_15_lower_triangular_cells"] = total_cells == 15
    status = "PASS" if all(gates.values()) else "FAIL"
    summary = {
        "status": status,
        "method": METHOD,
        "ordinary_lora": True,
        "continual_learning_constraints": False,
        "replay": False,
        "adapter_semantics": "one cumulative LoRA, continued from the previous stage checkpoint",
        "historical_adapter_stack": False,
        "data_root": str(data),
        "output_root": str(root),
        "eval_limit_per_task": args.eval_limit,
        "gates": gates,
        "stages": stages,
        "metrics": metrics,
    }
    write_json(root / "sequential_experiment_summary.json", summary)
    report = [
        "# MedicalSkill-CL-v1.2 sequential native LoRA baseline",
        "",
        f"Status: **{status}**",
        "",
        "- Method: sequential_native_lora_r48; ordinary LoRA: true",
        "- One cumulative LoRA is continued across all five tasks.",
        "- No replay, orthogonal gradients, shared/private split, routing, or CL regularizer.",
        "- Evaluation: exactly 15 seen-task lower-triangular cells; no base/oracle/task-free matrix.",
        f"- Final average score: {metrics['final_average_score']:.6f}",
        f"- Average forgetting: {metrics['average_forgetting']:.6f}",
        f"- BWT: {metrics['BWT']:.6f}",
    ]
    (root / "sequential_experiment_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
