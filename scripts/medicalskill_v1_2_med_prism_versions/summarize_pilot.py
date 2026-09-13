#!/usr/bin/env python3
"""Summarize train-derived Primary/Oracle pilot diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def score(payload):
    for key in (
        "score", "accuracy", "all_concept_micro_f1", "micro_f1",
        "combined_score", "mean_iou", "final_answer_accuracy",
    ):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    metrics = payload.get("metrics")
    return score(metrics) if isinstance(metrics, dict) else None


def response(row):
    for key in ("prediction", "response", "generated_text", "output", "assistant_content"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--method-version", choices=("1.1", "1.2"), required=True)
    args = parser.parse_args()
    evaluation = args.output_root / "evaluation"
    cells = {}
    task_outputs = {"concept": {}, "grounding": {}}
    for path in sorted(evaluation.glob("stage_*/*.summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        mode = payload.get("mode")
        task = payload.get("eval_task") or payload.get("task_id")
        stage = payload.get("after_task") or payload.get("stage")
        if mode in {"primary_cumulative", "oracle_skill_aware"} and task and stage:
            cells[f"stage_{int(stage):02d}:{mode}:task_{int(task):02d}"] = {
                "score": score(payload),
                "summary": str(path),
            }
    for path in sorted(evaluation.glob("stage_*/*.predictions.jsonl")):
        name = path.name.lower()
        if "task_03" in name or "task3" in name:
            kind = "concept"
        elif "task_04" in name or "task4" in name:
            kind = "grounding"
        else:
            continue
        values = [
            response(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        task_outputs[kind][str(path)] = {
            "count": len(values),
            "mean_output_chars": sum(map(len, values)) / max(len(values), 1),
            "empty_output_count": sum(not value.strip() for value in values),
            "reasoning_prefix_count": sum(value.lstrip().lower().startswith("reasoning:") for value in values),
        }
    gaps = {}
    for key, item in cells.items():
        if ":primary_cumulative:" not in key:
            continue
        oracle_key = key.replace(":primary_cumulative:", ":oracle_skill_aware:")
        if oracle_key in cells and item["score"] is not None and cells[oracle_key]["score"] is not None:
            gaps[key] = item["score"] - cells[oracle_key]["score"]
    report = {
        "status": "PASS",
        "method": "Med-PRISM",
        "method_version": args.method_version,
        "data_scope": "locked_training_split_pilot_diagnostic_only",
        "formal_test_used_for_tuning": False,
        "cells": cells,
        "primary_minus_oracle_gaps": gaps,
        "output_interface_diagnostics": task_outputs,
    }
    target = args.output_root / "pilot_diagnostic_summary.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
