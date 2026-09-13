#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from med_prism.evaluation.cl import build_cl_metrics


ROOT = Path("/remote-home/wangbomin/medagentcl_v4_phase6b")


def read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise RuntimeError(f"Evaluation is incomplete: {path}")
    return payload


def write_matrix(path: Path, cells: list[dict]) -> None:
    indexed = {(cell["after_task"], cell["eval_task"]): cell for cell in cells}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "after_task",
            "R_task1_accuracy",
            "R_task1_invalid_rate",
            "R_task2_accuracy",
            "R_task2_invalid_rate",
            "scale",
        ])
        for after_task in (1, 2):
            first = indexed[(after_task, 1)]
            second = indexed.get((after_task, 2))
            writer.writerow([
                after_task,
                first["accuracy"],
                first["invalid_rate"],
                "" if second is None else second["accuracy"],
                "" if second is None else second["invalid_rate"],
                "0..1",
            ])


def cells(method: str, mode: str) -> list[dict]:
    prediction = ROOT / method / "predictions"
    result = []
    for after_task, eval_task in ((1, 1), (2, 1), (2, 2)):
        name = f"{mode}_after_task{after_task}_eval_task{eval_task}.summary.json"
        if mode == "oracle_skill_aware" and after_task == 1:
            name = "primary_cumulative_after_task1_eval_task1.summary.json"
        payload = read(prediction / name)
        result.append({
            "after_task": after_task,
            "eval_task": eval_task,
            "accuracy": payload["accuracy"],
            "invalid_rate": payload["invalid_rate"],
            "total": payload["total"],
            "prediction_path": payload["prediction_path"],
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=[
            "base_zero_shot",
            "sequential_native_lora_r48",
            "pure_rank1_e16",
            "shared32_private16",
        ],
        required=True,
    )
    args = parser.parse_args()
    output = ROOT / args.method
    if args.method == "base_zero_shot":
        payloads = [
            read(output / f"predictions/zero_shot_eval_task{task}.summary.json")
            for task in (1, 2)
        ]
        result = {
            "status": "PASS",
            "method": args.method,
            "scale": "0..1",
            "zero_shot": {
                str(item["eval_task"]): {
                    "accuracy": item["accuracy"],
                    "invalid_rate": item["invalid_rate"],
                    "total": item["total"],
                }
                for item in payloads
            },
        }
        (output / "zero_shot_metrics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2))
        return

    primary = cells(args.method, "primary_cumulative")
    write_matrix(output / "primary_cumulative_matrix.csv", primary)
    result = {
        "status": "PASS",
        "method": args.method,
        "scale": "0..1",
        "primary_cumulative": {
            "cells": primary,
            "metrics": build_cl_metrics(primary),
        },
    }
    if args.method == "shared32_private16":
        oracle = cells(args.method, "oracle_skill_aware")
        write_matrix(output / "oracle_skill_aware_matrix.csv", oracle)
        result["oracle_skill_aware"] = {
            "diagnostic_only": True,
            "cells": oracle,
            "metrics": build_cl_metrics(oracle),
        }
    (output / "cl_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
