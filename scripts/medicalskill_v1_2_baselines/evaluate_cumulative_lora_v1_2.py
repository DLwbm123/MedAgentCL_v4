#!/usr/bin/env python3
"""Formal evaluator handoff for methods that sum independent PEFT LoRAs."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

ROOT = Path("/root/MedAgentCL_v4")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.medicalskill_v1_2_baselines.baseline_harness import (
    EVALUATION_PROTOCOL,
    MODEL_ID,
    MODEL_REVISION,
    SNAPSHOT,
    inspect_adapter,
)
from scripts.medicalskill_v1_2_med_prism.evaluate_formal_v1_2 import (
    TASK_DIRS,
    TASK_NAMES,
    evaluate_rows,
    generate,
    read_jsonl,
    sha256,
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def cumulative_adapter_manifest(
    adapters: list[Path], method: str, *, expected_rank: int | None = None
) -> dict[str, Any]:
    if not adapters:
        raise ValueError("At least one adapter is required")
    components = []
    targets = None
    total = 0
    for task, checkpoint in enumerate(adapters, 1):
        info = inspect_adapter(checkpoint)
        if expected_rank is not None and info["r"] != expected_rank:
            raise RuntimeError(
                f"Task {task} rank mismatch: {info['r']} != {expected_rank}"
            )
        if targets is None:
            targets = info["target_modules"]
        elif info["target_modules"] != targets:
            raise RuntimeError("Cumulative adapters use different target modules")
        total += info["adapter_parameters"]
        components.append(
            {
                "task": task,
                "role": "stage2",
                "checkpoint": info["checkpoint"],
                "adapter_weights_sha256": info["adapter_weights_sha256"],
                "adapter_config_sha256": info["adapter_config_sha256"],
                "parameters": info["adapter_parameters"],
                "r": info["r"],
                "target_modules": info["target_modules"],
            }
        )
    return {
        "method": method,
        "composition": "base_plus_cumulative_independent_adapters",
        "merge_order": "task_ascending",
        "adapter_load_count": len(components),
        "historical_adapter_stack": True,
        "stage1_adapters_active": False,
        "oracle_task_id": False,
        "active_adapter_parameters": total,
        "target_modules": targets,
        "components": components,
    }


def load_model(adapters: list[Path]):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(str(SNAPSHOT), local_files_only=True)
    if hasattr(processor.image_processor, "min_pixels"):
        processor.image_processor.min_pixels = 200704
    if hasattr(processor.image_processor, "max_pixels"):
        processor.image_processor.max_pixels = 200704
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(SNAPSHOT),
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.requires_grad_(False)
    for checkpoint in adapters:
        peft_model = PeftModel.from_pretrained(
            model, str(checkpoint), is_trainable=False
        )
        model = peft_model.merge_and_unload()
    return model.to("cuda:0").eval(), processor


def read_raw_predictions(
    path: Path, rows: list[dict[str, Any]], active: dict[str, Any]
) -> list[dict[str, Any]]:
    values = read_jsonl(path) if path.is_file() else []
    if len(values) > len(rows):
        raise RuntimeError(f"Prediction file is longer than dataset: {path}")
    for index, value in enumerate(values):
        if value.get("id") != rows[index].get("id"):
            raise RuntimeError(f"Prediction resume ID mismatch at {path}:{index + 1}")
        if value.get("active_adapter") != active:
            raise RuntimeError(f"Prediction resume adapter mismatch at {path}:{index + 1}")
    return values


def append_prediction(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--stage", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--adapter", action="append", type=Path, required=True)
    parser.add_argument("--expected-rank", type=int)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-limit", type=int, default=0)
    args = parser.parse_args()
    if len(args.adapter) != args.stage:
        raise ValueError("Exactly one cumulative adapter per learned task is required")
    if args.eval_limit < 0:
        raise ValueError("--eval-limit must be non-negative")
    adapters = [path.resolve() for path in args.adapter]
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    active = cumulative_adapter_manifest(
        adapters, args.method, expected_rank=args.expected_rank
    )
    stage_dir = output_root / "evaluation" / f"stage_{args.stage:02d}"
    stage_summary = stage_dir / "stage_evaluation_summary.json"
    if stage_summary.is_file():
        previous = json.loads(stage_summary.read_text(encoding="utf-8"))
        if (
            previous.get("status") == "PASS"
            and previous.get("method") == args.method
            and previous.get("evaluation_protocol") == EVALUATION_PROTOCOL
            and previous.get("data_root") == str(data_root)
            and previous.get("eval_limit") == args.eval_limit
            and previous.get("active_adapter") == active
        ):
            print(f"SKIP completed evaluation stage {args.stage}")
            return 0

    torch.manual_seed(42)
    started = time.time()
    model, processor = load_model(adapters)
    summaries: dict[str, Any] = {}
    for task_id in range(1, args.stage + 1):
        dataset = data_root / TASK_DIRS[task_id] / "test.jsonl"
        rows = read_jsonl(dataset, args.eval_limit)
        if not rows:
            raise RuntimeError(f"Empty evaluation dataset: {dataset}")
        stem = f"primary_cumulative_task_{task_id:02d}"
        raw_path = stage_dir / f"{stem}.raw_predictions.jsonl"
        raw_values = read_raw_predictions(raw_path, rows, active)
        for index in range(len(raw_values), len(rows)):
            prediction = generate(model, processor, rows[index], task_id)
            value = {
                "id": rows[index]["id"],
                "prediction": prediction,
                "active_adapter": active,
            }
            append_prediction(raw_path, value)
            raw_values.append(value)
        predictions = [value["prediction"] for value in raw_values]
        metric, details = evaluate_rows(task_id, rows, predictions)
        detail_path = stage_dir / f"{stem}.predictions.jsonl"
        detail_path.write_text(
            "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in details),
            encoding="utf-8",
        )
        metric.update(
            {
                "mode": "primary_cumulative",
                "stage": args.stage,
                "eval_task": task_id,
                "task_name": TASK_NAMES[task_id],
                "method": args.method,
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "dataset": str(dataset),
                "dataset_sha256": sha256(dataset),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "active_adapter": active,
                "predictions": str(detail_path),
                "predictions_sha256": sha256(detail_path),
                "raw_predictions": str(raw_path),
                "raw_predictions_sha256": sha256(raw_path),
                "fresh_process_reload": True,
                "oracle_task_id": False,
            }
        )
        atomic_json(stage_dir / f"{stem}.summary.json", metric)
        summaries[stem] = metric
    status = (
        "PASS"
        if len(summaries) == args.stage
        and all(value.get("status") == "PASS" for value in summaries.values())
        else "FAIL"
    )
    final = {
        "status": status,
        "stage": args.stage,
        "method": args.method,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "matrix_scope": "seen_tasks_only",
        "evaluated_task_ids": list(range(1, args.stage + 1)),
        "cell_count": len(summaries),
        "model_load_count": 1,
        "elapsed_seconds": time.time() - started,
        "data_root": str(data_root),
        "eval_limit": args.eval_limit,
        "active_adapter": active,
        "cells": summaries,
    }
    atomic_json(stage_summary, final)
    print(json.dumps({"status": status, "stage": args.stage, "cells": len(summaries)}))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
