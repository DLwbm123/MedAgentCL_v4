#!/usr/bin/env python3
"""Evaluate one cumulative sequential-LoRA checkpoint on its seen tasks."""
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

from scripts.medicalskill_v1_2_med_prism.evaluate_formal_v1_2 import (
    EVALUATION_PROTOCOL,
    MODEL_ID,
    MODEL_REVISION,
    SNAPSHOT,
    TASK_DIRS,
    TASK_NAMES,
    evaluate_rows,
    generate,
    read_jsonl,
    sha256,
)


METHOD = "sequential_native_lora_r48"


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


def adapter_manifest(checkpoint: Path, stage: int) -> dict[str, Any]:
    config = checkpoint / "adapter_config.json"
    weights = checkpoint / "adapter_model.safetensors"
    if not weights.is_file():
        weights = checkpoint / "adapter_model.bin"
    if not config.is_file() or not weights.is_file():
        raise FileNotFoundError(f"Incomplete PEFT adapter checkpoint: {checkpoint}")
    value = json.loads(config.read_text(encoding="utf-8"))
    return {
        "method": METHOD,
        "stage": stage,
        "checkpoint": str(checkpoint),
        "adapter_config_sha256": sha256(config),
        "adapter_weights_file": str(weights),
        "adapter_weights_sha256": sha256(weights),
        "adapter_load_count": 1,
        "historical_adapter_stack": False,
        "base_plus_cumulative_adapter": True,
        "r": value.get("r"),
        "lora_alpha": value.get("lora_alpha"),
        "target_modules": sorted(value.get("target_modules") or []),
    }


def load_model(checkpoint: Path):
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
    model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False)
    return model.to("cuda:0").eval(), processor


def read_raw_predictions(
    path: Path,
    rows: list[dict[str, Any]],
    active: dict[str, Any],
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
    parser.add_argument("--stage", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-limit", type=int, default=0)
    args = parser.parse_args()
    if args.eval_limit < 0:
        raise ValueError("--eval-limit must be non-negative")
    checkpoint = args.adapter.resolve()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    active = adapter_manifest(checkpoint, args.stage)
    stage_dir = output_root / "evaluation" / f"stage_{args.stage:02d}"
    stage_summary = stage_dir / "stage_evaluation_summary.json"
    if stage_summary.is_file():
        previous = json.loads(stage_summary.read_text(encoding="utf-8"))
        if (
            previous.get("status") == "PASS"
            and previous.get("method") == METHOD
            and previous.get("evaluation_protocol") == EVALUATION_PROTOCOL
            and previous.get("data_root") == str(data_root)
            and previous.get("eval_limit") == args.eval_limit
            and previous.get("active_adapter") == active
        ):
            print(f"SKIP completed sequential evaluation stage {args.stage}")
            return 0

    torch.manual_seed(42)
    started = time.time()
    model, processor = load_model(checkpoint)
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
            append_prediction(
                raw_path,
                {
                    "id": rows[index]["id"],
                    "prediction": prediction,
                    "active_adapter": active,
                },
            )
            raw_values.append(
                {"id": rows[index]["id"], "prediction": prediction, "active_adapter": active}
            )
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
                "method": METHOD,
                "ordinary_lora": True,
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
            }
        )
        atomic_json(stage_dir / f"{stem}.summary.json", metric)
        summaries[stem] = metric
    status = "PASS" if len(summaries) == args.stage and all(
        value.get("status") == "PASS" for value in summaries.values()
    ) else "FAIL"
    final = {
        "status": status,
        "stage": args.stage,
        "method": METHOD,
        "ordinary_lora": True,
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
