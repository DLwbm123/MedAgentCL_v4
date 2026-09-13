#!/usr/bin/env python3
"""Long-lived single-GPU worker for MedicalSkill v1.2 evaluation cells."""
from __future__ import annotations

import contextlib
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

import evaluate_formal_v1_2 as legacy
import evaluate_formal_v1_2_cell as cell_io
from evaluation_contract_v1_2 import (
    EVALUATION_PROTOCOL,
    MAX_NEW_TOKENS,
    TASK_DIRS,
    cell_is_complete,
    cell_stem,
    sha256,
)


ARTIFACT_ROOT = Path(os.environ["MED_PRISM_ARTIFACT_ROOT"]).resolve()
DATA_ROOT = Path(os.environ["MED_PRISM_DATA_ROOT"]).resolve()
EVALUATION_ROOT = ARTIFACT_ROOT / "evaluation"
PHYSICAL_GPU = os.environ.get("MED_PRISM_PHYSICAL_GPU", "unknown")


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def profiled_generate(model, processor, rows, task: int, batch_size: int):
    from qwen_vl_utils import process_vision_info

    processor.tokenizer.padding_side = "left"
    predictions: list[str] = []
    preprocessing_seconds = 0.0
    generation_seconds = 0.0
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        started = time.time()
        conversations = [legacy.prompt_messages(row) for row in batch]
        texts = [
            processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in conversations
        ]
        images, videos = process_vision_info(conversations)
        inputs = processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        ).to("cuda:0")
        prompt_width = inputs["input_ids"].shape[1]
        preprocessing_seconds += time.time() - started
        started = time.time()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS[task],
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        generation_seconds += time.time() - started
        predictions.extend(
            value.strip()
            for value in processor.tokenizer.batch_decode(
                generated[:, prompt_width:], skip_special_tokens=True
            )
        )
    generated_token_count = sum(
        len(processor.tokenizer.encode(value, add_special_tokens=False))
        for value in predictions
    )
    return predictions, {
        "preprocessing_seconds": preprocessing_seconds,
        "generation_seconds": generation_seconds,
        "generated_token_count": generated_token_count,
        "generated_tokens_per_second": (
            generated_token_count / generation_seconds
            if generation_seconds > 0
            else 0.0
        ),
    }


class Worker:
    def __init__(self) -> None:
        self.stage: int | None = None
        self.model = None
        self.processor = None
        self.active_manifest = None
        self.reload_probe = None
        self.stage_load_count = 0
        self.model_load_seconds_total = 0.0
        self.reload_probe_seconds_total = 0.0
        self.cells_completed = 0
        self.started = time.time()

    def load_stage(self, stage: int) -> dict[str, Any]:
        if self.stage == stage and self.model is not None:
            return {"stage_reused": True, "stage_load": None}
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        started = time.time()
        with contextlib.redirect_stdout(sys.stderr):
            model, processor, active = legacy.load_model(stage)
        model_load_seconds = time.time() - started
        started = time.time()
        with contextlib.redirect_stdout(sys.stderr):
            probe = legacy.reload_probe(model, stage)
        reload_probe_seconds = time.time() - started
        if probe["status"] == "FAIL":
            raise RuntimeError(f"Reload equivalence failed: {probe}")
        self.stage = stage
        self.model = model
        self.processor = processor
        self.active_manifest = active
        self.reload_probe = probe
        self.stage_load_count += 1
        self.model_load_seconds_total += model_load_seconds
        self.reload_probe_seconds_total += reload_probe_seconds
        return {
            "stage_reused": False,
            "stage_load": {
                "stage": stage,
                "model_load_seconds": model_load_seconds,
                "reload_probe_seconds": reload_probe_seconds,
                "reload_equivalence": probe,
                "active_component_manifest": active,
            },
        }

    def evaluate(self, job: dict[str, Any]) -> dict[str, Any]:
        stage = int(job["stage"])
        mode = str(job["mode"])
        task = int(job["task"])
        eval_limit = int(job["eval_limit"])
        batch_size = int(job["batch_size"])
        if cell_is_complete(
            evaluation_root=EVALUATION_ROOT,
            data_root=DATA_ROOT,
            stage=stage,
            mode=mode,
            task=task,
            eval_limit=eval_limit,
        ):
            return {
                "status": "SKIP",
                "stage": stage,
                "mode": mode,
                "task": task,
                "gpu": PHYSICAL_GPU,
            }
        load_info = self.load_stage(stage)
        cell_started = time.time()
        selected_tasks = cell_io.active_tasks(stage, mode, task)
        if stage:
            legacy.set_active(self.model, selected_tasks)
        dataset = DATA_ROOT / TASK_DIRS[task] / "test.jsonl"
        rows = legacy.read_jsonl(dataset, eval_limit)
        if not rows:
            raise RuntimeError(f"Empty evaluation dataset: {dataset}")
        predictions, generation_profile = profiled_generate(
            self.model, self.processor, rows, task, batch_size
        )
        started = time.time()
        metric, details = legacy.evaluate_rows(task, rows, predictions)
        metric_seconds = time.time() - started
        cell_elapsed_seconds = time.time() - cell_started
        metric.update(
            {
                "mode": mode,
                "stage": stage,
                "eval_task": task,
                "task_name": legacy.TASK_NAMES[task],
                "active_private_tasks": selected_tasks,
                "evaluation_protocol": EVALUATION_PROTOCOL,
                "dataset": str(dataset),
                "dataset_sha256": sha256(dataset),
                "model_id": legacy.MODEL_ID,
                "model_revision": legacy.MODEL_REVISION,
                "method": (
                    "base_zero_shot" if stage == 0 else "med_prism_shared_private"
                ),
                "fresh_process_reload": True,
                "eval_limit": eval_limit,
                "batch_size": batch_size,
                "worker_gpu": PHYSICAL_GPU,
                "model_load_seconds": (
                    load_info["stage_load"]["model_load_seconds"]
                    if load_info["stage_load"]
                    else 0.0
                ),
                "reload_probe_seconds": (
                    load_info["stage_load"]["reload_probe_seconds"]
                    if load_info["stage_load"]
                    else 0.0
                ),
                **generation_profile,
                "metric_seconds": metric_seconds,
                "cell_elapsed_seconds": cell_elapsed_seconds,
                "samples_per_second_generation": (
                    len(rows) / generation_profile["generation_seconds"]
                    if generation_profile["generation_seconds"] > 0
                    else 0.0
                ),
            }
        )
        stem = cell_stem(mode, task)
        stage_root = EVALUATION_ROOT / f"stage_{stage:02d}"
        predictions_path = stage_root / f"{stem}.predictions.jsonl"
        cell_io.atomic_write(
            predictions_path,
            "".join(
                json.dumps(item, ensure_ascii=False) + "\n" for item in details
            ),
        )
        metric["predictions"] = str(predictions_path)
        metric["predictions_sha256"] = sha256(predictions_path)
        cell_io.write_json(stage_root / f"{stem}.summary.json", metric)
        self.cells_completed += 1
        return {
            "status": metric["status"],
            "stage": stage,
            "mode": mode,
            "task": task,
            "gpu": PHYSICAL_GPU,
            "stage_reused": load_info["stage_reused"],
            "stage_load": load_info["stage_load"],
            "cell_profile": {
                key: metric[key]
                for key in (
                    "model_load_seconds",
                    "reload_probe_seconds",
                    "preprocessing_seconds",
                    "generation_seconds",
                    "metric_seconds",
                    "cell_elapsed_seconds",
                    "samples_per_second_generation",
                    "generated_token_count",
                    "generated_tokens_per_second",
                )
            },
        }

    def report(self) -> dict[str, Any]:
        return {
            "status": "PASS",
            "gpu": PHYSICAL_GPU,
            "stage_load_count": self.stage_load_count,
            "model_load_seconds_total": self.model_load_seconds_total,
            "reload_probe_seconds_total": self.reload_probe_seconds_total,
            "cells_completed": self.cells_completed,
            "wall_clock_seconds": time.time() - self.started,
        }


def main() -> int:
    worker = Worker()
    emit({"status": "READY", "gpu": PHYSICAL_GPU, "pid": os.getpid()})
    for line in sys.stdin:
        try:
            command = json.loads(line)
            if command.get("command") == "shutdown":
                emit({"status": "SHUTDOWN", "worker_report": worker.report()})
                return 0
            if command.get("command") != "evaluate":
                raise ValueError(f"Unsupported worker command: {command}")
            emit(worker.evaluate(command))
        except Exception as exc:
            emit(
                {
                    "status": "FAIL",
                    "gpu": PHYSICAL_GPU,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
