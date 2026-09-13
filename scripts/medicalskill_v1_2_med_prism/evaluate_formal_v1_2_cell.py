#!/usr/bin/env python3
"""Resumable, batched evaluation of one Med-PRISM stage/mode/task cell."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

import evaluate_formal_v1_2 as legacy


EVAL_ROOT = Path(
    os.environ.get("MED_PRISM_EVALUATION_ROOT", str(legacy.ARTIFACT / "evaluation"))
)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def generation_cells(stage: int) -> list[tuple[str, int]]:
    tasks = legacy.evaluation_task_ids(stage)
    if stage == 0:
        return [("base_zero_shot", task) for task in tasks]
    return [
        (mode, task)
        for mode in ("primary_cumulative", "oracle_skill_aware")
        for task in tasks
    ]


def stem(mode: str, task: int) -> str:
    return f"{mode}_task_{task:02d}"


def expected_total(task: int, eval_limit: int) -> int:
    dataset = legacy.DATA_ROOT / legacy.TASK_DIRS[task] / "test.jsonl"
    return len(legacy.read_jsonl(dataset, eval_limit))


def cell_is_complete(
    stage_dir: Path,
    stage: int,
    mode: str,
    task: int,
    eval_limit: int,
) -> bool:
    name = stem(mode, task)
    summary_path = stage_dir / f"{name}.summary.json"
    predictions_path = stage_dir / f"{name}.predictions.jsonl"
    if not summary_path.is_file() or not predictions_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        dataset = legacy.DATA_ROOT / legacy.TASK_DIRS[task] / "test.jsonl"
        return (
            summary.get("status") == "PASS"
            and summary.get("stage") == int(stage)
            and summary.get("mode") == mode
            and summary.get("eval_task") == int(task)
            and summary.get("dataset") == str(dataset)
            and summary.get("dataset_sha256") == legacy.sha256(dataset)
            and int(summary.get("total", -1)) == expected_total(task, eval_limit)
            and int(summary.get("nonempty_predictions", -1))
            == expected_total(task, eval_limit)
            and summary.get("predictions_sha256") == legacy.sha256(predictions_path)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def batched_generate(
    model,
    processor,
    rows: list[dict[str, Any]],
    task: int,
    batch_size: int,
) -> list[str]:
    from qwen_vl_utils import process_vision_info

    processor.tokenizer.padding_side = "left"
    limits = {1: 64, 2: 64, 3: 128, 4: 96, 5: 256}
    predictions: list[str] = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
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
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=limits[task],
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        predictions.extend(
            value.strip()
            for value in processor.tokenizer.batch_decode(
                generated[:, prompt_width:], skip_special_tokens=True
            )
        )
    return predictions


def active_tasks(stage: int, mode: str, task: int) -> list[int]:
    if stage == 0:
        return []
    if mode == "primary_cumulative":
        return list(range(1, stage + 1))
    return [task]


def evaluate_cell(
    stage: int,
    mode: str,
    task: int,
    eval_limit: int,
    batch_size: int,
) -> dict[str, Any]:
    stage_dir = EVAL_ROOT / f"stage_{stage:02d}"
    if cell_is_complete(stage_dir, stage, mode, task, eval_limit):
        return {"status": "SKIP", "stage": stage, "mode": mode, "task": task}

    started = time.time()
    model, processor, active_manifest = legacy.load_model(stage)
    probe = legacy.reload_probe(model, stage)
    if probe["status"] == "FAIL":
        raise RuntimeError(f"Reload equivalence failed: {probe}")
    selected_tasks = active_tasks(stage, mode, task)
    if stage:
        legacy.set_active(model, selected_tasks)

    dataset = legacy.DATA_ROOT / legacy.TASK_DIRS[task] / "test.jsonl"
    rows = legacy.read_jsonl(dataset, eval_limit)
    if not rows:
        raise RuntimeError(f"Empty evaluation dataset: {dataset}")
    predictions = batched_generate(model, processor, rows, task, batch_size)
    metric, details = legacy.evaluate_rows(task, rows, predictions)
    cell_elapsed = time.time() - started
    metric.update(
        {
            "mode": mode,
            "stage": stage,
            "eval_task": task,
            "task_name": legacy.TASK_NAMES[task],
            "active_private_tasks": selected_tasks,
            "evaluation_protocol": legacy.EVALUATION_PROTOCOL,
            "dataset": str(dataset),
            "dataset_sha256": legacy.sha256(dataset),
            "model_id": legacy.MODEL_ID,
            "model_revision": legacy.MODEL_REVISION,
            "method": "base_zero_shot" if stage == 0 else "med_prism_shared_private",
            "fresh_process_reload": bool(stage),
            "eval_limit": eval_limit,
            "batch_size": batch_size,
            "elapsed_seconds": cell_elapsed,
            "samples_per_second": len(rows) / cell_elapsed,
        }
    )
    name = stem(mode, task)
    predictions_path = stage_dir / f"{name}.predictions.jsonl"
    atomic_write(
        predictions_path,
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in details),
    )
    metric["predictions"] = str(predictions_path)
    metric["predictions_sha256"] = legacy.sha256(predictions_path)
    write_json(stage_dir / f"{name}.summary.json", metric)
    write_json(
        stage_dir / f".{name}.worker.json",
        {
            "status": metric["status"],
            "pid": os.getpid(),
            "stage": stage,
            "mode": mode,
            "task": task,
            "batch_size": batch_size,
            "elapsed_seconds": cell_elapsed,
            "reload_equivalence": probe,
            "active_component_manifest": active_manifest,
        },
    )
    return {"status": metric["status"], "stage": stage, "mode": mode, "task": task, "elapsed_seconds": cell_elapsed}


def finalize_stage(stage: int, eval_limit: int) -> dict[str, Any]:
    stage_dir = EVAL_ROOT / f"stage_{stage:02d}"
    summaries: dict[str, Any] = {}
    missing = []
    for mode, task in generation_cells(stage):
        name = stem(mode, task)
        if not cell_is_complete(stage_dir, stage, mode, task, eval_limit):
            missing.append(name)
            continue
        summaries[name] = json.loads(
            (stage_dir / f"{name}.summary.json").read_text(encoding="utf-8")
        )
    if missing:
        raise RuntimeError(f"Stage {stage} is incomplete: {missing}")

    if stage:
        for task in legacy.evaluation_task_ids(stage):
            primary = dict(summaries[stem("primary_cumulative", task)])
            primary["mode"] = "task_free"
            primary["task_free_semantics"] = (
                "all learned private experts composed without eval-task adapter routing"
            )
            name = stem("task_free", task)
            write_json(stage_dir / f"{name}.summary.json", primary)
            summaries[name] = primary

    workers = []
    for path in sorted(stage_dir.glob(".*.worker.json")):
        try:
            workers.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    probe = (
        {"status": "NOT_APPLICABLE", "stage": 0}
        if stage == 0
        else next(
            (
                item["reload_equivalence"]
                for item in workers
                if item.get("reload_equivalence", {}).get("status") == "PASS"
            ),
            None,
        )
    )
    if probe is None:
        raise RuntimeError(f"Stage {stage} has no successful reload probe")
    active_manifest = next(
        (item.get("active_component_manifest") for item in workers if item.get("active_component_manifest")),
        {"method": "base_zero_shot", "private_task_ids": []},
    )
    status = (
        "PASS"
        if probe["status"] in {"PASS", "NOT_APPLICABLE"}
        and all(value.get("status") == "PASS" for value in summaries.values())
        else "FAIL"
    )
    generated = [
        value for key, value in summaries.items() if not key.startswith("task_free_")
    ]
    final = {
        "status": status,
        "stage": stage,
        "evaluation_protocol": legacy.EVALUATION_PROTOCOL,
        "evaluated_task_ids": legacy.evaluation_task_ids(stage),
        "matrix_scope": "base_zero_shot_all_tasks" if stage == 0 else "seen_tasks_only",
        "model_load_count": len(workers),
        "parallel_cell_evaluation": True,
        "elapsed_seconds": sum(float(item.get("elapsed_seconds", 0.0)) for item in generated),
        "data_root": str(legacy.DATA_ROOT),
        "eval_limit": eval_limit,
        "active_component_manifest": active_manifest,
        "reload_equivalence": probe,
        "cells": summaries,
    }
    write_json(stage_dir / "stage_evaluation_summary.json", final)
    return {"status": status, "stage": stage, "cells": len(summaries)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=range(0, 6), required=True)
    parser.add_argument("--mode", choices=("base_zero_shot", "primary_cumulative", "oracle_skill_aware"))
    parser.add_argument("--task", type=int, choices=range(1, 6))
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if args.eval_limit < 0 or args.batch_size <= 0:
        parser.error("--eval-limit must be non-negative and --batch-size must be positive")
    if args.finalize_only:
        result = finalize_stage(args.stage, args.eval_limit)
    else:
        if args.mode is None or args.task is None:
            parser.error("--mode and --task are required unless --finalize-only is used")
        if (args.mode, args.task) not in generation_cells(args.stage):
            parser.error(f"Cell {args.mode}:{args.task} is invalid for stage {args.stage}")
        torch.manual_seed(42)
        result = evaluate_cell(
            args.stage, args.mode, args.task, args.eval_limit, args.batch_size
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] in {"PASS", "SKIP"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
