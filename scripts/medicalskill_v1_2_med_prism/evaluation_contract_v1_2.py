#!/usr/bin/env python3
"""Evaluation-only contracts shared by the MedicalSkill v1.2 scheduler."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
EVALUATION_PROTOCOL = "lower_triangular_seen_tasks_v1"
TASK_DIRS = {
    1: "task_01_vqa",
    2: "task_02_diagnosis_classification",
    3: "task_03_concept_recognition",
    4: "task_04_visual_grounding",
    5: "task_05_reasoning_vqa",
}
MAX_NEW_TOKENS = {1: 64, 2: 64, 3: 128, 4: 96, 5: 256}
IMAGE_PREPROCESSING = {"min_pixels": 200704, "max_pixels": 200704}
GENERATION_CONFIG = {
    "do_sample": False,
    "num_beams": 1,
    "temperature": None,
    "top_p": None,
    "top_k": None,
    "max_new_tokens_by_task": MAX_NEW_TOKENS,
}
TASK_METRICS = {
    1: "accuracy",
    2: "accuracy",
    3: "all_concept_micro_f1/core_vocabulary_micro_f1",
    4: "Acc@IoU>=0.5/mean IoU; Hungarian one-to-one matching",
    5: "final-answer accuracy/reasoning ROUGE-L",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def stage_dir(evaluation_root: Path, stage: int) -> Path:
    return evaluation_root / f"stage_{stage:02d}"


def cell_stem(mode: str, task: int) -> str:
    return f"{mode}_task_{task:02d}"


def required_generation_cells(
    *, with_stage0: bool, with_oracle: bool
) -> list[tuple[int, str, int]]:
    cells: list[tuple[int, str, int]] = []
    if with_stage0:
        cells.extend((0, "base_zero_shot", task) for task in range(1, 6))
    for stage in range(1, 6):
        cells.extend(
            (stage, "primary_cumulative", task)
            for task in range(1, stage + 1)
        )
        if with_oracle:
            cells.extend(
                (stage, "oracle_skill_aware", task)
                for task in range(1, stage + 1)
            )
    return cells


def expected_total(data_root: Path, task: int, eval_limit: int) -> int:
    path = data_root / TASK_DIRS[task] / "test.jsonl"
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
                if eval_limit > 0 and count >= eval_limit:
                    break
    return count


def cell_is_complete(
    *,
    evaluation_root: Path,
    data_root: Path,
    stage: int,
    mode: str,
    task: int,
    eval_limit: int,
) -> bool:
    root = stage_dir(evaluation_root, stage)
    stem = cell_stem(mode, task)
    summary_path = root / f"{stem}.summary.json"
    predictions_path = root / f"{stem}.predictions.jsonl"
    dataset = data_root / TASK_DIRS[task] / "test.jsonl"
    if not summary_path.is_file() or not predictions_path.is_file():
        return False
    try:
        summary = load_json(summary_path)
        total = expected_total(data_root, task, eval_limit)
        return (
            summary.get("status") == "PASS"
            and summary.get("stage") == stage
            and summary.get("mode") == mode
            and summary.get("eval_task") == task
            and summary.get("evaluation_protocol") == EVALUATION_PROTOCOL
            and summary.get("model_id") == MODEL_ID
            and summary.get("model_revision") == MODEL_REVISION
            and summary.get("dataset") == str(dataset)
            and summary.get("dataset_sha256") == sha256(dataset)
            and int(summary.get("total", -1)) == total
            and int(summary.get("nonempty_predictions", -1)) == total
            and summary.get("predictions_sha256") == sha256(predictions_path)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def required_stage_cells(stage: int, with_oracle: bool) -> list[tuple[str, int]]:
    if stage == 0:
        return [("base_zero_shot", task) for task in range(1, 6)]
    result = [
        ("primary_cumulative", task) for task in range(1, stage + 1)
    ]
    if with_oracle:
        result.extend(
            ("oracle_skill_aware", task) for task in range(1, stage + 1)
        )
    return result


def stage_is_complete(
    *,
    evaluation_root: Path,
    data_root: Path,
    stage: int,
    with_oracle: bool,
    eval_limit: int,
) -> bool:
    return all(
        cell_is_complete(
            evaluation_root=evaluation_root,
            data_root=data_root,
            stage=stage,
            mode=mode,
            task=task,
            eval_limit=eval_limit,
        )
        for mode, task in required_stage_cells(stage, with_oracle)
    )


def resolve_base_stage_dir(base_eval_root: Path) -> Path:
    candidates = [
        base_eval_root / "evaluation" / "stage_00",
        base_eval_root / "stage_00",
        base_eval_root,
    ]
    for candidate in candidates:
        if candidate.name == "stage_00" and candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot locate canonical stage_00 under base evaluation root: {base_eval_root}"
    )


def validate_base_evaluation(
    *, base_eval_root: Path, data_root: Path, eval_limit: int
) -> dict[str, Any]:
    base_stage = resolve_base_stage_dir(base_eval_root)
    summary_path = base_stage / "stage_evaluation_summary.json"
    errors: list[str] = []
    try:
        summary = load_json(summary_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid canonical Stage 0 summary: {exc}") from exc
    if summary.get("status") != "PASS":
        errors.append("stage0_status_not_pass")
    if summary.get("stage") != 0:
        errors.append("stage0_identity_mismatch")
    if summary.get("evaluation_protocol") != EVALUATION_PROTOCOL:
        errors.append("evaluation_protocol_mismatch")
    if summary.get("eval_limit") != eval_limit:
        errors.append("eval_limit_mismatch")
    if summary.get("data_root") != str(data_root.resolve()):
        errors.append("dataset_root_mismatch")
    evaluation_root = base_stage.parent
    cells = {}
    for task in range(1, 6):
        valid = cell_is_complete(
            evaluation_root=evaluation_root,
            data_root=data_root.resolve(),
            stage=0,
            mode="base_zero_shot",
            task=task,
            eval_limit=eval_limit,
        )
        if not valid:
            errors.append(f"base_zero_shot_task_{task:02d}_invalid")
            continue
        cell = load_json(
            base_stage / f"base_zero_shot_task_{task:02d}.summary.json"
        )
        cells[str(task)] = {
            "dataset": cell["dataset"],
            "dataset_sha256": cell["dataset_sha256"],
            "predictions": cell["predictions"],
            "predictions_sha256": cell["predictions_sha256"],
            "total": cell["total"],
            "nonempty_predictions": cell["nonempty_predictions"],
            "metric_definition": TASK_METRICS[task],
        }
    model_ids = {
        value.get("model_id") for value in summary.get("cells", {}).values()
    }
    revisions = {
        value.get("model_revision")
        for value in summary.get("cells", {}).values()
    }
    if model_ids != {MODEL_ID}:
        errors.append("model_id_mismatch")
    if revisions != {MODEL_REVISION}:
        errors.append("model_revision_mismatch")
    result = {
        "status": "PASS" if not errors else "FAIL",
        "format_version": "medicalskill_v1_2_base_evaluation_v1",
        "base_stage_dir": str(base_stage),
        "stage_summary": str(summary_path),
        "stage_summary_sha256": sha256(summary_path),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "data_root": str(data_root.resolve()),
        "dataset_version": data_root.resolve().name,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "eval_limit": eval_limit,
        "generation_config": GENERATION_CONFIG,
        "image_preprocessing": IMAGE_PREPROCESSING,
        "task_metrics": TASK_METRICS,
        "cells": cells,
        "errors": errors,
    }
    if errors:
        raise RuntimeError(
            "Canonical Stage 0 is incompatible; rerun with --with-stage0. "
            + json.dumps(result, ensure_ascii=False)
        )
    return result


def build_run_config(
    *,
    data_root: Path,
    output_root: Path,
    eval_limit: int,
    batch_size: int,
    with_oracle: bool,
    with_stage0: bool,
    base_eval_root: Path | None,
) -> dict[str, Any]:
    cells = required_generation_cells(
        with_stage0=with_stage0, with_oracle=with_oracle
    )
    base = None
    if not with_stage0:
        if base_eval_root is None:
            raise RuntimeError(
                "Stage 0 is disabled by default; provide a compatible --base-eval-root"
            )
        base = validate_base_evaluation(
            base_eval_root=base_eval_root,
            data_root=data_root,
            eval_limit=eval_limit,
        )
    return {
        "status": "PASS",
        "format_version": "medicalskill_v1_2_evaluation_run_v2",
        "data_root": str(data_root.resolve()),
        "output_root": str(output_root.resolve()),
        "eval_limit": eval_limit,
        "batch_size": batch_size,
        "oracle_evaluation_enabled": with_oracle,
        "stage0_evaluation_enabled": with_stage0,
        "base_zero_shot_reused": not with_stage0,
        "base_evaluation_root": str(base_eval_root.resolve()) if base_eval_root else None,
        "base_evaluation": base,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "generation_config": GENERATION_CONFIG,
        "image_preprocessing": IMAGE_PREPROCESSING,
        "generation_cell_count": len(cells),
        "generation_cells": [
            {"stage": stage, "mode": mode, "task": task}
            for stage, mode, task in cells
        ],
    }
