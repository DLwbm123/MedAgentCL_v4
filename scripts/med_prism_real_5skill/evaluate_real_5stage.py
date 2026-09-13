#!/usr/bin/env python3
"""Fresh-process reload and real five-skill evaluation for one CL stage."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from scipy.optimize import linear_sum_assignment

from med_prism.adapters.injection import iter_shared_private_wrappers
from med_prism.checkpoint.shared_private_checkpoint import compose_shared_private
from med_prism.config import MODEL_ID, MODEL_REVISION


ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/medicalskill_cl_v1_1_and_medprism_smoke"
SNAPSHOT = Path("/remote-home/wangbomin/huggingface_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots") / MODEL_REVISION
TASK_NAMES = {1: "vqa", 2: "diagnosis_classification", 3: "concept_recognition", 4: "visual_grounding", 5: "reasoning_vqa"}


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    user = str(row["messages"][0]["content"])
    for _ in row["images"]:
        user = user.replace("<image>", "", 1)
    content = [{"type": "image", "image": image} for image in row["images"]]
    content.append({"type": "text", "text": user.strip()})
    return [{"role": "user", "content": content}]


def generate(model, processor, row: dict[str, Any], task_id: int) -> str:
    from qwen_vl_utils import process_vision_info

    messages = prompt_messages(row)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda:0")
    limits = {1: 64, 2: 64, 3: 128, 4: 96, 5: 256}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=limits[task_id], do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None)
    return processor.tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def options(row: dict[str, Any]) -> dict[str, str]:
    prompt = str(row["messages"][0]["content"])
    return {letter.upper(): text.strip() for letter, text in re.findall(r"(?m)^\s*([A-Z])\.\s*(.+?)\s*$", prompt)}


def answer_match(row: dict[str, Any], raw: str) -> bool:
    reference = str(row["messages"][-1]["content"])
    choices = options(row)
    text = raw.strip()
    match = re.search(r"(?:final\s+answer\s*[:\-]?\s*)?\b([A-Z])(?:[\.\):]|\b)", text, re.I)
    prediction = choices.get(match.group(1).upper(), "") if match else text
    reference_match = re.search(r"(?:final\s+answer\s*[:\-]?\s*)?\b([A-Z])(?:[\.\):]|\b)", reference, re.I)
    reference_text = choices.get(reference_match.group(1).upper(), "") if reference_match else reference
    return bool(norm(prediction)) and (norm(prediction) == norm(reference_text) or norm(reference_text) in norm(text))


def concept_sets(row: dict[str, Any], raw: str) -> tuple[set[str], set[str], set[str], set[str]]:
    def segments(text: str) -> list[str]:
        return [norm(value) for value in re.split(r"[;\n,]+", text) if norm(value)]
    predicted_all = set(segments(raw))
    reference_all = set(segments(str(row["messages"][-1]["content"])))
    def core(values: set[str]) -> set[str]:
        result = set()
        for value in values:
            value = re.sub(r"^(?:uncertain|negated)\s+", "", value)
            result.add(value)
        return result
    return predicted_all, reference_all, core(predicted_all), core(reference_all)


def boxes(text: str) -> list[list[float]]:
    result = []
    for raw in re.findall(r"<box>\s*([^<]+?)\s*</box>", text, re.I):
        values = re.findall(r"-?\d+(?:\.\d+)?", raw)
        if len(values) >= 4:
            box = [max(0.0, min(1000.0, float(value))) for value in values[:4]]
            if box[2] > box[0] and box[3] > box[1]:
                result.append(box)
    return result


def iou(a: list[float], b: list[float]) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def hungarian_scores(predicted: list[list[float]], reference: list[list[float]]) -> tuple[float, float, list[float]]:
    if not predicted or not reference:
        return 0.0, 0.0, []
    matrix = [[iou(left, right) for right in reference] for left in predicted]
    row_ids, col_ids = linear_sum_assignment([[-value for value in row] for row in matrix])
    matched_by_ref = [0.0] * len(reference)
    for row_id, col_id in zip(row_ids, col_ids):
        matched_by_ref[col_id] = matrix[row_id][col_id]
    return sum(value >= 0.5 for value in matched_by_ref) / len(reference), sum(matched_by_ref) / len(reference), matched_by_ref


def rouge_l(reference: str, prediction: str) -> float:
    left, right = norm(reference).split(), norm(prediction).split()
    if not left or not right:
        return 0.0
    dp = [0] * (len(right) + 1)
    for token in left:
        previous = 0
        for index, other in enumerate(right, 1):
            saved = dp[index]
            dp[index] = previous + 1 if token == other else max(dp[index], dp[index - 1])
            previous = saved
    lcs = dp[-1]
    precision, recall = lcs / len(right), lcs / len(left)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def reasoning_part(text: str) -> str:
    return re.split(r"(?i)final\s+answer\s*:", text, maxsplit=1)[0].replace("Reasoning:", "").strip()


def evaluate_rows(task_id: int, rows: list[dict[str, Any]], predictions: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details = []
    if task_id in (1, 2):
        values = [answer_match(row, raw) for row, raw in zip(rows, predictions)]
        metric = {"accuracy": sum(values) / len(values), "primary_score": sum(values) / len(values)}
        details = [{"id": row["id"], "prediction": raw, "reference": row["messages"][-1]["content"], "correct": value} for row, raw, value in zip(rows, predictions, values)]
    elif task_id == 3:
        totals = {"all_tp": 0, "all_fp": 0, "all_fn": 0, "core_tp": 0, "core_fp": 0, "core_fn": 0}
        for row, raw in zip(rows, predictions):
            pa, ra, pc, rc = concept_sets(row, raw)
            values = {"all_tp": len(pa & ra), "all_fp": len(pa - ra), "all_fn": len(ra - pa), "core_tp": len(pc & rc), "core_fp": len(pc - rc), "core_fn": len(rc - pc)}
            for key, value in values.items(): totals[key] += value
            details.append({"id": row["id"], "prediction": raw, "reference": row["messages"][-1]["content"], **values})
        def f1(prefix):
            tp, fp, fn = totals[f"{prefix}_tp"], totals[f"{prefix}_fp"], totals[f"{prefix}_fn"]
            return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        metric = {"all_concept_micro_f1": f1("all"), "core_vocabulary_micro_f1": f1("core")}
        metric["primary_score"] = (metric["all_concept_micro_f1"] + metric["core_vocabulary_micro_f1"]) / 2
    elif task_id == 4:
        acc, means = [], []
        for row, raw in zip(rows, predictions):
            predicted, reference = boxes(raw), (row.get("metadata") or {}).get("boxes_0_1000") or []
            a, m, matching = hungarian_scores(predicted, reference)
            acc.append(a); means.append(m)
            details.append({"id": row["id"], "prediction": raw, "predicted_boxes": predicted, "reference_boxes": reference, "hungarian_matched_ious": matching, "acc_iou_0_5": a, "mean_iou": m})
        metric = {"acc_iou_ge_0_5": sum(acc) / len(acc), "mean_iou": sum(means) / len(means), "matching": "scipy.optimize.linear_sum_assignment one-to-one Hungarian"}
        metric["primary_score"] = (metric["acc_iou_ge_0_5"] + metric["mean_iou"]) / 2
    else:
        correct, rouge = [], []
        for row, raw in zip(rows, predictions):
            reference = str(row["messages"][-1]["content"])
            correct.append(answer_match(row, raw)); rouge.append(rouge_l(reasoning_part(reference), reasoning_part(raw)))
            details.append({"id": row["id"], "prediction": raw, "reference": reference, "final_answer_correct": correct[-1], "reasoning_rouge_l": rouge[-1]})
        metric = {"final_answer_accuracy": sum(correct) / len(correct), "reasoning_rouge_l": sum(rouge) / len(rouge)}
        metric["primary_score"] = (metric["final_answer_accuracy"] + metric["reasoning_rouge_l"]) / 2
    metric.update({"status": "PASS" if all(prediction.strip() for prediction in predictions) else "FAIL", "total": len(rows), "nonempty_predictions": sum(bool(value.strip()) for value in predictions)})
    return metric, details


def set_active(model, task_ids: list[int]) -> None:
    for _, wrapper in iter_shared_private_wrappers(model):
        wrapper.set_active_tasks(task_ids)


def load_model(stage: int):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(str(SNAPSHOT), local_files_only=True)
    if hasattr(processor.image_processor, "min_pixels"): processor.image_processor.min_pixels = 200704
    if hasattr(processor.image_processor, "max_pixels"): processor.image_processor.max_pixels = 200704
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(SNAPSHOT), local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.requires_grad_(False)
    active = {"method": "base_zero_shot", "private_task_ids": []}
    if stage:
        base = ARTIFACT / "med_prism"
        active = compose_shared_private(model, shared_manifest=base / f"stage_{stage:02d}" / "shared/shared_manifest.json", private_manifests=[base / f"stage_{task:02d}" / "private/private_manifest.json" for task in range(1, stage + 1)])
    return model.to("cuda:0").eval(), processor, active


def reload_probe(model, stage: int) -> dict[str, Any]:
    if not stage:
        return {"status": "NOT_APPLICABLE", "stage": 0}
    root = ARTIFACT / "med_prism" / f"stage_{stage:02d}" / "runtime_audits"
    metadata = json.loads((root / f"task{stage}_pre_reload_probe.json").read_text())
    inputs = torch.load(metadata["inputs"], map_location="cpu", weights_only=True)
    inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
    set_active(model, list(range(1, stage + 1)))
    with torch.inference_mode():
        actual = model(**inputs).logits[:, -1, :].detach().float().cpu()
    expected = load_file(metadata["logits"])["last_token_logits"].float()
    difference = (actual - expected).abs()
    result = {"status": "PASS" if torch.allclose(actual, expected, rtol=1e-4, atol=1e-5) else "FAIL", "stage": stage, "fresh_process": True, "shape": list(actual.shape), "max_abs_diff": float(difference.max()), "mean_abs_diff": float(difference.mean()), "rtol": 1e-4, "atol": 1e-5, "pre_reload_probe": metadata}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=range(0, 6), required=True)
    args = parser.parse_args()
    stage_dir = ARTIFACT / "evaluation" / f"stage_{args.stage:02d}"
    summary_path = stage_dir / "stage_evaluation_summary.json"
    if summary_path.is_file() and json.loads(summary_path.read_text()).get("status") == "PASS":
        print(f"SKIP completed evaluation stage {args.stage}")
        return 0
    torch.manual_seed(42)
    started = time.time()
    model, processor, active = load_model(args.stage)
    probe = reload_probe(model, args.stage)
    if probe["status"] == "FAIL":
        raise RuntimeError(f"Reload equivalence failed: {probe}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    modes = ["base_zero_shot"] if args.stage == 0 else ["primary_cumulative", "oracle_skill_aware"]
    summaries = {}
    primary_cache = {}
    for mode in modes:
        for task_id in range(1, 6):
            active_tasks = [] if args.stage == 0 else (list(range(1, args.stage + 1)) if mode == "primary_cumulative" else ([task_id] if task_id <= args.stage else []))
            if args.stage: set_active(model, active_tasks)
            dataset = ARTIFACT / "smoke_data" / f"task_{task_id:02d}_test_fixed.jsonl"
            values = read_jsonl(dataset)
            predictions = [generate(model, processor, row, task_id) for row in values]
            metric, details = evaluate_rows(task_id, values, predictions)
            metric.update({"mode": mode, "stage": args.stage, "eval_task": task_id, "task_name": TASK_NAMES[task_id], "active_private_tasks": active_tasks, "dataset": str(dataset), "dataset_sha256": sha256(dataset), "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "method": "base_zero_shot" if args.stage == 0 else "med_prism_shared_private", "fresh_process_reload": bool(args.stage)})
            stem = f"{mode}_task_{task_id:02d}"
            path = stage_dir / f"{stem}.predictions.jsonl"
            path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in details), encoding="utf-8")
            metric["predictions"] = str(path); metric["predictions_sha256"] = sha256(path)
            (stage_dir / f"{stem}.summary.json").write_text(json.dumps(metric, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            summaries[stem] = metric
            if mode == "primary_cumulative": primary_cache[task_id] = metric
    if args.stage:
        for task_id, metric in primary_cache.items():
            task_free = dict(metric)
            task_free["mode"] = "task_free"
            task_free["task_free_semantics"] = "all learned private experts composed without eval-task adapter routing"
            stem = f"task_free_task_{task_id:02d}"
            (stage_dir / f"{stem}.summary.json").write_text(json.dumps(task_free, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            summaries[stem] = task_free
    status = "PASS" if probe["status"] in {"PASS", "NOT_APPLICABLE"} and all(value["status"] == "PASS" for value in summaries.values()) else "FAIL"
    final = {"status": status, "stage": args.stage, "model_load_count": 1, "elapsed_seconds": time.time() - started, "active_component_manifest": active, "reload_equivalence": probe, "cells": summaries}
    summary_path.write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "stage": args.stage, "elapsed_seconds": final["elapsed_seconds"], "cells": len(summaries), "reload": probe["status"]}, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
