#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from med_prism.adapters.injection import iter_rank1_wrappers
from med_prism.checkpoint import load_rank1_checkpoint
from med_prism.checkpoint.shared_private_checkpoint import compose_shared_private
from med_prism.config import MODEL_ID, MODEL_REVISION
from med_prism.evaluation.cl import component_task_ids, evaluate_prediction


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sample_id(row: dict[str, Any]) -> str:
    return str(
        row.get("source_sample_id")
        or row.get("question_id")
        or row.get("cl_sample_id")
        or row.get("id")
    )


def prompt_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    user = row["messages"][0]["content"]
    for _ in row["images"]:
        user = user.replace("<image>", "", 1)
    content = [
        {"type": "image", "image": image}
        for image in row["images"]
    ]
    content.append({"type": "text", "text": user.strip()})
    return [{"role": "user", "content": content}]


def generate(model, processor, row: dict[str, Any]) -> str:
    messages = prompt_messages(row)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    images, videos = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        return_tensors="pt",
    ).to("cuda:0")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=24,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    generated_ids = generated[0, inputs["input_ids"].shape[1] :]
    return processor.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()


def load_model(args: argparse.Namespace):
    snapshot = snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        snapshot,
        revision=MODEL_REVISION,
        local_files_only=True,
    )
    if hasattr(processor.image_processor, "min_pixels"):
        processor.image_processor.min_pixels = 200704
    if hasattr(processor.image_processor, "max_pixels"):
        processor.image_processor.max_pixels = 802816
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        snapshot,
        revision=MODEL_REVISION,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.requires_grad_(False)

    active: dict[str, Any]
    if args.method == "base_zero_shot":
        if args.after_task != 0 or any(
            (args.adapter, args.rank1_manifest, args.shared_manifest, args.private_manifest)
        ):
            raise ValueError("Base zero-shot must use after-task 0 and no adapters")
        active = {
            "method": args.method,
            "base_only": True,
            "merged": False,
            "adapter_load_count": 0,
        }
    elif args.method == "sequential_native_lora_r48":
        if not args.adapter or args.after_task not in (1, 2):
            raise ValueError("Native LoRA requires exactly one current continuous checkpoint")
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        active = {
            "method": args.method,
            "continuous_checkpoint_after_task": args.after_task,
            "adapter": str(args.adapter),
            "adapter_load_count": 1,
            "historical_adapter_stack": False,
            "merged": False,
        }
    elif args.method == "pure_rank1_e16":
        if not args.rank1_manifest or args.after_task not in (1, 2):
            raise ValueError("Pure Rank-1 requires one cumulative rank1 manifest")
        load_rank1_checkpoint(
            model,
            args.rank1_manifest,
            expected_revision=MODEL_REVISION,
            trainable_task_id=None,
        )
        wrappers = list(iter_rank1_wrappers(model))
        task_ids = list(wrappers[0][1].active_tasks)
        expected = list(range(1, args.after_task + 1))
        if task_ids != expected:
            raise RuntimeError(f"Rank-1 task bank mismatch: {task_ids} != {expected}")
        active = {
            "method": args.method,
            "rank1_manifest": str(args.rank1_manifest),
            "private_task_ids": task_ids,
            "manifest_load_count": 1,
            "merged": False,
        }
    else:
        expected = component_task_ids(
            args.mode,
            after_task=args.after_task,
            eval_task=args.eval_task,
        )
        if not args.shared_manifest or len(args.private_manifest) != len(expected):
            raise ValueError("Shared/private component count does not match evaluation contract")
        active = compose_shared_private(
            model,
            shared_manifest=args.shared_manifest,
            private_manifests=args.private_manifest,
        )
        if (
            active["shared_load_count"] != 1
            or active["private_task_ids"] != expected
            or active["merged"]
        ):
            raise RuntimeError(f"Shared/private contract failed: {active}")
    return model.to("cuda:0").eval(), processor, active


def validate_existing(
    existing: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    active: dict[str, Any],
) -> None:
    if len(existing) > len(rows):
        raise RuntimeError("Prediction file is longer than the dataset")
    for index, prediction in enumerate(existing):
        if prediction["sample_id"] != sample_id(rows[index]):
            raise RuntimeError(f"Resume sample mismatch at row {index}")
        if prediction["active_component_manifest"] != active:
            raise RuntimeError(f"Resume component mismatch at row {index}")


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
    parser.add_argument(
        "--mode",
        choices=["primary_cumulative", "oracle_skill_aware"],
        default="primary_cumulative",
    )
    parser.add_argument("--after-task", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--eval-task", type=int, choices=[1, 2], required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--rank1-manifest", type=Path)
    parser.add_argument("--shared-manifest", type=Path)
    parser.add_argument("--private-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.method != "shared32_private16" and args.mode != "primary_cumulative":
        parser.error("Oracle mode is diagnostic-only and only valid for shared32_private16")

    rows = read_jsonl(args.dataset)
    if not rows:
        raise RuntimeError(f"Empty evaluation dataset: {args.dataset}")
    for path in (
        args.dataset,
        args.adapter,
        args.rank1_manifest,
        args.shared_manifest,
        *args.private_manifest,
    ):
        if path is not None and not path.exists():
            raise FileNotFoundError(path)

    torch.manual_seed(42)
    model, processor, active = load_model(args)
    existing = read_jsonl(args.output) if args.output.is_file() else []
    validate_existing(existing, rows, active)
    replay_a = generate(model, processor, rows[0])
    replay_b = generate(model, processor, rows[0])
    deterministic = replay_a == replay_b
    if not deterministic:
        raise RuntimeError("Deterministic first-sample replay mismatch")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in range(len(existing), len(rows)):
        row = rows[index]
        raw = replay_a if index == 0 else generate(model, processor, row)
        parsed = evaluate_prediction(row, raw)
        result = {
            "sample_id": sample_id(row),
            "task_id": args.eval_task,
            "after_task": args.after_task,
            **parsed,
            "correct": bool(parsed["correct"]),
            "image_path": row["images"][0],
            "active_component_manifest": active,
        }
        with args.output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=True) + "\n")
        existing.append(result)

    correct = sum(item["correct"] for item in existing)
    invalid = sum(item["invalid"] for item in existing)
    summary = {
        "status": "PASS",
        "method": args.method,
        "mode": args.mode,
        "after_task": args.after_task,
        "eval_task": args.eval_task,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "do_sample": False,
        "num_beams": 1,
        "seed": 42,
        "total": len(existing),
        "correct": correct,
        "invalid": invalid,
        "accuracy": correct / len(existing),
        "invalid_rate": invalid / len(existing),
        "scale": "0..1",
        "deterministic_first_sample_replay": deterministic,
        "prediction_path": str(args.output),
        "prediction_sha256": sha256(args.output),
        "dataset_path": str(args.dataset),
        "dataset_sha256": sha256(args.dataset),
        "active_component_manifest": active,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "method": args.method,
        "mode": args.mode,
        "cell": f"R[{args.after_task},{args.eval_task}]",
        "accuracy": summary["accuracy"],
        "invalid_rate": summary["invalid_rate"],
    }))


if __name__ == "__main__":
    main()
