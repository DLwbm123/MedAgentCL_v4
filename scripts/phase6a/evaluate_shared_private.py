#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from med_prism.checkpoint.shared_private_checkpoint import compose_shared_private
from med_prism.config import MODEL_ID, MODEL_REVISION
from med_prism.evaluation.cl import component_task_ids, evaluate_prediction


ROOT = Path("/root/MedAgentCL_v4")
PILOT = Path("/remote-home/wangbomin/medagentcl_v4_phase6a/pilot")
ARTIFACT = ROOT / "artifacts/phase6a"


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


def generate(
    model,
    processor,
    row: dict[str, Any],
) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["primary_cumulative", "oracle_skill_aware"],
        required=True,
    )
    parser.add_argument("--after-task", type=int, choices=[1, 2], required=True)
    parser.add_argument("--eval-task", type=int, choices=[1, 2], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    task_ids = component_task_ids(
        args.mode,
        after_task=args.after_task,
        eval_task=args.eval_task,
    )
    dataset = ARTIFACT / f"data/pilot_test_task{args.eval_task}.jsonl"
    rows = read_jsonl(dataset)
    if len(rows) != 64:
        raise RuntimeError(f"Expected 64 pilot test rows, got {len(rows)}")
    shared = PILOT / f"shared/after_task_{args.after_task}/shared_manifest.json"
    private = {
        task: PILOT / f"private/task_{task}/private_manifest.json"
        for task in task_ids
    }
    for path in (shared, *private.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

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
    torch.manual_seed(42)
    model = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            snapshot,
            revision=MODEL_REVISION,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda:0")
        .eval()
    )
    model.requires_grad_(False)
    active = compose_shared_private(
        model,
        shared_manifest=shared,
        private_manifests=[private[task] for task in task_ids],
    )
    if (
        active["shared_load_count"] != 1
        or active["private_task_ids"] != task_ids
        or active["merged"]
    ):
        raise RuntimeError(f"Active component contract failed: {active}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions = []
    deterministic_replay = None
    for index, row in enumerate(rows):
        raw = generate(model, processor, row)
        if index == 0:
            replay = generate(model, processor, row)
            deterministic_replay = raw == replay
            if not deterministic_replay:
                raise RuntimeError("Deterministic first-sample replay mismatch")
        parsed = evaluate_prediction(row, raw)
        predictions.append(
            {
                "sample_id": str(
                    row.get("source_sample_id")
                    or row.get("question_id")
                    or row.get("id")
                ),
                "task_id": args.eval_task,
                "after_task": args.after_task,
                **parsed,
                "correct": bool(parsed["correct"]),
                "image_path": row["images"][0],
                "active_component_manifest": active,
            }
        )
        with args.output.open("a" if index else "w", encoding="utf-8") as handle:
            handle.write(json.dumps(predictions[-1], ensure_ascii=True) + "\n")

    correct = sum(item["correct"] for item in predictions)
    invalid = sum(item["invalid"] for item in predictions)
    summary = {
        "status": "PASS",
        "method": "med_prism_shared_private",
        "mode": args.mode,
        "after_task": args.after_task,
        "eval_task": args.eval_task,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "do_sample": False,
        "num_beams": 1,
        "seed": 42,
        "total": len(predictions),
        "correct": correct,
        "invalid": invalid,
        "accuracy": correct / len(predictions),
        "invalid_rate": invalid / len(predictions),
        "scale": "0..1",
        "deterministic_first_sample_replay": deterministic_replay,
        "prediction_path": str(args.output),
        "prediction_sha256": sha256(args.output),
        "dataset_path": str(dataset),
        "dataset_sha256": sha256(dataset),
        "active_component_manifest": active,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "mode": args.mode,
                "cell": f"R[{args.after_task},{args.eval_task}]",
                "accuracy": summary["accuracy"],
                "invalid_rate": summary["invalid_rate"],
            }
        )
    )


if __name__ == "__main__":
    main()
