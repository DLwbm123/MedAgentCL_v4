#!/usr/bin/env python3
"""Offline Qwen3-VL forward-only smoke for all v1.2 task splits."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


ROOT = Path("/remote-home/wangbomin/MedicalSkill-CL-v1.2")
OUTPUT = ROOT / "smoke/model_smoke_summary.json"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
SNAPSHOT_ROOT = Path(
    "/remote-home/wangbomin/huggingface_cache/hub/"
    "models--Qwen--Qwen3-VL-8B-Instruct/snapshots"
)


def first_shortest(path: Path) -> dict:
    best = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            score = (
                len(row.get("images") or []),
                sum(
                    len(str(item.get("content") or ""))
                    for item in row.get("messages") or []
                ),
                row.get("id", ""),
            )
            if best is None or score < best[0]:
                best = (score, row)
    if best is None:
        raise RuntimeError(f"empty smoke source: {path}")
    return best[1]


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous()
    return hashlib.sha256(array.view(torch.uint8).numpy().tobytes()).hexdigest()


def conversation(row: dict, include_assistant: bool) -> list[dict]:
    user_text = row["messages"][0]["content"].replace("<image>", "").strip()
    content = [
        {"type": "image", "image": image}
        for image in row["images"]
    ]
    content.append({"type": "text", "text": user_text})
    messages = [{"role": "user", "content": content}]
    if include_assistant:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": row["messages"][1]["content"],
                    }
                ],
            }
        )
    return messages


def main() -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    snapshot = snapshot_download(
        MODEL_ID,
        revision=REVISION,
        cache_dir=str(SNAPSHOT_ROOT.parents[1]),
        local_files_only=True,
    )
    expected = SNAPSHOT_ROOT / REVISION
    if Path(snapshot).resolve() != expected.resolve():
        raise RuntimeError(f"unexpected model snapshot: {snapshot}")
    processor = AutoProcessor.from_pretrained(
        snapshot, revision=REVISION, local_files_only=True
    )
    if hasattr(processor.image_processor, "min_pixels"):
        processor.image_processor.min_pixels = 200704
    if hasattr(processor.image_processor, "max_pixels"):
        processor.image_processor.max_pixels = 802816
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        snapshot,
        revision=REVISION,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda:0").eval()
    model.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    tasks = {}
    for task_dir in sorted(ROOT.glob("task_*")):
        split_results = {}
        for split in ("train", "test"):
            row = first_shortest(task_dir / f"{split}.jsonl")
            prompt_messages = conversation(row, include_assistant=False)
            full_messages = conversation(row, include_assistant=True)
            prompt_text = processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            image_inputs, video_inputs = process_vision_info(full_messages)
            prompt_image_inputs, prompt_video_inputs = process_vision_info(
                prompt_messages
            )
            prompt_batch = processor(
                text=[prompt_text],
                images=prompt_image_inputs,
                videos=prompt_video_inputs,
                return_tensors="pt",
            )
            full = processor(
                text=[full_text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            )
            prompt_ids = prompt_batch["input_ids"]
            labels = full["input_ids"].clone()
            prompt_length = min(prompt_ids.shape[1], labels.shape[1] - 1)
            labels[:, :prompt_length] = -100
            if not (labels[:, :prompt_length] == -100).all():
                raise RuntimeError("prompt label masking failed")
            if not (labels[:, prompt_length:] != -100).any():
                raise RuntimeError("assistant labels are empty")
            full["labels"] = labels
            full = full.to("cuda:0")
            with torch.inference_mode():
                output = model(**full)
            loss = float(output.loss.detach().float().cpu())
            if not math.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss: {task_dir.name}/{split}"
                )
            image_key = next(
                key
                for key in ("pixel_values", "pixel_values_videos")
                if key in full
            )
            split_results[split] = {
                "status": "PASS",
                "id": row["id"],
                "image_count": len(row["images"]),
                "images_readable": all(
                    Path(path).is_file() and Path(path).stat().st_size > 0
                    for path in row["images"]
                ),
                "processor_template_encoded": True,
                "input_keys": sorted(full),
                "input_token_count": int(full["input_ids"].numel()),
                "masked_prompt_token_count": prompt_length,
                "assistant_label_token_count": int(
                    (labels != -100).sum().item()
                ),
                "label_masking_valid": True,
                "image_tensor_key": image_key,
                "image_tensor_shape": list(full[image_key].shape),
                "image_tensor_sha256": tensor_sha256(full[image_key]),
                "forward_loss": loss,
                "forward_loss_finite": True,
                "logits_shape": list(output.logits.shape),
            }
        tasks[task_dir.name] = {
            "status": "PASS",
            "splits": split_results,
        }
    report = {
        "status": "PASS" if len(tasks) == 5 else "FAIL",
        "smoke_only_not_training": True,
        "optimizer_created": False,
        "parameters_updated": False,
        "checkpoint_saved": False,
        "offline": True,
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "snapshot": snapshot,
        "processor_class": type(processor).__name__,
        "model_class": type(model).__name__,
        "tasks": tasks,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
