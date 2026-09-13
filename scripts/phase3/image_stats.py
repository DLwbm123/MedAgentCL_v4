#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    snapshot = snapshot_download(MODEL_ID, revision=REVISION, cache_dir=str(args.cache_dir), local_files_only=True)
    processor = AutoProcessor.from_pretrained(snapshot, revision=REVISION, local_files_only=True)
    processor.image_processor.min_pixels = 200704
    processor.image_processor.max_pixels = 802816
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    records = []
    for line in args.train_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sample = json.loads(line)
        user = sample["messages"][0]["content"].replace("<image>", "").strip()
        messages = [{"role": "user", "content": [
            {"type": "image", "image": sample["images"][0]},
            {"type": "text", "text": user},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        encoded = processor(text=[text], images=images, videos=videos, return_tensors="pt")
        records.append({
            "id": sample["id"],
            "grid": encoded["image_grid_thw"].tolist(),
            "visual_tokens": int((encoded["input_ids"] == image_token_id).sum()),
            "pixel_tensor_shape": list(encoded["pixel_values"].shape),
        })
    counts = [row["visual_tokens"] for row in records]
    summary = {
        "sample_count": len(records),
        "visual_tokens_min": min(counts),
        "visual_tokens_max": max(counts),
        "visual_tokens_mean": sum(counts) / len(counts),
        "unique_grids": sorted({json.dumps(row["grid"]) for row in records}),
        "min_pixels": processor.image_processor.min_pixels,
        "max_pixels": processor.image_processor.max_pixels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "PASS", "summary": summary, "records": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
