#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from swift.template import TEMPLATE_MAPPING
from swift.template.constant import MLLMTemplateType
from swift.template.template_inputs import StdTemplateInputs


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def shapes(inputs):
    return {key: list(value.shape) for key, value in inputs.items() if hasattr(value, "shape")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--sample-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "huggingface" / "hub")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample = json.loads(args.sample_jsonl.read_text(encoding="utf-8").splitlines()[0])
    image_path = sample["resolved_images"][0]["resolved"]
    image = Image.open(image_path).convert("RGB")
    second = image.resize((896, 896))

    snapshot = snapshot_download(MODEL_ID, revision=args.revision, cache_dir=str(args.cache_dir), local_files_only=True)
    processor = AutoProcessor.from_pretrained(snapshot, revision=args.revision, local_files_only=True)
    image_processor = processor.image_processor
    min_pixels = 256 * 28 * 28
    max_pixels = 1024 * 28 * 28
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = min_pixels
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = max_pixels

    official_messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "Answer the medical image question briefly: " + sample["messages"][0]["content"].replace("<image>", "").strip()},
        ],
    }]
    text = processor.apply_chat_template(official_messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(official_messages)
    encoded = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")

    swift_record = {"messages": sample["messages"], "images": [image_path]}
    swift_inputs = StdTemplateInputs.from_dict(swift_record)
    assert swift_inputs.is_multimodal
    assert swift_inputs.messages[0]["content"].count("<image>") == len(swift_inputs.images)

    pair_text = [text, text]
    pair = processor(text=pair_text, images=[image, second], return_tensors="pt", padding=True)
    grid_key = next((key for key in ("image_grid_thw", "image_grid_hw") if key in encoded), None)
    pixel_key = next((key for key in ("pixel_values", "pixel_values_videos") if key in encoded), None)
    if pixel_key is None or encoded[pixel_key].numel() == 0:
        raise RuntimeError("No non-empty visual pixel tensor")
    if grid_key is None or encoded[grid_key].numel() == 0:
        raise RuntimeError("No non-empty visual grid tensor")
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    visual_token_count = int((encoded["input_ids"] == image_token_id).sum())
    if visual_token_count <= 0:
        raise RuntimeError("No visual tokens found in input_ids")

    grid_rows = pair.get(grid_key)
    dynamic_resolution = bool(grid_rows is not None and len(grid_rows) >= 2 and not grid_rows[0].equal(grid_rows[1]))
    tensor_payload = {
        "single": shapes(encoded), "different_resolution_pair": shapes(pair),
        "grid_key": grid_key, "pixel_key": pixel_key,
        "single_grid": encoded[grid_key].tolist(),
        "pair_grid": None if grid_rows is None else grid_rows.tolist(),
        "visual_token_id": image_token_id, "visual_token_count": visual_token_count,
        "dynamic_resolution_evidence": dynamic_resolution,
    }
    comparison = {
        "official_messages": official_messages,
        "swift_jsonl_record": swift_record,
        "swift_normalized_messages": swift_inputs.messages,
        "swift_normalized_images": [str(item) for item in swift_inputs.images],
        "placeholder_count": swift_inputs.messages[0]["content"].count("<image>"),
        "image_count": len(swift_inputs.images),
        "template_registry_found": MLLMTemplateType.qwen3_vl in TEMPLATE_MAPPING,
    }
    summary = {
        "status": "PASS" if dynamic_resolution else "PARTIAL",
        "model_id": MODEL_ID, "revision": args.revision, "snapshot": snapshot,
        "processor_class": type(processor).__name__, "image_processor_class": type(image_processor).__name__,
        "processor_settings": {
            "min_pixels": getattr(image_processor, "min_pixels", None),
            "max_pixels": getattr(image_processor, "max_pixels", None),
            "patch_size": getattr(image_processor, "patch_size", None),
            "merge_size": getattr(image_processor, "merge_size", None),
        },
        "real_image": image_path, "real_image_size": list(image.size), "second_image_size": list(second.size),
        "dynamic_resolution": dynamic_resolution,
    }
    (args.output_dir / "qwen3_vl_processor_smoke.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "qwen3_vl_tensor_shapes.json").write_text(json.dumps(tensor_payload, indent=2) + "\n")
    (args.output_dir / "qwen3_vl_template_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
