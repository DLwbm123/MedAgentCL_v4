#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from huggingface_hub import snapshot_download
from qwen_vl_utils import process_vision_info
import torch
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
PROJECTION_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
EXCLUDED = ("visual", "vision", "merger", "projector", "aligner", "patch_embed")


def nvidia_state() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free", "--format=csv,noheader"], text=True
    )


def module_inventory(model):
    names = []
    linear = []
    categories = {name: [] for name in PROJECTION_NAMES}
    language = []
    vision = []
    aligner = []
    special = {"merger": [], "deepstack_merger_list": [], "projector": [], "aligner": []}
    for name, module in model.named_modules():
        names.append(name)
        lower = name.lower()
        if isinstance(module, nn.Linear):
            linear.append(name)
            leaf = name.rsplit(".", 1)[-1]
            if leaf in categories:
                categories[leaf].append(name)
        if name.startswith("model.language_model"):
            language.append(name)
        if name.startswith("model.visual") or ".visual." in name or ".vision." in name:
            vision.append(name)
        if any(token in lower for token in ("merger", "projector", "aligner")):
            aligner.append(name)
        for key in special:
            if key in lower:
                special[key].append(name)

    candidates = [
        name for name in linear
        if name.startswith("model.language_model")
        and name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}
        and not any(token in name.lower() for token in EXCLUDED)
    ]
    candidate_classes = {
        "language": candidates,
        "vision": [name for name in candidates if any(token in name.lower() for token in ("visual", "vision"))],
        "aligner": [name for name in candidates if any(token in name.lower() for token in ("merger", "projector", "aligner"))],
    }
    return names, {
        "total_modules": len(names), "linear_modules": len(linear),
        "language_modules": len(language), "vision_modules": len(vision), "aligner_modules": len(aligner),
        "projection_counts": {key: len(value) for key, value in categories.items()},
        "q_proj_paths": categories["q_proj"], "v_proj_paths": categories["v_proj"],
        "special_paths": special,
        "target_filter_preview": {
            "language_candidate_count": len(candidate_classes["language"]),
            "vision_candidate_count": len(candidate_classes["vision"]),
            "aligner_candidate_count": len(candidate_classes["aligner"]),
            "candidates": candidates,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--sample-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "huggingface" / "hub")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = json.loads(args.sample_jsonl.read_text().splitlines()[0])
    image_path = sample["resolved_images"][0]["resolved"]
    question = sample["messages"][0]["content"].replace("<image>", "").strip()
    messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question}]}]

    before = nvidia_state()
    snapshot = snapshot_download(MODEL_ID, revision=args.revision, cache_dir=str(args.cache_dir), local_files_only=True)
    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(snapshot, revision=args.revision, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        snapshot, revision=args.revision, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(args.device).eval()
    load_seconds = time.perf_counter() - load_start
    names, inventory = module_inventory(model)
    preview = inventory["target_filter_preview"]
    if preview["language_candidate_count"] <= 0 or preview["vision_candidate_count"] or preview["aligner_candidate_count"]:
        raise RuntimeError(f"Invalid language-only target preview: {preview}")

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to(args.device)
    grid_key = next(key for key in ("image_grid_thw", "image_grid_hw") if key in inputs)
    input_tokens = int(inputs["attention_mask"].sum())
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    inference_seconds = time.perf_counter() - start
    output_ids = generated[0, inputs.input_ids.shape[1]:]
    raw = processor.decode(output_ids, skip_special_tokens=True)
    if not raw.strip():
        raise RuntimeError("Empty Qwen3-VL response")

    record = {
        "model_id": MODEL_ID, "revision": args.revision, "snapshot": snapshot,
        "image": image_path, "prompt": messages, "raw_output": raw,
        "normalized_output": raw.strip(), "input_tokens": input_tokens,
        "output_tokens": int(output_ids.numel()), "visual_grid": inputs[grid_key].tolist(),
        "single_sample_not_performance_evaluation": True,
    }
    memory = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu": torch.cuda.get_device_name(),
        "dtype": "bfloat16", "attention_implementation": "sdpa",
        "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved(),
        "load_seconds": load_seconds, "inference_seconds": inference_seconds,
        "cpu_offload": False, "oom": False, "nvidia_smi_before": before, "nvidia_smi_after": nvidia_state(),
    }
    for filename, rows in (("qwen3_vl_inference.jsonl", [record]), ("qwen3_vl_raw_outputs.jsonl", [{"raw_output": raw}])):
        (args.output_dir / filename).write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows))
    (args.output_dir / "qwen3_vl_memory.json").write_text(json.dumps(memory, indent=2) + "\n")
    (args.output_dir / "qwen3_vl_module_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (args.output_dir / "qwen3_vl_module_names.txt").write_text("\n".join(names) + "\n")
    (args.output_dir / "qwen3_vl_target_filter_preview.txt").write_text(json.dumps(preview, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "output": raw.strip(), "memory": memory, "preview": preview}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
