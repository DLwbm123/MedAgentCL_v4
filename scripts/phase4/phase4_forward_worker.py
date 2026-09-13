#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from med_prism.adapters.injection import iter_rank1_wrappers
from med_prism.checkpoint import load_rank1_checkpoint
from med_prism.checkpoint.rank1_checkpoint import sha256_file
from med_prism.config import MODEL_ID, MODEL_REVISION


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def file_row(path: Path) -> dict:
    return json.loads(next(line for line in path.read_text().splitlines() if line))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-jsonl", type=Path, required=True)
    parser.add_argument("--variant", choices=["base", "adapter"], required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--logits-output", type=Path, required=True)
    args = parser.parse_args()
    if args.variant == "adapter" and not args.manifest:
        parser.error("--manifest is required for adapter")

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
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda:0").eval()
    model.requires_grad_(False)

    manifest = None
    adapter_sha = None
    active_tasks = []
    task_scaling = {}
    if args.variant == "adapter":
        manifest = load_rank1_checkpoint(
            model,
            args.manifest,
            expected_revision=MODEL_REVISION,
            trainable_task_id=None,
        )
        weights = args.manifest.parent / manifest["weights_file"]
        adapter_sha = sha256_file(weights)
        wrappers = list(iter_rank1_wrappers(model))
        active_tasks = list(wrappers[0][1].active_tasks)
        task_scaling = manifest["task_scaling"]
    model.eval()

    sample = file_row(args.sample_jsonl)
    image_path = sample["images"][0]
    question = sample["messages"][0]["content"].replace("<image>", "").strip()
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question},
        ],
    }]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to("cuda:0")
    input_ids = inputs["input_ids"].detach().cpu()
    image_key = next(
        key for key in ("pixel_values", "pixel_values_videos") if key in inputs
    )
    image_tensor = inputs[image_key].detach().cpu()
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].float().cpu()
        generated = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    generated_ids = generated[0, inputs["input_ids"].shape[1]:]
    generated_text = processor.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )
    args.logits_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(logits, args.logits_output)
    payload = {
        "status": "PASS",
        "run_label": args.run_label,
        "variant": args.variant,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "snapshot": snapshot,
        "base_identity": f"{MODEL_ID}@{MODEL_REVISION}",
        "manifest": str(args.manifest) if args.manifest else None,
        "adapter_weights_sha256": adapter_sha,
        "active_task_ids": active_tasks,
        "task_scaling": task_scaling,
        "input_token_ids": input_ids.tolist(),
        "input_token_ids_sha256": tensor_hash(input_ids),
        "image_path": image_path,
        "image_file_sha256": sha256_file(Path(image_path)),
        "image_tensor_key": image_key,
        "image_tensor_shape": list(image_tensor.shape),
        "image_tensor_sha256": tensor_hash(image_tensor),
        "logit_position": -1,
        "logits_shape": list(logits.shape),
        "logits_sha256": tensor_hash(logits),
        "generated_text": generated_text,
        "merged": False,
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "run_label": args.run_label,
        "active_task_ids": active_tasks,
        "logits_sha256": payload["logits_sha256"],
    }))


if __name__ == "__main__":
    main()
