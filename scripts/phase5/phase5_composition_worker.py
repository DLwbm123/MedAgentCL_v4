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

from med_prism.checkpoint.rank1_checkpoint import sha256_file
from med_prism.checkpoint.shared_private_checkpoint import (
    _read_manifest,
    _validate,
    compose_shared_private,
    load_private_component,
    load_shared_component,
)
from med_prism.config import MODEL_ID, MODEL_REVISION


ROOT = Path("/root/MedAgentCL_v4")
SHARED = ROOT / "output/phase5_shared_private/shared/after_task_2/shared_manifest.json"
PRIVATE = {
    1: ROOT / "output/phase5_shared_private/private/task_1/private_manifest.json",
    2: ROOT / "output/phase5_shared_private/private/task_2/private_manifest.json",
}
SAMPLE = ROOT / "artifacts/phase5_shared_private/data/task1_vqa_train.jsonl"


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def first_row(path: Path) -> dict:
    return json.loads(next(line for line in path.read_text().splitlines() if line))


def negative_tests(model) -> dict:
    checks = {}
    try:
        load_shared_component(model, SHARED)
    except ValueError as error:
        checks["duplicate_shared"] = str(error)
    try:
        load_private_component(model, PRIVATE[1], trainable=False)
    except ValueError as error:
        checks["duplicate_private"] = str(error)
    try:
        _read_manifest(
            Path("/definitely/missing/shared_manifest.json"), "shared_manifest.json"
        )
    except FileNotFoundError as error:
        checks["missing_manifest"] = str(error)

    path, manifest = _read_manifest(SHARED, "shared_manifest.json")
    wrong_revision = dict(manifest)
    wrong_revision["model_revision"] = "wrong"
    try:
        _validate(model, path, wrong_revision, component_type="shared")
    except ValueError as error:
        checks["wrong_revision"] = str(error)
    wrong_schema = dict(manifest)
    wrong_schema["tensor_schema"] = manifest["tensor_schema"][:-1]
    try:
        _validate(model, path, wrong_schema, component_type="shared")
    except ValueError as error:
        checks["wrong_tensor_schema"] = str(error)
    expected = {
        "duplicate_shared": "Duplicate shared component",
        "duplicate_private": "Duplicate private task component",
        "missing_manifest": "Explicit shared_manifest.json path is required",
        "wrong_revision": "Component model revision mismatch",
        "wrong_tensor_schema": "Component tensor schema key mismatch",
    }
    return {
        "status": (
            "PASS"
            if all(expected[key] in checks.get(key, "") for key in expected)
            else "BLOCKED"
        ),
        "errors": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["task1_skill_aware", "task2_skill_aware", "cumulative"],
        required=True,
    )
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--private-order", default=None)
    parser.add_argument("--negative-tests", action="store_true")
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--logits-output", type=Path, required=True)
    args = parser.parse_args()

    task_ids = {
        "task1_skill_aware": [1],
        "task2_skill_aware": [2],
        "cumulative": [1, 2],
    }[args.mode]
    if args.private_order:
        task_ids = [int(value) for value in args.private_order.split(",")]
        if sorted(task_ids) != [1, 2] or args.mode != "cumulative":
            raise ValueError("--private-order is only valid for cumulative 1,2")

    snapshot = snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(
        snapshot, revision=MODEL_REVISION, local_files_only=True
    )
    if hasattr(processor.image_processor, "min_pixels"):
        processor.image_processor.min_pixels = 200704
    if hasattr(processor.image_processor, "max_pixels"):
        processor.image_processor.max_pixels = 802816
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
        shared_manifest=SHARED,
        private_manifests=[PRIVATE[task_id] for task_id in task_ids],
    )
    model.eval()

    sample = first_row(SAMPLE)
    image_path = sample["images"][0]
    question = sample["messages"][0]["content"].replace("<image>", "").strip()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to("cuda:0")
    image_key = next(
        key for key in ("pixel_values", "pixel_values_videos") if key in inputs
    )
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
    generated_ids = generated[0, inputs["input_ids"].shape[1] :]
    generated_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    shared_manifest = json.loads(SHARED.read_text())
    private_manifests = [json.loads(PRIVATE[item].read_text()) for item in task_ids]
    payload = {
        "status": "PASS",
        "mode": args.mode,
        "run_label": args.run_label,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "base_identity": f"{MODEL_ID}@{MODEL_REVISION}",
        "active_component_manifest": active,
        "shared_weights_sha256": shared_manifest["weights_sha256"],
        "private_weights_sha256": {
            str(item["task_id"]): item["weights_sha256"] for item in private_manifests
        },
        "input_token_ids": inputs["input_ids"].detach().cpu().tolist(),
        "input_token_ids_sha256": tensor_hash(inputs["input_ids"]),
        "image_path": image_path,
        "image_file_sha256": sha256_file(Path(image_path)),
        "image_tensor_key": image_key,
        "image_tensor_shape": list(inputs[image_key].shape),
        "image_tensor_sha256": tensor_hash(inputs[image_key]),
        "logits_sha256": tensor_hash(logits),
        "generated_text": generated_text,
        "negative_tests": negative_tests(model) if args.negative_tests else None,
        "merged": False,
    }
    args.logits_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(logits, args.logits_output)
    args.metadata_output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "mode": args.mode,
                "run_label": args.run_label,
                "tasks": active["private_task_ids"],
                "logits_sha256": payload["logits_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
