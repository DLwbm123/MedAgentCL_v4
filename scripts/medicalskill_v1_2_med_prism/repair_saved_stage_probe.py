#!/usr/bin/env python3
"""Rebuild a missing reload probe from immutable saved Med-PRISM components."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = Path("/root/MedAgentCL_v4")
EVALUATOR = REPO / "scripts/medicalskill_v1_2_med_prism/evaluate_formal_v1_2.py"
TASK_DIRS = {
    1: "task_01_vqa",
    2: "task_02_diagnosis_classification",
    3: "task_03_concept_recognition",
    4: "task_04_visual_grounding",
    5: "task_05_reasoning_vqa",
}


def first_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise RuntimeError(f"Empty JSONL: {path}")


def load_evaluator():
    spec = importlib.util.spec_from_file_location("formal_eval_probe_repair", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    os.environ["MED_PRISM_DATA_ROOT"] = str(args.data_root.resolve())
    os.environ["MED_PRISM_ARTIFACT_ROOT"] = str(args.output_root.resolve())
    evaluation = load_evaluator()
    from qwen_vl_utils import process_vision_info

    row = first_jsonl(args.data_root / TASK_DIRS[args.stage] / "test.jsonl")
    model, processor, active = evaluation.load_model(args.stage)
    evaluation.set_active(model, list(range(1, args.stage + 1)))
    messages = evaluation.prompt_messages(row)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt")
    cpu_inputs = {key: value.detach().cpu().contiguous() for key, value in inputs.items() if isinstance(value, torch.Tensor)}
    gpu_inputs = {key: value.to("cuda:0") for key, value in cpu_inputs.items()}
    with torch.inference_mode():
        first = model(**gpu_inputs).logits[:, -1, :].detach().float().cpu().contiguous()
    del gpu_inputs, model, processor
    gc.collect()
    torch.cuda.empty_cache()

    second_model, _, second_active = evaluation.load_model(args.stage)
    evaluation.set_active(second_model, list(range(1, args.stage + 1)))
    second_inputs = {key: value.to("cuda:0") for key, value in cpu_inputs.items()}
    with torch.inference_mode():
        second = second_model(**second_inputs).logits[:, -1, :].detach().float().cpu().contiguous()
    difference = (first - second).abs()
    equivalent = torch.allclose(first, second, rtol=1e-4, atol=1e-5)
    finite = bool(torch.isfinite(first).all() and torch.isfinite(second).all())

    audit = args.output_root / "med_prism" / f"stage_{args.stage:02d}" / "runtime_audits"
    audit.mkdir(parents=True, exist_ok=True)
    inputs_path = audit / f"task{args.stage}_reload_probe_inputs.pt"
    logits_path = audit / f"task{args.stage}_pre_reload_logits.safetensors"
    metadata_path = audit / f"task{args.stage}_pre_reload_probe.json"
    torch.save(cpu_inputs, inputs_path)
    save_file({"last_token_logits": first}, str(logits_path))
    digest = hashlib.sha256(first.numpy().tobytes()).hexdigest()
    payload = {
        "status": "PASS" if equivalent and finite else "FAIL",
        "task_id": args.stage,
        "source": "two_independent_reloads_from_saved_components_after_training",
        "sample_id": row.get("id"),
        "image_count": len(row.get("images") or []),
        "inputs": str(inputs_path),
        "input_tensor_schema": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in cpu_inputs.items()},
        "logits": str(logits_path),
        "shape": list(first.shape),
        "float32_sha256": digest,
        "finite": finite,
        "independent_reload_max_abs_diff": float(difference.max()),
        "independent_reload_mean_abs_diff": float(difference.mean()),
        "rtol": 1e-4,
        "atol": 1e-5,
        "first_active_components": active,
        "second_active_components": second_active,
    }
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())