#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import Qwen3VLForConditionalGeneration

from med_prism.adapters.injection import collect_language_targets, iter_rank1_wrappers
from med_prism.checkpoint import load_rank1_checkpoint
from med_prism.checkpoint.rank1_checkpoint import sha256_file
from med_prism.config import MODEL_ID, MODEL_REVISION


def tensor_hash(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    weights = args.manifest.parent / manifest["weights_file"]
    state = load_file(str(weights))
    payload = {
        "status": "PENDING",
        "model_id": MODEL_ID,
        "immutable_revision": MODEL_REVISION,
        "manifest": str(args.manifest),
        "weights": str(weights),
        "weights_sha256_matches": sha256_file(weights) == manifest["safetensors_sha256"],
        "tensor_count": len(state),
        "task_ids": manifest["task_ids"],
        "wrapper_count": manifest["wrapper_count"],
        "expert_count": manifest["expert_count"],
        "load_model_requested": args.load_model,
    }
    if args.load_model:
        snapshot = snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            snapshot,
            revision=MODEL_REVISION,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to("cuda:0")
        model.requires_grad_(False)
        loaded = load_rank1_checkpoint(
            model,
            args.manifest,
            expected_revision=MODEL_REVISION,
            trainable_task_id=None,
        )
        inventory = collect_language_targets(model)
        wrappers = list(iter_rank1_wrappers(model))
        parameters = dict(model.named_parameters())
        mismatches = []
        for key, tensor in state.items():
            if key not in parameters or tensor_hash(tensor) != tensor_hash(parameters[key]):
                mismatches.append(key)
        payload.update({
            "snapshot": snapshot,
            "loaded_manifest_revision": loaded["immutable_revision"],
            "loaded_wrapper_count": len(wrappers),
            "loaded_target_hash": inventory.sha256,
            "loaded_task_ids": list(wrappers[0][1].task_ids),
            "loaded_active_tasks": list(wrappers[0][1].active_tasks),
            "all_adapter_parameters_frozen": all(
                not parameter.requires_grad
                for name, parameter in model.named_parameters()
                if ".experts." in name
            ),
            "tensor_hash_mismatches": mismatches,
        })
    passed = (
        payload["weights_sha256_matches"]
        and payload["wrapper_count"] == 72
        and payload["task_ids"] == [1, 2]
        and payload["tensor_count"] == 72 * 2 * 4 * 2
    )
    if args.load_model:
        passed = passed and (
            payload["loaded_wrapper_count"] == 72
            and payload["loaded_task_ids"] == [1, 2]
            and payload["loaded_active_tasks"] == [1, 2]
            and payload["all_adapter_parameters_frozen"]
            and not payload["tensor_hash_mismatches"]
        )
    payload["status"] = "PASS" if passed else "BLOCKED"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise SystemExit("Phase 4 checkpoint reload audit failed")


if __name__ == "__main__":
    main()
