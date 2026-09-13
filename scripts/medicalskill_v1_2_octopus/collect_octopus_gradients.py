#!/usr/bin/env python3
"""Collect original Octopus historical-prefix gradients on current-task data."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed.fsdp as fsdp

if not hasattr(fsdp, "FSDPModule"):
    class FSDPModule:
        pass

    fsdp.FSDPModule = FSDPModule

from accelerate.utils import send_to_device
from peft import PeftModel
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from swift.arguments import SftArguments
from swift.pipelines.train.sft import SwiftSft

from med_prism.baselines.octopus import canonical_base_weight_name
from scripts.medicalskill_v1_2_baselines.baseline_harness import (
    MODEL_ID,
    MODEL_REVISION,
    SEED,
    atomic_json,
    inspect_adapter,
    sha256_file,
    sha256_json,
    utc_now,
)


def target_weight_keys(checkpoint: Path) -> set[str]:
    from safetensors import safe_open

    info = inspect_adapter(checkpoint)
    if not info["adapter_weights"].endswith(".safetensors"):
        state = torch.load(info["adapter_weights"], map_location="cpu", weights_only=True)
        keys = state
    else:
        with safe_open(info["adapter_weights"], framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
    return {
        canonical_base_weight_name(key)
        for key in keys
        if ".lora_B." in key or key.endswith(".lora_B.weight")
    }


def atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        save_file(tensors, str(temporary))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def expected_identity(
    *,
    task_id: int,
    dataset: Path,
    samples: int,
    adapters: list[dict[str, Any]],
    target_modules: list[str],
) -> dict[str, Any]:
    value = {
        "task_id": task_id,
        "source_current_task_dataset": str(dataset.resolve()),
        "source_current_task_dataset_sha256": sha256_file(dataset),
        "sample_policy": "deterministic_prefix_from_locked_current_task_train_split",
        "sample_count_requested": samples,
        "historical_stage2_prefix": adapters,
        "historical_prefix_length": len(adapters),
        "target_modules": sorted(target_modules),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": SEED,
        "algorithm": "octopus_historical_prefix_weight_gradient_response",
    }
    value["artifact_identity_sha256"] = sha256_json(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, choices=range(2, 6), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--historical-adapter", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--target-module", action="append", default=[])
    args = parser.parse_args()
    target_modules = sorted(args.target_module or ["q_proj", "v_proj"])
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("Sample and batch sizes must be positive")
    adapters = [path.resolve() for path in args.historical_adapter]
    if not 1 <= len(adapters) <= args.task_id - 1:
        raise ValueError("Historical adapters must form a non-empty prefix")
    adapter_metadata = []
    expected_targets = None
    for task, checkpoint in enumerate(adapters, 1):
        info = inspect_adapter(checkpoint)
        if info["r"] != 16:
            raise RuntimeError(f"Historical Stage-2 task {task} is not rank 16")
        if expected_targets is None:
            expected_targets = info["target_modules"]
        if info["target_modules"] != target_modules:
            raise RuntimeError("Historical adapter target-module mismatch")
        adapter_metadata.append(
            {
                "task": task,
                "role": "stage2",
                "checkpoint": info["checkpoint"],
                "adapter_weights_sha256": info["adapter_weights_sha256"],
                "adapter_config_sha256": info["adapter_config_sha256"],
            }
        )
    dataset = args.dataset.resolve()
    identity = expected_identity(
        task_id=args.task_id,
        dataset=dataset,
        samples=args.samples,
        adapters=adapter_metadata,
        target_modules=target_modules,
    )
    output = args.output_dir.resolve()
    artifact = output / "historical_prefix_gradients.safetensors"
    manifest_path = output / "gradient_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            previous.get("status") == "PASS"
            and previous.get("artifact_identity_sha256")
            == identity["artifact_identity_sha256"]
            and artifact.is_file()
            and previous.get("gradient_artifact_sha256") == sha256_file(artifact)
        ):
            print(f"SKIP valid gradient artifact: {manifest_path}")
            return 0
        raise RuntimeError(f"Refusing stale gradient manifest: {manifest_path}")
    if artifact.exists():
        raise RuntimeError(f"Refusing unmanifested gradient artifact: {artifact}")

    swift_args = SftArguments(
        model=MODEL_ID,
        model_revision=MODEL_REVISION,
        template="qwen3_vl",
        dataset=[f"{dataset}#{args.samples}"],
        output_dir=str(output / "swift_preparation"),
        torch_dtype="bfloat16",
        attn_impl="sdpa",
        max_length=args.max_length,
        max_pixels=200704,
        per_device_train_batch_size=args.batch_size,
        dataset_num_proc=1,
        dataloader_num_workers=0,
        split_dataset_ratio=0,
        dataset_shuffle=False,
        train_dataloader_shuffle=False,
        use_hf=True,
        seed=SEED,
        data_seed=SEED,
        report_to=[],
    )
    started = time.monotonic()
    pipeline = SwiftSft(swift_args)
    train_dataset, _ = pipeline._prepare_dataset()
    model = pipeline.model
    template = pipeline.template
    for checkpoint in adapters:
        wrapped = PeftModel.from_pretrained(
            model, str(checkpoint), is_trainable=False
        )
        model = wrapped.merge_and_unload()
    if template.use_model:
        template.model = model
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "language_model") and hasattr(model.language_model, "config"):
        model.language_model.config.use_cache = False

    target_names = set()
    for checkpoint in adapters:
        target_names.update(target_weight_keys(checkpoint))
    model_names: dict[str, tuple[str, torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        key = canonical_base_weight_name(name)
        if key in target_names:
            if key in model_names:
                raise RuntimeError(f"Duplicate canonical base weight: {key}")
            model_names[key] = (name, parameter)
    missing = sorted(target_names - set(model_names))
    if missing:
        raise RuntimeError(f"Qwen target weights are missing: {missing[:10]}")
    model.requires_grad_(False)
    for _, parameter in model_names.values():
        parameter.requires_grad_(True)
    device = next(model.parameters()).device
    if device.type == "cpu":
        model = model.to("cuda:0")
        device = torch.device("cuda:0")
    model.train()
    collator = partial(template.data_collator, padding_to=None)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        pin_memory=True,
    )
    sums: dict[str, torch.Tensor] = {}
    batches = 0
    examples = 0
    for batch in loader:
        batch = send_to_device(batch, device)
        model.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        loss.backward()
        current = {}
        for key, (_, parameter) in model_names.items():
            if parameter.grad is None:
                raise RuntimeError(f"Missing base-weight gradient for {key}")
            current[key] = parameter.grad.detach().cpu().float()
        for key, value in current.items():
            sums[key] = value if key not in sums else sums[key] + value
        batches += 1
        first = next(iter(batch.values()))
        examples += int(first.shape[0]) if hasattr(first, "shape") else args.batch_size
    if not batches:
        raise RuntimeError("Gradient collection received no batches")
    averaged = {
        key: (value / batches).to(torch.bfloat16).contiguous()
        for key, value in sums.items()
    }
    atomic_safetensors(artifact, averaged)
    payload = {
        "status": "PASS",
        "format_version": "medicalskill_cl_octopus_gradient_v1",
        "created_at": utc_now(),
        **identity,
        "sample_count_observed": min(len(train_dataset), examples),
        "batch_count": batches,
        "gradient_tensor_count": len(averaged),
        "gradient_dtype": "bfloat16",
        "gradient_artifact": str(artifact),
        "gradient_artifact_sha256": sha256_file(artifact),
        "gradient_artifact_bytes": artifact.stat().st_size,
        "collection_seconds_internal": time.monotonic() - started,
        "current_task_data_only": True,
        "old_raw_examples_loaded": False,
        "historical_semantics": (
            "Each artifact measures a cumulative historical Stage-2 prefix "
            "on the locked current-task train subset, matching original Octopus."
        ),
    }
    atomic_json(manifest_path, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
