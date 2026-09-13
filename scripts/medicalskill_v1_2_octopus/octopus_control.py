#!/usr/bin/env python3
"""Failure-safe lifecycle controller for formal Octopus runs."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from med_prism.baselines.octopus import (
    inspect_octopus_stage2_serialization,
    stage2_inference_stack,
    validate_gradient_manifest,
)
from scripts.medicalskill_v1_2_baselines.baseline_harness import (
    HistoricalMemory,
    TASK_ORDER,
    atomic_json,
    build_run_manifest,
    inspect_adapter,
    load_json,
    require_adapter_contract,
    sha256_file,
    trainer_state,
    validate_completion,
    validate_runtime_record,
    write_or_validate_run_manifest,
)

METHOD = "octopus_qwen3vl_r16"
TARGET_MODULES = ["q_proj", "v_proj"]
RANK = 16


def completion_path(root: Path, stage: int) -> Path:
    return root / "octopus" / f"stage_{stage:02d}" / "completion.json"


def load_run(root: Path) -> dict[str, Any]:
    value = load_json(root / "run_manifest.json")
    if value.get("status") != "PASS" or value.get("method") != METHOD:
        raise RuntimeError("Invalid Octopus run manifest")
    return value


def init_run(args) -> int:
    payload = build_run_manifest(
        method=METHOD,
        method_family="gradient_space_orthogonalization",
        data_root=args.data_root,
        output_root=args.output_root,
        capacity_policy={
            "class": "task_growing",
            "rank_per_task": RANK,
            "lora_alpha": 32,
            "target_modules": TARGET_MODULES,
            "stage1_and_stage2_same_rank": True,
            "parameter_equality_claim": False,
        },
        historical_memory=HistoricalMemory(
            raw_examples=False,
            features_or_prototypes=False,
            statistics_or_masks=True,
            historical_parameters=True,
            notes=(
                "Retains frozen Stage-2 adapters and current-task gradient "
                "response artifacts; does not explicitly retain old-task raw examples."
            ),
        ),
        method_config={
            "lifecycle": (
                "Stage-1; same-task Stage-1 initializes Stage-2; historical "
                "Stage-2 prefix gradients on current-task data; freeze Stage-2"
            ),
            "stage1_initialization": (
                "base for task 1, previous Stage-1 cumulative adapter thereafter"
            ),
            "stage2_initialization": "same_task_stage1",
            "stage2_training_forward": (
                "same-task trainable/default LoRA plus frozen historical "
                "Stage-2 extras_A/extras_B"
            ),
            "stage2_serialization": "current trainable/default LoRA only",
            "primary_inference": "base_plus_stage2_adapters_tasks_1_to_t",
            "stage1_active_at_primary_inference": False,
            "oracle_task_id": False,
            "target_modules": TARGET_MODULES,
            "rank": RANK,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lambda_1": args.lambda_1,
            "lambda_2": args.lambda_2,
            "gradient_samples": args.gradient_samples,
            "gradient_sample_policy": (
                "deterministic prefix of locked current-task train split"
            ),
            "historical_gradient_semantics": (
                "one artifact per cumulative historical Stage-2 prefix, "
                "summed by the Stage-2 trainer as in original Octopus"
            ),
            "learning_rate": args.learning_rate,
            "epochs_per_training_stage": 1,
            "task1_stage2": (
                "official second training stage initialized from Task1 Stage-1; "
                "task loss only with lambda_1=lambda_2=0"
            ),
        },
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2))
    else:
        args.output_root.mkdir(parents=True, exist_ok=True)
        write_or_validate_run_manifest(args.output_root / "run_manifest.json", payload)
        print(json.dumps(payload, indent=2))
    return 0


def stage_ready(args) -> int:
    run = load_run(args.output_root)
    if args.stage > 1:
        validate_completion(
            completion_path(args.output_root, args.stage - 1),
            method=METHOD,
            stage=args.stage - 1,
            immutable_config_sha256=run["immutable_config_sha256"],
        )
    print("PASS")
    return 0


def write_stage2_initialization(
    path: Path, stage: int, stage1_checkpoint: Path, mode: str
) -> dict[str, Any]:
    info = require_adapter_contract(
        stage1_checkpoint, rank=RANK, target_modules=TARGET_MODULES
    )
    value = {
        "status": "PASS",
        "format_version": "medicalskill_cl_octopus_stage2_init_v1",
        "stage": stage,
        "source_role": "same_task_stage1",
        "source_task": stage,
        "source_checkpoint": info["checkpoint"],
        "source_adapter_weights_sha256": info["adapter_weights_sha256"],
        "source_adapter_config_sha256": info["adapter_config_sha256"],
        "rank": RANK,
        "target_modules": TARGET_MODULES,
        "initialization_mode": mode,
    }
    atomic_json(path, value)
    return value


def record_stage2_init(args) -> int:
    value = write_stage2_initialization(
        args.manifest.resolve(),
        args.stage,
        args.stage1_checkpoint.resolve(),
        "peft_from_pretrained_trainable_same_task_stage1",
    )
    print(json.dumps(value, indent=2))
    return 0


def copy_task1(args) -> int:
    if args.stage != 1:
        raise ValueError("The copy lifecycle is only valid for Task 1")
    source = args.stage1_checkpoint.resolve()
    destination = args.stage2_checkpoint.resolve()
    original = require_adapter_contract(
        source, rank=RANK, target_modules=TARGET_MODULES
    )
    if destination == source:
        copied = original
        mode = "exact_same_task_stage1_checkpoint_alias"
    else:
        if destination.exists():
            raise RuntimeError(f"Refusing to overwrite Stage-2 copy: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        copied = require_adapter_contract(
            destination, rank=RANK, target_modules=TARGET_MODULES
        )
        mode = "exact_same_task_stage1_copy"
    if copied["adapter_weights_sha256"] != original["adapter_weights_sha256"]:
        raise RuntimeError("Task-1 Stage-1 -> Stage-2 identity mismatch")
    value = write_stage2_initialization(
        args.manifest.resolve(), 1, source, mode
    )
    value["stage2_checkpoint"] = str(destination)
    value["stage2_adapter_weights_sha256"] = copied["adapter_weights_sha256"]
    atomic_json(args.manifest.resolve(), value)
    print(json.dumps(value, indent=2))
    return 0


def _validate_gradient_set(
    output_root: Path, data_root: Path, stage: int, manifests: list[Path]
) -> list[dict[str, Any]]:
    if len(manifests) != stage - 1:
        raise RuntimeError(
            f"Task {stage} requires {stage - 1} gradient manifests, "
            f"received {len(manifests)}"
        )
    dataset = data_root.resolve() / TASK_ORDER[stage - 1] / "train.jsonl"
    dataset_hash = sha256_file(dataset)
    previous_stack = stage2_inference_stack(
        completion_path(output_root, task) for task in range(1, stage)
    )
    result = []
    for prefix_length, path in enumerate(manifests, 1):
        value = validate_gradient_manifest(
            path,
            current_dataset_sha256=dataset_hash,
            target_modules=TARGET_MODULES,
        )
        if value["task_id"] != stage:
            raise RuntimeError(f"Gradient task mismatch: {path}")
        if value["historical_prefix_length"] != prefix_length:
            raise RuntimeError(f"Gradient prefix order mismatch: {path}")
        expected = previous_stack[:prefix_length]
        actual = value["historical_stage2_prefix"]
        if [
            (item["task"], item["adapter_weights_sha256"]) for item in actual
        ] != [
            (item["task"], item["adapter_weights_sha256"]) for item in expected
        ]:
            raise RuntimeError(f"Gradient historical adapter mismatch: {path}")
        result.append(value)
    return result


def validate_gradients(args) -> int:
    values = _validate_gradient_set(
        args.output_root,
        args.data_root,
        args.stage,
        [path.resolve() for path in args.manifest],
    )
    print(json.dumps({"status": "PASS", "count": len(values)}, indent=2))
    return 0


def _runtime_values(paths: list[Path], category: str) -> list[dict[str, Any]]:
    return [validate_runtime_record(path, category) for path in paths]


def finalize(args) -> int:
    output_root = args.output_root.resolve()
    data_root = args.data_root.resolve()
    run = load_run(output_root)
    stage1 = require_adapter_contract(
        args.stage1_checkpoint, rank=RANK, target_modules=TARGET_MODULES
    )
    stage2 = require_adapter_contract(
        args.stage2_checkpoint, rank=RANK, target_modules=TARGET_MODULES
    )
    initialization = load_json(args.stage2_initialization)
    if (
        initialization.get("status") != "PASS"
        or initialization.get("stage") != args.stage
        or initialization.get("source_task") != args.stage
        or initialization.get("source_role") != "same_task_stage1"
        or initialization.get("source_adapter_weights_sha256")
        != stage1["adapter_weights_sha256"]
    ):
        raise RuntimeError("Stage-2 did not validate against same-task Stage-1")
    stage1_runtime = validate_runtime_record(args.stage1_runtime, "stage1_train")
    if stage1_runtime["status"] != "PASS":
        raise RuntimeError("Stage-1 runtime is not PASS")
    stage2_runtime = None
    if args.stage2_runtime is None:
        raise RuntimeError("Stage-2 runtime is required for every task")
    stage2_runtime = validate_runtime_record(args.stage2_runtime, "stage2_train")
    if stage2_runtime["status"] != "PASS":
        raise RuntimeError("Stage-2 runtime is not PASS")
    gradient_paths = [path.resolve() for path in args.gradient_manifest]
    gradient_values = (
        _validate_gradient_set(output_root, data_root, args.stage, gradient_paths)
        if args.stage > 1
        else []
    )
    gradient_runtimes = _runtime_values(
        [path.resolve() for path in args.gradient_runtime],
        "historical_gradient_collection",
    )
    if len(gradient_runtimes) != len(gradient_values):
        raise RuntimeError("Gradient runtime/artifact count mismatch")
    if any(value["status"] != "PASS" for value in gradient_runtimes):
        raise RuntimeError("A gradient collection runtime failed")

    prior = [
        validate_completion(
            completion_path(output_root, task),
            method=METHOD,
            stage=task,
            immutable_config_sha256=run["immutable_config_sha256"],
        )
        for task in range(1, args.stage)
    ]
    previous_stage1 = prior[-1]["stage1_adapter"] if prior else None
    stage2_serialization = inspect_octopus_stage2_serialization(
        Path(stage2["checkpoint"])
    )
    current_stub = {
        "status": "PASS",
        "stage": args.stage,
        "stage2_checkpoint": stage2["checkpoint"],
    }
    temporary_completion = output_root / ".octopus_current_completion.json"
    atomic_json(temporary_completion, current_stub)
    try:
        inference_stack = stage2_inference_stack(
            [completion_path(output_root, task) for task in range(1, args.stage)]
            + [temporary_completion]
        )
    finally:
        temporary_completion.unlink(missing_ok=True)

    all_stage1 = [item["stage1_adapter"] for item in prior] + [stage1]
    all_stage2 = [item["stage2_adapter"] for item in prior] + [stage2]
    unique_stored = {
        item["checkpoint"]: item for item in all_stage1 + all_stage2
    }
    stored_parameters = sum(
        int(item["adapter_parameters"]) for item in unique_stored.values()
    )
    active_parameters = sum(
        int(item["adapter_parameters"]) for item in all_stage2
    )
    checkpoint_bytes = sum(
        int(item["checkpoint_bytes"]) for item in unique_stored.values()
    )
    current_aux = sum(
        Path(value["gradient_artifact"]).stat().st_size
        + path.stat().st_size
        for value, path in zip(gradient_values, gradient_paths)
    )
    previous_aux = (
        int(prior[-1]["parameter_accounting"]["accumulated_auxiliary_state_bytes"])
        if prior
        else 0
    )

    stage1_seconds = float(stage1_runtime["wall_clock_seconds"])
    stage2_seconds = (
        float(stage2_runtime["wall_clock_seconds"]) if stage2_runtime else 0.0
    )
    gradient_seconds = sum(
        float(value["wall_clock_seconds"]) for value in gradient_runtimes
    )
    gpu_hours = float(stage1_runtime["gpu_hours"]) + sum(
        float(value["gpu_hours"]) for value in gradient_runtimes
    )
    if stage2_runtime:
        gpu_hours += float(stage2_runtime["gpu_hours"])
    formal_task_samples = int(
        run["dataset"]["tasks"][TASK_ORDER[args.stage - 1]]["train"]["records"]
    )
    task_samples = int(
        stage1_runtime.get("number_of_samples") or formal_task_samples
    )
    stage2_samples = int(
        (stage2_runtime or {}).get("number_of_samples") or 0
    )
    stage1_state = trainer_state(Path(stage1["checkpoint"]))
    stage2_state = (
        trainer_state(Path(stage2["checkpoint"]))
        if args.stage > 1
        else {"training_steps": 0, "trainer_state": None}
    )
    steps = sum(
        int(value or 0)
        for value in (
            stage1_state.get("training_steps"),
            stage2_state.get("training_steps"),
        )
    )
    peak_values = []
    gpu_models = []
    runtime_values = [stage1_runtime, *gradient_runtimes]
    if stage2_runtime:
        runtime_values.append(stage2_runtime)
    for value in runtime_values:
        if value.get("peak_gpu_memory_mib"):
            peak_values.extend(value["peak_gpu_memory_mib"].values())
        gpu_models.extend(value.get("gpu_models") or [])

    payload = {
        "status": "PASS",
        "format_version": "medicalskill_cl_baseline_stage_v1",
        "method": METHOD,
        "stage": args.stage,
        "task_name": TASK_ORDER[args.stage - 1],
        "immutable_config_sha256": run["immutable_config_sha256"],
        "stage1_checkpoint": stage1["checkpoint"],
        "stage2_checkpoint": stage2["checkpoint"],
        "stage1_adapter": stage1,
        "stage2_adapter": stage2,
        "stage1_initialization": {
            "mode": (
                "base_only"
                if args.stage == 1
                else "base_plus_previous_stage1_cumulative_adapter"
            ),
            "source_checkpoint": (
                previous_stage1["checkpoint"] if previous_stage1 else None
            ),
            "source_adapter_weights_sha256": (
                previous_stage1["adapter_weights_sha256"]
                if previous_stage1
                else None
            ),
        },
        "stage2_initialization": initialization,
        "same_task_stage1_to_stage2_confirmed": True,
        "stage2_checkpoint_serialization": stage2_serialization,
        "historical_gradient_current_task_only": True,
        "gradient_manifests": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "artifact": value["gradient_artifact"],
                "artifact_sha256": value["gradient_artifact_sha256"],
                "historical_prefix_length": value["historical_prefix_length"],
            }
            for path, value in zip(gradient_paths, gradient_values)
        ],
        "inference_stage2_checkpoints": inference_stack,
        "primary_inference": {
            "composition": "base_plus_stage2_adapters_tasks_1_to_t",
            "stage1_adapters_active": False,
            "oracle_task_id": False,
        },
        "historical_memory": run["historical_memory"],
        "parameter_accounting": {
            "trainable_parameters_current_stage": stage2["adapter_parameters"],
            "added_parameters_current_task": stage2["adapter_parameters"],
            "stage1_adapter_parameters": stage1["adapter_parameters"],
            "stage2_adapter_parameters": stage2["adapter_parameters"],
            "accumulated_stored_adapter_parameters": stored_parameters,
            "active_adapter_parameters_at_inference": active_parameters,
            "router_parameters": 0,
            "auxiliary_state_bytes": current_aux,
            "accumulated_auxiliary_state_bytes": previous_aux + current_aux,
            "checkpoint_bytes": checkpoint_bytes,
            "stage1_retained_for_provenance_not_primary_inference": True,
        },
        "runtime_accounting": {
            "stage1_train_seconds": stage1_seconds,
            "historical_gradient_collection_seconds": gradient_seconds,
            "stage2_train_seconds": stage2_seconds,
            "main_train_seconds": stage1_seconds + stage2_seconds,
            "auxiliary_seconds": gradient_seconds,
            "evaluation_seconds": 0.0,
            "total_stage_seconds": stage1_seconds
            + gradient_seconds
            + stage2_seconds,
            "training_gpu_hours": gpu_hours,
            "gpu_count_main_training": int(stage1_runtime["gpu_count"]),
            "gpu_models": gpu_models,
            "peak_gpu_memory_mib_observed": max(peak_values)
            if peak_values
            else None,
            "training_steps": steps,
            "number_of_unique_current_task_samples": task_samples,
            "formal_locked_train_records": formal_task_samples,
            "current_task_sample_exposures": task_samples + stage2_samples,
            "evaluation_reported_separately": True,
            "runtime_records": {
                "stage1": str(args.stage1_runtime.resolve()),
                "gradients": [
                    str(path.resolve()) for path in args.gradient_runtime
                ],
                "stage2": str(args.stage2_runtime.resolve())
                if args.stage2_runtime
                else None,
            },
        },
    }
    atomic_json(completion_path(output_root, args.stage), payload)
    print(json.dumps(payload, indent=2))
    return 0


def completion_ok(args) -> int:
    run = load_run(args.output_root)
    validate_completion(
        completion_path(args.output_root, args.stage),
        method=METHOD,
        stage=args.stage,
        immutable_config_sha256=run["immutable_config_sha256"],
    )
    print("PASS")
    return 0


def print_checkpoint(args) -> int:
    value = load_json(completion_path(args.output_root, args.stage))
    print(value[f"{args.role}_checkpoint"])
    return 0


def print_stack(args) -> int:
    values = stage2_inference_stack(
        completion_path(args.output_root, task)
        for task in range(1, args.stage + 1)
    )
    for value in values:
        print(value["checkpoint"])
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)

    item = sub.add_parser("init-run")
    item.add_argument("--data-root", type=Path, required=True)
    item.add_argument("--output-root", type=Path, required=True)
    item.add_argument("--lambda-1", type=float, default=0.01)
    item.add_argument("--lambda-2", type=float, default=0.01)
    item.add_argument("--gradient-samples", type=int, default=256)
    item.add_argument("--learning-rate", type=float, default=1e-4)
    item.add_argument("--dry-run", action="store_true")
    item.set_defaults(func=init_run)

    item = sub.add_parser("stage-ready")
    item.add_argument("--output-root", type=Path, required=True)
    item.add_argument("--stage", type=int, choices=range(1, 6), required=True)
    item.set_defaults(func=stage_ready)

    item = sub.add_parser("record-stage2-init")
    item.add_argument("--stage", type=int, choices=range(1, 6), required=True)
    item.add_argument("--stage1-checkpoint", type=Path, required=True)
    item.add_argument("--manifest", type=Path, required=True)
    item.set_defaults(func=record_stage2_init)

    item = sub.add_parser("copy-task1")
    item.add_argument("--stage", type=int, default=1)
    item.add_argument("--stage1-checkpoint", type=Path, required=True)
    item.add_argument("--stage2-checkpoint", type=Path, required=True)
    item.add_argument("--manifest", type=Path, required=True)
    item.set_defaults(func=copy_task1)

    item = sub.add_parser("validate-gradients")
    item.add_argument("--output-root", type=Path, required=True)
    item.add_argument("--data-root", type=Path, required=True)
    item.add_argument("--stage", type=int, choices=range(2, 6), required=True)
    item.add_argument("--manifest", action="append", type=Path, required=True)
    item.set_defaults(func=validate_gradients)

    item = sub.add_parser("finalize")
    item.add_argument("--output-root", type=Path, required=True)
    item.add_argument("--data-root", type=Path, required=True)
    item.add_argument("--stage", type=int, choices=range(1, 6), required=True)
    item.add_argument("--stage1-checkpoint", type=Path, required=True)
    item.add_argument("--stage2-checkpoint", type=Path, required=True)
    item.add_argument("--stage2-initialization", type=Path, required=True)
    item.add_argument("--stage1-runtime", type=Path, required=True)
    item.add_argument("--stage2-runtime", type=Path)
    item.add_argument("--gradient-manifest", action="append", type=Path, default=[])
    item.add_argument("--gradient-runtime", action="append", type=Path, default=[])
    item.set_defaults(func=finalize)

    item = sub.add_parser("completion-ok")
    item.add_argument("--output-root", type=Path, required=True)
    item.add_argument("--stage", type=int, choices=range(1, 6), required=True)
    item.set_defaults(func=completion_ok)

    item = sub.add_parser("print-checkpoint")
    item.add_argument("--output-root", type=Path, required=True)
    item.add_argument("--stage", type=int, choices=range(1, 6), required=True)
    item.add_argument("--role", choices=("stage1", "stage2"), required=True)
    item.set_defaults(func=print_checkpoint)

    item = sub.add_parser("print-stack")
    item.add_argument("--output-root", type=Path, required=True)
    item.add_argument("--stage", type=int, choices=range(1, 6), required=True)
    item.set_defaults(func=print_stack)
    return result


def main() -> int:
    args = parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
