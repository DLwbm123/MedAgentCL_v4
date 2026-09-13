#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.medicalskill_v1_2_baselines.baseline_harness import (
    HistoricalMemory, TASK_ORDER, atomic_json, build_run_manifest, inspect_adapter,
    load_json, sha256_file, trainer_state, validate_runtime_record, write_or_validate_run_manifest,
)
from scripts.medicalskill_v1_2_replay.replay_memory import (
    TRAINING_REPLAY, build_memory, concatenate_current_and_replay,
)

METHOD = "replay_lora_r48_balanced_total_v1"


def run_manifest(args: argparse.Namespace) -> dict:
    payload = build_run_manifest(
        method=METHOD,
        method_family="single_evolving_cumulative_lora_with_bounded_training_replay",
        data_root=args.data_root,
        output_root=args.output_root,
        capacity_policy={"kind": "fixed_single_adapter", "rank": 48, "lora_alpha": 96, "target_modules": ["q_proj", "v_proj"]},
        historical_memory=HistoricalMemory(
            raw_examples=True, features_or_prototypes=False, statistics_or_masks=False,
            historical_parameters=False,
            notes="Bounded historical raw examples replayed during sequential LoRA training; evolving model state is true; no formal privacy guarantee.",
        ),
        method_config={
            "replay_purpose": TRAINING_REPLAY,
            "total_replay_budget": args.replay_budget,
            "supported_budgets": [100, 500, 1000],
            "buffer_policy": "balanced_per_task_stratified_stable_sha256_v1",
            "buffer_timing": "pre-stage tasks < t; post-stage tasks <= t",
            "one_epoch": "one pass through full current task plus historical buffer",
            "oracle_task_id": False,
            "training_data_limit": args.train_limit,
            "evaluation_data_limit": args.eval_limit,
            "requested_stage_count": args.max_stages,
        },
    )
    payload["historical_memory"]["evolving_model_state"] = True
    return payload


def prepare_stage(args: argparse.Namespace) -> dict:
    stage_dir = args.output_root / "replay_lora" / f"stage_{args.stage:02d}"
    memory_dir = stage_dir / "memory_pre_train"
    previous = None
    if args.stage > 1:
        previous = args.output_root / "replay_lora" / f"stage_{args.stage - 1:02d}" / "memory_post_train" / "manifest.json"
        if not previous.is_file():
            raise FileNotFoundError(f"Previous post-stage buffer is missing: {previous}")
    manifest = build_memory(
        data_root=args.data_root, stage=args.stage, purpose=TRAINING_REPLAY,
        output_dir=memory_dir, total_budget=args.replay_budget, include_current=False,
        seed=args.seed, previous_manifest=previous,
    )
    current = args.data_root / TASK_ORDER[args.stage - 1] / "train.jsonl"
    mix_path = stage_dir / "training_mixture.jsonl"
    mixture = concatenate_current_and_replay(
        current, Path(manifest["materialized_dataset"]), mix_path, current_limit=args.train_limit
    )
    global_batch = args.global_batch
    mixture["estimated_optimizer_steps"] = math.ceil(mixture["total_effective_training_examples"] / global_batch)
    mixture["global_batch_size"] = global_batch
    mixture["current_dataset_sha256"] = sha256_file(current)
    mixture["pre_train_buffer_hash"] = manifest["resulting_buffer_hash"]
    atomic_json(stage_dir / "mixture_manifest.json", mixture)
    return {"status": "PASS", "dataset": str(mix_path), "memory": manifest, "mixture": mixture}


def update_buffer(args: argparse.Namespace) -> dict:
    previous = args.output_root / "replay_lora" / f"stage_{args.stage:02d}" / "memory_pre_train" / "manifest.json"
    return build_memory(
        data_root=args.data_root, stage=args.stage, purpose=TRAINING_REPLAY,
        output_dir=args.output_root / "replay_lora" / f"stage_{args.stage:02d}" / "memory_post_train",
        total_budget=args.replay_budget, include_current=True, seed=args.seed,
        previous_manifest=previous,
    )


def complete(args: argparse.Namespace) -> dict:
    manifest = load_json(args.output_root / "run_manifest.json")
    stage_dir = args.output_root / "replay_lora" / f"stage_{args.stage:02d}"
    adapter = inspect_adapter(args.checkpoint)
    if adapter["r"] != 48 or adapter["target_modules"] != ["q_proj", "v_proj"]:
        raise RuntimeError("Replay+LoRA checkpoint violates rank48 q_proj/v_proj contract")
    runtime = validate_runtime_record(args.runtime, "replay_lora_train")
    if runtime["status"] != "PASS":
        raise RuntimeError("Training runtime record is not PASS")
    mixture = load_json(stage_dir / "mixture_manifest.json")
    post = load_json(stage_dir / "memory_post_train" / "manifest.json")
    previous_checkpoint = None
    if args.stage > 1:
        previous_checkpoint = load_json(args.output_root / "replay_lora" / f"stage_{args.stage - 1:02d}" / "completion.json")["checkpoint"]
    payload = {
        "status": "PASS", "method": METHOD, "stage": args.stage,
        "immutable_config_sha256": manifest["immutable_config_sha256"],
        "checkpoint": str(args.checkpoint.resolve()), "stage2_checkpoint": str(args.checkpoint.resolve()),
        "initialization": "base_only" if args.stage == 1 else "base_plus_previous_cumulative_adapter",
        "previous_checkpoint": previous_checkpoint,
        "one_cumulative_adapter": True, "historical_adapter_stack": False, "router": False,
        "adapter": adapter,
        "inference_stage2_checkpoints": [{"task": args.stage, "checkpoint": adapter["checkpoint"], "adapter_weights_sha256": adapter["adapter_weights_sha256"]}],
        "parameter_accounting": {"trainable_adapter_parameters": adapter["adapter_parameters"], "stored_adapter_parameters": adapter["adapter_parameters"], "active_adapter_parameters": adapter["adapter_parameters"]},
        "runtime_accounting": {"main_train_seconds": runtime["wall_clock_seconds"], "auxiliary_seconds": 0.0, "training_gpu_hours": runtime["gpu_hours"], "optimizer_steps": runtime.get("training_steps")},
        "replay_accounting": {**mixture, "post_train_buffer_hash": post["resulting_buffer_hash"], "post_train_buffer_count": post["memory_count"]},
        **trainer_state(args.checkpoint),
    }
    atomic_json(stage_dir / "completion.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", type=Path, required=True)
    common.add_argument("--output-root", type=Path, required=True)
    common.add_argument("--replay-budget", type=int, choices=(100, 500, 1000), default=1000)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--train-limit", type=int, default=0)
    common.add_argument("--eval-limit", type=int, default=0)
    common.add_argument("--max-stages", type=int, default=5)
    check = sub.add_parser("check", parents=[common]); check.add_argument("--write", action="store_true")
    prep = sub.add_parser("prepare-stage", parents=[common]); prep.add_argument("--stage", type=int, required=True); prep.add_argument("--global-batch", type=int, default=16)
    update = sub.add_parser("update-buffer", parents=[common]); update.add_argument("--stage", type=int, required=True)
    done = sub.add_parser("complete", parents=[common]); done.add_argument("--stage", type=int, required=True); done.add_argument("--checkpoint", type=Path, required=True); done.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        payload = run_manifest(args)
        if args.write: write_or_validate_run_manifest(args.output_root / "run_manifest.json", payload)
    elif args.command == "prepare-stage": payload = prepare_stage(args)
    elif args.command == "update-buffer": payload = update_buffer(args)
    else: payload = complete(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
