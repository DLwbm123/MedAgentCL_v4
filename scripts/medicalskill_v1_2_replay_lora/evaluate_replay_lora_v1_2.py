#!/usr/bin/env python3
"""Replay+LoRA uses the verified single-cumulative-LoRA evaluator unchanged."""
import argparse
import json
from pathlib import Path
from scripts.medicalskill_v1_2_sequential import evaluate_sequential_v1_2 as evaluator

evaluator.METHOD = "replay_lora_r48_balanced_total_v1"

def verify_stage_owned_checkpoint() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args, _ = parser.parse_known_args()
    completion = args.output_root.resolve() / "replay_lora" / f"stage_{args.stage:02d}" / "completion.json"
    value = json.loads(completion.read_text(encoding="utf-8"))
    if Path(value["checkpoint"]).resolve() != args.adapter.resolve():
        raise RuntimeError("Evaluation row t must load cumulative checkpoint_t")

if __name__ == "__main__":
    verify_stage_owned_checkpoint()
    raise SystemExit(evaluator.main())
