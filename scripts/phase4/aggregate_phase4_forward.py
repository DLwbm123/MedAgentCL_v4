#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def delta(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    difference = (a.float() - b.float()).abs()
    return float(difference.max()), float(difference.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = [
        "base",
        "task1_reference",
        "task1_reload_1",
        "task1_reload_2",
        "task2_reference",
        "task2_reload_1",
        "task2_reload_2",
    ]
    metadata = {
        label: read_json(args.run_dir / f"{label}.json")
        for label in labels
    }
    logits = {
        label: torch.load(
            args.run_dir / f"{label}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for label in labels
    }
    input_hashes = {item["input_token_ids_sha256"] for item in metadata.values()}
    image_hashes = {item["image_tensor_sha256"] for item in metadata.values()}
    base_identities = {item["base_identity"] for item in metadata.values()}
    task_results = {}
    for task_id in (1, 2):
        prefix = f"task{task_id}"
        base_max, base_mean = delta(logits["base"], logits[f"{prefix}_reference"])
        presave_max, presave_mean = delta(
            logits[f"{prefix}_reference"],
            logits[f"{prefix}_reload_1"],
        )
        reload_max, reload_mean = delta(
            logits[f"{prefix}_reload_1"],
            logits[f"{prefix}_reload_2"],
        )
        reference = metadata[f"{prefix}_reference"]
        task_results[str(task_id)] = {
            "base_vs_adapter_max_abs_logit_diff": base_max,
            "base_vs_adapter_mean_abs_logit_diff": base_mean,
            "pre_save_vs_reload_1_max_abs_logit_diff": presave_max,
            "pre_save_vs_reload_1_mean_abs_logit_diff": presave_mean,
            "reload_1_vs_reload_2_max_abs_logit_diff": reload_max,
            "reload_1_vs_reload_2_mean_abs_logit_diff": reload_mean,
            "adapter_weights_sha256": reference["adapter_weights_sha256"],
            "active_task_ids": reference["active_task_ids"],
            "task_scaling": reference["task_scaling"],
            "reference_semantics": (
                "original checkpoint inference in a fresh process; "
                "accepted as the pre-save/original-checkpoint reference"
            ),
            "generated_text": {
                name: metadata[f"{prefix}_{name}"]["generated_text"]
                for name in ("reference", "reload_1", "reload_2")
            },
            "pass": (
                base_max > 1e-6
                and presave_max <= 1e-5
                and reload_max <= 1e-5
            ),
        }
    payload = {
        "status": "PASS",
        "model_revision": metadata["base"]["model_revision"],
        "shared_model_base_identity": {
            "consistent": len(base_identities) == 1,
            "identities": sorted(base_identities),
            "snapshot": metadata["base"]["snapshot"],
        },
        "input_token_ids": metadata["base"]["input_token_ids"],
        "input_token_ids_hash": metadata["base"]["input_token_ids_sha256"],
        "image_path": metadata["base"]["image_path"],
        "image_tensor_hash": metadata["base"]["image_tensor_sha256"],
        "image_tensor_shape": metadata["base"]["image_tensor_shape"],
        "base_vs_adapter_max_abs_logit_diff": {
            task: result["base_vs_adapter_max_abs_logit_diff"]
            for task, result in task_results.items()
        },
        "pre_save_vs_reload_1_max_abs_logit_diff": {
            task: result["pre_save_vs_reload_1_max_abs_logit_diff"]
            for task, result in task_results.items()
        },
        "reload_1_vs_reload_2_max_abs_logit_diff": {
            task: result["reload_1_vs_reload_2_max_abs_logit_diff"]
            for task, result in task_results.items()
        },
        "mean_abs_logit_diff": {
            task: {
                "base_vs_adapter": result["base_vs_adapter_mean_abs_logit_diff"],
                "pre_save_vs_reload_1": result["pre_save_vs_reload_1_mean_abs_logit_diff"],
                "reload_1_vs_reload_2": result["reload_1_vs_reload_2_mean_abs_logit_diff"],
            }
            for task, result in task_results.items()
        },
        "tasks": task_results,
        "run_metadata": {
            label: str(args.run_dir / f"{label}.json") for label in labels
        },
    }
    if (
        len(input_hashes) != 1
        or len(image_hashes) != 1
        or len(base_identities) != 1
        or not all(result["pass"] for result in task_results.values())
    ):
        payload["status"] = "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    if payload["status"] != "PASS":
        raise SystemExit("Phase 4 forward/reload comparison failed")
    print(json.dumps({"status": "PASS", "tasks": task_results}, indent=2))


if __name__ == "__main__":
    main()
