#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


MODES = ("task1_skill_aware", "task2_skill_aware", "cumulative")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def difference(left: torch.Tensor, right: torch.Tensor) -> dict:
    delta = (left - right).abs()
    return {
        "max_abs_logit_diff": float(delta.max()),
        "mean_abs_logit_diff": float(delta.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    args.artifact_root.mkdir(parents=True, exist_ok=True)

    reloads = {}
    samples = []
    for mode in MODES:
        reference = read(args.run_dir / f"{mode}_reference.json")
        reload = read(args.run_dir / f"{mode}_reload.json")
        compare = difference(
            torch.load(args.run_dir / f"{mode}_reference.pt", weights_only=True),
            torch.load(args.run_dir / f"{mode}_reload.pt", weights_only=True),
        )
        compare.update(
            {
                "reference_logits_sha256": reference["logits_sha256"],
                "reload_logits_sha256": reload["logits_sha256"],
                "reference_text": reference["generated_text"],
                "reload_text": reload["generated_text"],
                "active_component_manifest": reference["active_component_manifest"],
                "pass": compare["max_abs_logit_diff"] <= 1e-5,
            }
        )
        reloads[mode] = compare
        samples.append(
            {
                "mode": mode,
                "generated_text": reference["generated_text"],
                "active_component_manifest": reference["active_component_manifest"],
                "input_token_ids_sha256": reference["input_token_ids_sha256"],
                "image_tensor_sha256": reference["image_tensor_sha256"],
                "logits_sha256": reference["logits_sha256"],
            }
        )
    order = difference(
        torch.load(args.run_dir / "cumulative_reference.pt", weights_only=True),
        torch.load(args.run_dir / "cumulative_order_21.pt", weights_only=True),
    )
    order["pass"] = order["max_abs_logit_diff"] <= 1e-5
    negative = read(args.run_dir / "cumulative_reference.json")["negative_tests"]
    manifests = [reloads[mode]["active_component_manifest"] for mode in MODES]
    composition_checks = {
        "all_modes_shared_loaded_once": all(
            item["shared_load_count"] == 1 for item in manifests
        ),
        "task1_private_only": manifests[0]["private_task_ids"] == [1],
        "task2_private_only": manifests[1]["private_task_ids"] == [2],
        "cumulative_private_1_2": manifests[2]["private_task_ids"] == [1, 2],
        "all_unmerged": all(item["merged"] is False for item in manifests),
        "private_order_equivalent": order["pass"],
        "fail_fast_cases_pass": negative["status"] == "PASS",
    }
    reload_payload = {
        "status": (
            "PASS" if all(item["pass"] for item in reloads.values()) else "BLOCKED"
        ),
        "model_revision": read(args.run_dir / "cumulative_reference.json")[
            "model_revision"
        ],
        "modes": reloads,
    }
    composition_payload = {
        "status": "PASS" if all(composition_checks.values()) else "BLOCKED",
        "checks": composition_checks,
        "private_order_comparison": order,
        "negative_tests": negative,
        "active_component_manifests": {
            mode: reloads[mode]["active_component_manifest"] for mode in MODES
        },
    }
    (args.artifact_root / "reload_comparison.json").write_text(
        json.dumps(reload_payload, indent=2, ensure_ascii=True) + "\n"
    )
    (args.artifact_root / "composition_audit.json").write_text(
        json.dumps(composition_payload, indent=2, ensure_ascii=True) + "\n"
    )
    with (args.artifact_root / "inference_samples.jsonl").open("w") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
    print(
        json.dumps(
            {
                "reload_status": reload_payload["status"],
                "composition_status": composition_payload["status"],
                "order_max_abs_diff": order["max_abs_logit_diff"],
            }
        )
    )
    if reload_payload["status"] != "PASS" or composition_payload["status"] != "PASS":
        raise SystemExit("Phase 5 composition checks failed")


if __name__ == "__main__":
    main()
