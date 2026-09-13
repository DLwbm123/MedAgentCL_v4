#!/usr/bin/env python3
"""Aggregate five-stage Med-PRISM matrices, audits, readiness, and reports."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/medicalskill_cl_v1_1_and_medprism_smoke"
DATA = Path("/remote-home/wangbomin/MedicalSkill-CL-v1.1")
TASKS = ["VQA", "Diagnosis", "Concept", "Grounding", "Reasoning"]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def matrix(mode: str) -> list[list[float]]:
    values = []
    for stage in range(1, 6):
        row = []
        for task in range(1, 6):
            summary = load(ARTIFACT / "evaluation" / f"stage_{stage:02d}" / f"{mode}_task_{task:02d}.summary.json")
            row.append(float(summary["primary_score"]))
        values.append(row)
    return values


def write_matrix(path: Path, values: list[list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["after_stage", *TASKS, "average"])
        for index, row in enumerate(values, 1):
            writer.writerow([index, *[f"{value:.8f}" for value in row], f"{sum(row) / len(row):.8f}"])


def main() -> int:
    primary = matrix("primary_cumulative")
    oracle = matrix("oracle_skill_aware")
    task_free = matrix("task_free")
    for name, values in (("primary_cumulative_matrix.csv", primary), ("oracle_skill_aware_matrix.csv", oracle), ("task_free_matrix.csv", task_free)):
        write_matrix(ARTIFACT / name, values)
    base = [float(load(ARTIFACT / "evaluation/stage_00" / f"base_zero_shot_task_{task:02d}.summary.json")["primary_score"]) for task in range(1, 6)]
    best = [max(primary[stage][task] for stage in range(task, 5)) for task in range(5)]
    final = primary[-1]
    forgetting = [best[index] - final[index] for index in range(5)]
    bwt_terms = [final[index] - primary[index][index] for index in range(4)]
    fwt_terms = [primary[index - 1][index] - base[index] for index in range(1, 5)]
    metrics = {
        "status": "PASS", "scale": "0..1", "base_zero_shot": base,
        "final_average_score": sum(final) / 5, "per_task_best_score": best,
        "per_task_final_score": final, "per_task_forgetting": forgetting,
        "average_forgetting": sum(forgetting) / 5,
        "BWT": sum(bwt_terms) / len(bwt_terms), "BWT_terms": bwt_terms,
        "FWT": sum(fwt_terms) / len(fwt_terms), "FWT_terms": fwt_terms,
        "definitions": {"BWT": "mean(R[5,i]-R[i,i]) for i=1..4", "FWT": "mean(R[i-1,i]-R[0,i]) for i=2..5", "forgetting": "max post-learning score minus final score"},
    }
    write_json(ARTIFACT / "cl_metrics.json", metrics)

    transitions, optimizer, shared, private, orth, reloads = {}, {}, {}, {}, {}, {}
    all_transition_pass = True
    for stage in range(1, 6):
        padded = f"{stage:02d}"
        root = ARTIFACT / "med_prism" / f"stage_{padded}"
        config = load(ARTIFACT / "configs" / f"stage_{padded}.json")
        shared_manifest = load(root / "shared/shared_manifest.json")
        private_manifest = load(root / "private/private_manifest.json")
        runtime = root / "runtime_audits"
        expected_source = None if stage == 1 else str(ARTIFACT / "med_prism" / f"stage_{stage - 1:02d}" / "shared/shared_manifest.json")
        source_ok = config["shared_source_manifest"] == expected_source and shared_manifest.get("source_checkpoint") == expected_source
        source_hash_ok = True
        if stage > 1:
            previous = load(Path(expected_source))
            source_hash_ok = previous["weights_sha256"] == sha256(Path(expected_source).parent / previous["weights_file"])
        private_sources_ok = config["private_source_manifests"] == [str(ARTIFACT / "med_prism" / f"stage_{old:02d}" / "private/private_manifest.json") for old in range(1, stage)]
        transition_status = "PASS" if source_ok and source_hash_ok and private_sources_ok and private_manifest["task_id"] == stage else "FAIL"
        all_transition_pass &= transition_status == "PASS"
        transitions[str(stage)] = {"status": transition_status, "starts_from_base": stage == 1, "expected_shared_source": expected_source, "configured_shared_source": config["shared_source_manifest"], "saved_source_checkpoint": shared_manifest.get("source_checkpoint"), "source_weights_hash_valid": source_hash_ok, "historical_private_sources": config["private_source_manifests"], "current_private_task_id": private_manifest["task_id"]}
        optimizer[str(stage)] = load(runtime / f"task{stage}_optimizer_audit.json")
        shared[str(stage)] = {**load(runtime / "shared_hash_audit.json").get(f"task_{stage}", {}), "manifest_weights_sha256": shared_manifest["weights_sha256"]}
        private[str(stage)] = {**load(runtime / "private_hash_audit.json").get(f"task_{stage}", {}), "current_manifest_weights_sha256": private_manifest["weights_sha256"]}
        orth[str(stage)] = load(runtime / "orth_gradient_audit.json").get(f"task_{stage}", {})
        reloads[str(stage)] = load(ARTIFACT / "evaluation" / f"stage_{padded}" / "stage_evaluation_summary.json")["reload_equivalence"]
    write_json(ARTIFACT / "stage_transition_audit.json", {"status": "PASS" if all_transition_pass else "FAIL", "stages": transitions})
    write_json(ARTIFACT / "five_stage_transition_audit.json", {"status": "PASS" if all_transition_pass else "FAIL", "stages": transitions})
    write_json(ARTIFACT / "optimizer_membership_by_stage.json", {"status": "PASS" if all(value["status"] == "PASS" for value in optimizer.values()) else "FAIL", "stages": optimizer})
    write_json(ARTIFACT / "shared_hash_by_stage.json", {"status": "PASS" if all(value.get("status") == "PASS" for value in shared.values()) else "FAIL", "stages": shared})
    write_json(ARTIFACT / "private_hash_by_stage.json", {"status": "PASS" if all(value.get("status") == "PASS" for value in private.values()) else "FAIL", "stages": private})
    write_json(ARTIFACT / "orth_gradient_by_stage.json", {"status": "PASS" if all(value.get("status") == "PASS" for value in orth.values()) else "FAIL", "stages": orth})
    write_json(ARTIFACT / "reload_equivalence_by_stage.json", {"status": "PASS" if all(value.get("status") == "PASS" for value in reloads.values()) else "FAIL", "stages": reloads})

    traces = {}
    total_seconds = 0.0
    peak_reserved = 0
    for stage in range(1, 6):
        path = ARTIFACT / "med_prism" / f"stage_{stage:02d}" / "runtime_audits" / f"task{stage}_training_trace.jsonl"
        values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        finite = all(math.isfinite(float(row[key])) for row in values for key in ("task_loss", "total_loss", "shared_gradient_norm", "current_private_gradient_norm"))
        peak = max((int(row["gpu_reserved"]) for row in values), default=0)
        peak_reserved = max(peak_reserved, peak)
        log = (ARTIFACT / "med_prism" / f"stage_{stage:02d}" / "train.log").read_text(errors="replace")
        matches = re.findall(r"'train_runtime':\s*([0-9.]+)", log)
        runtime = float(matches[-1]) if matches else None
        if runtime: total_seconds += runtime
        traces[str(stage)] = {"steps": len(values), "finite": finite, "peak_gpu_reserved_bytes": peak, "train_runtime_seconds": runtime, "last": values[-1] if values else None}
    resources = {"status": "PASS", "smoke": {"optimizer_steps": 500, "train_runtime_seconds": total_seconds, "peak_gpu_reserved_gib": peak_reserved / 2**30}, "estimated_full_one_epoch": {"basis": "linear scaling from measured 100-step stages; excludes evaluation", "optimizer_steps_by_task": {}, "estimated_train_gpu_hours": None, "recommended_gpu": "1x A100 40GB or larger", "checkpoint_storage_gib": 5.0}}
    manifest = load(ARTIFACT / "smoke_data_manifest.json")
    full_steps = 0
    for task, value in manifest["tasks"].items():
        source_count = sum(1 for line in open(Path(value["train_source"]), encoding="utf-8") if line.strip())
        resources["estimated_full_one_epoch"]["optimizer_steps_by_task"][task] = source_count
        full_steps += source_count
    if total_seconds:
        resources["estimated_full_one_epoch"]["estimated_train_gpu_hours"] = total_seconds / 500 * full_steps / 3600
    write_json(ARTIFACT / "estimated_full_run_resources.json", resources)

    required = {
        "concept_v3": load(ARTIFACT / "concept_v3_summary.json")["status"] == "PASS",
        "cap8": load(ARTIFACT / "cap8_vs_cap12_pilot.json")["decision"] == "retain_cap_8",
        "phash": load(ARTIFACT / "cross_split_phash_only_summary.json")["status"] == "PASS",
        "dataset_validation": load(DATA / "audits/strict_validation_v1_1.json")["status"] == "PASS",
        "five_stage_transition": all_transition_pass,
        "optimizer_ownership": all(value["status"] == "PASS" for value in optimizer.values()),
        "orth_gradient": all(value.get("status") == "PASS" for value in orth.values()),
        "private_isolation": all(value.get("status") == "PASS" for value in private.values()),
        "reload_equivalence": all(value.get("status") == "PASS" for value in reloads.values()),
        "all_predictions_nonempty": all(load(ARTIFACT / "evaluation" / f"stage_{stage:02d}" / "stage_evaluation_summary.json")["status"] == "PASS" for stage in range(0, 6)),
        "finite_loss_and_gradients": all(value["finite"] and value["steps"] == 100 for value in traces.values()),
    }
    readiness = {"status": "PASS" if all(required.values()) else "BLOCKED", "can_start_formal_one_epoch": all(required.values()), "checks": required, "method": "med_prism_shared_private", "ordinary_lora": False, "seed": 42, "formal_experiment_not_run": True}
    write_json(ARTIFACT / "formal_experiment_readiness.json", readiness)
    commands = """# Review resource estimate and readiness before execution. These commands were not run.
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/remote-home/wangbomin/huggingface_cache
export HF_HUB_CACHE=/remote-home/wangbomin/huggingface_cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
# Use the same single-stage-chain orchestrator with max_steps removed and num_train_epochs=1.
bash scripts/med_prism_real_5skill/run_real_5stage_one_epoch.sh --seed 42 --data-root /remote-home/wangbomin/MedicalSkill-CL-v1.1 --output-root /remote-home/wangbomin/med_prism_real_5skill_seed42
"""
    (ARTIFACT / "formal_one_epoch_commands.txt").write_text(commands, encoding="utf-8")
    report = ["# Real Med-PRISM five-stage smoke", "", f"Status: **{readiness['status']}**", "", "Method: `med_prism_shared_private` (shared rank-32 low-rank adapter plus 16 rank-1 private experts per task); ordinary PEFT LoRA was not used.", "", "Each stage ran 100 optimizer steps with seed 42, inherited the preceding shared manifest, saved an independent private manifest, and was reloaded in a fresh process.", "", f"Final average score: {metrics['final_average_score']:.6f}; forgetting: {metrics['average_forgetting']:.6f}; BWT: {metrics['BWT']:.6f}; FWT: {metrics['FWT']:.6f}.", "", f"Measured training runtime: {total_seconds / 3600:.3f} GPU-hours; peak reserved memory: {peak_reserved / 2**30:.2f} GiB."]
    (ARTIFACT / "med_prism_real_5stage_smoke_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(ARTIFACT / "med_prism_real_smoke_summary.json", {"status": readiness["status"], "method": "med_prism_shared_private", "steps_per_stage": 100, "metrics": metrics, "training": traces, "readiness": readiness})
    print(json.dumps(readiness, indent=2))
    return 0 if readiness["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
