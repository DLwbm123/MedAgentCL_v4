#!/usr/bin/env python3
"""Finalize the v1.1 candidate after the cap-8 gate blocked release/training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path("/root/MedAgentCL_v4")
DATA = Path("/remote-home/wangbomin/MedicalSkill-CL-v1.1")
ARTIFACTS = REPO / "artifacts/medicalskill_cl_v1_1_and_medprism_smoke"
TASKS = [
    "task_01_vqa",
    "task_02_diagnosis_classification",
    "task_03_concept_recognition",
    "task_04_visual_grounding",
    "task_05_reasoning_vqa",
]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def known_regression() -> dict:
    needle = "isic_0001131"
    refs = []
    for task in TASKS:
        for split in ("train", "test"):
            path = DATA / task / f"{split}.jsonl"
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if needle not in line.lower():
                        continue
                    row = json.loads(line)
                    refs.append(
                        {
                            "task": task,
                            "split": split,
                            "line": line_number,
                            "id": row.get("id"),
                            "images": [p for p in row.get("images", []) if needle in p.lower()],
                        }
                    )
    train_count = sum(ref["split"] == "train" for ref in refs)
    test_count = sum(ref["split"] == "test" for ref in refs)
    return {
        "status": "PASS" if train_count == 0 and test_count > 0 else "FAIL",
        "known_case": "ISIC_0001131",
        "canonical_token": "isic:isic_0001131",
        "ref_count": len(refs),
        "splits": sorted({ref["split"] for ref in refs}),
        "train_ref_count": train_count,
        "test_ref_count": test_count,
        "refs": refs,
    }


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    known = known_regression()
    write_json(ARTIFACTS / "known_leakage_regression_v2.json", known)

    task_hashes = {}
    for task in TASKS:
        task_hashes[task] = {
            split: sha256(DATA / task / f"{split}.jsonl") for split in ("train", "test")
        }

    manifest = {
        "status": "BLOCKED_NOT_FROZEN",
        "candidate": "MedicalSkill-CL-v1.1",
        "candidate_root": str(DATA),
        "release_manifest": False,
        "blocking_gate": "cap8_vs_cap12_supported_clinical_extra_ratio",
        "observed_ratio": 0.56,
        "threshold": 0.20,
        "concept_v3": "PASS",
        "strict_validation": "PASS",
        "phash_closure": "PASS",
        "known_leakage_regression": known["status"],
        "task_file_sha256": task_hashes,
        "note": "Candidate validation evidence only. Do not publish or train until the concept cap policy is revised and revalidated.",
    }
    write_json(ARTIFACTS / "medicalskill_cl_v1_1_manifest.json", manifest)
    write_json(DATA / "audits/strict_validation_v1_1.json", manifest)

    resources = {
        "status": "ESTIMATE_ONLY_NOT_RUN",
        "full_one_epoch_total_optimizer_steps": 233981,
        "full_one_epoch_gpu_hours_a100_40gb_range": [200, 500],
        "single_a100_40gb_wall_days_range": [8.3, 20.8],
        "estimated_peak_gpu_memory_gib_range": [28, 40],
        "estimated_additional_storage_gib_range": [10, 30],
        "five_stage_short_smoke_gpu_hours_range": [0.5, 1.5],
        "basis": "Planning estimate; measure throughput in a gated short smoke before committing to the full run.",
    }
    write_json(ARTIFACTS / "estimated_full_run_resources.json", resources)

    readiness = {
        "status": "BLOCKED",
        "ready": False,
        "dataset_formally_frozen": False,
        "concept_v3_closure": "PASS",
        "cap8_sufficiency_gate": "FAIL",
        "cap8_supported_clinical_extra_ratio": 0.56,
        "cap8_gate_threshold": 0.20,
        "phash_closure": "PASS",
        "known_leakage_regression": known["status"],
        "candidate_strict_validation": "PASS",
        "med_prism_five_stage_smoke": "NOT_RUN_BY_GATE",
        "formal_one_epoch_experiment": "DO_NOT_RUN",
        "blocking_reasons": [
            "The cap-12 pilot found supported clinically useful extra concepts in 56.0% of sampled pre-v3 cap-8 rows (threshold 20.0%).",
            "The specification requires stopping dataset freeze and all downstream Med-PRISM training when this gate fails.",
        ],
    }
    write_json(ARTIFACTS / "formal_experiment_readiness.json", readiness)

    commands = """BLOCKED - DO NOT RUN THESE COMMANDS YET.

The fixed seed-42 formal one-epoch commands are intentionally withheld because
the cap-8 sufficiency gate failed (56.0% > 20.0%). Revise the concept cap policy,
rebuild task_03, and repeat independent closure before generating runnable commands.
"""
    (ARTIFACTS / "formal_one_epoch_commands.txt").write_text(commands, encoding="utf-8")

    report = """# Real Med-PRISM five-stage smoke

Status: **NOT RUN (required gate stop)**

The implementation path targets the repository's real shared/private Med-PRISM
tuner, composition, optimizer ownership, orthogonal-gradient audit, checkpoint
reload, and task-aware/task-free evaluation. No ordinary-LoRA substitute and no
Med-PRISM training were run because the cap-8 sufficiency pilot failed its release
gate. Consequently no empirical transition, optimizer, hash, reload, gradient, or
continual-learning matrices are claimed.
"""
    (ARTIFACTS / "med_prism_real_5stage_smoke_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "BLOCKED_NOT_FROZEN", "known": known, "files": 5}))


if __name__ == "__main__":
    main()
