#!/usr/bin/env python3
"""Build immutable Med-PRISM v1.1/v1.2 run contracts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from med_prism.config import SharedPrivateConfig

HERE = Path(__file__).resolve().parent
REPO = Path("/root/MedAgentCL_v4")
LEGACY_SCRIPTS = REPO / "scripts/medicalskill_v1_2_med_prism"
sys.path.insert(0, str(LEGACY_SCRIPTS))
from prepare_formal_v1_2 import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    TASKS,
    assert_contract,
    sha256,
)

DEFAULT_DATA = Path("/remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k")
ABLATIONS = {
    "no_geo_orth": "medicalskill_v1_2_med_prism_ablation_no_geo_orth",
    "no_shared_drift": "medicalskill_v1_2_med_prism_ablation_no_shared_drift",
    "rank16_private": "medicalskill_v1_2_med_prism_ablation_rank16_private",
}
SOURCE_FILES = (
    "med_prism/config.py",
    "med_prism/adapters/shared_private.py",
    "med_prism/projection/orth_losses.py",
    "med_prism/projection/shared_private.py",
    "med_prism/optimization/shared_private.py",
    "med_prism/swift_plugins/shared_private_trainer.py",
    "med_prism/swift_plugins/med_prism_shared_private_plugin.py",
    "med_prism/checkpoint/shared_private_checkpoint.py",
    "med_prism/evaluation/cl.py",
    "scripts/medicalskill_v1_2_med_prism/evaluate_formal_v1_2_cell.py",
    "scripts/medicalskill_v1_2_med_prism/evaluation_contract_v1_2.py",
    "scripts/medicalskill_v1_2_med_prism/aggregate_formal_v1_2.py",
    "scripts/medicalskill_v1_2_med_prism_versions/prepare_versioned.py",
    "scripts/medicalskill_v1_2_med_prism_versions/run_versioned_train.sh",
    "scripts/medicalskill_v1_2_med_prism_versions/run_pilot_evaluation.sh",
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def hash_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def git_provenance(script_dir: Path) -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    ).splitlines()
    source_hashes = {}
    for relative in SOURCE_FILES:
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        source_hashes[relative] = sha256(path)
    return {
        "git_head": head,
        "git_dirty": bool(dirty),
        "git_status_porcelain": dirty,
        "git_status_sha256": hashlib.sha256(
            ("\n".join(dirty) + "\n").encode()
        ).hexdigest(),
        "source_hashes": source_hashes,
        "script_directory": str(script_dir),
        "script_directory_sha256": hash_directory(script_dir),
    }


def refresh_failed_run_manifest(
    run_manifest: Path,
    previous: dict[str, Any],
    manifest: dict[str, Any],
) -> Path:
    """Archive and refresh provenance only when no stage has completed."""
    completed = []
    for completion in sorted(
        run_manifest.parent.glob("med_prism/stage_*/completion.json")
    ):
        payload = json.loads(completion.read_text(encoding="utf-8"))
        if payload.get("status") == "PASS":
            completed.append(str(completion))
    if completed:
        raise RuntimeError(
            "Refusing to change source provenance after completed stages: "
            f"{completed}"
        )
    history = run_manifest.parent / "run_manifest_history"
    archive = history / f"run_manifest_{sha256(run_manifest)[:16]}.json"
    if archive.is_file():
        if json.loads(archive.read_text(encoding="utf-8")) != previous:
            raise RuntimeError(f"Run-manifest history collision: {archive}")
    else:
        atomic_json(archive, previous)
    atomic_json(run_manifest, manifest)
    return archive


def pilot_contract(data: Path) -> dict[str, Any]:
    manifest_path = data / "pilot_selection_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("seed") != 42:
        raise RuntimeError("Pilot selection manifest is not locked/PASS")
    tasks = {}
    for task_id, task_name in TASKS.items():
        train = data / task_name / "train.jsonl"
        diagnostic = data / task_name / "test.jsonl"
        record = manifest["tasks"][str(task_id)]
        if sha256(train) != record["pilot_train_sha256"]:
            raise RuntimeError(f"Pilot train hash mismatch: {train}")
        if sha256(diagnostic) != record["pilot_diagnostic_sha256"]:
            raise RuntimeError(f"Pilot diagnostic hash mismatch: {diagnostic}")
        tasks[str(task_id)] = {
            "task_name": task_name,
            "train": str(train),
            "test": str(diagnostic),
            "train_count": record["pilot_train_count"],
            "test_count": record["pilot_diagnostic_count"],
            "train_sha256": record["pilot_train_sha256"],
            "test_sha256": record["pilot_diagnostic_sha256"],
        }
    return {
        "dataset_root": str(data),
        "dataset_version": "MedicalSkill-CL-v1.2-train-derived-pilot",
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": sha256(manifest_path),
        "dataset_lock": {"pilot": True, "seed": 42},
        "tasks": tasks,
    }


def method_fields(
    version: str,
    *,
    shared_lr: float,
    private_lr: float,
    drift: float,
    key: float,
    ablation_name: str | None = None,
) -> dict[str, Any]:
    if version == "1.1":
        return {
            "method": "Med-PRISM",
            "method_version": "1.1",
            "method_variant": "shared_fix",
            "shared_optimizer_mode": "separate_lr_param_groups",
            "shared_lr": shared_lr,
            "private_lr": private_lr,
            "shared_gradient_hook_enabled": False,
            "shared_drift_mode": "effective_BA",
            "shared_drift_lambda": drift,
            "key_isolation_enabled": False,
            "key_loss_mode": "disabled",
            "key_lambda": 0.0,
            "key_history_scope": "none",
        }
    if version == "1.2":
        rank16_private = ablation_name == "rank16_private"
        return {
            "method": "Med-PRISM",
            "method_version": "1.2",
            "method_variant": (
                "shared_fix_plus_standard_rank16_task_private"
                if rank16_private
                else "shared_fix_plus_key_isolation"
            ),
            "ablation_name": ablation_name,
            "shared_optimizer_mode": "separate_lr_param_groups",
            "shared_lr": shared_lr,
            "private_lr": private_lr,
            "shared_gradient_hook_enabled": False,
            "shared_drift_mode": "effective_BA",
            "shared_drift_lambda": drift,
            "key_isolation_enabled": not rank16_private,
            "key_loss_mode": "disabled" if rank16_private else "rms_cosine_A",
            "key_lambda": key,
            "key_history_scope": (
                "none" if rank16_private else "all_previous_tasks_same_layer"
            ),
        }
    raise ValueError("Method version must be 1.1 or 1.2")


def manifest_method_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Add publication-facing lambda names without changing runtime config keys."""
    return {
        **fields,
        "lambda_shared_drift": fields["shared_drift_lambda"],
        "lambda_key": fields["key_lambda"],
    }


def build_contract(args) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    if args.seed != 42:
        raise ValueError("The controlled comparison locks seed=42")
    if args.ablation_name and args.method_version != "1.2":
        raise ValueError("Ablations are defined only against Med-PRISM v1.2")
    expected = (
        (0.1, 0.01, 0.0)
        if args.method_version == "1.1"
        else {
            None: (0.1, 0.01, 0.1),
            "no_geo_orth": (0.0, 0.01, 0.1),
            "no_shared_drift": (0.1, 0.0, 0.1),
            "rank16_private": (0.0, 0.01, 0.0),
        }[args.ablation_name]
    )
    actual = (args.orth_lambda, args.lambda_drift, args.lambda_key)
    if any(abs(left - right) > 1e-12 for left, right in zip(actual, expected)):
        raise ValueError(
            f"Ablation {args.ablation_name!r} locks "
            f"(lambda_geo_orth, lambda_shared_drift, lambda_key)={expected}"
        )
    if args.shared_lr <= 0 or args.private_lr <= 0:
        raise ValueError("Learning rates must be positive")
    if abs(args.shared_lr / args.private_lr - 0.1) > 1e-12:
        raise ValueError("Shared/private LR ratio must be 0.1")
    fields = method_fields(
        args.method_version,
        shared_lr=args.shared_lr,
        private_lr=args.private_lr,
        drift=args.lambda_drift,
        key=args.lambda_key,
        ablation_name=args.ablation_name,
    )
    manifest_fields = manifest_method_fields(fields)
    data = pilot_contract(args.data_root) if args.pilot_mode else assert_contract(args.data_root)
    script_name = ABLATIONS.get(
        args.ablation_name,
        f"medicalskill_v1_2_med_prism_v{args.method_version}",
    )
    script_dir = REPO / "scripts" / script_name
    provenance = git_provenance(script_dir)
    rank16_private = args.ablation_name == "rank16_private"
    private_adapter_type = (
        "standard_rank16_lora" if rank16_private else "rank1_expert_bank"
    )
    private_experts_per_task = 1 if rank16_private else 16
    configs = {}
    for stage in range(1, 6):
        root = args.output_root / "med_prism" / f"stage_{stage:02d}"
        configs[stage] = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "current_task_id": stage,
            "shared_rank": 32,
            "shared_alpha": 32.0,
            "shared_lr_scale": 0.1,
            "private_experts_per_task": private_experts_per_task,
            "private_rank": 16,
            "private_adapter_type": private_adapter_type,
            "private_alpha": 16.0,
            "dropout": 0.05,
            "orth_loss_type": "rms",
            "orth_lambda": args.orth_lambda,
            "orth_eps": 1e-8,
            **fields,
            "trace_every_optimizer_steps": 10,
            "geometry_every_optimizer_steps": 50,
            "component_gradient_diagnostics_interval": args.gradient_diagnostics_interval,
            "seed": args.seed,
            "shared_source_manifest": None if stage == 1 else str(args.output_root / "med_prism" / f"stage_{stage - 1:02d}" / "shared/shared_manifest.json"),
            "private_source_manifests": [str(args.output_root / "med_prism" / f"stage_{old:02d}" / "private/private_manifest.json") for old in range(1, stage)],
            "shared_output_dir": str(root / "shared"),
            "private_output_dir": str(root / "private"),
            "dataset_manifest": data["dataset_manifest"],
            "dataset_sha256": data["tasks"][str(stage)]["train_sha256"],
            "output_dir": str(root),
            "artifact_root": str(root / "runtime_audits"),
        }
        SharedPrivateConfig.from_dict(configs[stage])
    immutable = {
        **manifest_fields,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "seed": 42,
        "dataset_manifest_sha256": data["dataset_manifest_sha256"],
        "shared_rank": 32,
        "shared_alpha": 32.0,
        "private_rank1_experts_per_task": 0 if rank16_private else 16,
        "private_adapter_type": private_adapter_type,
        "private_rank": 16,
        "private_units_per_task": private_experts_per_task,
        "private_alpha": 16.0,
        "dropout": 0.05,
        "orth_loss_type": "rms",
        "orth_lambda": args.orth_lambda,
        "orth_epsilon": 1e-8,
        "task1_effective_shared_drift": "exact_differentiable_zero",
        "primary_composition": "base_plus_latest_shared_once_plus_all_private_banks",
        "router": False,
        "replay": False,
        "stage0_default": "disabled_reuse_compatible_cache",
        "oracle_default": "disabled_diagnostic_only",
        "ablation_name": args.ablation_name,
        "enabled_components": {
            "geometric_rank1_orthogonality": args.orth_lambda > 0,
            "effective_BA_shared_drift": args.lambda_drift > 0,
            "rank1_key_isolation": args.lambda_key > 0,
            "explicit_rank1_private_experts": not rank16_private,
            "conventional_rank16_task_private_lora": rank16_private,
        },
    }
    immutable_sha256 = hashlib.sha256(
        json.dumps(immutable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    evaluator_contract = (
        REPO
        / "scripts/medicalskill_v1_2_med_prism/evaluation_contract_v1_2.py"
    )
    manifest = {
        "status": "PASS",
        **manifest_fields,
        "ordinary_lora": False,
        "private_branch_ordinary_lora": rank16_private,
        "ablation_name": args.ablation_name,
        "pilot_mode": args.pilot_mode,
        "data": data,
        "output_root": str(args.output_root),
        "stage_order": list(TASKS.values()),
        "immutable_config": immutable,
        "immutable_config_sha256": immutable_sha256,
        "evaluator_contract_sha256": sha256(evaluator_contract),
        "source_provenance": provenance,
    }
    return manifest, configs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-version", choices=("1.1", "1.2"), required=True)
    parser.add_argument("--ablation-name", choices=tuple(ABLATIONS))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shared-lr", type=float, default=1e-5)
    parser.add_argument("--private-lr", type=float, default=1e-4)
    parser.add_argument("--orth-lambda", type=float, default=0.1)
    parser.add_argument("--lambda-drift", type=float, default=0.01)
    parser.add_argument("--lambda-key", type=float, default=0.1)
    parser.add_argument("--gradient-diagnostics-interval", type=int, default=0)
    parser.add_argument("--pilot-mode", action="store_true")
    parser.add_argument(
        "--refresh-source-provenance-after-failed-run", action="store_true"
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.method_version == "1.1":
        args.lambda_key = 0.0
    manifest, configs = build_contract(args)
    if not args.check_only:
        args.output_root.mkdir(parents=True, exist_ok=True)
        run_manifest = args.output_root / "run_manifest.json"
        if run_manifest.is_file():
            previous = json.loads(run_manifest.read_text(encoding="utf-8"))
            if previous.get("immutable_config") != manifest["immutable_config"]:
                raise RuntimeError(f"Existing immutable run contract differs: {run_manifest}")
            if previous.get("source_provenance", {}).get("source_hashes") != manifest["source_provenance"]["source_hashes"]:
                if not args.refresh_source_provenance_after_failed_run:
                    raise RuntimeError(
                        "Existing source hashes differ. For a failed run with no "
                        "completed stages, rerun with "
                        "--refresh-source-provenance-after-failed-run: "
                        f"{run_manifest}"
                    )
                archive = refresh_failed_run_manifest(
                    run_manifest, previous, manifest
                )
                print(f"Archived stale run manifest to {archive}", file=sys.stderr)
        else:
            atomic_json(run_manifest, manifest)
        for stage, config in configs.items():
            path = args.output_root / "configs" / f"stage_{stage:02d}.json"
            if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != config:
                raise RuntimeError(f"Existing stage config differs: {path}")
            if not path.is_file():
                atomic_json(path, config)
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
