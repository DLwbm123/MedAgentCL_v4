#!/usr/bin/env python3
"""Aggregate MedicalSkill-v1.2 Med-PRISM results under the selected eval contract."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from evaluation_contract_v1_2 import (
    EVALUATION_PROTOCOL,
    atomic_json,
    cell_is_complete,
    load_json,
    sha256,
    validate_base_evaluation,
)

TASKS = ["VQA", "Diagnosis", "Concept", "Grounding", "Reasoning"]


def metric_path(root: Path, stage: int, mode: str, task: int) -> Path:
    return root / "evaluation" / f"stage_{stage:02d}" / f"{mode}_task_{task:02d}.summary.json"


def matrix(root: Path, mode: str) -> list[list[float | None]]:
    return [[
        float(load_json(metric_path(root, stage, mode, task))["primary_score"])
        if task <= stage else None
        for task in range(1, 6)
    ] for stage in range(1, 6)]


def write_matrix(path: Path, values: list[list[float | None]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["after_stage", *TASKS, "average"])
        for stage, row in enumerate(values, 1):
            seen = [value for value in row if value is not None]
            if len(seen) != stage:
                raise ValueError(f"Stage {stage} must contain exactly {stage} cells")
            writer.writerow([stage, *["" if x is None else f"{x:.8f}" for x in row],
                             f"{sum(seen) / len(seen):.8f}"])


def required(values: list[float | None], label: str) -> list[float]:
    if any(value is None for value in values):
        raise ValueError(f"Missing lower-triangular score: {label}")
    return [float(value) for value in values if value is not None]


def evaluation_config(root: Path) -> dict[str, Any]:
    path = root / "evaluation_run_config.json"
    if path.is_file():
        return load_json(path)
    # Outputs created before v2 always used the full 35-generation-cell contract.
    return {
        "status": "PASS",
        "format_version": "legacy_full_35_cell",
        "oracle_evaluation_enabled": True,
        "stage0_evaluation_enabled": True,
        "base_zero_shot_reused": False,
        "base_evaluation_root": str(root),
        "generation_cell_count": 35,
    }


def audit_evaluation_cells(root: Path, data: Path, eval_limit: int,
                           with_oracle: bool) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for stage in range(1, 6):
        for task in range(1, stage + 1):
            modes = ["primary_cumulative"]
            if with_oracle:
                modes.append("oracle_skill_aware")
            for mode in modes:
                if not cell_is_complete(evaluation_root=root / "evaluation", data_root=data,
                                        stage=stage, mode=mode, task=task,
                                        eval_limit=eval_limit):
                    errors.append(f"stage_{stage:02d}/{mode}_task_{task:02d}")
    return not errors, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-limit", type=int, default=0)
    args = parser.parse_args()
    root, data = args.output_root.resolve(), args.data_root.resolve()
    config = evaluation_config(root)
    with_oracle = bool(config.get("oracle_evaluation_enabled"))
    with_stage0 = bool(config.get("stage0_evaluation_enabled"))

    primary = matrix(root, "primary_cumulative")
    task_free = matrix(root, "task_free")
    write_matrix(root / "primary_cumulative_matrix.csv", primary)
    write_matrix(root / "task_free_matrix.csv", task_free)
    oracle = matrix(root, "oracle_skill_aware") if with_oracle else None
    if oracle is not None:
        write_matrix(root / "oracle_skill_aware_matrix.csv", oracle)

    if with_stage0:
        base_root = root
    else:
        base_root = Path(config["base_evaluation_root"]).resolve()
    base_contract = validate_base_evaluation(base_eval_root=base_root, data_root=data,
                                             eval_limit=args.eval_limit)
    base_stage = Path(base_contract["base_stage_dir"])
    base = [float(load_json(base_stage / f"base_zero_shot_task_{task:02d}.summary.json")
                  ["primary_score"]) for task in range(1, 6)]

    best = [max(required([primary[s][task] for s in range(task, 5)],
                         f"task {task + 1} post-learning")) for task in range(5)]
    final = required(primary[-1], "final stage")
    forgetting = [best[i] - final[i] for i in range(5)]
    diagonal = [float(primary[i][i]) for i in range(5)]
    bwt_terms = [final[i] - diagonal[i] for i in range(4)]
    metrics = {
        "status": "PASS", "scale": "0..1",
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "primary_matrix": "primary_cumulative_matrix.csv",
        "oracle_matrix": "oracle_skill_aware_matrix.csv" if with_oracle else None,
        "task_free_matrix": "task_free_matrix.csv",
        "concept_primary_metric": "all_concept_micro_f1",
        "grounding_matching": "Hungarian one-to-one",
        "base_zero_shot": base,
        "base_evaluation_reused": not with_stage0,
        "base_evaluation_stage_dir": str(base_stage),
        "final_average_score": sum(final) / 5,
        "per_task_best_score": best, "per_task_final_score": final,
        "per_task_forgetting": forgetting,
        "average_forgetting": sum(forgetting) / 5,
        "BWT": sum(bwt_terms) / len(bwt_terms), "BWT_terms": bwt_terms,
        "FWT": None, "FWT_terms": [],
        "FWT_status": "NOT_COMPUTED_BY_SEEN_TASKS_ONLY_PROTOCOL",
    }
    atomic_json(root / "cl_metrics.json", metrics)

    transitions, optimizer, shared, private, orth, drift, reloads, traces = {}, {}, {}, {}, {}, {}, {}, {}
    gates = {
        "dataset_acceptance": load_json(data / "audits/acceptance_matrix.json").get("status") == "PASS",
        "run_manifest": load_json(root / "run_manifest.json").get("status") == "PASS",
    }
    for stage in range(1, 6):
        stage_root = root / "med_prism" / f"stage_{stage:02d}"
        stage_config = load_json(root / "configs" / f"stage_{stage:02d}.json")
        completion = load_json(stage_root / "completion.json")
        shared_manifest_path = stage_root / "shared/shared_manifest.json"
        private_manifest_path = stage_root / "private/private_manifest.json"
        shared_manifest = load_json(shared_manifest_path)
        private_manifest = load_json(private_manifest_path)
        expected_shared = None if stage == 1 else str(root / "med_prism" / f"stage_{stage-1:02d}" / "shared/shared_manifest.json")
        expected_privates = [str(root / "med_prism" / f"stage_{old:02d}" / "private/private_manifest.json") for old in range(1, stage)]
        source_hash_valid = True
        if stage > 1:
            previous = load_json(Path(expected_shared))
            source_hash_valid = sha256(Path(expected_shared).parent / previous["weights_file"]) == previous["weights_sha256"]
        transition_ok = (completion.get("status") == "PASS"
                         and stage_config.get("shared_source_manifest") == expected_shared
                         and shared_manifest.get("source_checkpoint") == expected_shared
                         and stage_config.get("private_source_manifests") == expected_privates
                         and private_manifest.get("task_id") == stage and source_hash_valid)
        transitions[str(stage)] = {
            "status": "PASS" if transition_ok else "FAIL", "starts_from_base": stage == 1,
            "expected_shared_source": expected_shared,
            "configured_shared_source": stage_config.get("shared_source_manifest"),
            "saved_shared_source": shared_manifest.get("source_checkpoint"),
            "historical_private_sources": stage_config.get("private_source_manifests"),
            "source_hash_valid": source_hash_valid,
        }
        runtime = stage_root / "runtime_audits"
        optimizer[str(stage)] = load_json(runtime / f"task{stage}_optimizer_audit.json")
        shared[str(stage)] = load_json(runtime / "shared_hash_audit.json").get(f"task_{stage}", {})
        private[str(stage)] = load_json(runtime / "private_hash_audit.json").get(f"task_{stage}", {})
        orth[str(stage)] = load_json(runtime / "orth_gradient_audit.json").get(f"task_{stage}", {})
        drift[str(stage)] = load_json(runtime / "shared_drift_gradient_audit.json").get(f"task_{stage}", {})
        reloads[str(stage)] = load_json(root / "evaluation" / f"stage_{stage:02d}" / "stage_evaluation_summary.json")["reload_equivalence"]
        trace_path = runtime / f"task{stage}_training_trace.jsonl"
        rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
        finite = bool(rows) and all(math.isfinite(float(row[key])) for row in rows
                                    for key in ("task_loss", "total_loss", "shared_gradient_norm",
                                                "current_private_gradient_norm"))
        traces[str(stage)] = {"status": "PASS" if finite else "FAIL",
                              "optimizer_steps": len(rows), "last": rows[-1] if rows else None}

    audit_sets = {
        "five_stage_transition_audit.json": transitions,
        "optimizer_membership_by_stage.json": optimizer,
        "shared_hash_by_stage.json": shared,
        "private_hash_by_stage.json": private,
        "orth_gradient_by_stage.json": orth,
        "shared_drift_by_stage.json": drift,
        "reload_equivalence_by_stage.json": reloads,
        "training_trace_by_stage.json": traces,
    }
    for name, values in audit_sets.items():
        item_status = "PASS" if all(value.get("status") == "PASS" for value in values.values()) else "FAIL"
        atomic_json(root / name, {"status": item_status, "stages": values})
        gates[name.removesuffix(".json")] = item_status == "PASS"

    cells_ok, cell_errors = audit_evaluation_cells(root, data, args.eval_limit, with_oracle)
    gates["required_generation_cell_hashes"] = cells_ok
    gates["canonical_base_evaluation"] = base_contract.get("status") == "PASS"
    summaries = [load_json(root / "evaluation" / f"stage_{stage:02d}" / "stage_evaluation_summary.json")
                 for stage in range(1, 6)]
    gates["all_evaluations_pass"] = all(item.get("status") == "PASS" for item in summaries)
    gates["lower_triangular_evaluation_protocol"] = all(
        item.get("evaluation_protocol") == EVALUATION_PROTOCOL
        and item.get("evaluated_task_ids") == list(range(1, stage + 1))
        for stage, item in enumerate(summaries, 1))
    expected_multiplier = 3 if with_oracle else 2
    gates["lower_triangular_cell_counts"] = all(
        len(item.get("cells", {})) == expected_multiplier * stage
        for stage, item in enumerate(summaries, 1))
    gates["eval_limit_matches"] = all(item.get("eval_limit") == args.eval_limit for item in summaries)
    gates["task_free_is_primary_alias"] = all(
        primary[s][t] == task_free[s][t] for s in range(5) for t in range(s + 1))
    status = "PASS" if all(gates.values()) else "FAIL"
    summary = {
        "status": status, "method": "med_prism_shared_private", "ordinary_lora": False,
        "data_root": str(data), "output_root": str(root), "eval_limit_per_task": args.eval_limit,
        "full_test_evaluation": args.eval_limit == 0,
        "oracle_evaluation_enabled": with_oracle,
        "stage0_evaluation_enabled": with_stage0,
        "base_zero_shot_reused": not with_stage0,
        "configured_generation_cell_count": config.get("generation_cell_count"),
        "required_cell_errors": cell_errors,
        "evaluation_run_config": config, "gates": gates, "metrics": metrics,
    }
    atomic_json(root / "formal_experiment_summary.json", summary)
    report = [
        "# MedicalSkill-CL-v1.2 formal Med-PRISM experiment", "", f"Status: **{status}**", "",
        "- Method: med_prism_shared_private; ordinary LoRA: false",
        f"- Dataset: {data}", f"- Full test evaluation: {args.eval_limit == 0}",
        f"- Stage 0 generated in this run: {with_stage0}",
        f"- Stage 0 reused: {not with_stage0} ({base_stage})",
        f"- Oracle generated: {with_oracle}",
        f"- Configured generation cells: {config.get('generation_cell_count')}",
        f"- Final average score: {metrics['final_average_score']:.6f}",
        f"- Average forgetting: {metrics['average_forgetting']:.6f}",
        f"- BWT: {metrics['BWT']:.6f}",
        "- FWT: not computed under the seen-tasks-only lower-triangular protocol",
    ]
    (root / "formal_experiment_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
