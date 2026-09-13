#!/usr/bin/env python3
"""Schedule MedicalSkill formal evaluation on persistent per-GPU workers."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from evaluation_contract_v1_2 import (
    EVALUATION_PROTOCOL,
    atomic_json,
    build_run_config,
    cell_is_complete,
    cell_stem,
    load_json,
    required_generation_cells,
    required_stage_cells,
    sha256,
)

HERE = Path(__file__).resolve().parent
WORKER_SCRIPT = HERE / "persistent_eval_worker_v1_2.py"
TASK_WEIGHT = {1: 1.0, 2: 2.0, 3: 10.0, 4: 20.0, 5: 100.0}


def priority(job: tuple[int, str, int]) -> float:
    stage, mode, task = job
    mode_weight = 1.15 if mode == "primary_cumulative" else 1.0
    return TASK_WEIGHT[task] * mode_weight * (1.0 + 0.04 * stage)


def select_job(pending, current_stage):
    same_stage = [job for job in pending if job[0] == current_stage]
    selected = max(same_stage or pending, key=priority)
    pending.remove(selected)
    return selected


def load_recovery_records(path: Path | None) -> list[dict[str, Any]]:
    """Recover completed worker records from an append-only scheduler log."""
    if path is None or not path.is_file():
        return []
    recovered: dict[tuple[int, str, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(record, dict) or not isinstance(record.get("response"), dict):
                continue
            try:
                key = (int(record["stage"]), str(record["mode"]), int(record["task"]))
            except (KeyError, TypeError, ValueError):
                continue
            if record["response"].get("status") in {"PASS", "SKIP"}:
                recovered[key] = record
    return list(recovered.values())


def finalize_stage(*, evaluation_root, data_root, stage, with_oracle, eval_limit, records):
    stage_root = evaluation_root / f"stage_{stage:02d}"
    cells = {}
    for mode, task in required_stage_cells(stage, with_oracle):
        if not cell_is_complete(
            evaluation_root=evaluation_root, data_root=data_root, stage=stage,
            mode=mode, task=task, eval_limit=eval_limit,
        ):
            raise RuntimeError(f"Incomplete cell: stage={stage} mode={mode} task={task}")
        name = cell_stem(mode, task)
        cells[name] = load_json(stage_root / f"{name}.summary.json")
    if stage:
        for task in range(1, stage + 1):
            primary = dict(cells[cell_stem("primary_cumulative", task)])
            primary["mode"] = "task_free"
            primary["task_free_semantics"] = "all learned private experts composed without eval-task adapter routing"
            name = cell_stem("task_free", task)
            atomic_json(stage_root / f"{name}.summary.json", primary)
            cells[name] = primary
    loads = []
    for record in records:
        response = record.get("response") or {}
        stage_load = response.get("stage_load")
        if stage_load and stage_load.get("stage") == stage:
            loads.append(stage_load)
    previous_path = stage_root / "stage_evaluation_summary.json"
    previous = load_json(previous_path) if previous_path.is_file() else {}
    probe = loads[0]["reload_equivalence"] if loads else previous.get(
        "reload_equivalence",
        {"status": "NOT_APPLICABLE", "stage": 0} if stage == 0 else {"status": "MISSING", "stage": stage},
    )
    active = loads[0]["active_component_manifest"] if loads else previous.get(
        "active_component_manifest", {"method": "base_zero_shot", "private_task_ids": []}
    )
    status = "PASS" if probe.get("status") in {"PASS", "NOT_APPLICABLE"} and all(
        value.get("status") == "PASS" for value in cells.values()
    ) else "FAIL"
    summary = {
        "status": status, "stage": stage, "evaluation_protocol": EVALUATION_PROTOCOL,
        "evaluated_task_ids": list(range(1, 6)) if stage == 0 else list(range(1, stage + 1)),
        "matrix_scope": "base_zero_shot_all_tasks" if stage == 0 else "seen_tasks_only",
        "oracle_evaluation_enabled": bool(with_oracle and stage > 0),
        "task_free_derived_from_primary": bool(stage > 0),
        "model_load_count": len(loads), "persistent_worker_evaluation": True,
        "data_root": str(data_root), "eval_limit": eval_limit,
        "active_component_manifest": active, "reload_equivalence": probe, "cells": cells,
    }
    atomic_json(previous_path, summary)
    if status != "PASS":
        raise RuntimeError(f"Stage {stage} finalization failed")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--with-oracle", action="store_true")
    parser.add_argument("--with-stage0", action="store_true")
    parser.add_argument("--base-eval-root", type=Path)
    parser.add_argument("--recovery-records-log", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        parser.error("--gpus must contain distinct GPU IDs")
    if args.eval_limit < 0 or args.batch_size <= 0:
        parser.error("invalid eval limit or batch size")
    data_root = Path(os.environ["MED_PRISM_DATA_ROOT"]).resolve()
    output_root = Path(os.environ["MED_PRISM_ARTIFACT_ROOT"]).resolve()
    evaluation_root = output_root / "evaluation"
    config = build_run_config(
        data_root=data_root, output_root=output_root, eval_limit=args.eval_limit,
        batch_size=args.batch_size, with_oracle=args.with_oracle,
        with_stage0=args.with_stage0, base_eval_root=args.base_eval_root,
    )
    configured_jobs = required_generation_cells(with_stage0=args.with_stage0, with_oracle=args.with_oracle)
    recovered_records = load_recovery_records(args.recovery_records_log)
    pending = [job for job in configured_jobs if not cell_is_complete(
        evaluation_root=evaluation_root, data_root=data_root, stage=job[0],
        mode=job[1], task=job[2], eval_limit=args.eval_limit,
    )]
    plan = {
        "status": "PASS", "configured_generation_cell_count": len(configured_jobs),
        "pending_generation_cell_count": len(pending),
        "cached_generation_cell_count": len(configured_jobs) - len(pending),
        "oracle_evaluation_enabled": args.with_oracle,
        "stage0_evaluation_enabled": args.with_stage0,
        "recovered_worker_record_count": len(recovered_records),
        "jobs_priority_order": [
            {"stage": s, "mode": m, "task": t}
            for s, m, t in sorted(pending, key=priority, reverse=True)
        ],
    }
    if args.plan_only:
        print(json.dumps({"run_config": config, "plan": plan}, indent=2))
        return 0
    evaluation_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "evaluation_run_config.json", config)
    atomic_json(evaluation_root / "evaluation_plan.json", plan)
    started = time.time()
    pending_lock, records_lock = threading.Lock(), threading.Lock()
    failure = threading.Event()
    records, worker_reports = [], []

    def worker_thread(gpu: str) -> None:
        log_path = evaluation_root / f"persistent_worker_gpu_{gpu}.log"
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"], env["MED_PRISM_PHYSICAL_GPU"] = gpu, gpu
        with log_path.open("a", encoding="utf-8") as error_log:
            process = subprocess.Popen(
                [sys.executable, str(WORKER_SCRIPT)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=error_log, text=True, bufsize=1, env=env,
            )
            assert process.stdin and process.stdout
            ready = json.loads(process.stdout.readline())
            if ready.get("status") != "READY":
                failure.set(); return
            current_stage = None
            while not failure.is_set():
                with pending_lock:
                    if not pending: break
                    stage, mode, task = select_job(pending, current_stage)
                command = {
                    "command": "evaluate", "stage": stage, "mode": mode, "task": task,
                    "eval_limit": args.eval_limit, "batch_size": args.batch_size,
                }
                job_started = time.time()
                process.stdin.write(json.dumps(command) + "\n"); process.stdin.flush()
                response = json.loads(process.stdout.readline())
                record = {
                    "gpu": gpu, "stage": stage, "mode": mode, "task": task,
                    "elapsed_seconds": time.time() - job_started, "response": response,
                }
                with records_lock:
                    records.append(record); print(json.dumps(record, ensure_ascii=False), flush=True)
                if response.get("status") not in {"PASS", "SKIP"}:
                    failure.set(); break
                current_stage = stage
            process.stdin.write(json.dumps({"command": "shutdown"}) + "\n"); process.stdin.flush()
            shutdown = json.loads(process.stdout.readline())
            with records_lock: worker_reports.append(shutdown.get("worker_report", shutdown))
            process.wait(timeout=60)
            if process.returncode: failure.set()

    threads = [threading.Thread(target=worker_thread, args=(gpu,)) for gpu in gpu_ids]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    status = "FAIL" if failure.is_set() else "PASS"
    finalized = {}
    if status == "PASS":
        for stage in range(0 if args.with_stage0 else 1, 6):
            finalized[str(stage)] = finalize_stage(
                evaluation_root=evaluation_root, data_root=data_root, stage=stage,
                with_oracle=args.with_oracle, eval_limit=args.eval_limit,
                records=[*recovered_records, *records],
            )
    report = {
        "status": status,
        "scheduler": "persistent_gpu_workers_reasoning_first_with_stage_affinity",
        "gpus": gpu_ids, "batch_size": args.batch_size, "eval_limit": args.eval_limit,
        "oracle_evaluation_enabled": args.with_oracle,
        "stage0_evaluation_enabled": args.with_stage0,
        "configured_generation_cell_count": len(configured_jobs),
        "scheduled_generation_cell_count": len(records),
        "cached_generation_cell_count": len(configured_jobs) - len(records),
        "recovered_worker_record_count": len(recovered_records),
        "completed_generation_cell_count": sum(
            item["response"].get("status") in {"PASS", "SKIP"} for item in records
        ),
        "elapsed_seconds": time.time() - started,
        "run_config": str(output_root / "evaluation_run_config.json"),
        "run_config_sha256": sha256(output_root / "evaluation_run_config.json"),
        "worker_reports": worker_reports, "records": records,
        "finalized_stages": sorted(finalized),
    }
    atomic_json(evaluation_root / "parallel_scheduler_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
