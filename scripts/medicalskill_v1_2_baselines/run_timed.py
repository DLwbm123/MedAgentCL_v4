#!/usr/bin/env python3
"""Run one command while recording reusable wall-clock and GPU metadata."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.medicalskill_v1_2_baselines.baseline_harness import atomic_json


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def query_gpus(requested: set[str]) -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for raw in result.stdout.splitlines():
        parts = [item.strip() for item in raw.split(",", 2)]
        if len(parts) != 3 or (requested and parts[0] not in requested):
            continue
        try:
            memory = int(parts[2])
        except ValueError:
            continue
        rows.append({"index": parts[0], "model": parts[1], "memory_used_mib": memory})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--number-of-samples", type=int)
    parser.add_argument("--training-steps", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("A command is required after --")
    if args.sample_interval <= 0:
        raise ValueError("--sample-interval must be positive")
    gpu_ids = {item.strip() for item in args.gpus.split(",") if item.strip()}
    initial = query_gpus(gpu_ids) if gpu_ids else []
    peak = {item["index"]: item["memory_used_mib"] for item in initial}
    models = {item["index"]: item["model"] for item in initial}
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(args.sample_interval):
            for item in (query_gpus(gpu_ids) if gpu_ids else []):
                models[item["index"]] = item["model"]
                peak[item["index"]] = max(
                    peak.get(item["index"], 0), item["memory_used_mib"]
                )

    start_stamp = timestamp()
    started = time.monotonic()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    exit_code = 255
    failure = None
    try:
        process = subprocess.run(command)
        exit_code = int(process.returncode)
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        ended = time.monotonic()
        stop.set()
        thread.join(timeout=max(2.0, args.sample_interval * 2))
        final = query_gpus(gpu_ids) if gpu_ids else []
        for item in final:
            models[item["index"]] = item["model"]
            peak[item["index"]] = max(
                peak.get(item["index"], 0), item["memory_used_mib"]
            )
        gpu_count = len(gpu_ids)
        payload = {
            "status": "PASS" if exit_code == 0 and failure is None else "FAIL",
            "format_version": "medicalskill_cl_runtime_v1",
            "category": args.category,
            "start_timestamp": start_stamp,
            "end_timestamp": timestamp(),
            "wall_clock_seconds": ended - started,
            "gpu_count": gpu_count,
            "requested_gpu_ids": sorted(gpu_ids),
            "gpu_models": [
                {"index": key, "model": models[key]} for key in sorted(models)
            ],
            "peak_gpu_memory_mib": {
                key: peak[key] for key in sorted(peak)
            } or None,
            "peak_memory_measurement": (
                "nvidia-smi sampled whole-device memory; may include other processes"
                if peak
                else "UNAVAILABLE"
            ),
            "gpu_hours": (ended - started) * gpu_count / 3600.0,
            "training_steps": args.training_steps,
            "number_of_samples": args.number_of_samples,
            "command": command,
            "command_exit_code": exit_code,
            "failure": failure,
            "pid": os.getpid(),
        }
        atomic_json(args.record.resolve(), payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
