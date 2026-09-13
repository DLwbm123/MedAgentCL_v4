#!/usr/bin/env python3
"""Cycle 02 closure runner for the frozen MedicalSkill-CL Concept 35K."""

from __future__ import annotations

import argparse
import asyncio
import collections
import fcntl
import hashlib
import json
import os
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


ROOT = Path("/root/MedAgentCL_v4")
SCRIPT_DIR = ROOT / "scripts/medicalskill_kimi_k3"
sys.path.insert(0, str(SCRIPT_DIR))

import run_kimi_k3_v3_1 as core  # noqa: E402
import v3_1_first_round_policy as first_policy  # noqa: E402


ARTIFACT = core.ARTIFACT
CYCLE_ID = "cycle_02"
CYCLE_LIMIT = 18_000
TOTAL_TARGET = 35_000
FIRST_ROUND_SIZE = 18_000
SECOND_ROUND_SIZE = 17_000
CYCLE_DIR = ARTIFACT / "cycle_02_35k_closure"
REPORT_DIR = CYCLE_DIR / "batch_reports"
PREFLIGHT_JSON = CYCLE_DIR / "cycle_02_preflight.json"
PREFLIGHT_MD = CYCLE_DIR / "cycle_02_preflight.md"
TARGET_MANIFEST = CYCLE_DIR / "cycle_02_target_manifest.json"
RESUME_PATH = CYCLE_DIR / "cycle_02_resume_state.json"
RUNNER_STATE_PATH = CYCLE_DIR / "cycle_02_runner_state.json"
SUMMARY_PATH = CYCLE_DIR / "cycle_02_summary.json"
EVENTS_PATH = CYCLE_DIR / "cycle_02_events.jsonl"
COMMAND_PATH = CYCLE_DIR / "cycle_02_execution_command.txt"
LOCK_PATH = CYCLE_DIR / ".cycle_02.lock"
EXPECTED_START_RESULTS_HASH = (
    "34eecbef3ca8b30613e8390ddd3fb983f6a7debf4866806d0da1c95eae125c2c"
)
EXPECTED_HASHES = {
    "prompt_hash": (
        "fb63d4b1963d5b9c93188ed65b7ec5677eeaf7ecb8e38b86bb73e640f516ae68"
    ),
    "schema_hash": (
        "14830fcd9b334ac7ce24c17cd0704a373ccec247da03bf1435dad139a8d38aea"
    ),
    "run_config_hash": (
        "382703cd787c9c49cce0ccdc2cda71615ccbb16f35820cc4a9ad290243d230a1"
    ),
}


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(values: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{value}\n" for value in values).encode()
    ).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    first_policy.atomic_write_json(path, value)


def append_event(event_type: str, **values: Any) -> None:
    first_policy.append_jsonl(
        EVENTS_PATH,
        {
            "event_type": event_type,
            "timestamp": iso_now(),
            **values,
        },
    )


def api_rows(
    history: list[dict[str, Any]],
    cycle_id: str | None = None,
) -> list[dict[str, Any]]:
    rows = [
        row for row in history
        if (
            row.get("api_call_performed") is True
            or (
                row.get("api_call_performed") is not False
                and row.get("http_status") is not None
            )
        )
    ]
    if cycle_id is not None:
        rows = [
            row for row in rows
            if row.get("cycle_id") == cycle_id
        ]
    return rows


def matching_runner_processes() -> list[dict[str, Any]]:
    matches = []
    current_pid = os.getpid()
    for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc_dir.name)
            if pid == current_pid:
                continue
            command = (
                proc_dir / "cmdline"
            ).read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            )
        except (OSError, ValueError):
            continue
        if "--preflight" in command:
            continue
        if (
            "run_kimi_k3_v3_1.py" in command
            or "run_kimi_k3_v3_1_cycle01.py" in command
            or "run_kimi_k3_v3_1_cycle02" in command
        ):
            matches.append({
                "pid": pid,
                "command": command[:500],
            })
    return matches


def latest_rows(
    history: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in history:
        record_id = row.get("stable_id")
        if record_id:
            latest[str(record_id)] = row
    return latest


def active_success_ids(
    stable_ids: list[str],
    sources: dict[str, dict[str, Any]],
    prompt: dict[str, Any],
    config: dict[str, Any],
    history: list[dict[str, Any]],
) -> set[str]:
    latest = latest_rows(history)
    return {
        record_id for record_id in stable_ids
        if core.active_success(
            latest.get(record_id),
            sources[record_id],
            prompt,
            config,
        )
    }


def formal_first_round_success(
    stable_ids: list[str],
    history: list[dict[str, Any]],
) -> set[str]:
    classification = first_policy.classify_records(
        stable_ids,
        history,
        first_policy.load_permanent_registry(),
    )
    return set(classification["success"])


def batch_plan(
    stable_ids: list[str],
    formal_first_success: set[str],
) -> list[dict[str, Any]]:
    stage1 = [
        record_id for record_id in stable_ids[17_000:18_000]
        if record_id not in formal_first_success
    ]
    plan = [{
        "phase": "first_round_closure",
        "logical_batch": "batch_17_closure",
        "batch_index": 17,
        "selected_ids": stable_ids[17_000:18_000],
        "required_ids": stage1,
    }]
    for offset in range(17):
        start = 18_000 + offset * 1_000
        selected = stable_ids[start:start + 1_000]
        plan.append({
            "phase": "second_round",
            "logical_batch": f"round2_batch_{offset:02d}",
            "batch_index": 18 + offset,
            "selected_ids": selected,
            "required_ids": selected,
        })
    return plan


def preflight_payload() -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    prompt, schema, config, _ = core.load_contract(ARTIFACT)
    sources, stable_ids = core.load_sources(ARTIFACT, config)
    results_path = ARTIFACT / "v3_1_results.jsonl"
    history: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    with results_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                history.append(json.loads(line))
            except Exception as error:
                parse_errors.append({
                    "line": line_number,
                    "error": type(error).__name__,
                })

    formal_success = formal_first_round_success(
        stable_ids, history
    )
    active_success = active_success_ids(
        stable_ids, sources, prompt, config, history
    )
    first_ids = set(stable_ids[:FIRST_ROUND_SIZE])
    second_ids = set(stable_ids[FIRST_ROUND_SIZE:])
    first_remaining = [
        record_id for record_id in stable_ids[:FIRST_ROUND_SIZE]
        if record_id not in formal_success
    ]
    second_ordered = stable_ids[FIRST_ROUND_SIZE:]
    second_cache = [
        record_id for record_id in second_ordered
        if record_id in active_success
    ]
    source_mismatch = []
    latest = latest_rows(history)
    for record_id in active_success:
        if (
            latest[record_id].get("source_caption_sha256")
            != sources[record_id]["source_caption_sha256"]
        ):
            source_mismatch.append(record_id)

    old_resume_path = ARTIFACT / "v3_1_resume_state.json"
    old_resume = json.loads(old_resume_path.read_text())
    results_hash = sha256(results_path)
    credential = (
        os.environ.get("MOONSHOT_API_KEY")
        or os.environ.get("KIMI_API_KEY")
        or ""
    )
    running_processes = matching_runner_processes()
    target_ids = first_remaining + second_ordered
    unresolved_api_ids = [
        record_id for record_id in target_ids
        if record_id not in active_success
    ]
    partition_union = (
        formal_success
        | set(first_remaining)
        | second_ids
    )
    checks = {
        "no_old_process": not running_processes,
        "credential_present": bool(credential),
        "results_hash_matches_known": (
            results_hash == EXPECTED_START_RESULTS_HASH
        ),
        "jsonl_parse_errors_zero": not parse_errors,
        "stable_ids_total_35000": (
            len(stable_ids) == len(set(stable_ids)) == TOTAL_TARGET
        ),
        "sources_total_35000": len(sources) == TOTAL_TARGET,
        "formal_success_17725": len(formal_success) == 17_725,
        "first_remaining_275": len(first_remaining) == 275,
        "batch17_success_725": (
            sum(
                record_id in formal_success
                for record_id in stable_ids[17_000:18_000]
            )
            == 725
        ),
        "second_round_target_17000": (
            len(second_ordered) == 17_000
        ),
        "partitions_disjoint": not bool(
            (formal_success & set(first_remaining))
            | (formal_success & second_ids)
            | (set(first_remaining) & second_ids)
        ),
        "partition_union_35000": (
            partition_union == set(stable_ids)
        ),
        "source_hash_mismatch_zero": not source_mismatch,
        "old_resume_results_hash_matches": (
            old_resume.get("results_hash") == results_hash
        ),
        "prompt_hash_frozen": (
            prompt["prompt_hash"]
            == EXPECTED_HASHES["prompt_hash"]
        ),
        "schema_hash_frozen": (
            prompt["schema_hash"]
            == EXPECTED_HASHES["schema_hash"]
        ),
        "run_config_hash_frozen": (
            config["run_config_hash"]
            == EXPECTED_HASHES["run_config_hash"]
        ),
        "request_model_k3": core.REQUEST_MODEL == "k3",
        "expected_route_k2_6": (
            core.ROUTED_MODEL == "kimi-k2.6"
        ),
        "reasoning_effort_none": core.EFFORT == "none",
        "temperature_0_6": core.TEMPERATURE == 0.6,
        "top_p_omitted": (
            "top_p" not in core.request_kwargs(
                sources[stable_ids[0]], prompt, schema
            )
        ),
        "cycle_02_calls_zero": not api_rows(
            history, CYCLE_ID
        ),
        "target_count_17275": len(target_ids) == 17_275,
        "unresolved_api_count_17255": (
            len(unresolved_api_ids) == 17_255
        ),
    }
    payload = {
        "status": (
            "PASS_WITH_EXPLAINED_DIFFERENCES"
            if all(checks.values()) else "FAIL"
        ),
        "cycle_id": CYCLE_ID,
        "created_at": iso_now(),
        "api_called": False,
        "checks": checks,
        "credential": {
            "present": bool(credential),
            "length": len(credential),
        },
        "results": {
            "path": str(results_path),
            "sha256": results_hash,
            "bytes": results_path.stat().st_size,
            "jsonl_rows": len(history),
            "jsonl_parse_error_count": len(parse_errors),
            "jsonl_parse_error_examples": parse_errors[:10],
            "append_only_event_identity_unique": (
                len(history) == len({
                    (
                        row.get("run_id"),
                        row.get("stable_id"),
                        row.get("attempt_count"),
                        row.get("timestamp"),
                    )
                    for row in history
                })
            ),
        },
        "sets": {
            "frozen_stable_ids": len(stable_ids),
            "formal_success": len(formal_success),
            "first_round_remaining": len(first_remaining),
            "second_round_target": len(second_ordered),
            "second_round_existing_cache": len(second_cache),
            "cycle_target_unique_records": len(target_ids),
            "cycle_new_http_originals_expected": (
                len(unresolved_api_ids)
            ),
            "initial_retry_margin": (
                CYCLE_LIMIT - len(unresolved_api_ids)
            ),
            "first_remaining_sha256": canonical_sha(
                first_remaining
            ),
            "second_round_sha256": canonical_sha(
                second_ordered
            ),
            "target_sha256": canonical_sha(target_ids),
            "source_hash_mismatch_count": (
                len(source_mismatch)
            ),
        },
        "contract": {
            "prompt_hash": prompt["prompt_hash"],
            "schema_hash": prompt["schema_hash"],
            "run_config_hash": config["run_config_hash"],
            "request_model": core.REQUEST_MODEL,
            "expected_routed_model": core.ROUTED_MODEL,
            "reasoning_effort": core.EFFORT,
            "temperature": core.TEMPERATURE,
            "top_p_sent": False,
            "max_completion_tokens": core.MAX_TOKENS,
            "runtime_policy_revision": (
                core.RUNTIME_POLICY_REVISION
            ),
        },
        "resume": {
            "path": str(old_resume_path),
            "results_hash_matches": (
                old_resume.get("results_hash") == results_hash
            ),
            "status": old_resume.get("status"),
            "completed": len(
                old_resume.get("completed") or []
            ),
            "pending": len(old_resume.get("pending") or []),
            "failed": len(old_resume.get("failed") or []),
        },
        "running_processes": running_processes,
        "explained_differences": [
            {
                "name": "second_round_existing_cache",
                "count": len(second_cache),
                "handling": (
                    "Reuse as cache hits; never issue API calls."
                ),
            },
            {
                "name": "legacy_batch17_resume_failed",
                "count": len(old_resume.get("failed") or []),
                "handling": (
                    "Formal success view is authoritative; include "
                    "the legacy rejected ID in the 275 remaining."
                ),
            },
            {
                "name": "append_only_repeated_stable_ids",
                "handling": (
                    "Rows are API/reclassification events. Stable-ID "
                    "uniqueness is enforced on the active success view."
                ),
            },
        ],
    }
    manifest = {
        "cycle_id": CYCLE_ID,
        "created_at": iso_now(),
        "stable_ids_sha256": sha256(
            ARTIFACT / "concept_35k_stable_ids.txt"
        ),
        "input_manifest_sha256": sha256(
            ARTIFACT / "concept_35k_v3_1_input_manifest.jsonl"
        ),
        "first_round_remaining_ids": first_remaining,
        "second_round_ids": second_ordered,
        "second_round_cache_ids": second_cache,
        "target_ids_sha256": canonical_sha(target_ids),
        "first_round_remaining_ids_sha256": canonical_sha(
            first_remaining
        ),
        "second_round_ids_sha256": canonical_sha(
            second_ordered
        ),
    }
    return payload, manifest


def write_preflight() -> dict[str, Any]:
    CYCLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload, manifest = preflight_payload()
    atomic_json(PREFLIGHT_JSON, payload)
    atomic_json(TARGET_MANIFEST, manifest)
    markdown = (
        "# Cycle 02 preflight\n\n"
        f"Status: **{payload['status']}**\n\n"
        f"- Formal success: {payload['sets']['formal_success']}\n"
        f"- First-round remaining: "
        f"{payload['sets']['first_round_remaining']}\n"
        f"- Second-round target: "
        f"{payload['sets']['second_round_target']}\n"
        f"- Existing second-round cache: "
        f"{payload['sets']['second_round_existing_cache']}\n"
        f"- Expected new HTTP originals: "
        f"{payload['sets']['cycle_new_http_originals_expected']}\n"
        f"- Initial retry margin: "
        f"{payload['sets']['initial_retry_margin']}\n"
        f"- Results SHA-256: "
        f"`{payload['results']['sha256']}`\n"
    )
    PREFLIGHT_MD.write_text(markdown, encoding="utf-8")
    if payload["status"] == "FAIL":
        raise RuntimeError("cycle_02_preflight_failed")
    return payload


def budget_reservation_count() -> int:
    if not EVENTS_PATH.exists():
        return 0
    count = 0
    with EVENTS_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event_type") == "API_BUDGET_RESERVED":
                count += 1
    return count

class AtomicCycleBudget:
    def __init__(
        self,
        used: int,
        unresolved: set[str],
    ):
        self.used = used
        self.limit = CYCLE_LIMIT
        self.unresolved = set(unresolved)
        self.funded: set[str] = set()
        self.lock = asyncio.Lock()
        self.zero_retry_mode = False

    async def reserve(
        self,
        record_id: str,
        extra_attempt: bool,
    ) -> bool:
        async with self.lock:
            remaining_budget = self.limit - self.used
            required = len(self.unresolved - self.funded)
            margin = remaining_budget - required
            if remaining_budget <= 0:
                return False
            if extra_attempt and margin <= 0:
                self.zero_retry_mode = True
                return False
            if not extra_attempt:
                if record_id not in self.unresolved:
                    return False
                self.funded.add(record_id)
            append_event(
                "API_BUDGET_RESERVED",
                reservation_number=self.used + 1,
                stable_id=record_id,
                extra_attempt=extra_attempt,
                remaining_budget_after=self.limit - self.used - 1,
            )
            self.used += 1
            return True

    async def complete(self, record_id: str) -> None:
        async with self.lock:
            self.unresolved.discard(record_id)
            self.funded.discard(record_id)

    async def fail(self, record_id: str) -> None:
        async with self.lock:
            self.funded.discard(record_id)

    async def snapshot(self) -> dict[str, Any]:
        async with self.lock:
            remaining_budget = self.limit - self.used
            remaining_unique = len(self.unresolved)
            return {
                "cycle_api_attempts": self.used,
                "remaining_call_budget": remaining_budget,
                "remaining_unique_records": remaining_unique,
                "retry_margin": (
                    remaining_budget - remaining_unique
                ),
                "zero_retry_mode": self.zero_retry_mode,
            }


class RecordBudget:
    def __init__(
        self,
        shared: AtomicCycleBudget,
        record_id: str,
    ):
        self.shared = shared
        self.record_id = record_id
        self.calls = 0

    async def reserve(self) -> bool:
        allowed = await self.shared.reserve(
            self.record_id,
            extra_attempt=self.calls > 0,
        )
        if allowed:
            self.calls += 1
        return allowed


def cycle_rows(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return api_rows(history, CYCLE_ID)


def is_retryable_failure(
    row: dict[str, Any] | None,
) -> bool:
    if not row:
        return True
    return bool(
        row.get("http_status") in {429, 500, 502, 503, 504}
        or row.get("error_type") in {
            "timeout",
            "connection",
            "rate_limit",
            "engine_overloaded",
            "http_500",
            "http_502",
            "http_503",
            "http_504",
        }
        or row.get("quality_status") in {
            "TECHNICAL_RETRY_REQUIRED",
            "TECHNICAL_FAILED_AFTER_RETRY",
        }
    )


def hard_failure(
    rows: list[dict[str, Any]],
) -> str | None:
    for row in rows:
        status = row.get("http_status")
        if status in {400, 401, 404}:
            return f"http_{status}"
        if status == 403:
            if row.get("quota_exhausted"):
                message = str(
                    row.get("provider_error_message") or ""
                ).lower()
                if "billing cycle" in message:
                    return "BILLING_CYCLE_QUOTA"
                if row.get("quota_scope") == "weekly":
                    return "WEEKLY_QUOTA"
                return "PROVIDER_USAGE_LIMIT"
            return "non_quota_http_403"
        if row.get("quality_status") == "CONTRACT_VIOLATION":
            return "contract_violation"
        if (
            status == 200
            and row.get("response_model_id") not in {None, "k3"}
        ):
            return "response_model_changed"
    return None


def summarize_rows(
    active_rows: list[dict[str, Any]],
    attempt_rows: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    latencies = [
        float(row.get("latency") or 0)
        for row in attempt_rows
    ]
    quality_status = collections.Counter(
        str(row.get("quality_status")) for row in active_rows
    )
    return {
        "api_success_rate": (
            sum(row.get("http_status") == 200 for row in attempt_rows)
            / len(attempt_rows) if attempt_rows else 1.0
        ),
        "reasoning_token_total": sum(
            int(row.get("reasoning_token_usage") or 0)
            for row in attempt_rows
        ),
        "prompt_token_total": sum(
            int(row.get("prompt_tokens") or 0)
            for row in attempt_rows
        ),
        "completion_token_total": sum(
            int(row.get("completion_tokens") or 0)
            for row in attempt_rows
        ),
        "cached_token_total": sum(
            int(row.get("cached_tokens") or 0)
            for row in attempt_rows
        ),
        "latency_seconds_total": round(sum(latencies), 6),
        "latency_seconds_mean": (
            round(sum(latencies) / len(latencies), 6)
            if latencies else 0.0
        ),
        "modality_distribution": dict(sorted(
            collections.Counter(
                str(row.get("modality") or "unknown")
                for row in active_rows
            ).items()
        )),
        "split_distribution": dict(sorted(
            collections.Counter(
str(
                    sources[row["stable_id"]].get("split")
                    or (
                        "test"
                        if "_test_" in row["stable_id"]
                        else "train"
                        if "_train_" in row["stable_id"]
                        else "unknown"
                    )
                )
                for row in active_rows
            ).items()
        )),
        "quality_status_distribution": dict(sorted(
            quality_status.items()
        )),
        "pass_count": quality_status.get("PASS", 0),
        "pass_with_qc_flags_count": quality_status.get(
            "PASS_WITH_QC_FLAGS", 0
        ),
        "response_model_distribution": dict(sorted(
            collections.Counter(
                str(row.get("response_model_id") or "unknown")
                for row in attempt_rows
                if row.get("http_status") == 200
            ).items()
        )),
    }

def batch_report(
    entry: dict[str, Any],
    start_history_len: int,
    start_success: set[str],
    stable_ids: list[str],
    sources: dict[str, dict[str, Any]],
    prompt: dict[str, Any],
    config: dict[str, Any],
    history: list[dict[str, Any]],
    status: str,
    technical_pending: list[str],
    budget_snapshot: dict[str, Any],
) -> dict[str, Any]:
    selected = entry["selected_ids"]
    selected_set = set(selected)
    success = active_success_ids(
        stable_ids, sources, prompt, config, history
    )
    new_rows = history[start_history_len:]
    attempts = [
        row for row in new_rows
        if row.get("cycle_id") == CYCLE_ID
        and row.get("api_call_performed") is not False
    ]
    active_rows = [
        latest_rows(history)[record_id]
        for record_id in selected
        if record_id in success
    ]
    concept_counts = [
        len(row.get("final_concepts") or [])
        for row in active_rows
    ]
    qc = collections.Counter(
        flag
        for row in active_rows
        for flag in (row.get("audit_flags") or [])
    )
    http = collections.Counter(
        str(row.get("http_status"))
        if row.get("http_status") is not None
        else "transport_error"
        for row in attempts
    )
    row_stats = summarize_rows(active_rows, attempts, sources)
    cycle_history = cycle_rows(history)
    report = {
        "status": status,
        "cycle_id": CYCLE_ID,
        "phase": entry["phase"],
        "logical_batch": entry["logical_batch"],
        "batch_index": entry["batch_index"],
        "planned_unique_records": len(selected),
        "success": sum(
            record_id in success for record_id in selected
        ),
        "cache_hits_at_start": sum(
            record_id in start_success for record_id in selected
        ),
        "api_attempts_this_batch": len(attempts),
        "cycle_api_attempts": len(cycle_history),
        "lifetime_api_attempts": len(api_rows(history)),
        "http_status_distribution": dict(sorted(http.items())),
        "row_statistics": row_stats,
        "technical_retry_count": max(
            0,
            len(attempts)
            - len({
                row.get("stable_id") for row in attempts
            }),
        ),
        "technical_pending": technical_pending,
        "remaining_unique_records": (
            TOTAL_TARGET - len(success)
        ),
        "contract_valid_rate": (
            sum(
                first_policy.is_contract_valid_http_200(row)
                for row in attempts
                if row.get("http_status") == 200
            )
            / max(
                1,
                sum(
                    row.get("http_status") == 200
                    for row in attempts
                ),
            )
        ),
        "concept_count_histogram": dict(sorted(
            collections.Counter(concept_counts).items()
        )),
        "cap8_count": sum(value == 8 for value in concept_counts),
        "cap8_ratio": (
            sum(value == 8 for value in concept_counts)
            / len(concept_counts)
            if concept_counts else 0.0
        ),
        "qc_flag_distribution": dict(sorted(qc.items())),
        "budget": budget_snapshot,
        "selected_ids_sha256": canonical_sha(selected),
        "success_ids_sha256": canonical_sha([
            record_id for record_id in selected
            if record_id in success
        ]),
        "results_sha256": sha256(
            ARTIFACT / "v3_1_results.jsonl"
        ),
        "resume_sha256": (
            sha256(RESUME_PATH)
            if RESUME_PATH.exists() else None
        ),
        "prompt_hash": prompt["prompt_hash"],
        "schema_hash": prompt["schema_hash"],
        "run_config_hash": config["run_config_hash"],
        "temperature": core.TEMPERATURE,
        "last_update": iso_now(),
    }
    name = entry["logical_batch"]
    path = REPORT_DIR / f"cycle_02_{name}.json"
    atomic_json(path, report)
    atomic_json(
        REPORT_DIR / f"cycle_02_{name}.hash.json",
        {
            "report_sha256": sha256(path),
            "results_sha256": report["results_sha256"],
            "resume_sha256": report["resume_sha256"],
        },
    )
    (REPORT_DIR / f"cycle_02_{name}.md").write_text(
        (
            f"# {name}\n\n"
            f"Status: **{status}**\n\n"
            f"- Success: {report['success']}/"
            f"{report['planned_unique_records']}\n"
            f"- API attempts this batch: "
            f"{report['api_attempts_this_batch']}\n"
            f"- Cycle API attempts: "
            f"{report['cycle_api_attempts']}\n"
            f"- Retry margin: "
            f"{budget_snapshot['retry_margin']}\n"
            f"- Cap8 ratio: {report['cap8_ratio']:.4%}\n"
        ),
        encoding="utf-8",
    )
    return report


def write_resume(
    status: str,
    current_entry: dict[str, Any] | None,
    stable_ids: list[str],
    sources: dict[str, dict[str, Any]],
    prompt: dict[str, Any],
    config: dict[str, Any],
    history: list[dict[str, Any]],
    technical_pending: list[str],
    budget: dict[str, Any],
    stop_reason: str | None = None,
) -> None:
    success = active_success_ids(
        stable_ids, sources, prompt, config, history
    )
    cycle_history = cycle_rows(history)
    http = collections.Counter(
        str(row.get("http_status"))
        if row.get("http_status") is not None
        else "transport_error"
        for row in cycle_history
    )
    state = {
        "status": status,
        "cycle_id": CYCLE_ID,
        "current_phase": (
            current_entry.get("phase")
            if current_entry else None
        ),
        "current_logical_batch": (
            current_entry.get("logical_batch")
            if current_entry else None
        ),
        "current_batch_index": (
            current_entry.get("batch_index")
            if current_entry else None
        ),
        "unique_records_completed": len(success),
        "remaining_unique_records": (
            TOTAL_TARGET - len(success)
        ),
        "cycle_api_attempts": len(cycle_history),
        "lifetime_api_attempts": len(api_rows(history)),
        "http_status_distribution": dict(sorted(http.items())),
        "technical_retry_count": (
            len(cycle_history)
            - len({
                row.get("stable_id") for row in cycle_history
            })
        ),
        "technical_pending": technical_pending,
        "budget": budget,
        "stop_reason": stop_reason,
        "results_sha256": sha256(
            ARTIFACT / "v3_1_results.jsonl"
        ),
        "prompt_hash": prompt["prompt_hash"],
        "schema_hash": prompt["schema_hash"],
        "run_config_hash": config["run_config_hash"],
        "temperature": core.TEMPERATURE,
        "last_update": iso_now(),
    }
    atomic_json(RESUME_PATH, state)
    atomic_json(SUMMARY_PATH, state)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def write_progress_checkpoint(
    plan_position: int,
    entry: dict[str, Any],
    stable_ids: list[str],
    sources: dict[str, dict[str, Any]],
    prompt: dict[str, Any],
    config: dict[str, Any],
    history: list[dict[str, Any]],
    budget: dict[str, Any],
) -> None:
    success = active_success_ids(
        stable_ids, sources, prompt, config, history
    )
    payload = {
        "cycle_id": CYCLE_ID,
        "plan_position": plan_position,
        "logical_batch": entry["logical_batch"],
        "unique_records_completed": len(success),
        "remaining_unique_records": TOTAL_TARGET - len(success),
        "cycle_result_rows": len(cycle_rows(history)),
        "cycle_budget_reservations": budget_reservation_count(),
        "budget": budget,
        "results_sha256": sha256(ARTIFACT / "v3_1_results.jsonl"),
        "resume_sha256": sha256(RESUME_PATH),
        "last_update": iso_now(),
    }
    atomic_json(CYCLE_DIR / "cycle_02_progress.json", payload)
    atomic_json(
        CYCLE_DIR / f"checkpoint_{plan_position:02d}.json",
        payload,
    )
    if plan_position % 5 == 0:
        milestone = CYCLE_DIR / (
            f"cycle_02_milestone_{plan_position:02d}.json"
        )
        atomic_json(milestone, payload)
        atomic_text(
            milestone.with_suffix(".md"),
            "# Cycle 02 milestone\n\n"
            f"- Plan position: {plan_position}\n"
            f"- Completed: {len(success)}/{TOTAL_TARGET}\n"
            f"- Budget reservations: "
            f"{payload['cycle_budget_reservations']}/{CYCLE_LIMIT}\n"
            f"- Results SHA-256: `{payload['results_sha256']}`\n",
        )


def write_first_round_closure(
    stable_ids: list[str],
    success: set[str],
) -> None:
    first_ids = stable_ids[:FIRST_ROUND_SIZE]
    complete = all(record_id in success for record_id in first_ids)
    payload = {
        "status": "PASS" if complete else "FAIL",
        "cycle_id": CYCLE_ID,
        "first_round_target": FIRST_ROUND_SIZE,
        "first_round_success": sum(
            record_id in success for record_id in first_ids
        ),
        "first_round_success_ids_sha256": canonical_sha([
            record_id for record_id in first_ids
            if record_id in success
        ]),
        "results_sha256": sha256(ARTIFACT / "v3_1_results.jsonl"),
        "last_update": iso_now(),
    }
    atomic_json(CYCLE_DIR / "first_round_18k_closure.json", payload)
    atomic_text(
        CYCLE_DIR / "first_round_18k_closure.md",
        "# First-round 18K closure\n\n"
        f"Status: **{payload['status']}**\n\n"
        f"- Success: {payload['first_round_success']}/18000\n"
        f"- Results SHA-256: `{payload['results_sha256']}`\n",
    )
    if not complete:
        raise RuntimeError("first_round_closure_failed")


def write_final_outputs(
    stable_ids: list[str],
    sources: dict[str, dict[str, Any]],
    prompt: dict[str, Any],
    config: dict[str, Any],
    history: list[dict[str, Any]],
    budget: dict[str, Any],
) -> dict[str, Any]:
    latest = latest_rows(history)
    success = active_success_ids(
        stable_ids, sources, prompt, config, history
    )
    active = [latest[record_id] for record_id in stable_ids if record_id in success]
    concept_counts = [len(row.get("final_concepts") or []) for row in active]
    qc = collections.Counter(
        flag for row in active for flag in (row.get("audit_flags") or [])
    )
    modality = collections.Counter(
        str(row.get("modality") or "unknown") for row in active
    )
    split = collections.Counter(
        str(sources[row["stable_id"]].get("split") or "unknown")
        for row in active
    )
    audit_rows = [
        {
            "stable_id": row["stable_id"],
            "modality": row.get("modality"),
            "split": sources[row["stable_id"]].get("split"),
            "audit_flags": row.get("audit_flags") or [],
            "final_concepts": row.get("final_concepts") or [],
            "run_id": row.get("run_id"),
        }
        for row in active if row.get("audit_flags")
    ]
    atomic_text(
        CYCLE_DIR / "manual_audit_queue.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in audit_rows
        ),
    )
    manifest_rows = [
        {
            "stable_id": row["stable_id"],
            "source_caption_sha256": row["source_caption_sha256"],
            "result_run_id": row.get("run_id"),
            "final_concepts": row.get("final_concepts") or [],
            "quality_status": row.get("quality_status"),
            "audit_flags": row.get("audit_flags") or [],
        }
        for row in active
    ]
    final_manifest = CYCLE_DIR / "final_35k_active_manifest.jsonl"
    atomic_text(
        final_manifest,
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in manifest_rows
        ),
    )
    cycle_attempts = cycle_rows(history)
    payload = {
        "status": "PASS" if len(success) == TOTAL_TARGET else "FAIL",
        "cycle_id": CYCLE_ID,
        "target_unique_records": TOTAL_TARGET,
        "completed_unique_records": len(success),
        "cycle_result_rows": len(cycle_attempts),
        "cycle_budget_reservations": budget_reservation_count(),
        "lifetime_api_attempts": len(api_rows(history)),
        "technical_retry_count": max(
            0,
            len(cycle_attempts) - len({
                row.get("stable_id") for row in cycle_attempts
            }),
        ),
        "budget": budget,
        "concept_count_histogram": dict(sorted(
            collections.Counter(concept_counts).items()
        )),
        "cap8_ratio": (
            sum(value == 8 for value in concept_counts) / len(concept_counts)
            if concept_counts else 0.0
        ),
        "modality_distribution": dict(sorted(modality.items())),
        "split_distribution": dict(sorted(split.items())),
        "qc_flag_distribution": dict(sorted(qc.items())),
        "manual_audit_queue_count": len(audit_rows),
        "prompt_hash": prompt["prompt_hash"],
        "schema_hash": prompt["schema_hash"],
        "run_config_hash": config["run_config_hash"],
        "temperature": core.TEMPERATURE,
        "results_sha256": sha256(ARTIFACT / "v3_1_results.jsonl"),
        "resume_sha256": sha256(RESUME_PATH),
        "final_manifest_sha256": sha256(final_manifest),
        "last_update": iso_now(),
    }
    summary = CYCLE_DIR / "cycle_02_completion_summary.json"
    atomic_json(summary, payload)
    atomic_text(
        CYCLE_DIR / "cycle_02_completion_summary.md",
        "# MedicalSkill-CL Concept v3.1 35K closure\n\n"
        f"Status: **{payload['status']}**\n\n"
        f"- Unique records: {len(success)}/{TOTAL_TARGET}\n"
        f"- Cycle budget reservations: "
        f"{payload['cycle_budget_reservations']}/{CYCLE_LIMIT}\n"
        f"- Cycle result rows: {payload['cycle_result_rows']}\n"
        f"- Manual audit queue: {len(audit_rows)}\n"
        f"- Cap8 ratio: {payload['cap8_ratio']:.4%}\n"
        f"- Results SHA-256: `{payload['results_sha256']}`\n",
    )
    hashes = {}
    for path in sorted(CYCLE_DIR.rglob("*")):
        if path.is_file() and path.name != "cycle_02_file_hashes.json":
            hashes[str(path.relative_to(CYCLE_DIR))] = sha256(path)
    atomic_json(
        CYCLE_DIR / "cycle_02_file_hashes.json",
        {"files": hashes, "created_at": iso_now()},
    )
    return payload

async def run_wave(
    record_ids: list[str],
    entry: dict[str, Any],
    args: argparse.Namespace,
    shared_budget: AtomicCycleBudget,
    sources: dict[str, dict[str, Any]],
    prompt: dict[str, Any],
    schema: dict[str, Any],
    config: dict[str, Any],
) -> tuple[
    dict[str, dict[str, Any] | None],
    list[dict[str, Any]],
]:
    results_path = ARTIFACT / "v3_1_results.jsonl"
    history, latest, ordinals = core.load_history(results_path)
    writer = core.ResultWriter(results_path)
    tracker = core.StateTracker(
        RUNNER_STATE_PATH,
        results_path,
        record_ids,
        latest,
        sources,
        prompt,
        config,
        "formal_batch",
        CYCLE_ID,
        entry["batch_index"],
    )
    limiter = core.RollingWindowLimiter(
        args.requests_per_second,
        history,
        CYCLE_ID,
    )
    client = AsyncOpenAI(
        api_key=core.common.resolve_credential(
            os.environ
        )[0],
        base_url=core.BASE_URL,
        timeout=args.timeout,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    stop = asyncio.Event()
    run_id = (
        datetime.now(timezone.utc).strftime(
            "v3.1-cycle02-%Y%m%dT%H%M%SZ-"
        )
        + uuid.uuid4().hex[:12]
    )
    outcomes: dict[str, dict[str, Any] | None] = {}

    async def worker(record_id: str) -> None:
        if stop.is_set():
            return
        async with semaphore:
            if stop.is_set():
                return
            record_budget = RecordBudget(
                shared_budget, record_id
            )
            row = await core.request_one(
                client,
                limiter,
                record_budget,
                writer,
                tracker,
                sources[record_id],
                prompt,
                schema,
                config,
                run_id,
                ordinals[record_id],
                "formal_batch",
                CYCLE_ID,
                entry["batch_index"],
                args.max_retries,
                None,
                args.concurrency,
            )
            outcomes[record_id] = row
            if row and core.active_success(
                row,
                sources[record_id],
                prompt,
                config,
            ):
                await shared_budget.complete(record_id)
            else:
                await shared_budget.fail(record_id)
            if row and (
                row.get("http_status") in {400, 401, 403, 404}
                or row.get("quality_status")
                == "CONTRACT_VIOLATION"
            ):
                stop.set()

    before = len(history)
    await asyncio.gather(*(
        worker(record_id) for record_id in record_ids
    ))
    await client.close()
    final_history = first_policy.read_jsonl(results_path)
    return outcomes, final_history[before:]


async def run_batch(
    entry: dict[str, Any],
    args: argparse.Namespace,
    shared_budget: AtomicCycleBudget,
    stable_ids: list[str],
    sources: dict[str, dict[str, Any]],
    prompt: dict[str, Any],
    schema: dict[str, Any],
    config: dict[str, Any],
) -> tuple[str, list[str], str | None]:
    history = first_policy.read_jsonl(
        ARTIFACT / "v3_1_results.jsonl"
    )
    start_history_len = len(history)
    start_success = active_success_ids(
        stable_ids, sources, prompt, config, history
    )
    selected = entry["selected_ids"]
    pending = [
        record_id for record_id in selected
        if record_id not in start_success
    ]
    attempts_by_id = collections.Counter(
        row.get("stable_id")
        for row in cycle_rows(history)
    )
    technical_pending: list[str] = []
    hard_reason: str | None = None
    retry_round = 0

    while pending:
        snapshot = await shared_budget.snapshot()
        if snapshot["remaining_call_budget"] <= 0:
            technical_pending.extend(pending)
            break
        outcomes, new_rows = await run_wave(
            pending,
            entry,
            args,
            shared_budget,
            sources,
            prompt,
            schema,
            config,
        )
        hard_reason = hard_failure(new_rows)
        if hard_reason:
            break
        history = first_policy.read_jsonl(
            ARTIFACT / "v3_1_results.jsonl"
        )
        success = active_success_ids(
            stable_ids, sources, prompt, config, history
        )
        failed = [
            record_id for record_id in pending
            if record_id not in success
        ]
        if not failed:
            pending = []
            break
        retry_round += 1
        snapshot = await shared_budget.snapshot()
        attempts_by_id.update(
            row.get("stable_id") for row in new_rows
            if row.get("api_call_performed") is not False
        )
        retryable = [
            record_id for record_id in failed
            if is_retryable_failure(outcomes.get(record_id))
            and attempts_by_id[record_id] < 6
        ]
        terminal = [
            record_id for record_id in failed
            if record_id not in retryable
        ]
        technical_pending.extend(terminal)
        if snapshot["retry_margin"] <= 0:
            technical_pending.extend(retryable)
            pending = []
            break

        rate_rows = [
            row for row in new_rows
            if row.get("http_status") == 429
        ]
        if rate_rows:
            detector = first_policy.RateLimitBurstDetector()
            burst = None
            for row in new_rows:
                burst = detector.record(
                    row.get("http_status") == 429,
                    str(row.get("stable_id", "")),
                ) or burst
            delay = (
                first_policy.COOLDOWN_SECONDS
                if burst else max(
                    float(
                        row.get("retry_after_seconds") or 0
                    )
                    for row in rate_rows
                )
            )
            delay = max(delay, min(120, 2 ** retry_round))
            append_event(
                "RATE_LIMIT_WAIT",
                logical_batch=entry["logical_batch"],
                burst=burst,
                seconds=delay,
                records=len(rate_rows),
            )
            await asyncio.sleep(delay)
        else:
            await asyncio.sleep(
                min(120, 2 ** retry_round + 0.5)
            )
        pending = retryable

    history = first_policy.read_jsonl(
        ARTIFACT / "v3_1_results.jsonl"
    )
    success = active_success_ids(
        stable_ids, sources, prompt, config, history
    )
    if hard_reason:
        technical_pending.extend(
            record_id for record_id in selected
            if record_id not in success
        )
    technical_pending = list(dict.fromkeys(
        record_id for record_id in technical_pending
        if record_id not in success
    ))
    batch_success = sum(
        record_id in success for record_id in selected
    )
    status = (
        "HARD_STOP"
        if hard_reason
        else "COMPLETE"
        if batch_success == len(selected)
        else "PARTIAL"
    )
    snapshot = await shared_budget.snapshot()
    report = batch_report(
        entry,
        start_history_len,
        start_success,
        stable_ids,
        sources,
        prompt,
        config,
        history,
        status,
        technical_pending,
        snapshot,
    )
    if report["cap8_ratio"] >= 0.25:
        hard_reason = "cap8_ratio_at_or_above_25_percent"
        status = "HARD_STOP"
    write_resume(
        status,
        entry,
        stable_ids,
        sources,
        prompt,
        config,
        history,
        technical_pending,
        snapshot,
        hard_reason,
    )
    return status, technical_pending, hard_reason


async def execute(args: argparse.Namespace) -> int:
    preflight = json.loads(PREFLIGHT_JSON.read_text())
    if preflight.get("status") not in {
        "PASS", "PASS_WITH_EXPLAINED_DIFFERENCES"
    }:
        raise RuntimeError("passing_preflight_required")
    prompt, schema, config, _ = core.load_contract(ARTIFACT)
    sources, stable_ids = core.load_sources(
        ARTIFACT, config
    )
    history = first_policy.read_jsonl(
        ARTIFACT / "v3_1_results.jsonl"
    )
    existing_cycle_rows = cycle_rows(history)
    current_results_hash = sha256(
        ARTIFACT / "v3_1_results.jsonl"
    )
    if (
        not existing_cycle_rows
        and current_results_hash != preflight["results"]["sha256"]
    ):
        raise RuntimeError("results_changed_after_preflight")
    manifest = json.loads(TARGET_MANIFEST.read_text())
    if manifest.get("stable_ids_sha256") != sha256(
        ARTIFACT / "concept_35k_stable_ids.txt"
    ):
        raise RuntimeError("stable_ids_changed_after_preflight")
    if manifest.get("input_manifest_sha256") != sha256(
        ARTIFACT / "concept_35k_v3_1_input_manifest.jsonl"
    ):
        raise RuntimeError("input_manifest_changed_after_preflight")
    formal_success = formal_first_round_success(
        stable_ids, history
    )
    plan = batch_plan(stable_ids, formal_success)
    success = active_success_ids(
        stable_ids, sources, prompt, config, history
    )
    target = set(
        plan[0]["required_ids"]
        + [
            record_id
            for entry in plan[1:]
            for record_id in entry["required_ids"]
        ]
    )
    unresolved = target - success
    used = max(
        len(cycle_rows(history)),
        budget_reservation_count(),
    )
    if used > CYCLE_LIMIT:
        raise RuntimeError("cycle_02_budget_already_exceeded")
    budget = AtomicCycleBudget(used, unresolved)
    append_event(
        "CYCLE_STARTED",
        cycle_api_attempts=used,
        unique_records_completed=len(success),
        unresolved=len(unresolved),
    )
    COMMAND_PATH.write_text(
        " ".join(sys.argv) + "\n",
        encoding="utf-8",
    )

    for plan_position, entry in enumerate(plan, 1):
        history = first_policy.read_jsonl(
            ARTIFACT / "v3_1_results.jsonl"
        )
        success = active_success_ids(
            stable_ids, sources, prompt, config, history
        )
        if all(
            record_id in success
            for record_id in entry["selected_ids"]
        ):
            append_event(
                "BATCH_CACHE_COMPLETE",
                logical_batch=entry["logical_batch"],
            )
            if entry["phase"] == "first_round_closure":
                write_first_round_closure(stable_ids, success)
            continue
        append_event(
            "BATCH_STARTED",
            logical_batch=entry["logical_batch"],
            batch_index=entry["batch_index"],
            success_at_start=sum(
                record_id in success
                for record_id in entry["selected_ids"]
            ),
        )
        status, technical_pending, hard_reason = (
            await run_batch(
                entry,
                args,
                budget,
                stable_ids,
                sources,
                prompt,
                schema,
                config,
            )
        )
        append_event(
            "BATCH_FINISHED",
            logical_batch=entry["logical_batch"],
            status=status,
            technical_pending=len(technical_pending),
            hard_reason=hard_reason,
        )
        checkpoint_history = first_policy.read_jsonl(
            ARTIFACT / "v3_1_results.jsonl"
        )
        write_progress_checkpoint(
            plan_position,
            entry,
            stable_ids,
            sources,
            prompt,
            config,
            checkpoint_history,
            await budget.snapshot(),
        )
        if entry["phase"] == "first_round_closure":
            history = first_policy.read_jsonl(
                ARTIFACT / "v3_1_results.jsonl"
            )
            success = active_success_ids(
                stable_ids,
                sources,
                prompt,
                config,
                history,
            )
            if not all(
                record_id in success
                for record_id in stable_ids[:FIRST_ROUND_SIZE]
            ):
                return 2
            write_first_round_closure(stable_ids, success)
        if hard_reason or status != "COMPLETE" or technical_pending:
            return 2

    history = first_policy.read_jsonl(
        ARTIFACT / "v3_1_results.jsonl"
    )
    success = active_success_ids(
        stable_ids, sources, prompt, config, history
    )
    snapshot = await budget.snapshot()
    final_status = (
        "COMPLETE"
        if len(success) == TOTAL_TARGET
        else "STOPPED_INCOMPLETE"
    )
    write_resume(
        final_status,
        None,
        stable_ids,
        sources,
        prompt,
        config,
        history,
        [
            record_id for record_id in stable_ids
            if record_id not in success
        ],
        snapshot,
        None if final_status == "COMPLETE"
        else "remaining_records",
    )
    append_event(
        "CYCLE_FINISHED",
        status=final_status,
        unique_records_completed=len(success),
        cycle_api_attempts=snapshot["cycle_api_attempts"],
        completion_summary_status=final_status,
    )
    closure = write_final_outputs(
        stable_ids,
        sources,
        prompt,
        config,
        history,
        snapshot,
    )
    return 0 if final_status == "COMPLETE" else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument(
        "--run-cycle-02-35k",
        action="store_true",
    )
    parser.add_argument(
        "--confirm-cycle-02-35k",
        action="store_true",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--requests-per-second", type=float, default=2.0
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()
    if args.preflight:
        payload = write_preflight()
        print(json.dumps({
            "status": payload["status"],
            "api_called": False,
            "preflight_json": str(PREFLIGHT_JSON),
            "preflight_markdown": str(PREFLIGHT_MD),
            "target_manifest": str(TARGET_MANIFEST),
        }))
        return
    if not args.confirm_cycle_02_35k:
        raise SystemExit("explicit_cycle_02_confirmation_required")
    if args.concurrency < 1 or args.max_retries < 0:
        raise SystemExit("invalid_runtime_arguments")
    CYCLE_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.touch(exist_ok=True)
    with LOCK_PATH.open("r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(
                "cycle_02_already_running"
            ) from error
        code = asyncio.run(execute(args))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
