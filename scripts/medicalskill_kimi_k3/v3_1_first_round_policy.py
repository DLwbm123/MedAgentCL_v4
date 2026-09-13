#!/usr/bin/env python3
"""State, accounting, and reporting helpers for the frozen v3.1 first round."""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/medicalskill_cl_kimi_k3_v3_1_k2_6_routing"
RESULTS_PATH = ARTIFACT / "v3_1_results.jsonl"
RUNNER_RESUME_PATH = ARTIFACT / "v3_1_resume_state.json"
FIRST_ROUND_RESUME_PATH = ARTIFACT / "first_round_resume_state.json"
CACHE_INDEX_PATH = ARTIFACT / "first_round_cache_index.json"
PENDING_QUEUE_PATH = ARTIFACT / "first_round_pending_queue.json"
PERMANENT_FAILURE_PATH = ARTIFACT / "first_round_permanent_failures.json"
RUNTIME_EVENTS_PATH = ARTIFACT / "first_round_runtime_events.jsonl"
SUMMARY_PATH = ARTIFACT / "first_round_summary.json"
REPORT_DIR = ARTIFACT / "batch_reports"

FIRST_BATCH = 0
LAST_BATCH = 17
BATCH_SIZE = 1000
FIRST_ROUND_SIZE = 18_000
CYCLE_CALL_LIMIT = 18_000
MAX_PERMANENT_FAILURE_RATE = 0.01
MIN_SUCCESS_RATE = 0.99
MIN_CONTRACT_VALID_RATE = 0.995
COOLDOWN_SECONDS = 5 * 60 * 60 + 5 * 60
RUNTIME_POLICY_REVISION = 3
TEMPERATURE = 0.6
EXPECTED_HASHES = {
    "prompt_hash": "fb63d4b1963d5b9c93188ed65b7ec5677eeaf7ecb8e38b86bb73e640f516ae68",
    "schema_hash": "14830fcd9b334ac7ce24c17cd0704a373ccec247da03bf1435dad139a8d38aea",
    "run_config_hash": "382703cd787c9c49cce0ccdc2cda71615ccbb16f35820cc4a9ad290243d230a1",
    "stable_id_set_sha256": "476ab533eca00fa47610a9cc541aa3ed7c1183510a4d24b452d97f095ec09aec",
    "input_manifest_sha256": "dd5dd8aedead3ac751b85cab0fcaa1738bde22bee393a170b27284fa82e57831",
}
SUCCESS_QUALITY = {"PASS", "PASS_WITH_QC_FLAGS", "PASS_LEGACY_AUDITED"}
CYCLE_STATE_MODES = {
    "formal_batch",
    "recovery_queue",
    "batch0_quality_reclassification_revision_001",
}
TECHNICAL_TERMINAL = {
    "TECHNICAL_FAILED_AFTER_RETRY",
    "REJECTED_AFTER_RETRY",
}
RECOVERABLE_ERRORS = {
    "rate_limit",
    "timeout",
    "connection",
    "engine_overloaded",
    "http_500",
    "http_502",
    "http_503",
    "http_504",
}
SECRET_PATTERN = re.compile(
    r"(?:authorization\s*:|bearer\s+[a-z0-9._-]{12,}|"
    r"(?:api[_-]?key|moonshot_api_key|kimi_api_key)\s*[=:]\s*\S+)",
    re.I,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return digest.hexdigest()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists() or path.stat().st_size == 0:
        return default
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid_jsonl:{path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise RuntimeError(f"non_object_jsonl:{path}:{line_number}")
            rows.append(value)
    return rows


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temp_path.unlink(missing_ok=True)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def load_contract_files() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(ARTIFACT / "prompt_v3_1_manifest.json")
    config = read_json(ARTIFACT / "run_config_v3_1.json")
    if not isinstance(manifest, dict) or not isinstance(config, dict):
        raise RuntimeError("missing_frozen_contract")
    actual = {
        "prompt_hash": manifest.get("prompt_hash"),
        "schema_hash": manifest.get("schema_hash"),
        "run_config_hash": config.get("run_config_hash"),
        "stable_id_set_sha256": config.get("stable_id_set_sha256"),
        "input_manifest_sha256": config.get("input_manifest_sha256"),
    }
    mismatches = [
        key for key, expected in EXPECTED_HASHES.items()
        if actual.get(key) != expected
    ]
    if mismatches:
        raise RuntimeError(
            f"frozen_contract_hash_mismatch:{','.join(mismatches)}"
        )
    if sha256(ARTIFACT / "concept_35k_stable_ids.txt") != EXPECTED_HASHES[
        "stable_id_set_sha256"
    ]:
        raise RuntimeError("stable_id_file_hash_mismatch")
    if sha256(
        ARTIFACT / "concept_35k_v3_1_input_manifest.jsonl"
    ) != EXPECTED_HASHES["input_manifest_sha256"]:
        raise RuntimeError("input_manifest_file_hash_mismatch")
    return manifest, config


def load_stable_ids() -> list[str]:
    values = (
        ARTIFACT / "concept_35k_stable_ids.txt"
    ).read_text(encoding="utf-8").splitlines()
    if len(values) != 35_000 or len(set(values)) != 35_000:
        raise RuntimeError("stable_id_manifest_count_mismatch")
    return values


def first_round_ids(stable_ids: list[str] | None = None) -> list[str]:
    values = stable_ids if stable_ids is not None else load_stable_ids()
    return values[:FIRST_ROUND_SIZE]


def batch_ids(stable_ids: list[str], batch_index: int) -> list[str]:
    if not FIRST_BATCH <= batch_index <= LAST_BATCH:
        raise RuntimeError("batch_outside_frozen_first_round")
    start = batch_index * BATCH_SIZE
    return stable_ids[start:start + BATCH_SIZE]


def batch_index_map(stable_ids: list[str]) -> dict[str, int]:
    return {
        record_id: position // BATCH_SIZE
        for position, record_id in enumerate(stable_ids[:FIRST_ROUND_SIZE])
    }


def latest_rows(history: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in history:
        record_id = row.get("stable_id")
        if record_id:
            latest[str(record_id)] = row
    return latest


def effective_latest_rows(
    stable_ids: list[str],
    history: Iterable[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], set[int]]:
    """Resolve only started batches, with explicit cache inheritance."""
    positions = batch_index_map(stable_ids)
    cycle_history = [
        row for row in history
        if row.get("cycle_id") == "cycle_01"
        and row.get("mode") in CYCLE_STATE_MODES
        and row.get("stable_id") in positions
    ]
    started_batches = {
        positions[row["stable_id"]] for row in cycle_history
    }
    cycle_latest = latest_rows(cycle_history)
    all_latest = latest_rows(history)
    effective: dict[str, dict[str, Any]] = {}
    for record_id, batch_index in positions.items():
        if batch_index not in started_batches:
            continue
        if record_id in cycle_latest:
            effective[record_id] = cycle_latest[record_id]
            continue
        inherited = all_latest.get(record_id)
        if is_success(inherited):
            effective[record_id] = inherited
    return effective, started_batches


def is_success(row: dict[str, Any] | None) -> bool:
    concepts = (row or {}).get("final_concepts") or []
    return bool(row) and all((
        row.get("cache_status") == "active_v3_1_success",
        row.get("quality_status") in SUCCESS_QUALITY,
        row.get("output_source") in {"raw_model", "retry_model"},
        row.get("http_status") == 200,
        row.get("error_type") is None,
        row.get("prompt_hash") == EXPECTED_HASHES["prompt_hash"],
        row.get("schema_hash") == EXPECTED_HASHES["schema_hash"],
        row.get("run_config_hash") == EXPECTED_HASHES["run_config_hash"],
        row.get("reasoning_content_present") is False,
        int(row.get("reasoning_token_usage", 0) or 0) == 0,
        1 <= len(concepts) <= 8,
    ))


def is_contract_valid_http_200(row: dict[str, Any]) -> bool:
    concepts = (
        row.get("final_concepts")
        or row.get("retry_model_concepts")
        or row.get("raw_model_concepts")
        or []
    )
    return all((
        row.get("http_status") == 200,
        row.get("prompt_hash") == EXPECTED_HASHES["prompt_hash"],
        row.get("schema_hash") == EXPECTED_HASHES["schema_hash"],
        row.get("run_config_hash") == EXPECTED_HASHES["run_config_hash"],
        row.get("reasoning_content_present") is False,
        int(row.get("reasoning_token_usage", 0) or 0) == 0,
        row.get("finish_reason") != "length",
        1 <= len(concepts) <= 8,
    ))


def api_rows(history: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in history
        if row.get("api_call_performed") is not False
        and row.get("mode") in {"formal_batch", "recovery_queue"}
        and row.get("cycle_id") == "cycle_01"
    ]


def attempt_counts(history: Iterable[dict[str, Any]]) -> collections.Counter:
    return collections.Counter(
        row["stable_id"] for row in api_rows(history)
    )


def cycle_calls_used(history: Iterable[dict[str, Any]]) -> int:
    return len(api_rows(history))


def load_permanent_registry() -> dict[str, dict[str, Any]]:
    value = read_json(PERMANENT_FAILURE_PATH, {"records": {}})
    if not isinstance(value, dict) or not isinstance(
        value.get("records"), dict
    ):
        raise RuntimeError("invalid_permanent_failure_registry")
    return value["records"]


def save_permanent_registry(records: dict[str, dict[str, Any]]) -> None:
    atomic_write_json(
        PERMANENT_FAILURE_PATH,
        {
            "policy_revision": 2,
            "records": dict(sorted(records.items())),
            "count": len(records),
            "last_update": iso_now(),
        },
    )


def classify_records(
    stable_ids: list[str],
    history: list[dict[str, Any]],
    permanent: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    first_ids = stable_ids[:FIRST_ROUND_SIZE]
    first_set = set(first_ids)
    first_positions = {
        record_id: index for index, record_id in enumerate(first_ids)
    }
    latest, started_batches = effective_latest_rows(
        stable_ids, history
    )
    counts = attempt_counts(history)
    success: list[str] = []
    permanent_failed: list[str] = []
    pending: list[str] = []
    not_started: list[str] = []
    pending_details: dict[str, dict[str, Any]] = {}

    for record_id in first_ids:
        row = latest.get(record_id)
        if is_success(row):
            success.append(record_id)
        elif record_id in permanent:
            permanent_failed.append(record_id)
        elif (
            first_positions[record_id] // BATCH_SIZE
            not in started_batches
            or row is None
            or counts[record_id] == 0
        ):
            not_started.append(record_id)
        else:
            pending.append(record_id)
            pending_details[record_id] = {
                "stable_id": record_id,
                "batch_index": first_positions[record_id] // BATCH_SIZE,
                "lifetime_attempts": counts[record_id],
                "latest_http_status": row.get("http_status"),
                "latest_error_type": row.get("error_type"),
                "latest_quality_status": row.get("quality_status"),
                "latest_timestamp": row.get("timestamp"),
                "latest_temperature": row.get("temperature"),
                "latest_runtime_policy_revision": row.get(
                    "runtime_policy_revision"
                ),
                "recoverable": (
                    row.get("error_type") in RECOVERABLE_ERRORS
                    or row.get("http_status") == 429
                    or (
                        row.get("http_status") == 400
                        and row.get("temperature") == 1.0
                        and int(row.get(
                            "runtime_policy_revision", 0
                        ) or 0) == 2
                        and RUNTIME_POLICY_REVISION == 3
                        and TEMPERATURE == 0.6
                    )
                ),
            }

    leaked = [
        str(row.get("stable_id")) for row in history
        if row.get("stable_id") not in first_set
        and row.get("mode") in CYCLE_STATE_MODES
        and row.get("cycle_id") == "cycle_01"
    ]
    if leaked:
        raise RuntimeError(
            f"results_outside_first_round:{len(leaked)}"
        )
    return {
        "success": success,
        "permanent_failed": permanent_failed,
        "pending": pending,
        "not_started": not_started,
        "pending_details": pending_details,
        "counts": {
            "planned": FIRST_ROUND_SIZE,
            "success": len(success),
            "permanent_failed": len(permanent_failed),
            "pending": len(pending),
            "not_started": len(not_started),
        },
    }


def reconcile_permanent_failures(
    stable_ids: list[str],
    history: list[dict[str, Any]],
    permanent: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    first_ids = stable_ids[:FIRST_ROUND_SIZE]
    first_set = set(first_ids)
    first_positions = {
        record_id: index for index, record_id in enumerate(first_ids)
    }
    latest, _ = effective_latest_rows(stable_ids, history)
    counts = attempt_counts(history)
    changed = False
    for record_id, row in latest.items():
        if record_id not in first_set or is_success(row):
            continue
        reason = None
        if row.get("quality_status") in TECHNICAL_TERMINAL:
            reason = "contract_failure_after_single_retry"
        elif (
            counts[record_id] >= 6
            and row.get("http_status") != 429
            and row.get("error_type") in RECOVERABLE_ERRORS
        ):
            reason = "transport_lifetime_attempt_limit"
        if reason and record_id not in permanent:
            permanent[record_id] = {
                "stable_id": record_id,
                "batch_index": first_positions[record_id] // BATCH_SIZE,
                "reason": reason,
                "lifetime_attempts": counts[record_id],
                "latest_http_status": row.get("http_status"),
                "latest_error_type": row.get("error_type"),
                "timestamp": iso_now(),
            }
            changed = True
    if changed or not PERMANENT_FAILURE_PATH.exists():
        save_permanent_registry(permanent)
    return permanent


class RateLimitBurstDetector:
    """Detect concentrated 429 responses without inspecting sensitive payloads."""

    def __init__(self, recent_size: int = 20):
        self.recent_size = recent_size
        self.recent: collections.deque[tuple[bool, str, datetime]] = (
            collections.deque(maxlen=recent_size)
        )
        self.consecutive = 0

    def record(
        self,
        is_429: bool,
        stable_id: str,
        timestamp: datetime | None = None,
    ) -> str | None:
        timestamp = timestamp or utc_now()
        self.recent.append((is_429, stable_id, timestamp))
        self.consecutive = self.consecutive + 1 if is_429 else 0
        if self.consecutive >= 3:
            return "three_consecutive_429"
        if len(self.recent) >= 5 and sum(item[0] for item in self.recent) >= 5:
            return "five_429_in_recent_20"
        cutoff = timestamp - timedelta(minutes=5)
        distinct = {
            item[1] for item in self.recent
            if item[0] and item[2] >= cutoff
        }
        if len(distinct) >= 3:
            return "multiple_ids_429_in_short_window"
        return None


def detect_latest_burst(
    history: list[dict[str, Any]],
) -> dict[str, Any] | None:
    detector = RateLimitBurstDetector()
    decision = None
    for row in sorted(
        api_rows(history), key=lambda item: item.get("timestamp", "")
    ):
        try:
            timestamp = datetime.fromisoformat(row["timestamp"])
        except (KeyError, TypeError, ValueError):
            timestamp = utc_now()
        reason = detector.record(
            row.get("http_status") == 429,
            str(row.get("stable_id", "")),
            timestamp,
        )
        if reason:
            decision = {
                "reason": reason,
                "trigger_timestamp": timestamp.isoformat(),
                "cooldown_until": (
                    timestamp + timedelta(seconds=COOLDOWN_SECONDS)
                ).isoformat(),
            }
    return decision


def record_runtime_event(event_type: str, **values: Any) -> dict[str, Any]:
    event = {
        "event_type": event_type,
        "timestamp": iso_now(),
        "policy_revision": 2,
        **values,
    }
    append_jsonl(RUNTIME_EVENTS_PATH, event)
    return event


def runtime_events() -> list[dict[str, Any]]:
    return read_jsonl(RUNTIME_EVENTS_PATH)


def latest_cooldown() -> dict[str, Any] | None:
    events = runtime_events()
    starts = [
        event for event in events
        if event.get("event_type") == "COOLDOWN_STARTED"
    ]
    if not starts:
        return None
    latest = starts[-1]
    matching_end = [
        event for event in events
        if event.get("event_type") == "COOLDOWN_ENDED"
        and event.get("cooldown_id") == latest.get("cooldown_id")
    ]
    return None if matching_end else latest


def credential_leakage_hits(
    paths: Iterable[Path],
    credential_values: Iterable[str] = (),
) -> list[dict[str, Any]]:
    secrets = [value for value in credential_values if value]
    hits = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        reasons = []
        if SECRET_PATTERN.search(text):
            reasons.append("sensitive_header_or_key_pattern")
        if any(secret in text for secret in secrets):
            reasons.append("exact_runtime_credential")
        if reasons:
            hits.append({"path": str(path), "reasons": reasons})
    return hits


def batch_metrics(
    batch_index: int,
    stable_ids: list[str],
    history: list[dict[str, Any]],
    classification: dict[str, Any],
    concurrency_changes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    planned = batch_ids(stable_ids, batch_index)
    planned_set = set(planned)
    latest, _ = effective_latest_rows(stable_ids, history)
    success = [
        record_id for record_id in planned
        if record_id in set(classification["success"])
    ]
    permanent = [
        record_id for record_id in planned
        if record_id in set(classification["permanent_failed"])
    ]
    pending = [
        record_id for record_id in planned
        if record_id in set(classification["pending"])
    ]
    not_started = [
        record_id for record_id in planned
        if record_id in set(classification["not_started"])
    ]
    attempts = [
        row for row in api_rows(history)
        if row.get("stable_id") in planned_set
    ]
    http_counts = collections.Counter(
        str(row.get("http_status"))
        if row.get("http_status") is not None
        else "transport_error"
        for row in attempts
    )
    http_200 = [row for row in attempts if row.get("http_status") == 200]
    contract_valid_events = [
        row for row in http_200 if is_contract_valid_http_200(row)
    ]
    active_rows = [latest[record_id] for record_id in success]
    concept_counts = [
        len(row.get("final_concepts") or []) for row in active_rows
    ]
    usage = {
        key: sum(int(row.get(key, 0) or 0) for row in attempts)
        for key in (
            "prompt_tokens",
            "cached_tokens",
            "completion_tokens",
            "reasoning_token_usage",
        )
    }
    latencies = sorted(
        float(row.get("latency", 0) or 0) for row in attempts
    )
    cycle_used = cycle_calls_used(history)
    return {
        "status": (
            "COMPLETE"
            if not not_started and not pending
            and len(success) >= int(len(planned) * MIN_SUCCESS_RATE)
            and len(permanent) <= int(len(planned) * MAX_PERMANENT_FAILURE_RATE)
            else "IN_PROGRESS"
        ),
        "policy_revision": 2,
        "cycle_id": "cycle_01",
        "batch_index": batch_index,
        "planned_unique_records": len(planned),
        "success": len(success),
        "permanent_failed": len(permanent),
        "pending": len(pending),
        "not_started": len(not_started),
        "success_ids_sha256": hashlib.sha256(
            "".join(f"{value}\n" for value in success).encode()
        ).hexdigest(),
        "permanent_failure_ids": permanent,
        "pending_ids": pending,
        "api_attempts": len(attempts),
        "http_status_distribution": dict(sorted(http_counts.items())),
        "attempt_success_rate": (
            len(http_200) / len(attempts) if attempts else 1.0
        ),
        "unique_record_completion_rate": len(success) / len(planned),
        "contract_valid_rate_among_http_200": (
            len(contract_valid_events) / len(http_200)
            if http_200 else 1.0
        ),
        "permanent_failure_rate": len(permanent) / len(planned),
        "pending_rate": len(pending) / len(planned),
        "retry_amplification": (
            len(attempts) / len(planned) if planned else 0.0
        ),
        "usage": usage,
        "latency": {
            "mean_seconds": (
                sum(latencies) / len(latencies) if latencies else None
            ),
            "p95_seconds": (
                latencies[
                    min(len(latencies) - 1, int(len(latencies) * 0.95))
                ]
                if latencies else None
            ),
            "total_seconds": sum(latencies),
        },
        "concept_count_histogram": dict(
            sorted(collections.Counter(concept_counts).items())
        ),
        "cap8_count": sum(value == 8 for value in concept_counts),
        "cap8_rate": (
            sum(value == 8 for value in concept_counts)
            / len(concept_counts)
            if concept_counts else 0.0
        ),
        "qc_flag_counts": dict(sorted(collections.Counter(
            flag for row in active_rows
            for flag in (row.get("audit_flags") or [])
        ).items())),
        "concurrency_changes": concurrency_changes or [],
        "cooldowns": [
            event for event in runtime_events()
            if event.get("batch_index") == batch_index
            and event.get("event_type") in {
                "COOLDOWN_STARTED", "COOLDOWN_ENDED"
            }
        ],
        "cycle_calls_used": cycle_used,
        "cycle_calls_remaining": max(0, CYCLE_CALL_LIMIT - cycle_used),
        "results_hash": sha256(RESULTS_PATH),
        "runner_resume_hash": sha256(RUNNER_RESUME_PATH),
        "last_update": iso_now(),
    }


def write_batch_metrics(report: dict[str, Any]) -> Path:
    path = REPORT_DIR / (
        f"cycle_01_batch_{report['batch_index']:02d}_"
        "first_round_policy_revision_002.json"
    )
    atomic_write_json(path, report)
    atomic_write_json(
        path.with_suffix(".hash.json"),
        {
            "report_sha256": sha256(path),
            "results_sha256": sha256(RESULTS_PATH),
            "runner_resume_sha256": sha256(RUNNER_RESUME_PATH),
            "first_round_resume_sha256": sha256(
                FIRST_ROUND_RESUME_PATH
            ),
        },
    )
    return path


def write_runtime_artifacts(
    status: str,
    stable_ids: list[str],
    history: list[dict[str, Any]],
    permanent: dict[str, dict[str, Any]],
    current_batch: int | None,
    current_concurrency: int,
    stop_reason: str | None = None,
    next_record_id: str | None = None,
) -> dict[str, Any]:
    classification = classify_records(stable_ids, history, permanent)
    counts = classification["counts"]
    used = cycle_calls_used(history)
    results_hash = sha256(RESULTS_PATH)
    cache = {
        "policy_revision": 2,
        "first_round_batch_range": [FIRST_BATCH, LAST_BATCH],
        "success_ids": classification["success"],
        "success_count": counts["success"],
        "success_ids_sha256": hashlib.sha256(
            "".join(
                f"{value}\n" for value in classification["success"]
            ).encode()
        ).hexdigest(),
        "results_hash": results_hash,
        "last_update": iso_now(),
    }
    atomic_write_json(CACHE_INDEX_PATH, cache)
    atomic_write_json(
        PENDING_QUEUE_PATH,
        {
            "policy_revision": 2,
            "records": [
                classification["pending_details"][record_id]
                for record_id in classification["pending"]
            ],
            "count": counts["pending"],
            "last_update": iso_now(),
        },
    )
    state = {
        "status": status,
        "policy_revision": 2,
        "cycle_id": "cycle_01",
        "first_round_batch_range": [FIRST_BATCH, LAST_BATCH],
        "current_batch": current_batch,
        "current_concurrency": current_concurrency,
        "counts": counts,
        "cycle_api_calls": used,
        "cycle_api_call_limit": CYCLE_CALL_LIMIT,
        "cycle_remaining_calls": max(0, CYCLE_CALL_LIMIT - used),
        "stop_reason": stop_reason,
        "next_record_id": next_record_id,
        "cooldown": latest_cooldown(),
        "prompt_hash": EXPECTED_HASHES["prompt_hash"],
        "schema_hash": EXPECTED_HASHES["schema_hash"],
        "run_config_hash": EXPECTED_HASHES["run_config_hash"],
        "stable_id_set_sha256": EXPECTED_HASHES[
            "stable_id_set_sha256"
        ],
        "input_manifest_sha256": EXPECTED_HASHES[
            "input_manifest_sha256"
        ],
        "results_hash": results_hash,
        "runner_resume_hash": sha256(RUNNER_RESUME_PATH),
        "cache_index_hash": sha256(CACHE_INDEX_PATH),
        "pending_queue_hash": sha256(PENDING_QUEUE_PATH),
        "permanent_failure_registry_hash": sha256(
            PERMANENT_FAILURE_PATH
        ),
        "last_update": iso_now(),
    }
    atomic_write_json(FIRST_ROUND_RESUME_PATH, state)
    state["first_round_resume_hash"] = sha256(
        FIRST_ROUND_RESUME_PATH
    )
    return state


def write_summary(
    status: str,
    stable_ids: list[str],
    history: list[dict[str, Any]],
    permanent: dict[str, dict[str, Any]],
    stop_reason: str | None,
    current_batch: int | None,
) -> dict[str, Any]:
    classification = classify_records(stable_ids, history, permanent)
    counts = classification["counts"]
    attempts = api_rows(history)
    http_200 = [row for row in attempts if row.get("http_status") == 200]
    contract_valid = [
        row for row in http_200 if is_contract_valid_http_200(row)
    ]
    reports = []
    for batch_index in range(FIRST_BATCH, LAST_BATCH + 1):
        report = batch_metrics(
            batch_index, stable_ids, history, classification
        )
        reports.append({
            key: report[key] for key in (
                "batch_index",
                "status",
                "success",
                "permanent_failed",
                "pending",
                "not_started",
                "api_attempts",
                "unique_record_completion_rate",
                "attempt_success_rate",
                "contract_valid_rate_among_http_200",
            )
        })
    success_rate = counts["success"] / FIRST_ROUND_SIZE
    permanent_rate = counts["permanent_failed"] / FIRST_ROUND_SIZE
    contract_rate = (
        len(contract_valid) / len(http_200) if http_200 else 1.0
    )
    first_round_complete = (
        counts["not_started"] == 0
        and counts["pending"] == 0
        and counts["success"] + counts["permanent_failed"] == FIRST_ROUND_SIZE
        and success_rate >= MIN_SUCCESS_RATE
        and permanent_rate <= MAX_PERMANENT_FAILURE_RATE
        and contract_rate >= MIN_CONTRACT_VALID_RATE
    )
    summary = {
        "status": status,
        "first_round_complete": first_round_complete,
        "first_round_batch_range": [FIRST_BATCH, LAST_BATCH],
        "counts": counts,
        "success_rate": success_rate,
        "permanent_failure_rate": permanent_rate,
        "contract_valid_rate_among_http_200": contract_rate,
        "attempt_success_rate": (
            len(http_200) / len(attempts) if attempts else 1.0
        ),
        "retry_amplification": (
            len(attempts) / FIRST_ROUND_SIZE
        ),
        "total_api_calls": len(attempts),
        "http_status_distribution": dict(sorted(collections.Counter(
            str(row.get("http_status"))
            if row.get("http_status") is not None
            else "transport_error"
            for row in attempts
        ).items())),
        "permanent_failures": list(permanent.values()),
        "pending_ids": classification["pending"],
        "cooldown_periods": [
            event for event in runtime_events()
            if event.get("event_type") in {
                "COOLDOWN_STARTED", "COOLDOWN_ENDED"
            }
        ],
        "batch_status": reports,
        "stop_reason": stop_reason,
        "stop_batch": current_batch,
        "next_record_id": (
            classification["not_started"][0]
            if classification["not_started"]
            else (
                classification["pending"][0]
                if classification["pending"] else None
            )
        ),
        "results_hash": sha256(RESULTS_PATH),
        "runner_resume_hash": sha256(RUNNER_RESUME_PATH),
        "first_round_resume_hash": sha256(
            FIRST_ROUND_RESUME_PATH
        ),
        "cache_index_hash": sha256(CACHE_INDEX_PATH),
        "pending_queue_hash": sha256(PENDING_QUEUE_PATH),
        "permanent_failure_registry_hash": sha256(
            PERMANENT_FAILURE_PATH
        ),
        "next_round_authorized": False,
        "last_update": iso_now(),
    }
    atomic_write_json(SUMMARY_PATH, summary)
    return summary





