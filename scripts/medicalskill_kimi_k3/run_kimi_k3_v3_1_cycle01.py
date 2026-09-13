#!/usr/bin/env python3
"""Unattended, resumable controller for frozen v3.1 Batch 00-17."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/root/MedAgentCL_v4")
SCRIPT_DIR = ROOT / "scripts/medicalskill_kimi_k3"
sys.path.insert(0, str(SCRIPT_DIR))

import v3_1_first_round_policy as policy  # noqa: E402


PYTHON = Path("/root/anaconda3/envs/medagentcl_v4/bin/python")
RUNNER = SCRIPT_DIR / "run_kimi_k3_v3_1.py"
ARTIFACT = policy.ARTIFACT
LOCK_PATH = ARTIFACT / ".first_round_controller.lock"
CONTROLLER_LOG = (
    ARTIFACT / "controller_logs/first_round_unattended.log"
)
CONTROLLER_STATE = ARTIFACT / "first_round_controller_state.json"
MAX_CONCURRENCY = 8
MIN_CONCURRENCY = 1
INITIAL_CONCURRENCY = 8
POST_BURST_CONCURRENCY = 4
SUCCESS_WINDOW_FOR_INCREASE = 100


def log(message: str, **values: Any) -> None:
    CONTROLLER_LOG.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": policy.iso_now(),
        "message": message,
        **values,
    }
    line = json.dumps(event, ensure_ascii=False, sort_keys=True)
    with CONTROLLER_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(line, flush=True)


def latest_acceptance() -> dict[str, Any]:
    rows = policy.read_jsonl(
        ARTIFACT / "v3_1_pilot_acceptance_revisions.jsonl"
    )
    if not rows:
        raise RuntimeError("missing_acceptance_revision")
    return rows[-1]


def validate_authorization() -> None:
    revision = latest_acceptance()
    if revision.get("revision") != 6:
        raise RuntimeError("acceptance_revision_006_required")
    if set(revision.get("authorized_formal_batch_indices") or []) != set(
        range(policy.FIRST_BATCH, policy.LAST_BATCH + 1)
    ):
        raise RuntimeError("first_round_authorized_batch_set_mismatch")
    if revision.get("batch_18_and_later_authorized") is not False:
        raise RuntimeError("second_round_must_remain_blocked")
    if revision.get("temperature") != 0.6:
        raise RuntimeError("temperature_0_6_required")
    if revision.get("runtime_policy_revision") != 3:
        raise RuntimeError("runtime_policy_revision_003_required")


def write_controller_state(**values: Any) -> None:
    state = {
        "policy_revision": 3,
        "cycle_id": "cycle_01",
        "first_round_batch_range": [0, 17],
        "last_update": policy.iso_now(),
        **values,
    }
    policy.atomic_write_json(CONTROLLER_STATE, state)


def run_subprocess(
    command: list[str],
    label: str,
) -> tuple[int, list[str]]:
    log_path = (
        ARTIFACT / "controller_logs" / f"{label}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_lines = []
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n[{policy.iso_now()}] START {' '.join(command)}\n"
        )
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            output_lines.append(line.rstrip("\n"))
            handle.write(line)
            handle.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
        handle.write(f"[{policy.iso_now()}] EXIT {return_code}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return return_code, output_lines


def runner_command(
    args: argparse.Namespace,
    concurrency: int,
    batch_index: int | None = None,
    recovery_ids: list[str] | None = None,
) -> list[str]:
    command = [
        str(PYTHON),
        str(RUNNER),
        "--cycle-id",
        "cycle_01",
        "--cycle-start-batch",
        "0",
        "--concurrency",
        str(concurrency),
        "--requests-per-second",
        str(args.requests_per_second),
        "--timeout",
        str(args.timeout),
        "--max-retries",
        str(args.max_retries),
    ]
    if batch_index is not None:
        command.extend([
            "--formal-batch-index",
            str(batch_index),
            "--confirm-formal-batch",
        ])
    elif recovery_ids:
        command.append("--confirm-recovery")
        for record_id in recovery_ids:
            command.extend(["--recovery-stable-id", record_id])
        command.extend([
            "--max-api-calls",
            str(len(recovery_ids) * 2),
        ])
    else:
        raise RuntimeError("runner_selection_required")
    return command


def run_batch(
    args: argparse.Namespace,
    batch_index: int,
    concurrency: int,
) -> tuple[int, list[dict[str, Any]]]:
    before = policy.read_jsonl(policy.RESULTS_PATH)
    return_code, _ = run_subprocess(
        runner_command(
            args, concurrency, batch_index=batch_index
        ),
        f"cycle_01_batch_{batch_index:02d}",
    )
    after = policy.read_jsonl(policy.RESULTS_PATH)
    return return_code, after[len(before):]


def run_recovery(
    args: argparse.Namespace,
    record_ids: list[str],
    concurrency: int,
    label: str,
) -> tuple[int, list[dict[str, Any]]]:
    before = policy.read_jsonl(policy.RESULTS_PATH)
    return_code, _ = run_subprocess(
        runner_command(
            args, concurrency, recovery_ids=record_ids
        ),
        label,
    )
    after = policy.read_jsonl(policy.RESULTS_PATH)
    return return_code, after[len(before):]


def quota_hint(row: dict[str, Any]) -> bool:
    return bool(
        row.get("quota_exhausted")
        or row.get("quota_scope") in {
            "weekly", "monthly", "account", "cycle"
        }
    )


def hard_failure_reason(rows: list[dict[str, Any]]) -> str | None:
    for row in rows:
        status = row.get("http_status")
        if status in {400, 401, 404}:
            return f"http_{status}"
        if status == 403 and not quota_hint(row):
            return "non_quota_http_403"
        if row.get("quality_status") == "CONTRACT_VIOLATION":
            return "reasoning_or_output_contract_violation"
        if status == 200 and row.get("response_model_id") not in {
            None, "k3"
        }:
            return "returned_model_changed"
        if row.get("prompt_hash") != policy.EXPECTED_HASHES[
            "prompt_hash"
        ]:
            return "prompt_hash_changed"
        if row.get("schema_hash") != policy.EXPECTED_HASHES[
            "schema_hash"
        ]:
            return "schema_hash_changed"
        if row.get("run_config_hash") != policy.EXPECTED_HASHES[
            "run_config_hash"
        ]:
            return "run_config_hash_changed"
    return None


def quota_exhausted(
    history: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> bool:
    return (
        policy.cycle_calls_used(history) >= policy.CYCLE_CALL_LIMIT
        or any(quota_hint(row) for row in new_rows)
    )


def burst_reason(rows: list[dict[str, Any]]) -> str | None:
    detector = policy.RateLimitBurstDetector()
    decision = None
    for row in rows:
        decision = detector.record(
            row.get("http_status") == 429,
            str(row.get("stable_id", "")),
        ) or decision
    return decision


def schedule_cooldown(
    reason: str,
    batch_index: int | None,
    concurrency_before: int,
) -> tuple[str, datetime]:
    cooldown_id = uuid.uuid4().hex
    start = policy.utc_now()
    end = start.timestamp() + policy.COOLDOWN_SECONDS
    cooldown_until = datetime.fromtimestamp(
        end, tz=timezone.utc
    )
    policy.record_runtime_event(
        "COOLDOWN_STARTED",
        cooldown_id=cooldown_id,
        reason=reason,
        batch_index=batch_index,
        cooldown_seconds=policy.COOLDOWN_SECONDS,
        cooldown_until=cooldown_until.isoformat(),
        concurrency_before=concurrency_before,
        concurrency_after=max(
            MIN_CONCURRENCY, concurrency_before // 2
        ),
    )
    log(
        "cooldown_started",
        cooldown_id=cooldown_id,
        reason=reason,
        batch_index=batch_index,
        cooldown_until=cooldown_until.isoformat(),
    )
    return cooldown_id, cooldown_until


def wait_for_cooldown(
    args: argparse.Namespace,
    cooldown: dict[str, Any],
) -> None:
    cooldown_until = datetime.fromisoformat(
        cooldown["cooldown_until"]
    )
    remaining = max(
        0.0,
        cooldown_until.timestamp() - policy.utc_now().timestamp(),
    )
    write_controller_state(
        status="COOLDOWN_429",
        current_batch=cooldown.get("batch_index"),
        cooldown_id=cooldown.get("cooldown_id"),
        cooldown_until=cooldown["cooldown_until"],
        remaining_seconds=remaining,
    )
    if remaining:
        time.sleep(remaining)
    policy.record_runtime_event(
        "COOLDOWN_ENDED",
        cooldown_id=cooldown.get("cooldown_id"),
        batch_index=cooldown.get("batch_index"),
        scheduled_seconds=policy.COOLDOWN_SECONDS,
        actual_wait_seconds=remaining,
    )
    log(
        "cooldown_ended",
        cooldown_id=cooldown.get("cooldown_id"),
        actual_wait_seconds=remaining,
    )


def initialize_historical_cooldown(
    history: list[dict[str, Any]],
) -> None:
    if policy.runtime_events():
        return
    burst = policy.detect_latest_burst(history)
    if not burst:
        return
    cooldown_id = "historical-batch01-429"
    policy.record_runtime_event(
        "COOLDOWN_STARTED",
        cooldown_id=cooldown_id,
        reason=burst["reason"],
        batch_index=1,
        trigger_timestamp=burst["trigger_timestamp"],
        cooldown_seconds=policy.COOLDOWN_SECONDS,
        cooldown_until=burst["cooldown_until"],
        concurrency_before=INITIAL_CONCURRENCY,
        concurrency_after=POST_BURST_CONCURRENCY,
        reconstructed_from_append_only_results=True,
    )
    if datetime.fromisoformat(
        burst["cooldown_until"]
    ) <= policy.utc_now():
        policy.record_runtime_event(
            "COOLDOWN_ENDED",
            cooldown_id=cooldown_id,
            batch_index=1,
            scheduled_seconds=policy.COOLDOWN_SECONDS,
            actual_wait_seconds=policy.COOLDOWN_SECONDS,
            reconstructed_from_append_only_results=True,
        )


def safe_flush(
    status: str,
    stable_ids: list[str],
    history: list[dict[str, Any]],
    permanent: dict[str, dict[str, Any]],
    current_batch: int | None,
    concurrency: int,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    classification = policy.classify_records(
        stable_ids, history, permanent
    )
    next_record_id = (
        classification["not_started"][0]
        if classification["not_started"]
        else (
            classification["pending"][0]
            if classification["pending"] else None
        )
    )
    state = policy.write_runtime_artifacts(
        status,
        stable_ids,
        history,
        permanent,
        current_batch,
        concurrency,
        stop_reason,
        next_record_id,
    )
    for batch_index in range(
        policy.FIRST_BATCH, policy.LAST_BATCH + 1
    ):
        report = policy.batch_metrics(
            batch_index, stable_ids, history, classification
        )
        policy.write_batch_metrics(report)
    summary = policy.write_summary(
        status,
        stable_ids,
        history,
        permanent,
        stop_reason,
        current_batch,
    )
    exact_credentials = [
        os.environ.get("KIMI_API_KEY", ""),
        os.environ.get("MOONSHOT_API_KEY", ""),
    ]
    leakage = policy.credential_leakage_hits(
        [
            policy.RESULTS_PATH,
            policy.RUNNER_RESUME_PATH,
            policy.FIRST_ROUND_RESUME_PATH,
            policy.CACHE_INDEX_PATH,
            policy.PENDING_QUEUE_PATH,
            policy.PERMANENT_FAILURE_PATH,
            policy.SUMMARY_PATH,
        ],
        exact_credentials,
    )
    if leakage:
        raise RuntimeError("credential_leakage_detected")
    write_controller_state(
        status=status,
        current_batch=current_batch,
        current_concurrency=concurrency,
        stop_reason=stop_reason,
        cycle_api_calls=state["cycle_api_calls"],
        counts=state["counts"],
        next_record_id=next_record_id,
        first_round_summary_hash=policy.sha256(
            policy.SUMMARY_PATH
        ),
    )
    return summary


def recover_after_cooldown(
    args: argparse.Namespace,
    stable_ids: list[str],
    history: list[dict[str, Any]],
    permanent: dict[str, dict[str, Any]],
    concurrency: int,
) -> tuple[list[dict[str, Any]], int, str | None]:
    classification = policy.classify_records(
        stable_ids, history, permanent
    )
    recoverable = [
        record_id for record_id in classification["pending"]
        if classification["pending_details"][record_id]["recoverable"]
        and classification["pending_details"][record_id][
            "lifetime_attempts"
        ] < 6
    ]
    if not recoverable:
        return history, concurrency, None

    probe_id = recoverable[0]
    log(
        "recovery_probe_started",
        stable_id=probe_id,
        concurrency=1,
    )
    _, probe_rows = run_recovery(
        args, [probe_id], 1, "recovery_probe"
    )
    history = policy.read_jsonl(policy.RESULTS_PATH)
    if not probe_rows:
        if policy.cycle_calls_used(history) >= policy.CYCLE_CALL_LIMIT:
            return history, concurrency, "STOPPED_WEEKLY_QUOTA"
        return history, concurrency, "HARD_CONTRACT_FAILURE"
    reason = hard_failure_reason(probe_rows)
    if reason:
        return history, concurrency, f"HARD_CONTRACT_FAILURE:{reason}"
    if quota_exhausted(history, probe_rows):
        return history, concurrency, "STOPPED_WEEKLY_QUOTA"
    if probe_rows[-1].get("http_status") == 429:
        schedule_cooldown(
            "recovery_probe_429",
            probe_rows[-1].get("batch_index"),
            concurrency,
        )
        return (
            history,
            max(MIN_CONCURRENCY, concurrency // 2),
            "COOLDOWN_429",
        )
    if not policy.is_success(probe_rows[-1]):
        return history, concurrency, None

    remaining = recoverable[1:]
    if remaining:
        log(
            "recovery_queue_resumed",
            records=len(remaining),
            concurrency=max(MIN_CONCURRENCY, concurrency),
        )
        _, recovery_rows = run_recovery(
            args,
            remaining,
            max(MIN_CONCURRENCY, concurrency),
            "recovery_queue",
        )
        history = policy.read_jsonl(policy.RESULTS_PATH)
        reason = hard_failure_reason(recovery_rows)
        if reason:
            return history, concurrency, (
                f"HARD_CONTRACT_FAILURE:{reason}"
            )
        if quota_exhausted(history, recovery_rows):
            return history, concurrency, "STOPPED_WEEKLY_QUOTA"
        burst = burst_reason(recovery_rows)
        if burst:
            schedule_cooldown(
                burst,
                recovery_rows[-1].get("batch_index")
                if recovery_rows else None,
                concurrency,
            )
            return (
                history,
                max(MIN_CONCURRENCY, concurrency // 2),
                "COOLDOWN_429",
            )
    return history, concurrency, None


def next_batch_with_not_started(
    stable_ids: list[str],
    classification: dict[str, Any],
) -> int | None:
    if not classification["not_started"]:
        return None
    positions = {
        record_id: position
        for position, record_id in enumerate(
            stable_ids[:policy.FIRST_ROUND_SIZE]
        )
    }
    return min(
        positions[record_id] // policy.BATCH_SIZE
        for record_id in classification["not_started"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirm-first-round-batches-00-17",
        action="store_true",
    )
    parser.add_argument(
        "--requests-per-second", type=float, default=2.0
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()
    if not args.confirm_first_round_batches_00_17:
        raise SystemExit("explicit_first_round_confirmation_required")
    if args.max_retries < 0:
        raise SystemExit("invalid_retry_limit")

    policy.load_contract_files()
    validate_authorization()
    stable_ids = policy.load_stable_ids()
    LOCK_PATH.touch(exist_ok=True)
    with LOCK_PATH.open("r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(
                "first_round_controller_already_running"
            ) from error

        history = policy.read_jsonl(policy.RESULTS_PATH)
        initialize_historical_cooldown(history)
        permanent = policy.load_permanent_registry()
        permanent = policy.reconcile_permanent_failures(
            stable_ids, history, permanent
        )
        concurrency = (
            POST_BURST_CONCURRENCY
            if policy.detect_latest_burst(history)
            else INITIAL_CONCURRENCY
        )
        ready_classification = policy.classify_records(
            stable_ids, history, permanent
        )
        initial_recovery_due = bool(
            ready_classification["pending"]
        )
        safe_flush(
            "READY",
            stable_ids,
            history,
            permanent,
            1,
            concurrency,
        )

        while True:
            history = policy.read_jsonl(policy.RESULTS_PATH)
            permanent = policy.reconcile_permanent_failures(
                stable_ids, history, permanent
            )
            classification = policy.classify_records(
                stable_ids, history, permanent
            )
            existing_hard = next((
                details for details in
                classification["pending_details"].values()
                if details.get("latest_http_status") in {400, 401, 404}
                and not details.get("recoverable")
            ), None)
            if existing_hard:
                reason = f"existing_http_{existing_hard['latest_http_status']}"
                safe_flush(
                    "HARD_CONTRACT_FAILURE", stable_ids, history,
                    permanent, existing_hard.get("batch_index"),
                    concurrency, reason,
                )
                raise SystemExit(2)
            permanent_rate = (
                classification["counts"]["permanent_failed"]
                / policy.FIRST_ROUND_SIZE
            )
            if permanent_rate > policy.MAX_PERMANENT_FAILURE_RATE:
                safe_flush(
                    "HARD_CONTRACT_FAILURE",
                    stable_ids,
                    history,
                    permanent,
                    None,
                    concurrency,
                    "global_permanent_failure_rate_above_1_percent",
                )
                raise SystemExit(2)
            if policy.cycle_calls_used(history) >= policy.CYCLE_CALL_LIMIT:
                safe_flush(
                    "STOPPED_WEEKLY_QUOTA",
                    stable_ids,
                    history,
                    permanent,
                    None,
                    concurrency,
                    "local_cycle_call_limit_reached",
                )
                return

            cooldown = policy.latest_cooldown()
            if initial_recovery_due and not cooldown:
                initial_recovery_due = False
                history, concurrency, recovery_status = (
                    recover_after_cooldown(
                        args,
                        stable_ids,
                        history,
                        permanent,
                        POST_BURST_CONCURRENCY,
                    )
                )
                if recovery_status == "COOLDOWN_429":
                    safe_flush(
                        "COOLDOWN_429", stable_ids, history,
                        permanent, 1, concurrency,
                        "initial_recovery_probe_or_queue_429",
                    )
                    continue
                if recovery_status == "STOPPED_WEEKLY_QUOTA":
                    safe_flush(
                        recovery_status, stable_ids, history,
                        permanent, 1, concurrency,
                        "weekly_quota_during_initial_recovery",
                    )
                    return
                if recovery_status and recovery_status.startswith(
                    "HARD_CONTRACT_FAILURE"
                ):
                    safe_flush(
                        "HARD_CONTRACT_FAILURE", stable_ids,
                        history, permanent, 1, concurrency,
                        recovery_status,
                    )
                    raise SystemExit(2)

            cooldown = policy.latest_cooldown()
            if cooldown:
                wait_for_cooldown(args, cooldown)
                history = policy.read_jsonl(policy.RESULTS_PATH)
                history, concurrency, recovery_status = (
                    recover_after_cooldown(
                        args,
                        stable_ids,
                        history,
                        permanent,
                        max(
                            MIN_CONCURRENCY,
                            int(
                                cooldown.get(
                                    "concurrency_after",
                                    POST_BURST_CONCURRENCY,
                                )
                            ),
                        ),
                    )
                )
                if recovery_status == "COOLDOWN_429":
                    safe_flush(
                        "COOLDOWN_429",
                        stable_ids,
                        history,
                        permanent,
                        None,
                        concurrency,
                        "recovery_probe_or_queue_429",
                    )
                    continue
                if recovery_status == "STOPPED_WEEKLY_QUOTA":
                    safe_flush(
                        recovery_status,
                        stable_ids,
                        history,
                        permanent,
                        None,
                        concurrency,
                        "weekly_quota_during_recovery",
                    )
                    return
                if recovery_status and recovery_status.startswith(
                    "HARD_CONTRACT_FAILURE"
                ):
                    safe_flush(
                        "HARD_CONTRACT_FAILURE",
                        stable_ids,
                        history,
                        permanent,
                        None,
                        concurrency,
                        recovery_status,
                    )
                    raise SystemExit(2)

            classification = policy.classify_records(
                stable_ids, history, permanent
            )
            batch_index = next_batch_with_not_started(
                stable_ids, classification
            )
            if batch_index is None:
                # A normal completion cannot hide unresolved records. Give
                # the bounded recovery queue another pass before accepting.
                if classification["pending"]:
                    before_count = len(history)
                    history, concurrency, recovery_status = (
                        recover_after_cooldown(
                            args,
                            stable_ids,
                            history,
                            permanent,
                            max(MIN_CONCURRENCY, concurrency),
                        )
                    )
                    if recovery_status == "COOLDOWN_429":
                        safe_flush(
                            "COOLDOWN_429", stable_ids, history,
                            permanent, None, concurrency,
                            "final_recovery_429",
                        )
                        continue
                    if recovery_status == "STOPPED_WEEKLY_QUOTA":
                        safe_flush(
                            recovery_status, stable_ids, history,
                            permanent, None, concurrency,
                            "weekly_quota_during_final_recovery",
                        )
                        return
                    if recovery_status and recovery_status.startswith(
                        "HARD_CONTRACT_FAILURE"
                    ):
                        safe_flush(
                            "HARD_CONTRACT_FAILURE", stable_ids,
                            history, permanent, None, concurrency,
                            recovery_status,
                        )
                        raise SystemExit(2)
                    if len(history) == before_count:
                        safe_flush(
                            "HARD_CONTRACT_FAILURE", stable_ids,
                            history, permanent, None, concurrency,
                            "unrecoverable_pending_records",
                        )
                        raise SystemExit(2)
                    continue
                summary = safe_flush(
                    "FIRST_ROUND_COMPLETE",
                    stable_ids,
                    history,
                    permanent,
                    None,
                    concurrency,
                )
                if not summary["first_round_complete"]:
                    safe_flush(
                        "HARD_CONTRACT_FAILURE",
                        stable_ids,
                        history,
                        permanent,
                        None,
                        concurrency,
                        "first_round_acceptance_not_met",
                    )
                    raise SystemExit(2)
                return

            safe_flush(
                "RUNNING",
                stable_ids,
                history,
                permanent,
                batch_index,
                concurrency,
            )
            log(
                "batch_started",
                batch_index=batch_index,
                concurrency=concurrency,
                cycle_calls_used=policy.cycle_calls_used(history),
            )
            _, new_rows = run_batch(
                args, batch_index, concurrency
            )
            history = policy.read_jsonl(policy.RESULTS_PATH)
            hard_reason = hard_failure_reason(new_rows)
            if hard_reason:
                safe_flush(
                    "HARD_CONTRACT_FAILURE",
                    stable_ids,
                    history,
                    permanent,
                    batch_index,
                    concurrency,
                    hard_reason,
                )
                raise SystemExit(2)
            if quota_exhausted(history, new_rows):
                safe_flush(
                    "STOPPED_WEEKLY_QUOTA",
                    stable_ids,
                    history,
                    permanent,
                    batch_index,
                    concurrency,
                    "weekly_quota_or_cycle_limit",
                )
                return

            burst = burst_reason(new_rows)
            count_429 = sum(
                row.get("http_status") == 429 for row in new_rows
            )
            success_count = sum(
                policy.is_success(row) for row in new_rows
            )
            if burst:
                _, _ = schedule_cooldown(
                    burst, batch_index, concurrency
                )
                concurrency = max(
                    MIN_CONCURRENCY, concurrency // 2
                )
                safe_flush(
                    "COOLDOWN_429",
                    stable_ids,
                    history,
                    permanent,
                    batch_index,
                    concurrency,
                    burst,
                )
                continue
            if count_429:
                old_concurrency = concurrency
                concurrency = max(
                    MIN_CONCURRENCY,
                    int(concurrency * 0.8),
                )
                policy.record_runtime_event(
                    "CONCURRENCY_CHANGED",
                    reason="isolated_429",
                    batch_index=batch_index,
                    before=old_concurrency,
                    after=concurrency,
                )
            elif success_count >= SUCCESS_WINDOW_FOR_INCREASE:
                old_concurrency = concurrency
                concurrency = min(
                    MAX_CONCURRENCY, concurrency + 1
                )
                if concurrency != old_concurrency:
                    policy.record_runtime_event(
                        "CONCURRENCY_CHANGED",
                        reason="successful_window",
                        batch_index=batch_index,
                        before=old_concurrency,
                        after=concurrency,
                    )

            summary = safe_flush(
                "BATCH_COMPLETE",
                stable_ids,
                history,
                permanent,
                batch_index,
                concurrency,
            )
            log(
                "batch_finished",
                batch_index=batch_index,
                new_api_rows=len(new_rows),
                new_success_rows=success_count,
                new_429=count_429,
                cycle_calls_used=summary["total_api_calls"],
                current_concurrency=concurrency,
            )


if __name__ == "__main__":
    main()



