#!/usr/bin/env python3
"""Asynchronous, resumable Kimi K3 runner restricted to the frozen 2,000-ID pilot."""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import os
import random
import re
import signal
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError


ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/medicalskill_cl_kimi_k3_pilot"
MODEL = "kimi-k3"
REASONING_EFFORT = "low"
PILOT_LIMIT = 2_000
MAX_COMPLETION_TOKENS = 512
DEFAULT_BASE_URL = "https://api.kimi.com/coding/v1"
EXPECTED_HOSTNAME = "api.kimi.com"
EXPECTED_BASE_PATH = "/coding/v1"
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
GENERIC_ONLY = {
    "disease", "disease process", "pathology", "abnormality", "finding",
    "medical image", "anatomical association", "region of interest",
}
ARTIFACT_PATTERNS = re.compile(
    r"\b(?:roi|region of interest|bounding box|bbox|area ratio|highlighted region|colored box|central position|cross-sectional view)\b",
    re.I,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"Non-object JSON at {path}:{line_number}")
            yield value


def write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def resolve_credential(environment: Mapping[str, str]) -> tuple[str, str]:
    kimi = environment.get("KIMI_API_KEY", "")
    moonshot = environment.get("MOONSHOT_API_KEY", "")
    if kimi and moonshot and kimi != moonshot:
        raise RuntimeError("ambiguous_credentials")
    if kimi:
        return kimi, "KIMI_API_KEY"
    if moonshot:
        return moonshot, "MOONSHOT_API_KEY"
    raise RuntimeError("credential_missing")


def resolve_base_url(environment: Mapping[str, str]) -> dict[str, str]:
    if environment.get("KIMI_BASE_URL"):
        raw, source = environment["KIMI_BASE_URL"], "KIMI_BASE_URL"
    elif environment.get("MOONSHOT_BASE_URL"):
        raw, source = environment["MOONSHOT_BASE_URL"], "MOONSHOT_BASE_URL"
    else:
        raw, source = DEFAULT_BASE_URL, "default"
    normalized = raw.strip().rstrip("/")
    parsed = urlparse(normalized)
    if (
        parsed.scheme != "https" or parsed.hostname != EXPECTED_HOSTNAME
        or parsed.path.rstrip("/") != EXPECTED_BASE_PATH
        or parsed.params or parsed.query or parsed.fragment
    ):
        raise RuntimeError("unsupported_base_url")
    return {
        "base_url": DEFAULT_BASE_URL,
        "base_url_source": source,
        "scheme": "https",
        "hostname": EXPECTED_HOSTNAME,
        "base_path": EXPECTED_BASE_PATH,
        "chat_completions_url": DEFAULT_BASE_URL + "/chat/completions",
    }


def normalize_concept(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return " ".join(value.split()).strip(" ;,.").casefold()


def validate_content(raw: str) -> list[str]:
    if "```" in raw:
        raise ValueError("markdown_code_fence")
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"concepts"}:
        raise ValueError("schema_object")
    concepts = value["concepts"]
    if not isinstance(concepts, list):
        raise ValueError("schema_concepts_array")
    if not 1 <= len(concepts) <= 8:
        raise ValueError("concept_count")
    normalized = []
    seen = set()
    for item in concepts:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("concept_nonempty_string")
        clean = normalize_concept(item)
        if not clean:
            raise ValueError("concept_empty_after_normalization")
        if clean in seen:
            raise ValueError("concept_duplicate")
        normalized.append(clean)
        seen.add(clean)
    if any(ARTIFACT_PATTERNS.search(item) for item in normalized):
        raise ValueError("obvious_artifact")
    if all(item in GENERIC_ONLY for item in normalized):
        raise ValueError("all_generic")
    return normalized


class AdaptiveLimiter:
    def __init__(self, requests_per_second: float):
        self.minimum_interval = 1.0 / max(0.05, requests_per_second)
        self.current_interval = self.minimum_interval
        self.next_time = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_time - now)
            if delay:
                await asyncio.sleep(delay)
            self.next_time = max(self.next_time, time.monotonic()) + self.current_interval

    def limited(self) -> None:
        self.current_interval = min(30.0, self.current_interval * 1.8)

    def success(self) -> None:
        self.current_interval = max(self.minimum_interval, self.current_interval * 0.98)


class AttemptBudget:
    def __init__(self, used: int, maximum: int):
        self.used = used
        self.maximum = maximum
        self.lock = asyncio.Lock()

    async def reserve(self) -> bool:
        async with self.lock:
            if self.used >= self.maximum:
                return False
            self.used += 1
            return True


class ResultWriter:
    def __init__(self, path: Path):
        self.path = path
        self.lock = asyncio.Lock()

    async def append(self, value: dict[str, Any]) -> None:
        async with self.lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())


def retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None)
    return {
        "prompt": int(getattr(usage, "prompt_tokens", 0) or 0),
        "cached": int(getattr(details, "cached_tokens", 0) or 0),
        "completion": int(getattr(usage, "completion_tokens", 0) or 0),
        "total": int(getattr(usage, "total_tokens", 0) or 0),
    }


def redact_secrets(text: str, environment: Mapping[str, str]) -> str:
    for name in ("KIMI_API_KEY", "MOONSHOT_API_KEY"):
        secret = environment.get(name, "")
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)(authorization|api[_ -]?key|bearer)\s*[:=]?\s*[^\s,;]+", r"\1=[REDACTED]", text)
    return re.sub(r"(?:sk-|kimi-)[A-Za-z0-9_-]{12,}", "[REDACTED]", text)[:1_000]


def load_history(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], collections.Counter]:
    history = list(read_jsonl(path)) if path.exists() else []
    latest: dict[str, dict[str, Any]] = {}
    ordinals: collections.Counter = collections.Counter()
    for row in history:
        row_id = str(row["stable_record_id"])
        latest[row_id] = row
        ordinal = int(row.get("attempt_ordinal", row.get("attempt_count", 1)) or 1)
        ordinals[row_id] = max(ordinals[row_id], ordinal)
    return history, latest, ordinals


def active_success(cached: dict[str, Any] | None, source: dict[str, Any], prompt: dict[str, Any], endpoint: dict[str, str]) -> bool:
    if not cached:
        return False
    return all((
        cached.get("status") == "success",
        cached.get("contract_status") == "PASS",
        cached.get("stable_record_id") == source["stable_record_id"],
        cached.get("source_caption_sha256") == source["source_caption_sha256"],
        cached.get("prompt_hash") == prompt["prompt_hash"],
        cached.get("schema_hash") == prompt["schema_hash"],
        cached.get("model_returned") == MODEL,
        cached.get("endpoint_hostname") == endpoint["hostname"],
        cached.get("endpoint_base_path") == endpoint["base_path"],
        cached.get("max_completion_tokens") == MAX_COMPLETION_TOKENS,
        cached.get("finish_reason") not in {"length", "max_tokens"},
        1 <= len(cached.get("parsed_concepts") or []) <= 8,
    ))


def build_result(
    row: dict[str, Any], prompt: dict[str, Any], endpoint: dict[str, str], credential_source: str,
    run_id: str, attempt_ordinal: int, model_returned: str | None, raw: str,
    concepts: list[str], usage: dict[str, int], latency: float, status: str,
    contract_status: str, error: str | None, finish_reason: str | None, retry_reason: str | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "attempt_id": f"{run_id}:{row['stable_record_id']}:attempt_{attempt_ordinal:04d}",
        "attempt_ordinal": attempt_ordinal,
        "stable_record_id": row["stable_record_id"],
        "source_caption_sha256": row["source_caption_sha256"],
        "raw_model_response": raw,
        "parsed_concepts": concepts,
        "model_requested": MODEL,
        "model_returned": model_returned,
        "reasoning_effort": REASONING_EFFORT,
        "prompt_version": prompt["prompt_version"],
        "prompt_hash": prompt["prompt_hash"],
        "schema_version": prompt["schema_version"],
        "schema_hash": prompt["schema_hash"],
        "temperature": 1.0,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "structured_output_mode": "json_schema_strict",
        "endpoint_base_url": endpoint["base_url"],
        "endpoint_scheme": endpoint["scheme"],
        "endpoint_hostname": endpoint["hostname"],
        "endpoint_base_path": endpoint["base_path"],
        "base_url_source": endpoint["base_url_source"],
        "credential_present": True,
        "credential_source": credential_source,
        "token_usage": usage,
        "latency_seconds": round(latency, 6),
        "attempt_count": attempt_ordinal,
        "retry_reasons": [retry_reason] if retry_reason else [],
        "status": status,
        "contract_status": contract_status,
        "error_category": error,
        "finish_reason": finish_reason,
        "created_timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def request_one(
    client: AsyncOpenAI, limiter: AdaptiveLimiter, budget: AttemptBudget, writer: ResultWriter,
    row: dict[str, Any], prompt: dict[str, Any], schema: dict[str, Any], endpoint: dict[str, str],
    credential_source: str, run_id: str, max_retries: int, prior_attempts: int,
) -> dict[str, Any] | None:
    for local_attempt in range(1, max_retries + 2):
        if not await budget.reserve():
            return None
        attempt_ordinal = prior_attempts + local_attempt
        await limiter.wait()
        started = time.monotonic()
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": prompt["system_prompt"]},
                    {"role": "user", "content": prompt["user_template"].format(caption=row["source_caption"])},
                ],
                reasoning_effort=REASONING_EFFORT,
                temperature=1.0,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "medical_concepts", "strict": True, "schema": schema},
                },
            )
        except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as error:
            latency = time.monotonic() - started
            status_code = getattr(error, "status_code", None)
            retryable = isinstance(error, (RateLimitError, APITimeoutError, APIConnectionError)) or status_code in RETRYABLE_STATUS
            category = (
                "rate_limit" if isinstance(error, RateLimitError) or status_code == 429
                else "timeout" if isinstance(error, APITimeoutError)
                else "connection" if isinstance(error, APIConnectionError)
                else f"http_{status_code or 'error'}"
            )
            result = build_result(
                row, prompt, endpoint, credential_source, run_id, attempt_ordinal, None, "", [],
                {"prompt": 0, "cached": 0, "completion": 0, "total": 0}, latency,
                "retryable_api_error" if retryable else "contract_or_api_rejected", "FAIL",
                redact_secrets(str(error), os.environ), None, category if retryable else None,
            )
            await writer.append(result)
            if not retryable or local_attempt > max_retries:
                return result
            if category == "rate_limit":
                limiter.limited()
            server_delay = retry_after_seconds(error)
            await asyncio.sleep(server_delay if server_delay is not None else min(60.0, 2 ** (local_attempt - 1) + random.random()))
            continue
        except Exception as error:
            result = build_result(
                row, prompt, endpoint, credential_source, run_id, attempt_ordinal, None, "", [],
                {"prompt": 0, "cached": 0, "completion": 0, "total": 0}, time.monotonic() - started,
                "runner_error", "FAIL", redact_secrets(str(error), os.environ), None, None,
            )
            await writer.append(result)
            return result

        latency = time.monotonic() - started
        returned_model = str(getattr(response, "model", "") or "")
        content = str(response.choices[0].message.content or "")
        finish_reason = str(response.choices[0].finish_reason or "")
        usage = usage_dict(response)
        if returned_model != MODEL:
            result = build_result(row, prompt, endpoint, credential_source, run_id, attempt_ordinal, returned_model, content, [], usage, latency, "contract_model_mismatch", "FAIL", f"requested={MODEL}, returned={returned_model}", finish_reason, None)
        elif finish_reason in {"length", "max_tokens"}:
            result = build_result(row, prompt, endpoint, credential_source, run_id, attempt_ordinal, returned_model, content, [], usage, latency, "contract_finish_reason_length", "FAIL", "finish_reason_length_at_fixed_512", finish_reason, None)
        else:
            try:
                concepts = validate_content(content)
            except (ValueError, json.JSONDecodeError) as error:
                result = build_result(row, prompt, endpoint, credential_source, run_id, attempt_ordinal, returned_model, content, [], usage, latency, "contract_invalid_output", "FAIL", str(error), finish_reason, None)
            else:
                limiter.success()
                result = build_result(row, prompt, endpoint, credential_source, run_id, attempt_ordinal, returned_model, content, concepts, usage, latency, "success", "PASS", None, finish_reason, None)
        await writer.append(result)
        return result
    raise AssertionError("unreachable")


def load_contract(artifact: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prompt = json.loads((artifact / "prompt_manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((artifact / "kimi_prompt_schema.json").read_text(encoding="utf-8"))
    selection = json.loads((artifact / "pilot_selection_manifest.json").read_text(encoding="utf-8"))
    if prompt["model"] != MODEL or prompt["reasoning_effort"] != REASONING_EFFORT:
        raise RuntimeError("frozen_model_reasoning_contract_mismatch")
    if sha_text(canonical_json(schema)) != prompt["schema_hash"]:
        raise RuntimeError("frozen_schema_hash_mismatch")
    prompt_contract = {key: prompt[key] for key in ("prompt_version", "system_prompt", "user_template", "model", "reasoning_effort", "temperature", "max_completion_tokens")}
    if sha_text(canonical_json(prompt_contract)) != prompt["prompt_hash"]:
        raise RuntimeError("frozen_prompt_hash_mismatch")
    selected_ids = list(selection["stable_record_ids"])
    if selection.get("hard_unique_record_limit") != PILOT_LIMIT or len(selected_ids) != PILOT_LIMIT or len(set(selected_ids)) != PILOT_LIMIT:
        raise RuntimeError("pilot_hard_limit_contract_failed")
    if selected_ids[0] != "caption_ct_train_005626":
        raise RuntimeError("frozen_canary_id_mismatch")
    return prompt, schema, selection


async def run(args: argparse.Namespace) -> int:
    api_key, credential_source = resolve_credential(os.environ)
    endpoint = resolve_base_url(os.environ)
    artifact = Path(args.artifact_root)
    prompt, schema, selection = load_contract(artifact)
    if args.config_audit:
        print(json.dumps({
            "status": "CONFIG_PASS", "credential_present": True, "credential_source": credential_source,
            **endpoint, "model": MODEL, "reasoning_effort": REASONING_EFFORT,
            "temperature": 1.0, "max_completion_tokens": MAX_COMPLETION_TOKENS, "structured_output_mode": "json_schema_strict",
            "prompt_version": prompt["prompt_version"], "prompt_hash": prompt["prompt_hash"],
            "schema_version": prompt["schema_version"], "schema_hash": prompt["schema_hash"], "api_called": False,
        }))
        return 0

    selected_ids = list(selection["stable_record_ids"])
    frozen = {str(row["stable_record_id"]): row for row in read_jsonl(artifact / "input_freeze_manifest.jsonl")}
    if len(frozen) != 70_356 or any(row_id not in frozen for row_id in selected_ids):
        raise RuntimeError("frozen_input_contract_failed")
    for record in selection["records"]:
        source = frozen[record["stable_record_id"]]
        if source["source_caption_sha256"] != record["source_caption_sha256"] or sha_text(source["source_caption"]) != source["source_caption_sha256"]:
            raise RuntimeError(f"caption_hash_mismatch:{record['stable_record_id']}")

    result_path = artifact / "pilot_results.jsonl"
    history, latest, ordinals = load_history(result_path)
    run_ids = selected_ids[:1] if args.canary else selected_ids
    cached_ids = [row_id for row_id in run_ids if active_success(latest.get(row_id), frozen[row_id], prompt, endpoint)]
    cached_set = set(cached_ids)
    pending = [row_id for row_id in run_ids if row_id not in cached_set]
    if args.canary and not pending:
        print(json.dumps({"status": "CANARY_PASS_CACHED", "stable_record_id": run_ids[0], "api_called": False}))
        return 0

    run_id = datetime.now(timezone.utc).strftime("kimi-k3-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]
    client = AsyncOpenAI(api_key=api_key, base_url=endpoint["base_url"], timeout=args.timeout)
    limiter = AdaptiveLimiter(args.requests_per_second)
    writer = ResultWriter(result_path)
    budget = AttemptBudget(len(history), args.max_api_calls)
    semaphore = asyncio.Semaphore(args.concurrency)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    outcomes = collections.Counter()

    async def worker(row_id: str) -> None:
        if stop.is_set():
            return
        async with semaphore:
            result = await request_one(client, limiter, budget, writer, frozen[row_id], prompt, schema, endpoint, credential_source, run_id, args.max_retries, ordinals[row_id])
        if result is None:
            stop.set()
            return
        outcomes[result["status"]] += 1
        if args.canary and result["status"] != "success":
            stop.set()
        if result["status"] in {"contract_or_api_rejected", "contract_model_mismatch", "contract_finish_reason_length"}:
            stop.set()

    tasks = [asyncio.create_task(worker(row_id)) for row_id in pending]
    if tasks:
        await asyncio.gather(*tasks)
    await client.close()

    history, latest, ordinals = load_history(result_path)
    active_cache = {row_id for row_id in selected_ids if active_success(latest.get(row_id), frozen[row_id], prompt, endpoint)}
    summary = {
        "status": "CANARY_PASS" if args.canary and run_ids[0] in active_cache else "RUN_COMPLETE",
        "run_id": run_id, "mode": "canary" if args.canary else "pilot",
        "selected_unique_limit": PILOT_LIMIT, "pending_at_start": len(pending),
        "cache_hits_at_start": len(cached_ids), "outcomes_this_invocation": dict(outcomes),
        "active_success_cache_total": len(active_cache), "attempt_rows_total": len(history),
        "api_attempt_budget_used": budget.used, **endpoint, "credential_present": True,
        "credential_source": credential_source, "model": MODEL, "reasoning_effort": REASONING_EFFORT,
        "temperature": 1.0, "max_completion_tokens": MAX_COMPLETION_TOKENS, "structured_output_mode": "json_schema_strict",
        "stopped": stop.is_set(),
    }
    write_json(artifact / "resume_state.json", summary)
    print(json.dumps(summary))
    if args.canary:
        canary = latest.get(selected_ids[0])
        if selected_ids[0] not in active_cache:
            return 2
        print(json.dumps({"status": "CANARY_PASS", "stable_record_id": selected_ids[0], "attempt_id": canary["attempt_id"], "finish_reason": canary["finish_reason"], "usage": canary["token_usage"], "contract_status": canary["contract_status"]}))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default=str(ARTIFACT))
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--config-audit", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests-per-second", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-api-calls", type=int, default=2_400)
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_retries < 0 or args.max_api_calls < PILOT_LIMIT:
        raise SystemExit("Invalid safety/rate arguments")
    try:
        code = asyncio.run(run(args))
    except RuntimeError as error:
        print(json.dumps({"status": "CONFIG_OR_CONTRACT_ERROR", "error": str(error)}))
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()
