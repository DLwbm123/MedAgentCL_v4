#!/usr/bin/env python3
"""Resumable Kimi Code k3 non-thinking runner for the isolated v3 contract."""

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
import sys
import time
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_kimi_k3_v2_pilot as common  # noqa: E402

ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/medicalskill_cl_kimi_k3_v3_k2_6_routing"
V2 = ROOT / "artifacts/medicalskill_cl_kimi_k3_v2_pilot"
REQUEST_MODEL = "k3"
ROUTED_MODEL = "kimi-k2.6"
EFFORT = "none"
MAX_TOKENS = 256
BASE_URL = "https://api.kimi.com/coding/v1"
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
FORBIDDEN = re.compile(
    r"\b(?:roi|region of interest|bounding box|bbox|area ratio|highlighted region|colored box|"
    r"central position|cross-sectional view|anatomical association)\b", re.I
)
GENERIC = {
    "disease", "pathology", "abnormality", "finding", "process",
    "disease process", "pathological process", "anatomical association", "region of interest",
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).strip(" ;,.").casefold()


def validate_content(raw: str) -> list[str]:
    if "```" in raw:
        raise ValueError("markdown_code_fence")
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"concepts"}:
        raise ValueError("schema_object")
    concepts = value["concepts"]
    if not isinstance(concepts, list) or not 1 <= len(concepts) <= 8:
        raise ValueError("concept_count")
    output, seen = [], set()
    for item in concepts:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("concept_nonempty_string")
        clean = normalize(item)
        if not clean:
            raise ValueError("concept_empty")
        if clean.startswith("uncertain:"):
            raise ValueError("uncertainty_prefix_forbidden")
        if FORBIDDEN.search(clean) or clean in GENERIC:
            raise ValueError("generic_or_artifact_forbidden")
        if re.search(r"\b(?:abdominal ct scan|brain mri)\b", clean):
            raise ValueError("compound_modality_axis_not_split")
        if clean not in seen:
            output.append(clean)
            seen.add(clean)
    for child in ("lesion", "mass", "carcinoma"):
        if child in seen and any(value != child and re.search(rf"\b{child}\b", value) for value in seen):
            raise ValueError("parent_child_duplicate")
    if not output:
        raise ValueError("empty_after_normalization")
    return output


def request_kwargs(source: dict[str, Any], prompt: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": REQUEST_MODEL,
        "messages": [
            {"role": "system", "content": prompt["system_prompt"]},
            {"role": "user", "content": prompt["user_template"].format(caption=source["source_caption"])},
        ],
        "reasoning_effort": EFFORT,
        "max_completion_tokens": MAX_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "medical_concepts_v3", "strict": True, "schema": schema},
        },
    }


def usage_fields(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "cached_tokens": int(getattr(prompt_details, "cached_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "reasoning_token_usage": int(getattr(completion_details, "reasoning_tokens", 0) or 0),
    }


def reasoning_content(message: Any) -> str:
    value = getattr(message, "reasoning_content", None)
    if value is None and hasattr(message, "model_extra"):
        value = (message.model_extra or {}).get("reasoning_content")
    return str(value or "")


def load_contract(artifact: Path = ARTIFACT) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    prompt = json.loads((artifact / "prompt_v3_manifest.json").read_text())
    schema = json.loads((artifact / "kimi_prompt_schema_v3.json").read_text())
    config = json.loads((artifact / "v3_run_config.json").read_text())
    paired = json.loads((artifact / "v3_paired_pilot_200_manifest.json").read_text())
    prompt_contract = {key: prompt[key] for key in ("prompt_version", "system_prompt", "user_template", "normalization_contract")}
    if sha_text(canonical(prompt_contract)) != prompt["prompt_hash"]:
        raise RuntimeError("prompt_hash_mismatch")
    if sha_text(canonical(schema)) != prompt["schema_hash"]:
        raise RuntimeError("schema_hash_mismatch")
    unhashed = {key: value for key, value in config.items() if key != "run_config_hash"}
    if sha_text(canonical(unhashed)) != config["run_config_hash"]:
        raise RuntimeError("run_config_hash_mismatch")
    required = (
        prompt["request_model_id"] == config["request_model_id"] == REQUEST_MODEL,
        prompt["reasoning_effort"] == config["reasoning_effort"] == EFFORT,
        prompt["expected_routed_model"] == config["expected_routed_model"] == ROUTED_MODEL,
        prompt["base_url"] == config["base_url"] == BASE_URL,
        prompt["max_completion_tokens"] == config["max_completion_tokens"] == MAX_TOKENS,
        not any(prompt[field] for field in ("temperature_sent", "top_p_sent", "presence_penalty_sent",
                                             "frequency_penalty_sent", "thinking_field_sent")),
        config["v2_cache_reusable_as_v3"] is False,
        paired["total"] == len(paired["records"]) == 200,
    )
    if not all(required):
        raise RuntimeError("v3_contract_mismatch")
    return prompt, schema, config, paired


def load_sources(artifact: Path, config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    compact = list(common.read_jsonl(artifact / "concept_35k_v3_input_manifest.jsonl"))
    candidate = {row["stable_record_id"]: row for row in common.read_jsonl(V2 / "concept_35k_candidate_manifest.jsonl")}
    stable_ids = (artifact / "concept_35k_stable_ids.txt").read_text().splitlines()
    if len(compact) != 35_000 or len(candidate) != 35_000:
        raise RuntimeError("input_count_mismatch")
    if len(compact) != 35_000 or len(stable_ids) != 35_000 or len(set(stable_ids)) != 35_000:
        raise RuntimeError("input_count_mismatch")
    stable_text = "".join(f"{value}\n" for value in stable_ids)
    if sha_text(stable_text) != config["stable_id_set_sha256"]:
        raise RuntimeError("stable_id_hash_mismatch")
    for row in compact:
        source = candidate.get(row["stable_id"])
        if not source or source["source_caption_sha256"] != row["source_caption_sha256"]:
            raise RuntimeError(f"caption_hash_mismatch:{row['stable_id']}")
        if sha_text(source["source_caption"]) != source["source_caption_sha256"]:
            raise RuntimeError(f"caption_content_hash_mismatch:{row['stable_id']}")
    return candidate, stable_ids


def load_history(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], collections.Counter]:
    history = list(common.read_jsonl(path)) if path.exists() else []
    latest: dict[str, dict[str, Any]] = {}
    ordinals: collections.Counter = collections.Counter()
    for row in history:
        record_id = str(row["stable_id"])
        latest[record_id] = row
        ordinals[record_id] = max(ordinals[record_id], int(row.get("attempt_count", 1) or 1))
    return history, latest, ordinals


def active_success(row: dict[str, Any] | None, source: dict[str, Any], prompt: dict[str, Any], config: dict[str, Any]) -> bool:
    return bool(row) and all((
        row.get("cache_status") == "active_v3_success",
        row.get("stable_id") == source["stable_record_id"],
        row.get("source_caption_sha256") == source["source_caption_sha256"],
        row.get("request_model_id") == REQUEST_MODEL,
        row.get("reasoning_effort") == EFFORT,
        row.get("expected_routed_model") == ROUTED_MODEL,
        row.get("prompt_hash") == prompt["prompt_hash"],
        row.get("schema_hash") == prompt["schema_hash"],
        row.get("run_config_hash") == config["run_config_hash"],
        row.get("reasoning_content_present") is False,
        int(row.get("reasoning_token_usage", 0) or 0) == 0,
        row.get("http_status") == 200,
        row.get("error_type") is None,
        1 <= len(row.get("parsed_concepts") or []) <= 8,
    ))


def make_result(source: dict[str, Any], prompt: dict[str, Any], config: dict[str, Any],
                run_id: str, attempt: int, **values: Any) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "stable_id": source["stable_record_id"],
        "source_caption": source["source_caption"],
        "source_caption_sha256": source["source_caption_sha256"],
        "raw_response": values.get("raw_response", ""),
        "parsed_concepts": values.get("parsed_concepts", []),
        "base_url": BASE_URL,
        "request_model_id": REQUEST_MODEL,
        "response_model_id": values.get("response_model_id"),
        "expected_routed_model": ROUTED_MODEL,
        "reasoning_effort": EFFORT,
        "reasoning_content_present": values.get("reasoning_content_present", False),
        "reasoning_token_usage": values.get("reasoning_token_usage", 0),
        "prompt_version": prompt["prompt_version"],
        "prompt_hash": prompt["prompt_hash"],
        "schema_version": prompt["schema_version"],
        "schema_hash": prompt["schema_hash"],
        "run_config_hash": config["run_config_hash"],
        "prompt_tokens": values.get("prompt_tokens", 0),
        "cached_tokens": values.get("cached_tokens", 0),
        "completion_tokens": values.get("completion_tokens", 0),
        "latency": round(float(values.get("latency", 0)), 6),
        "attempt_count": attempt,
        "http_status": values.get("http_status"),
        "error_type": values.get("error_type"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cache_status": values.get("cache_status", "failed_v3_attempt"),
        "finish_reason": values.get("finish_reason"),
    }


async def request_one(client: AsyncOpenAI, limiter: common.AdaptiveLimiter, budget: common.AttemptBudget,
                      writer: common.ResultWriter, source: dict[str, Any], prompt: dict[str, Any],
                      schema: dict[str, Any], config: dict[str, Any], run_id: str,
                      prior: int, max_retries: int) -> dict[str, Any] | None:
    for local_attempt in range(1, max_retries + 2):
        if not await budget.reserve():
            return None
        attempt = prior + local_attempt
        await limiter.wait()
        started = time.monotonic()
        try:
            response = await client.chat.completions.create(**request_kwargs(source, prompt, schema))
        except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as error:
            code = getattr(error, "status_code", None)
            retryable = isinstance(error, (RateLimitError, APITimeoutError, APIConnectionError)) or code in RETRYABLE_STATUS
            category = (
                "rate_limit" if isinstance(error, RateLimitError) or code == 429
                else "timeout" if isinstance(error, APITimeoutError)
                else "connection" if isinstance(error, APIConnectionError)
                else f"http_{code or 'error'}"
            )
            result = make_result(source, prompt, config, run_id, attempt, latency=time.monotonic() - started,
                                 http_status=code, error_type=category)
            await writer.append(result)
            if not retryable or local_attempt > max_retries:
                return result
            if category == "rate_limit":
                limiter.limited()
            await asyncio.sleep(min(60.0, 2 ** (local_attempt - 1) + random.random()))
            continue
        except Exception as error:
            result = make_result(source, prompt, config, run_id, attempt, latency=time.monotonic() - started,
                                 error_type=f"runner_{type(error).__name__}")
            await writer.append(result)
            return result

        message = response.choices[0].message
        raw = str(message.content or "")
        reasoning = reasoning_content(message)
        usage = usage_fields(response)
        finish = str(response.choices[0].finish_reason or "")
        values = {
            **usage,
            "raw_response": raw,
            "response_model_id": str(getattr(response, "model", "") or ""),
            "reasoning_content_present": bool(reasoning.strip()),
            "latency": time.monotonic() - started,
            "http_status": 200,
            "finish_reason": finish,
        }
        error_type = None
        concepts: list[str] = []
        if reasoning.strip() or usage["reasoning_token_usage"] > 0:
            error_type = "reasoning_not_disabled"
        elif finish in {"length", "max_tokens"}:
            error_type = "finish_reason_length"
        else:
            try:
                concepts = validate_content(raw)
            except (ValueError, json.JSONDecodeError) as error:
                error_type = f"invalid_output:{error}"
        values["parsed_concepts"] = concepts
        values["error_type"] = error_type
        values["cache_status"] = "active_v3_success" if error_type is None else "failed_v3_attempt"
        result = make_result(source, prompt, config, run_id, attempt, **values)
        await writer.append(result)
        if error_type is None:
            limiter.success()
        return result
    raise AssertionError("unreachable")


def acceptance_pass(artifact: Path, filename: str) -> bool:
    path = artifact / filename
    return path.exists() and json.loads(path.read_text()).get("status") == "PASS"


async def run(args: argparse.Namespace) -> int:
    artifact = Path(args.artifact_root)
    prompt, schema, config, paired = load_contract(artifact)
    sources, stable_ids = load_sources(artifact, config)
    if args.offline_config_audit:
        print(json.dumps({
            "status": "CONFIG_PASS", "api_called": False, "base_url": BASE_URL,
            "request_model_id": REQUEST_MODEL, "reasoning_effort": EFFORT,
            "expected_routed_model": ROUTED_MODEL, "max_completion_tokens": MAX_TOKENS,
            "temperature_sent": False, "top_p_sent": False, "presence_penalty_sent": False,
            "frequency_penalty_sent": False, "thinking_field_sent": False,
            "structured_output": "json_schema_strict", "run_config_hash": config["run_config_hash"],
        }))
        return 0

    if args.full:
        if not args.confirm_full_run:
            raise RuntimeError("full_run_requires_--confirm-full-run")
        if not acceptance_pass(artifact, "v3_canary_acceptance.json"):
            raise RuntimeError("canary_acceptance_not_pass")
        if not acceptance_pass(artifact, "v3_paired_pilot_acceptance.json"):
            raise RuntimeError("paired_pilot_acceptance_not_pass")
        selected_ids, mode = stable_ids, "full"
    elif args.paired_pilot:
        selected_ids, mode = [row["stable_id"] for row in paired["records"]], "paired"
    elif args.canary:
        selected_ids, mode = [paired["canary_stable_id"]], "canary"
    else:
        raise RuntimeError("execution_mode_required")

    api_key, _ = common.resolve_credential(os.environ)
    endpoint = common.resolve_base_url(os.environ)
    if endpoint["base_url"] != BASE_URL:
        raise RuntimeError("endpoint_mismatch")
    result_path = artifact / "v3_results.jsonl"
    history, latest, ordinals = load_history(result_path)
    # v2 results are intentionally absent from this cache lookup.
    cached = {record_id for record_id in selected_ids if active_success(latest.get(record_id), sources[record_id], prompt, config)}
    pending = [record_id for record_id in selected_ids if record_id not in cached]
    budget_limit = args.max_api_calls or {"canary": 1, "paired": 250, "full": 50_000}[mode]
    budget = common.AttemptBudget(len(history), budget_limit)
    limiter = common.AdaptiveLimiter(args.requests_per_second)
    writer = common.ResultWriter(result_path)
    client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL, timeout=args.timeout)
    stop, semaphore = asyncio.Event(), asyncio.Semaphore(args.concurrency)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    run_id = datetime.now(timezone.utc).strftime("k2.6-routing-v3-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]
    outcomes = collections.Counter()

    async def worker(record_id: str) -> None:
        if stop.is_set():
            return
        async with semaphore:
            if stop.is_set():
                return
            result = await request_one(client, limiter, budget, writer, sources[record_id], prompt, schema,
                                       config, run_id, ordinals[record_id], args.max_retries)
        if result is None:
            stop.set()
            return
        outcomes[result["cache_status"]] += 1
        error = result.get("error_type")
        if error and not (error in {"rate_limit", "timeout", "connection"} or error.startswith("http_5")):
            stop.set()

    for offset in range(0, len(pending), args.batch_size):
        if stop.is_set():
            break
        batch = pending[offset:offset + args.batch_size]
        await asyncio.gather(*(worker(record_id) for record_id in batch))
        common.write_json(artifact / "v3_resume_state.json", {
            "status": "STOPPED" if stop.is_set() else "RUNNING",
            "mode": mode, "run_id": run_id, "selected": len(selected_ids),
            "cache_hits_at_start": len(cached), "attempt_budget_used": budget.used,
            "next_batch_offset": offset + len(batch), "batch_size": args.batch_size,
            "outcomes": dict(outcomes), "api_called": True,
        })
    await client.close()
    history, latest, _ = load_history(result_path)
    active = sum(active_success(latest.get(record_id), sources[record_id], prompt, config) for record_id in selected_ids)
    summary = {
        "status": "STOPPED" if stop.is_set() else "RUN_COMPLETE",
        "mode": mode, "selected": len(selected_ids), "active_success": active,
        "attempt_rows_total": len(history), "cache_hits_at_start": len(cached),
        "outcomes": dict(outcomes), "request_model_id": REQUEST_MODEL,
        "reasoning_effort": EFFORT, "expected_routed_model": ROUTED_MODEL,
        "api_called": bool(pending), "stopped": stop.is_set(),
    }
    common.write_json(artifact / "v3_resume_state.json", summary)
    print(json.dumps(summary))
    return 0 if not stop.is_set() else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline-config-audit", action="store_true")
    mode.add_argument("--canary", action="store_true")
    mode.add_argument("--paired-pilot", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--confirm-full-run", action="store_true")
    parser.add_argument("--artifact-root", default=str(ARTIFACT))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests-per-second", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, choices=(1000, 2000), default=1000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-api-calls", type=int)
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_retries < 0 or (args.max_api_calls is not None and args.max_api_calls < 1):
        raise SystemExit("invalid safety arguments")
    try:
        code = asyncio.run(run(args))
    except RuntimeError as error:
        print(json.dumps({"status": "CONFIG_OR_GATE_ERROR", "error": str(error), "api_called": False}))
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()
