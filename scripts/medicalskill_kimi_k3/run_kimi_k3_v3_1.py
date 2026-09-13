#!/usr/bin/env python3
"""Resumable Kimi K2.6-routing v3.1 runner with atomic state recovery."""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import os
import re
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_kimi_k3_v2_pilot as common  # noqa: E402
import v3_1_quality_policy as quality  # noqa: E402

ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/medicalskill_cl_kimi_k3_v3_1_k2_6_routing"
V2 = ROOT / "artifacts/medicalskill_cl_kimi_k3_v2_pilot"
REQUEST_MODEL = "k3"
ROUTED_MODEL = "kimi-k2.6"
EFFORT = "none"
MAX_TOKENS = 256
TEMPERATURE = 0.6
RUNTIME_POLICY_REVISION = 3
BASE_URL = "https://api.kimi.com/coding/v1"
BATCH_SIZE = 1000
CYCLE_HARD_LIMIT = 18_000
RETRYABLE_STATUS = {429}
TWO_HOUR_WINDOW_SECONDS = 2 * 60 * 60
TWO_HOUR_CALL_LIMIT = 6_600
MAX_AUTHORIZED_BATCH_INDEX = 17
SELECTIVE_RETRY_IDS = (
    "caption_microscopy_train_006806",
    "caption_histopathology_train_007647",
    "caption_microscopy_train_004840",
    "caption_pet_train_004376",
    "caption_xray_train_005620",
)
NEGATED = re.compile(
    r"\b(?:no|not|without|absence(?: of)?|absent|negative for|free of|lack(?:ing)?|no evidence of)\b",
    re.I,
)
FORBIDDEN = re.compile(
    r"\b(?:roi|region of interest|bounding box|bbox|area ratio|highlighted region|colored box|"
    r"central position|cross-sectional view|anatomical association)\b",
    re.I,
)
GENERIC = {
    "disease", "pathology", "abnormality", "finding", "process",
    "disease process", "pathological process", "anatomical association", "region of interest",
}
BACKGROUND_ORGANS = {
    "brain", "heart", "liver", "spleen", "kidney", "kidneys", "lung", "lungs",
    "stomach", "bowel", "intestine", "intestines", "gastrointestinal tract",
    "pancreas", "bladder", "colon", "small bowel", "large bowel", "spine",
}
NEGATION_SENSITIVE = {
    "necrosis", "pleomorphism", "colloid", "metastasis", "mitotic activity",
    "mitotic figures", "lipoblasts", "myelin",
}
CONTRADICTIONS = (
    (("high-grade tumor", "high grade tumor", "high-grade malignancy", "high grade malignancy"),
     ("low-grade tumor", "low grade tumor", "low-grade malignancy", "low grade malignancy")),
    (("benign tumor", "benign neoplasm", "benign lesion"),
     ("malignant tumor", "malignant neoplasm", "malignancy", "sarcoma", "carcinoma")),
    (("well-differentiated", "well differentiated"),
     ("poorly differentiated", "poor differentiated", "poorly-differentiated")),
    (("high-grade", "high grade"),
     ("well-differentiated", "well differentiated")),
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha_file(path: Path) -> str:
    if not path.exists():
        return hashlib.sha256(b"").hexdigest()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value: str) -> str:
    return " ".join(value.split()).strip(" ;,.").casefold()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    payload = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    try:
        with temp.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp.exists():
            temp.unlink()


def read_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    output = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise RuntimeError(f"corrupt_results_jsonl_line:{index + 1}")
        if not isinstance(value, dict):
            raise RuntimeError(f"non_object_results_jsonl_line:{index + 1}")
        output.append(value)
    return output


def _caption_negates_concept(caption: str, concept: str) -> bool:
    if concept not in NEGATION_SENSITIVE and not re.fullmatch(r"(?:cd\d+|erg|sox10|s100)", concept):
        return False
    text = normalize(caption)
    for cue in NEGATED.finditer(text):
        clause = text[cue.end():cue.end() + 96]
        if re.search(rf"\b{re.escape(concept)}\b", clause):
            return True
    return False


def _matching_values(group: tuple[str, ...], concepts: list[str]) -> set[str]:
    return {value for value in concepts if any(term in value for term in group)}


def _resolve_contradictions(concepts: list[str], caption: str) -> tuple[list[str], bool]:
    output = list(concepts)
    text = normalize(caption)
    changed = False
    for left, right in CONTRADICTIONS:
        left_values = _matching_values(left, output)
        right_values = _matching_values(right, output)
        if not left_values or not right_values:
            continue
        left_score = sum(text.count(term) for term in left)
        right_score = sum(text.count(term) for term in right)
        if left_score > right_score:
            drop = right_values
        elif right_score > left_score:
            drop = left_values
        else:
            drop = left_values | right_values
        output = [value for value in output if value not in drop]
        changed = True
    return output, changed


def _pet_background_only(caption: str, concept: str) -> bool:
    text = normalize(caption)
    if text.count(concept) != 1 or "showing various organs" not in text:
        return False
    association = re.compile(
        rf"(?:lesion|uptake|abnormality|affected|primary organ)[^.]*\b{re.escape(concept)}\b|"
        rf"\b{re.escape(concept)}\b[^.]*(?:lesion|uptake|abnormality|affected|primary organ)"
    )
    return association.search(text) is None


def parse_with_repairs(raw: str, source_caption: str = "", modality: str = "") -> tuple[list[str], list[str], list[str]]:
    if chr(96) * 3 in raw:
        raise ValueError("markdown_code_fence")
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"concepts"}:
        raise ValueError("schema_object")
    concepts = value["concepts"]
    if not isinstance(concepts, list) or not 1 <= len(concepts) <= 8:
        raise ValueError("concept_count")
    normalized, seen = [], set()
    for item in concepts:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("concept_nonempty_string")
        clean = normalize(item)
        if not clean:
            raise ValueError("concept_empty")
        if clean not in seen:
            normalized.append(clean)
            seen.add(clean)
    repairs = []
    output = [value for value in normalized if not NEGATED.search(value)]
    if len(output) != len(normalized):
        repairs.append("remove_explicitly_negated_concept")
    caption_trimmed = [value for value in output if not _caption_negates_concept(source_caption, value)]
    if len(caption_trimmed) != len(output):
        output = caption_trimmed
        repairs.append("remove_caption_negated_positive_concept")
    output, contradiction_changed = _resolve_contradictions(output, source_caption)
    if contradiction_changed:
        repairs.append("resolve_caption_internal_contradiction")
    organ_count = sum(value in BACKGROUND_ORGANS for value in output)
    if organ_count >= 4 or (len(output) == 8 and organ_count >= 3):
        trimmed = [value for value in output if value not in BACKGROUND_ORGANS]
        if trimmed:
            output = trimmed
            repairs.append("remove_obvious_background_organ_enumeration")
    if modality.casefold() == "pet":
        trimmed = [value for value in output if value not in BACKGROUND_ORGANS or not _pet_background_only(source_caption, value)]
        if trimmed and len(trimmed) != len(output):
            output = trimmed
            repairs.append("remove_pet_single_mention_background_organ")
    if not output:
        raise ValueError("empty_after_deterministic_repairs")
    return output, repairs, normalized


def validate_content(raw: str) -> list[str]:
    concepts, _, _ = parse_with_repairs(raw)
    return concepts


def contradiction_hits(concepts: list[str]) -> list[str]:
    hits = []
    for left, right in CONTRADICTIONS:
        left_values = _matching_values(left, concepts)
        right_values = _matching_values(right, concepts)
        if left_values and right_values:
            hits.append(f"{sorted(left_values)[0]}::{sorted(right_values)[0]}")
    return hits


def semantic_violations(concepts: list[str]) -> list[str]:
    violations = []
    if any(value.startswith("uncertain:") for value in concepts):
        violations.append("uncertain_prefix")
    if any(NEGATED.search(value) for value in concepts):
        violations.append("negated_concept")
    if any(FORBIDDEN.search(value) or value in GENERIC for value in concepts):
        violations.append("roi_or_generic_artifact")
    if contradiction_hits(concepts):
        violations.append("mutually_exclusive_concepts")
    seen = set(concepts)
    for child in ("lesion", "mass", "carcinoma"):
        if child in seen and any(value != child and re.search(rf"\b{child}\b", value) for value in seen):
            violations.append("parent_child_duplicate")
            break
    organ_count = sum(value in BACKGROUND_ORGANS for value in concepts)
    if organ_count >= 4 or (len(concepts) == 8 and organ_count >= 3):
        violations.append("background_organ_enumeration")
    return sorted(set(violations))


def request_kwargs(source: dict[str, Any], prompt: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": REQUEST_MODEL,
        "messages": [
            {"role": "system", "content": prompt["system_prompt"]},
            {"role": "user", "content": prompt["user_template"].format(caption=source["source_caption"])},
        ],
        "reasoning_effort": EFFORT,
        "temperature": TEMPERATURE,
        "max_completion_tokens": MAX_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "medical_concepts_v3_1", "strict": True, "schema": schema},
        },
    }


def load_contract(artifact: Path = ARTIFACT):
    prompt = json.loads((artifact / "prompt_v3_1_manifest.json").read_text())
    schema = json.loads((artifact / "kimi_prompt_schema_v3_1.json").read_text())
    config = json.loads((artifact / "run_config_v3_1.json").read_text())
    pilot = json.loads((artifact / "v3_1_pilot_50_manifest.json").read_text())
    prompt_contract = {
        key: prompt[key]
        for key in ("prompt_version", "system_prompt", "user_template", "normalization_contract")
    }
    if sha_text(canonical(prompt_contract)) != prompt["prompt_hash"]:
        raise RuntimeError("prompt_hash_mismatch")
    if sha_text(canonical(schema)) != prompt["schema_hash"]:
        raise RuntimeError("schema_hash_mismatch")
    unhashed = {key: value for key, value in config.items() if key != "run_config_hash"}
    if sha_text(canonical(unhashed)) != config["run_config_hash"]:
        raise RuntimeError("run_config_hash_mismatch")
    checks = (
        prompt["request_model_id"] == config["request_model_id"] == REQUEST_MODEL,
        prompt["reasoning_effort"] == config["reasoning_effort"] == EFFORT,
        prompt["expected_routed_model"] == config["expected_routed_model"] == ROUTED_MODEL,
        prompt["base_url"] == config["base_url"] == BASE_URL,
        prompt["max_completion_tokens"] == config["max_completion_tokens"] == MAX_TOKENS,
        config["formal_batch_size"] == BATCH_SIZE,
        config["formal_cycle_hard_limit"] == CYCLE_HARD_LIMIT,
        config["v3_cache_reusable_as_v3_1"] is False,
        pilot["total"] == len(pilot["records"]) == 50,
    )
    if not all(checks):
        raise RuntimeError("v3_1_contract_mismatch")
    return prompt, schema, config, pilot


def load_sources(artifact: Path, config: dict[str, Any]):
    with (V2 / "concept_35k_candidate_manifest.jsonl").open(encoding="utf-8-sig") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    sources = {row["stable_record_id"]: row for row in rows}
    stable_ids = (artifact / "concept_35k_stable_ids.txt").read_text().splitlines()
    if len(sources) != 35_000 or len(stable_ids) != 35_000 or len(set(stable_ids)) != 35_000:
        raise RuntimeError("input_count_mismatch")
    if sha_text("".join(f"{value}\n" for value in stable_ids)) != config["stable_id_set_sha256"]:
        raise RuntimeError("stable_id_hash_mismatch")
    return sources, stable_ids


def load_history(path: Path):
    history = read_results(path)
    latest: dict[str, dict[str, Any]] = {}
    ordinals: collections.Counter = collections.Counter()
    for row in history:
        record_id = str(row["stable_id"])
        latest[record_id] = row
        ordinals[record_id] = max(ordinals[record_id], int(row.get("attempt_count", 1) or 1))
    return history, latest, ordinals


def selective_retry_context(record_id: str, history: list[dict[str, Any]],
                            latest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    api_rows = [
        row for row in history
        if row.get("stable_id") == record_id and row.get("api_call_performed") is not False
        and row.get("http_status") == 200 and row.get("raw_response")
    ]
    if not api_rows:
        raise RuntimeError(f"missing_original_api_response:{record_id}")
    original = api_rows[0]
    latest_row = latest[record_id]
    reasons = list(latest_row.get("parser_repairs") or [])
    if not reasons:
        concepts = original.get("raw_model_concepts") or original.get("raw_parsed_concepts") or original.get("parsed_concepts") or []
        audited = quality.audit_concepts(concepts, original.get("source_caption", ""), original.get("modality", ""))
        reasons = audited["retry_reasons"] or ["legacy_deterministic_semantic_repair"]
    return {
        "original_run_id": original.get("original_run_id") or original.get("run_id"),
        "original_raw_response": original.get("raw_response", ""),
        "raw_model_concepts": original.get("raw_model_concepts") or original.get("raw_parsed_concepts") or original.get("parsed_concepts") or [],
        "retry_reasons": reasons,
    }


def selective_retry_terminal(row: dict[str, Any] | None) -> bool:
    return bool(row) and row.get("retry_run_id") and row.get("output_source") == "retry_model" and row.get("quality_status") in {"PASS", "REJECTED_AFTER_RETRY"}


def active_success(row: dict[str, Any] | None, source: dict[str, Any],
                   prompt: dict[str, Any], config: dict[str, Any]) -> bool:
    final_concepts = (row or {}).get("final_concepts") or []
    return bool(row) and all((
        row.get("cache_status") == "active_v3_1_success",
        row.get("quality_status") in {"PASS", "PASS_WITH_QC_FLAGS", "PASS_LEGACY_AUDITED"},
        row.get("output_source") in {"raw_model", "retry_model"},
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
        1 <= len(final_concepts) <= 8,
    ))


def terminal_rejected(row: dict[str, Any] | None) -> bool:
    return bool(row) and row.get("quality_status") in {
        "REJECTED_AFTER_RETRY", "TECHNICAL_FAILED_AFTER_RETRY", "CONTRACT_VIOLATION"
    }


def state_payload(status: str, selected_ids: list[str], latest: dict[str, dict[str, Any]],
                  sources: dict[str, dict[str, Any]], prompt: dict[str, Any],
                  config: dict[str, Any], results_path: Path, mode: str,
                  cycle_id: str | None, batch_index: int | None,
                  rebuilt_from_results: bool = False) -> dict[str, Any]:
    completed = [
        record_id for record_id in selected_ids
        if active_success(latest.get(record_id), sources[record_id], prompt, config)
    ]
    failed = [record_id for record_id in selected_ids if terminal_rejected(latest.get(record_id))]
    terminal = set(completed) | set(failed)
    pending = [record_id for record_id in selected_ids if record_id not in terminal]
    return {
        "status": status,
        "mode": mode,
        "cycle_id": cycle_id,
        "batch_index": batch_index,
        "completed": completed,
        "failed": failed,
        "pending": pending,
        "counts": {
            "selected": len(selected_ids),
            "completed": len(completed),
            "failed": len(failed),
            "pending": len(pending),
        },
        "last_update": datetime.now(timezone.utc).isoformat(),
        "prompt_hash": prompt["prompt_hash"],
        "schema_hash": prompt["schema_hash"],
        "run_config_hash": config["run_config_hash"],
        "results_hash": sha_file(results_path),
        "rebuilt_from_results": rebuilt_from_results,
    }


def valid_state(path: Path, prompt: dict[str, Any], config: dict[str, Any],
                results_path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return all((
        state.get("prompt_hash") == prompt["prompt_hash"],
        state.get("schema_hash") == prompt["schema_hash"],
        state.get("run_config_hash") == config["run_config_hash"],
        state.get("results_hash") == sha_file(results_path),
        isinstance(state.get("completed"), list),
        isinstance(state.get("failed"), list),
        isinstance(state.get("pending"), list),
    ))


def rebuild_resume_state(state_path: Path, results_path: Path, selected_ids: list[str],
                         latest: dict[str, dict[str, Any]], sources: dict[str, dict[str, Any]],
                         prompt: dict[str, Any], config: dict[str, Any], mode: str,
                         cycle_id: str | None = None, batch_index: int | None = None) -> dict[str, Any]:
    rebuilt = not valid_state(state_path, prompt, config, results_path)
    state = state_payload(
        "REBUILT" if rebuilt else "READY",
        selected_ids, latest, sources, prompt, config, results_path,
        mode, cycle_id, batch_index, rebuilt,
    )
    atomic_write_json(state_path, state)
    return state


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


class StateTracker:
    def __init__(self, path: Path, results_path: Path, selected_ids: list[str],
                 latest: dict[str, dict[str, Any]], sources: dict[str, dict[str, Any]],
                 prompt: dict[str, Any], config: dict[str, Any], mode: str,
                 cycle_id: str | None, batch_index: int | None):
        self.path, self.results_path = path, results_path
        self.selected_ids, self.latest, self.sources = selected_ids, latest, sources
        self.prompt, self.config, self.mode = prompt, config, mode
        self.cycle_id, self.batch_index = cycle_id, batch_index
        self.lock = asyncio.Lock()

    async def update(self, row: dict[str, Any], status: str = "RUNNING") -> None:
        async with self.lock:
            self.latest[row["stable_id"]] = row
            atomic_write_json(
                self.path,
                state_payload(
                    status, self.selected_ids, self.latest, self.sources,
                    self.prompt, self.config, self.results_path, self.mode,
                    self.cycle_id, self.batch_index,
                ),
            )

    async def finalize(self, status: str) -> dict[str, Any]:
        async with self.lock:
            state = state_payload(
                status, self.selected_ids, self.latest, self.sources,
                self.prompt, self.config, self.results_path, self.mode,
                self.cycle_id, self.batch_index,
            )
            atomic_write_json(self.path, state)
            return state


def response_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "cached_tokens": int(getattr(prompt_details, "cached_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "reasoning_token_usage": int(getattr(completion_details, "reasoning_tokens", 0) or 0),
    }


def response_reasoning(message: Any) -> str:
    value = getattr(message, "reasoning_content", None)
    if value is None and hasattr(message, "model_extra"):
        value = (message.model_extra or {}).get("reasoning_content")
    return str(value or "")


def sanitize_provider_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    for secret_name in ("KIMI_API_KEY", "MOONSHOT_API_KEY"):
        secret = os.environ.get(secret_name, "")
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(
        r"(?i)bearer\\s+[a-z0-9._-]+",
        "Bearer [REDACTED]",
        text,
    )
    text = re.sub(
        r"(?i)(?:api[_-]?key|authorization)(\\s*[=:]\\s*)\\S+",
        r"\\1[REDACTED]",
        text,
    )
    return text[:2000]


def provider_error_details(error: Exception) -> dict[str, Any]:
    payload = getattr(error, "body", None)
    if not isinstance(payload, dict):
        response = getattr(error, "response", None)
        try:
            payload = response.json() if response is not None else {}
        except Exception:
            payload = {}
    detail = payload.get("error", payload) if isinstance(payload, dict) else {}
    if not isinstance(detail, dict):
        detail = {"message": detail}
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    return {
        "provider_error_code": sanitize_provider_text(
            detail.get("code")
        ),
        "provider_error_type": sanitize_provider_text(
            detail.get("type")
        ),
        "provider_error_param": sanitize_provider_text(
            detail.get("param")
        ),
        "provider_error_message": sanitize_provider_text(
            detail.get("message") or str(error)
        ),
        "provider_request_id": sanitize_provider_text(
            headers.get("x-request-id")
            or headers.get("request-id")
        ),
    }


def make_result(source: dict[str, Any], prompt: dict[str, Any], config: dict[str, Any],
                run_id: str, attempt: int, mode: str, cycle_id: str | None,
                batch_index: int | None, **values: Any) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "mode": mode,
        "cycle_id": cycle_id,
        "batch_index": batch_index,
        "stable_id": source["stable_record_id"],
        "source_caption": source["source_caption"],
        "source_caption_sha256": source["source_caption_sha256"],
        "modality": source.get("modality"),
        "raw_response": values.get("raw_response", ""),
        "original_raw_response": values.get("original_raw_response", ""),
        "raw_model_concepts": values.get("raw_model_concepts", []),
        "retry_model_concepts": values.get("retry_model_concepts", []),
        "final_concepts": values.get("final_concepts", []),
        "repair_or_retry_reason": values.get("repair_or_retry_reason", []),
        "output_source": values.get("output_source"),
        "original_run_id": values.get("original_run_id"),
        "retry_run_id": values.get("retry_run_id"),
        "audit_flags": values.get("audit_flags", []),
        "audit_details": values.get("audit_details", {}),
        "parsed_concepts": values.get("final_concepts", []),
        "raw_parsed_concepts": values.get("raw_model_concepts", []),
        "parser_repairs": [],
        "semantic_violations": values.get("audit_flags", []),
        "quality_status": values.get("quality_status", "NOT_EVALUATED"),
        "base_url": BASE_URL,
        "request_model_id": REQUEST_MODEL,
        "response_model_id": values.get("response_model_id"),
        "expected_routed_model": ROUTED_MODEL,
        "reasoning_effort": EFFORT,
        "temperature": TEMPERATURE,
        "top_p_sent": False,
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
        "finish_reason": values.get("finish_reason"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cache_status": values.get("cache_status", "failed_v3_1_attempt"),
        "api_call_performed": values.get("api_call_performed", True),
        "api_attempt_kind": values.get("api_attempt_kind"),
        "transport_attempt": values.get("transport_attempt", 1),
        "retry_after_seconds": values.get("retry_after_seconds"),
        "rate_limit_reset": values.get("rate_limit_reset"),
        "quota_exhausted": values.get("quota_exhausted", False),
        "quota_scope": values.get("quota_scope"),
        "safe_error_message": values.get("safe_error_message"),
        "provider_error_code": values.get("provider_error_code"),
        "provider_error_type": values.get("provider_error_type"),
        "provider_error_param": values.get("provider_error_param"),
        "provider_error_message": values.get("provider_error_message"),
        "provider_request_id": values.get("provider_request_id"),
        "concurrency_at_attempt": values.get("concurrency_at_attempt"),
        "runtime_policy_revision": RUNTIME_POLICY_REVISION,
        "audit_policy_revision": values.get(
            "audit_policy_revision", quality.AUDIT_POLICY_REVISION
        ),
    }


class RollingWindowLimiter:
    """Global RPS limiter plus a persisted two-hour call window."""

    def __init__(self, requests_per_second: float, history: list[dict[str, Any]],
                 cycle_id: str | None):
        self.base = common.AdaptiveLimiter(requests_per_second)
        self.lock = asyncio.Lock()
        self.timestamps: collections.deque[float] = collections.deque()
        self.total_wait_seconds = 0.0
        now = time.time()
        for row in history:
            if row.get("api_call_performed") is False:
                continue
            if row.get("mode") not in {"formal_batch", "recovery_queue"} or row.get("cycle_id") != cycle_id:
                continue
            try:
                stamp = datetime.fromisoformat(row["timestamp"]).timestamp()
            except (KeyError, TypeError, ValueError):
                continue
            if stamp > now - TWO_HOUR_WINDOW_SECONDS:
                self.timestamps.append(stamp)

    async def wait(self) -> None:
        while True:
            async with self.lock:
                now = time.time()
                cutoff = now - TWO_HOUR_WINDOW_SECONDS
                while self.timestamps and self.timestamps[0] <= cutoff:
                    self.timestamps.popleft()
                if len(self.timestamps) < TWO_HOUR_CALL_LIMIT:
                    self.timestamps.append(now)
                    break
                delay = max(0.25, self.timestamps[0] + TWO_HOUR_WINDOW_SECONDS - now)
            self.total_wait_seconds += delay
            await asyncio.sleep(delay)
        await self.base.wait()

    def limited(self) -> None:
        self.base.limited()

    def success(self) -> None:
        self.base.success()


async def request_one(client: AsyncOpenAI, limiter: RollingWindowLimiter,
                      budget: common.AttemptBudget, writer: ResultWriter,
                      tracker: StateTracker, source: dict[str, Any],
                      prompt: dict[str, Any], schema: dict[str, Any],
                      config: dict[str, Any], run_id: str, prior: int,
                      mode: str, cycle_id: str | None, batch_index: int | None,
                      max_transport_retries: int,
                      retry_only_context: dict[str, Any] | None = None,
                      concurrency_at_attempt: int = 1):
    if retry_only_context is not None:
        raise RuntimeError("semantic_retry_disabled_by_policy_revision_1")

    record_id = source["stable_record_id"]
    original_event_id = f"{run_id}:original:{record_id}"
    original_raw_response = ""
    original_concepts: list[str] = []
    technical_retry_reasons: list[str] = []
    api_ordinal = prior

    for technical_stage in range(2):
        attempt_kind = "original" if technical_stage == 0 else "technical_content_retry"
        event_id = (
            original_event_id if technical_stage == 0
            else f"{run_id}:technical-retry:{record_id}"
        )
        response = None
        for transport_attempt in range(1, max_transport_retries + 2):
            if not await budget.reserve():
                return None
            api_ordinal += 1
            await limiter.wait()
            started = time.monotonic()
            try:
                response = await client.chat.completions.create(
                    **request_kwargs(source, prompt, schema)
                )
            except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as error:
                code = getattr(error, "status_code", None)
                retryable = (
                    isinstance(error, (RateLimitError, APITimeoutError, APIConnectionError))
                    or (isinstance(code, int) and 500 <= code < 600)
                )
                error_type = (
                    "rate_limit" if isinstance(error, RateLimitError) or code == 429
                    else "timeout" if isinstance(error, APITimeoutError)
                    else "connection" if isinstance(error, APIConnectionError)
                    else f"http_{code or 'unknown'}"
                )
                # A 429 leaves this process immediately. The controller owns
                # cooldown, probing, and bounded recovery across windows.
                will_retry = (
                    retryable
                    and code not in {403, 429}
                    and transport_attempt <= max_transport_retries
                )
                provider_details = provider_error_details(error)
                error_text = str(
                    provider_details.get("provider_error_message")
                    or error
                ).lower()
                quota_scope = next((
                    scope for scope in ("weekly", "monthly", "account")
                    if scope in error_text
                ), None)
                quota_markers = (
                    "quota", "usage limit", "weekly limit",
                    "monthly limit", "account limit",
                )
                quota_exhausted = bool(
                    code in {403, 429}
                    and any(marker in error_text for marker in quota_markers)
                )
                if quota_exhausted and quota_scope is None:
                    quota_scope = "account"
                retry_after = common.retry_after_seconds(error)
                row = make_result(
                    source, prompt, config, run_id, api_ordinal, mode, cycle_id, batch_index,
                    original_raw_response=original_raw_response,
                    raw_model_concepts=original_concepts,
                    final_concepts=[],
                    repair_or_retry_reason=technical_retry_reasons + [error_type],
                    output_source="retry_model" if technical_stage else "raw_model",
                    original_run_id=original_event_id,
                    retry_run_id=event_id if technical_stage else None,
                    quality_status="TRANSPORT_RETRYING" if will_retry else "TRANSPORT_FAILED",
                    latency=time.monotonic() - started,
                    http_status=code,
                    error_type=error_type,
                    cache_status="failed_v3_1_attempt",
                    api_attempt_kind=attempt_kind,
                    transport_attempt=transport_attempt,
                    retry_after_seconds=retry_after,
                    quota_exhausted=quota_exhausted,
                    quota_scope=quota_scope,
                    safe_error_message=error_type,
                    concurrency_at_attempt=concurrency_at_attempt,
                    **provider_details,
                )
                await writer.append(row)
                await tracker.update(row)
                if will_retry:
                    limiter.limited()
                    backoff = min(120.0, (2 ** (transport_attempt - 1)) + 0.25)
                    await asyncio.sleep(max(backoff, retry_after or 0.0))
                    continue
                return row
            break

        assert response is not None
        message = response.choices[0].message
        raw = str(message.content or "")
        reasoning = response_reasoning(message)
        usage = response_usage(response)
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        evaluated = quality.evaluate_response(
            raw, source.get("source_caption", ""), source.get("modality", ""), finish_reason
        )
        current_concepts = evaluated["raw_model_concepts"]
        parse_errors = list(evaluated["parse_errors"])
        retryable_parse_errors = list(evaluated["retry_reasons"])

        if technical_stage == 0:
            original_raw_response = raw
            original_concepts = current_concepts
            technical_retry_reasons = parse_errors

        if reasoning or usage["reasoning_token_usage"]:
            row = make_result(
                source, prompt, config, run_id, api_ordinal, mode, cycle_id, batch_index,
                raw_response=raw,
                original_raw_response=original_raw_response if technical_stage else "",
                raw_model_concepts=original_concepts,
                retry_model_concepts=current_concepts if technical_stage else [],
                final_concepts=[],
                repair_or_retry_reason=["reasoning_contract_violation"],
                output_source="retry_model" if technical_stage else "raw_model",
                original_run_id=original_event_id,
                retry_run_id=event_id if technical_stage else None,
                audit_flags=evaluated["audit_flags"],
                audit_details=evaluated["audit_details"],
                quality_status="CONTRACT_VIOLATION",
                response_model_id=getattr(response, "model", None),
                reasoning_content_present=bool(reasoning),
                latency=time.monotonic() - started,
                http_status=200,
                error_type="reasoning_contract_violation",
                finish_reason=finish_reason,
                cache_status="failed_v3_1_attempt",
                api_attempt_kind=attempt_kind,
                transport_attempt=transport_attempt,
                **usage,
            )
            await writer.append(row)
            await tracker.update(row)
            return row

        if parse_errors:
            will_retry = technical_stage == 0 and bool(retryable_parse_errors)
            row = make_result(
                source, prompt, config, run_id, api_ordinal, mode, cycle_id, batch_index,
                raw_response=raw,
                original_raw_response=original_raw_response if technical_stage else "",
                raw_model_concepts=original_concepts,
                retry_model_concepts=current_concepts if technical_stage else [],
                final_concepts=[],
                repair_or_retry_reason=parse_errors,
                output_source="retry_model" if technical_stage else "raw_model",
                original_run_id=original_event_id,
                retry_run_id=event_id if technical_stage else None,
                audit_flags=[],
                audit_details={},
                quality_status=(
                    "TECHNICAL_RETRY_REQUIRED" if will_retry
                    else "TECHNICAL_FAILED_AFTER_RETRY"
                ),
                response_model_id=getattr(response, "model", None),
                reasoning_content_present=False,
                latency=time.monotonic() - started,
                http_status=200,
                error_type=(
                    "technical_content_retry_required" if will_retry
                    else "technical_content_failed"
                ),
                finish_reason=finish_reason,
                cache_status="technical_retry_required" if will_retry else "technical_failed",
                api_attempt_kind=attempt_kind,
                transport_attempt=transport_attempt,
                **usage,
            )
            await writer.append(row)
            await tracker.update(row)
            if will_retry:
                continue
            return row

        qc_flags = list(evaluated["audit_flags"])
        status = "PASS_WITH_QC_FLAGS" if qc_flags else "PASS"
        row = make_result(
            source, prompt, config, run_id, api_ordinal, mode, cycle_id, batch_index,
            raw_response=raw,
            original_raw_response=original_raw_response if technical_stage else "",
            raw_model_concepts=original_concepts if technical_stage else current_concepts,
            retry_model_concepts=current_concepts if technical_stage else [],
            final_concepts=current_concepts,
            repair_or_retry_reason=technical_retry_reasons if technical_stage else [],
            output_source="retry_model" if technical_stage else "raw_model",
            original_run_id=original_event_id,
            retry_run_id=event_id if technical_stage else None,
            audit_flags=qc_flags,
            audit_details=evaluated["audit_details"],
            quality_status=status,
            response_model_id=getattr(response, "model", None),
            reasoning_content_present=False,
            latency=time.monotonic() - started,
            http_status=200,
            error_type=None,
            finish_reason=finish_reason,
            cache_status="active_v3_1_success",
            api_attempt_kind=attempt_kind,
            transport_attempt=transport_attempt,
            **usage,
        )
        await writer.append(row)
        await tracker.update(row)
        limiter.success()
        return row

    raise AssertionError("unreachable")

def offline_recertify(artifact: Path, prompt: dict[str, Any], config: dict[str, Any],
                      pilot: dict[str, Any], sources: dict[str, dict[str, Any]]) -> int:
    raise RuntimeError("deterministic_recertification_disabled_by_v3_1_policy_revision")


def latest_acceptance_revision(artifact: Path) -> dict[str, Any] | None:
    path = artifact / "v3_1_pilot_acceptance_revisions.jsonl"
    if not path.exists():
        return None
    rows = read_results(path)
    return rows[-1] if rows else None


def acceptance_pass(artifact: Path) -> bool:
    revision = latest_acceptance_revision(artifact)
    return bool(revision) and revision.get("technical_acceptance_status") == "PASS" and revision.get("human_review", {}).get("status") == "NOT_VERIFIED_BY_USER"


def formal_selection(args: argparse.Namespace, stable_ids: list[str], config: dict[str, Any]) -> list[str]:
    if not args.confirm_formal_batch:
        raise RuntimeError("formal_batch_requires_confirmation")
    if not acceptance_pass(Path(args.artifact_root)):
        raise RuntimeError("v3_1_pilot_acceptance_not_pass")
    if args.formal_batch_index is None or not 0 <= args.formal_batch_index < config["formal_batches_total"]:
        raise RuntimeError("invalid_formal_batch_index")
    if args.cycle_id != "cycle_01" or args.cycle_start_batch != 0:
        raise RuntimeError("cycle_01_and_start_0_required")
    revision = latest_acceptance_revision(Path(args.artifact_root)) or {}
    authorized = set(revision.get("authorized_formal_batch_indices") or [])
    if args.formal_batch_index not in authorized:
        if args.formal_batch_index >= 18:
            raise RuntimeError("batch_18_and_later_not_authorized")
        raise RuntimeError("formal_batch_requires_explicit_authorization")
    if args.formal_batch_index > MAX_AUTHORIZED_BATCH_INDEX:
        raise RuntimeError("batch_above_cycle_01_authorized_maximum")
    start = args.formal_batch_index * BATCH_SIZE
    return stable_ids[start:start + BATCH_SIZE]


def recovery_selection(
    args: argparse.Namespace,
    stable_ids: list[str],
    config: dict[str, Any],
) -> list[str]:
    if not args.confirm_recovery:
        raise RuntimeError("recovery_requires_confirmation")
    if not acceptance_pass(Path(args.artifact_root)):
        raise RuntimeError("v3_1_pilot_acceptance_not_pass")
    if args.cycle_id != "cycle_01" or args.cycle_start_batch != 0:
        raise RuntimeError("cycle_01_and_start_0_required")
    requested = list(dict.fromkeys(args.recovery_stable_id or []))
    if not requested:
        raise RuntimeError("empty_recovery_selection")
    first_round = set(
        stable_ids[: (MAX_AUTHORIZED_BATCH_INDEX + 1) * BATCH_SIZE]
    )
    if any(record_id not in first_round for record_id in requested):
        raise RuntimeError("recovery_id_outside_first_round")
    return requested


def write_batch_report(artifact: Path, run_id: str, mode: str, cycle_id: str | None,
                       batch_index: int | None, selected_ids: list[str],
                       run_rows: list[dict[str, Any]], state: dict[str, Any],
                       results_path: Path, sources: dict[str, dict[str, Any]],
                       prompt: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    report_dir = artifact / "batch_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    label = (
        "pilot_50" if mode == "pilot_50"
        else "selective_retry_5" if mode == "selective_retry_5"
        else f"{cycle_id}_batch_{batch_index:02d}" if mode == "formal_batch"
        else f"{cycle_id}_recovery_{run_id}"
    )
    history, latest, _ = load_history(results_path)
    final_rows = [latest[record_id] for record_id in selected_ids if record_id in latest]
    report_rows = (
        [
            row for row in history
            if row.get("mode") in {"formal_batch", "recovery_queue"}
            and row.get("cycle_id") == cycle_id
            and row.get("batch_index") == batch_index
        ]
        if mode == "formal_batch"
        else run_rows
    )
    api_rows = [
        row for row in report_rows if row.get("api_call_performed") is not False
    ]
    api_record_ids = {row["stable_id"] for row in api_rows}
    accepted_rows = [
        row for row in final_rows
        if active_success(row, sources[row["stable_id"]], prompt, config)
    ]
    accepted_ids = {row["stable_id"] for row in accepted_rows}
    failed_rows = [row for row in final_rows if row["stable_id"] not in accepted_ids]
    concept_counts = [len(row.get("final_concepts") or []) for row in accepted_rows]
    qc_counts = collections.Counter(
        flag for row in accepted_rows for flag in (row.get("audit_flags") or [])
    )
    status_counts = collections.Counter(row.get("quality_status") for row in accepted_rows)
    http_counts = collections.Counter(
        str(row.get("http_status") if row.get("http_status") is not None else "transport_error")
        for row in api_rows
    )
    technical_retry_rows = max(0, len(api_rows) - len(api_record_ids))
    technical_content_retry_rows = sum(
        row.get("api_attempt_kind") == "technical_content_retry" for row in api_rows
    )
    transport_retry_rows = sum(
        int(row.get("transport_attempt", 1) or 1) > 1 for row in api_rows
    )
    semantic_retry_violations = [
        row["stable_id"] for row in report_rows
        if row.get("quality_status") in {"RETRY_REQUIRED", "REJECTED_AFTER_RETRY"}
        or bool(
            set(row.get("repair_or_retry_reason") or [])
            & quality.QC_FLAGS
        )
    ]
    usage = {
        key: sum(int(row.get(key, 0) or 0) for row in api_rows)
        for key in ("prompt_tokens", "cached_tokens", "completion_tokens", "reasoning_token_usage")
    }
    latencies = sorted(float(row.get("latency", 0) or 0) for row in api_rows)
    cycle_rows = [
        row for row in history
        if row.get("mode") in {"formal_batch", "recovery_queue"}
        and row.get("cycle_id") == cycle_id
        and row.get("api_call_performed") is not False
    ] if cycle_id else []
    cutoff = datetime.now(timezone.utc).timestamp() - TWO_HOUR_WINDOW_SECONDS
    recent_cycle_calls = 0
    for row in cycle_rows:
        try:
            recent_cycle_calls += datetime.fromisoformat(row["timestamp"]).timestamp() > cutoff
        except (KeyError, TypeError, ValueError):
            pass
    modalities = collections.Counter(
        sources[record_id].get("modality", "unknown") for record_id in selected_ids
    )
    splits = collections.Counter(
        sources[record_id].get("split", "unknown") for record_id in selected_ids
    )
    api_success_rate = (
        sum(row.get("http_status") == 200 for row in api_rows) / len(api_rows)
        if api_rows else (1.0 if not api_record_ids else 0.0)
    )
    schema_rate = len(accepted_rows) / len(selected_ids) if selected_ids else 1.0
    cap8_ratio = (
        sum(value == 8 for value in concept_counts) / len(concept_counts)
        if concept_counts else 0.0
    )
    mean_concepts = sum(concept_counts) / len(concept_counts) if concept_counts else 0.0
    empty_or_count_failures = sum(
        bool({"empty_output", "concept_count_out_of_range"} & set(row.get("repair_or_retry_reason") or []))
        for row in report_rows
    )
    max_consecutive_429 = 0
    streak = 0
    for row in api_rows:
        if row.get("http_status") == 429:
            streak += 1
            max_consecutive_429 = max(max_consecutive_429, streak)
        else:
            streak = 0
    state_hash_consistent = state.get("results_hash") == sha_file(results_path)
    checks = {
        "api_success_rate_gte_99_5": api_success_rate >= 0.995,
        "json_schema_compliance_gte_99_5": schema_rate >= 0.995,
        "reasoning_tokens_zero": usage["reasoning_token_usage"] == 0,
        "empty_output_not_increased": empty_or_count_failures <= max(5, int(len(selected_ids) * 0.005)),
        "cap8_ratio_below_25_percent": cap8_ratio < 0.25,
        "mean_concepts_between_2_and_7": 2 <= mean_concepts <= 7,
        "results_resume_hash_consistent": state_hash_consistent,
        "http_403_zero": http_counts.get("403", 0) == 0,
        "consecutive_429_below_20": max_consecutive_429 < 20,
        "semantic_qc_never_retried": not semantic_retry_violations,
        "audit_policy_revision_at_least_1": all(
            int(row.get("audit_policy_revision", 0) or 0) >= 1 for row in accepted_rows
            if row.get("batch_index") != 0
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "HOLD",
        "run_status": state["status"],
        "run_id": run_id,
        "mode": mode,
        "cycle_id": cycle_id,
        "batch_index": batch_index,
        "selected": len(selected_ids),
        "completed": len(accepted_rows),
        "failed": len(failed_rows),
        "cache_hits": len(selected_ids) - len(api_record_ids),
        "api_attempts": len(api_rows),
        "technical_retry_count": technical_retry_rows,
        "technical_content_retry_count": technical_content_retry_rows,
        "transport_retry_count": transport_retry_rows,
        "http_status_distribution": dict(sorted(http_counts.items())),
        "api_success_rate": api_success_rate,
        "json_schema_compliance_rate": schema_rate,
        "reasoning_token_usage": usage["reasoning_token_usage"],
        "quality_status_distribution": dict(sorted(status_counts.items())),
        "concept_count_histogram": dict(sorted(collections.Counter(concept_counts).items())),
        "mean_concept_count": mean_concepts,
        "cap8_count": sum(value == 8 for value in concept_counts),
        "cap8_ratio": cap8_ratio,
        "qc_flag_counts": dict(sorted(qc_counts.items())),
        "modality_counts": dict(sorted(modalities.items())),
        "split_counts": dict(sorted(splits.items())),
        "empty_or_count_failure_events": empty_or_count_failures,
        "semantic_qc_retry_violation_ids": semantic_retry_violations,
        "usage": usage,
        "latency": {
            "mean_seconds": sum(latencies) / len(latencies) if latencies else None,
            "p95_seconds": latencies[min(len(latencies) - 1, int(len(latencies) * .95))] if latencies else None,
            "total_seconds": sum(latencies),
        },
        "quota": {
            "cycle_hard_api_call_limit": CYCLE_HARD_LIMIT,
            "cycle_api_calls_used": len(cycle_rows),
            "cycle_api_calls_remaining": CYCLE_HARD_LIMIT - len(cycle_rows),
            "two_hour_rolling_limit": TWO_HOUR_CALL_LIMIT,
            "two_hour_calls_at_report_time": recent_cycle_calls,
            "estimated_two_hour_quota_percent": recent_cycle_calls / (20_000 / 3) * 100,
            "manual_remaining_weekly_quota_percent": None,
        },
        "checks": checks,
        "failures": [
            {
                "stable_id": row["stable_id"],
                "http_status": row.get("http_status"),
                "error_type": row.get("error_type"),
                "quality_status": row.get("quality_status"),
            }
            for row in failed_rows
        ],
        "results_hash": sha_file(results_path),
        "resume_state_hash": sha_file(artifact / "v3_1_resume_state.json"),
        "prompt_hash": prompt["prompt_hash"],
        "schema_hash": prompt["schema_hash"],
        "run_config_hash": config["run_config_hash"],
        "audit_policy_revision": quality.AUDIT_POLICY_REVISION,
        "last_update": datetime.now(timezone.utc).isoformat(),
    }
    path = report_dir / f"{label}.json"
    if path.exists():
        archived = report_dir / "runs" / f"{label}_{run_id}.json"
        archived.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(archived, report)
    atomic_write_json(path, report)
    atomic_write_json(report_dir / f"{label}.hash.json", {
        "report_sha256": sha_file(path),
        "results_sha256": sha_file(results_path),
        "resume_state_sha256": sha_file(artifact / "v3_1_resume_state.json"),
    })
    markdown = (
        f"# {label} quality report\n\n"
        f"Status: **{report['status']}**\n\n"
        f"Completed: {report['completed']}/{report['selected']}; "
        f"API attempts: {report['api_attempts']}; technical retries: "
        f"{report['technical_retry_count']}; QC flagged: "
        f"{report['quality_status_distribution'].get('PASS_WITH_QC_FLAGS', 0)}.\n"
    )
    (report_dir / f"{label}.md").write_text(markdown, encoding="utf-8")
    return report

async def run(args: argparse.Namespace) -> int:
    artifact = Path(args.artifact_root)
    prompt, schema, config, pilot = load_contract(artifact)
    sources, stable_ids = load_sources(artifact, config)
    if args.offline_config_audit:
        print(json.dumps({
            "status": "CONFIG_PASS", "api_called": False,
            "base_url": BASE_URL, "request_model_id": REQUEST_MODEL,
            "reasoning_effort": EFFORT, "expected_routed_model": ROUTED_MODEL,
            "temperature": TEMPERATURE, "top_p_sent": False,
            "max_completion_tokens": MAX_TOKENS, "batch_size": BATCH_SIZE,
            "first_round_batch_range": [0, 17],
            "cycle_hard_limit": CYCLE_HARD_LIMIT, "full_mode_exists": False,
        }))
        return 0
    if args.offline_recertify:
        return offline_recertify(artifact, prompt, config, pilot, sources)

    if args.pilot_50:
        selected_ids = [row["stable_id"] for row in pilot["records"]]
        mode, cycle_id, batch_index = "pilot_50", None, None
    elif args.retry_repaired_five:
        raise RuntimeError("semantic_retry_disabled_by_policy_revision_1")
    elif args.formal_batch_index is not None:
        selected_ids = formal_selection(args, stable_ids, config)
        mode, cycle_id, batch_index = "formal_batch", args.cycle_id, args.formal_batch_index
    elif args.recovery_stable_id:
        selected_ids = recovery_selection(args, stable_ids, config)
        mode, cycle_id, batch_index = "recovery_queue", args.cycle_id, None
    else:
        raise RuntimeError("execution_mode_required")

    api_key, _ = common.resolve_credential(os.environ)
    endpoint = common.resolve_base_url(os.environ)
    if endpoint["base_url"] != BASE_URL:
        raise RuntimeError("endpoint_mismatch")

    results_path = artifact / "v3_1_results.jsonl"
    state_path = artifact / "v3_1_resume_state.json"
    history, latest, ordinals = load_history(results_path)
    rebuild_resume_state(
        state_path, results_path, selected_ids, latest, sources,
        prompt, config, mode, cycle_id, batch_index,
    )
    cached = {
        record_id for record_id in selected_ids
        if active_success(latest.get(record_id), sources[record_id], prompt, config)
    }
    if mode == "selective_retry_5":
        terminal = {record_id for record_id in selected_ids if selective_retry_terminal(latest.get(record_id))}
        pending = [record_id for record_id in selected_ids if record_id not in terminal]
    else:
        terminal = cached | {record_id for record_id in selected_ids if terminal_rejected(latest.get(record_id))}
        pending = [record_id for record_id in selected_ids if record_id not in terminal]

    if mode == "pilot_50":
        used = sum(row.get("mode") == "pilot_50" for row in history)
        limit = args.max_api_calls or 50
        if limit > 50:
            raise RuntimeError("pilot_api_limit_exceeds_50")
    elif mode == "selective_retry_5":
        used = sum(row.get("mode") == "selective_retry_5" and row.get("api_call_performed") is not False for row in history)
        limit = 5
        if used >= limit and pending:
            raise RuntimeError("selective_retry_5_api_limit_reached")
    else:
        used = sum(
            row.get("mode") in {"formal_batch", "recovery_queue"}
            and row.get("cycle_id") == cycle_id
            and row.get("api_call_performed") is not False
            for row in history
        )
        limit = CYCLE_HARD_LIMIT
        if mode == "recovery_queue" and args.max_api_calls is not None:
            limit = min(CYCLE_HARD_LIMIT, used + args.max_api_calls)
        if used >= limit:
            raise RuntimeError("cycle_18000_api_attempt_limit_reached")
    budget = common.AttemptBudget(used, limit)
    limiter = RollingWindowLimiter(args.requests_per_second, history, cycle_id)
    writer = ResultWriter(results_path)
    tracker = StateTracker(
        state_path, results_path, selected_ids, latest, sources,
        prompt, config, mode, cycle_id, batch_index,
    )
    client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL, timeout=args.timeout)
    stop, semaphore = asyncio.Event(), asyncio.Semaphore(args.concurrency)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    run_id = datetime.now(timezone.utc).strftime("v3.1-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:12]
    outcomes = collections.Counter()
    recent_rate_limits: collections.deque[tuple[bool, str, float]] = (
        collections.deque(maxlen=20)
    )
    consecutive_429 = 0

    def rate_limit_burst(row: dict[str, Any]) -> bool:
        nonlocal consecutive_429
        is_429 = row.get("http_status") == 429
        now = time.monotonic()
        consecutive_429 = consecutive_429 + 1 if is_429 else 0
        recent_rate_limits.append((is_429, row["stable_id"], now))
        count_429 = sum(value[0] for value in recent_rate_limits)
        short_ids = {
            record_id for hit, record_id, stamp in recent_rate_limits
            if hit and now - stamp <= 120
        }
        return bool(
            consecutive_429 >= 3
            or count_429 >= 5
            or len(short_ids) >= 3
        )

    async def worker(record_id: str) -> None:
        if stop.is_set():
            return
        async with semaphore:
            if stop.is_set():
                return
            retry_context = selective_retry_context(record_id, history, latest) if mode == "selective_retry_5" else None
            row = await request_one(
                client, limiter, budget, writer, tracker, sources[record_id],
                prompt, schema, config, run_id, ordinals[record_id],
                mode, cycle_id, batch_index, args.max_retries,
                retry_context, args.concurrency,
            )
        if row is None:
            stop.set()
            return
        outcomes[row["cache_status"]] += 1
        error = row.get("error_type")
        if row.get("http_status") == 429 and rate_limit_burst(row):
            outcomes["rate_limit_burst"] += 1
            stop.set()
        elif row.get("http_status") == 403:
            stop.set()
        elif row.get("http_status") in {400, 401, 404}:
            stop.set()
        elif row.get("quality_status") == "CONTRACT_VIOLATION":
            stop.set()

    await asyncio.gather(*(worker(record_id) for record_id in pending))
    await client.close()
    final_status = "STOPPED" if stop.is_set() else "RUN_COMPLETE"
    state = await tracker.finalize(final_status)
    final_history, final_latest, _ = load_history(results_path)
    run_rows = [row for row in final_history if row.get("run_id") == run_id]
    report = write_batch_report(
        artifact, run_id, mode, cycle_id, batch_index,
        selected_ids, run_rows, state, results_path, sources, prompt, config,
    )
    active = sum(
        active_success(final_latest.get(record_id), sources[record_id], prompt, config)
        for record_id in selected_ids
    )
    summary = {
        "status": final_status,
        "mode": mode,
        "selected": len(selected_ids),
        "active_success": active,
        "rejected_after_retry": sum(terminal_rejected(final_latest.get(record_id)) for record_id in selected_ids),
        "cache_hits_at_start": len(cached),
        "api_attempts_this_run": len(run_rows),
        "cycle_api_attempts_used": budget.used,
        "outcomes": dict(outcomes),
        "api_called": bool(run_rows),
        "stopped": stop.is_set(),
        "batch_acceptance": report["status"],
        "last_provider_error": (
            {
                key: run_rows[-1].get(key)
                for key in (
                    "http_status", "error_type",
                    "provider_error_code", "provider_error_type",
                    "provider_error_param", "provider_error_message",
                    "provider_request_id",
                )
            }
            if run_rows and run_rows[-1].get("error_type")
            else None
        ),
        "full_mode_exists": False,
    }
    print(json.dumps(summary))
    return 0 if not stop.is_set() and report["status"] == "PASS" else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--offline-config-audit", action="store_true")
    mode.add_argument("--offline-recertify", action="store_true")
    mode.add_argument("--pilot-50", action="store_true")
    mode.add_argument("--retry-repaired-five", action="store_true")
    mode.add_argument("--formal-batch-index", type=int)
    mode.add_argument("--recovery-stable-id", action="append")
    parser.add_argument("--confirm-selective-retry", action="store_true")
    parser.add_argument("--confirm-recovery", action="store_true")
    parser.add_argument("--confirm-formal-batch", action="store_true")
    parser.add_argument("--cycle-id")
    parser.add_argument("--cycle-start-batch", type=int)
    parser.add_argument("--artifact-root", default=str(ARTIFACT))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--requests-per-second", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-api-calls", type=int)
    args = parser.parse_args()
    if args.concurrency < 1 or args.max_retries < 0:
        raise SystemExit("invalid_safety_arguments")
    try:
        code = asyncio.run(run(args))
    except RuntimeError as error:
        print(json.dumps({"status": "CONFIG_OR_GATE_ERROR", "error": str(error), "api_called": False}))
        code = 2
    raise SystemExit(code)


if __name__ == "__main__":
    main()




