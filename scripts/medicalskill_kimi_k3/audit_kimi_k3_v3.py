#!/usr/bin/env python3
"""Offline acceptance audit for future v3 canary and paired Pilot results."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_kimi_k3_v3 as runner  # noqa: E402

ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/medicalskill_cl_kimi_k3_v3_k2_6_routing"
V2 = ROOT / "artifacts/medicalskill_cl_kimi_k3_v2_pilot"
MANUAL_FIELDS = (
    "caption_supported", "image_learnable", "generic_or_speculative",
    "semantic_duplicate", "important_lesion_omitted", "preferred_output", "reviewer_notes",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def proportion(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def is_generic_or_artifact(value: Any) -> bool:
    text = str(value)
    return bool(runner.FORBIDDEN.search(text)) or runner.normalize(text) in runner.GENERIC


def active_results(artifact: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    prompt, _, config, paired = runner.load_contract(artifact)
    sources, _ = runner.load_sources(artifact, config)
    _, latest, _ = runner.load_history(artifact / "v3_results.jsonl")
    active = {
        record_id: row for record_id, row in latest.items()
        if record_id in sources and runner.active_success(row, sources[record_id], prompt, config)
    }
    return active, paired, config, sources


def existing_manual(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["stable_id"]: row for row in csv.DictReader(handle)}


def write_human_table(path: Path, selected: list[str], active: dict[str, dict[str, Any]],
                      sources: dict[str, dict[str, Any]], paired: dict[str, Any]) -> None:
    old = existing_manual(path)
    selection = {row["stable_id"]: row for row in paired["records"]}
    v2_latest = {row["stable_record_id"]: row for row in read_jsonl(V2 / "v2_pilot_results.jsonl")}
    fields = [
        "stable_id", "modality", "split", "source_kind", "selection_stratum", "source_caption",
        "k3_v2_concepts", "k2_6_routing_v3_concepts", "response_model_id",
        "reasoning_content_present", "reasoning_token_usage", "latency", "completion_tokens",
        *MANUAL_FIELDS,
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record_id in selected:
            if record_id not in active:
                continue
            source, result, selected_row = sources[record_id], active[record_id], selection[record_id]
            previous = old.get(record_id, {})
            writer.writerow({
                "stable_id": record_id,
                "modality": source["modality"],
                "split": source["original_split"],
                "source_kind": source["source_kind"],
                "selection_stratum": selected_row["selection_stratum"],
                "source_caption": source["source_caption"],
                "k3_v2_concepts": json.dumps(v2_latest.get(record_id, {}).get("parsed_concepts") or [], ensure_ascii=False),
                "k2_6_routing_v3_concepts": json.dumps(result["parsed_concepts"], ensure_ascii=False),
                "response_model_id": result.get("response_model_id"),
                "reasoning_content_present": result.get("reasoning_content_present"),
                "reasoning_token_usage": result.get("reasoning_token_usage"),
                "latency": result.get("latency"),
                "completion_tokens": result.get("completion_tokens"),
                **{field: previous.get(field, "") for field in MANUAL_FIELDS},
            })


def canary_audit(artifact: Path, active: dict[str, dict[str, Any]], paired: dict[str, Any]) -> dict[str, Any]:
    record_id = paired["canary_stable_id"]
    result = active.get(record_id)
    gates = {
        "exact_canary_id_success": result is not None,
        "request_model_k3": bool(result) and result.get("request_model_id") == "k3",
        "reasoning_effort_none": bool(result) and result.get("reasoning_effort") == "none",
        "json_and_1_to_8": bool(result) and 1 <= len(result.get("parsed_concepts") or []) <= 8,
        "reasoning_content_absent": bool(result) and result.get("reasoning_content_present") is False,
        "reasoning_tokens_zero": bool(result) and int(result.get("reasoning_token_usage", 0) or 0) == 0,
        "http_200": bool(result) and result.get("http_status") == 200,
    }
    output = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "stable_id": record_id,
        "gates": gates,
        "response_model_id": result.get("response_model_id") if result else None,
        "routing_note": (
            "A response model alias of k3 does not alone disprove routing; reasoning_effort, "
            "reasoning content/tokens, and the provider routing contract are evaluated together."
        ),
    }
    write_json(artifact / "v3_canary_acceptance.json", output)
    return output


def paired_audit(artifact: Path, active: dict[str, dict[str, Any]], paired: dict[str, Any],
                 sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    selected = [row["stable_id"] for row in paired["records"]]
    rows = [active[record_id] for record_id in selected if record_id in active]
    result_path = artifact / "v3_results.jsonl"
    attempts = read_jsonl(result_path)
    selected_attempts = [row for row in attempts if row.get("stable_id") in set(selected)]
    uncertain = sum(any(str(value).casefold().startswith("uncertain:") for value in row["parsed_concepts"]) for row in rows)
    artifact_hits = sum(any(is_generic_or_artifact(value) for value in row["parsed_concepts"]) for row in rows)
    duplicate_hits = sum(len(row["parsed_concepts"]) != len({runner.normalize(value) for value in row["parsed_concepts"]}) for row in rows)
    prompt_tokens = [int(row.get("prompt_tokens", 0) or 0) for row in rows]
    cached_tokens = [int(row.get("cached_tokens", 0) or 0) for row in rows]
    completion = [int(row.get("completion_tokens", 0) or 0) for row in rows]
    total_tokens = [prompt + output for prompt, output in zip(prompt_tokens, completion)]
    latency = [float(row.get("latency", 0) or 0) for row in rows]
    gates = {
        "api_success_rate_ge_99_5pct": proportion(len(rows), 200) >= .995,
        "json_parse_rate_100pct": len(rows) == 200,
        "concept_count_1_to_8_rate_100pct": len(rows) == 200 and all(1 <= len(row["parsed_concepts"]) <= 8 for row in rows),
        "empty_rate_lt_0_5pct": all(row["parsed_concepts"] for row in rows),
        "generic_roi_artifact_rate_le_1pct": proportion(artifact_hits, 200) <= .01,
        "semantic_duplicate_rate_le_0_5pct": proportion(duplicate_hits, 200) <= .005,
        "uncertain_prefix_absent": uncertain == 0,
        "reasoning_content_absent": len(rows) == 200 and all(row.get("reasoning_content_present") is False for row in rows),
        "reasoning_tokens_zero": len(rows) == 200 and all(int(row.get("reasoning_token_usage", 0) or 0) == 0 for row in rows),
    }
    human_path = artifact / "v3_paired_human_audit_200.csv"
    write_human_table(human_path, selected, active, sources, paired)
    manual = existing_manual(human_path)
    reviewed = [row for row in manual.values() if all(row.get(field, "").strip() for field in MANUAL_FIELDS[:-2])]
    caption_supported = sum(row["caption_supported"].strip().casefold() in {"yes", "1", "true"} for row in reviewed)
    important_omission = sum(row["important_lesion_omitted"].strip().casefold() in {"yes", "1", "true"} for row in reviewed)
    manual_gates = {
        "all_200_reviewed": len(reviewed) == 200,
        "caption_supported_precision_ge_98pct": len(reviewed) == 200 and proportion(caption_supported, 200) >= .98,
        "no_obvious_important_lesion_omission": len(reviewed) == 200 and important_omission == 0,
    }
    machine_pass = all(gates.values())
    status = "PASS" if machine_pass and all(manual_gates.values()) else "PENDING_HUMAN_REVIEW" if machine_pass else "FAIL"
    v2_latest = {row["stable_record_id"]: row for row in read_jsonl(V2 / "v2_pilot_results.jsonl")}
    v2_counts = [len(v2_latest[item].get("parsed_concepts") or []) for item in selected if item in v2_latest]
    output = {
        "status": status,
        "selected": 200,
        "active_success": len(rows),
        "attempts": len(selected_attempts),
        "machine_gates": gates,
        "manual_gates": manual_gates,
        "manual_reviewed": len(reviewed),
        "metrics": {
            "mean_concept_count_v3": statistics.fmean(len(row["parsed_concepts"]) for row in rows) if rows else None,
            "mean_concept_count_k3_v2": statistics.fmean(v2_counts) if v2_counts else None,
            "cap8_count_v3": sum(len(row["parsed_concepts"]) == 8 for row in rows),
            "uncertain_prefix_count": uncertain,
            "generic_roi_artifact_count": artifact_hits,
            "semantic_duplicate_count": duplicate_hits,
            "mean_latency": statistics.fmean(latency) if latency else None,
            "mean_prompt_tokens": statistics.fmean(prompt_tokens) if prompt_tokens else None,
            "mean_cached_tokens": statistics.fmean(cached_tokens) if cached_tokens else None,
            "mean_uncached_prompt_tokens": statistics.fmean(
                max(prompt - cached, 0) for prompt, cached in zip(prompt_tokens, cached_tokens)
            ) if prompt_tokens else None,
            "mean_completion_tokens": statistics.fmean(completion) if completion else None,
            "mean_total_tokens_per_request": statistics.fmean(total_tokens) if total_tokens else None,
            "total_prompt_tokens": sum(prompt_tokens),
            "total_cached_tokens": sum(cached_tokens),
            "total_completion_tokens": sum(completion),
            "total_tokens": sum(total_tokens),
            "response_model_distribution": dict(collections.Counter(row.get("response_model_id") for row in rows)),
        },
        "human_audit_path": str(human_path),
    }
    write_json(artifact / "v3_paired_pilot_acceptance.json", output)
    write_json(artifact / "v3_paired_comparison_report.json", output)
    (artifact / "v3_paired_comparison_report.md").write_text(
        f"# K3-thinking vs K2.6-routing non-thinking paired Pilot\n\n"
        f"Status: **{status}**\n\nActive success: {len(rows)}/200. "
        f"Human reviewed: {len(reviewed)}/200. Full 35K remains blocked unless status is PASS.\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default=str(ARTIFACT))
    parser.add_argument("--stage", choices=("canary", "paired"), required=True)
    args = parser.parse_args()
    artifact = Path(args.artifact_root)
    active, paired, _, sources = active_results(artifact)
    output = canary_audit(artifact, active, paired) if args.stage == "canary" else paired_audit(artifact, active, paired, sources)
    print(json.dumps({"status": output["status"], "stage": args.stage, "api_called": False}))


if __name__ == "__main__":
    main()
