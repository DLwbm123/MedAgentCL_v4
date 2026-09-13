#!/usr/bin/env python3
"""Prepare the isolated MedicalSkill-CL Kimi K2.6-routing v3.1 contract."""

from __future__ import annotations

import collections
import difflib
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/root/MedAgentCL_v4")
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import prepare_kimi_k3_v3 as v3  # noqa: E402

V2 = ROOT / "artifacts/medicalskill_cl_kimi_k3_v2_pilot"
V3 = ROOT / "artifacts/medicalskill_cl_kimi_k3_v3_k2_6_routing"
OUT = ROOT / "artifacts/medicalskill_cl_kimi_k3_v3_1_k2_6_routing"
TARGET, PILOT, SEED = 35_000, 50, 43

SYSTEM_PROMPT = v3.SYSTEM_PROMPT
USER_TEMPLATE = """Extract 1 to 8 final English medical concepts from the caption in one pass.

Rules:
1. Never output an `uncertain:` prefix. For a concrete imaging finding directly weakened by possible, may, suggestive, or likely, output the concrete finding normally: `possible pulmonary nodule` becomes `pulmonary nodule`; `findings may suggest cerebral edema` becomes `cerebral edema`.
2. Do not convert downstream speculation into a positive concept. `may affect adjacent tissue` does not establish adjacent-tissue involvement. `could be benign or malignant` establishes neither benign nor malignant disease. For mutually exclusive differential diagnoses, keep their shared observed imaging finding rather than all candidate diseases.
3. Split independent modality and broad anatomy axes. `abdominal CT scan` becomes `computed tomography` and `abdomen`; `brain MRI` becomes `magnetic resonance imaging` and `brain`. Do not also retain the compound modality phrase.
4. Preserve localization that changes lesion meaning as one clinical phrase, including `pelvic bone lesion`, `right renal mass`, and `left upper lobe pulmonary nodule`.
5. Preserve indivisible medical terms, including `ground-glass opacity`, `pleural effusion`, and `invasive ductal carcinoma`.
6. Do not emit parent-child duplicates. With `pelvic bone lesion`, omit `lesion`; with `right renal mass`, omit `mass`; with `invasive ductal carcinoma`, omit `carcinoma`.
7. Remove ROI, bounding box, bbox, area ratio, highlighted or colored region, geometric image position, central position, cross-sectional view, anatomical association, and generic `disease`, `pathology`, `abnormality`, `finding`, or `process` concepts.
8. Keep only caption-supported diagnoses/pathology, salient abnormal findings, clinically localized findings, affected anatomy, modality/procedure/device, and discriminative observable attributes. Never infer an unstated disease.
9. Rank: specific diagnosis/pathology > salient abnormal finding > clinically localized finding > affected anatomy > modality/procedure/device > discriminative attribute > ordinary visible anatomy.
10. Never pad to eight. Each item must be a concise medical phrase. Merge synonyms and semantic duplicates.
11. Remove explicitly negated or absent concepts. Do not turn `no necrosis`, `absence of pleomorphism`, `absence of colloid`, `no evidence of metastasis`, or `without mitotic activity` into positive concepts. A missing sign such as `absent lung markings` is not itself a concept; retain a supported positive diagnosis or finding such as `pneumothorax`, `pleural air`, or `increased radiolucency`.
12. Never output mutually exclusive labels together, including high-grade with low-grade tumor, benign with malignant tumor or sarcoma/carcinoma, and well-differentiated with poorly differentiated. If the caption conflicts, prefer the repeatedly and explicitly supported specific diagnosis. If it cannot be resolved, keep only the shared observable morphology or a neutral lesion concept.
13. Limit ordinary anatomy to the lesion organ, the primary imaged organ, or anatomy required for lesion localization. Do not enumerate every visible background organ. In a PET caption listing brain, heart, liver, spleen, and gastrointestinal tract when uptake is confined to the upper abdomen, prioritize `positron emission tomography`, `increased fdg uptake`, `upper abdominal lesion` or `upper abdomen`, and an explicitly involved organ. Never pad with normal background organs.

<caption_data>
{caption}
</caption_data>"""

SCHEMA = v3.SCHEMA
NORMALIZATION = {
    **v3.NORMALIZATION,
    "version": "medicalskill-kimi-code-k2.6-routing-normalization-v3.1",
}
NEGATIVE_OUTPUT = __import__("re").compile(
    r"\b(?:no|not|without|absence(?: of)?|absent|negative for|free of|lack(?:ing)?|no evidence of)\b",
    __import__("re").I,
)
CONTRADICTIONS = (
    (("high-grade tumor", "high grade tumor", "high-grade malignancy", "high grade malignancy"),
     ("low-grade tumor", "low grade tumor", "low-grade malignancy", "low grade malignancy")),
    (("benign tumor", "benign neoplasm", "benign lesion"),
     ("malignant tumor", "malignant neoplasm", "malignancy", "sarcoma", "carcinoma")),
    (("well-differentiated", "well differentiated"),
     ("poorly differentiated", "poor differentiated", "poorly-differentiated")),
)


def contradictory(concepts: list[str]) -> bool:
    values = {v3.normalize_concept(value) if hasattr(v3, "normalize_concept") else " ".join(value.casefold().split()) for value in concepts}
    return any(
        any(left_term in values for left_term in left)
        and any(right_term in values for right_term in right)
        for left, right in CONTRADICTIONS
    )


def rank(record_id: str, purpose: str) -> str:
    return v3.sha_text(f"{SEED}|v3.1|{purpose}|{record_id}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_pilot(candidate: dict[str, dict[str, Any]], paired: dict[str, Any],
                 latest: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    paired_by_id = {row["stable_id"]: row for row in paired["records"]}
    selected: dict[str, set[str]] = collections.defaultdict(set)
    for record_id, meta in paired_by_id.items():
        result = latest[record_id]
        concepts = result["parsed_concepts"]
        if any(NEGATIVE_OUTPUT.search(value) for value in concepts):
            selected[record_id].add("v3_negative_output")
        if contradictory(concepts):
            selected[record_id].add("v3_mutually_exclusive_output")
        if len(concepts) == 8:
            selected[record_id].add("v3_exactly_8_concepts")
    mandatory = set(selected)
    if len(mandatory) > PILOT:
        raise RuntimeError(f"mandatory_v3_1_risks_exceed_50:{len(mandatory)}")

    all_modalities = {row["modality"] for row in paired["records"]}
    present = {paired_by_id[record_id]["modality"] for record_id in selected}
    for modality in sorted(all_modalities - present):
        pool = [
            record_id for record_id, row in paired_by_id.items()
            if row["modality"] == modality and record_id not in selected
        ]
        record_id = min(pool, key=lambda item: rank(item, f"coverage-{modality}"))
        selected[record_id].add("modality_coverage")

    pool = [record_id for record_id in paired_by_id if record_id not in selected]
    modality_counts = collections.Counter(paired_by_id[record_id]["modality"] for record_id in selected)
    while len(selected) < PILOT:
        record_id = min(
            pool,
            key=lambda item: (
                modality_counts[paired_by_id[item]["modality"]],
                rank(item, "stratified-random"),
            ),
        )
        pool.remove(record_id)
        selected[record_id].add("stratified_random")
        modality_counts[paired_by_id[record_id]["modality"]] += 1

    output = []
    for record_id in sorted(selected, key=lambda item: rank(item, "final")):
        source, prior, meta = candidate[record_id], latest[record_id], paired_by_id[record_id]
        output.append({
            "stable_id": record_id,
            "source_caption_sha256": source["source_caption_sha256"],
            "modality": source["modality"],
            "split": source["original_split"],
            "source_kind": source["source_kind"],
            "selection_reasons": sorted(selected[record_id]),
            "v3_concepts": prior["parsed_concepts"],
            "v3_concept_count": len(prior["parsed_concepts"]),
            "v3_result_is_reference_only": True,
            "v3_response_model_id": prior.get("response_model_id"),
            "paired_selection_stratum": meta["selection_stratum"],
        })
    if len(output) != PILOT or len({row["stable_id"] for row in output}) != PILOT:
        raise RuntimeError("pilot_50_selection_failed")
    if {row["modality"] for row in output} != all_modalities:
        raise RuntimeError("pilot_50_missing_modality")
    for record_id in mandatory:
        if record_id not in {row["stable_id"] for row in output}:
            raise RuntimeError(f"mandatory_risk_missing:{record_id}")
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    candidate_rows = read_jsonl(V2 / "concept_35k_candidate_manifest.jsonl")
    candidate = {row["stable_record_id"]: row for row in candidate_rows}
    if len(candidate) != TARGET:
        raise RuntimeError("frozen_35k_changed")

    shutil.copyfile(V3 / "concept_35k_stable_ids.txt", OUT / "concept_35k_stable_ids.txt")
    shutil.copyfile(V3 / "concept_35k_v3_input_manifest.jsonl", OUT / "concept_35k_v3_1_input_manifest.jsonl")
    stable_hash = v3.sha_file(OUT / "concept_35k_stable_ids.txt")
    input_hash = v3.sha_file(OUT / "concept_35k_v3_1_input_manifest.jsonl")
    v3_config = json.loads((V3 / "v3_run_config.json").read_text())
    if stable_hash != v3_config["stable_id_set_sha256"]:
        raise RuntimeError("stable_id_hash_changed")

    prompt_contract = {
        "prompt_version": "medicalskill-kimi-code-k2.6-routing-concepts-v3.1",
        "system_prompt": SYSTEM_PROMPT,
        "user_template": USER_TEMPLATE,
        "normalization_contract": NORMALIZATION,
    }
    prompt_hash = v3.sha_text(v3.canonical(prompt_contract))
    schema_hash = v3.sha_text(v3.canonical(SCHEMA))
    prompt = {
        **prompt_contract,
        "prompt_hash": prompt_hash,
        "schema_version": "medicalskill-kimi-code-k2.6-routing-schema-v3.1",
        "schema_hash": schema_hash,
        "request_model_id": v3.REQUEST_MODEL,
        "reasoning_effort": v3.EFFORT,
        "expected_routed_model": v3.ROUTED_MODEL,
        "endpoint_type": "kimi_code_subscription",
        "base_url": v3.BASE_URL,
        "max_completion_tokens": v3.MAX_TOKENS,
        "temperature_sent": False,
        "top_p_sent": False,
        "presence_penalty_sent": False,
        "frequency_penalty_sent": False,
        "thinking_field_sent": False,
        "structured_output": "json_schema_strict",
    }
    v3.write_json(OUT / "prompt_v3_1_manifest.json", prompt)
    v3.write_json(OUT / "kimi_prompt_schema_v3_1.json", SCHEMA)
    (OUT / "prompt_v3_1.md").write_text(
        f"# System\n\n{SYSTEM_PROMPT}\n\n# User template\n\n{USER_TEMPLATE}\n",
        encoding="utf-8",
    )
    diff = difflib.unified_diff(
        v3.USER_TEMPLATE.splitlines(),
        USER_TEMPLATE.splitlines(),
        fromfile="prompt_v3",
        tofile="prompt_v3_1",
        lineterm="",
    )
    (OUT / "prompt_v3_1_diff.md").write_text(
        "# Prompt v3 to v3.1\n\n```diff\n" + "\n".join(diff) + "\n```\n",
        encoding="utf-8",
    )

    config = {
        "contract_version": "medicalskill-kimi-code-k2.6-routing-run-v3.1",
        "request_model_id": v3.REQUEST_MODEL,
        "reasoning_effort": v3.EFFORT,
        "expected_routed_model": v3.ROUTED_MODEL,
        "endpoint_type": "kimi_code_subscription",
        "base_url": v3.BASE_URL,
        "max_completion_tokens": v3.MAX_TOKENS,
        "structured_output": "json_schema_strict",
        "omitted_request_fields": ["temperature", "top_p", "presence_penalty", "frequency_penalty", "thinking"],
        "prompt_hash": prompt_hash,
        "schema_hash": schema_hash,
        "stable_id_set_sha256": stable_hash,
        "input_manifest_sha256": input_hash,
        "v3_cache_reusable_as_v3_1": False,
        "v3_cache_role": "pilot_selection_and_quality_reference_only",
        "pilot_size": PILOT,
        "formal_batch_size": 1000,
        "formal_cycle_hard_limit": 18_000,
        "formal_total_records": 35_000,
        "formal_batches_total": 35,
        "current_cycle_id": "cycle_01",
        "current_cycle_first_batch": 0,
        "current_cycle_batch_count": 18,
    }
    config["run_config_hash"] = v3.sha_text(v3.canonical(config))
    v3.write_json(OUT / "run_config_v3_1.json", config)

    paired = json.loads((V3 / "v3_paired_pilot_200_manifest.json").read_text())
    v3_rows = read_jsonl(V3 / "v3_results.jsonl")
    latest = {}
    for row in v3_rows:
        if row.get("cache_status") == "active_v3_success":
            latest[row["stable_id"]] = row
    missing = {row["stable_id"] for row in paired["records"]} - set(latest)
    if missing:
        raise RuntimeError(f"v3_reference_missing:{len(missing)}")
    pilot = select_pilot(candidate, paired, latest)
    reason_counts = collections.Counter(reason for row in pilot for reason in row["selection_reasons"])
    v3.write_json(OUT / "v3_1_pilot_50_manifest.json", {
        "status": "PASS",
        "seed": SEED,
        "total": PILOT,
        "selection_interpretation": (
            "absence/absent/without selection applies to negated phrases present in v3 output concepts; "
            "caption-only negation is audited but is not mandatory because it would exceed 50."
        ),
        "caption_negation_count_in_source_200": sum(
            bool(__import__("re").search(r"\b(?:absence|absent|without|no evidence of)\b", latest[row["stable_id"]]["source_caption"], __import__("re").I))
            for row in paired["records"]
        ),
        "mandatory_reason_counts": dict(reason_counts),
        "distribution_by_modality": dict(collections.Counter(row["modality"] for row in pilot)),
        "v3_results_are_reference_not_v3_1_cache": True,
        "records": pilot,
    })

    plan = {
        "status": "PREPARED_NOT_EXECUTED",
        "measured_quota_assumptions": {
            "records": 200,
            "weekly_quota_percent": 1,
            "two_hour_quota_percent": 3,
        },
        "batch_size": 1000,
        "current_cycle": {
            "cycle_id": "cycle_01",
            "batch_indices": list(range(18)),
            "maximum_records": 18_000,
            "hard_api_attempt_limit": 18_000,
        },
        "next_cycle": {
            "cycle_id": "cycle_02",
            "batch_indices": list(range(18, 35)),
            "maximum_records": 17_000,
        },
        "pilot_50_cache_reused": True,
        "stop_on_403": True,
        "per_batch_outputs": [
            "usage", "failures", "manual_remaining_quota", "results_hash", "resume_state_hash"
        ],
        "single_uninterruptible_35k_mode_exists": False,
    }
    v3.write_json(OUT / "formal_batch_plan_v3_1.json", plan)

    commands = [
        "# Authorized targeted Pilot only",
        "scripts/medicalskill_kimi_k3/run_kimi_k3_v3_1_secure.sh --pilot-50 --max-api-calls 50",
        "python scripts/medicalskill_kimi_k3/audit_kimi_k3_v3_1.py",
        "",
        "# Future formal cycle_01. Do not execute until v3.1 acceptance is PASS.",
    ]
    commands.extend(
        f"scripts/medicalskill_kimi_k3/run_kimi_k3_v3_1_secure.sh --formal-batch-index {index} "
        f"--cycle-id cycle_01 --cycle-start-batch 0 --confirm-formal-batch"
        for index in range(18)
    )
    commands.extend(["", "# Next weekly cycle continues remaining batches"])
    commands.extend(
        f"scripts/medicalskill_kimi_k3/run_kimi_k3_v3_1_secure.sh --formal-batch-index {index} "
        f"--cycle-id cycle_02 --cycle-start-batch 18 --confirm-formal-batch"
        for index in range(18, 35)
    )
    (OUT / "v3_1_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    (OUT / "v3_1_rollback.md").write_text(
        "# Rollback\n\nStop v3.1 and preserve its append-only results. "
        "The v1/v2/v3 code, prompts, caches, and artifacts remain independent and unchanged.\n",
        encoding="utf-8",
    )

    tracked = sorted(path for path in OUT.iterdir() if path.is_file() and path.name != "v3_1_artifact_hashes.json")
    v3.write_json(OUT / "v3_1_artifact_hashes.json", {
        "status": "PASS",
        "files": {path.name: {"sha256": v3.sha_file(path), "bytes": path.stat().st_size} for path in tracked},
    })
    print(json.dumps({
        "status": "PASS",
        "candidate_total": TARGET,
        "pilot_total": len(pilot),
        "prompt_hash": prompt_hash,
        "schema_hash": schema_hash,
        "run_config_hash": config["run_config_hash"],
        "stable_id_set_sha256": stable_hash,
        "api_called": False,
    }))


if __name__ == "__main__":
    main()
