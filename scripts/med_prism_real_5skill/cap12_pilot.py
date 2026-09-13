#!/usr/bin/env python3
"""Run and audit the fixed-revision max-12 Concept saturation pilot."""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/root/MedAgentCL_v4")
DATA = Path("/remote-home/wangbomin/MedicalSkill-CL-v1.1")
ARTIFACT = ROOT / "artifacts/medicalskill_cl_v1_1_and_medprism_smoke"
MODEL_ID = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SNAPSHOT = Path(os.environ.get(
    "MEDICALSKILL_QWEN3_TEXT_SNAPSHOT",
    str(Path("/remote-home/wangbomin/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots") / REVISION),
))
PROMPT_VERSION = "medicalskill_concept_qwen3_v3_max12_pilot"
SYSTEM_PROMPT = """You extract medical concepts only from the supplied caption. Return one JSON object with a concepts list, ordered by clinical importance. Include only explicitly caption-supported concepts with medical value. Do not fill a quota. Exclude spatial position templates, quadrant/portion/region/area, ROI or annotation geometry, and generic terms such as diagnosis, disease process, pathological process, generic pathology, or generic abnormality. Concepts 9 through 12 are permitted only when each adds distinct clinically useful information beyond the first eight. Preserve uncertainty, negation, and laterality. Every object must contain canonical, type, status, laterality, and an exact supporting mention. Allowed types: anatomy, finding, pathology, modality, device, procedure, attribute. Allowed status: present, uncertain, negated. Return at most 12 concepts and JSON only."""
USER_TEMPLATE = "Caption:\n{caption}\n\nSchema: {{\"concepts\":[{{\"canonical\":\"\",\"type\":\"anatomy|finding|pathology|modality|device|procedure|attribute\",\"status\":\"present|uncertain|negated\",\"laterality\":\"left|right|bilateral|midline|unspecified\",\"mention\":\"exact supporting phrase\"}}]}}"
ALLOWED_TYPES = {"anatomy", "finding", "pathology", "modality", "device", "procedure", "attribute"}
ALLOWED_STATUS = {"present", "uncertain", "negated"}
ALLOWED_LATERALITY = {"left", "right", "bilateral", "midline", "unspecified"}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def prompt_hash() -> str:
    return hashlib.sha256((PROMPT_VERSION + "\n" + SYSTEM_PROMPT + "\n" + USER_TEMPLATE).encode()).hexdigest()


def parse(raw: str, caption: str) -> list[dict[str, str]]:
    values = None
    try:
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            values = json.loads(match.group(0)).get("concepts")
    except json.JSONDecodeError:
        values = None
    if not isinstance(values, list):
        values = []
        for fragment in re.findall(r"\{[^{}]*\}", raw, re.S):
            try:
                candidate = json.loads(fragment)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "canonical" in candidate:
                values.append(candidate)
        if not values:
            raise ValueError("no complete concept objects in raw response")
    result = []
    seen = set()
    caption_norm = norm(caption)
    for value in values:
        if not isinstance(value, dict):
            continue
        item = {
            "canonical": re.sub(r"\s+", " ", str(value.get("canonical") or "")).strip(),
            "type": str(value.get("type") or "").casefold(),
            "status": str(value.get("status") or "").casefold(),
            "laterality": str(value.get("laterality") or "unspecified").casefold(),
            "mention": re.sub(r"\s+", " ", str(value.get("mention") or "")).strip(),
        }
        key = norm(item["canonical"])
        if not key or key in seen or item["type"] not in ALLOWED_TYPES or item["status"] not in ALLOWED_STATUS or item["laterality"] not in ALLOWED_LATERALITY:
            continue
        if not norm(item["mention"]) or norm(item["mention"]) not in caption_norm:
            continue
        seen.add(key)
        result.append(item)
        if len(result) == 12:
            break
    return result


def select(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    eligible = [row for row in rows if len(row.get("concepts_before_v3") or row.get("concepts") or []) == 8]
    rng = random.Random(seed)
    strata: dict[tuple[str, str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in eligible:
        source = str((row.get("source_metadata") or {}).get("source_dataset") or "unknown")
        strata[(str(row.get("split")), str(row.get("modality")), str(row.get("source_kind")), source)].append(row)
    buckets = []
    for key in sorted(strata):
        values = strata[key]
        rng.shuffle(values)
        buckets.append(values)
    selected = []
    while len(selected) < count and any(buckets):
        next_buckets = []
        for values in buckets:
            if values and len(selected) < count:
                selected.append(values.pop())
            if values:
                next_buckets.append(values)
        buckets = next_buckets
    if len(selected) != count:
        raise RuntimeError(f"Only selected {len(selected)} of {count}")
    return selected


def load_auditor():
    path = ROOT / "scripts/med_prism_real_5skill/concept_v3_auditor.py"
    spec = importlib.util.spec_from_file_location("cap12_independent_auditor", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_report(selected: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
    auditor = load_auditor()
    by_id = {str(row["id"]): row for row in outputs}
    extra_total = supported = useful = generic = artifact = 0
    rows_with_useful_extra = 0
    exact_prefix = 0
    top8_overlap = []
    modality: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    details = []
    for source in selected:
        result = by_id[str(source["id"])]
        old = [norm(item["canonical"]) for item in (source.get("concepts_before_v3") or source["concepts"])]
        new = result.get("concepts") or []
        new_names = [norm(item["canonical"]) for item in new]
        exact_prefix += new_names[:8] == old
        top8_overlap.append(len(set(old) & set(new_names[:8])) / 8)
        row_useful = 0
        extra_details = []
        for item in new[8:12]:
            extra_total += 1
            is_supported = norm(item["mention"]) in norm(source["source_caption"])
            supported += is_supported
            audit = auditor.audit_row(source, [item])
            reasons = audit["reason_counts"]
            is_generic = bool(reasons.get("generic_nonclinical_concept"))
            is_artifact = bool(reasons.get("pure_positional_concept") or reasons.get("annotation_or_roi_concept"))
            is_useful = is_supported and not is_generic and not is_artifact and audit["status"] == "PASS"
            generic += is_generic
            artifact += is_artifact
            useful += is_useful
            row_useful += is_useful
            extra_details.append({"concept": item, "supported": is_supported, "clinically_useful": is_useful, "generic": is_generic, "artifact": is_artifact, "audit_reasons": reasons})
        rows_with_useful_extra += row_useful > 0
        key = str(source.get("modality") or "unknown")
        modality[key]["rows"] += 1
        modality[key]["extra"] += len(new[8:12])
        modality[key]["useful_extra"] += row_useful
        details.append({"id": source["id"], "modality": key, "max8": source["concepts"], "max12": new, "top8_overlap": top8_overlap[-1], "extra": extra_details})
    ratio = rows_with_useful_extra / len(selected)
    cap8_sufficient = ratio < 0.20
    return {
        "status": "PASS" if len(outputs) == len(selected) and all(not row.get("error") for row in outputs) else "FAIL",
        "decision": "retain_cap_8" if cap8_sufficient else "cap_8_insufficient_stop_freeze",
        "cap8_definition": "At most 8 core concepts ranked by clinical importance" if cap8_sufficient else None,
        "threshold": 0.20, "sample_count": len(selected),
        "rows_with_at_least_one_supported_clinically_useful_extra": rows_with_useful_extra,
        "rows_with_useful_extra_ratio": ratio,
        "top8_exact_order_stability": exact_prefix / len(selected),
        "top8_mean_set_overlap": sum(top8_overlap) / len(top8_overlap),
        "extra_concept_count": extra_total, "extra_supported": supported,
        "extra_unsupported": extra_total - supported, "extra_clinically_useful": useful,
        "extra_generic": generic, "extra_artifact": artifact,
        "per_modality": {key: {**dict(value), "useful_extra_per_row": value["useful_extra"] / value["rows"]} for key, value in sorted(modality.items())},
        "rows": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reparse-existing", action="store_true")
    args = parser.parse_args()
    pilot_dir = DATA / "_concept/cap12_pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    input_path = pilot_dir / "cap12_input_1000.jsonl"
    if input_path.is_file() and not args.prepare_only:
        selected = list(iter_jsonl(input_path))
        if len(selected) != args.sample_count:
            raise RuntimeError(f"Frozen pilot input count mismatch: {len(selected)}")
    else:
        rows = []
        for part in (DATA / "_concept/full_output_part_00.jsonl", DATA / "_concept/full_output_part_01.jsonl"):
            rows.extend(iter_jsonl(part))
        selected = select(rows, args.sample_count, 42)
        input_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    if args.prepare_only:
        print(json.dumps({"status": "PASS", "selected": len(selected), "input": str(input_path)}, indent=2))
        return 0
    if not SNAPSHOT.is_dir():
        raise FileNotFoundError(f"Missing fixed snapshot: {SNAPSHOT}")
    output_path = pilot_dir / "cap12_output_1000.jsonl"
    partial = output_path.with_suffix(".jsonl.partial")
    existing = {str(row["id"]): row for row in iter_jsonl(partial)} if args.resume and partial.is_file() else {}
    if args.reparse_existing:
        source_by_id = {str(row["id"]): row for row in selected}
        for key, result in existing.items():
            try:
                result["concepts"] = parse(result["raw_response"], source_by_id[key]["source_caption"])
                result["error"] = ""
                result["deterministic_truncation_repair"] = True
            except Exception as exc:
                result["concepts"] = []
                result["error"] = f"{type(exc).__name__}: {exc}"
        partial.write_text("".join(json.dumps(existing[str(row["id"])], ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), local_files_only=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(SNAPSHOT), local_files_only=True, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto").eval()
    pending = [row for row in selected if str(row["id"]) not in existing]
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        prompts = []
        for row in batch:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_TEMPLATE.format(caption=row["source_caption"])}]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**encoded, do_sample=False, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id)
        raw_values = tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        with partial.open("a", encoding="utf-8") as handle:
            for source, raw in zip(batch, raw_values):
                error = ""
                try:
                    concepts = parse(raw, source["source_caption"])
                except Exception as exc:
                    concepts = []
                    error = f"{type(exc).__name__}: {exc}"
                result = {"id": source["id"], "concepts": concepts, "raw_response": raw, "error": error, "model_id": MODEL_ID, "revision": REVISION, "prompt_version": PROMPT_VERSION, "prompt_hash": prompt_hash(), "do_sample": False, "seed": 42}
                existing[str(source["id"])] = result
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        print(f"completed={len(existing)}/{len(selected)}", flush=True)
    outputs = [existing[str(row["id"])] for row in selected]
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs), encoding="utf-8")
    report = build_report(selected, outputs)
    (ARTIFACT / "cap8_vs_cap12_pilot.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (ARTIFACT / "cap8_vs_cap12_pilot_report.md").write_text(
        "# Cap-8 vs cap-12 pilot\n\n"
        f"Status: **{report['status']}**\n\nDecision: **{report['decision']}**\n\n"
        f"Sample: {report['sample_count']}; rows with useful extra concepts: {report['rows_with_at_least_one_supported_clinically_useful_extra']} ({100 * report['rows_with_useful_extra_ratio']:.2f}%).\n\n"
        f"Top-8 exact order stability: {100 * report['top8_exact_order_stability']:.2f}%; mean set overlap: {100 * report['top8_mean_set_overlap']:.2f}%.\n\n"
        f"Extra concepts: {report['extra_concept_count']}; supported: {report['extra_supported']}; clinically useful: {report['extra_clinically_useful']}; generic: {report['extra_generic']}; artifact: {report['extra_artifact']}.\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0 if report["status"] == "PASS" and report["decision"] == "retain_cap_8" else 1


if __name__ == "__main__":
    raise SystemExit(main())
