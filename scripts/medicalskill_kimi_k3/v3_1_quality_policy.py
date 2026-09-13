#!/usr/bin/env python3
"""Audit-only quality policy for Kimi K2.6-routing v3.1 outputs.

This module never rewrites, deletes, or inserts a model concept. It only parses
strict output and records semantic QC flags. Only technical parse failures may
request one same-contract retry.
"""
from __future__ import annotations

import json
import re
from typing import Any

NEGATED = re.compile(
    r"\b(?:no|not|without|absence(?: of)?|absent|negative for|free of|lack(?:ing)?|no evidence of)\b",
    re.I,
)
GEOMETRIC = re.compile(
    r"\b(?:roi|region of interest|bounding box|bbox|area ratio|highlighted region|colored box|"
    r"cross-sectional view|coronal section|sagittal section|axial section|central position|"
    r"upper[- ]?left quadrant|upper[- ]?right quadrant|lower[- ]?left quadrant|lower[- ]?right quadrant|"
    r"left[- ]?center|right[- ]?center)\b",
    re.I,
)
GENERIC_EXACT = {
    "disease", "pathology", "abnormality", "finding", "process", "disease process",
    "pathological process", "anatomical association", "region of interest", "lesion",
    "mass", "tumor", "medical finding", "abnormal tissue",
}
BACKGROUND_ORGANS = {
    "brain", "heart", "liver", "spleen", "kidney", "kidneys", "lung", "lungs",
    "stomach", "bowel", "intestine", "intestines", "gastrointestinal tract", "pancreas",
    "bladder", "colon", "small bowel", "large bowel", "spine",
}
CONTRADICTIONS = (
    (("high-grade", "high grade"), ("low-grade", "low grade")),
    (("benign tumor", "benign neoplasm", "benign lesion"),
     ("malignant tumor", "malignant neoplasm", "malignancy", "sarcoma", "carcinoma", "cancer")),
    (("well-differentiated", "well differentiated"),
     ("poorly differentiated", "poor differentiated", "poorly-differentiated")),
    (("high-grade", "high grade"), ("well-differentiated", "well differentiated")),
)
DIAGNOSIS_TERMS = re.compile(
    r"\b(?:carcinoma|sarcoma|melanoma|glioma|glioblastoma|cancer|adenoma|granuloma|"
    r"neurofibroma|pneumothorax|atelectasis|lymphoma|metastasis|malignancy|neoplasm|tumou?r)\b",
    re.I,
)
STOPWORDS = {
    "high", "grade", "low", "well", "poorly", "differentiated", "malignant", "benign",
    "tumor", "tumour", "cancer", "lesion", "disease", "process", "the", "and", "of",
}
QC_FLAGS = {
    "negated_concept", "mutually_exclusive_diagnosis", "mutually_exclusive_differential",
    "parent_child_duplicate", "semantic_duplicate", "background_organ_enumeration",
    "geometric_view_artifact", "unsupported_specific_diagnosis", "generic_only",
    "cap8_review", "unsupported_or_over-specific_concept", "uncertain_prefix",
}
# Compatibility symbol: semantic QC flags are never retry reasons.
RETRY_FLAGS: set[str] = set()
RETRYABLE_PARSE_ERRORS = {
    "empty_output", "invalid_json", "markdown_code_fence", "schema_object",
    "concepts_not_list", "concept_count_out_of_range", "concept_nonempty_string",
    "truncated_output",
}
AUDIT_POLICY_REVISION = 1


def normalize(value: str) -> str:
    return " ".join(value.split()).strip(" ;,.").casefold()


def _contains_any(value: str, terms: tuple[str, ...]) -> bool:
    return any(term in value for term in terms)


def _parent_child(concepts: list[str]) -> list[str]:
    hits: set[str] = set()
    heads = {"tumor", "carcinoma", "cancer", "adenoma", "lesion", "mass", "malignancy", "neoplasm"}
    for i, left in enumerate(concepts):
        lt = set(re.findall(r"[a-z0-9]+", left))
        for right in concepts[i + 1:]:
            rt = set(re.findall(r"[a-z0-9]+", right))
            # General anatomy is intentionally emitted separately from a localized
            # lesion phrase (for example, ``lung`` + ``lung lesion``). Only flag
            # duplication when the parent is itself a clinical lesion concept.
            if lt < rt and lt & heads:
                hits.add(f"{left}::{right}")
            elif rt < lt and rt & heads:
                hits.add(f"{left}::{right}")
    known_relations = (
        ("neuroendocrine tumor", "grade 2 neuroendocrine tumor"),
        ("nerve sheath tumor", "neurofibroma"),
        ("breast cancer", "high-grade malignancy"),
        ("malignancy", "breast carcinoma"),
    )
    values = set(concepts)
    for left, right in known_relations:
        if left in values and right in values:
            hits.add(f"{left}::{right}")
    return sorted(hits)


def _contradictions(concepts: list[str]) -> list[str]:
    hits = []
    for left, right in CONTRADICTIONS:
        lv = sorted(value for value in concepts if _contains_any(value, left))
        rv = sorted(value for value in concepts if _contains_any(value, right))
        if lv and rv:
            hits.append(f"{lv[0]}::{rv[0]}")
    return hits


def _differential_hits(concepts: list[str]) -> list[str]:
    values = set(concepts)
    hits = []
    if {"villous adenoma", "tubular adenoma"} <= values:
        hits.append("villous adenoma::tubular adenoma")
    if "granuloma" in values and ({"nerve sheath tumor", "neurofibroma"} & values):
        hits.append("nerve_sheath_or_neurofibroma::granuloma")
    diagnosis_like = [value for value in concepts if DIAGNOSIS_TERMS.search(value)]
    unrelated = []
    for value in diagnosis_like:
        tokens = set(re.findall(r"[a-z0-9]+", value)) - STOPWORDS
        if tokens and all(not (tokens & (set(re.findall(r"[a-z0-9]+", other)) - STOPWORDS)) for other in diagnosis_like if other != value):
            unrelated.append(value)
    if len(unrelated) >= 3:
        hits.append("multiple_unresolved_differential_diagnoses")
    return sorted(set(hits))


def _unsupported_specific(concepts: list[str], caption: str) -> list[str]:
    text = normalize(caption)
    hits = []
    for concept in concepts:
        if not DIAGNOSIS_TERMS.search(concept) or concept in GENERIC_EXACT or concept in text:
            continue
        tokens = [token for token in re.findall(r"[a-z0-9]+", concept) if len(token) >= 5 and token not in STOPWORDS]
        if tokens and not any(token in text for token in tokens):
            hits.append(concept)
    return hits


def audit_concepts(concepts: list[str], caption: str = "", modality: str = "") -> dict[str, Any]:
    flags: list[str] = []
    details: dict[str, Any] = {}
    if len(concepts) != len(set(concepts)):
        flags.append("semantic_duplicate")
    if any(value.startswith("uncertain:") for value in concepts):
        flags.append("uncertain_prefix")
    negated = [value for value in concepts if NEGATED.search(value)]
    if negated:
        flags.append("negated_concept")
        details["negated_concept"] = negated
    geometric = [value for value in concepts if GEOMETRIC.search(value)]
    if geometric:
        flags.append("geometric_view_artifact")
        details["geometric_view_artifact"] = geometric
    contradictions = _contradictions(concepts)
    if contradictions:
        flags.append("mutually_exclusive_diagnosis")
        details["mutually_exclusive_diagnosis"] = contradictions
    parent_child = _parent_child(concepts)
    if parent_child:
        flags.append("parent_child_duplicate")
        details["parent_child_duplicate"] = parent_child
    differential = _differential_hits(concepts)
    if differential:
        flags.append("mutually_exclusive_differential")
        details["mutually_exclusive_differential"] = differential
    organ_count = sum(value in BACKGROUND_ORGANS for value in concepts)
    if organ_count >= 4 or (len(concepts) == 8 and organ_count >= 3):
        flags.append("background_organ_enumeration")
        details["background_organ_enumeration"] = [value for value in concepts if value in BACKGROUND_ORGANS]
    if concepts and all(value in GENERIC_EXACT for value in concepts):
        flags.append("generic_only")
    if len(concepts) == 8:
        flags.append("cap8_review")
    unsupported = _unsupported_specific(concepts, caption)
    if unsupported:
        flags.append("unsupported_specific_diagnosis")
        details["unsupported_specific_diagnosis"] = unsupported
        flags.append("unsupported_or_over-specific_concept")
        details["unsupported_or_over-specific_concept"] = unsupported
    flags = sorted(set(flags))
    return {
        "audit_flags": flags,
        "audit_details": details,
        "retry_reasons": [],
        "modality": modality,
        "audit_policy_revision": AUDIT_POLICY_REVISION,
    }


def parse_raw_response(raw: str, finish_reason: str | None = None) -> dict[str, Any]:
    errors = []
    concepts: list[str] = []
    if finish_reason == "length":
        errors.append("truncated_output")
    if not raw.strip():
        errors.append("empty_output")
    elif "```" in raw:
        errors.append("markdown_code_fence")
    else:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            errors.append("invalid_json")
        else:
            if not isinstance(value, dict) or set(value) != {"concepts"}:
                errors.append("schema_object")
            elif not isinstance(value["concepts"], list):
                errors.append("concepts_not_list")
            elif not 1 <= len(value["concepts"]) <= 8:
                errors.append("concept_count_out_of_range")
            else:
                for item in value["concepts"]:
                    if not isinstance(item, str) or not item.strip():
                        errors.append("concept_nonempty_string")
                        concepts = []
                        break
                    concepts.append(normalize(item))
    return {"raw_model_concepts": concepts, "parse_errors": sorted(set(errors))}


def evaluate_response(raw: str, caption: str, modality: str, finish_reason: str | None) -> dict[str, Any]:
    parsed = parse_raw_response(raw, finish_reason)
    if parsed["parse_errors"]:
        return {
            **parsed,
            "audit_flags": [],
            "audit_details": {},
            "retry_reasons": sorted(set(parsed["parse_errors"]) & RETRYABLE_PARSE_ERRORS),
            "audit_policy_revision": AUDIT_POLICY_REVISION,
            "technical_parse_failure": True,
        }
    audited = audit_concepts(parsed["raw_model_concepts"], caption, modality)
    return {**parsed, **audited, "technical_parse_failure": False}
