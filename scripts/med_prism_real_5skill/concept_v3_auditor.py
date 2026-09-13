#!/usr/bin/env python3
"""Independent semantic auditor for MedicalSkill-CL-v1.1 Concept v3."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any


def audit_normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


POSITION_ONLY = re.compile(
    r"^(?:(?:upper|lower|left|right|central|center|middle|peripheral|anterior|"
    r"posterior|medial|lateral)\s+)*(?:quadrant|portion|region|area|position|"
    r"location|side|view)$|^(?:upper|lower|left|right|central|center|middle|"
    r"peripheral|quadrant|position|location)$"
)
ROI_TERMS = re.compile(
    r"\b(?:region of interest|roi|bounding box|highlighted (?:region|area)|"
    r"area ratio|image annotation|colored overlay)\b"
)
GENERIC_EXACT = {
    "diagnosis", "disease", "disease process", "pathological process",
    "pathological condition", "pathological change", "generic pathology",
    "generic disease", "generic structure", "generic tissue",
    "generic process", "pathology", "abnormality", "generic abnormality",
    "finding", "generic finding", "process", "structure", "tissue",
    "cross sectional view", "cross section", "anatomical association",
    "spatial relationship", "relative position", "proximity",
}
SYNONYM_GROUPS = (
    {"ct", "ct scan", "computed tomography", "computed tomography scan"},
    {"mri", "mr imaging", "magnetic resonance imaging"},
    {"x ray", "xray", "radiograph", "plain radiograph"},
    {"ultrasound", "ultrasonography", "sonography"},
    {"pet", "positron emission tomography"},
)
UNCERTAIN = re.compile(
    r"\b(?:possible|possibly|potential|potentially|may|might|could|likely|"
    r"suggest(?:s|ed|ing|ive)?|indicative of)\b"
)
NEGATED = re.compile(
    r"\b(?:no|not|without|absence of|absent|negative for|free of|"
    r"does not show|did not show)\b"
)
MODALITY_NAMES = {
    "ct": {"computed tomography", "ct", "ct scan"},
    "mri": {"magnetic resonance imaging", "mri", "mr imaging"},
    "xray": {"x ray", "xray", "radiograph"},
    "ultrasound": {"ultrasound", "ultrasonography", "sonography"},
}


def _sentence_for_mention(caption: str, mention: str) -> str:
    needle = audit_normalize(mention)
    for sentence in re.split(r"(?<=[.!?;])\s+|\n+", caption):
        if needle and needle in audit_normalize(sentence):
            return sentence
    return ""


def _nearby_cue(sentence: str, mention: str, cue: re.Pattern[str], before: int, after: int) -> bool:
    sentence_norm = audit_normalize(sentence)
    tokens = sentence_norm.split()
    needle = audit_normalize(mention).split()
    locations = [index for index in range(len(tokens) - len(needle) + 1) if needle and tokens[index:index + len(needle)] == needle]
    for match in cue.finditer(sentence_norm):
        cue_start = len(sentence_norm[:match.start()].split())
        cue_end = cue_start + len(match.group(0).split())
        for start in locations:
            end = start + len(needle)
            if cue_end <= start and start - cue_end <= before:
                return True
            if end <= cue_start and cue_start - end <= after:
                return True
    return False

def audit_row(row: dict[str, Any], concepts: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit without importing or calling Concept v3 repair code."""
    reasons: Counter[str] = Counter()
    caption = str(row.get("source_caption") or "")
    caption_norm = audit_normalize(caption)
    canonicals = [audit_normalize(item.get("canonical")) for item in concepts]

    for value, count in Counter(canonicals).items():
        if value and count > 1:
            reasons["exact_duplicate"] += count - 1
    for group in SYNONYM_GROUPS:
        hits = [value for value in canonicals if value in group]
        if len(set(hits)) > 1:
            reasons["synonym_duplicate"] += len(set(hits)) - 1

    for item, canonical in zip(concepts, canonicals):
        mention = str(item.get("mention") or item.get("canonical") or "")
        mention_norm = audit_normalize(mention)
        if POSITION_ONLY.fullmatch(canonical):
            reasons["pure_positional_concept"] += 1
        if ROI_TERMS.search(canonical) or ROI_TERMS.search(mention_norm):
            reasons["annotation_or_roi_concept"] += 1
        if canonical in GENERIC_EXACT:
            reasons["generic_nonclinical_concept"] += 1
        if not mention_norm or mention_norm not in caption_norm:
            reasons["caption_support_failure"] += 1

        sentence = _sentence_for_mention(caption, mention)
        status = str(item.get("status") or "")
        if sentence and _nearby_cue(sentence, mention, UNCERTAIN, 6, 5) and status == "present":
            reasons["uncertainty_cue_inconsistency"] += 1
        if sentence and _nearby_cue(sentence, mention, NEGATED, 4, 2) and status != "negated":
            reasons["negation_inconsistency"] += 1

        if item.get("type") == "modality":
            declared = audit_normalize(row.get("modality"))
            for key, names in MODALITY_NAMES.items():
                if declared.startswith(key) and canonical not in names and mention_norm not in caption_norm:
                    reasons["modality_anatomy_contradiction"] += 1

    return {
        "status": "PASS" if not reasons else "FAIL",
        "reason_counts": dict(sorted(reasons.items())),
    }

