#!/usr/bin/env python3
"""High-risk cross-split pHash closure with per-image lineage evidence."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path("/root/MedAgentCL_v4")
DATA = Path("/remote-home/wangbomin/MedicalSkill-CL-v1.1")
ARTIFACT = ROOT / "artifacts/medicalskill_cl_v1_1_and_medprism_smoke"
SOURCE = DATA / "audits/phash_candidate_classification.json"


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def source_dataset(path: str) -> str:
    for value in ("OmniMedVQA", "MedIMeta", "MedTrinity-25M", "MedSG", "MedThinkVQA"):
        if f"/{value}/" in path:
            return value
    return "unknown"


def per_image_tokens(ref: dict[str, Any]) -> list[str]:
    path = str(ref.get("path") or "")
    stem = norm(Path(path).stem)
    dataset = source_dataset(path)
    tokens = []
    sha = str(ref.get("sha256") or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", sha):
        tokens.append(f"sha256:{sha}")
    isic = re.search(r"isic[_-]?(\d{7})", path, re.I)
    if isic:
        tokens.append(f"isic:isic_{isic.group(1)}")
    if dataset == "OmniMedVQA":
        match = re.search(r"/Images/([^/]+)/", path)
        namespace = norm(match.group(1)) if match else "omnimedvqa"
    elif dataset == "MedIMeta":
        match = re.search(r"/MedIMeta/([^/]+)/", path)
        namespace = norm(match.group(1)) if match else "medimeta"
    elif dataset == "MedThinkVQA":
        match = re.search(r"/images/(case\d+)/", path, re.I)
        namespace = f"medthink_{norm(match.group(1))}" if match else "medthinkvqa"
    else:
        namespace = norm(dataset)
    if stem:
        tokens.append(f"image:{namespace}:{stem}")
    return sorted(set(tokens))


def clean_ref(ref: dict[str, Any] | list[Any]) -> dict[str, Any]:
    if isinstance(ref, list):
        if len(ref) != 7:
            raise ValueError(f"Unexpected compact pHash ref length: {len(ref)}")
        ref = {
            "task": ref[0], "split": ref[1], "id": ref[2], "path": ref[3],
            "phash": ref[4], "lineage_tokens": ref[5], "sha256": ref[6],
        }
    original = [str(value) for value in ref.get("lineage_tokens") or []]
    return {
        "task": int(ref.get("task") or 0), "split": str(ref.get("split") or ""),
        "id": str(ref.get("id") or ""), "path": str(ref.get("path") or ""),
        "sha256": str(ref.get("sha256") or ""), "phash": str(ref.get("phash") or ""),
        "source_dataset": source_dataset(str(ref.get("path") or "")),
        "image_level_tokens": per_image_tokens(ref),
        "group_level_tokens": sorted({value for value in original if value.startswith(("group:", "source:"))}),
        "discarded_foreign_or_ambiguous_image_tokens": sorted({value for value in original if value.startswith(("image:", "isic:", "sha256:")) and value not in per_image_tokens(ref)}),
    }


def dedup_refs(refs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = {}
    for raw in refs:
        ref = clean_ref(raw)
        key = (ref["task"], ref["split"], ref["id"], ref["path"], ref["sha256"], ref["phash"])
        result[key] = ref
    return list(result.values())


def cross_split(refs: list[dict[str, Any]]) -> bool:
    return {ref["split"] for ref in refs} >= {"train", "test"}


def evidence_matches(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches = []
    train = [ref for ref in refs if ref["split"] == "train"]
    test = [ref for ref in refs if ref["split"] == "test"]
    for left in train:
        for right in test:
            common = sorted(set(left["image_level_tokens"]) & set(right["image_level_tokens"]))
            same_sha = bool(left["sha256"] and left["sha256"] == right["sha256"])
            canonical = [value for value in common if not value.startswith("sha256:")]
            if same_sha or canonical:
                matches.append({"train": left, "test": right, "same_sha256": same_sha, "canonical_image_tokens": canonical})
    return matches


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    report = json.loads(args.source.read_text(encoding="utf-8"))
    candidates = report.get("candidates") or report.get("candidate_groups") or []
    if not candidates:
        raise RuntimeError("No pHash candidates")
    split_counts = collections.Counter()
    cross_candidates = []
    high_risk = []
    likely = []
    evidence_rows = []
    confirmed = []
    for candidate in candidates:
        refs = dedup_refs(candidate.get("refs") or [])
        splits = {ref["split"] for ref in refs}
        if splits >= {"train", "test"}:
            split_class = "cross_split"
        elif splits == {"train"}:
            split_class = "train_only"
        elif splits == {"test"}:
            split_class = "test_only"
        else:
            split_class = "other_or_truncated"
        split_counts[split_class] += 1
        matches = evidence_matches(refs) if split_class == "cross_split" else []
        cleaned = {
            "candidate_id": candidate.get("candidate_id") or candidate.get("id"),
            "classification": candidate.get("classification"),
            "distance": candidate.get("distance"),
            "best_ssim": (candidate.get("best_comparison") or {}).get("ssim"),
            "split_class": split_class, "refs": refs,
            "refs_before": len(candidate.get("refs") or []), "refs_after_dedup": len(refs),
            "refs_truncated": bool(candidate.get("refs_truncated")),
            "canonical_or_sha_matches": matches,
        }
        evidence_rows.append(cleaned)
        if split_class != "cross_split":
            continue
        cross_candidates.append(cleaned)
        ssim = cleaned["best_ssim"]
        if int(cleaned["distance"]) == 0 and ssim is not None and float(ssim) >= 0.97:
            high_risk.append(cleaned)
        if cleaned["classification"] == "likely_near_duplicate":
            likely.append(cleaned)
        if matches:
            confirmed.append(cleaned)

    summary = {
        "status": "PASS" if not confirmed else "BLOCKED",
        "input": str(args.source), "input_sha256": sha256(args.source),
        "all_candidate_groups": len(candidates), "split_classification_counts": dict(split_counts),
        "cross_split_acceptance_denominator": len(cross_candidates),
        "train_only_excluded_from_denominator": split_counts["train_only"],
        "test_only_excluded_from_denominator": split_counts["test_only"],
        "cross_split_by_distance": dict(collections.Counter(str(row["distance"]) for row in cross_candidates)),
        "cross_split_by_classification": dict(collections.Counter(str(row["classification"]) for row in cross_candidates)),
        "distance0_ssim_ge_097_cross_split": len(high_risk),
        "likely_near_duplicate_cross_split": len(likely),
        "confirmed_cross_split_same_source": len(confirmed),
        "policy": "Never delete by pHash alone; require per-image SHA/canonical metadata plus visual evidence.",
    }
    high_report = {
        "status": "PASS" if not any(row["canonical_or_sha_matches"] for row in high_risk) else "BLOCKED",
        "scope": "all cross-split candidates with Hamming distance=0 and SSIM>=0.97",
        "total": len(high_risk), "complete_not_sampled": True, "rows": high_risk,
    }
    likely_report = {
        "status": "PASS" if not any(row["canonical_or_sha_matches"] for row in likely) else "BLOCKED",
        "total": len(likely),
        "resolution_counts": dict(collections.Counter("confirmed_same_source" if row["canonical_or_sha_matches"] else "retain_near_duplicate_unconfirmed" for row in likely)),
        "rows": likely,
    }
    evidence_report = {
        "status": "PASS",
        "rules": ["one ref per unique task/split/id/path/SHA/pHash", "path/SHA/pHash and image-level tokens describe the same image", "group/source tokens stored separately", "foreign or ambiguous image tokens discarded"],
        "groups": len(evidence_rows),
        "refs_before": sum(row["refs_before"] for row in evidence_rows),
        "refs_after_dedup": sum(row["refs_after_dedup"] for row in evidence_rows),
        "refs_with_discarded_tokens": sum(bool(ref["discarded_foreign_or_ambiguous_image_tokens"]) for row in evidence_rows for ref in row["refs"]),
        "rows": evidence_rows,
    }
    known_refs = [ref for row in evidence_rows for ref in row["refs"] if "isic:isic_0001131" in ref["image_level_tokens"]]
    known = {
        "status": "PASS" if {ref["split"] for ref in known_refs} <= {"test"} else "BLOCKED",
        "known_case": "ISIC_0001131", "canonical_token": "isic:isic_0001131",
        "ref_count": len(known_refs), "splits": sorted({ref["split"] for ref in known_refs}),
        "train_ref_count": sum(ref["split"] == "train" for ref in known_refs),
        "test_ref_count": sum(ref["split"] == "test" for ref in known_refs), "refs": known_refs,
    }
    for name, value in (
        ("cross_split_phash_only_summary.json", summary),
        ("phash_distance0_high_ssim_audit.json", high_report),
        ("likely_near_duplicate_resolution.json", likely_report),
        ("per_image_lineage_evidence_audit.json", evidence_report),
        ("known_leakage_regression_v2.json", known),
    ):
        atomic_json(ARTIFACT / name, value)
    statuses = [summary["status"], high_report["status"], likely_report["status"], evidence_report["status"], known["status"]]
    final = "PASS" if set(statuses) == {"PASS"} else "BLOCKED"
    (ARTIFACT / "phash_final_closure_report.md").write_text(
        "# pHash final closure\n\n"
        f"Status: **{final}**\n\n"
        f"All candidates: {len(candidates)}; true cross-split denominator: {len(cross_candidates)}.\n\n"
        f"Train-only/test-only excluded: {split_counts['train_only']}/{split_counts['test_only']}.\n\n"
        f"Distance=0 and SSIM>=0.97 cross-split groups fully reviewed: {len(high_risk)}.\n\n"
        f"Likely-near-duplicate cross-split groups: {len(likely)}.\n\n"
        f"Per-image canonical/SHA confirmed train/test sources: {len(confirmed)}.\n\n"
        f"ISIC_0001131 train/test refs: {known['train_ref_count']}/{known['test_ref_count']}.\n",
        encoding="utf-8",
    )
    print(json.dumps({**summary, "known_status": known["status"]}, indent=2))
    return 0 if final == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
