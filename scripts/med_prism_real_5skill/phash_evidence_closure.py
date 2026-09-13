#!/usr/bin/env python3
"""Summarize and stratify MedicalSkill-CL-v1 pHash evidence without distance-only deletion."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path("/root/MedAgentCL_v4")
DATA = Path("/remote-home/wangbomin/MedicalSkill-CL-v1")
DEFAULT_INPUT = DATA / "audits/phash_candidate_classification.json"
DEFAULT_OUTPUT = ROOT / "artifacts/med_prism_real_5skill_smoke"
KNOWN_TOKEN = "isic:isic_0001131"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_dataset(path: str) -> str:
    lower = path.casefold()
    rules = (
        ("/medsg/", "MedSG"),
        ("/medtrinity-25m/", "MedTrinity-25M"),
        ("/omnimedvqa/", "OmniMedVQA"),
        ("/medimeta/", "MedIMeta"),
        ("/medthinkvqa/", "MedThinkVQA"),
        ("/isic", "ISIC"),
        ("/brats", "BraTS"),
        ("/tcga", "TCGA"),
        ("/ihc4bc", "IHC4BC"),
        ("/vindr", "VinDr"),
    )
    for needle, name in rules:
        if needle in lower:
            return name
    parts = Path(path).parts
    return parts[4] if len(parts) > 4 else "unknown"


def ref_value(ref: list[Any]) -> dict[str, Any]:
    task, split, row_id, path, phash, tokens, sha256 = ref
    return {
        "task": int(task),
        "split": str(split),
        "id": str(row_id),
        "path": str(path),
        "phash": str(phash),
        "lineage_tokens": list(tokens),
        "sha256": str(sha256),
        "source_dataset": source_dataset(str(path)),
    }


def classification_bucket(value: str) -> str:
    lower = value.casefold()
    if "confirmed" in lower:
        return "confirmed"
    if "likely" in lower:
        return "likely"
    if "false_positive" in lower:
        return "false_positive"
    return "unresolved"


def pair_keys(refs: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    train = [ref for ref in refs if ref["split"] == "train"]
    test = [ref for ref in refs if ref["split"] == "test"]
    task_pairs = sorted({f"train_task_{left['task']}__test_task_{right['task']}" for left in train for right in test})
    source_pairs = sorted({"__".join(sorted((left["source_dataset"], right["source_dataset"]))) for left in refs for right in refs if left is not right})
    return task_pairs, source_pairs


def canonical_train_test_matches(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train = [ref for ref in refs if ref["split"] == "train"]
    test = [ref for ref in refs if ref["split"] == "test"]
    matches = []
    for left in train:
        left_tokens = set(left["lineage_tokens"])
        for right in test:
            common = sorted(left_tokens & set(right["lineage_tokens"]))
            if common:
                matches.append({
                    "train_task": left["task"], "train_id": left["id"],
                    "test_task": right["task"], "test_id": right["id"],
                    "tokens": common,
                })
    return matches


def stratum(candidate: dict[str, Any]) -> tuple[Any, ...]:
    refs = [ref_value(ref) for ref in candidate["refs"]]
    task_pairs, source_pairs = pair_keys(refs)
    task_key = "+".join(task_pairs) if task_pairs else "+".join(sorted({f"{ref['split']}_task_{ref['task']}" for ref in refs}))
    source_key = "+".join(source_pairs) if source_pairs else "+".join(sorted({ref["source_dataset"] for ref in refs}))
    return candidate["distance"], task_key, source_key


def stratified_sample(candidates: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for candidate in candidates:
        buckets[stratum(candidate)].append(candidate)
    rng = random.Random(seed)
    ordered = []
    for key in sorted(buckets, key=lambda item: tuple(map(str, item))):
        values = buckets[key]
        rng.shuffle(values)
        ordered.append((key, values))
    selected = []
    index = 0
    while len(selected) < min(count, len(candidates)):
        progress = False
        for key, values in ordered:
            if index < len(values):
                selected.append(values[index])
                progress = True
                if len(selected) == count:
                    break
        if not progress:
            break
        index += 1
    return selected


def audit_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    refs = [ref_value(ref) for ref in candidate["refs"]]
    task_pairs, source_pairs = pair_keys(refs)
    canonical = canonical_train_test_matches(refs)
    best = candidate.get("best_comparison") or {}
    return {
        "candidate_id": candidate["candidate_id"],
        "classification": candidate["classification"],
        "distance": candidate["distance"],
        "task_pairs": task_pairs,
        "source_pairs": source_pairs,
        "best_ssim": best.get("ssim"),
        "best_std_left": best.get("std_left"),
        "best_std_right": best.get("std_right"),
        "refs": refs,
        "refs_truncated": candidate.get("refs_truncated", False),
        "canonical_train_test_matches": canonical,
        "decision": "confirmed_canonical_train_test" if canonical else "retain_unresolved_no_canonical_confirmation",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    candidates = report["candidates"]
    classification = collections.Counter()
    buckets = collections.Counter()
    distances = collections.Counter()
    task_pairs = collections.Counter()
    source_pairs = collections.Counter()
    all_canonical_matches = []
    known_candidates = []
    unresolved = []
    for candidate in candidates:
        classification[candidate["classification"]] += 1
        buckets[classification_bucket(candidate["classification"])] += 1
        distances[int(candidate["distance"])] += 1
        refs = [ref_value(ref) for ref in candidate["refs"]]
        candidate_task_pairs, candidate_source_pairs = pair_keys(refs)
        task_pairs.update(candidate_task_pairs)
        source_pairs.update(candidate_source_pairs)
        canonical = canonical_train_test_matches(refs)
        if canonical:
            all_canonical_matches.append({"candidate_id": candidate["candidate_id"], "matches": canonical})
        if any(KNOWN_TOKEN in ref["lineage_tokens"] or "isic_0001131" in ref["path"].casefold() for ref in refs):
            known_candidates.append(audit_candidate(candidate))
        if classification_bucket(candidate["classification"]) == "unresolved":
            unresolved.append(candidate)
    sampled = [audit_candidate(candidate) for candidate in stratified_sample(unresolved, args.sample_size, args.seed)]
    sampled_matches = [item for item in sampled if item["canonical_train_test_matches"]]
    summary = {
        "status": "PASS" if not all_canonical_matches else "BLOCKED",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "candidate_group_count": len(candidates),
        "classification_counts": dict(sorted(classification.items())),
        "normalized_classification_counts": dict(sorted(buckets.items())),
        "hamming_distance_counts": {str(key): value for key, value in sorted(distances.items())},
        "train_task_test_task_counts": dict(sorted(task_pairs.items())),
        "source_dataset_pair_counts": dict(sorted(source_pairs.items())),
        "canonical_train_test_match_count": len(all_canonical_matches),
        "canonical_train_test_matches": all_canonical_matches[:100],
        "policy": "No deletion by pHash distance alone; only canonical lineage confirmation may remove a complete train group.",
    }
    known_refs = [ref for candidate in known_candidates for ref in candidate["refs"] if KNOWN_TOKEN in ref["lineage_tokens"] or "isic_0001131" in ref["path"].casefold()]
    known_splits = sorted({ref["split"] for ref in known_refs})
    known = {
        "status": "PASS" if known_refs and not ({"train", "test"} <= set(known_splits)) else "BLOCKED",
        "known_case": "ISIC_0001131",
        "canonical_token": KNOWN_TOKEN,
        "candidate_count": len(known_candidates),
        "known_ref_count": len(known_refs),
        "known_ref_splits": known_splits,
        "known_refs": known_refs,
        "candidate_evidence": known_candidates,
        "train_test_leakage_present": {"train", "test"} <= set(known_splits),
        "regression_rule": "The known canonical case may remain as evidence but cannot retain both train and test membership.",
    }
    audit = {
        "status": "PASS" if len(sampled) >= 300 and not sampled_matches else "BLOCKED",
        "seed": args.seed,
        "requested": args.sample_size,
        "sampled": len(sampled),
        "population": len(unresolved),
        "stratification": ["task/split combination", "source dataset combination", "Hamming distance"],
        "canonical_confirmed_in_sample": len(sampled_matches),
        "samples": sampled,
    }
    output = args.output_root
    atomic_json(output / "phash_classification_summary.json", summary)
    atomic_json(output / "phash_unresolved_stratified_audit.json", audit)
    atomic_json(output / "known_leakage_regression.json", known)
    lines = [
        "# pHash closure",
        "",
        f"Status: **{'PASS' if summary['status'] == audit['status'] == known['status'] == 'PASS' else 'BLOCKED'}**",
        "",
        f"Candidate groups: {len(candidates)}",
        f"Confirmed / likely / false-positive / unresolved: {buckets['confirmed']} / {buckets['likely']} / {buckets['false_positive']} / {buckets['unresolved']}",
        f"Stratified unresolved audit: {len(sampled)} of {len(unresolved)}",
        f"Canonical train/test matches in full report: {len(all_canonical_matches)}",
        f"Known ISIC_0001131 splits: {known_splits}; leakage present: {known['train_test_leakage_present']}",
        "",
        "No candidate was deleted from pHash distance alone.",
    ]
    (output / "phash_closure_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "summary": summary["status"], "audit": audit["status"], "known": known["status"],
        "sampled": len(sampled), "canonical_matches": len(all_canonical_matches),
    }, indent=2))
    return 0 if summary["status"] == audit["status"] == known["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
