#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator


OLD_ROOT = Path("/root/MedAgentCL")
DATA_ROOT = OLD_ROOT / "data"
PATH_REMAPS = [
    ("/remote-home/wangbomin/OmniMedVQA/Images", str(DATA_ROOT / "OmniMedVQA" / "Images")),
]


def iter_samples(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    yield line_number, json.loads(line)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError(f"Expected JSON array: {path}")
        for index, sample in enumerate(payload):
            yield index, sample


def resolve_image(path_text: str, json_path: Path) -> tuple[Path, str]:
    candidate = Path(path_text)
    if candidate.is_file():
        return candidate, "direct"
    for source, target in PATH_REMAPS:
        if path_text == source or path_text.startswith(source + "/"):
            remapped = Path(target + path_text[len(source):])
            if remapped.is_file():
                return remapped, "known_root_remap"
    if not candidate.is_absolute():
        relative = (json_path.parent / candidate).resolve()
        if relative.is_file():
            return relative, "json_parent_relative"
    return candidate, "missing"


def candidate_files() -> list[Path]:
    paths = []
    for root in (DATA_ROOT / "MedSkill_CL_4Skill", DATA_ROOT / "MedTrinity_ConceptCaption_CL"):
        if root.is_dir():
            paths.extend(sorted(path for path in root.glob("*.jsonl") if "_intermediate" not in path.parts))
    for path in (
        DATA_ROOT / "OmniMedVQA_CL" / "train" / "all_tasks_train.json",
        DATA_ROOT / "OmniMedVQA_CL" / "test" / "all_tasks_test.json",
    ):
        if path.is_file():
            paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    selected = None
    for path in candidate_files():
        counts = {
            "total": 0,
            "messages_missing_or_invalid": 0,
            "images_empty": 0,
            "image_paths_total": 0,
            "image_paths_direct_existing": 0,
            "image_paths_remapped_existing": 0,
            "image_paths_missing": 0,
            "absolute_image_paths": 0,
            "relative_image_paths": 0,
            "placeholder_mismatch": 0,
            "assistant_target_empty": 0,
        }
        first_keys = None
        for reference, sample in iter_samples(path):
            counts["total"] += 1
            if first_keys is None:
                first_keys = sorted(sample)
            messages = sample.get("messages")
            if not isinstance(messages, list) or not messages:
                counts["messages_missing_or_invalid"] += 1
                messages = []
            user_content = "\n".join(
                str(message.get("content", ""))
                for message in messages
                if isinstance(message, dict) and message.get("role") == "user"
            )
            assistant_contents = [
                str(message.get("content", "")).strip()
                for message in messages
                if isinstance(message, dict) and message.get("role") == "assistant"
            ]
            if not assistant_contents or not any(assistant_contents):
                counts["assistant_target_empty"] += 1

            images = sample.get("images")
            if images is None and sample.get("image"):
                images = [sample["image"]]
            if not isinstance(images, list):
                images = []
            if not images:
                counts["images_empty"] += 1
            if user_content.count("<image>") != len(images):
                counts["placeholder_mismatch"] += 1

            resolved_images = []
            for image in images:
                image_text = str(image)
                counts["image_paths_total"] += 1
                if Path(image_text).is_absolute():
                    counts["absolute_image_paths"] += 1
                else:
                    counts["relative_image_paths"] += 1
                resolved, resolution = resolve_image(image_text, path)
                if resolution == "direct":
                    counts["image_paths_direct_existing"] += 1
                elif resolution == "missing":
                    counts["image_paths_missing"] += 1
                else:
                    counts["image_paths_remapped_existing"] += 1
                resolved_images.append({
                    "original": image_text,
                    "resolved": str(resolved),
                    "resolution": resolution,
                    "exists": resolved.is_file(),
                    "size": resolved.stat().st_size if resolved.is_file() else None,
                })

            if (
                selected is None
                and path.name == "task_02_vqa_test.jsonl"
                and resolved_images
                and all(item["exists"] and item["size"] for item in resolved_images)
                and user_content.count("<image>") == len(images)
                and any(assistant_contents)
            ):
                selected = {
                    "source_json": str(path),
                    "source_reference": reference,
                    "sample_id": sample.get("id"),
                    "dataset": sample.get("dataset"),
                    "messages": messages,
                    "images": images,
                    "resolved_images": resolved_images,
                    "assistant_target": assistant_contents[-1],
                }

        split = "train" if "train" in path.name or "/train/" in str(path) else "test" if "test" in path.name or "/test/" in str(path) else "other"
        reports.append({
            "path": str(path),
            "format": "jsonl" if path.suffix == ".jsonl" else "json_array",
            "split": split,
            "size_bytes": path.stat().st_size,
            "first_sample_fields": first_keys,
            "counts": counts,
        })

    manifest = {
        "status": "PASS" if selected is not None else "BLOCKED",
        "old_project_read_only": str(OLD_ROOT),
        "searched_roots": [
            str(DATA_ROOT / "MedSkill_CL_4Skill"),
            str(DATA_ROOT / "MedTrinity_ConceptCaption_CL"),
            str(DATA_ROOT / "OmniMedVQA_CL"),
            str(DATA_ROOT / "OmniMedVQA"),
        ],
        "candidate_files": [str(path) for path in candidate_files()],
        "selected_real_medical_sample": selected,
        "path_remaps": PATH_REMAPS,
    }
    schema = {
        "status": manifest["status"],
        "reports": reports,
        "notes": [
            "OmniMedVQA_CL uses JSON arrays rather than JSONL.",
            "MedSkill_CL_4Skill provides JSONL derived from OmniMedVQA.",
            "Original OmniMedVQA image paths use /remote-home/wangbomin and require the recorded root remap on this server.",
        ],
    }

    (output_dir / "phase2_data_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
    (output_dir / "jsonl_schema_report.json").write_text(json.dumps(schema, indent=2, ensure_ascii=True) + "\n")
    selected_path = output_dir / "selected_smoke_samples.jsonl"
    selected_path.write_text("" if selected is None else json.dumps(selected, ensure_ascii=True) + "\n")

    print(json.dumps({"status": manifest["status"], "files": len(reports), "selected": selected}, indent=2, ensure_ascii=True))
    return 0 if selected is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
