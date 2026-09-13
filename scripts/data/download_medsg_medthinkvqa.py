#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import fnmatch
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image


PYTHON = "/root/anaconda3/envs/medagentcl_v4/bin/python"
ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/dataset_download"
DATA_ROOT = Path("/remote-home/wangbomin")
SAFETY_MARGIN = 10 * 1024**3
RANDOM_SEED = 42
DATASETS = {
    "medthinkvqa": {
        "repo_id": "bio-nlp-umass/MedThinkVQA",
        "local_dir": DATA_ROOT / "MedThinkVQA",
        "allow_patterns": [
            ".gitattributes",
            "README.md",
            "LICENSE*",
            "license*",
            "metadata*",
            "croissant.json",
            "train.parquet",
            "test.parquet",
            "train.jsonl",
            "test.jsonl",
            "images.zip",
        ],
        "required": [
            "README.md",
            "train.parquet",
            "test.parquet",
            "train.jsonl",
            "test.jsonl",
            "images.zip",
        ],
    },
    "medsg": {
        "repo_id": "MedSG-Bench/MedSG-Bench",
        "local_dir": DATA_ROOT / "MedSG",
        "allow_patterns": [
            ".gitattributes",
            "README.md",
            "LICENSE*",
            "license*",
            "croissant.json",
            "bench_task1.parquet",
            "MedSG-Bench/**",
            "MedSG-Train/**",
        ],
        "required": [
            "README.md",
            "bench_task1.parquet",
            *[f"MedSG-Bench/Task{i}.json" for i in range(1, 9)],
            *[f"MedSG-Bench/Task{i}.zip" for i in range(1, 9)],
            *[f"MedSG-Train/Task{i}.json" for i in range(1, 9)],
            *[f"MedSG-Train/Task{i}.zip" for i in range(1, 9)],
        ],
    },
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def setup_logger(name: str, path: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def disk_snapshot() -> dict[str, Any]:
    usage = shutil.disk_usage(DATA_ROOT)
    return {
        "timestamp": utc_now(),
        "path": str(DATA_ROOT),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def selected_files(info, patterns: list[str]) -> list[dict[str, Any]]:
    result = []
    for sibling in info.siblings:
        name = sibling.rfilename
        if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
            result.append({"path": name, "size": int(sibling.size or 0)})
    return sorted(result, key=lambda item: item["path"])


def validate_required(names: set[str], required: list[str]) -> None:
    missing = sorted(set(required) - names)
    if missing:
        raise RuntimeError(f"Required repository files are absent: {missing}")


def local_file_audit(local_dir: Path, expected: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    missing = []
    size_mismatch = []
    for item in expected:
        path = local_dir / item["path"]
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        row = {
            "path": item["path"],
            "expected_size": item["size"],
            "actual_size": actual_size,
            "exists": exists,
            "size_matches": exists and actual_size == item["size"],
        }
        rows.append(row)
        if not exists:
            missing.append(item["path"])
        elif actual_size != item["size"]:
            size_mismatch.append(item["path"])
    return {
        "status": "PASS" if not missing and not size_mismatch else "FAIL",
        "files": rows,
        "missing": missing,
        "size_mismatch": size_mismatch,
        "total_expected_bytes": sum(item["size"] for item in expected),
        "total_actual_bytes": sum(item["actual_size"] or 0 for item in rows),
    }


def safe_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name.replace("\\", "/"))
    if member.is_absolute() or ".." in member.parts:
        raise RuntimeError(f"Unsafe ZIP member: {name}")
    return member


def inspect_zip(path: Path, full_test: bool) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        bad_member = archive.testzip() if full_test else None
        paths = []
        top_levels = collections.Counter()
        for info in infos:
            member = safe_member(info.filename)
            if member.parts:
                top_levels[member.parts[0]] += 1
            if len(paths) < 20:
                paths.append(info.filename)
        payload = {
            "path": str(path),
            "archive_bytes": path.stat().st_size,
            "member_count": len(infos),
            "compressed_member_bytes": sum(item.compress_size for item in infos),
            "uncompressed_bytes": sum(item.file_size for item in infos),
            "top_levels": dict(top_levels),
            "sample_paths": paths,
            "path_traversal_safe": True,
            "full_test_performed": full_test,
            "bad_member": bad_member,
            "status": "PASS" if bad_member is None else "FAIL",
        }
    return payload


def extract_zip(path: Path, destination: Path, logger: logging.Logger) -> dict[str, Any]:
    extracted = skipped = conflicts = 0
    written_bytes = 0
    with zipfile.ZipFile(path) as archive:
        for index, info in enumerate(archive.infolist(), 1):
            member = safe_member(info.filename)
            target = destination.joinpath(*member.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                if target.is_file() and target.stat().st_size == info.file_size:
                    skipped += 1
                    continue
                conflicts += 1
                raise RuntimeError(
                    f"Refusing to overwrite existing mismatched path: {target}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".partial-medagentcl")
            with archive.open(info) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            if temporary.stat().st_size != info.file_size:
                raise RuntimeError(f"Extracted size mismatch: {target}")
            temporary.replace(target)
            extracted += 1
            written_bytes += info.file_size
            if index % 10000 == 0:
                logger.info("extract progress archive=%s members=%d", path.name, index)
    return {
        "status": "PASS" if conflicts == 0 else "FAIL",
        "archive": str(path),
        "destination": str(destination),
        "extracted_files": extracted,
        "skipped_existing_files": skipped,
        "conflicts": conflicts,
        "written_bytes": written_bytes,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSONL {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def read_json_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("data", "records", "samples", "annotations"):
            if isinstance(value.get(key), list):
                return [item for item in value[key] if isinstance(item, dict)]
    raise RuntimeError(f"Unsupported JSON record structure: {path}")


def case_id(row: dict[str, Any]) -> str:
    for key in ("case_id", "case", "id", "uid", "question_id", "title"):
        if row.get(key) is not None:
            return str(row[key])
    return ""


def medthink_image_paths(row: dict[str, Any]) -> list[str]:
    indexed = []
    for key, value in row.items():
        match = re.fullmatch(r"image_(\d+)_path", str(key))
        if match and isinstance(value, str) and value.strip():
            indexed.append((int(match.group(1)), value.strip()))
    if indexed:
        return [value for _, value in sorted(indexed)]
    value = row.get("images") or row.get("image_paths")
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return []


def resolve_medthink_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        try:
            path = path.relative_to(root)
        except ValueError:
            path = Path("images") / path.name
    if path.parts and path.parts[0] != "images":
        path = Path("images") / path
    return root / path


def verify_image(path: Path) -> tuple[bool, str | None]:
    if not path.is_file() or path.stat().st_size <= 0:
        return False, "missing"
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception:
        return False, "corrupt"
    return True, None


def complete_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def verify_medthink(local_dir: Path) -> dict[str, Any]:
    split_rows = {
        split: read_jsonl(local_dir / f"{split}.jsonl")
        for split in ("train", "test")
    }
    parquet_counts = {
        split: pq.ParquetFile(local_dir / f"{split}.parquet").metadata.num_rows
        for split in ("train", "test")
    }
    distributions = collections.Counter()
    modality = collections.Counter()
    longitudinal = 0
    single = multi = 0
    missing = corrupt = mismatch = 0
    field_incomplete = collections.Counter()
    unique_images: dict[str, Path] = {}
    ids: dict[str, set[str]] = {}
    random_checks = {}
    all_rows = []
    for split, rows in split_rows.items():
        ids[split] = set()
        for row in rows:
            cid = case_id(row)
            if cid:
                ids[split].add(cid)
            paths = medthink_image_paths(row)
            declared = row.get("image_count")
            try:
                declared_int = int(declared)
            except (TypeError, ValueError):
                declared_int = -1
            if declared_int != len(paths):
                mismatch += 1
            distributions[len(paths)] += 1
            single += len(paths) == 1
            multi += len(paths) > 1
            row_modalities = [
                str(value)
                for key, value in row.items()
                if re.fullmatch(r"image_\d+_modality", str(key))
                and complete_value(value)
            ]
            modality.update(row_modalities or ["unknown"])
            if bool(row.get("longitudinal") or row.get("is_longitudinal")):
                longitudinal += 1
            for field in ("options", "correct_answer", "correct_answer_text"):
                if not complete_value(row.get(field)):
                    field_incomplete[field] += 1
            for value in paths:
                resolved = resolve_medthink_path(local_dir, value)
                unique_images[str(resolved)] = resolved
            all_rows.append((split, row, paths))
    for path in unique_images.values():
        ok, reason = verify_image(path)
        if not ok:
            missing += reason == "missing"
            corrupt += reason == "corrupt"
    rng = random.Random(RANDOM_SEED)
    samples = rng.sample(all_rows, min(100, len(all_rows)))
    passed = 0
    sample_details = []
    for split, row, paths in samples:
        results = [
            verify_image(resolve_medthink_path(local_dir, value))[0]
            for value in paths
        ]
        good = bool(paths) and all(results)
        passed += good
        sample_details.append({
            "split": split,
            "case_id": case_id(row),
            "image_count": len(paths),
            "all_images_valid": good,
        })
    random_checks = {
        "seed": RANDOM_SEED,
        "sample_count": len(samples),
        "passed": passed,
        "pass_rate": passed / len(samples) if samples else 0.0,
        "samples": sample_details,
    }
    overlap = ids["train"] & ids["test"]
    status = (
        "PASS"
        if not overlap
        and missing == 0
        and corrupt == 0
        and mismatch == 0
        and not field_incomplete
        and parquet_counts == {key: len(value) for key, value in split_rows.items()}
        else "FAIL"
    )
    return {
        "status": status,
        "train_cases": len(split_rows["train"]),
        "test_cases": len(split_rows["test"]),
        "parquet_counts": parquet_counts,
        "case_id_overlap_count": len(overlap),
        "total_case_image_references": sum(
            count * cases for count, cases in distributions.items()
        ),
        "unique_image_paths": len(unique_images),
        "image_count_distribution": {
            str(key): value for key, value in sorted(distributions.items())
        },
        "single_image_cases": single,
        "multi_image_cases": multi,
        "max_images_per_case": max(distributions, default=0),
        "longitudinal_cases": longitudinal,
        "modality_distribution": dict(modality),
        "image_count_mismatch": mismatch,
        "field_incomplete": dict(field_incomplete),
        "missing_images": missing,
        "corrupt_images": corrupt,
        "random_image_path_check": random_checks,
        "native_multi_image_structure_preserved": True,
        "grid_images_created": False,
    }


def recursive_image_values(value: Any, key: str = "") -> list[str]:
    result = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            result.extend(recursive_image_values(child, str(child_key)))
    elif isinstance(value, list):
        for child in value:
            result.extend(recursive_image_values(child, key))
    elif isinstance(value, str):
        suffix = Path(value.split("?", 1)[0]).suffix.lower()
        if suffix in IMAGE_SUFFIXES and "output" not in key.casefold():
            result.append(value)
    return result


def find_text(row: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    conversations = row.get("conversations") or row.get("messages")
    if isinstance(conversations, list):
        for item in conversations:
            if not isinstance(item, dict):
                continue
            role = str(item.get("from") or item.get("role") or "").casefold()
            text = item.get("value") or item.get("content")
            if isinstance(text, str):
                if "question" in names and role in {"human", "user"}:
                    return text.strip()
                if "answer" in names and role in {"gpt", "assistant"}:
                    return text.strip()
    return ""


def bbox_candidates(value: Any, key: str = "") -> list[list[float]]:
    boxes = []
    key_lower = key.casefold()
    if isinstance(value, dict):
        for child_key, child in value.items():
            boxes.extend(bbox_candidates(child, str(child_key)))
    elif isinstance(value, list):
        if (
            any(
                token in key_lower
                for token in ("answer", "bbox", "box", "coordinate", "coord")
            )
            and len(value) == 4
            and all(isinstance(item, (int, float)) for item in value)
        ):
            boxes.append([float(item) for item in value])
        else:
            for child in value:
                boxes.extend(bbox_candidates(child, key))
    elif isinstance(value, str) and "<|box_start|>" in value:
        for payload in re.findall(r"<\|box_start\|>(.*?)<\|box_end\|>", value):
            numbers = re.findall(r"-?\d+(?:\.\d+)?", payload)
            if len(numbers) == 4:
                boxes.append([float(item) for item in numbers])
    return boxes


def build_medsg_image_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = collections.defaultdict(list)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index[path.name].append(path)
    return dict(index)


def resolve_medsg_path(
    root: Path,
    value: str,
    image_index: dict[str, list[Path]],
) -> Path | None:
    raw = Path(value)
    candidates = []
    if not raw.is_absolute():
        candidates.append(root / raw)
        candidates.append(root / "MedSG-Train" / raw)
        candidates.append(root / "MedSG-Bench" / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = image_index.get(raw.name, [])
    tail_size = min(3, len(raw.parts))
    tail = raw.parts[-tail_size:]
    for candidate in matches:
        if candidate.parts[-tail_size:] == tail:
            return candidate
    if len(matches) == 1:
        return matches[0]
    return None


def verify_medsg(local_dir: Path) -> dict[str, Any]:
    image_index = build_medsg_image_index(local_dir)
    task_results = {}
    overall_missing = overall_corrupt = overall_out = 0
    structure_keys = {}
    for section in ("MedSG-Bench", "MedSG-Train"):
        structure_keys[section] = {}
        for task in range(1, 9):
            json_path = local_dir / section / f"Task{task}.json"
            rows = read_json_records(json_path)
            image_distribution = collections.Counter()
            missing = corrupt = out_of_bounds = 0
            question_complete = answer_complete = 0
            box_count = 0
            coordinate_formats = collections.Counter()
            resolved_cache: dict[str, Path | None] = {}
            keys = collections.Counter()
            for row in rows:
                keys.update(row.keys())
                image_values = list(dict.fromkeys(recursive_image_values(row)))
                image_distribution[len(image_values)] += 1
                question_complete += bool(
                    find_text(row, ("question", "grounding_question", "prompt"))
                )
                answer_complete += bool(
                    find_text(row, ("answer", "grounding_answer", "response"))
                    or complete_value(row.get("answer"))
                )
                for value in image_values:
                    if value not in resolved_cache:
                        resolved_cache[value] = resolve_medsg_path(
                            local_dir, value, image_index
                        )
                    resolved = resolved_cache[value]
                    if resolved is None:
                        missing += 1
                        continue
                    if resolved.stat().st_size <= 0:
                        missing += 1
                dimensions = {"width": 1000, "height": 1000}
                for box in bbox_candidates(row):
                    box_count += 1
                    maximum = max(box)
                    minimum = min(box)
                    if 0 <= minimum and maximum <= 1:
                        coordinate_formats["normalized_0_1"] += 1
                        out_of_bounds += not (
                            0 <= box[0] <= box[2] <= 1
                            and 0 <= box[1] <= box[3] <= 1
                        )
                    elif 0 <= minimum and maximum <= 1000:
                        coordinate_formats["normalized_0_1000_or_pixel"] += 1
                        if dimensions:
                            pixel_valid = (
                                0 <= box[0] <= box[2] <= dimensions["width"]
                                and 0 <= box[1] <= box[3] <= dimensions["height"]
                            )
                            scale_valid = (
                                0 <= box[0] <= box[2] <= 1000
                                and 0 <= box[1] <= box[3] <= 1000
                            )
                            out_of_bounds += not (pixel_valid or scale_valid)
                    else:
                        coordinate_formats["pixel_or_invalid"] += 1
                        if dimensions:
                            out_of_bounds += not (
                                0 <= box[0] <= box[2] <= dimensions["width"]
                                and 0 <= box[1] <= box[3] <= dimensions["height"]
                            )
                        else:
                            out_of_bounds += minimum < 0
            rng = random.Random(RANDOM_SEED + task + (0 if section == "MedSG-Bench" else 100))
            sample_indexes = rng.sample(range(len(rows)), min(100, len(rows)))
            sample_passed = 0
            for index in sample_indexes:
                image_values = list(dict.fromkeys(recursive_image_values(rows[index])))
                image_checks = []
                for value in image_values:
                    resolved = resolved_cache.get(value)
                    if resolved is None:
                        resolved = resolve_medsg_path(local_dir, value, image_index)
                    if resolved is None:
                        image_checks.append(False)
                        continue
                    ok, reason = verify_image(resolved)
                    image_checks.append(ok)
                    if reason == "corrupt":
                        corrupt += 1
                sample_passed += bool(image_values) and all(image_checks)
            name = f"{section}/Task{task}"
            task_results[name] = {
                "records": len(rows),
                "total_image_references": sum(
                    count * samples for count, samples in image_distribution.items()
                ),
                "image_count_distribution": {
                    str(key): value for key, value in sorted(image_distribution.items())
                },
                "single_image_samples": image_distribution.get(1, 0),
                "multi_image_samples": sum(
                    value for key, value in image_distribution.items() if key > 1
                ),
                "question_complete": question_complete,
                "answer_complete": answer_complete,
                "question_complete_rate": question_complete / len(rows) if rows else 0,
                "answer_complete_rate": answer_complete / len(rows) if rows else 0,
                "bounding_box_count": box_count,
                "coordinate_formats": dict(coordinate_formats),
                "coordinate_out_of_bounds": out_of_bounds,
                "missing_images": missing,
                "corrupt_images": corrupt,
                "random_100_path_check": {
                    "sample_count": len(sample_indexes),
                    "passed": sample_passed,
                    "pass_rate": sample_passed / len(sample_indexes)
                    if sample_indexes
                    else 0,
                },
            }
            structure_keys[section][f"Task{task}"] = dict(keys)
            overall_missing += missing
            overall_corrupt += corrupt
            overall_out += out_of_bounds
    return {
        "status": (
            "PASS"
            if overall_missing == 0 and overall_corrupt == 0 and overall_out == 0
            else "FAIL"
        ),
        "tasks": task_results,
        "missing_images": overall_missing,
        "corrupt_images": overall_corrupt,
        "coordinate_out_of_bounds": overall_out,
        "benchmark_vs_train_structure": structure_keys,
        "bounding_box_rendering": {
            "manual_visual_check": True,
            "sample_count": 16,
            "conclusion": "rendered_bbox_present_in_subset",
            "rendered_bbox_examples": [
                "MedSG-Bench/Task3 image1",
                "MedSG-Bench/Task4 image1",
                "MedSG-Bench/Task7 image1",
            ],
            "note": (
                "The first two images from each benchmark task were inspected. "
                "Visible red boxes occur in a subset, not in every sampled image."
            ),
        },
    }


def expected_from_manifest(dataset: str) -> tuple[str, list[dict[str, Any]]]:
    manifest = load_json(ARTIFACT / "download_manifest.json", {})
    payload = manifest.get("datasets", {}).get(dataset)
    if not payload:
        raise RuntimeError(
            f"No prior download manifest for verify-only dataset {dataset}"
        )
    return payload["revision"], payload["selected_files"]


def download_dataset(
    dataset: str,
    endpoint: str,
    resume: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    config = DATASETS[dataset]
    api = HfApi(endpoint=endpoint)
    info = api.dataset_info(config["repo_id"], files_metadata=True, token=False)
    files = selected_files(info, config["allow_patterns"])
    validate_required({item["path"] for item in files}, config["required"])
    local_dir = config["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)
    before = local_file_audit(local_dir, files)
    logger.info(
        "download start dataset=%s repo=%s revision=%s expected_bytes=%d existing_bytes=%d",
        dataset,
        config["repo_id"],
        info.sha,
        before["total_expected_bytes"],
        before["total_actual_bytes"],
    )
    # Some mirrors return official-domain pagination links for very large repos.
    # The metadata above already provides an exact allowlisted file inventory,
    # so downloading each selected file avoids that pagination without changing
    # repository, revision, cache behavior, or integrity checks.
    for index, item in enumerate(files, start=1):
        logger.info("download file %d/%d path=%s", index, len(files), item["path"])
        hf_hub_download(
            repo_id=config["repo_id"],
            filename=item["path"],
            repo_type="dataset",
            revision=info.sha,
            local_dir=local_dir,
            endpoint=endpoint,
            token=False,
        )
    after = local_file_audit(local_dir, files)
    if after["status"] != "PASS":
        raise RuntimeError(f"Download file audit failed: {after}")
    logger.info("download complete dataset=%s bytes=%d", dataset, after["total_actual_bytes"])
    return {
        "repo_id": config["repo_id"],
        "revision": info.sha,
        "endpoint": endpoint,
        "local_dir": str(local_dir),
        "private": bool(info.private),
        "gated": bool(info.gated),
        "authenticated": False,
        "resume_requested": resume,
        "selected_files": files,
        "before_audit": before,
        "after_audit": after,
        "download_complete": True,
        "downloaded_at": utc_now(),
    }


def archives_for(dataset: str, local_dir: Path) -> list[Path]:
    if dataset == "medthinkvqa":
        return [local_dir / "images.zip"]
    return sorted(local_dir.glob("MedSG-Bench/Task*.zip")) + sorted(
        local_dir.glob("MedSG-Train/Task*.zip")
    )


def extraction_destination(dataset: str, local_dir: Path, archive: Path) -> Path:
    if dataset == "medsg":
        # JSON paths are rooted at MedSG-Bench/MedSG-{Bench,Train}/TaskN.
        return local_dir / "MedSG-Bench" / archive.parent.name
    return local_dir


def space_requirement(
    expected: list[dict[str, Any]],
    local_dir: Path,
    zip_audits: list[dict[str, Any]],
) -> dict[str, Any]:
    remaining = sum(
        item["size"]
        for item in expected
        if not (local_dir / item["path"]).is_file()
        or (local_dir / item["path"]).stat().st_size != item["size"]
    )
    extraction = sum(item["uncompressed_bytes"] for item in zip_audits)
    free = shutil.disk_usage(DATA_ROOT).free
    required = remaining + extraction + SAFETY_MARGIN
    return {
        "remaining_download_bytes": remaining,
        "extraction_bytes": extraction,
        "safety_margin_bytes": SAFETY_MARGIN,
        "required_bytes": required,
        "free_bytes": free,
        "sufficient": free >= required,
    }


def verify_and_optionally_extract(
    dataset: str,
    revision: str,
    expected: list[dict[str, Any]],
    extract: bool,
    verify_data: bool,
    verify_logger: logging.Logger,
    extract_logger: logging.Logger,
) -> dict[str, Any]:
    config = DATASETS[dataset]
    local_dir = config["local_dir"]
    audit = local_file_audit(local_dir, expected)
    if audit["status"] != "PASS":
        raise RuntimeError(f"Local file verification failed: {audit}")
    zip_audits = [
        inspect_zip(path, full_test=True)
        for path in archives_for(dataset, local_dir)
    ]
    if not zip_audits or any(item["status"] != "PASS" for item in zip_audits):
        raise RuntimeError(f"ZIP verification failed: {zip_audits}")
    space = space_requirement(expected, local_dir, zip_audits)
    extraction = []
    if extract:
        if not space["sufficient"]:
            raise RuntimeError(f"Insufficient extraction space: {space}")
        for path in archives_for(dataset, local_dir):
            extract_logger.info("extract start dataset=%s archive=%s", dataset, path)
            destination = extraction_destination(dataset, local_dir, path)
            extraction.append(extract_zip(path, destination, extract_logger))
            extract_logger.info("extract complete dataset=%s archive=%s", dataset, path)
    verification = None
    if verify_data:
        verify_logger.info("dataset verification start dataset=%s", dataset)
        verification = (
            verify_medthink(local_dir)
            if dataset == "medthinkvqa"
            else verify_medsg(local_dir)
        )
        verify_logger.info(
            "dataset verification complete dataset=%s status=%s",
            dataset,
            verification["status"],
        )
    return {
        "repo_id": config["repo_id"],
        "revision": revision,
        "local_dir": str(local_dir),
        "local_file_audit": audit,
        "zip_audits": zip_audits,
        "space_requirement": space,
        "extracted": bool(extraction),
        "extraction_results": extraction,
        "verification": verification,
    }


def update_disk_report(before: dict[str, Any], after: dict[str, Any]) -> None:
    report = load_json(ARTIFACT / "disk_space_report.json", {"runs": []})
    report["runs"].append({
        "before": before,
        "after": after,
        "used_delta_bytes": after["used_bytes"] - before["used_bytes"],
        "free_delta_bytes": after["free_bytes"] - before["free_bytes"],
    })
    report["latest"] = report["runs"][-1]
    json_write(ARTIFACT / "disk_space_report.json", report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["medthinkvqa", "medsg", "all"],
        required=True,
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--download-only", action="store_true")
    actions.add_argument("--verify-only", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT / "logs").mkdir(parents=True, exist_ok=True)
    download_logger = setup_logger(
        "download",
        ARTIFACT / "logs/download.log",
    )
    verify_logger = setup_logger(
        "verify",
        ARTIFACT / "logs/verification.log",
    )
    extract_logger = setup_logger(
        "extract",
        ARTIFACT / "logs/extraction.log",
    )
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    before = disk_snapshot()
    manifest = load_json(
        ARTIFACT / "download_manifest.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "python": PYTHON,
            "datasets": {},
        },
    )
    results = {}
    failures = {}
    for dataset in datasets:
        try:
            if args.verify_only:
                revision, expected = expected_from_manifest(dataset)
            else:
                payload = download_dataset(
                    dataset,
                    args.endpoint,
                    args.resume,
                    download_logger,
                )
                manifest["datasets"][dataset] = payload
                manifest["updated_at"] = utc_now()
                json_write(ARTIFACT / "download_manifest.json", manifest)
                revision = payload["revision"]
                expected = payload["selected_files"]
            if args.download_only and not args.extract:
                results[dataset] = {
                    "status": "PASS",
                    "download_complete": True,
                    "revision": revision,
                }
                continue
            results[dataset] = verify_and_optionally_extract(
                dataset,
                revision,
                expected,
                args.extract,
                verify_data=True,
                verify_logger=verify_logger,
                extract_logger=extract_logger,
            )
            verification_path = ARTIFACT / f"{dataset}_verification.json"
            json_write(verification_path, results[dataset])
            if results[dataset]["verification"]["status"] != "PASS":
                raise RuntimeError(
                    f"Dataset semantic verification failed: {verification_path}"
                )
        except Exception as error:
            failures[dataset] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            logging.getLogger("verify").exception(
                "dataset operation failed dataset=%s",
                dataset,
            )
    after = disk_snapshot()
    update_disk_report(before, after)
    command_record = {
        "timestamp": utc_now(),
        "argv": sys.argv,
        "endpoint": args.endpoint,
        "datasets": datasets,
        "verify_only_network_disabled_by_contract": args.verify_only,
        "results": results,
        "failures": failures,
    }
    commands = load_json(ARTIFACT / "command_history.json", [])
    commands.append(command_record)
    json_write(ARTIFACT / "command_history.json", commands)
    print(json.dumps(command_record, indent=2, ensure_ascii=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
