#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from huggingface_hub import HfApi, scan_cache_dir
import huggingface_hub
import transformers


TARGETS = ["Qwen/Qwen3-8B", "Qwen/Qwen3-VL-8B-Instruct"]


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def repo_record(repo) -> dict[str, Any]:
    revisions = []
    for revision in sorted(repo.revisions, key=lambda item: item.commit_hash):
        revisions.append({
            "commit_hash": revision.commit_hash,
            "snapshot_path": str(revision.snapshot_path),
            "size_on_disk": revision.size_on_disk,
            "files": sorted(str(file.file_path) for file in revision.files),
            "refs": sorted(revision.refs),
        })
    return {
        "repo_id": repo.repo_id,
        "repo_type": repo.repo_type,
        "repo_path": str(repo.repo_path),
        "size_on_disk": repo.size_on_disk,
        "revisions": revisions,
    }


def git_value(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo_root), *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "huggingface" / "hub")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.resolve()

    cache_info = scan_cache_dir(cache_dir)
    repos = sorted(cache_info.repos, key=lambda item: (item.repo_type, item.repo_id))
    repo_map = {repo.repo_id: repo for repo in repos}
    incomplete = sorted(str(path) for path in cache_dir.rglob("*.incomplete")) if cache_dir.exists() else []
    disk = shutil.disk_usage(cache_dir)

    cache_payload = {
        "cache_dir": str(cache_dir),
        "huggingface_hub_version": huggingface_hub.__version__,
        "scan_size_on_disk": cache_info.size_on_disk,
        "repo_count": len(repos),
        "warnings": [str(warning) for warning in cache_info.warnings],
        "incomplete_files": incomplete,
        "filesystem": {"total": disk.total, "used": disk.used, "free": disk.free},
        "targets": {},
        "repositories": [repo_record(repo) for repo in repos],
    }
    for model_id in TARGETS:
        repo = repo_map.get(model_id)
        cache_payload["targets"][model_id] = {
            "present": repo is not None,
            "size_on_disk": 0 if repo is None else repo.size_on_disk,
            "revisions": [] if repo is None else [item["commit_hash"] for item in repo_record(repo)["revisions"]],
        }

    api = HfApi()
    upstream_commit = git_value(repo_root, "rev-parse", "v4.4.1^{commit}")
    current_commit = git_value(repo_root, "rev-parse", "HEAD")
    manifest = {
        "status": "BLOCKED_PENDING_SPACE_CHECK",
        "transformers_version": transformers.__version__,
        "huggingface_hub_version": huggingface_hub.__version__,
        "ms_swift": {
            "tag": "v4.4.1",
            "upstream_commit": upstream_commit,
            "current_commit": current_commit,
        },
        "models": [],
    }

    total_expected = 0
    total_cached = 0
    for model_id in TARGETS:
        info = api.model_info(model_id, revision="main", files_metadata=True)
        files = []
        expected_size = 0
        for sibling in info.siblings:
            size = sibling.size or 0
            expected_size += size
            files.append({
                "path": sibling.rfilename,
                "size": size,
                "blob_id": sibling.blob_id,
                "lfs": None if sibling.lfs is None else json_safe(sibling.lfs),
            })
        cached_size = cache_payload["targets"][model_id]["size_on_disk"]
        total_expected += expected_size
        total_cached += min(cached_size, expected_size)
        manifest["models"].append({
            "model_id": model_id,
            "requested_revision": "main",
            "immutable_revision": info.sha,
            "repository_commit": info.sha,
            "last_modified": None if info.last_modified is None else info.last_modified.isoformat(),
            "expected_files": files,
            "expected_total_bytes": expected_size,
            "cached_reusable_bytes_upper_bound": min(cached_size, expected_size),
            "snapshot_cached": info.sha in cache_payload["targets"][model_id]["revisions"],
            "download_status": "not_started",
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "processor_revision": info.sha,
            "tokenizer_revision": info.sha,
            "local_files_only_after_download": True,
        })

    estimated_new = max(total_expected - total_cached, 0)
    space_payload = {
        "status": "PASS" if disk.free >= estimated_new else "BLOCKED",
        "cache_filesystem": {"path": str(cache_dir), "total": disk.total, "used": disk.used, "free": disk.free},
        "project_filesystem": {
            "path": str(repo_root),
            "total": shutil.disk_usage(repo_root).total,
            "used": shutil.disk_usage(repo_root).used,
            "free": shutil.disk_usage(repo_root).free,
        },
        "target_expected_bytes": total_expected,
        "cached_reusable_bytes_upper_bound": total_cached,
        "estimated_new_bytes": estimated_new,
        "space_deficit_bytes": max(estimated_new - disk.free, 0),
        "download_permitted": disk.free >= estimated_new,
        "policy": "Do not delete cache or change cache directory automatically when space is insufficient.",
    }
    manifest["status"] = "READY" if space_payload["download_permitted"] else "BLOCKED_INSUFFICIENT_SPACE"

    (output_dir / "cache_inventory.json").write_text(
        json.dumps(cache_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (output_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    (output_dir / "disk_space_check.json").write_text(
        json.dumps(space_payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    print(json.dumps(space_payload, indent=2, ensure_ascii=True))
    for model in manifest["models"]:
        print(model["model_id"], model["immutable_revision"], model["expected_total_bytes"])
    return 0 if space_payload["download_permitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
