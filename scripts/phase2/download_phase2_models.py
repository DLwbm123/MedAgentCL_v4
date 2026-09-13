#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "huggingface" / "hub")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    required = sum(model["expected_total_bytes"] for model in manifest["models"] if not model["snapshot_cached"])
    free = shutil.disk_usage(args.cache_dir).free
    if free < required:
        raise RuntimeError(f"Insufficient cache space: free={free}, required_upper_bound={required}")

    for model in manifest["models"]:
        snapshot = snapshot_download(
            repo_id=model["model_id"],
            revision=model["immutable_revision"],
            cache_dir=str(args.cache_dir),
        )
        resolved = Path(snapshot).resolve()
        if resolved.name != model["immutable_revision"]:
            raise RuntimeError(f"Unexpected snapshot revision: {resolved}")
        print(model["model_id"], resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
