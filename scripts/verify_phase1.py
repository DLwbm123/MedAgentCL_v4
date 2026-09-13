#!/usr/bin/env python3
"""Offline Phase 1 verification for the fixed ms-swift v4 environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


EXPECTED_UPSTREAM = "98a09c18cdf95ff07051324b9b8cc90f5184b24b"
EXPECTED_OLD_FINGERPRINT = "eb7a06159a8aa85f31ce4acecbab9447ee4e88d243eb97fa381ad26f8ae08749"
EXPECTED_OLD_STAT = "size=4096|mtime=2026-07-07 05:37:19.989160086 +0000|mode=drwxr-xr-x"
EXPECTED_PYTHON = "3.12.7"
EXPECTED_REPO = Path("/root/MedAgentCL_v4")
EXPECTED_ENV = Path("/root/anaconda3/envs/medagentcl_v4")
OLD_REPO = Path("/root/MedAgentCL")


def run(command: list[str], *, cwd: Path | None = None, timeout: int = 180) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=os.environ.copy(),
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def package_version(distribution: str) -> str:
    return importlib.metadata.version(distribution)


def import_record(module_name: str) -> dict[str, Any]:
    module = importlib.import_module(module_name)
    return {
        "module": module_name,
        "file": getattr(module, "__file__", None),
        "version": getattr(module, "__version__", None),
    }


def cache_state(cache_root: Path) -> dict[str, Any]:
    total_bytes = 0
    file_count = 0
    entries: list[str] = []
    if cache_root.exists():
        for path in sorted(cache_root.rglob("*")):
            if path.is_file():
                stat = path.stat()
                total_bytes += stat.st_size
                file_count += 1
                entries.append(f"{path.relative_to(cache_root)}\0{stat.st_size}")
    digest = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
    return {
        "path": str(cache_root),
        "exists": cache_root.exists(),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "path_size_sha256": digest,
    }


def old_repo_fingerprint() -> str:
    command = (
        "find /root/MedAgentCL/med_prism /root/MedAgentCL/scripts "
        "/root/MedAgentCL/examples -type f ! -path '*/__pycache__/*' -print0 "
        "| sort -z | xargs -0 sha256sum | sha256sum"
    )
    result = subprocess.run(
        ["bash", "-o", "pipefail", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.split()[0]


def old_repo_stat() -> str:
    result = subprocess.run(
        ["stat", "-c", "size=%s|mtime=%y|mode=%A", str(OLD_REPO)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def is_under(path_text: str | None, parent: Path) -> bool:
    if not path_text:
        return False
    try:
        Path(path_text).resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("Offline environment flags must both be set to 1")

    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache_before = cache_state(cache_root)

    imports = {}
    for module_name in (
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "qwen_vl_utils",
        "decord",
        "peft",
        "accelerate",
        "datasets",
        "trl",
        "swift",
    ):
        imports[module_name] = import_record(module_name)

    import torch
    import torchaudio
    import torchvision
    import transformers
    from transformers import AutoProcessor, Qwen3ForCausalLM, Qwen3VLForConditionalGeneration
    import swift
    from swift.model import LLMModelType, MLLMModelType, MODEL_MAPPING
    from swift.template import TEMPLATE_MAPPING
    from swift.template.constant import LLMTemplateType, MLLMTemplateType

    class_imports = {
        "AutoProcessor": f"{AutoProcessor.__module__}.{AutoProcessor.__name__}",
        "Qwen3ForCausalLM": f"{Qwen3ForCausalLM.__module__}.{Qwen3ForCausalLM.__name__}",
        "Qwen3VLForConditionalGeneration": (
            f"{Qwen3VLForConditionalGeneration.__module__}.{Qwen3VLForConditionalGeneration.__name__}"
        ),
    }

    registry_checks = {}
    model_keys = {
        "qwen3": LLMModelType.qwen3,
        "qwen3_vl": MLLMModelType.qwen3_vl,
    }
    for label, key in model_keys.items():
        meta = MODEL_MAPPING.get(key)
        registry_checks[f"model_{label}"] = {
            "query_entrypoint": "swift.model.MODEL_MAPPING.get",
            "key": key,
            "found": meta is not None,
            "result_type": None if meta is None else f"{type(meta).__module__}.{type(meta).__name__}",
            "template": None if meta is None else meta.template,
            "architectures": None if meta is None else meta.architectures,
        }

    template_keys = {
        "qwen3": LLMTemplateType.qwen3,
        "qwen3_vl": MLLMTemplateType.qwen3_vl,
    }
    for label, key in template_keys.items():
        meta = TEMPLATE_MAPPING.get(key)
        registry_checks[f"template_{label}"] = {
            "query_entrypoint": "swift.template.TEMPLATE_MAPPING.get",
            "key": key,
            "found": meta is not None,
            "result_type": None if meta is None else f"{type(meta).__module__}.{type(meta).__name__}",
            "template_class": None if meta is None else f"{meta.template_cls.__module__}.{meta.template_cls.__name__}",
        }

    registry_payload = {
        "model_registry_import_path": str(Path(importlib.import_module("swift.model").__file__).resolve()),
        "template_registry_import_path": str(Path(importlib.import_module("swift.template").__file__).resolve()),
        "checks": registry_checks,
    }
    write_json(output_dir / "registry_check.json", registry_payload)

    pip_bin = Path(sys.executable).with_name("pip")
    swift_bin = Path(sys.executable).with_name("swift")
    pip_check = run([str(pip_bin), "check"])
    (output_dir / "pip_check.txt").write_text(
        pip_check["stdout"] + pip_check["stderr"], encoding="utf-8"
    )

    cli_results = [
        run([str(swift_bin), "--help"]),
        run([str(swift_bin), "sft", "--help"]),
    ]
    cli_text = []
    for result in cli_results:
        cli_text.extend(
            [
                f"$ {' '.join(result['command'])}",
                f"returncode={result['returncode']}",
                result["stdout"],
                result["stderr"],
            ]
        )
    (output_dir / "cli_help_check.txt").write_text("\n".join(cli_text), encoding="utf-8")

    pip_show = run([str(pip_bin), "show", "ms-swift"])
    git_branch = run(["git", "branch", "--show-current"], cwd=repo_root)
    git_remote = run(["git", "remote", "-v"], cwd=repo_root)
    git_tag = run(["git", "rev-parse", "v4.4.1^{commit}"], cwd=repo_root)
    git_head = run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    git_ancestor = run(["git", "merge-base", "--is-ancestor", EXPECTED_UPSTREAM, "HEAD"], cwd=repo_root)
    git_status = run(["git", "status", "--short"], cwd=repo_root)
    git_log = run(["git", "log", "--oneline", "--decorate", "-5"], cwd=repo_root)
    git_diff_stat = run(["git", "diff", "--stat", f"{EXPECTED_UPSTREAM}..HEAD"], cwd=repo_root)
    git_state = "\n".join(
        [
            "=== branch ===",
            git_branch["stdout"],
            "=== remotes ===",
            git_remote["stdout"],
            "=== v4.4.1 commit ===",
            git_tag["stdout"],
            "=== HEAD ===",
            git_head["stdout"],
            f"=== upstream ancestor returncode: {git_ancestor['returncode']} ===",
            "=== log ===",
            git_log["stdout"],
            "=== upstream..HEAD diff stat ===",
            git_diff_stat["stdout"],
            "=== status --short ===",
            git_status["stdout"],
        ]
    )
    (output_dir / "git_state.txt").write_text(git_state, encoding="utf-8")

    old_fingerprint = old_repo_fingerprint()
    old_stat = old_repo_stat()
    forbidden_installed = {}
    for distribution in ("flash-attn", "vllm", "deepspeed"):
        try:
            forbidden_installed[distribution] = package_version(distribution)
        except importlib.metadata.PackageNotFoundError:
            forbidden_installed[distribution] = None

    custom_sources = [
        repo_root / "scripts" / "bootstrap_phase1.sh",
        repo_root / "scripts" / "install_pytorch_cu124.sh",
        repo_root / "scripts" / "run_phase1_verify.sh",
        repo_root / "scripts" / "verify_phase1.py",
    ]
    forbidden_call_text = ".from" + "_pretrained("
    custom_source_has_forbidden_call = any(
        forbidden_call_text in source.read_text(encoding="utf-8") for source in custom_sources
    )

    cache_after = cache_state(cache_root)
    cache_unchanged = (
        cache_before["file_count"] == cache_after["file_count"]
        and cache_before["total_bytes"] == cache_after["total_bytes"]
        and cache_before["path_size_sha256"] == cache_after["path_size_sha256"]
    )

    python_path = str(Path(sys.executable).resolve())
    swift_path = str(Path(swift.__file__).resolve())
    old_sys_path_entries = [entry for entry in sys.path if is_under(entry, OLD_REPO)]
    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torchvision": torchvision.__version__,
        "torchaudio": torchaudio.__version__,
        "transformers": transformers.__version__,
        "tokenizers": package_version("tokenizers"),
        "qwen-vl-utils": package_version("qwen-vl-utils"),
        "decord": package_version("decord"),
        "peft": package_version("peft"),
        "accelerate": package_version("accelerate"),
        "datasets": package_version("datasets"),
        "safetensors": package_version("safetensors"),
        "trl": package_version("trl"),
        "huggingface-hub": package_version("huggingface-hub"),
        "ms-swift": package_version("ms-swift"),
    }

    checks = {
        "python_exact_3_12_7": platform.python_version() == EXPECTED_PYTHON,
        "python_path_is_new_env": Path(sys.executable).resolve() == EXPECTED_ENV / "bin" / "python3.12",
        "repo_root_is_expected": repo_root == EXPECTED_REPO,
        "torch_2_5_1_cu124": torch.__version__ == "2.5.1+cu124" and torch.version.cuda == "12.4",
        "torchvision_0_20_1_cu124": torchvision.__version__ == "0.20.1+cu124",
        "torchaudio_2_5_1_cu124": torchaudio.__version__ == "2.5.1+cu124",
        "cuda_available": torch.cuda.is_available(),
        "two_a100_visible": len(gpu_names) == 2 and all("A100" in name for name in gpu_names),
        "qwen_class_imports": len(class_imports) == 3,
        "model_registry_qwen3": registry_checks["model_qwen3"]["found"],
        "model_registry_qwen3_vl": registry_checks["model_qwen3_vl"]["found"],
        "template_registry_qwen3": registry_checks["template_qwen3"]["found"],
        "template_registry_qwen3_vl": registry_checks["template_qwen3_vl"]["found"],
        "pip_check": pip_check["returncode"] == 0,
        "swift_help": cli_results[0]["returncode"] == 0,
        "swift_sft_help": cli_results[1]["returncode"] == 0,
        "swift_import_from_new_repo": is_under(swift_path, EXPECTED_REPO),
        "no_old_repo_sys_path": not old_sys_path_entries,
        "editable_show_points_to_new_repo": str(EXPECTED_REPO) in pip_show["stdout"] and str(OLD_REPO) + "\n" not in pip_show["stdout"],
        "upstream_tag_exact": git_tag["stdout"].strip() == EXPECTED_UPSTREAM,
        "upstream_is_ancestor": git_ancestor["returncode"] == 0,
        "branch_medagentcl_v4": git_branch["stdout"].strip() == "medagentcl-v4",
        "git_worktree_clean": not git_status["stdout"].strip(),
        "hf_cache_unchanged": cache_unchanged,
        "offline_flags_set": os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "no_forbidden_model_processor_call_in_phase1_scripts": not custom_source_has_forbidden_call,
        "forbidden_accelerators_absent": all(value is None for value in forbidden_installed.values()),
        "old_repo_code_fingerprint_unchanged": old_fingerprint == EXPECTED_OLD_FINGERPRINT,
        "old_repo_root_stat_unchanged": old_stat == EXPECTED_OLD_STAT,
        "med_prism_not_migrated": not (repo_root / "med_prism").exists(),
    }
    overall = "PASS" if all(checks.values()) else "PARTIAL"

    environment_report = {
        "status": overall,
        "offline_mode": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "paths": {
            "repo_root": str(repo_root),
            "python": python_path,
            "swift_module": swift_path,
            "sys_path": sys.path,
            "old_repo_sys_path_entries": old_sys_path_entries,
        },
        "versions": versions,
        "imports": imports,
        "class_imports": class_imports,
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_names": gpu_names,
        },
        "pip_show_ms_swift": pip_show,
        "forbidden_packages": forbidden_installed,
        "huggingface_cache": {
            "before": cache_before,
            "after": cache_after,
            "unchanged": cache_unchanged,
        },
        "old_repo_guard": {
            "fingerprint_before": EXPECTED_OLD_FINGERPRINT,
            "fingerprint_after": old_fingerprint,
            "root_stat_before": EXPECTED_OLD_STAT,
            "root_stat_after": old_stat,
        },
        "checks": checks,
    }
    write_json(output_dir / "environment_report.json", environment_report)

    package_lines = [f"- `{name}`: `{version}`" for name, version in versions.items()]
    check_lines = [f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in checks.items()]
    report = "\n".join(
        [
            "# Phase 1 Verification Report",
            "",
            f"Overall status: **{overall}**",
            "",
            "## Scope",
            "",
            "Phase 1 initialized the fixed ms-swift v4.4.1 baseline, created an isolated",
            "Python/CUDA environment, locked dependencies, and ran offline import, registry,",
            "template, CLI, Git, cache, and legacy-project integrity checks. No model or",
            "processor weights were loaded, no model cache files were downloaded, no training",
            "was started, and Med-PRISM was not migrated.",
            "",
            "## Installed Versions",
            "",
            *package_lines,
            "",
            "## Candidate Pin Deviation",
            "",
            "The requested `decord==0.6.0` version is unchanged, but its package source is",
            "the exact conda-forge Python 3.12 build `np2py312h48de876_2`. The PyPI wheel",
            "contains an internal CPython 3.6-only tag and fails `pip check` on Python 3.12.",
            "The Conda solve changed the transitive NumPy build to `2.5.1`; Python remained",
            "exactly `3.12.7`. Resolver evidence is preserved in `output/phase1/`.",
            "",
            "## Acceptance Checks",
            "",
            "| Check | Result |",
            "| --- | --- |",
            *check_lines,
            "",
            "## Registry Evidence",
            "",
            "The model registry was queried through `swift.model.MODEL_MAPPING.get`; the",
            "template registry was queried through `swift.template.TEMPLATE_MAPPING.get`.",
            "Detailed keys, result types, template classes, and import paths are recorded in",
            "`registry_check.json`.",
            "",
            "## Remaining Phase 2 Risks",
            "",
            "- Qwen3-VL processor/model artifact compatibility is still untested because Phase 1 forbids weight or processor loading.",
            "- Med-PRISM adapter, dataset, template, and training argument migration remains intentionally unimplemented.",
            "- Full training memory use, distributed launch behavior, and optional accelerator packages remain unvalidated.",
            "",
            "Phase 1 stops here; Phase 2 has not started.",
            "",
        ]
    )
    (output_dir / "phase1_report.md").write_text(report, encoding="utf-8")

    print(f"Phase 1 verification status: {overall}")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    print(f"Artifacts: {output_dir}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
