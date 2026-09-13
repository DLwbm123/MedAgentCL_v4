#!/usr/bin/env python3
"""Migrate the frozen Kimi request contract to temperature 1.0 / 512 tokens without touching IDs or history."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/root/MedAgentCL_v4")
ARTIFACT = ROOT / "artifacts/medicalskill_cl_kimi_k3_pilot"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    artifact = ARTIFACT
    prompt_path = artifact / "prompt_manifest.json"
    selection_path = artifact / "pilot_selection_manifest.json"
    prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    old_hash = prompt["prompt_hash"]
    old_temperature = prompt["temperature"]
    old_max_tokens = prompt["max_completion_tokens"]
    if old_temperature not in (0, 1.0) or old_max_tokens not in (128, 512):
        raise RuntimeError("Unexpected pre-migration generation contract")
    if len(selection["stable_record_ids"]) != 2_000 or len(set(selection["stable_record_ids"])) != 2_000:
        raise RuntimeError("Pilot IDs changed before migration")
    if selection["stable_record_ids"][0] != "caption_ct_train_005626":
        raise RuntimeError("Canary ID changed before migration")
    attempts_before = sum(1 for line in (artifact / "pilot_results.jsonl").open() if line.strip())
    if attempts_before != 2:
        raise RuntimeError(f"Expected two historical failed attempts, found {attempts_before}")

    prompt["temperature"] = 1.0
    prompt["max_completion_tokens"] = 512
    prompt["generation_contract_version"] = "kimi-code-temperature-1-max512-v1"
    basis = {
        key: prompt[key]
        for key in (
            "prompt_version", "system_prompt", "user_template", "model",
            "reasoning_effort", "temperature", "max_completion_tokens",
        )
    }
    prompt["prompt_hash"] = sha_text(canonical_json(basis))
    write_json(prompt_path, prompt)
    selection["prompt_hash"] = prompt["prompt_hash"]
    write_json(selection_path, selection)

    hashes_path = artifact / "input_file_hashes.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    for path_text in list(hashes):
        if path_text.endswith("/prompt_manifest.json"):
            hashes[path_text] = sha_file(prompt_path)
        elif path_text.endswith("/pilot_selection_manifest.json"):
            hashes[path_text] = sha_file(selection_path)
    write_json(hashes_path, hashes)

    write_json(
        artifact / "runtime_api_contract.json",
        {
            "status": "READY_FOR_SINGLE_CANARY",
            "effective_base_url": "https://api.kimi.com/coding/v1",
            "scheme": "https",
            "hostname": "api.kimi.com",
            "base_path": "/coding/v1",
            "chat_completions_url": "https://api.kimi.com/coding/v1/chat/completions",
            "base_url_priority": ["KIMI_BASE_URL", "MOONSHOT_BASE_URL", "default"],
            "credential_priority": ["KIMI_API_KEY", "MOONSHOT_API_KEY"],
            "model": "kimi-k3",
            "reasoning_effort": "low",
            "temperature": 1.0,
            "max_completion_tokens": 512,
            "top_p_sent": False,
            "structured_output_mode": "json_schema_strict",
            "prompt_version": prompt["prompt_version"],
            "prompt_hash": prompt["prompt_hash"],
            "schema_version": prompt["schema_version"],
            "schema_hash": prompt["schema_hash"],
            "pilot_manifest_count": 2_000,
            "canary_stable_record_id": selection["stable_record_ids"][0],
        },
    )
    write_json(
        artifact / "temperature_contract_migration.json",
        {
            "status": "PASS",
            "reason": "Kimi Code API requires temperature=1 for kimi-k3; explicitly authorized by user after preserved HTTP 400.",
            "old_temperature": old_temperature,
            "new_temperature": 1.0,
            "old_max_completion_tokens": old_max_tokens,
            "new_max_completion_tokens": 512,
            "old_prompt_hash": old_hash,
            "new_prompt_hash": prompt["prompt_hash"],
            "medical_prompt_text_changed": False,
            "schema_changed": False,
            "pilot_ids_changed": False,
            "historical_attempt_rows_before": attempts_before,
            "historical_attempt_rows_after": sum(1 for line in (artifact / "pilot_results.jsonl").open() if line.strip()),
        },
    )
    (artifact / "commands.txt").write_text(
        "# No-API contract audit\n"
        "scripts/medicalskill_kimi_k3/run_kimi_k3_secure.sh --config-audit\n\n"
        "# Run only the frozen canary; temperature is explicitly 1.0\n"
        "scripts/medicalskill_kimi_k3/run_kimi_k3_secure.sh --canary\n\n"
        "# Run the remaining Pilot only after canary PASS\n"
        "scripts/medicalskill_kimi_k3/run_kimi_k3_secure.sh\n\n"
        "# Audit the completed 2,000-ID Pilot and stop\n"
        "/root/anaconda3/envs/medagentcl_v4/bin/python scripts/medicalskill_kimi_k3/audit_kimi_k3_pilot.py\n",
        encoding="utf-8",
    )
    targets = sorted(path for path in artifact.iterdir() if path.is_file() and path.name != "artifact_hashes.json")
    write_json(artifact / "artifact_hashes.json", {path.name: sha_file(path) for path in targets})
    print(json.dumps({"status": "PASS", "temperature": 1.0, "max_completion_tokens": 512, "prompt_hash": prompt["prompt_hash"], "attempt_rows": attempts_before}))


if __name__ == "__main__":
    main()
