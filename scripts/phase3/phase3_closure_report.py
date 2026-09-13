from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir
    exit_audit = load_json(output / "phase3_exit_code.json")
    trainer_state = load_json(output / "train/checkpoint-10/trainer_state.json")
    adapter = load_json(output / "adapter_state_audit.json")
    reload_audit = load_json(output / "reload_comparison.json")
    gc_audit = load_json(output / "gradient_checkpointing_audit.json")
    log_text = (output / "native_lora_closure.log").read_text(
        encoding="utf-8", errors="replace"
    )
    git_status = subprocess.check_output(
        ["git", "status", "--short"], text=True
    ).strip()
    checks = {
        "swift_sft_exit_code_zero": exit_audit["swift_sft_exit_code"] == 0,
        "checkpoint_exists": exit_audit["checkpoint_exists"],
        "ten_steps_completed": trainer_state["global_step"] == 10,
        "adapter_audit_pass": adapter["status"] == "PASS",
        "reload_pass": reload_audit["status"] == "PASS",
        "independent_reload_consistent": reload_audit[
            "independent_reload_consistent"
        ],
        "gradient_checkpointing_pass": gc_audit["status"] == "PASS",
        "language_layers_checkpointed": (
            gc_audit["language_layer_count"] == 36
            and gc_audit["language_layer_flags_enabled"] == 36
            and gc_audit["dynamic_wrapped_layer_count"] == 36
        ),
        "activation_checkpoint_executed": (
            gc_audit["activation_checkpoint_calls"] > 0
        ),
        "vision_frozen_without_checkpointing": (
            gc_audit["vision_trainable_parameter_tensors"] == 0
            and not gc_audit["vision_gradient_checkpointing"]
        ),
        "no_traceback": "Traceback (most recent call last)" not in log_text,
        "no_invalid_generation_warning": (
            "generation flags are not valid" not in log_text
        ),
        "no_ddp_unused_parameter_warning": (
            "find_unused_parameters=True was specified" not in log_text
        ),
        "git_worktree_clean": not git_status,
    }
    status = "PASS" if all(checks.values()) else "BLOCKED"
    payload = {
        "status": status,
        "checks": checks,
        "swift_sft_exit_code": exit_audit["swift_sft_exit_code"],
        "global_step": trainer_state["global_step"],
        "checkpoint": str(output / "train/checkpoint-10"),
        "adapter_status": adapter["status"],
        "reload_status": reload_audit["status"],
        "gradient_checkpointing": gc_audit,
        "git_status_short": git_status,
        "failures": [key for key, value in checks.items() if not value],
    }
    (output / "phase3_closure_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    report = f"""# Phase 3 Closure Report

Overall status: **{status}**

- swift sft exit code: `{exit_audit['swift_sft_exit_code']}`
- completed optimizer steps: `{trainer_state['global_step']}`
- checkpoint exists: `{exit_audit['checkpoint_exists']}`
- adapter audit: `{adapter['status']}`
- independent reload: `{reload_audit['status']}`
- traceback present: `{not checks['no_traceback']}`
- Git worktree clean: `{not bool(git_status)}`

## Gradient Checkpointing

- Language tower: `{gc_audit['language_class']}`
- Language layers enabled/dynamically wrapped: `{gc_audit['language_layer_flags_enabled']}/{gc_audit['dynamic_wrapped_layer_count']}`
- Activation-checkpoint calls observed: `{gc_audit['activation_checkpoint_calls']}`
- Vision trainable parameter tensors: `{gc_audit['vision_trainable_parameter_tensors']}`
- Vision gradient checkpointing: `{gc_audit['vision_gradient_checkpointing']}`
- Warning scope: `{gc_audit['warning_scope']}`

The caught warning originates while ms-swift disables input-gradient hooks on the
already frozen vision tower. The language tower was enabled first, all 36 decoder
layers were dynamically wrapped, and runtime checkpoint calls were observed.

## Warning Cleanup

- Invalid deterministic generation flags present: `{not checks['no_invalid_generation_warning']}`
- DDP unused-parameter warning present: `{not checks['no_ddp_unused_parameter_warning']}`
- TRANSFORMERS_CACHE is not used by the closure script.
"""
    (output / "phase3_closure_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
