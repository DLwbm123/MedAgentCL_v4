#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import MISSING, fields
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import HfArgumentParser

from swift.arguments import SftArguments
from swift.arguments.base_args.model_args import ModelArguments
from swift.model import LLMModelType, MODEL_MAPPING
from swift.utils.utils import _patch_get_type_hints


PLANNED_FIELDS = [
    "freeze_vit", "freeze_aligner", "freeze_llm", "tuner_backend",
    "tuner_type", "target_modules", "external_plugins", "dataset",
    "template", "use_hf",
]


def default_value(field):
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:
        return field.default_factory()
    return None


def fake_init_model_info(self):
    self.model_meta = MODEL_MAPPING[LLMModelType.qwen3]
    self.model_info = SimpleNamespace(
        task_type="causal_lm",
        num_labels=None,
        model_dir="/tmp/offline-qwen3-config",
        model_type=LLMModelType.qwen3,
        max_model_len=32768,
        rope_scaling=None,
        torch_dtype=torch.bfloat16,
    )
    self.task_type = self.model_info.task_type
    self.num_labels = self.model_info.num_labels
    self.model_dir = self.model_info.model_dir
    self.model_type = self.model_info.model_type
    return torch.bfloat16


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataclass_fields = {field.name: field for field in fields(SftArguments)}
    inventory = {
        name: {
            "exists": name in dataclass_fields,
            "type": None if name not in dataclass_fields else str(dataclass_fields[name].type),
            "default": None if name not in dataclass_fields else default_value(dataclass_fields[name]),
        }
        for name in PLANNED_FIELDS + ["dataset_format"]
    }

    with _patch_get_type_hints():
        full_parser = HfArgumentParser([SftArguments])
    full_help = full_parser.format_help()
    help_flags = {
        name: f"--{name}" in full_help
        for name in PLANNED_FIELDS
    }

    construction_stdout = io.StringIO()
    construction_stderr = io.StringIO()
    with patch.object(ModelArguments, "_init_model_info", fake_init_model_info):
        with redirect_stdout(construction_stdout), redirect_stderr(construction_stderr):
            instance = SftArguments(
                model="offline-qwen3-config",
                dataset=["dummy"],
                output_dir="/tmp/medagentcl_phase2_args",
                add_version=False,
                report_to=[],
                use_hf=False,
                tuner_backend="peft",
                tuner_type="lora",
                target_modules=["q_proj", "v_proj"],
                freeze_vit=True,
                freeze_aligner=True,
                freeze_llm=False,
                external_plugins=[],
                template="qwen3",
            )

    constructed = {
        name: getattr(instance, name)
        for name in PLANNED_FIELDS
    }
    constructed["class"] = type(instance).__name__
    constructed["model_weights_loaded"] = False
    constructed["model_info_source"] = "test monkeypatch; registry metadata only"

    all_required_exist = all(inventory[name]["exists"] for name in PLANNED_FIELDS)
    all_help_flags_exist = all(help_flags.values())
    constructed_matches = (
        constructed["tuner_backend"] == "peft"
        and constructed["tuner_type"] == "lora"
        and constructed["target_modules"] == ["q_proj", "v_proj"]
        and constructed["freeze_vit"] is True
        and constructed["freeze_aligner"] is True
        and constructed["freeze_llm"] is False
    )

    payload = {
        "status": "PASS" if all_required_exist and all_help_flags_exist and constructed_matches else "BLOCKED",
        "sft_argument_field_count": len(dataclass_fields),
        "planned_fields": inventory,
        "full_hf_argument_parser_help_line_count": len(full_help.splitlines()),
        "help_flags": help_flags,
        "minimal_construction": constructed,
        "construction_stdout": construction_stdout.getvalue(),
        "construction_stderr": construction_stderr.getvalue(),
        "cli_help_analysis": {
            "official_swift_sft_help_is_full": False,
            "reason": "swift/cli/sft.py try_init_unsloth uses argparse with help enabled and exits before sft_main",
            "tuner_backend_preparse_purpose": "conditionally import unsloth before the full SFT pipeline",
            "dataset_format_exists": inventory["dataset_format"]["exists"],
            "dataset_format_migration": "Do not pass this legacy flag; v4 consumes messages/images schema directly.",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    help_path = args.output.with_name("sft_full_parser_help.txt")
    help_path.write_text(full_help, encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
