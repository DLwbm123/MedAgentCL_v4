from __future__ import annotations

import json
from pathlib import Path

import torch

from swift.trainers import TrainerFactory
from swift.tuner_plugin import Tuner, tuners_map
from swift.utils import get_logger

from med_prism.adapters.injection import (
    add_task_bank,
    collect_language_targets,
    freeze_except_task,
    inject_rank1_wrappers,
    iter_rank1_wrappers,
)
from med_prism.checkpoint import load_rank1_checkpoint, save_rank1_checkpoint
from med_prism.config import Rank1BankConfig
from med_prism.swift_plugins.med_prism_trainer import MedPrismRank1Trainer


logger = get_logger()
TUNER_KEY = "med_prism_rank1"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


class MedPrismRank1Tuner(Tuner):
    @staticmethod
    def prepare_model(args, model: torch.nn.Module) -> torch.nn.Module:
        config = Rank1BankConfig.from_environment()
        model.requires_grad_(False)
        if config.source_manifest:
            load_rank1_checkpoint(
                model,
                config.source_manifest,
                trainable_task_id=None,
            )
        else:
            inject_rank1_wrappers(model, dropout=config.dropout)
        loaded_tasks = set(next(iter(iter_rank1_wrappers(model)))[1].task_ids)
        if config.current_task_id in loaded_tasks:
            raise ValueError(
                f"Source manifest already contains current task {config.current_task_id}"
            )
        add_task_bank(
            model,
            task_id=config.current_task_id,
            experts_per_task=config.experts_per_task,
            alpha=config.alpha,
            trainable=True,
        )
        freeze_except_task(model, config.current_task_id)
        wrappers = list(iter_rank1_wrappers(model))
        active_tasks = sorted(
            set(task_id for _, wrapper in wrappers for task_id in wrapper.task_ids)
        )
        for _, wrapper in wrappers:
            wrapper.set_active_tasks(active_tasks)
        model.med_prism_config = config

        inventory = collect_language_targets(model)
        wrapper_classes = sorted({type(wrapper).__name__ for _, wrapper in wrappers})
        trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        old_trainable = [
            name for name in trainable
            if f"task_{config.current_task_id:04d}__" not in name
        ]
        peft_wrappers = [
            name for name, module in model.named_modules()
            if hasattr(module, "lora_A") or hasattr(module, "lora_B")
        ]
        if (
            len(wrappers) != 72
            or inventory.vision
            or inventory.merger
            or old_trainable
            or peft_wrappers
        ):
            raise RuntimeError(
                "Med-PRISM tuner activation failed: "
                f"wrappers={len(wrappers)}, vision={inventory.vision}, "
                f"merger={inventory.merger}, old_trainable={old_trainable[:3]}, "
                f"peft_wrappers={peft_wrappers[:3]}"
            )
        audit = {
            "status": "PASS",
            "tuner_type": TUNER_KEY,
            "tuner_registry_class": f"{MedPrismRank1Tuner.__module__}.{MedPrismRank1Tuner.__name__}",
            "wrapper_class": wrapper_classes,
            "wrapper_count": len(wrappers),
            "q_proj": inventory.q_proj,
            "v_proj": inventory.v_proj,
            "language": inventory.language,
            "vision": inventory.vision,
            "merger": inventory.merger,
            "target_module_hash": inventory.sha256,
            "target_modules": list(inventory.names),
            "task_ids": active_tasks,
            "current_task_id": config.current_task_id,
            "experts_per_wrapper": {
                str(task_id): len(wrappers[0][1].task_experts(task_id))
                for task_id in active_tasks
            },
            "expert_count": sum(wrapper.num_experts for _, wrapper in wrappers),
            "expert_tensor_count": sum(wrapper.num_experts * 2 for _, wrapper in wrappers),
            "trainable_parameter_names": trainable,
            "old_trainable_parameter_names": old_trainable,
            "peft_lora_wrapper_count": len(peft_wrappers),
        }
        output = Path(config.output_dir or args.output_dir)
        _write_json(output / "target_module_audit.json", audit)
        logger.info(f"Med-PRISM rank-1 activation audit: {audit}")
        return model

    @staticmethod
    def save_pretrained(
        model,
        save_directory: str,
        state_dict=None,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        save_rank1_checkpoint(model, save_directory)

    @staticmethod
    def from_pretrained(model, model_id: str, **kwargs):
        config = Rank1BankConfig.from_environment()
        load_rank1_checkpoint(
            model,
            model_id,
            trainable_task_id=config.current_task_id,
        )
        model.med_prism_config = config
        return model


_ORIGINAL_GET_TRAINER_CLS = TrainerFactory.get_trainer_cls.__func__


def _conditional_get_trainer_cls(cls, args):
    if getattr(args, "tuner_type", None) == TUNER_KEY:
        return MedPrismRank1Trainer
    return _ORIGINAL_GET_TRAINER_CLS(cls, args)


if not getattr(TrainerFactory.get_trainer_cls, "_med_prism_conditional", False):
    conditional = classmethod(_conditional_get_trainer_cls)
    TrainerFactory.get_trainer_cls = conditional
    TrainerFactory.get_trainer_cls.__func__._med_prism_conditional = True

tuners_map[TUNER_KEY] = MedPrismRank1Tuner
logger.info(
    "Registered tuner med_prism_rank1 and conditional TrainerFactory extension"
)
