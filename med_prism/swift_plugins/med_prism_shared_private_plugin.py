from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed as dist

from swift.trainers import TrainerFactory
from swift.tuner_plugin import Tuner, tuners_map
from swift.utils import get_logger

from med_prism.adapters.injection import (
    add_task_bank,
    freeze_shared_private_except,
    inject_shared_private_wrappers,
    iter_shared_private_wrappers,
)
from med_prism.checkpoint.shared_private_checkpoint import (
    load_private_component,
    load_shared_component,
    save_shared_private_components,
)
from med_prism.config import SharedPrivateConfig
from med_prism.swift_plugins.shared_private_trainer import (
    MedPrismSharedPrivateTrainer,
)


logger = get_logger()
TUNER_KEY = "med_prism_shared_private"


def _is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _write_json(path: Path, payload: dict) -> None:
    if not _is_main_process():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


class MedPrismSharedPrivateTuner(Tuner):
    @staticmethod
    def prepare_model(args, model: torch.nn.Module) -> torch.nn.Module:
        config = SharedPrivateConfig.from_environment()
        model.requires_grad_(False)
        inventory = inject_shared_private_wrappers(
            model,
            shared_rank=config.shared_rank,
            shared_alpha=config.shared_alpha,
            shared_lr_scale=(
                config.shared_lr_scale
                if config.shared_gradient_hook_enabled
                else 1.0
            ),
            dropout=config.dropout,
            shared_trainable=True,
            private_adapter_type=config.private_adapter_type,
        )
        if config.shared_source_manifest:
            load_shared_component(model, config.shared_source_manifest)
        else:
            model._med_prism_shared_loaded = True
            for _, wrapper in iter_shared_private_wrappers(model):
                wrapper.shared.reset_anchor()

        loaded_private = []
        for manifest_path in config.private_source_manifests:
            loaded_private.append(
                load_private_component(model, manifest_path, trainable=False)
            )
        loaded_tasks = {int(item["task_id"]) for item in loaded_private}
        if config.current_task_id in loaded_tasks:
            raise ValueError(
                f"Current private task already exists: {config.current_task_id}"
            )
        add_task_bank(
            model,
            task_id=config.current_task_id,
            experts_per_task=config.private_experts_per_task,
            private_rank=config.effective_private_rank,
            alpha=config.private_alpha,
            trainable=True,
        )
        freeze_shared_private_except(model, config.current_task_id)
        active_tasks = sorted(loaded_tasks | {config.current_task_id})
        wrappers = list(iter_shared_private_wrappers(model))
        for _, wrapper in wrappers:
            wrapper.set_active_tasks(active_tasks)
        model.med_prism_shared_private_config = config

        trainable = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        current_markers = (
            f"task_{config.current_task_id:04d}__",
            f".task_adapters.task_{config.current_task_id:04d}.",
        )
        invalid_trainable = [
            name
            for name in trainable
            if ".shared." not in name
            and not any(marker in name for marker in current_markers)
        ]
        peft_wrappers = [
            name
            for name, module in model.named_modules()
            if hasattr(module, "lora_A") or hasattr(module, "lora_B")
        ]
        if (
            len(wrappers) != 72
            or inventory.language != 72
            or inventory.vision
            or inventory.merger
            or invalid_trainable
            or peft_wrappers
        ):
            raise RuntimeError(
                "Shared/private tuner activation failed: "
                f"wrappers={len(wrappers)}, invalid={invalid_trainable[:3]}, "
                f"peft={peft_wrappers[:3]}"
            )
        audit = {
            "status": "PASS",
            "tuner_type": TUNER_KEY,
            "wrapper_count": len(wrappers),
            "wrapper_class": sorted(
                {type(wrapper).__name__ for _, wrapper in wrappers}
            ),
            "q_proj": inventory.q_proj,
            "v_proj": inventory.v_proj,
            "language": inventory.language,
            "vision": inventory.vision,
            "merger": inventory.merger,
            "target_modules": list(inventory.names),
            "target_module_hash": inventory.sha256,
            "shared_rank": config.shared_rank,
            "shared_alpha": config.shared_alpha,
            "shared_scaling": config.shared_scaling,
            "method": config.method,
            "method_version": config.method_version,
            "method_variant": config.method_variant,
            "shared_optimizer_mode": config.shared_optimizer_mode,
            "shared_lr": config.shared_lr,
            "private_lr": config.private_lr,
            "shared_gradient_hook_enabled": config.shared_gradient_hook_enabled,
            "shared_drift_mode": config.shared_drift_mode,
            "key_isolation_enabled": config.key_isolation_enabled,
            "shared_load_count": 1,
            "private_experts_per_task": config.private_experts_per_task,
            "private_rank": config.effective_private_rank,
            "private_adapter_type": config.private_adapter_type,
            "private_alpha": config.private_alpha,
            "private_scaling": config.private_scaling,
            "active_private_tasks": active_tasks,
            "old_private_forward_active": sorted(loaded_tasks),
            "trainable_parameter_names": trainable,
            "invalid_trainable_parameter_names": invalid_trainable,
            "peft_lora_wrapper_count": len(peft_wrappers),
        }
        output = Path(config.output_dir or args.output_dir)
        _write_json(output / "target_module_audit.json", audit)
        logger.info(f"Med-PRISM shared/private activation audit: {audit}")
        return model

    @staticmethod
    def save_pretrained(
        model,
        save_directory: str,
        state_dict=None,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        if not _is_main_process():
            return
        config = getattr(
            model,
            "med_prism_shared_private_config",
            SharedPrivateConfig.from_environment(),
        )
        components = save_shared_private_components(model, config)
        _write_json(
            Path(save_directory) / "shared_private_checkpoint.json",
            {
                "method": TUNER_KEY,
                "shared_manifest": str(
                    Path(config.shared_output_dir) / "shared_manifest.json"
                ),
                "private_manifest": str(
                    Path(config.private_output_dir) / "private_manifest.json"
                ),
                "shared_weights_sha256": components["shared"]["weights_sha256"],
                "private_weights_sha256": components["private"]["weights_sha256"],
                "merged": False,
            },
        )

    @staticmethod
    def from_pretrained(model, model_id: str, **kwargs):
        raise RuntimeError("Use explicit shared/private component manifests")


_PREVIOUS_GET_TRAINER_CLS = TrainerFactory.get_trainer_cls.__func__


def _shared_private_get_trainer_cls(cls, args):
    if getattr(args, "tuner_type", None) == TUNER_KEY:
        return MedPrismSharedPrivateTrainer
    return _PREVIOUS_GET_TRAINER_CLS(cls, args)


if not getattr(
    TrainerFactory.get_trainer_cls, "_med_prism_shared_private_conditional", False
):
    TrainerFactory.get_trainer_cls = classmethod(_shared_private_get_trainer_cls)
    TrainerFactory.get_trainer_cls.__func__._med_prism_shared_private_conditional = True

tuners_map[TUNER_KEY] = MedPrismSharedPrivateTuner
logger.info("Registered tuner med_prism_shared_private")
