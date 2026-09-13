"""Select the Octopus trainer only for explicitly marked Stage-2 runs."""
from __future__ import annotations

import os
import torch.distributed.fsdp as fsdp

if not hasattr(fsdp, "FSDPModule"):
    class FSDPModule:
        pass

    fsdp.FSDPModule = FSDPModule

from swift.trainers import TrainerFactory
from swift.utils import get_logger

from med_prism.swift_plugins.octopus_trainer import OctopusStage2Trainer

logger = get_logger()
_PREVIOUS_GET_TRAINER_CLS = TrainerFactory.get_trainer_cls.__func__


def _get_trainer_cls(cls, args):
    if os.environ.get("OCTOPUS_STAGE2_ENABLED") == "1":
        if getattr(args, "tuner_type", None) != "lora":
            raise RuntimeError("Octopus Stage-2 requires standard PEFT LoRA")
        return OctopusStage2Trainer
    return _PREVIOUS_GET_TRAINER_CLS(cls, args)


if not getattr(TrainerFactory.get_trainer_cls, "_octopus_conditional", False):
    TrainerFactory.get_trainer_cls = classmethod(_get_trainer_cls)
    TrainerFactory.get_trainer_cls.__func__._octopus_conditional = True

logger.info("Registered conditional Octopus Stage-2 trainer")
