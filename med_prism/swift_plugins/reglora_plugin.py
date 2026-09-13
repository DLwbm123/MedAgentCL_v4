"""Conditional RegLoRA trainer registration."""
import os, torch.distributed.fsdp as fsdp
if not hasattr(fsdp,"FSDPModule"):
    class FSDPModule: pass
    fsdp.FSDPModule=FSDPModule
from swift.trainers import TrainerFactory
from med_prism.swift_plugins.reglora_trainer import RegLoRATrainer
_PREV=TrainerFactory.get_trainer_cls.__func__
def _get(cls,args): return RegLoRATrainer if os.environ.get("REGLORA_ENABLED")=="1" else _PREV(cls,args)
if not getattr(TrainerFactory.get_trainer_cls,"_reglora_conditional",False):
    TrainerFactory.get_trainer_cls=classmethod(_get); TrainerFactory.get_trainer_cls.__func__._reglora_conditional=True
