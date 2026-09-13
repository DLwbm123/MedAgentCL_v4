"""Conditional O-LoRA trainer registration."""
import os, torch.distributed.fsdp as fsdp
if not hasattr(fsdp,"FSDPModule"):
    class FSDPModule: pass
    fsdp.FSDPModule=FSDPModule
from swift.trainers import TrainerFactory
from med_prism.swift_plugins.olora_trainer import OLoRATrainer
_PREV=TrainerFactory.get_trainer_cls.__func__
def _get(cls,args): return OLoRATrainer if os.environ.get("OLORA_ENABLED")=="1" else _PREV(cls,args)
if not getattr(TrainerFactory.get_trainer_cls,"_olora_conditional",False):
    TrainerFactory.get_trainer_cls=classmethod(_get); TrainerFactory.get_trainer_cls.__func__._olora_conditional=True
