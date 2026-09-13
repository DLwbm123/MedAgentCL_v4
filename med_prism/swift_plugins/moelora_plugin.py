"""Install CoIN dense-router PEFT patch before adapter construction."""
import os
from med_prism.baselines.moelora import enable_moelora_patch
enable_moelora_patch()
import torch.distributed.fsdp as fsdp
if not hasattr(fsdp,"FSDPModule"):
    class FSDPModule: pass
    fsdp.FSDPModule=FSDPModule
from swift.trainers import TrainerFactory
from med_prism.swift_plugins.moelora_trainer import MoELoRATrainer
_PREV=TrainerFactory.get_trainer_cls.__func__
def _get(cls,args): return MoELoRATrainer if os.environ.get("MOELORA_ENABLED")=="1" else _PREV(cls,args)
if not getattr(TrainerFactory.get_trainer_cls,"_moelora_conditional",False):
    TrainerFactory.get_trainer_cls=classmethod(_get); TrainerFactory.get_trainer_cls.__func__._moelora_conditional=True
