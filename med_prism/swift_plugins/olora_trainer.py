"""ms-swift trainer for official O-LoRA loss."""
from __future__ import annotations
import json, os
from pathlib import Path
import torch
import torch.distributed as dist
import torch.distributed.fsdp as fsdp
if not hasattr(fsdp, "FSDPModule"):
    class FSDPModule: pass
    fsdp.FSDPModule = FSDPModule
from swift.trainers import Seq2SeqTrainer
from med_prism.baselines.octopus import attach_historical_stage2_extras, historical_extra_runtime_stats, lora_pairs
from med_prism.baselines.olora import OFFICIAL_COMMIT, OFFICIAL_REPOSITORY, olora_regularizers
from scripts.medicalskill_v1_2_baselines.baseline_harness import atomic_json

def _main(): return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

class OLoRATrainer(Seq2SeqTrainer):
    def __init__(self,*args,**kwargs):
        self.task_id=int(os.environ["OLORA_TASK_ID"])
        history=[Path(x).resolve() for x in json.loads(os.environ.get("OLORA_HISTORY","[]"))]
        if len(history)!=self.task_id-1: raise RuntimeError("O-LoRA history count mismatch")
        model=kwargs.get("model")
        if model is None: raise RuntimeError("O-LoRA trainer requires keyword model")
        self.history_meta=(attach_historical_stage2_extras(model,history) if history else
            {"historical_adapter_count":0,"frozen_extra_tensor_count":0})
        super().__init__(*args,**kwargs)
        self.lambda_1=float(os.environ.get("OLORA_LAMBDA_1","0.5"))
        self.lambda_2=float(os.environ.get("OLORA_LAMBDA_2","0"))
        self.audit_root=Path(os.environ["OLORA_AUDIT_DIR"])
        pairs=lora_pairs(self.accelerator.unwrap_model(self.model))
        if sorted({int(a.shape[0]) for _,a,_ in pairs}) != [16]: raise RuntimeError("O-LoRA rank must be 16")
        if any(not a.requires_grad or not b.requires_grad for _,a,b in pairs): raise RuntimeError("Current O-LoRA must be trainable")
        self._last=-1; self._forward_written=False
        if _main(): atomic_json(self.audit_root/"olora_trainer_activation.json",{
            "status":"PASS","task_id":self.task_id,"official_repository":OFFICIAL_REPOSITORY,
            "official_commit":OFFICIAL_COMMIT,"rank":16,"lambda_1":self.lambda_1,
            "lambda_2":self.lambda_2,"historical":self.history_meta,
            "loss":"task + lambda_1*sum(abs(A_old@A_current.T)) + lambda_2*sum(L2(current A/B))",
            "checkpoint":"fresh current-task adapter only"})
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        result=super().compute_loss(model,inputs,return_outputs=return_outputs,num_items_in_batch=num_items_in_batch)
        task,outputs=result if return_outputs else (result,None)
        raw=self.accelerator.unwrap_model(model)
        orth,secondary,meta=olora_regularizers(raw)
        if self.task_id>1:
            stats=historical_extra_runtime_stats(raw)
            if not stats["all_historical_extras_active_in_forward"] or stats["trainable_extra_tensor_count"]: raise RuntimeError("O-LoRA historical freeze/forward failure")
            if not self._forward_written and _main(): atomic_json(self.audit_root/"olora_forward_audit.json",stats); self._forward_written=True
            if meta["historical_a_block_count"]==0: raise RuntimeError("O-LoRA penalty has no historical A")
        w_orth=self.lambda_1*orth; w_secondary=self.lambda_2*secondary; total=task+w_orth+w_secondary
        step=int(self.state.global_step)
        if step!=self._last and (step==0 or step%max(1,int(self.args.logging_steps))==0):
            self._last=step; self.log({"olora/task_loss":float(task.detach().float()),"olora/raw_orthogonal_loss":float(orth.detach().float()),"olora/weighted_orthogonal_loss":float(w_orth.detach().float()),"olora/raw_secondary_l2":float(secondary.detach().float()),"olora/weighted_secondary_l2":float(w_secondary.detach().float()),"olora/total_loss":float(total.detach().float())})
        return (total,outputs) if return_outputs else total
