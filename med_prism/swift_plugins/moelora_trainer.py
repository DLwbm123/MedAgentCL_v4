"""ms-swift trainer audit/logging for dense CoIN MoELoRA."""
from __future__ import annotations
import os
from pathlib import Path
import torch.distributed as dist
import torch.distributed.fsdp as fsdp
if not hasattr(fsdp,"FSDPModule"):
    class FSDPModule: pass
    fsdp.FSDPModule=FSDPModule
from swift.trainers import Seq2SeqTrainer
from med_prism.baselines.moelora import OFFICIAL_COMMIT,OFFICIAL_REPOSITORY,moelora_runtime_stats
from scripts.medicalskill_v1_2_baselines.baseline_harness import atomic_json

def _main(): return not dist.is_available() or not dist.is_initialized() or dist.get_rank()==0
class MoELoRATrainer(Seq2SeqTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); self.task_id=int(os.environ["MOELORA_TASK_ID"]); self.audit_root=Path(os.environ["MOELORA_AUDIT_DIR"]); self._last=-1; self._forward_written=False
        stats=moelora_runtime_stats(self.accelerator.unwrap_model(self.model))
        router=[p for n,p in self.model.named_parameters() if "lora_router" in n]
        expert=[p for n,p in self.model.named_parameters() if ".lora_A." in n or ".lora_B." in n]
        if not router or not expert or any(not p.requires_grad for p in router+expert): raise RuntimeError("MoELoRA router/experts must be jointly trainable")
        if _main(): atomic_json(self.audit_root/"moelora_trainer_activation.json",{
            **stats,"official_repository":OFFICIAL_REPOSITORY,"official_commit":OFFICIAL_COMMIT,
            "task_id":self.task_id,"lifecycle":"one persistent adapter/router state continued across tasks",
            "auxiliary_loss":"none in official CoIN implementation"})
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        result=super().compute_loss(model,inputs,return_outputs=return_outputs,num_items_in_batch=num_items_in_batch)
        task,outputs=result if return_outputs else (result,None); stats=moelora_runtime_stats(self.accelerator.unwrap_model(model))
        if stats["min_forward_calls"]<1 or stats["max_probability_sum_error"]>1e-4: raise RuntimeError("MoELoRA routing forward/normalization failure")
        step=int(self.state.global_step)
        if step!=self._last and (step==0 or step%max(1,int(self.args.logging_steps))==0):
            self._last=step; log={"moelora/task_loss":float(task.detach().float()),"moelora/total_loss":float(task.detach().float()),"moelora/routing_entropy":stats["routing_entropy"]}
            for i,v in enumerate(stats["mean_expert_utilization"]): log[f"moelora/expert_{i}_utilization"]=v
            self.log(log)
        if not self._forward_written and _main(): atomic_json(self.audit_root/"moelora_forward_audit.json",stats); self._forward_written=True
        return (task,outputs) if return_outputs else task
