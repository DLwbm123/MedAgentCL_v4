"""ms-swift trainer for the RegLoRA component of SEFE (no ASD)."""
from __future__ import annotations
import json, os
from pathlib import Path
import torch
import torch.distributed as dist
import torch.distributed.fsdp as fsdp
if not hasattr(fsdp,"FSDPModule"):
    class FSDPModule: pass
    fsdp.FSDPModule=FSDPModule
from swift.trainers import Seq2SeqTrainer
from med_prism.baselines.octopus import attach_historical_stage2_extras, historical_extra_runtime_stats, lora_pairs
from med_prism.baselines.reglora import DEFAULT_INDEX_CHUNK_SIZE, OFFICIAL_COMMIT, OFFICIAL_REPOSITORY, reglora_regularizer, inspect_importance_mask, load_importance_mask
from scripts.medicalskill_v1_2_baselines.baseline_harness import atomic_json, load_json, sha256_file

def _main(): return not dist.is_available() or not dist.is_initialized() or dist.get_rank()==0
class RegLoRATrainer(Seq2SeqTrainer):
    def __init__(self,*args,**kwargs):
        self.task_id=int(os.environ["REGLORA_TASK_ID"])
        history=[Path(x).resolve() for x in json.loads(os.environ.get("REGLORA_HISTORY","[]"))]
        if len(history)!=self.task_id-1: raise RuntimeError("RegLoRA reconstruction history mismatch")
        mask=os.environ.get("REGLORA_MASK"); self.mask_path=Path(mask).resolve() if mask else None
        if (self.task_id==1)!=(self.mask_path is None): raise RuntimeError("RegLoRA Task1 alone has no prior mask")
        model=kwargs.get("model")
        if model is None: raise RuntimeError("RegLoRA trainer requires keyword model")
        self.history_meta=(attach_historical_stage2_extras(model,history) if history else {"historical_adapter_count":0})
        super().__init__(*args,**kwargs)
        self.coefficient=float(os.environ.get("REGLORA_LAMBDA","2500"))
        self.audit_root=Path(os.environ["REGLORA_AUDIT_DIR"]); self._last=-1; self._forward_written=False
        pairs=lora_pairs(self.accelerator.unwrap_model(self.model))
        self.mask_value = None
        self.mask_meta = None
        if self.mask_path:
            meta_path = Path(os.environ["REGLORA_MASK_META"]).resolve()
            self.mask_meta = load_json(meta_path)
            if self.mask_meta.get("status") != "PASS" or self.mask_meta.get("mask_sha256") != sha256_file(self.mask_path):
                raise RuntimeError("RegLoRA mask metadata/hash mismatch")
            self.mask_value = load_importance_mask(self.mask_path)
            self.mask_value["masks"] = {name: value.to(dtype=torch.int32)
                                        for name, value in self.mask_value["masks"].items()}
            # Keep the cumulative int32 mask on CPU. The regularizer transfers
            # bounded chunks and recomputes in backward; Task-5 masks otherwise
            # consume about 0.9 GiB per rank before the first training step.
            self.mask_value["_runtime_meta"] = {**self.mask_meta,
                "runtime_index_device": "cpu",
                "runtime_index_dtype": "int32",
                "index_chunk_size": int(os.environ.get("REGLORA_INDEX_CHUNK_SIZE", DEFAULT_INDEX_CHUNK_SIZE))}
        if sorted({int(a.shape[0]) for _,a,_ in pairs}) != [16]: raise RuntimeError("RegLoRA rank must be 16")
        if _main(): atomic_json(self.audit_root/"reglora_trainer_activation.json",{
            "status":"PASS","task_id":self.task_id,"official_repository":OFFICIAL_REPOSITORY,
            "official_commit":OFFICIAL_COMMIT,"component":"RegLoRA only; ASD absent","rank":16,
            "coefficient":self.coefficient,"mask":self.mask_meta,
            "runtime_index_device":"cpu",
            "runtime_index_dtype":"int32",
            "index_chunk_size":int(os.environ.get("REGLORA_INDEX_CHUNK_SIZE",DEFAULT_INDEX_CHUNK_SIZE)),
            "historical_reconstruction":self.history_meta,
            "framework_adaptation":"frozen adapter reconstruction is algebraically base+sum(previous BA), matching official merged-base forward without serializing full 8B merged checkpoints"})
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        result=super().compute_loss(model,inputs,return_outputs=return_outputs,num_items_in_batch=num_items_in_batch)
        task,outputs=result if return_outputs else (result,None)
        raw=self.accelerator.unwrap_model(model); reg,meta=reglora_regularizer(raw,self.mask_value)
        if self.task_id>1:
            stats=historical_extra_runtime_stats(raw)
            if not stats["all_historical_extras_active_in_forward"] or stats["trainable_extra_tensor_count"]: raise RuntimeError("RegLoRA reconstruction failure")
            if not self._forward_written and _main(): atomic_json(self.audit_root/"reglora_forward_audit.json",stats); self._forward_written=True
        weighted=self.coefficient*reg; total=task+weighted; step=int(self.state.global_step)
        if step!=self._last and (step==0 or step%max(1,int(self.args.logging_steps))==0):
            self._last=step; self.log({"reglora/task_loss":float(task.detach().float()),"reglora/raw_reglora_loss":float(reg.detach().float()),"reglora/weighted_reglora_loss":float(weighted.detach().float()),"reglora/protected_position_count":meta["protected_position_count"],"reglora/protected_position_fraction":meta["protected_position_fraction"],"reglora/total_loss":float(total.detach().float())})
        return (total,outputs) if return_outputs else total
