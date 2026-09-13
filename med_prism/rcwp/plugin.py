"""Opt-in trainer subclass: existing no-geo compute_loss remains authoritative."""
import json
import os
import torch
from swift.trainers import TrainerFactory
from med_prism.swift_plugins.shared_private_trainer import MedPrismSharedPrivateTrainer
from med_prism.rcwp.runtime import WriteProtection, load_summary, wrappers


class RCWPTrainer(MedPrismSharedPrivateTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        c = self.sp_config
        if (c.orth_lambda != 0 or c.key_lambda != .1 or not c.key_isolation_enabled or
            c.shared_drift_lambda != .01 or c.private_rank != 16):
            raise ValueError("v2.1 requires the unchanged no-geo A-only baseline")
        settings = json.loads(os.environ["MED_PRISM_RCWP_CONFIG_JSON"])
        if settings["method_version"] != "2.1" or settings.get("tpm_enabled",False):
            raise ValueError("RCWP v2.1 is mutually exclusive with TPM")
        self.rcwp_weight = float(settings.get("rcwp_lambda",.1))
        if self.rcwp_weight != .1:
            raise ValueError("Initial v2.1 fixes lambda_rcwp=0.1")
        raw = self.accelerator.unwrap_model(self.model)
        summaries = [load_summary(p,wrappers(raw)) for p in settings["summaries"]]
        self.rcwp = WriteProtection(raw,c.current_task_id,summaries)

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        mask = inputs.get("attention_mask")
        if mask is None:
            ids = inputs.get("input_ids")
            if ids is None or ids.shape[0] != 1:
                raise ValueError("No token mask for RCWP")
            mask = torch.ones_like(ids)
        self.rcwp.begin(mask)
        try:
            result = super().compute_loss(model,inputs,return_outputs=return_outputs,
                                         num_items_in_batch=num_items_in_batch)
            auxiliary,diag = self.rcwp.finish()
        finally:
            self.rcwp.cancel()
        baseline, outputs = result if return_outputs else (result,None)
        total = baseline + self.rcwp_weight*auxiliary
        self.last_loss.update(diag,method_version="2.1",method_name="Med-PRISM-v2.1-RCWP",
            A_orth_loss=self.last_loss["raw_key_loss"],
            shared_drift_loss=self.last_loss["shared_drift_raw_loss"],total_loss=float(total.detach()))
        return (total,outputs) if return_outputs else total


_previous = TrainerFactory.get_trainer_cls.__func__


def get_trainer(cls,args):
    if os.environ.get("MED_PRISM_RCWP_CONFIG_JSON") and getattr(args,"tuner_type",None)=="med_prism_shared_private":
        return RCWPTrainer
    return _previous(cls,args)


TrainerFactory.get_trainer_cls = classmethod(get_trainer)
