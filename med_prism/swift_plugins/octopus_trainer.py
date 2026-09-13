"""ms-swift trainer implementing the original Octopus Stage-2 losses."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.fsdp as fsdp

if not hasattr(fsdp, "FSDPModule"):
    class FSDPModule:
        pass

    fsdp.FSDPModule = FSDPModule

from swift.trainers import Seq2SeqTrainer

from med_prism.baselines.octopus import (
    attach_historical_stage2_extras,
    historical_extra_runtime_stats,
    load_merged_gradients,
    lora_pairs,
    octopus_regularizers,
)
from scripts.medicalskill_v1_2_baselines.baseline_harness import atomic_json


def _is_main() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


class OctopusStage2Trainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        self.task_id = int(os.environ["OCTOPUS_STAGE2_TASK_ID"])
        raw_history = json.loads(os.environ["OCTOPUS_HISTORICAL_STAGE2_ADAPTERS"])
        historical_paths = [Path(value).resolve() for value in raw_history]
        expected_history = self.task_id - 1
        if self.task_id < 1 or len(historical_paths) != expected_history:
            raise RuntimeError(
                f"Task {self.task_id} requires {expected_history} historical extras"
            )
        trainer_model = kwargs.get("model")
        if trainer_model is None:
            raise RuntimeError("Octopus trainer requires model as a keyword argument")
        if historical_paths:
            self.historical_extra_metadata = attach_historical_stage2_extras(
                trainer_model, historical_paths
            )
        else:
            self.historical_extra_metadata = {
                "semantics": "official_task1_no_historical_extras",
                "historical_adapter_count": 0,
                "lora_layer_count": 0,
                "frozen_extra_tensor_count": 0,
                "trainable_extra_tensor_count": 0,
                "historical_adapters": [],
            }
        super().__init__(*args, **kwargs)
        self.lambda_1 = float(os.environ.get("OCTOPUS_LAMBDA_1", "0.01"))
        self.lambda_2 = float(os.environ.get("OCTOPUS_LAMBDA_2", "0.01"))
        self.eps = float(os.environ.get("OCTOPUS_EPS", "1e-8"))
        raw_paths = json.loads(os.environ["OCTOPUS_GRADIENT_MANIFESTS"])
        paths = [Path(value).resolve() for value in raw_paths]
        if len(paths) != self.task_id - 1:
            raise RuntimeError(
                f"Task {self.task_id} requires {self.task_id - 1} gradient artifacts"
            )
        if paths:
            self.merged_gradients, gradient_metadata = load_merged_gradients(paths)
        else:
            self.merged_gradients = {}
            gradient_metadata = {
                "aggregation": "none_for_official_task1_stage2",
                "gradient_manifest_count": 0,
                "gradient_tensor_count": 0,
                "manifests": [],
            }
        raw = self.accelerator.unwrap_model(self.model)
        pairs = lora_pairs(raw)
        ranks = sorted({int(parameter_a.shape[0]) for _, parameter_a, _ in pairs})
        trainable_b = [
            key for key, _, parameter_b in pairs if parameter_b.requires_grad
        ]
        if ranks != [16]:
            raise RuntimeError(f"Octopus formal rank must be 16, found {ranks}")
        if not trainable_b:
            raise RuntimeError("No trainable LoRA B parameters in Octopus Stage-2")
        self._octopus_last_log = -1
        self._octopus_forward_audit_written = False
        self.audit_root = Path(os.environ["OCTOPUS_AUDIT_DIR"])
        audit = {
            "status": "PASS",
            "method": "octopus_qwen3vl_r16",
            "task_id": self.task_id,
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
            "eps": self.eps,
            "rank": 16,
            "lora_layer_count": len(pairs),
            "trainable_lora_b_count": len(trainable_b),
            "gradient_metadata": gradient_metadata,
            "historical_frozen_forward_extras": self.historical_extra_metadata,
            "checkpoint_serialization_expected": (
                "current trainable/default LoRA only; extras_A/extras_B excluded"
            ),
            "loss_definition": (
                "task_loss + lambda_1*sum(abs(trace(B_norm @ "
                "normalize(A @ G.T)))) + lambda_2*sum(norm(B))"
            ),
            "loss_logging_components": [
                "task_loss",
                "orthogonal_loss_raw",
                "orthogonal_loss_weighted",
                "sparsity_b_loss_raw",
                "sparsity_b_loss_weighted",
                "total_loss",
            ],
        }
        if _is_main():
            atomic_json(self.audit_root / "octopus_stage2_trainer_activation.json", audit)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        result = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        task_loss, outputs = result if return_outputs else (result, None)
        raw = self.accelerator.unwrap_model(model)
        if self.task_id > 1:
            forward_stats = historical_extra_runtime_stats(raw)
            if not forward_stats["all_historical_extras_active_in_forward"]:
                raise RuntimeError("Historical Stage-2 extras did not participate in forward")
            if forward_stats["trainable_extra_tensor_count"] != 0:
                raise RuntimeError("Historical Stage-2 extras received trainable parameters")
            if not self._octopus_forward_audit_written and _is_main():
                atomic_json(
                    self.audit_root / "octopus_stage2_forward_audit.json", forward_stats
                )
                self._octopus_forward_audit_written = True
            orthogonal, norm, metadata = octopus_regularizers(
                raw, self.merged_gradients, eps=self.eps
            )
        else:
            pairs = lora_pairs(raw)
            orthogonal = task_loss.new_zeros(())
            norm = sum(torch.norm(parameter_b) for _, _, parameter_b in pairs)
            metadata = {"gradient_layer_count": 0}
        weighted_orthogonal = self.lambda_1 * orthogonal
        weighted_norm = self.lambda_2 * norm
        total = task_loss + weighted_orthogonal + weighted_norm
        step = int(self.state.global_step)
        if step != self._octopus_last_log and (
            step == 0 or step % max(1, int(self.args.logging_steps)) == 0
        ):
            self._octopus_last_log = step
            self.log(
                {
                    "octopus/task_loss": float(task_loss.detach().float().cpu()),
                    "octopus/orthogonal_loss_raw": float(
                        orthogonal.detach().float().cpu()
                    ),
                    "octopus/orthogonal_loss_weighted": float(
                        weighted_orthogonal.detach().float().cpu()
                    ),
                    "octopus/sparsity_b_loss_raw": float(norm.detach().float().cpu()),
                    "octopus/sparsity_b_loss_weighted": float(
                        weighted_norm.detach().float().cpu()
                    ),
                    "octopus/total_loss": float(total.detach().float().cpu()),
                    # Backward-compatible Wave-0 field names.
                    "octopus/orthogonal_loss": float(
                        orthogonal.detach().float().cpu()
                    ),
                    "octopus/norm_loss": float(norm.detach().float().cpu()),
                    "octopus/weighted_orthogonal_loss": float(
                        weighted_orthogonal.detach().float().cpu()
                    ),
                    "octopus/weighted_norm_loss": float(
                        weighted_norm.detach().float().cpu()
                    ),
                    "octopus/gradient_layers": metadata["gradient_layer_count"],
                }
            )
        if return_outputs:
            return total, outputs
        return total
