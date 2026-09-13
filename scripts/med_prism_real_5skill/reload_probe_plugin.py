"""Opt-in in-process logits probe for the existing shared/private trainer."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from swift.trainers import TrainerFactory

from med_prism.swift_plugins.shared_private_trainer import MedPrismSharedPrivateTrainer


def _is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


class MedPrismReloadProbeTrainer(MedPrismSharedPrivateTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reload_probe_inputs = None

    @staticmethod
    def _clone_probe_tensor(value: torch.Tensor) -> torch.Tensor:
        # Formal training uses per-device batch size 1. Keep every image feature
        # row belonging to that sample; slicing dim 0 corrupts multi-image inputs.
        return value.detach().cpu().contiguous().clone()

    def compute_loss(self, model, inputs, *args, **kwargs):
        result = super().compute_loss(model, inputs, *args, **kwargs)
        if os.environ.get("MED_PRISM_RELOAD_PROBE") == "1" and self._reload_probe_inputs is None:
            # Seq2SeqTrainer mutates `inputs` to remove trainer-only fields before
            # forwarding. Capture that post-cleanup mapping so fresh reload calls
            # the model with exactly the tensor fields accepted during training.
            self._reload_probe_inputs = {
                key: self._clone_probe_tensor(value)
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }
        return result

    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        probe_enabled = os.environ.get("MED_PRISM_RELOAD_PROBE") == "1" and bool(self._reload_probe_inputs)
        if not probe_enabled:
            return result
        if not _is_main_process():
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            return result
        raw = self.accelerator.unwrap_model(self.model)
        was_training = raw.training
        raw.eval()
        inputs = self._prepare_inputs(self._reload_probe_inputs)
        with torch.inference_mode():
            logits = raw(**inputs).logits[:, -1, :].detach().float().cpu().contiguous()
        if was_training:
            raw.train()
        probe_dir = Path(self.sp_config.artifact_root)
        inputs_path = probe_dir / f"task{self.sp_config.current_task_id}_reload_probe_inputs.pt"
        logits_path = probe_dir / f"task{self.sp_config.current_task_id}_pre_reload_logits.safetensors"
        torch.save(self._reload_probe_inputs, inputs_path)
        save_file({"last_token_logits": logits}, str(logits_path))
        digest = hashlib.sha256(logits.numpy().tobytes()).hexdigest()
        (probe_dir / f"task{self.sp_config.current_task_id}_pre_reload_probe.json").write_text(
            json.dumps({"status": "PASS" if torch.isfinite(logits).all() else "FAIL", "task_id": self.sp_config.current_task_id, "inputs": str(inputs_path), "input_tensor_schema": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in self._reload_probe_inputs.items()}, "logits": str(logits_path), "shape": list(logits.shape), "float32_sha256": digest, "finite": bool(torch.isfinite(logits).all())}, indent=2) + "\n",
            encoding="utf-8",
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return result


_PREVIOUS = TrainerFactory.get_trainer_cls.__func__


def _get_trainer_cls(cls, args):
    if getattr(args, "tuner_type", None) == "med_prism_shared_private":
        return MedPrismReloadProbeTrainer
    return _PREVIOUS(cls, args)


TrainerFactory.get_trainer_cls = classmethod(_get_trainer_cls)
