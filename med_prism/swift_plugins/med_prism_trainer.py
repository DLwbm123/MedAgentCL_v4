from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from swift.trainers import Seq2SeqTrainer
from swift.trainers.utils import _add_gradient_checkpointing

from med_prism.adapters.injection import iter_rank1_wrappers
from med_prism.config import Rank1BankConfig
from med_prism.projection import compute_rank1_orth_loss


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _parameter_hash(parameters) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(parameters):
        digest.update(name.encode())
        tensor = parameter.detach().cpu().contiguous().view(torch.uint8)
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _expert_parameters(model, current_task_id: int):
    current = []
    old = []
    marker = f"task_{current_task_id:04d}__"
    for name, parameter in model.named_parameters():
        if ".experts." not in name:
            continue
        (current if marker in name else old).append((name, parameter))
    return current, old


def _gradient_norm(named_parameters) -> float:
    total = 0.0
    for _, parameter in named_parameters:
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().float().norm().cpu())
        total += value * value
    return math.sqrt(total)


class MedPrismRank1Trainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.med_prism_config = Rank1BankConfig.from_environment()
        self.med_prism_output = Path(
            self.med_prism_config.output_dir or self.args.output_dir
        )
        self.med_prism_trace = self.med_prism_output / "med_prism_loss_trace.jsonl"
        self.med_prism_trace.parent.mkdir(parents=True, exist_ok=True)
        self.med_prism_trace.write_text("", encoding="utf-8")
        raw_model = self.accelerator.unwrap_model(self.model)
        current, old = _expert_parameters(
            raw_model, self.med_prism_config.current_task_id
        )
        self._current_hash_before = _parameter_hash(current)
        self._old_hash_before = _parameter_hash(old)
        self._last_loss_record = {}
        self._orth_gradient_norms = []
        self._gc_activation_calls = 0
        self._gc_project_wrap = False
        self._optimizer_audit = {}

    def _prepare_gradient_checkpointing(self, model):
        super()._prepare_gradient_checkpointing(model)
        language_matches = [
            module
            for module in model.modules()
            if module.__class__.__name__ == "Qwen3VLTextModel"
        ]
        if len(language_matches) != 1:
            raise RuntimeError(
                f"Expected one Qwen3VLTextModel, got {len(language_matches)}"
            )
        language = language_matches[0]
        layers = list(language.layers)
        wrapped = [hasattr(layer, "__old_forward") for layer in layers]
        if not all(wrapped):
            if any(wrapped):
                raise RuntimeError("Partially wrapped language checkpoint layers")
            checkpoint_func = language._gradient_checkpointing_func
            _add_gradient_checkpointing(language.layers)
            for layer in layers:
                layer.gradient_checkpointing = True
                layer._gradient_checkpointing_func = checkpoint_func
            self._gc_project_wrap = True
        for layer in layers:
            original = layer._gradient_checkpointing_func

            def counted(*call_args, _original=original, **call_kwargs):
                self._gc_activation_calls += 1
                return _original(*call_args, **call_kwargs)

            layer._gradient_checkpointing_func = counted
        self._write_trainer_activation_audit()

    def _write_trainer_activation_audit(self):
        raw_model = self.accelerator.unwrap_model(self.model)
        wrappers = list(iter_rank1_wrappers(raw_model))
        current, old = _expert_parameters(
            raw_model, self.med_prism_config.current_task_id
        )
        payload = {
            "status": "PASS" if len(wrappers) == 72 else "BLOCKED",
            "trainer_class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "is_med_prism_trainer": True,
            "tuner_type": self.args.tuner_type,
            "wrapper_class": sorted({type(wrapper).__name__ for _, wrapper in wrappers}),
            "wrapper_count": len(wrappers),
            "task_ids": sorted({
                task_id for _, wrapper in wrappers for task_id in wrapper.task_ids
            }),
            "current_task_id": self.med_prism_config.current_task_id,
            "current_parameter_names": [name for name, _ in current],
            "old_parameter_names": [name for name, _ in old],
            "optimizer_parameter_ids": self._optimizer_audit,
            "gradient_checkpointing": {
                "project_side_dynamic_wrap_applied": self._gc_project_wrap,
                "activation_checkpoint_calls": self._gc_activation_calls,
            },
        }
        _write_json(self.med_prism_output / "trainer_activation_audit.json", payload)

    def _capture_optimizer_audit(self):
        optimizer = self.optimizer
        if optimizer is None:
            raise RuntimeError("Optimizer audit ran before optimizer creation")
        raw_model = self.accelerator.unwrap_model(self.model)
        named = list(raw_model.named_parameters())
        id_to_name = {id(parameter): name for name, parameter in named}
        current, old = _expert_parameters(
            raw_model, self.med_prism_config.current_task_id
        )
        optimizer_ids = sorted({
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        })
        optimizer_id_set = set(optimizer_ids)
        optimizer_names = sorted(
            id_to_name.get(parameter_id, f"<unknown:{parameter_id}>")
            for parameter_id in optimizer_ids
        )
        old_ids = {id(parameter) for _, parameter in old}
        current_ids = {id(parameter) for _, parameter in current}
        vision_merger = [
            (name, parameter)
            for name, parameter in named
            if any(
                token in name
                for token in (
                    ".visual.",
                    "vision_tower",
                    "merger",
                    "deepstack_merger",
                    "aligner",
                    "projector",
                )
            )
        ]
        vision_merger_ids = {id(parameter) for _, parameter in vision_merger}
        expert_ids = current_ids | old_ids
        base = [
            (name, parameter)
            for name, parameter in named
            if id(parameter) not in expert_ids | vision_merger_ids
        ]
        base_ids = {id(parameter) for _, parameter in base}
        frozen = [
            (name, parameter)
            for name, parameter in named
            if not parameter.requires_grad
        ]
        missing_current = sorted(
            name for name, parameter in current
            if id(parameter) not in optimizer_id_set
        )
        old_intersection = sorted(
            name for name, parameter in old
            if id(parameter) in optimizer_id_set
        )
        frozen_intersection = sorted(
            name for name, parameter in frozen
            if id(parameter) in optimizer_id_set
        )
        unexpected = sorted(
            set(optimizer_names) - {name for name, _ in current}
        )
        self._optimizer_audit = {
            "status": "PASS",
            "captured_after_optimizer_creation": True,
            "optimizer_parameter_object_ids": optimizer_ids,
            "optimizer_parameter_names": optimizer_names,
            "current_task_expert_parameter_names": [name for name, _ in current],
            "old_expert_parameter_names": [name for name, _ in old],
            "base_parameter_names": [name for name, _ in base],
            "vision_merger_parameter_names": [name for name, _ in vision_merger],
            "missing_current_parameters": missing_current,
            "old_optimizer_intersection": old_intersection,
            "frozen_optimizer_intersection": frozen_intersection,
            "unexpected_optimizer_parameters": unexpected,
            "optimizer_parameter_count": len(optimizer_ids),
            "current_parameter_count": len(current_ids),
            "old_parameter_count": len(old_ids),
            "base_parameter_count": len(base_ids),
            "vision_merger_parameter_count": len(vision_merger_ids),
            "all_current_in_optimizer": current_ids <= optimizer_id_set,
            "old_in_optimizer_count": len(old_ids & optimizer_id_set),
            "base_in_optimizer_count": len(base_ids & optimizer_id_set),
            "vision_merger_in_optimizer_count": len(
                vision_merger_ids & optimizer_id_set
            ),
        }
        if (
            not current_ids
            or missing_current
            or old_intersection
            or frozen_intersection
            or unexpected
        ):
            self._optimizer_audit["status"] = "BLOCKED"
            raise RuntimeError(
                f"Optimizer expert isolation failed: {self._optimizer_audit}"
            )
        artifact = Path(self.med_prism_config.artifact_root) / (
            f"phase4_rank1_task{self.med_prism_config.current_task_id}_optimizer_audit.json"
        )
        _write_json(artifact, self._optimizer_audit)
        self._write_trainer_activation_audit()

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)
        self._capture_optimizer_audit()

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        result = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        if return_outputs:
            task_loss, outputs = result
        else:
            task_loss, outputs = result, None
        orth = compute_rank1_orth_loss(
            model,
            current_task_id=self.med_prism_config.current_task_id,
            loss_type=self.med_prism_config.orth_loss_type,
            eps=self.med_prism_config.orth_eps,
        )
        weighted = orth.loss * self.med_prism_config.orth_lambda
        total = task_loss + weighted
        raw_model = self.accelerator.unwrap_model(model)
        current, _ = _expert_parameters(
            raw_model, self.med_prism_config.current_task_id
        )
        orth_gradient_norm = 0.0
        if weighted.requires_grad and weighted.grad_fn is not None:
            gradients = torch.autograd.grad(
                weighted,
                [parameter for _, parameter in current],
                retain_graph=True,
                allow_unused=True,
            )
            orth_gradient_norm = math.sqrt(sum(
                float(gradient.detach().float().norm().cpu()) ** 2
                for gradient in gradients
                if gradient is not None
            ))
        self._last_loss_record = {
            "task_loss": float(task_loss.detach().float().cpu()),
            "orth_loss_raw": float(orth.loss.detach().float().cpu()),
            "orth_loss_weighted": float(weighted.detach().float().cpu()),
            "total_loss": float(total.detach().float().cpu()),
            "orth_loss_type": self.med_prism_config.orth_loss_type,
            "eps": self.med_prism_config.orth_eps,
            "lambda": self.med_prism_config.orth_lambda,
            "old_task_count": orth.old_task_count,
            "current_task_id": self.med_prism_config.current_task_id,
            "layer_count": orth.layer_count,
            "old_new_pair_count": orth.pair_count,
            "orth_gradient_norm": orth_gradient_norm,
        }
        self._orth_gradient_norms.append(orth_gradient_norm)
        if return_outputs:
            return total, outputs
        return total

    def training_step(self, model, inputs, *args, **kwargs):
        if not self._optimizer_audit:
            self._capture_optimizer_audit()
        loss = super().training_step(model, inputs, *args, **kwargs)
        raw_model = self.accelerator.unwrap_model(model)
        current, old = _expert_parameters(
            raw_model, self.med_prism_config.current_task_id
        )
        record = dict(self._last_loss_record)
        record.update({
            "step": int(self.state.global_step) + 1,
            "current_expert_grad_norm": _gradient_norm(current),
            "old_expert_grad_norm": _gradient_norm(old),
            "total_loss_identity_error": abs(
                record["total_loss"]
                - record["task_loss"]
                - record["orth_loss_weighted"]
            ),
            "gpu_allocated": torch.cuda.memory_allocated(),
            "gpu_reserved": torch.cuda.memory_reserved(),
        })
        with self.med_prism_trace.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        return loss

    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        raw_model = self.accelerator.unwrap_model(self.model)
        current, old = _expert_parameters(
            raw_model, self.med_prism_config.current_task_id
        )
        payload = {
            "status": "PASS",
            "current_task_id": self.med_prism_config.current_task_id,
            "current_hash_before": self._current_hash_before,
            "current_hash_after": _parameter_hash(current),
            "current_parameters_changed": (
                self._current_hash_before != _parameter_hash(current)
            ),
            "old_hash_before": self._old_hash_before,
            "old_hash_after": _parameter_hash(old),
            "old_parameters_unchanged": self._old_hash_before == _parameter_hash(old),
            "max_orth_gradient_norm": max(self._orth_gradient_norms, default=0.0),
            "orth_gradient_nonzero": any(
                value > 0 for value in self._orth_gradient_norms
            ),
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "activation_checkpoint_calls": self._gc_activation_calls,
        }
        if (
            not payload["current_parameters_changed"]
            or not payload["old_parameters_unchanged"]
            or (
                self.med_prism_config.current_task_id > 1
                and not payload["orth_gradient_nonzero"]
            )
        ):
            payload["status"] = "BLOCKED"
        _write_json(self.med_prism_output / "orth_gradient_audit.json", payload)
        self._write_trainer_activation_audit()
        return result
