from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist

from swift.trainers import Seq2SeqTrainer
from swift.trainers.utils import _add_gradient_checkpointing

from med_prism.config import SharedPrivateConfig
from med_prism.optimization.shared_private import (
    build_shared_private_optimizer_groups,
    optimizer_component_initial_lrs,
    optimizer_component_lrs,
)
from med_prism.projection.orth_losses import (
    compute_rank1_key_isolation_loss,
    compute_rank1_orth_loss,
)
from med_prism.projection.shared_private import compute_shared_drift


def _is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _write_json(path: Path, payload: dict) -> None:
    if not _is_main_process():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _update_json(path: Path, key: str, payload: dict) -> None:
    if not _is_main_process():
        return
    current = {}
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
    current[key] = payload
    current["status"] = (
        "PASS"
        if all(
            value.get("status") == "PASS"
            for name, value in current.items()
            if name.startswith("task_")
        )
        else "BLOCKED"
    )
    _write_json(path, current)


def _hash(named_parameters) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(named_parameters):
        digest.update(name.encode())
        digest.update(
            parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def _norm(named_parameters) -> float:
    total = 0.0
    for _, parameter in named_parameters:
        if parameter.grad is not None:
            value = float(parameter.grad.detach().float().norm().cpu())
            total += value * value
    return math.sqrt(total)


def _groups(model, task_id: int) -> dict[str, list[tuple[str, torch.nn.Parameter]]]:
    groups = {"shared": [], "current": [], "old": [], "base": [], "vision": []}
    markers = (f"task_{task_id:04d}__", f".task_adapters.task_{task_id:04d}.")
    for name, parameter in model.named_parameters():
        if ".shared." in name:
            groups["shared"].append((name, parameter))
        elif ".experts." in name or ".task_adapters." in name:
            groups[
                "current" if any(marker in name for marker in markers) else "old"
            ].append((name, parameter))
        elif any(
            token in name
            for token in (
                ".visual.",
                "vision_tower",
                "merger",
                "deepstack_merger",
                "aligner",
                "projector",
            )
        ):
            groups["vision"].append((name, parameter))
        else:
            groups["base"].append((name, parameter))
    return groups


class MedPrismSharedPrivateTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sp_config = SharedPrivateConfig.from_environment()
        self.sp_output = Path(self.sp_config.output_dir or self.args.output_dir)
        self.artifact_root = Path(self.sp_config.artifact_root)
        self.trace_path = self.artifact_root / (
            f"task{self.sp_config.current_task_id}_training_trace.jsonl"
        )
        if _is_main_process():
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.trace_path.write_text("", encoding="utf-8")
        raw = self.accelerator.unwrap_model(self.model)
        groups = _groups(raw, self.sp_config.current_task_id)
        self.hash_before = {key: _hash(value) for key, value in groups.items()}
        self.last_loss = {}
        self.orth_gradient_norms = []
        self.shared_drift_gradient_norms = []
        self.key_gradient_norms = []
        self.gc_calls = 0
        self.gc_wrapped = False
        self.optimizer_audit = {}
        self.optimizer_group_definition = None
        self._last_trace_step = -1
        self.shared_source_hash = self._manifest_weight_hash(
            self.sp_config.shared_source_manifest
        )
        self.historical_private_hashes = [
            self._manifest_weight_hash(path)
            for path in self.sp_config.private_source_manifests
        ]

    def _prepare_gradient_checkpointing(self, model):
        super()._prepare_gradient_checkpointing(model)
        matches = [
            module
            for module in model.modules()
            if module.__class__.__name__ == "Qwen3VLTextModel"
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one Qwen3VLTextModel, got {len(matches)}")
        language = matches[0]
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
            self.gc_wrapped = True
        for layer in layers:
            original = layer._gradient_checkpointing_func

            def counted(*call_args, _original=original, **call_kwargs):
                self.gc_calls += 1
                return _original(*call_args, **call_kwargs)

            layer._gradient_checkpointing_func = counted

    @staticmethod
    def _manifest_weight_hash(path: str | None) -> str | None:
        if not path:
            return None
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload.get("weights_sha256")

    def create_optimizer(self, model=None):
        if self.sp_config.shared_optimizer_mode == "legacy_gradient_hook":
            return super().create_optimizer(model=model)
        if self.optimizer is not None:
            return self.optimizer
        opt_model = model or self.model
        decay_names = set(self.get_decay_parameter_names(opt_model))
        grouped, audit = build_shared_private_optimizer_groups(
            opt_model,
            current_task_id=self.sp_config.current_task_id,
            decay_parameter_names=decay_names,
            weight_decay=self.args.weight_decay,
            shared_lr=self.sp_config.shared_lr,
            private_lr=self.sp_config.private_lr,
        )
        if self.optimizer_cls_and_kwargs is not None:
            optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
        else:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, opt_model
            )
        optimizer_kwargs = dict(optimizer_kwargs)
        forbidden = {"params", "model", "optimizer_dict"} & set(optimizer_kwargs)
        if forbidden:
            raise RuntimeError(
                "Shared-Fix supports the locked AdamW optimizer only; "
                f"unexpected optimizer kwargs: {sorted(forbidden)}"
            )
        self.optimizer = optimizer_cls(grouped, **optimizer_kwargs)
        self._optimizer_ori = self.optimizer
        self.optimizer_group_definition = audit
        return self.optimizer

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.sp_config.shared_optimizer_mode == "legacy_gradient_hook":
            super().create_optimizer_and_scheduler(num_training_steps)
        else:
            self.create_optimizer()
            self.create_scheduler(num_training_steps, optimizer=self.optimizer)
        self._capture_optimizer_audit()

    def _capture_optimizer_audit(self) -> None:
        if self.optimizer is None:
            raise RuntimeError("Optimizer audit ran before optimizer creation")
        raw = self.accelerator.unwrap_model(self.model)
        groups = _groups(raw, self.sp_config.current_task_id)
        named = dict(raw.named_parameters())
        id_to_name = {id(parameter): name for name, parameter in named.items()}
        optimizer_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        expected = {
            id(parameter)
            for key in ("shared", "current")
            for _, parameter in groups[key]
        }
        optimizer_names = sorted(id_to_name[item] for item in optimizer_ids)
        missing = sorted(
            name
            for key in ("shared", "current")
            for name, parameter in groups[key]
            if id(parameter) not in optimizer_ids
        )
        old_intersection = sorted(
            name for name, parameter in groups["old"] if id(parameter) in optimizer_ids
        )
        forbidden_intersection = sorted(
            name
            for key in ("base", "vision")
            for name, parameter in groups[key]
            if id(parameter) in optimizer_ids
        )
        frozen_intersection = sorted(
            name
            for name, parameter in named.items()
            if not parameter.requires_grad and id(parameter) in optimizer_ids
        )
        unexpected = sorted(
            id_to_name[item] for item in optimizer_ids if item not in expected
        )
        actual_shared_lr, actual_private_lr = optimizer_component_lrs(self.optimizer)
        shared_lr, private_lr = optimizer_component_initial_lrs(self.optimizer)
        if self.sp_config.shared_optimizer_mode == "legacy_gradient_hook":
            actual_shared_lr = actual_private_lr = float(
                self.optimizer.param_groups[0]["lr"]
            )
            shared_lr = private_lr = float(
                self.optimizer.param_groups[0].get(
                    "initial_lr", self.optimizer.param_groups[0]["lr"]
                )
            )
        lr_ratio = shared_lr / private_lr if shared_lr and private_lr else None
        actual_lr_ratio = (
            actual_shared_lr / actual_private_lr
            if actual_shared_lr is not None and actual_private_lr
            else None
        )
        separate_groups = (
            self.sp_config.shared_optimizer_mode == "separate_lr_param_groups"
        )
        configured_lrs_match = (
            shared_lr is not None
            and private_lr is not None
            and math.isclose(
                shared_lr, self.sp_config.shared_lr, rel_tol=1e-9, abs_tol=0.0
            )
            and math.isclose(
                private_lr, self.sp_config.private_lr, rel_tol=1e-9, abs_tol=0.0
            )
            and math.isclose(lr_ratio, 0.1, rel_tol=1e-9, abs_tol=1e-12)
        )
        scheduled_lrs_compatible = (
            actual_shared_lr == 0.0 and actual_private_lr == 0.0
        ) or (
            actual_shared_lr is not None
            and actual_private_lr is not None
            and actual_private_lr > 0.0
            and math.isclose(
                actual_shared_lr / actual_private_lr,
                self.sp_config.shared_lr / self.sp_config.private_lr,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        )
        optimizer_groups = [
            {
                "component": group.get("med_prism_group", "legacy"),
                "lr": float(group["lr"]),
                "initial_lr": float(group.get("initial_lr", group["lr"])),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "parameter_count": len(group["params"]),
            }
            for group in self.optimizer.param_groups
        ]
        failures = {
            "empty_expected_parameter_set": not expected,
            "missing_expected_parameters": bool(missing),
            "historical_private_in_optimizer": bool(old_intersection),
            "base_or_vision_in_optimizer": bool(forbidden_intersection),
            "frozen_parameter_in_optimizer": bool(frozen_intersection),
            "unexpected_optimizer_parameter": bool(unexpected),
            "configured_lr_contract": separate_groups and not configured_lrs_match,
            "scheduled_lr_ratio": separate_groups and not scheduled_lrs_compatible,
        }
        payload = {
            "status": "BLOCKED" if any(failures.values()) else "PASS",
            "captured_after_optimizer_creation": True,
            "optimizer_parameter_object_ids": sorted(optimizer_ids),
            "optimizer_parameter_names": optimizer_names,
            "shared_parameter_names": [name for name, _ in groups["shared"]],
            "current_private_parameter_names": [name for name, _ in groups["current"]],
            "old_private_parameter_names": [name for name, _ in groups["old"]],
            "base_parameter_names": [name for name, _ in groups["base"]],
            "vision_merger_parameter_names": [name for name, _ in groups["vision"]],
            "missing_expected_parameters": missing,
            "old_private_optimizer_intersection": old_intersection,
            "base_vision_optimizer_intersection": forbidden_intersection,
            "frozen_optimizer_intersection": frozen_intersection,
            "unexpected_optimizer_parameters": unexpected,
            "shared_optimizer_mode": self.sp_config.shared_optimizer_mode,
            "shared_gradient_hook_enabled": self.sp_config.shared_gradient_hook_enabled,
            "optimizer_groups": optimizer_groups,
            "shared_lr": shared_lr,
            "private_lr": private_lr,
            "shared_private_lr_ratio": lr_ratio,
            "actual_shared_lr": actual_shared_lr,
            "actual_private_lr": actual_private_lr,
            "actual_shared_private_lr_ratio": actual_lr_ratio,
            "scheduled_lrs_compatible": scheduled_lrs_compatible,
            "failure_reasons": [name for name, failed in failures.items() if failed],
        }
        self.optimizer_audit = payload
        audit_path = self.artifact_root / (
            f"task{self.sp_config.current_task_id}_optimizer_audit.json"
        )
        _write_json(audit_path, payload)
        if payload["status"] != "PASS":
            raise RuntimeError(
                "Shared/private optimizer isolation failed: "
                f"reasons={payload['failure_reasons']}; audit={audit_path}"
            )

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
        task_loss, outputs = result if return_outputs else (result, None)
        upcoming_step = int(self.state.global_step) + 1
        geometry_interval = self.sp_config.geometry_every_optimizer_steps
        include_factor_diagnostic = (
            self.sp_config.method_version == "1.0"
            or upcoming_step == 1
            or (geometry_interval > 0 and upcoming_step % geometry_interval == 0)
        )
        orth = compute_rank1_orth_loss(
            model,
            current_task_id=self.sp_config.current_task_id,
            loss_type=self.sp_config.orth_loss_type,
            eps=self.sp_config.orth_eps,
        )
        orth_loss = (
            orth.loss
            if self.sp_config.private_adapter_type == "rank1_expert_bank"
            else task_loss * 0.0
        )
        orth_weighted = orth_loss * self.sp_config.orth_lambda
        drift = compute_shared_drift(
            model,
            mode=self.sp_config.shared_drift_mode,
            current_task_id=self.sp_config.current_task_id,
            eps=self.sp_config.orth_eps,
            include_factor_diagnostic=include_factor_diagnostic,
        )
        drift_weighted = drift.loss * self.sp_config.shared_drift_lambda
        if self.sp_config.key_isolation_enabled:
            key = compute_rank1_key_isolation_loss(
                model,
                current_task_id=self.sp_config.current_task_id,
                eps=self.sp_config.orth_eps,
            )
            key_loss = key.loss
        else:
            key = None
            key_loss = task_loss * 0.0
        key_weighted = key_loss * self.sp_config.key_lambda
        total = task_loss + orth_weighted + drift_weighted + key_weighted

        raw = self.accelerator.unwrap_model(model)
        parameter_groups = _groups(raw, self.sp_config.current_task_id)
        current = parameter_groups["current"]
        shared = parameter_groups["shared"]
        diagnostic_interval = self.sp_config.component_gradient_diagnostics_interval
        diagnostic_now = self.sp_config.method_version == "1.0" or (
            diagnostic_interval > 0
            and upcoming_step % diagnostic_interval == 0
        )
        orth_gradient = shared_drift_gradient = key_gradient = 0.0
        if diagnostic_now and orth_weighted.requires_grad and current:
            gradients = torch.autograd.grad(
                orth_weighted,
                [parameter for _, parameter in current],
                retain_graph=True,
                allow_unused=True,
            )
            orth_gradient = math.sqrt(
                sum(
                    float(gradient.detach().float().norm().cpu()) ** 2
                    for gradient in gradients
                    if gradient is not None
                )
            )
        if diagnostic_now and drift_weighted.requires_grad and shared:
            gradients = torch.autograd.grad(
                drift_weighted,
                [parameter for _, parameter in shared],
                retain_graph=True,
                allow_unused=True,
            )
            shared_drift_gradient = math.sqrt(
                sum(
                    float(gradient.detach().float().norm().cpu()) ** 2
                    for gradient in gradients
                    if gradient is not None
                )
            )
        if (
            diagnostic_now
            and self.sp_config.key_isolation_enabled
            and key_weighted.requires_grad
            and current
        ):
            gradients = torch.autograd.grad(
                key_weighted,
                [parameter for _, parameter in current],
                retain_graph=True,
                allow_unused=True,
            )
            key_gradient = math.sqrt(
                sum(
                    float(gradient.detach().float().norm().cpu()) ** 2
                    for gradient in gradients
                    if gradient is not None
                )
            )
        self.orth_gradient_norms.append(orth_gradient)
        self.shared_drift_gradient_norms.append(shared_drift_gradient)
        self.key_gradient_norms.append(key_gradient)
        factor_value = (
            None
            if drift.factor_loss is None
            else float(drift.factor_loss.detach().float().cpu())
        )
        self.last_loss = {
            "method_version": self.sp_config.method_version,
            "ablation_name": self.sp_config.ablation_name,
            "private_adapter_type": self.sp_config.private_adapter_type,
            "task_loss": float(task_loss.detach().float().cpu()),
            "raw_geo_orth_loss": float(orth_loss.detach().float().cpu()),
            "weighted_geo_orth_loss": float(orth_weighted.detach().float().cpu()),
            "geo_pair_count": orth.pair_count,
            "geo_mean_abs_overlap": orth.mean_abs_overlap,
            "geo_rms_overlap": orth.rms_overlap,
            "geo_max_overlap": orth.max_abs_overlap,
            "raw_effective_shared_drift": (
                float(drift.loss.detach().float().cpu())
                if drift.mode == "effective_BA"
                else 0.0
            ),
            "weighted_effective_shared_drift": (
                float(drift_weighted.detach().float().cpu())
                if drift.mode == "effective_BA"
                else 0.0
            ),
            "diagnostic_factor_shared_drift": factor_value,
            "effective_shared_reference_norm": drift.effective_reference_norm,
            "effective_shared_delta_norm": drift.effective_delta_norm,
            "effective_shared_relative_shift": drift.effective_relative_shift,
            "shared_drift_mode": drift.mode,
            "raw_key_loss": float(key_loss.detach().float().cpu()),
            "weighted_key_loss": float(key_weighted.detach().float().cpu()),
            "key_isolation_enabled": self.sp_config.key_isolation_enabled,
            "key_pair_count": 0 if key is None else key.pair_count,
            "key_mean_abs_cosine": 0.0 if key is None else key.mean_abs_cosine,
            "key_rms_cosine": 0.0 if key is None else key.rms_cosine,
            "key_max_abs_cosine": 0.0 if key is None else key.max_abs_cosine,
            "total_loss": float(total.detach().float().cpu()),
            "shared_drift_gradient_norm": shared_drift_gradient,
            "orthogonal_gradient_norm": orth_gradient,
            "key_gradient_norm": key_gradient,
            "component_gradient_diagnostic": diagnostic_now,
            "old_private_task_count": orth.old_task_count,
            "orthogonal_pair_count": orth.pair_count,
            # Legacy aliases keep existing v1.0 audits and readers compatible.
            "orthogonal_raw_loss": float(orth_loss.detach().float().cpu()),
            "weighted_orthogonal_loss": float(orth_weighted.detach().float().cpu()),
            "shared_drift_raw_loss": float(drift.loss.detach().float().cpu()),
            "weighted_shared_drift_loss": float(drift_weighted.detach().float().cpu()),
        }
        return (total, outputs) if return_outputs else total
    def training_step(self, model, inputs, *args, **kwargs):
        loss = super().training_step(model, inputs, *args, **kwargs)
        raw = self.accelerator.unwrap_model(model)
        groups = _groups(raw, self.sp_config.current_task_id)
        record = dict(self.last_loss)
        weighted = (
            record["weighted_geo_orth_loss"]
            + record["weighted_shared_drift_loss"]
            + record["weighted_key_loss"]
        )
        if self.sp_config.shared_optimizer_mode == "separate_lr_param_groups":
            shared_lr, private_lr = optimizer_component_lrs(self.optimizer)
        else:
            private_lr = float(self.optimizer.param_groups[0]["lr"])
            shared_lr = private_lr
        step = int(self.state.global_step) + 1
        record.update(
            {
                "step": step,
                "loss_identity_error": abs(
                    record["total_loss"] - record["task_loss"] - weighted
                ),
                "full_loss_identity_error": abs(
                    record["total_loss"] - record["task_loss"] - weighted
                ),
                "requested_task_plus_orth_identity_error": abs(
                    record["total_loss"]
                    - record["task_loss"]
                    - record["weighted_geo_orth_loss"]
                ),
                "shared_gradient_norm": _norm(groups["shared"]),
                "current_private_gradient_norm": _norm(groups["current"]),
                "current_private_A_gradient_norm": _norm(
                    [(name, p) for name, p in groups["current"] if name.endswith(".A")]
                ),
                "current_private_B_gradient_norm": _norm(
                    [(name, p) for name, p in groups["current"] if name.endswith(".B")]
                ),
                "old_private_gradient_norm": _norm(groups["old"]),
                "learning_rate": private_lr,
                "shared_lr": shared_lr,
                "private_lr": private_lr,
                "shared_private_lr_ratio": (
                    shared_lr / private_lr if shared_lr is not None and private_lr else None
                ),
                "current_task": self.sp_config.current_task_id,
                "active_historical_private_task_ids": list(
                    range(1, self.sp_config.current_task_id)
                ),
                "shared_checkpoint_hash": self.shared_source_hash,
                "historical_private_hashes": self.historical_private_hashes,
                "gpu_allocated": torch.cuda.memory_allocated(),
                "gpu_reserved": torch.cuda.memory_reserved(),
                "activation_checkpoint_calls": self.gc_calls,
            }
        )
        interval = self.sp_config.trace_every_optimizer_steps
        should_trace = self.sp_config.method_version == "1.0" or (
            self.accelerator.sync_gradients
            and (step == 1 or (interval > 0 and step % interval == 0))
            and step != self._last_trace_step
        )
        if should_trace and _is_main_process():
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            self._last_trace_step = step
        return loss
    def train(self, *args, **kwargs):
        result = super().train(*args, **kwargs)
        raw = self.accelerator.unwrap_model(self.model)
        groups = _groups(raw, self.sp_config.current_task_id)
        after = {key: _hash(value) for key, value in groups.items()}
        task = f"task_{self.sp_config.current_task_id}"
        parameter_audit = {
            "status": "PASS",
            "task_id": self.sp_config.current_task_id,
            "trainable_parameter_names": [
                name
                for name, parameter in raw.named_parameters()
                if parameter.requires_grad
            ],
            "shared_changed": self.hash_before["shared"] != after["shared"],
            "current_private_changed": self.hash_before["current"] != after["current"],
            "old_private_unchanged": self.hash_before["old"] == after["old"],
            "old_private_max_gradient_norm": 0.0,
            "activation_checkpoint_calls": self.gc_calls,
        }
        traces = [
            json.loads(line)
            for line in self.trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        parameter_audit["old_private_max_gradient_norm"] = max(
            (item["old_private_gradient_norm"] for item in traces), default=0.0
        )
        if (
            not parameter_audit["shared_changed"]
            or not parameter_audit["current_private_changed"]
            or not parameter_audit["old_private_unchanged"]
            or parameter_audit["old_private_max_gradient_norm"] != 0.0
        ):
            parameter_audit["status"] = "BLOCKED"
        _write_json(
            self.artifact_root
            / f"task{self.sp_config.current_task_id}_parameter_audit.json",
            parameter_audit,
        )
        shared_payload = {
            "status": "PASS"
            if self.hash_before["shared"] != after["shared"]
            else "BLOCKED",
            "before": self.hash_before["shared"],
            "after": after["shared"],
            "source_manifest": self.sp_config.shared_source_manifest,
        }
        private_payload = {
            "status": parameter_audit["status"],
            "current_before": self.hash_before["current"],
            "current_after": after["current"],
            "old_before": self.hash_before["old"],
            "old_after": after["old"],
            "old_unchanged": self.hash_before["old"] == after["old"],
        }
        diagnostics_required = (
            self.sp_config.method_version == "1.0"
            or self.sp_config.component_gradient_diagnostics_interval > 0
        )
        orth_payload = {
            "status": "PASS",
            "enabled": self.sp_config.orth_lambda > 0,
            "private_adapter_type": self.sp_config.private_adapter_type,
            "expert_level_geometry_defined": (
                self.sp_config.private_adapter_type == "rank1_expert_bank"
            ),
            "lambda_geo_orth": self.sp_config.orth_lambda,
            "max_orthogonal_gradient_norm": max(self.orth_gradient_norms, default=0.0),
            "orthogonal_gradient_nonzero": any(
                value > 0 for value in self.orth_gradient_norms
            ),
            "gradient_diagnostic_enabled": diagnostics_required,
            "old_private_max_gradient_norm": parameter_audit[
                "old_private_max_gradient_norm"
            ],
        }
        if (
            diagnostics_required
            and self.sp_config.current_task_id > 1
            and self.sp_config.orth_lambda > 0
            and self.sp_config.private_adapter_type == "rank1_expert_bank"
            and not orth_payload["orthogonal_gradient_nonzero"]
        ):
            orth_payload["status"] = "BLOCKED"
        drift_nonzero = any(item["shared_drift_raw_loss"] > 0 for item in traces)
        drift_gradient_nonzero = any(
            value > 0 for value in self.shared_drift_gradient_norms
        )
        task1_effective_zero = (
            self.sp_config.current_task_id == 1
            and self.sp_config.shared_drift_mode == "effective_BA"
        )
        drift_payload = {
            "status": "PASS",
            "shared_drift_mode": self.sp_config.shared_drift_mode,
            "shared_drift_lambda": self.sp_config.shared_drift_lambda,
            "reference_semantics": (
                "no historical shared reference; exact differentiable zero"
                if task1_effective_zero
                else (
                    "shared initialization state"
                    if self.sp_config.current_task_id == 1
                    else "detached previous-stage final shared loaded from shared_source_manifest"
                )
            ),
            "reference_detached": True,
            "task1_effective_zero": task1_effective_zero,
            "raw_loss_nonzero": drift_nonzero,
            "weighted_loss_nonzero": any(
                item["weighted_shared_drift_loss"] > 0 for item in traces
            ),
            "max_shared_drift_gradient_norm": max(
                self.shared_drift_gradient_norms, default=0.0
            ),
            "shared_drift_gradient_nonzero": drift_gradient_nonzero,
            "gradient_diagnostic_enabled": diagnostics_required,
        }
        if task1_effective_zero and drift_nonzero:
            drift_payload["status"] = "BLOCKED"
        if (
            self.sp_config.current_task_id > 1
            and self.sp_config.shared_drift_lambda > 0
            and len(traces) > 1
            and not drift_nonzero
        ):
            drift_payload["status"] = "BLOCKED"
        if (
            diagnostics_required
            and self.sp_config.current_task_id > 1
            and self.sp_config.shared_drift_lambda > 0
            and len(traces) > 1
            and not drift_gradient_nonzero
        ):
            drift_payload["status"] = "BLOCKED"
        key_nonzero = any(item.get("raw_key_loss", 0.0) > 0 for item in traces)
        key_gradient_nonzero = any(value > 0 for value in self.key_gradient_norms)
        key_payload = {
            "status": "PASS",
            "enabled": self.sp_config.key_isolation_enabled,
            "lambda_key": self.sp_config.key_lambda,
            "history_scope": self.sp_config.key_history_scope,
            "raw_loss_nonzero": key_nonzero,
            "key_gradient_nonzero": key_gradient_nonzero,
            "max_key_gradient_norm": max(self.key_gradient_norms, default=0.0),
            "gradient_diagnostic_enabled": diagnostics_required,
        }
        if (
            self.sp_config.key_isolation_enabled
            and self.sp_config.current_task_id > 1
            and not key_nonzero
        ):
            key_payload["status"] = "BLOCKED"
        if (
            diagnostics_required
            and self.sp_config.key_isolation_enabled
            and self.sp_config.current_task_id > 1
            and not key_gradient_nonzero
        ):
            key_payload["status"] = "BLOCKED"
        hash_payload = {
            "status": (
                "PASS"
                if shared_payload["status"] == "PASS"
                and private_payload["status"] == "PASS"
                else "BLOCKED"
            ),
            "shared_before": self.hash_before["shared"],
            "shared_after": after["shared"],
            "current_private_before": self.hash_before["current"],
            "current_private_after": after["current"],
            "old_private_before": self.hash_before["old"],
            "old_private_after": after["old"],
            "old_private_unchanged": self.hash_before["old"] == after["old"],
            "shared_source_manifest": self.sp_config.shared_source_manifest,
        }
        _update_json(self.artifact_root / "shared_hash_audit.json", task, shared_payload)
        _update_json(self.artifact_root / "private_hash_audit.json", task, private_payload)
        _update_json(self.artifact_root / "orth_gradient_audit.json", task, orth_payload)
        _update_json(self.artifact_root / "key_isolation_audit.json", task, key_payload)
        _update_json(
            self.artifact_root / "shared_drift_gradient_audit.json",
            task,
            drift_payload,
        )
        _update_json(
            self.artifact_root / "shared_drift_hash_audit.json",
            task,
            hash_payload,
        )
        return result
