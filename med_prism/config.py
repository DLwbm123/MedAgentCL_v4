from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


@dataclass(frozen=True)
class Rank1BankConfig:
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    current_task_id: int = 1
    experts_per_task: int = 4
    alpha: float = 4.0
    dropout: float = 0.05
    orth_loss_type: str = "rms"
    orth_lambda: float = 0.1
    orth_eps: float = 1e-8
    seed: int = 42
    source_manifest: str | None = None
    dataset_manifest: str | None = None
    dataset_sha256: str | None = None
    output_dir: str | None = None
    artifact_root: str = (
        "/root/MedAgentCL_v4/artifacts/phase4_closure"
    )

    @property
    def scaling(self) -> float:
        return self.alpha / self.experts_per_task

    def validate(self) -> None:
        if self.model_id != MODEL_ID:
            raise ValueError(f"Unexpected backbone: {self.model_id}")
        if self.model_revision != MODEL_REVISION:
            raise ValueError(f"Unexpected immutable revision: {self.model_revision}")
        if self.current_task_id <= 0:
            raise ValueError("current_task_id must be positive")
        if self.experts_per_task <= 0:
            raise ValueError("experts_per_task must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.orth_loss_type not in {"squared", "rms"}:
            raise ValueError("orth_loss_type must be squared or rms")
        if self.orth_lambda < 0:
            raise ValueError("orth_lambda cannot be negative")
        if self.orth_eps <= 0:
            raise ValueError("orth_eps must be positive")
        if not self.artifact_root:
            raise ValueError("artifact_root is required")
        if self.current_task_id > 1 and not self.source_manifest:
            raise ValueError("Tasks after Task1 require an explicit source_manifest")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Rank1BankConfig":
        config = cls(**payload)
        config.validate()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "Rank1BankConfig":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing explicit Med-PRISM config: {path}")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_environment(cls) -> "Rank1BankConfig":
        path = os.environ.get("MED_PRISM_CONFIG")
        if not path:
            raise RuntimeError("MED_PRISM_CONFIG must point to an explicit JSON config")
        return cls.from_json(path)


@dataclass(frozen=True)
class SharedPrivateConfig:
    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    current_task_id: int = 1
    shared_rank: int = 32
    shared_alpha: float = 32.0
    shared_lr_scale: float = 0.1
    private_experts_per_task: int = 16
    private_rank: int | None = None
    private_adapter_type: str = "rank1_expert_bank"
    private_alpha: float = 16.0
    dropout: float = 0.05
    orth_loss_type: str = "rms"
    orth_lambda: float = 0.1
    orth_eps: float = 1e-8
    shared_drift_lambda: float = 0.01
    method: str = "Med-PRISM"
    method_version: str = "1.0"
    method_variant: str = "legacy"
    ablation_name: str | None = None
    shared_optimizer_mode: str = "legacy_gradient_hook"
    shared_lr: float = 1e-4
    private_lr: float = 1e-4
    shared_gradient_hook_enabled: bool = True
    shared_drift_mode: str = "factor"
    key_isolation_enabled: bool = False
    key_loss_mode: str = "disabled"
    key_lambda: float = 0.0
    key_history_scope: str = "none"
    trace_every_optimizer_steps: int = 0
    geometry_every_optimizer_steps: int = 0
    component_gradient_diagnostics_interval: int = 0
    seed: int = 42
    shared_source_manifest: str | None = None
    private_source_manifests: tuple[str, ...] = ()
    shared_output_dir: str | None = None
    private_output_dir: str | None = None
    dataset_manifest: str | None = None
    dataset_sha256: str | None = None
    output_dir: str | None = None
    artifact_root: str = (
        "/root/MedAgentCL_v4/artifacts/phase5_shared_private"
    )

    @property
    def shared_scaling(self) -> float:
        return self.shared_alpha / self.shared_rank

    @property
    def private_scaling(self) -> float:
        return self.private_alpha / self.effective_private_rank

    @property
    def effective_private_rank(self) -> int:
        return (
            self.private_experts_per_task
            if self.private_rank is None
            else self.private_rank
        )

    def validate(self) -> None:
        if self.model_id != MODEL_ID or self.model_revision != MODEL_REVISION:
            raise ValueError("Shared/private backbone identity mismatch")
        if self.current_task_id <= 0:
            raise ValueError("current_task_id must be positive")
        if (
            self.shared_rank <= 0
            or self.private_experts_per_task <= 0
            or self.effective_private_rank <= 0
        ):
            raise ValueError("Shared and private ranks must be positive")
        if self.private_adapter_type not in {
            "rank1_expert_bank",
            "standard_rank16_lora",
        }:
            raise ValueError("Unknown private_adapter_type")
        if self.private_adapter_type == "rank1_expert_bank":
            if self.effective_private_rank != self.private_experts_per_task:
                raise ValueError(
                    "Rank-1 banks require private_rank == private_experts_per_task"
                )
        elif self.effective_private_rank != 16 or self.private_experts_per_task != 1:
            raise ValueError(
                "The structural ablation requires exactly one rank-16 private LoRA"
            )
        if self.shared_alpha <= 0 or self.private_alpha <= 0:
            raise ValueError("Shared and private alpha must be positive")
        if self.shared_lr_scale <= 0:
            raise ValueError("shared_lr_scale must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.orth_loss_type not in {"squared", "rms"}:
            raise ValueError("orth_loss_type must be squared or rms")
        if self.orth_lambda < 0 or self.shared_drift_lambda < 0 or self.key_lambda < 0:
            raise ValueError("Loss weights cannot be negative")
        if self.orth_eps <= 0:
            raise ValueError("orth_eps must be positive")
        if not self.shared_output_dir or not self.private_output_dir:
            raise ValueError(
                "Independent shared/private output directories are required"
            )
        if not self.artifact_root:
            raise ValueError("artifact_root is required")
        if self.method != "Med-PRISM":
            raise ValueError("Unexpected shared/private method name")
        if self.method_version not in {"1.0", "1.1", "1.2"}:
            raise ValueError("method_version must be 1.0, 1.1, or 1.2")
        if self.shared_optimizer_mode not in {
            "legacy_gradient_hook",
            "separate_lr_param_groups",
        }:
            raise ValueError("Unknown shared optimizer mode")
        if self.shared_lr <= 0 or self.private_lr <= 0:
            raise ValueError("Shared/private learning rates must be positive")
        if self.shared_drift_mode not in {"factor", "effective_BA"}:
            raise ValueError("shared_drift_mode must be factor or effective_BA")
        if self.key_loss_mode not in {"disabled", "rms_cosine_A"}:
            raise ValueError("Unknown key loss mode")
        if min(
            self.trace_every_optimizer_steps,
            self.geometry_every_optimizer_steps,
            self.component_gradient_diagnostics_interval,
        ) < 0:
            raise ValueError("Logging intervals cannot be negative")
        if self.method_version == "1.0":
            if (
                self.shared_optimizer_mode != "legacy_gradient_hook"
                or not self.shared_gradient_hook_enabled
                or self.shared_drift_mode != "factor"
                or self.key_isolation_enabled
            ):
                raise ValueError("Med-PRISM v1.0 must preserve legacy semantics")
        else:
            if (
                self.shared_optimizer_mode != "separate_lr_param_groups"
                or self.shared_gradient_hook_enabled
                or self.shared_drift_mode != "effective_BA"
            ):
                raise ValueError("Med-PRISM v1.1/v1.2 require Shared-Fix semantics")
            if abs(self.shared_lr / self.private_lr - 0.1) > 1e-12:
                raise ValueError("Shared-Fix requires a 0.1 shared/private LR ratio")
        if self.method_version == "1.1" and self.key_isolation_enabled:
            raise ValueError("Med-PRISM v1.1 cannot enable key isolation")
        if self.method_version == "1.2":
            rank16_ablation = self.ablation_name == "rank16_private"
            if rank16_ablation:
                if (
                    self.private_adapter_type != "standard_rank16_lora"
                    or self.key_isolation_enabled
                    or self.key_loss_mode != "disabled"
                    or self.key_lambda != 0
                    or self.orth_lambda != 0
                ):
                    raise ValueError(
                        "rank16_private must disable expert-level key/geometry losses"
                    )
            elif (
                not self.key_isolation_enabled
                or self.key_loss_mode != "rms_cosine_A"
                or self.key_history_scope != "all_previous_tasks_same_layer"
            ):
                raise ValueError("Med-PRISM v1.2 requires rank-1 key isolation")
        elif self.key_loss_mode != "disabled" or self.key_lambda != 0:
            raise ValueError("Key loss must be disabled before Med-PRISM v1.2")
        if self.current_task_id == 1:
            if self.shared_source_manifest or self.private_source_manifests:
                raise ValueError("Task 1 cannot load previous components")
        else:
            if not self.shared_source_manifest:
                raise ValueError("Task 2+ requires shared_source_manifest")
            if len(self.private_source_manifests) != self.current_task_id - 1:
                raise ValueError("Task 2+ requires one private manifest per old task")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["private_source_manifests"] = list(self.private_source_manifests)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SharedPrivateConfig":
        payload = dict(payload)
        payload["private_source_manifests"] = tuple(
            payload.get("private_source_manifests", ())
        )
        config = cls(**payload)
        config.validate()
        return config

    @classmethod
    def from_environment(cls) -> "SharedPrivateConfig":
        path_value = os.environ.get("MED_PRISM_SHARED_PRIVATE_CONFIG")
        if not path_value:
            raise RuntimeError(
                "MED_PRISM_SHARED_PRIVATE_CONFIG must point to an explicit JSON config"
            )
        path = Path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"Missing shared/private config: {path}")
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
