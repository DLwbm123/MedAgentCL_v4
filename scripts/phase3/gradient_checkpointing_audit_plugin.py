from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from swift.callbacks import callbacks_map


OUTPUT = Path(os.environ.get(
    "PHASE3_OUTPUT_DIR",
    "/root/MedAgentCL_v4/output/phase3_native_lora",
))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


class Phase3GradientCheckpointingAudit(TrainerCallback):
    def __init__(self, *callback_args, **callback_kwargs) -> None:
        self.activation_calls = 0
        self.audit: dict[str, Any] = {}

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            raise RuntimeError("Gradient checkpointing audit did not receive model")

        language_modules = [
            (name, module)
            for name, module in model.named_modules()
            if module.__class__.__name__ == "Qwen3VLTextModel"
        ]
        vision_modules = [
            (name, module)
            for name, module in model.named_modules()
            if module.__class__.__name__ == "Qwen3VLVisionModel"
        ]
        if len(language_modules) != 1 or len(vision_modules) != 1:
            raise RuntimeError(
                "Could not identify unique Qwen3-VL towers: "
                f"language={len(language_modules)}, vision={len(vision_modules)}"
            )

        language_name, language = language_modules[0]
        vision_name, vision = vision_modules[0]
        layers = list(language.layers)
        project_side_dynamic_wrap_applied = False
        initially_wrapped = [
            hasattr(layer, "__old_forward") for layer in layers
        ]
        if not all(initially_wrapped):
            if any(initially_wrapped):
                raise RuntimeError("Language layers are only partially checkpoint-wrapped")
            from swift.trainers.utils import _add_gradient_checkpointing

            checkpoint_func = language._gradient_checkpointing_func
            _add_gradient_checkpointing(language.layers)
            for layer in layers:
                layer.gradient_checkpointing = True
                layer._gradient_checkpointing_func = checkpoint_func
            project_side_dynamic_wrap_applied = True
        dynamic_wrapped = [
            hasattr(layer, "__old_forward") for layer in layers
        ]
        layer_flags = [
            bool(getattr(layer, "gradient_checkpointing", False))
            for layer in layers
        ]
        language_trainable = sum(
            parameter.requires_grad for parameter in language.parameters()
        )
        vision_trainable = sum(
            parameter.requires_grad for parameter in vision.parameters()
        )
        if (
            len(layers) != 36
            or not all(dynamic_wrapped)
            or not all(layer_flags)
            or not language_trainable
            or vision_trainable
        ):
            raise RuntimeError(
                "Language gradient checkpointing audit failed: "
                f"layers={len(layers)}, wrapped={sum(dynamic_wrapped)}, "
                f"enabled={sum(layer_flags)}, language_trainable={language_trainable}, "
                f"vision_trainable={vision_trainable}"
            )

        for layer in layers:
            original = layer._gradient_checkpointing_func

            def counted_checkpoint(
                *call_args,
                _original=original,
                **call_kwargs,
            ):
                self.activation_calls += 1
                return _original(*call_args, **call_kwargs)

            layer._gradient_checkpointing_func = counted_checkpoint

        self.audit = {
            "status": "RUNNING",
            "project_side_dynamic_wrap_applied": project_side_dynamic_wrap_applied,
            "warning_call_path": [
                "swift.trainers.mixin.SwiftMixin.train",
                "SwiftMixin._prepare_gradient_checkpointing",
                "vision_tower.gradient_checkpointing_disable",
                "vision_tower.disable_input_require_grads",
            ],
            "warning_scope": "frozen_vision_tower_input_grad_hook_cleanup",
            "warning_is_caught": True,
            "language_module": language_name,
            "language_class": language.__class__.__name__,
            "language_model_flag": bool(
                getattr(language, "gradient_checkpointing", False)
            ),
            "language_layer_count": len(layers),
            "language_layer_flags_enabled": sum(layer_flags),
            "dynamic_wrapped_layer_count": sum(dynamic_wrapped),
            "language_trainable_parameter_tensors": language_trainable,
            "vision_module": vision_name,
            "vision_class": vision.__class__.__name__,
            "vision_gradient_checkpointing": bool(
                getattr(vision, "gradient_checkpointing", False)
            ),
            "vision_trainable_parameter_tensors": vision_trainable,
            "activation_checkpoint_calls": 0,
        }
        write_json(OUTPUT / "gradient_checkpointing_audit.json", self.audit)

    def on_train_end(self, args, state, control, **kwargs):
        self.audit.update({
            "status": (
                "PASS"
                if self.activation_calls > 0 and int(state.global_step) == 10
                else "BLOCKED"
            ),
            "activation_checkpoint_calls": self.activation_calls,
            "completed_global_steps": int(state.global_step),
        })
        write_json(OUTPUT / "gradient_checkpointing_audit.json", self.audit)


callbacks_map["phase3_gc_audit"] = Phase3GradientCheckpointingAudit
