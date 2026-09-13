from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

from swift.callbacks import callbacks_map


OUTPUT = Path(os.environ.get("MEDICALSKILL_V1_SMOKE_OUTPUT", "/root/MedAgentCL_v4/output/phase3_native_lora"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def classify(names: list[str]) -> dict[str, int]:
    return {
        "total": len(names),
        "language": sum("model.language_model." in name for name in names),
        "q_proj": sum(".q_proj" in name for name in names),
        "v_proj": sum(".v_proj" in name for name in names),
        "vision": sum(".visual." in name or "vision_tower" in name for name in names),
        "merger_aligner": sum(
            any(token in name for token in ("merger", "deepstack_merger", "aligner", "projector")) for name in names
        ),
    }


class MedicalSkillV1AuditCallback(TrainerCallback):
    def __init__(self, *callback_args, **callback_kwargs) -> None:
        self.initial_lora: dict[str, torch.Tensor] = {}
        self.gradient_records: dict[int, dict[str, Any]] = {}
        self.trace_path = OUTPUT / "training_trace.jsonl"
        self.image_stats = {}
        stats_path = OUTPUT / "data" / "image_token_grid_stats.json"
        if stats_path.is_file():
            self.image_stats = json.loads(stats_path.read_text(encoding="utf-8"))

    @staticmethod
    def wrappers(model) -> list[str]:
        return sorted(
            name for name, module in model.named_modules()
            if hasattr(module, "lora_A") and hasattr(module, "lora_B")
        )

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            raise RuntimeError("Phase3 callback did not receive the training model")
        wrappers = self.wrappers(model)
        classes = classify(wrappers)
        expected = {"total": 72, "language": 72, "q_proj": 36, "v_proj": 36, "vision": 0, "merger_aligner": 0}
        if classes != expected:
            raise RuntimeError(f"Post-injection LoRA wrapper audit failed: {classes}")
        (OUTPUT / "target_modules_post_injection.txt").write_text("\n".join(wrappers) + "\n", encoding="utf-8")
        target_path = OUTPUT / "target_module_audit.json"
        target = json.loads(target_path.read_text(encoding="utf-8"))
        target.update({"status": "PASS", "post_injection": classes, "post_injection_full_names": wrappers})
        write_json(target_path, target)

        trainable_names = []
        total_count = trainable_count = vision_count = merger_count = lora_count = 0
        unexpected = []
        for name, parameter in model.named_parameters():
            total_count += parameter.numel()
            if not parameter.requires_grad:
                continue
            trainable_names.append(name)
            trainable_count += parameter.numel()
            if ".visual." in name or "vision_tower" in name:
                vision_count += parameter.numel()
            if any(token in name for token in ("merger", "deepstack_merger", "aligner", "projector")):
                merger_count += parameter.numel()
            if ".lora_A." in name or ".lora_B." in name:
                lora_count += parameter.numel()
                self.initial_lora[name] = parameter.detach().float().cpu().clone()
            else:
                unexpected.append(name)
        if not trainable_count or trainable_count != lora_count or vision_count or merger_count or unexpected:
            raise RuntimeError(
                f"Trainable parameter audit failed: trainable={trainable_count}, lora={lora_count}, "
                f"vision={vision_count}, merger={merger_count}, unexpected={unexpected[:5]}"
            )
        (OUTPUT / "trainable_parameters.txt").write_text("\n".join(trainable_names) + "\n", encoding="utf-8")
        parameter_audit = {
            "status": "PASS",
            "total_parameter_count": total_count,
            "trainable_parameter_count": trainable_count,
            "trainable_ratio": trainable_count / total_count,
            "lora_trainable_parameter_count": lora_count,
            "vision_trainable_parameter_count": vision_count,
            "merger_trainable_parameter_count": merger_count,
            "unexpected_trainable_parameters": unexpected,
            "modules_to_save": [],
            "target_wrapper_count": len(wrappers),
            "trainable_parameter_names": trainable_names,
        }
        write_json(OUTPUT / "parameter_audit.json", parameter_audit)

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        step = int(state.global_step) + 1
        q_nonzero = v_nonzero = vision_nonzero = merger_nonzero = 0
        q_norm_sq = v_norm_sq = 0.0
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            finite = bool(torch.isfinite(gradient).all())
            norm = float(gradient.detach().float().norm().cpu())
            if not finite:
                raise RuntimeError(f"Non-finite gradient at step {step}: {name}")
            if ".q_proj." in name and ".lora_" in name and norm > 0:
                q_nonzero += 1
                q_norm_sq += norm * norm
            if ".v_proj." in name and ".lora_" in name and norm > 0:
                v_nonzero += 1
                v_norm_sq += norm * norm
            if (".visual." in name or "vision_tower" in name) and norm > 0:
                vision_nonzero += 1
            if any(token in name for token in ("merger", "deepstack_merger", "aligner", "projector")) and norm > 0:
                merger_nonzero += 1
        if not q_nonzero or not v_nonzero or vision_nonzero or merger_nonzero:
            raise RuntimeError(
                f"Gradient audit failed step={step}: q={q_nonzero}, v={v_nonzero}, "
                f"vision={vision_nonzero}, merger={merger_nonzero}"
            )
        self.gradient_records[step] = {
            "step": step,
            "q_nonzero_gradient_parameters": q_nonzero,
            "v_nonzero_gradient_parameters": v_nonzero,
            "q_gradient_norm": math.sqrt(q_norm_sq),
            "v_gradient_norm": math.sqrt(v_norm_sq),
            "vision_nonzero_gradient_parameters": vision_nonzero,
            "merger_nonzero_gradient_parameters": merger_nonzero,
            "gpu_allocated": torch.cuda.memory_allocated(),
            "gpu_reserved": torch.cuda.memory_reserved(),
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" not in logs:
            return
        step = int(state.global_step)
        row = dict(self.gradient_records.get(step, {"step": step}))
        row.update({
            "loss": float(logs["loss"]),
            "learning_rate": float(logs.get("learning_rate", 0.0)),
            "grad_norm": float(logs.get("grad_norm", 0.0)),
            "image_token_grid_stats": self.image_stats.get("summary", {}),
        })
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    def on_train_end(self, args, state, control, model=None, **kwargs):
        q_changed = v_changed = 0
        max_change = 0.0
        for name, parameter in model.named_parameters():
            if name not in self.initial_lora:
                continue
            change = float((parameter.detach().float().cpu() - self.initial_lora[name]).abs().max())
            max_change = max(max_change, change)
            if change > 0 and ".q_proj." in name:
                q_changed += 1
            if change > 0 and ".v_proj." in name:
                v_changed += 1
        records = list(self.gradient_records.values())
        summary = {
            "status": "PASS" if len(records) == 2 and q_changed and v_changed else "BLOCKED",
            "steps_with_gradients": len(records),
            "q_steps_nonzero": sum(row["q_nonzero_gradient_parameters"] > 0 for row in records),
            "v_steps_nonzero": sum(row["v_nonzero_gradient_parameters"] > 0 for row in records),
            "vision_or_merger_gradient_steps": sum(
                bool(row["vision_nonzero_gradient_parameters"] or row["merger_nonzero_gradient_parameters"])
                for row in records
            ),
            "q_changed_parameters": q_changed,
            "v_changed_parameters": v_changed,
            "max_lora_parameter_change": max_change,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "oom": False,
            "cpu_offload": False,
        }
        write_json(OUTPUT / "training_gradient_summary.json", summary)


callbacks_map["medicalskill_v1_audit"] = MedicalSkillV1AuditCallback
