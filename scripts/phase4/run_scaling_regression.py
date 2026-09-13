#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from med_prism.adapters.rank1_lora import Rank1ExpertLinear


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(42)
    cases = []
    for experts_per_task in (1, 4, 16):
        for task_count in (1, 2, 3):
            base = nn.Linear(5, 3, bias=False)
            with torch.no_grad():
                base.weight.zero_()
            wrapper = Rank1ExpertLinear(
                base,
                module_identity="model.language_model.layers.0.self_attn.q_proj",
                dropout=0.0,
            )
            alpha = float(experts_per_task)
            for task_id in range(1, task_count + 1):
                wrapper.add_task(
                    task_id,
                    experts_per_task=experts_per_task,
                    alpha=alpha,
                    trainable=task_id == task_count,
                )
                for expert_id, expert in enumerate(
                    wrapper.task_experts(task_id), start=1
                ):
                    with torch.no_grad():
                        expert.A.fill_(0.01 * (task_id + expert_id))
                        expert.B.fill_(0.02 * (2 * task_id + expert_id))

            x = torch.arange(10, dtype=torch.float32).reshape(2, 5) / 10
            expected = base(x)
            for task_id in range(1, task_count + 1):
                for expert in wrapper.task_experts(task_id):
                    expected = expected + F.linear(
                        F.linear(x, expert.A), expert.B
                    ) * (alpha / experts_per_task)
            actual = wrapper(x)
            explicit_error = float((actual - expected).abs().max())

            task1_before = wrapper.task_contribution(x, 1).detach().clone()
            task1_scaling = [
                float(expert.scaling) for expert in wrapper.task_experts(1)
            ]
            if task_count < 3:
                for task_id in range(task_count + 1, 4):
                    wrapper.add_task(
                        task_id,
                        experts_per_task=experts_per_task,
                        alpha=alpha,
                        trainable=False,
                    )
            task1_after = wrapper.task_contribution(x, 1).detach()
            old_task_error = float((task1_before - task1_after).abs().max())
            expected_scaling = alpha / experts_per_task
            scaling_error = max(
                abs(value - expected_scaling) for value in task1_scaling
            )
            passed = max(explicit_error, old_task_error, scaling_error) <= 1e-6
            cases.append({
                "experts_per_task": experts_per_task,
                "task_count": task_count,
                "alpha": alpha,
                "expected_per_expert_scaling": expected_scaling,
                "observed_task_1_scaling": task1_scaling,
                "explicit_sum_max_abs_error": explicit_error,
                "task_1_before_vs_after_new_tasks_max_abs_error": old_task_error,
                "scaling_max_abs_error": scaling_error,
                "pass": passed,
            })

    payload = {
        "status": "PASS" if all(case["pass"] for case in cases) else "BLOCKED",
        "formula": "per_expert_scaling = alpha / experts_per_task",
        "covered_experts_per_task": [1, 4, 16],
        "covered_task_counts": [1, 2, 3],
        "case_count": len(cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "case_count": len(cases)}))
    if payload["status"] != "PASS":
        raise SystemExit("Scaling regression failed")


if __name__ == "__main__":
    main()
