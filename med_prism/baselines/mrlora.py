"""MR-LoRA routing contracts independent of a particular MLLM backend."""
from __future__ import annotations
import hashlib
import re
from collections import Counter
from typing import Any


def parse_router_output(output: str, available_experts: list[int], sample_id: str, seed: int = 42) -> dict[str, Any]:
    """Parse a generated expert token; invalid output gets deterministic non-oracle fallback."""
    matches = [int(value) for value in re.findall(r"EXPERT[_\s-]*0*([1-9]\d*)", output.upper())]
    valid = len(matches) == 1 and matches[0] in available_experts
    if valid:
        selected = matches[0]
    else:
        digest = hashlib.sha256(f"{seed}|{sample_id}|{output}".encode()).digest()
        selected = available_experts[int.from_bytes(digest[:8], "big") % len(available_experts)]
    return {"router_output": output, "selected_expert": selected, "valid": valid, "fallback_used": not valid, "fallback_policy": None if valid else "deterministic_hash_over_available_experts"}


def router_diagnostics(records: list[dict[str, Any]], stage: int) -> dict[str, Any]:
    labels = list(range(1, stage + 1)); confusion = {str(t): {str(p): 0 for p in labels} for t in labels}
    per_true = Counter(); correct = Counter(); selected = Counter(); invalid = fallback = 0
    for record in records:
        true, predicted = int(record["true_task_id"]), int(record["selected_expert"])
        if true not in labels or predicted not in labels: raise RuntimeError("Router diagnostic contains a future expert")
        confusion[str(true)][str(predicted)] += 1; per_true[true] += 1; selected[predicted] += 1
        correct[true] += int(true == predicted); invalid += int(not record["valid"]); fallback += int(record["fallback_used"])
    total=sum(per_true.values()); hits=sum(correct.values())
    return {"status":"PASS","stage":stage,"available_experts":labels,"router_accuracy_overall":hits/total if total else 0.0,"router_accuracy_per_true_task":{str(t):correct[t]/per_true[t] if per_true[t] else None for t in labels},"confusion_matrix":confusion,"expert_selection_frequency":{str(t):selected[t] for t in labels},"invalid_router_outputs":invalid,"fallback_frequency":fallback,"total":total}


def assert_stage_router(stage: int, router_stage: int, expert_ids: list[int]) -> None:
    if router_stage != stage: raise RuntimeError(f"Row {stage} must load router_{stage}, not router_{router_stage}")
    if expert_ids != list(range(1, stage + 1)): raise RuntimeError("Evaluation expert set contains missing/future experts")

