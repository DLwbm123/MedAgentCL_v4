"""Current-task-only deterministic sampling and teacher-forced token selection."""
import copy
import hashlib
import json
from collections import Counter
import torch


def stable_id(row):
    value = row.get("id") or row.get("stable_id") or row.get("sample_id")
    if value is None:
        raise ValueError("Calibration requires stable sample IDs")
    return str(value)


def ordered_rows(rows, seed):
    ids = [stable_id(r) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate calibration IDs")
    return sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:{stable_id(r)}".encode()).hexdigest())


def select_tokens(attention_mask, labels, visual_mask, *, count, seed):
    if attention_mask.ndim != 1 or not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
        raise ValueError("Token selector requires an explicit 1-D binary attention mask")
    valid = attention_mask.bool().cpu()
    if not valid.any():
        raise ValueError("No valid calibration tokens")
    labels = labels.cpu() if labels is not None else None
    visual = visual_mask.bool().cpu() if visual_mask is not None else None
    if any(x is not None and x.shape != valid.shape for x in (labels, visual)):
        raise ValueError("Token metadata shape mismatch")
    groups = {}
    types = {}
    for i in valid.nonzero().flatten().tolist():
        kind = "vision" if visual is not None and visual[i] else (
            "answer" if labels is not None and labels[i] != -100 else
            "prompt" if labels is not None and visual is not None else "unknown_other")
        groups.setdefault(kind, []).append(i)
        types[i] = kind
    generator = torch.Generator().manual_seed(seed)
    for kind, values in groups.items():
        groups[kind] = [values[i] for i in torch.randperm(len(values), generator=generator).tolist()]
    selected = []
    # Round-robin existing categories, then deterministic within-category random selection.
    while len(selected) < min(count, int(valid.sum())):
        for kind in sorted(groups):
            if groups[kind] and len(selected) < count:
                selected.append(groups[kind].pop())
    selected.sort()
    return torch.tensor(selected, dtype=torch.long), {
        "positions": selected, "selected_types": dict(Counter(types[i] for i in selected)),
        "valid_tokens": int(valid.sum()), "requested_tokens": count,
        "visual_mapping": "native_visual_pos_masks" if visual is not None else "unavailable_unknown_fallback",
    }


def shuffled_target(Z, seed):
    permutation = torch.randperm(len(Z), generator=torch.Generator().manual_seed(seed))
    if len(Z) > 1 and torch.equal(permutation, torch.arange(len(Z))):
        permutation = permutation.roll(1)
    return Z[permutation.to(Z.device)], permutation


def encode_current_rows(rows, template, config):
    """Same ms-swift train template/collator; rejects overlength rows explicitly.

    No image/answer data are written to diagnostics. Counts and IDs are recorded.
    Attention is explicit for batch-size-one, where the collator can omit it.
    """
    from swift.template import MaxLengthError
    selected, rejected = [], []
    need = config.calibration_samples + config.holdout_samples
    for row in ordered_rows(rows, config.seed):
        try:
            encoded = template.encode(copy.deepcopy({k: row[k] for k in ("messages", "images", "videos") if k in row}))
        except MaxLengthError as exc:
            rejected.append({"id": stable_id(row), "reason": type(exc).__name__})
            continue
        batch = template.data_collator([encoded])
        if "input_ids" not in batch or batch["input_ids"].shape[0] != 1:
            raise ValueError("TPM requires unpacked, batch-size-one teacher-forced input_ids")
        if "attention_mask" not in batch:
            batch["attention_mask"] = torch.ones_like(batch["input_ids"])
        if "labels" not in batch:
            raise ValueError("Use train-mode template with teacher-forced labels")
        selected.append({"id": stable_id(row), "split": "fit" if len(selected) < config.calibration_samples else "holdout",
                         "batch": batch})
        if len(selected) == need:
            break
    if len(selected) != need:
        raise ValueError(f"Only {len(selected)} encodable current rows; need {need}")
    assert not ({r["id"] for r in selected if r["split"] == "fit"} &
                {r["id"] for r in selected if r["split"] == "holdout"})
    return selected, rejected
