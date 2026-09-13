from __future__ import annotations

from dataclasses import dataclass

from med_prism.adapters.injection import iter_rank1_wrappers


@dataclass(frozen=True)
class ExpertBankRecord:
    module_identity: str
    task_id: int
    expert_id: int
    alpha: float
    per_task_rank: int
    scaling: float
    frozen: bool
    active: bool


def summarize_expert_bank(model) -> list[ExpertBankRecord]:
    records = []
    for _, wrapper in iter_rank1_wrappers(model):
        for identity in wrapper.expert_identities():
            records.append(
                ExpertBankRecord(
                    module_identity=wrapper.module_identity,
                    task_id=identity.task_id,
                    expert_id=identity.expert_id,
                    alpha=identity.alpha,
                    per_task_rank=identity.per_task_rank,
                    scaling=identity.scaling,
                    frozen=not identity.trainable,
                    active=identity.active,
                )
            )
    return records
