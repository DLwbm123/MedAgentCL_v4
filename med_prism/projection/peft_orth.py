"""Compatibility module for the migrated explicit rank-1 bank.

Phase 4 does not use PEFT LoraLayer. Orthogonality is computed directly from the
explicit Rank1Expert A/B tensors.
"""

from .orth_losses import OrthLossResult, compute_rank1_orth_loss, rank1_overlap_squared

__all__ = ["OrthLossResult", "compute_rank1_orth_loss", "rank1_overlap_squared"]
