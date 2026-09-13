# Phase 4 Med-PRISM Explicit Rank-1 Expert Bank

## Scope

Phase 4 migrates only the original Med-PRISM explicit rank-1 expert bank and
orthogonality regularizer. It does not implement routing, shared/private
experts, Octopus, a full continual-learning driver, merging, DDP, or multi-GPU.
No ms-swift core file is modified.

The immutable backbone is Qwen/Qwen3-VL-8B-Instruct at revision
0c351dd01ed87e9c1b53cbc748cba10e6187ff3b; ms-swift is fixed at 4.4.1.

## Adapter Semantics

Each selected nn.Linear is replaced by a project-side Rank1ExpertLinear. For
task t, expert e stores explicit tensors A[t,e] with shape [1, in_features] and
B[t,e] with shape [out_features, 1]. Its contribution is:

    (alpha_t / E_t) * B[t,e] * A[t,e] * x

The layer output is the frozen base output plus every active task/expert
contribution. Scaling is permanently attached to each task bank. Adding later
tasks cannot rescale an older task.

Only the 36 q_proj and 36 v_proj modules under the actual Qwen3VLTextModel
object are eligible. Vision, merger/aligner/projector, k_proj, and o_proj are
rejected by the target audit.

## Continual Update

Task 1 creates four rank-1 experts per target. Task 2 explicitly loads Task 1's
rank1_manifest.json, freezes every Task 1 tensor, adds four Task 2 experts, and
activates both banks. The optimizer must contain every current expert and no old
expert. There is no PEFT LoRA wrapper in this path.

The regularizer uses the squared normalized Frobenius overlap between rank-1
updates:

    overlap^2 = ((normalize(B_new) @ normalize(B_old).T) *
                 (normalize(A_new) @ normalize(A_old).T))^2

The squared mode returns the mean overlap. RMS returns
sqrt(mean_overlap + eps). Old tensors are detached. Task 1 has an exact zero
orthogonality loss with a valid current-parameter gradient path.

## ms-swift Integration

med_prism_rank1 is registered through the ms-swift 4.4.1 project-side tuner
plugin map. A conditional project-side extension returns MedPrismRank1Trainer
only for this tuner; native lora continues to use the stock Swift trainer. The
custom trainer adds the orthogonality term, records per-step loss identities and
gradient norms, and audits optimizer and activation-checkpoint behavior.

## Checkpoint Contract

Every checkpoint contains rank1_adapter.safetensors and rank1_manifest.json.
The manifest pins the backbone revision, ms-swift version/commit, exact 72
target names and hash, task banks, fixed scaling, tensor schema/shapes/dtypes,
dataset provenance, and file checksum. Loading requires the explicit manifest
path and rejects wrong revisions, target hashes, schema counts, duplicate
identities, missing A/B pairs, altered scaling, PEFT or merged checkpoints.
