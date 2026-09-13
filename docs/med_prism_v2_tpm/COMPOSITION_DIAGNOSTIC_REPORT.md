# Frozen A–E composition diagnostic — 2026-09-08

## Outcome

Both tasks and all ten variant evaluations completed PASS. Each task uses all256
existing grouped diagnostic-dev rows, with identical order and generation settings
across A–E. GPU0 ran Task1; GPU1 ran Task2; both processes exited.
No training, TPM fitting/repair, parameter tuning or checkpoint publication.

| Composition | Task1 accuracy | Task1 KL | Task2 accuracy | Task2 KL |
|---|---:|---:|---:|---:|
| A: S2 + P1 + P2 |91.40625%|0|63.28125%|0|
| B: S3 + P1 + P2 + P3 |91.40625%|0.03611441|40.62500%|0.40450952|
| C: S3 + P1 + P2 |91.015625%|0.00023827|64.453125%|0.00144854|
| D: S2 + P1 + P2 + P3 |91.796875%|0.03563917|40.62500%|0.40308910|
| E: S3 + TPM(P1) + TPM(P2) + P3 |91.40625%|0.03597909|40.62500%|0.40251826|

All compositions also contain the identical base model. KL is mean full-vocabulary
KL(teacher || variant), in nats, at the FIRST next-token position of the identical
generation prompt. It is not sequence-level KL or teacher-forced answer NLL.

## Recovery relative to B

| Intervention | Task1 accuracy change (pp) | Task1 KL reduction | Task2 accuracy change (pp) | Task2 KL reduction |
|---|---:|---:|---:|---:|
| Remove P3 (C) |−0.390625|99.3402%|+23.828125|99.6419%|
| Restore S2 (D) |+0.390625|1.3159%|0|0.3511%|
| TPM (E) |0|0.3747%|0|0.4923%|

Task2: B loses58 correct answers relative to A; C recovers61 relative to B,
ending3 above A. D and E recover zero net correct answers. This strongly supports
new-private P3 interference as the main source of the Task2 retention loss on this
diagnostic cohort, not S2→S3 shared drift.

Task1: A and B already have equal aggregate accuracy. Removing P3 nearly eliminates
the measured distribution drift but loses one correct answer; restoring S2 gains
one. There is no Task1 aggregate accuracy deficit in this dev cohort to recover.
Do not interpret KL recovery as a promised accuracy gain.

TPM repairs historical-bank read behavior, but leaves P3 active. The present
results are consistent with that intervention missing the dominant source of
Task2 behavior drift. They do not establish that read transport never helps.

## Interaction

A–D form a 2×2 factorial: shared S2/S3 × P3 off/on. For centered raw logits define
I = B − C − D + A, and full drift F = B − A. Report
sqrt(sum_samples ||I||² / sum_samples ||F||²), not a sum of KL differences.

- Task1 interaction/full-drift L2:6.1382%; accuracy B−C−D+A:0pp.
- Task2 interaction/full-drift L2:5.2322%; accuracy interaction:−1.171875pp (3/256).

The compositions are not exactly additive. Interaction is modest relative to the
total measured drift and does not overturn the dominant-P3 explanation. These
descriptive values are not a statistical-significance claim, nor a causal
percentage decomposition of accuracy loss. Interaction can still matter relative
to the much smaller shared-only effect.

## Exact loading / protocol

Repair root R:
`/remote-home/wangbomin/MedPRISM_v2_TPM_repair_backtrack_20260907_110405_19036`

Legacy no-geo root L:
`/remote-home/wangbomin/med_prism_v1_2_ablation_no_geo_orth_medicalskill_v1_2_lite_10k1k_seed42`

- S2: L/med_prism/stage_02/shared/shared_manifest.json.
- A's original P1/P2: Task2 teacher legacy_state private manifests, recorded fully
  in each task's A.loaded.json. Their tensors were checked identical to pre P1/P2.
- S3: R/pre_tpm/shared/shared_manifest.json.
- B/C/D private banks: R/pre_tpm/private/task_0001, task_0002, task_0003,
  each containing private_manifest.json; C uses only1/2.
- E shared/private: all from R/accepted, including transported old banks.
- A/C active banks exactly[1,2]; B/D/E exactly[1,2,3], verified for all72 wrappers.
  P3 is allocated but inactive in A/C; it contributes nothing to their forward.
- Every switch checks all7056 adapter tensor values and dtypes against the selected
  composition. Final restoration of B matches the initial full-model fingerprint:
  14862 persistent tensors /8,805,469,312 scalars, zero changed tensors.
- All input manifest, checkpoint, data and tracked evaluator-source hashes unchanged.
- Base Qwen3-VL-8B-Instruct revision0c351dd01ed87e9c1b53cbc748cba10e6187ff3b;
  original evaluator model/processor recipe, BF16, SDPA, image pixels200704,
  batch4, seed42, greedy do_sample=false, num_beams1, max_new_tokens64.
- Uses existing batched_generate/evaluate_rows unchanged. A forward hook captures
  prefill logits during that same generation; no generation/metric reimplementation.
- Dev files: pilot root
  `/remote-home/wangbomin/MedPRISM_v2_TPM_implementation_plan_20260907_final/data/`
  task_01_vqa/development.jsonl and task_02_diagnosis_classification/development.jsonl.
  They are grouped held-outs for the future pilot but appeared in original no-geo
  training. This is mechanism attribution, not an unseen-data benchmark. No formal
  test data or results were used to select settings in this experiment.

## Artifacts and changes

Result root:
`/remote-home/wangbomin/MedPRISM_v2_TPM_composition_audit_20260908_01`

Within task1/ and task2/:

- summary.json: all variants, KL/MSE/cosine, interaction, invariants and completion.
- A.summary.json through E.summary.json: individual variant results.
- A.predictions.jsonl through E.predictions.jsonl: per-sample answers, correctness,
  KL/MSE/cosine and first-token teacher agreement.
- A.loaded.json through E.loaded.json: actual manifests/activity/tensor verification.
- provenance.json: data order, source hashes and fixed protocol.
- interactions.jsonl: per-sample centered-logit interaction and A–E correctness.

Added only scripts/medicalskill_v2_tpm/audit_compositions.py and
tests/med_prism_tpm/test_compositions.py plus this report/handoff updates.
All20 CPU tests PASS in4.247s. All ten real cells contain256 nonempty predictions;
post-run independent audit verifies sample order and recomputes summary averages
from per-sample diagnostics. Task1 post-load runtime1520.76s, Task2 1339.49s.

Local result copies: D:/postgraduate/med_prism/.work/tpm_v2/composition_results/.
No next experiment was started.
