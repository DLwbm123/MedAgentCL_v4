# Task3 pre/post TPM retention evaluation audit — 2026-09-07

## Status

Audit and minimal evaluation adapter COMPLETE. Formal evaluation NOT started.
No training, TPM repair, Stage0, oracle, dataset edit or hyperparameter tuning.

Run root:
`/remote-home/wangbomin/MedPRISM_v2_TPM_repair_backtrack_20260907_110405_19036`

State indexes: `<run>/pre_tpm/state.json` and `<run>/accepted/state.json`.
For EACH state, manifests are resolved relative to that state's directory:

- `shared/shared_manifest.json` (Task3 shared)
- `private/task_0001/private_manifest.json`
- `private/task_0002/private_manifest.json`
- `private/task_0003/private_manifest.json`

Post-TPM loads all four from accepted, never historical training-stage paths.

## Read-only findings

- Repair diagnostics: 144 candidates, 144 commits.
- Both indexes and all manifest/weight checksums validate.
- Direct saved-tensor comparison: 7056 adapter tensors, 2243 changed tensors,
  exclusively historical Task1/Task2 A. All other tensor values, shapes and dtypes
  are identical, including shared, current private and historical B.
- Component model identity, rank, alpha, scaling, task ids and targets agree.
- Recorded full-model repair invariants PASS: 14862 persistent tensors,
  8,805,469,312 scalars; base/shared/current/historical B unchanged.
- Base is not duplicated in adapter-only states. Both evaluations use the same
  local Qwen3-VL-8B-Instruct snapshot revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
- Shared safetensors file hashes differ due to serialization; tensor values were
  compared directly and are exactly equal. File hash differences are not weight differences.
- No eval_pre_tpm or eval_post_tpm output directory existed at final audit.

## Why a wrapper is needed

Existing `evaluate_formal_v1_2.load_model(stage)` constructs private manifests
from each original training stage. It cannot consume v2 state indexes directly.
Its old training-logit reload probe is also not a valid post-TPM reference.

New `scripts/medicalskill_v2_tpm/evaluate_retention.py` redirects ONLY the compose
call's manifest arguments while reusing the original load_model(3) recipe. It
checks loaded adapter tensors exactly against the selected state. It does not
claim equivalence to old, pre-repair training logits.

Generation calls the existing `evaluate_formal_v1_2_cell.batched_generate`;
metrics call the existing `evaluate_formal_v1_2.evaluate_rows`. Neither is copied,
edited or reimplemented. All three private banks remain active for every task.
The old evaluator main/finalizer is not called, so no Stage0/oracle requirement.

## Fixed common evaluation contract

- Dataset: `/remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k` (same as previous no-geo evaluation).
- Task1/2/3 complete test.jsonl splits: 1000 samples EACH; eval_limit=0.
- JSONL file order, no shuffle. File and ordered sample-id hashes recorded.
- Batch size4, seed42, BF16, SDPA, original processor min/max_pixels=200704.
- Greedy generation, do_sample=false, num_beams=1, temperature/top_p/top_k=None.
- max_new_tokens: Task1=64, Task2=64, Task3=128.
- R31=accuracy, R32=accuracy, R33=all_concept_micro_f1 (existing primary_score).
  Task3 core_vocabulary_micro_f1 is also preserved in its task summary.
- State, code, dataset and settings provenance must agree before comparing results.
- Existing output directories are refused, not overwritten or silently resumed.
  If an evaluation is interrupted, inspect its partial output before choosing any retry action.

## Exact server commands (not executed by this audit)

Run in one shell, sequentially on GPU0 for a matched hardware comparison:

```bash
cd /root/MedAgentCL_v4
RUN=/remote-home/wangbomin/MedPRISM_v2_TPM_repair_backtrack_20260907_110405_19036
PY=/root/anaconda3/envs/medagentcl_v4/bin/python

# Pre-TPM: R31/R32/R33
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/medicalskill_v2_tpm/evaluate_retention.py \
  --run-root "$RUN" --state pre_tpm

# Run after pre-TPM finishes successfully: post-TPM R31/R32/R33
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/medicalskill_v2_tpm/evaluate_retention.py \
  --run-root "$RUN" --state post_tpm

# After both succeed: prints existing evaluator scores, delta=post-pre
CUDA_VISIBLE_DEVICES='' "$PY" scripts/medicalskill_v2_tpm/evaluate_retention.py \
  --run-root "$RUN" --summarize
```

Outputs: `<run>/eval_pre_tpm/` and `<run>/eval_post_tpm/`, each containing
provenance.json, loaded_state.json, task_01/02/03 predictions+summary, completion.json.
Summary prints `state,task1,task2,task3` with pre_tpm/post_tpm/delta rows on a 0–1 scale.
No genuine retention scores exist yet; unit-test dummy scores are not experiment results.

## Changes and verification

- Added evaluation wrapper and `tests/med_prism_tpm/test_retention.py` (3 CPU tests).
- Added this report and updated handoff documentation. Existing evaluation and TPM
  algorithm sources unchanged; safety and analytic solver hashes still match previous patch.
- Read-only real-state/contract checks PASS, without base loading or generation.
- All17 unit tests PASS in4.236s: previous14 plus3 evaluation wiring tests.
  Wiring tests use mocks/temporary synthetic data, checking explicit transported
  bank paths, loaded tensor mismatch rejection, JSON-stable paired contracts,
  original generation/metric calls, no overwrite, summary and corrupt-result rejection.
- Actual GPU model-loading/generation path has not been smoke-run in this audit;
  the wrapper performs exact adapter load checks when the user starts evaluation.

Next action requires user execution/authorization of the above formal evaluations.
