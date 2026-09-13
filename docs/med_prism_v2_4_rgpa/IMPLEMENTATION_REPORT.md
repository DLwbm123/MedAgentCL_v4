# Med-PRISM v2.4 RGPA implementation

## A. Current-best audit

Baseline: `/remote-home/wangbomin/med_prism_v1_2_ablation_no_geo_orth_medicalskill_v1_2_lite_10k1k_seed42`.
Formal 5-task mean: 50.8524252, versus full v1.2 50.5503051, v1.1 49.7447922,
no-shared-drift 50.2519737, rank16-private 48.9387096 (0–100).
Entry: `scripts/medicalskill_v1_2_med_prism_ablation_no_geo_orth/run_train.sh`,
delegating to `scripts/medicalskill_v1_2_med_prism_versions/run_versioned_train.sh`.
All 16 source hashes recorded by this baseline matched the server at audit.
Exact hashes and configuration are retained in `rgpa_contract.json` and `run_manifest.json`.

Actual regularizer: `med_prism/projection/orth_losses.py::compute_rank1_key_isolation_loss`.
A rows are converted to FP32 and L2 normalized with eps=1e-8; history is detached.
Wrapper order is `iter_rank1_wrappers`; historical task IDs are sorted, then local expert order.
Every cosine Gram element is squared, summed over ALL wrappers/history, divided by total
pair count, then sqrt(squared+1e-8). No-history loss is differentiable zero.
Key lambda=.1, effective-BA shared drift lambda=.01, geometric BA lambda=0.
The old BA loss code still executes as before, multiplied by zero.

Qwen3-VL-8B-Instruct revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`;
shared rank/alpha32, private16 local rank1 experts/alpha16, dropout.05;
separate shared/private AdamW LR1e-5/1e-4, one epoch, effective batch16,
seed/data_seed42, BF16/SDPA, training max_length1024 for T1–T3, max_pixels200704.
The baseline script itself supplies all unchanged optimizer/training settings.

Group mapping reuses `ms_crc.composition.group_id`: `4*(layer//9)+rank//4`.
36 layers, q/v72 wrappers, 1152 local rank1 components per task, 16 groups of72.
Not sixteen global experts. Only this mapping is reused, not MS-CRC search/gates.

Official evaluation directly calls the existing cell evaluator with `primary_cumulative`:
latest shared once plus ALL private banks. No Stage0/oracle. T1/T2 accuracy and
Task3 all-concept micro-F1, greedy generation, original sample order and batch4.

## B. Code changes

All additions are isolated; no existing training/TPM/MS-CRC source edited.

- `med_prism/rgpa/core.py`: score normalization, real-output group-off hooks, frozen sigma and scoped pre-reduction Gram interception.
- `med_prism/rgpa/state.py`: checkpoint-local trainable tensors and atomic protection state.
- `med_prism/rgpa/plugin.py`: opt-in existing-trainer subclass, checkpoint/resume integration.
- `med_prism/rgpa/boundary.py`: current-train-only teacher-forced salience/timing.
- `med_prism/rgpa/run.py`: reuse baseline contracts, isolated T1 copy, sequential training/boundaries/official evaluations/report.
- `scripts/medicalskill_v2_4_rgpa/run.sh`: standalone entry.
- `tests/med_prism_rgpa/test_rgpa.py`: mechanical checks against real project wrappers and loss.

## C. RGPA semantics

Source train rows use existing deterministic `ordered_rows(seed42)`, no class resampling.
At most64 valid examples; identical encoded batch/gold answer mask for full and each off group.
Reuse existing encode/token_scores; NLL sum divided by answer-token count per example.
Positive part is taken per example before FP32 averaging. No per-example NLL/logits/IDs saved.
Eight-example timing never supplies formal scores. All persistent tensors are hashed before/after.

Per-history z=(s-mean(s))/max(s), or0 when max0; w=1+.25z.
Sigma is computed on the SAME normalized operands once at task initialization and detached;
it is usually1/sqrt(input_dim). Resume reads saved sigma rather than recomputing.
tau=.25*relu(-z)*sigma; historical columns are transformed by sqrt(w)*signed-soft-threshold.
The original loss implementation performs its original reduction without reimplementation.
The identity path avoids sign-at-zero derivative changes. Candidate B `--without-slack`
only disables tau; it is implemented but not automatically run.

## D. Mechanical tests

Executed on the server: all 9 new RGPA tests PASS (including reload-probe retention),
all 16 existing `medicalskill_v1_2_med_prism_versions` regression tests PASS.
Plugin import, Python compilation and shell syntax checks PASS.
Identity loss/current-A gradient comparisons and CPU next-step resumed loss/parameters
are bitwise equal (maximum absolute error 0). Tests cover:
rho0 exact real-project loss/current-A gradients; uniform recovery; mean/bounds/w-not-w²;
slack penalty/gradient; full/off/full output/NLL/tensor/scaling safety;
short AdamW training with saved trainable tensors/protection/optimizer and exact next-step
loss/parameters after reload; saved sigma restoration; inference invariance; missing-state rejection.
This CPU training smoke uses real project Rank1ExpertLinear and the real orth loss,
not full Qwen training. Any full-model integration smoke is recorded separately.

## E. Runtime

Pending measured eight-example timing; `rgpa/timing_task_01.json` will hold throughput and
64-example estimate. `runtime_approval.json` is mandatory before formal launch.
Boundary actual time is recorded in each `rgpa/task_XX.json`; training time in
`train_taskX_timing.json`. No claimed training/evaluation improvement before completion.

## F. State/resume

`rgpa/task_XX.json`: sixteen scores and derived z/w, fixed rho/slack, group/dataset hashes,
seed/valid count and minimal provenance. No historical examples or per-example predictions.
Each training checkpoint saves `rgpa_state.json` (including histories/frozen sigma) and
`rgpa_trainable.safetensors`, in addition to the original optimizer/scheduler/RNG state.
The trainer restores only current/shared trainable tensors, leaving original history and
shared drift anchors intact. Checkpoints missing RGPA state fail closed.

T1 reuse: all recorded source hashes match, original seed42, dual GPU world_size2,
accumulation8, batch1, one epoch; original component checksums verified. Only immutable
T1 adapter/probe artifacts are copied into the new run. Fresh T1 salience is computed
before T2. T2/T3 use only these source-boundary sidecars. A T3 sidecar is also generated
at T3 boundary to allow later T4 without retrospectively visiting old train data.

## G–I. Results / decision / next command

PENDING: no GO/NO-GO classification until the three-task official evaluation completes.
Runner writes `summary.json`, `results.csv` and appends the final table/decision/next command.
Predeclared GO thresholds and drop budgets are in `rgpa_contract.json`.
The default runner stops at Task3 and never starts Candidate B or T4/T5 automatically.
