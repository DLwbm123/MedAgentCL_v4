# Med-PRISM v2.0 TPM implementation report

Status: implementation and lightweight verification completed, 2026-09-07.
Prepared pilot commands are NOT completed training experiments. No retention claim is made.

## A. Repository audit and integration decision

Authoritative repository: `/root/MedAgentCL_v4`; local source additions are mirrored in `D:\postgraduate\med_prism`.
Existing dirty modifications were preserved. No v1 source, completed checkpoint, formal result, dataset source, evaluator or paper reference was edited.

| Audited file | Finding |
|---|---|
| `med_prism/adapters/rank1_lora.py` | Each expert `A=[1,d_in]`, `B=[d_out,1]`, output `linear(linear(dropout(x),A),B)*scaling`; scaling=alpha/per_task_rank=1 for the no-geo recipe. |
| `med_prism/adapters/shared_private.py` | Shared rank 32 plus sum of all active task-private experts. All 16 expert rows/columns form one historical bank. Shared/private add to the frozen base projection; no routing. |
| `med_prism/adapters/injection.py` | Dense Qwen language decoder q/v only, 36 blocks / 72 wrappers. q is 4096→4096, v is 4096→1024. |
| `med_prism/projection/orth_losses.py` | `compute_rank1_key_isolation_loss` reads live expert A tensors and detaches old rows. No permanently cached original A. RMS row-cosine uses all previous tasks at the same projection. |
| `med_prism/projection/shared_private.py` | Shared drift uses effective BA against previous shared anchor in the no-geo configuration. |
| `med_prism/optimization/shared_private.py` | Separate shared/current-private optimizer groups, historical A/B excluded; shared LR=1e-5 and private LR=1e-4. |
| `med_prism/swift_plugins/{med_prism_shared_private_plugin,shared_private_trainer}.py` | Load historical manifests then add new bank; freeze old banks. Existing no-geo loss is task + .1 key + .01 effective shared drift; geometry weight 0. |
| `med_prism/checkpoint/shared_private_checkpoint.py` | Existing save writes shared and CURRENT bank only. Existing compose accepts a list of task component manifests. |
| `scripts/medicalskill_v1_2_med_prism_versions/{prepare_versioned.py,run_versioned_train.sh}` | Legacy chain points to each bank's creation-stage path. This cannot be reused unchanged as the v2 state index: it would discard transported historical A at the next task. |
| `swift/template/templates/qwen.py`, `swift/template/base.py` | Actual API: `get_model_processor`, `get_template(processor, template_type='qwen3_vl')`. Train-mode encode/collate supplies labels, attention and multimodal position IDs; Qwen3VL `_post_encode` is identity. |
| Installed `transformers/models/qwen3_vl/modeling_qwen3_vl.py` | Native decoder kwargs are shared across layers. DeepStack visual features are injected by the parent after early decoder blocks. Must not omit this parent operation. |

Runtime audited: torch 2.5.1+cu124, transformers 4.57.6, local ms-swift checkout. Backbone is pinned by existing `MODEL_ID` / `MODEL_REVISION` constants.

Integration is an **independent boundary process and launcher**, with no changes to the existing trainer/plugin. Runtime training config remains `method_version=1.2`, `ablation_name=no_geo_orth`; the outer state machine and output manifests identify v2.0. This is deliberate reuse, not an assertion that v1 acquired new behavior.

## B. New files (v1.x impact: none)

All paths below are relative to `/root/MedAgentCL_v4` (the local mirror has the same paths).

| New file | Purpose |
|---|---|
| `med_prism/transport/analytic_transport.py` | Byte-identical supplied solver including all 15 self-tests and the unused research-only `profiled_read_loss`. |
| `med_prism/transport/__init__.py` | Narrow transport API; nothing imported into the v1 trainer. |
| `med_prism/transport/config.py` | Validated dataclass and argparse controls. |
| `med_prism/transport/banks.py` | Rank-1 expert↔combined-bank mapping, exact tensor hashes, invariant and cosine diagnostics. |
| `med_prism/transport/safety.py` | FP32/explicit FP64 solve, deployment casting, objective/edit/finite gates and rank diagnostics. |
| `med_prism/transport/calibration.py` | Deterministic current-row selection and attention-valid token stratification; shuffle control. |
| `med_prism/transport/hooks.py` | Native Qwen block/metadata/DeepStack bridge and exact one-batch equivalence check. |
| `med_prism/transport/engine.py` | Teacher snapshot swapping and sequential block transport, rollback and audit orchestration. |
| `med_prism/transport/checkpoint.py` | Full-bank pre/accepted indexes, v1-compatible component serialization and exact reload checks. |
| `med_prism/transport/diagnostics.py` | Per-bank JSONL and grouped JSON/CSV summaries. |
| `med_prism/transport/cli.py` | Existing Task2→Task3 checkpoint repair entry point; no training or generation. |
| `med_prism/transport/pilot_data.py` | Global grouped-development exclusion across the three current source training files. |
| `med_prism/transport/pilot.py` | Independent no-geo/TPM three-task chain with common Task1 and accepted-state restart markers. |
| `scripts/medicalskill_v2_tpm/run_checkpoint_repair.sh` | Offline, single-GPU checkpoint pilot wrapper. |
| `scripts/medicalskill_v2_tpm/run_three_task_pilot.sh` | Preparation by default, training only with explicit `--run`. |
| `tests/med_prism_tpm/test_transport.py` | Numerical, configuration, invariant, native small-model, state-chain, grouping and recipe tests. |
| `docs/med_prism_v2_tpm/IMPLEMENTATION_REPORT.md` | This report and commands. |

Local-only working records: `.work/tpm_v2/HANDOFF.md`, credential-free `remote_bridge.py`, read-only downloaded audit snapshots. The paper inputs were only read, never rewritten.

## C. Actual lifecycle

1. `pilot.train_stage` creates `SharedPrivateConfig` using unchanged no-geo settings and launches the existing `swift sft` tuner/plugins. There is no teacher forward or repair loss in this command.
2. Original trainer saves shared/current components and audits. `validate_train_completion` requires all existing relevant audit files to be PASS.
3. `materialize_state` saves a complete `pre_tpm/state.json`, including every accepted historical bank, not just the new bank.
4. `cli.repair` validates `teacher.task_id == student.task_id-1` and exact historical A/B plus scaling agreement. Explicit v2 teacher/previous indexes must be `ACCEPTED`, not `PRE_TPM`. Teacher must be the state that actually initialized gradient training.
5. `run_transport` hashes every persistent model tensor, snapshots adapter factors, and swaps the same base to teacher shared/historical state; current bank is inactive.
6. `QwenBlockBridge.collect_teacher` runs identical teacher-forced current inputs, captures selected historical responses and fit RMS sufficient statistics, and offloads full initial hidden states/native kwargs/visual side inputs to CPU.
7. Restore pre-TPM student factors. Exact native vs streamed equivalence is required on one teacher and one student batch, including DeepStack.
8. For each block: collect current q/v inputs from already repaired upstream states; solve every historical task bank jointly over that bank's 16 experts; cast and gate; commit/reject; then rerun that native block and native DeepStack operation to emit the next block's inputs.
9. Rehash all persistent tensors. Only explicitly committed historical A names may differ; shape/dtype/count unchanged. Invariant failure aborts publication.
10. `materialize_state(..., accepted=True)` writes every final bank, verifies exact file reload, and atomically publishes `accepted/state.json` in a new directory.
11. `pilot.training_config` for the next task takes shared and ALL private paths from `read_state(previous_accepted)`. Existing live-row key loss now sees transported A automatically.

`--tpm-mode off` skips teacher/model/calibration completely in the CLI and copies component weight bytes unchanged. Task1 has no historical banks and is accepted directly. No inference module or signature is altered; existing `compose_shared_private(model, shared_manifest=..., private_manifests=...)` consumes the accepted paths.

## D. Mathematical implementation check

- A is `[r,d]`, B is `[m,r]`, X is `[n,d]`, Z is `[n,r]`.
- B concatenates the historical experts' actual B times their fixed LoRA scaling. B parameters themselves are never modified.
- Objective: `|| (X(A+D)^T-Z) B^T ||²/n + eta ||BD||²`.
- Ridge is exactly `n*eta`; dual n×n when n≤d, primal otherwise. Cholesky, no inverse, no auto jitter.
- `repair_rank=2` applies to **effective BD for each complete historical rank-16 bank**, not per expert. `none` and `0` are supported.
- Reference RMS is one detached scalar per projection from teacher **fit** selected inputs. Student fit/holdout and teacher targets share this scalar. No tokenwise normalization and no holdout contribution to fitting/scaling.
- `rho=.05` caps `||BD||/||BA||`; this is directional damping, not the exact joint rank-plus-hard-cap constrained optimum.
- FP32 by default. FP64 fallback is opt-in, limited to explicit Cholesky failure, and records the failure reason/dtype without changing eta.
- Deployment dtype is the actual expert A dtype. Errors/objective/edit are remeasured from already-rounded candidate values in FP64 to reduce diagnostic cancellation. Solve D rank and post-cast difference rank are separately reported; BF16 exact rank≤s is NOT claimed.
- Default commit: finite, fit error nonincrease, regularized objective nonincrease, edit cap (with logged fixed tolerances). Holdout nonincrease is an optional gate. Strict mode aborts on any rejection/error and restores the student adapter snapshot; no accepted checkpoint is published.
- Shuffle negative control permutes FIT target rows only, preserves exact token multiset/counts, uses the same seeded permutation across banks, and logs its hash. Holdout always uses true pairs. Matched-fit diagnostic is also retained.
- Tokens use native `visual_pos_masks` when available, train labels for answer, explicit attention validity, and a logged unknown fallback. No hardcoded Qwen token IDs.
- Geometry is measured with existing row-cosine helper. No post-repair reorthogonalization.

## E. Verification

Final server suite: **12 test methods PASS in 4.691 s**. One supplied test method executes all 15 required numerical checks. Python compilation and both shell syntax checks passed. Existing Task2/Task3 checkpoint manifests and frozen-history transition passed CLI `--check-only`. All 15 audited existing v1 source files retained their initial SHA256.

Small integration uses three native `Qwen3VLTextDecoderLayer` blocks with synthetic visual injection, two historical banks and a current bank. It verifies bitwise native/blockwise equivalence, actual historical A changes, shared/current/B immutability, strict rollback, accepted serialization, rejection of PRE_TPM teachers, next-task live historical-key access, and unchanged shapes/counts. It also re-runs the FINAL repaired model and checks every normalized projection input hash against the input actually used by that bank's solver, detecting stale-input calibration. Both unbudgeted and shuffled modes run through the small-model boundary integration, not just argument parsing.

Real GPU0 smoke output:
`/remote-home/wangbomin/MedPRISM_v2_TPM_implementation_smoke_20260907_final`.

- Existing no-geo Task2 teacher / Task3 student; only 2 current fit + 2 fit-holdout samples, 4 tokens/sample; rank2, eta=.1, rho=.05.
- Teacher and student native/streamed hidden states both **bitwise equal**, max_abs=0, tested shape `[1,238,4096]`, including native DeepStack.
- All 36 blocks / 72 projections / 144 historical-bank candidates processed. 144 accepted, 0 rejected. 2240 historical A tensor rows changed; first-block zero edits are expected because its upstream input has not drifted. All other persistent tensors unchanged.
- Solve D effective-rank estimate maximum=2. Maximum deployed relative edit=0.0371805; mean=0.0147989. Post-cast difference-rank estimate reaches16: this is explicitly diagnostic, NOT a violated claim of exact BF16 rank2.
- Accepted local fit error aggregate 6.10840→0.29509; fit-holdout 8.36458→4.25972. These tiny-sample local errors **do not measure accuracy, NLL, or retention** and must not be used as evidence of improved medical performance.
- Native visual masks identified vision tokens reliably for these four samples. Each selection contains 1 vision, 1 prompt and 2 answer tokens, with disjoint fit/holdout IDs and no padding positions.
- Full model hash invariant and exact accepted checkpoint save/reload passed. Checkpoint source files remain untouched.
- Standalone real checkpoint `--tpm-mode off` also completed, without loading a model or running calibration: `/remote-home/wangbomin/MedPRISM_v2_TPM_implementation_off_20260907_final`.
- Off shared + all3 private weight hashes match the original components exactly. Real Task3 adapter parameter count is38,338,560 before/after repair, with unchanged tensor shapes. Smoke processes have exited; check live GPU availability before starting any requested experiment.

Grouped-data preparation passes with exactly 2000 training / 256 development rows per task and no cross-task grouping overlap. Concept has16 additional rows belonging to selected development groups; these are excluded from training. No pilot gradient step was run. Prepared final-code plan root:
`/remote-home/wangbomin/MedPRISM_v2_TPM_implementation_plan_20260907_final`.

Development history is preserved, not hidden: first attempt `_a` failed copying network-filesystem xattrs before any model load; fixed by copying bytes only with exact checks. Attempt `_b` passed with139 commits/5 conservative rank rejects. Its rank gate measured subtraction of rounded FP32 A tensors; this adds subtraction/addition roundoff. Final code gates rank on solver-returned D, while retaining the recovered-A and BF16 rank diagnostics separately. The **solver file remains byte-identical** to the supplied reference (SHA256 `c7e16d598e6d3cb5fcf8e2e49bb16ebb58cb61a4955b055862c8f8ca3b46563c`).

Not executed: 64+64 repair study, three-task gradient training, five-task training, full evaluation matrix, generation, multi-seed, or hyperparameter search.

## F. Runtime / memory

Core path uses one base model. Temporary adapter snapshots, full-sequence CPU hidden states/visual metadata and CPU teacher targets are real overhead. One teacher pass and two native student-block passes per sample, plus two one-batch native equivalence comparisons. q/v input/Gram sharing is not optimized in this first implementation; solver is faithful but called per bank.

Measured final smoke (4 examples total, NOT the default128-example repair study):

| Measurement | Result |
|---|---:|
| Teacher pass | 11.461 s |
| Sequential student calibration | 28.369 s |
| Whole engine, including full-base hashing, snapshot checks and other overhead | 172.550 s |
| Outer repair including model load and serialization | 282.501 s |
| Peak GPU allocated | 17,711,070,208 bytes (16.495 GiB) |
| Peak GPU reserved | 17,758,683,136 bytes |
| CPU process peak RSS | 5,846,804 KiB (5.576 GiB) |

No percentage-of-training overhead or accuracy claim is made. CPU peak RSS is process-lifetime Linux `ru_maxrss`, not a clean isolated increment. GPU peaks are allocation/reservation since engine entry; model-load time is included only in outer completion elapsed time. Full-model hashing is a substantial part of observed boundary wall time; forward-only figures must not be reported as total TPM cost. Selected diagnostics copied locally occupy875,024 bytes (this subset excludes per-module geometry JSONs and CSV).

## G. Runnable commands

Run on the server. Every invocation below creates a distinct new output root. Do not reuse a completed or failed repair root; failed attempts retain pre-TPM files for diagnosis.

```bash
cd /root/MedAgentCL_v4
RUN=/remote-home/wangbomin/MedPRISM_v2_TPM_repair_$(date +%Y%m%d_%H%M%S)_${RANDOM}
TPM_GPU=0 bash scripts/medicalskill_v2_tpm/run_checkpoint_repair.sh \
  --output-root "${RUN}_rank2" \
  --tpm-mode transport --tpm-repair-rank 2 --tpm-eta 0.1 --tpm-max-relative-edit 0.05
```

Defaults select the audited existing no-geo Task2 teacher, Task3 student, Task3 current training data, fit64/holdout64/tokens8. Add `--check-only` to verify inputs without loading the model or creating output.

```bash
# Baseline: byte-exact copy, no model load or calibration forward
TPM_GPU=0 bash scripts/medicalskill_v2_tpm/run_checkpoint_repair.sh \
  --output-root "${RUN}_off" --tpm-mode off

# Unbudgeted (same eta/rho and calibration samples)
TPM_GPU=0 bash scripts/medicalskill_v2_tpm/run_checkpoint_repair.sh \
  --output-root "${RUN}_unbudgeted" --tpm-repair-rank none

# Shuffled fit pairing negative control
TPM_GPU=0 bash scripts/medicalskill_v2_tpm/run_checkpoint_repair.sh \
  --output-root "${RUN}_shuffled" --tpm-repair-rank 2 --tpm-shuffle-teacher-pairing
```

Three-task pilot (common Task1, independently trained Task2/3 baseline and TPM branches):

```bash
PILOT=/remote-home/wangbomin/MedPRISM_v2_TPM_three_task_$(date +%Y%m%d_%H%M%S)_${RANDOM}
# First prepare only: 2000 training + 256 globally group-excluded development rows per task
TPM_GPU=0 bash scripts/medicalskill_v2_tpm/run_three_task_pilot.sh \
  --output-root "$PILOT" --gpus 0

# Explicitly start later; same root and flags allow completed-boundary continuation
TPM_GPU=0 bash scripts/medicalskill_v2_tpm/run_three_task_pilot.sh \
  --output-root "$PILOT" --gpus 0 --run
```

`--gpus 0,1` enables two-process gradient training with accumulation8 (global batch16); calibration stays single-card selected by `TPM_GPU`. Do not use an occupied GPU without checking capacity. `--branch baseline` runs only the off branch, `--branch tpm` only the selected TPM branch; common Task1 is reused. A changed source/configuration requires a new pilot root.

Accepted checkpoint use, without any TPM inference code:

```python
from med_prism.transport.checkpoint import read_state
from med_prism.checkpoint.shared_private_checkpoint import compose_shared_private
state = read_state('/absolute/MedPRISM_v2_TPM_run/accepted/state.json')
compose_shared_private(base_model, shared_manifest=state.shared_manifest,
                       private_manifests=state.private_manifests)
```

Test command:

```bash
cd /root/MedAgentCL_v4
/root/anaconda3/envs/medagentcl_v4/bin/python -m unittest discover -s tests/med_prism_tpm -v
```

## H. Limits / unresolved risks

- Synthetic/local reading error improvement is not old-task retention evidence. No accuracy or NLL improvement has been measured here.
- Native bridge is deliberately narrow: dense Qwen3VLTextModel, same-device base, q/v-only adapters. Unknown metadata, different output contracts, or failed exact equivalence abort rather than silently switching to stale inputs.
- Pilot development exclusion uses available connected patient/study/case/lineage/image metadata; incomplete source metadata cannot establish patient independence beyond recorded identifiers.
- No stale-order ablation or training-time repair loss was added.
- Per-bank rejection may yield a partially repaired accepted state. Rejections remain explicit, with original A kept. Strict mode offers all-or-abort publication.
- A failed repair can be rerun from preserved pre-TPM into a new output; the pilot can retry a failed boundary without rerunning gradient training. An interrupted gradient stage is preserved and requires explicit inspection/archive before restart; no silent optimizer-state resume is claimed.
- Key geometry was measured, not restored: historical/current RMS cosine 0.00429986→0.00432955; historical-between-bank RMS 0.0000275792→0.000271355 (about9.84× from a very small starting value). Historical-between-bank max absolute cosine 0.000197532→0.00216147. Do not hide this relative increase or infer its accuracy effect from the smoke; monitor it in the planned repair study.
- Existing formal reasoning parser is intentionally unchanged. New retention studies still need the separately planned evaluation-protocol audit; do not tune on formal test scores.

## Resuming implementation work

Read local `.work/tpm_v2/HANDOFF.md` and this report first. A final server handoff copy is stored beside this report as `RESUME.md`. They list completed work, exact files, tests and next experimental actions. Raw current-data calibration caches are not serialized; only diagnostic IDs/positions remain. No credentials are persisted. No commit/push was made.
