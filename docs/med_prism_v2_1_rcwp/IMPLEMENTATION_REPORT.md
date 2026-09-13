# Med-PRISM v2.1 RCWP implementation

Date: 2026-09-09. Authoritative repository: `/root/MedAgentCL_v4`.

## Status distinctions

- **Implemented:** isolated RCWP boundary fitter, factor-bound sidecar, signed-net
  training loss, opt-in trainer integration, gated Task1→2→3 runner.
- **Mechanically verified:** CPU synthetic/unit/regression tests and real GPU
  smoke PASS, including unchanged persistent tensors and source hashes.
- **Empirically validated:** NOT established. No continual-learning pilot was run;
  no retention/plasticity/Avg/BWT improvement is claimed. A 4+2 smoke cannot pass
  the research field-quality gate, even if its numerical AUROC is high.

No commit/push, existing checkpoint overwrite, TPM repair, old dev/test fitting,
raw-sample replay, router, inference gating or training was performed this turn.

## Files added

All paths below are relative to `/root/MedAgentCL_v4`:

- `med_prism/rcwp/__init__.py`: public method name/version.
- `med_prism/rcwp/core.py`: FP32 moments, affine ridge solve, reduced-rank folding,
  causal gold-token margin, low-rank signed write and final one-sided penalty.
- `med_prism/rcwp/runtime.py`: sidecar validation and original-forward-only hooks.
- `med_prism/rcwp/plugin.py`: opt-in subclass of the existing no-geo trainer.
- `med_prism/rcwp/data.py`: temporary, grouped, train-only boundary partitions.
- `med_prism/rcwp/boundary.py`: simultaneous scalar-margin backward and four-way
  field-quality comparison, without changing model parameters.
- `med_prism/rcwp/cli.py`: fixed128+64 boundary or fixed4+2 real smoke.
- `med_prism/rcwp/pilot.py`: explicit training lifecycle and quality-gated resume.
- `scripts/medicalskill_v2_1_rcwp/run_pilot.sh`: offline environment entrypoint.
- `tests/med_prism_rcwp/test_rcwp.py`, `run_tests.py`: new tests/regression runner.
- This report and `docs/med_prism_v2_1_rcwp/RESUME.md`.

No existing source files changed. Seven audited original files match their
pre-implementation SHA256 copies, including shared_private trainer/plugin,
adapter, config, no-geo run_train, and transport cli/pilot. Existing dirty state
is preserved. Local `.work/tpm_v2/remote_bridge.py` only gained upload allowlist
entries for the isolated new directories; it stores no credentials.

## Integration and fixed mathematics

The subclass calls the original `MedPrismSharedPrivateTrainer.compute_loss` once,
then adds `0.1 * rcwp_loss`. Original task loss, A-only weight0.1/definition,
effective shared drift weight0.01, optimizer, private rank16, frozen historical
banks, shared learning and cumulative forward remain unchanged. BA geometry
weight remains zero. TPM is retained and is OFF; RCWP config rejects TPM enabled.

The old `SharedPrivateConfig.method_version=1.2` remains an internal compatibility
schema so existing components are not rewritten. Public run contract, RCWP state,
summary and training trace record `method_version=2.1` and
`method_name=Med-PRISM-v2.1-RCWP`. RCWP-off uses the original trainer directly.

Boundary captures real wrapper inputs and gradients on **all valid nonpadding
tokens**, including image/prompt/context. Gold margin uses shifted answer labels,
averaged over gold answer positions. This differentiable proxy is not accuracy.
One scalar backward captures every q/v wrapper; no expert-wise backward.

Each effective B column includes that rank-1 expert's actual scaling. Economy SVD
finds B's numerical column span; it does NOT rotate/rewrite model factors. A
rank-deficient B uses its retained span and an equivalent lift. Zero B rank fails.
FP32 sufficient statistics fit standardized affine coordinates with std floor
1e-6 and ridge1e-3; intercept is unregularized. `solve`, with pseudoinverse fallback,
never explicit inverse. Store `M=lift@K`, so `B_eff M phi = Q K phi`.

`nu=max(RMS own-bank signed score,1)` is fixed from FIT examples' scalar-margin
sensitivities; holdout/future performance never tunes it. Negative budget is zero.

Training aggregates signed contributions over tokens, ranks and all wrappers
separately for each historical task, THEN computes `relu(-score/nu)^2`, averaged
over historical tasks/examples. Positive net writes have zero penalty. There is
no magnitude suppression and no rank-pair ReLU.

**Dropout:** original no-geo task forward retains dropout0.05. The auxiliary
branch implements the specified deterministic `B A stopgrad(h)`, with no new
dropout draws; boundary uses model.eval. This distinction is explicit, and RNG/
forward equality is unit-tested. Auxiliary arithmetic is FP32 even for BF16 model
factors; main forward dtype/order is untouched.

## Gradient ownership

Only current private A/B remain live in the auxiliary contraction. Hidden inputs,
historical A/B/scales and all summary tensors are detached. No auxiliary gradient
to base, shared, old factors, metadata or upstream hidden inputs. Tests deliberately
make otherwise forbidden tensors require gradients to verify isolation.

Hooks enable auxiliary grad during reentrant checkpoint's first no-grad forward,
then are removed before recomputation/backward. A checkpointed toy regression and
trainer-subclass add-on test verify one capture and one loss addition. Multi-GPU
DDP/full production optimization has not been empirically validated this turn;
initial pilot command below is single-GPU.

## Sidecar and history-free lifecycle

`summary.pt` schema `MedPRISM_RCWP_summary_v1` contains task/version, FP32 M/mu/scale
per wrapper, scalar nu, effective rank/condition/fit/holdout diagnostics, source
state/component/data hashes, config, and exact raw A/B/scaling/dtype factor hashes.
No Q matrix, raw row, sample feature, gradient, old activation or episodic logits
is persisted. Tensor payload is about85.5KiB/task; serialization and comprehensive
hash metadata add overhead, reported separately by the smoke.

Load rejects wrapper-name, task coverage, rank/shape, dtype, nonfinite values,
scaling or A/B hash mismatch. Gauge-equivalent refactorization also fails because
the factors, not just BA, are bound. Future TPM edits of A invalidate this summary.
No automatic canonicalization or refitting on old data is allowed.

Task1 trains with exactly zero RCWP, then creates its summary while Task1 is
current. Task2 reads only current Task2 training data plus summary1/frozen banks;
Task3 similarly uses summaries1/2. Summary fitting uses128 fit +64 independent
group-disjoint holdout rows from CURRENT `train.jsonl`, never diagnostic dev/test.
Temporary encoded data are discarded after the boundary. Existing grouped pilot
training files are reused, not copied into replay storage. The metadata is tiny
parameter-conditioned functional metadata, not a claim of absolutely no metadata.

Each task publishes an immutable compatible adapter `state/state.json` and a
`rcwp_state.json` index with shared/private manifests, ordered summary sidecars,
their hashes, source-state hash and quality status. The old state serializer is
reused only for component I/O, never to run TPM. Boundary completion binds summary
and source hashes. Next-task training verifies summary integrity and GO status;
trainer then verifies every factor binding. Incomplete outputs are preserved,
not silently retrained/overwritten. Reuse of completed stages requires the same
training contract. A failed boundary should be inspected before explicitly moving
that exact failed boundary aside; there is no automatic destructive recovery.

Inference continues to load ordinary shared/private manifests and sum all active
banks. It requires neither RCWP sidecars nor task identity/router/additional pass.

## Tests

New tests cover known affine recovery, full/deficient-rank folding, positive versus
same-norm negative writes, task-before-ReLU aggregation, Task1 exact zero,
auxiliary gradient ownership including metadata/hidden inputs, schema/hash/scaling
hard failures, BF16+FP32 arithmetic, checkpoint/sidecar roundtrip, forward and RNG
invariance, reentrant checkpoint capture, actual boundary all-token gradients,
group-disjoint splitting, subclass integration, and AUROC ties/single-class safety.

Reproduce the selected new/existing regression suites:

```bash
cd /root/MedAgentCL_v4
/root/anaconda3/envs/medagentcl_v4/bin/python tests/med_prism_rcwp/run_tests.py
```

Selected coverage:21 new RCWP +22 TPM/diagnostic +16 version/A-only regression
+5 existing evaluation-contract +2 lower-triangular checks. No claim that every
unrelated repository test suite was executed. Final test log:
`/remote-home/wangbomin/MedPRISM_v2_1_RCWP_cpu_tests_final_20260909.log`.

## Real smoke and field quality

Final attempt:
`/remote-home/wangbomin/MedPRISM_v2_1_RCWP_smoke_20260909_041552_20243`
(log is the same path plus `.log`). Uses existing Task3 **pre-TPM** state and only
Task3 grouped training rows, 4 fit +2 holdout. No new teacher, no optimizer step.
An ephemeral next-task bank tests real-input write/gradient ownership, then is
removed before full persistent-tensor invariants. No new model checkpoint saved.

The smoke is an explicitly isolated boundary reconstruction, not a legitimate
retroactive summary for historical Task1/2 in a completed training sequence. It
must not be inserted into old runs. The real gradient probe protects Task3 against
an ephemeral Task4 bank; full multiple-history ownership is CPU-tested. Sidecar
reload is bitwise; full real model state is read from existing components and
fingerprinted, not duplicated to a new checkpoint.

Field-quality protocol is fixed before observing results: one seeded unit-norm
rank-mixing write in the last/deep q and v wrappers, amplitude ±0.01, for each
holdout example. Actual BF16-deployed margin differences are compared with
conditional field, constant sensitivity, SRG-like historical response direction,
and response magnitude. Report sign accuracy, Spearman, harmful AUROC; no fitting
uses holdout outcomes. No-class AUROC is undefined, not fabricated.

Full128+64 boundary gate requires AUROC≥0.65 AND higher AUROC than constant/SRG.
That screening rule is not statistical significance or a theoretical guarantee.
The 4+2 smoke always reports `INCONCLUSIVE_SMALL_SMOKE`, never GO. The small, deep
wrapper perturbation family does not establish arbitrary future-write prediction.

First attempt `_20260909_040640_10995` completed sensitivity fitting but stopped
because sklearn was unavailable. It is preserved with failure.json. AUROC now
uses tested Mann–Whitney average ranks (SciPy), without installing packages.

GPU smoke **PASS**, elapsed293.867s (4m54s, including model load and full hashes),
peak PyTorch allocated19,828,485,632bytes (18.47GiB). All72 wrappers have effective
B rank16. Condition number range8.601–282.622, mean77.713; finite checks PASS.
Mean per-wrapper fit regression error9.7278e-5; holdout1.66461e-4 versus constant
1.77866e-4. These tiny-sample regression errors do not imply behavioral prediction.

| Predictor | Sign accuracy | Spearman | Harmful AUROC |
|---|---:|---:|---:|
| Conditional RCWP | 0.50 | -0.3333 | 0.3750 |
| Constant field | 0.50 | -0.1905 | 0.5625 |
| SRG-like direction | 0.25 | -0.4286 | 0.2500 |
| Response magnitude | 0.50 | 0.6429 | 0.8750 |

Only2 holdout examples/8 perturbations: **INCONCLUSIVE_SMALL_SMOKE, NOT GO**.
Conditional did not beat constant and is below0.65. This is insufficient evidence
to launch a full RCWP benchmark, not a reliable estimate of population AUROC.
Possible limitations are only4 fit examples, a projected affine sensitivity
approximation, and BF16 rounding of tiny perturbations; no specific failure cause
has been established. No tuning or additional experiment was launched.

Real next-bank signed score was positive9.9068944e-5, hence RCWP loss exactly0 as
required. Nonzero A/B gradient ownership was checked on the signed-score branch
(not a fabricated nonzero penalty): gradient norm sums1.81037/31.26052, no forbidden
gradients. Actual negative-penalty A/B ownership is separately unit-tested.
No optimizer steps. Sidecar reload exactly matches. Final14862 persistent tensors
(8,805,469,312 scalars) all unchanged; source/code hashes unchanged.

Tensor payload87,552bytes=85.5KiB, plus scalar nu in metadata; serialized sidecar
464,258bytes≈453.4KiB including per-expert hashes, schema and diagnostics. The
85.5KiB estimate refers to tensors, not the full metadata-rich serialized file.

Valuable files in the final smoke directory: `completion.json`, `summary.pt`,
`summary_diagnostics.json`, `field_quality.json`, `invariants.json`, `request.json`.
No per-sample features/gradients/logits were persisted.

## Next explicit pilot commands — NOT EXECUTED

**Do not run the full pilot yet based on these smoke numbers.** First a legally
current, fixed128+64 boundary-quality validation needs to provide adequate evidence.
For an explicitly approved Task3 boundary reconstruction (still train-only), the
implemented command is:

```bash
cd /root/MedAgentCL_v4
export HF_HOME=/remote-home/wangbomin/huggingface_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=/remote-home/wangbomin/MedPRISM_v2_1_RCWP_field_gate_$(date +%Y%m%d_%H%M%S)_${RANDOM}
CUDA_VISIBLE_DEVICES=0 /root/anaconda3/envs/medagentcl_v4/bin/python -m med_prism.rcwp.cli \
  --state /remote-home/wangbomin/MedPRISM_v2_TPM_repair_backtrack_20260907_110405_19036/pre_tpm/state.json \
  --current-task 3 \
  --current-train /remote-home/wangbomin/MedPRISM_v2_TPM_implementation_plan_20260907_final/data/task_03_concept_recognition/train.jsonl \
  --output-root "$OUT"
```

That command was NOT run and does not train/repair a model. Its summary must not
be backfilled into historical training runs. The following pilot recipe is for
a later user-approved run **after sufficient field-quality evidence**, not a
recommendation to start it now. It starts fresh Task1 rather
than inventing unavailable historical summaries. Existing reserved dev remains
excluded from both training and boundary fitting.

```bash
cd /root/MedAgentCL_v4
DATA=/remote-home/wangbomin/MedPRISM_v2_TPM_implementation_plan_20260907_final/data
OUT=/remote-home/wangbomin/MedPRISM_v2_1_RCWP_pilot_$(date +%Y%m%d_%H%M%S)_${RANDOM}

# Plan only: no training or data modification.
bash scripts/medicalskill_v2_1_rcwp/run_pilot.sh \
  --data-root "$DATA" --output-root "$OUT" --branch rcwp --gpus 0

# Run only after explicit approval. Stops on any non-GO boundary.
bash scripts/medicalskill_v2_1_rcwp/run_pilot.sh \
  --data-root "$DATA" --output-root "$OUT" --branch rcwp --gpus 0 --run

# Matched no-geo control, separate branch, unchanged recipe.
bash scripts/medicalskill_v2_1_rcwp/run_pilot.sh \
  --data-root "$DATA" --output-root "$OUT" --branch baseline --gpus 1 --run
```

Keep the same OUT for deliberate completed-stage resume. Incomplete stages hard
fail rather than overwrite prior experiments. No automatic benchmark or evaluator
is appended. Subsequent retention/plasticity/Avg/BWT/transfer analysis must reuse
the existing evaluator with the explicit published state manifests.

## Unverified risks

Finite/synthetic recovery and hooks do not demonstrate conditional field
generalization, current-to-old transfer, or continual-learning benefit. The full
128+64 gate is not run. Few-example smoke regression errors and correlations are
unstable. BF16 quantization can dominate very small interventions. Only a fixed
deep q/v perturbation family is evaluated. All-token backward and FP32 auxiliary
graphs add runtime/memory; full training/DDP overhead is unmeasured. Cumulative
history metadata grows linearly; fit conditioning and effective rank must be
inspected. No lambda/ridge/rank/rho tuning or new mechanism is proposed here.
