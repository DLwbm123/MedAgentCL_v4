# TPM BF16 deployment-aware gamma backtracking — 2026-09-07

## Scope and files

- `med_prism/transport/safety.py`: deployment-only gamma backtracking and bank diagnostics.
- `tests/med_prism_tpm/test_transport.py`: two new regression test methods.
- This report and `RESUME.md`: current handoff; previous implementation evidence is historical.
- No checkpoint, prior repair result, dataset, training code or launch script changed. No complete 64+64 repair or GPU job launched this session.

## Rule

The unchanged analytic solver returns `result.delta = initial_gamma * D_s`.
The initial candidate is exactly the existing `result.A_new.to(model_dtype)`.
Only when the complete gate has the single reason `post_cast:edit_cap`, reduce gamma by
`rho / measured_deployed_relative_edit * 0.999`. At most 16 reductions are allowed.
Each retry starts from the original A and original solver delta, scaled by
`final_gamma / initial_gamma`; it never accumulates rounded edits. Each retry casts
again, measures fit and holdout again in FP64 using the already-rounded factors,
then reruns all existing gates. Stop on success, any other rejection reason, or
the retry limit. Zero initial gamma is not backtracked.

No tolerance changed: rho remains 0.05, edit tolerance 1e-6, fit/objective tolerances
remain atol=1e-8 and rtol=1e-6. Finite, rank, and optional holdout gates are unchanged.
The fixed retry limit/safety factor are numerical safeguards, not tunable search parameters.
Solver, rank budget, eta, RMS scaling, teacher/student pairing, sequential block
calibration, training and key geometry are unchanged.

New per-bank fields: `initial_gamma`, `final_gamma`, `backtracking_steps`,
`initial_post_cast_relative_edit`, `final_post_cast_relative_edit`.
Existing solver-level `damping` and top-level solver metrics retain their original
meaning; `post_cast` and `holdout` describe the final proposed deployment candidate.
On rejection the existing caller retains A; the rejected proposal is not committed.

## Verification

Server CPU-only command:

```bash
cd /root/MedAgentCL_v4
CUDA_VISIBLE_DEVICES='' /root/anaconda3/envs/medagentcl_v4/bin/python -m unittest discover -s tests/med_prism_tpm -p 'test_*.py' -v
```

Result: **14 tests PASS in 6.020 seconds**, including all 12 existing tests and all
15 supplied solver numerical checks. Existing sequential/native equivalence,
A-only invariants, checkpoint roundtrip, next-key, strict rollback, off,
unbudgeted and shuffled tests continue to pass.

New real numerical test (seed 4; FP32 solve, BF16 factors; rank=2, eta=.1, rho=.05):

- FP32 measured relative edit: 0.04999999668530111.
- Initial BF16 relative edit: 0.05006400429945986.
- After one retry: 0.0498208437456339, commit=true.
- Gamma: 0.10902299641021429 -> 0.10877473240287648.
- Fit read error: 468.9013948097671 -> 422.88742682710983.
- Regularized objective: 468.9013948097671 -> 423.3049604205451.
- Input tensors unchanged; actual repeated measurements checked by call count.

Second new test simulates an unresolved quantization plateau and verifies rejection
after exactly 16 retries. Its mixed-failure subcase verifies objective failure is
not bypassed and no backtracking starts when another gate already fails.

Before documentation updates, 31 other files matched the previous SHA256 ledger:
16 untouched TPM implementation/docs files and all 15 audited v1 source files.
Analytic solver SHA256 remains
`c7e16d598e6d3cb5fcf8e2e49bb16ebb58cb61a4955b055862c8f8ca3b46563c`.

## Next session

Code and lightweight tests are complete. Do not automatically rerun repair.
If requested later, use a fresh output root for a new 64+64 run; keep
`MedPRISM_v2_TPM_repair_20260907_101442_6069` unchanged as the pre-fix comparison.
The synthetic regression does not establish the new full-run commit count or retention.
