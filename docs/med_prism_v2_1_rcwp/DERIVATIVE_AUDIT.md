# Frozen derivative-chain diagnostic (launch-only)

Launched GPU0, PID78598:
`/remote-home/wangbomin/MedPRISM_v2_1_RCWP_derivative_audit_20260909_061641_22602`.
Log is samepath + `.log`. Startup snapshot: process RNl, PRECHECK PASS with
original state/data/code/summary hashes verified; frozen model loading normally.
STOP monitoring after this confirmation. Completion/report are expected outputs,
not yet observed. User may request a later one-time status inspection.

Full gate reused:
`/remote-home/wangbomin/MedPRISM_v2_1_RCWP_field_gate_20260909_053055_12020`.
No summary refit, loss/ridge/lambda change, model training, TPM, or pilot.

New wrapper: `scripts/medicalskill_v2_1_rcwp/audit_derivatives.py`.
New CPU tests: `tests/med_prism_rcwp/test_derivative_audit.py`.
Existing RCWP files are unchanged; their original full-gate code hashes must match.
Regression log: `/remote-home/wangbomin/MedPRISM_v2_1_RCWP_derivative_cpu_20260909.log`.
All selected tests PASS, including FP32/BF16 hooks and parameter immutability.

The same hash-verified source data, encoder, grouped128+64 split and fitted summary
are reused. Only64 holdout samples receive derivative backward. No fit samples are
used for optimization. No fit API is called. Same final/deep q and v perturbations,
seed2100+wrapper-index, fixed amplitudes .01/.05/.1, and original gold margin.

Main predictor comparisons use actual deployed displacement `edited-out` after
the original BF16 cast and add. Central FD compares with the UN-SCALED ideal
direction derivative. A deployed-chord derivative is also recorded to separate
rounding from finite-displacement/nonlinear effects. Baseline gradient and no-grad
margin differences are recorded; wrapper input/output points must match bitwise.

Important limitation: ideal `B C A h` is already in col(B), so projected/exact
ideal derivatives should coincide despite potentially low total gradient-energy
coverage. Off-span BF16 rounding can alter this property for deployed edits.
Do not generalize this specific perturbation family to arbitrary future writes.

Only scalar observations are persisted. Results include original .01 RCWP metric
reproduction checks and final complete persistent tensor/source-file invariants.
Report answers six requested questions without proposing a new method/amplitude.

After detached formal process startup confirmation, STOP polling and end the turn.
The user will request results separately. Check `completion.json` or `failure.json`
then; no automation or continuing monitor is authorized.
