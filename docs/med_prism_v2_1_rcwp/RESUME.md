# v2.1 RCWP implementation ledger — 2026-09-09

## Latest complete

- Real GPU smoke PASS: PID77372 finished; finalroot
  `/remote-home/wangbomin/MedPRISM_v2_1_RCWP_smoke_20260909_041552_20243`.
  4fit+2holdout group-disjoint Task3 TRAIN only; 293.867s; peak18.47GiB.
  72B ranks16;14862persistent tensors unchanged;source/code hashes unchanged.
- Conditional AUROC .375 vs constant .5625, SRG .25, magnitude .875 on8perturbations;
  INCONCLUSIVE_SMALL_SMOKE, NOT GO. Do not run full benchmark or tune automatically.
- Real positive score9.9068944e-5 => loss0; score gradient only temporaryTask4A/B.
  EphemeralTask4removed; no optimizer step/newmodelcheckpoint. summaryreloadexact.
- Original no-geo dropout .05 preserved. Aux deterministic B A stopgrad(h) follows
  requested formula; no RNG draws. Checkpointed/trainer/dropout unit tests pass.
- No original v1/TPM source changes. Final report IMPLEMENTATION_REPORT.md contains
  exact future gate/pilot commands and limitations. Localcopies same relativepaths.
- Final test target21RCWP+22TPM+16versions+5evalcontract+2lowertriangular=66;
  rerun log `/remote-home/wangbomin/MedPRISM_v2_1_RCWP_cpu_tests_final_20260909.log`.

## Historical in-progress notes (superseded by above)

Authoritative repo `/root/MedAgentCL_v4`. No existing source modified; additions
under `med_prism/rcwp`, `scripts/medicalskill_v2_1_rcwp`, `tests/med_prism_rcwp`.
Do not train a pilot automatically. User authorized unit tests and tiny GPU smoke.

Implemented core moments/fold/signed-net loss, factor-bound sidecars, isolated
trainer subclass, all-valid-token boundary backward, real field-quality wrapper,
gated three-task orchestration. Original v1.2 runtime config remains a compatibility
detail; independent state/contract and RCWP trace identify method 2.1. TPM unused.

First tests: RCWP13 PASS2.437s; TPM22 PASS5.398s; version/A-isolation16 PASS2.243s.
Three additional integration tests added locally, need sync/run.

Current GPU0 tiny smoke PID76209:
`/remote-home/wangbomin/MedPRISM_v2_1_RCWP_smoke_20260909_040640_10995`
Log samepath + `.log`; 4 fit+2 holdout Task3 train only. No optimizer step. Temporary
Task4 bank is removed before final persistent invariants; no model checkpoint saved.
Need inspect outcome, fix scoped wrapper if necessary, run final tests/report.

Remaining review: grouped fit/holdout separation, source/summary lifecycle resume
checks; field-quality thresholds explicitly INCONCLUSIVE for smoke. Full formal
128+64 gate and CL benefit NOT validated. Do not misreport smoke as empirical GO.

SSH bridge `.work/tpm_v2/remote_bridge.py` supports isolated new paths. Current
session72117; oldsession65220 read-only also exists. Never persist credentials.
