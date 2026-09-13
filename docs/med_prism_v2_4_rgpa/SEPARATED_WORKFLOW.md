# Training-first continuation (2026-09-13)

New operational entry: `scripts/medicalskill_v2_4_rgpa/run_separated.sh`.
Do not use the old interleaved `run.sh` for this continuation.
The frozen RGPA algorithm, old sidecars, checkpoints and contract were not edited.

Changes: new `med_prism/rgpa/separated.py`, shell entry, and four regression tests.
All 13 RGPA tests passed. The fixed method source hashes still match the original contract.

Task3 uses GPU0, batch1, accumulation16 (effective batch16). Task1/2 are not retrained.
After Task3 training and its source-task boundary, evaluate six lower-triangle cells:
R31/R32/R33 first, then R21/R22/R11. Reuse the original cell generation and metrics.
Set MED_PRISM_DATA_ROOT explicitly to the manifest's lite-10k1k root; verify test hashes,
1000 examples per task, cumulative banks and prediction hashes. Never reuse the old full-test cells.
GPU0/1 workers take independent cells from one queue; occupied GPUs wait, no process is killed
and no GPU memory is reserved. Parallel use is conditional on GPU1 becoming available.

Current run root:
`/remote-home/wangbomin/MedPRISM_RGPA_A_20260912_1540`

New output subdirectory:
`evaluation_lite_20260913_train3_gpu0/`

`workflow.log` and `train_task3.log` show progress. `training_complete.json` marks training
and boundary completion; `completion_all.json` is PASS/FAIL for the whole workflow.
Final results: `summary.json`, `REPORT.md`, plus per-cell summaries and predictions.
Training checkpoints stay under the original root's `med_prism/stage_03/`.

Separate manual modes (do not start alongside the already-running workflow):

```bash
ROOT=/remote-home/wangbomin/MedPRISM_RGPA_A_20260912_1540
OUT="$ROOT/evaluation_lite_20260913_train3_gpu0"
bash /root/MedAgentCL_v4/scripts/medicalskill_v2_4_rgpa/run_separated.sh --root "$ROOT" --output "$OUT" --mode train
bash /root/MedAgentCL_v4/scripts/medicalskill_v2_4_rgpa/run_separated.sh --root "$ROOT" --output "$OUT" --mode eval
```

Default `--mode all` runs those two phases in order. No Stage0, oracle, Candidate B or T4/T5.
