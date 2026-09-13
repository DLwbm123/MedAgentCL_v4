# v2.3 interference diagnostic — 2026-09-11

## Scope and status

New code only: `med_prism/ms_crc/interference_diagnostic/{__init__,statistics,splits,runner}.py`, `tests/med_prism_ms_crc/test_interference_diagnostic.py`, and `scripts/medicalskill_v2_3_task_aware_ms_crc/run_interference_diagnostic.sh`.

No changes to existing evaluator, parser, candidate construction, atomic support, v2.3 selector, training, TPM, RCWP, locked artifacts or checkpoint weights. No commit/push.

CPU tests: 68 passed (57 existing, 11 new). Latest captured run: 19.396 seconds. Independent preflight passed both original locks and exact full/H historical summary reproduction.

Output root:

`/remote-home/wangbomin/MedPRISM_v2_3_interference_diagnostic_20260911_181431`

Source v2.3:

`/remote-home/wangbomin/MedPRISM_v2_3_TaskAware_MS_CRC_gate_20260910_142249_12471`

Local partial results:

`D:/postgraduate/med_prism/results/v2_3/interference_diagnostic_20260911_181431`

## Run / explicit resume

The first launch uses nohup with this command. Do not start a duplicate controller. Check process state and completion first.

```bash
cd /root/MedAgentCL_v4
OUT=/remote-home/wangbomin/MedPRISM_v2_3_interference_diagnostic_20260911_181431
nohup bash scripts/medicalskill_v2_3_task_aware_ms_crc/run_interference_diagnostic.sh run \
  --output-root "$OUT" > "$OUT/controller.log" 2>&1 < /dev/null &
```

For an interrupted run, after confirming the original controller and both workers have exited, use the same `run` command, appending to the controller log (`>>`) instead of replacing it. Valid per-composition caches have SHA sidecars and are reused; finished workers are skipped. Input/code hashes must still match. Do not modify the frozen plan or diagnostic code while this run is active.

GPU0/GPU1 have separate logs, PIDs and worker outputs. Startup requires at least 22000 MiB free on each pending GPU; other users' processes are never stopped. A controller file lock prevents duplicate controllers. No unattended rerun after failure is configured.

The launcher preserves the original v2.3 HF_HOME/HF_HUB_CACHE and offline mode. Initial PIDs 88018/88162/88163 were explicitly stopped before inference because the first direct launch omitted these environment variables; the corrected launch appends to logs and uses the same untouched diagnostic plan. Use the current gpu*_pid.json files, not those superseded PIDs.

## Work remaining after launch

28 fixed Task3 dev256 compositions (25 original candidates plus H/minus1/minus2), partitioned over GPU0/1. Same original dev order, batch4, BF16/SDPA, greedy settings and parser. Every candidate's dev micro-F1/counts/CE must reproduce v2.3 metric consistency results. Each worker checks all persistent tensor hashes at completion.

Then automatically aggregate 10 grouped fit128/holdout64 splits (2209..2218), using unchanged `AtomicSupport/select/accept`. All predictions are generated once in fixed original batches independently of selection. This is cached-forward resampling on an overlapping development pool, not ten independent inference trials or an unseen benchmark. Fit-only decisions are saved before selected/scalar/identity/prespecified off_07 holdout scoring. No holdout reranking.

The original Task3 fit128+holdout64 and v2.3 identity deployment remain untouched.

Historical results reuse source-identical v2.2 Task1/Task2 dev single-group ablations, with hashes, settings, sample counts and full/H summaries checked against v2.3. They never feed the repeated-split selector.

## Available preliminary results

- Memory-repair Spearman: Task1 -0.0123; Task2 0.5438.
- Task1 vs Task2 repair-vector Spearman 0.2689, Pearson 0.2500.
- Task2 off_07 gain +5.8594 pp (historical rank1); Task1 gain0.
- Task2 positive/negative groups15/1. Positive-repair HHI0.08843; effective groups11.31.
- Top3 groups7,4,5; descriptive summed gain14.8438 pp, NOT joint suppression effect.
- Preliminary evidence favors task-conditioned interference (CASE B), but repeated-split stability and final conclusions are still pending.

## Completion check

```bash
OUT=/remote-home/wangbomin/MedPRISM_v2_3_interference_diagnostic_20260911_181431
test -f "$OUT/completion.json" && cat "$OUT/completion.json"
tail -20 "$OUT/controller.log"
cat "$OUT/gpu0_pid.json" "$OUT/gpu1_pid.json"
```

`completion.json` PASS means diagnostic completion, not a method performance PASS. `DIAGNOSTIC_REPORT.md` currently clearly says PARTIAL and will be replaced **only in this new diagnostic root** by the completed report. `failure.json` may be stale after a successful explicit resume; inspect timestamps and completion.

Key final outputs: `DIAGNOSTIC_REPORT.md`, `historical_interference_matrix.json/.csv`, `group_signal_vs_historical_interference.csv`, `signal_correlations.json`, `task2_repair_decomposition.json`, `repeated_split_selection.json/.csv`, `decision.json`, `provenance.json`, `invariants.json`, and six PNG figures.

Do not label the run complete or make claims about repeated-split stability until these outputs actually exist and pass verification. No new method, training or repair is authorized.
