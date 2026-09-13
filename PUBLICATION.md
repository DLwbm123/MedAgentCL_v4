# Publication scope and provenance

- Source: `/root/MedAgentCL_v4` on the user-specified port-20032 server.
- Snapshot date: 2026-09-13.
- Source working tree: modified and untracked files were included; source HEAD was `a0e0142`.
- Baseline upstream attribution: see `UPSTREAM_MS_SWIFT.md` and `LICENSE`.
- Latest run: `/remote-home/wangbomin/MedPRISM_RGPA_A_20260912_1540`.
- Latest evaluated output: `evaluation_lite_20260913_train3_gpu0` under that run.
- Latest evidence is copied under `reports/latest_rgpa/` with its original relative layout.
- Included: source, scripts, configuration, existing tests, environment locks, documentation,
  RGPA contract/run manifest, aggregate source-boundary scores, timing, final report and six cell summaries.
- Excluded: original `.git` history, upstream GitHub automation, caches, build metadata,
  temporary upload directories, backups, old artifact/output collections, datasets, row-level
  medical inputs/answers/predictions, weights, optimizer checkpoints, large runtime outputs and logs.
- No training was started, stopped, resumed or rerun for publication. Server files were not changed.
- Existing report hashes are preserved as recorded provenance; no new dataset/model hashes were computed.
- Dataset/model paths in source and evidence refer to the original server. Reproduction requires
  separately obtained datasets and model weights and appropriate path configuration.
- Original implementation/workflow documents are historical; use the final report for results
  and `training_time.json` for the completed Task3 GPU configuration.
- Original source ancestry is recorded as provenance only; the new snapshot commit does not
  claim the upstream tag as a Git ancestor.
