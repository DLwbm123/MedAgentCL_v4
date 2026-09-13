# Phase 2 Model Smoke Runbook

Phase 2 scripts live in `scripts/phase2/`. They enforce this order:

1. Inspect cache and the model manifest.
2. Refuse download when the cache filesystem lacks required space.
3. Download each fixed immutable revision with `snapshot_download`.
4. Resolve the fixed snapshot and use `local_files_only=True` for every model,
   tokenizer, and processor load.
5. Run Qwen3 text and Qwen3-VL in separate Python processes on one GPU.
6. Stop after Qwen3-VL inference and module inventory; do not start Phase 3.

Fixed revisions resolved for this closure:

- `Qwen/Qwen3-8B`: `b968826d9c46dd6066d109eabc6255188de91218`
- `Qwen/Qwen3-VL-8B-Instruct`: `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`

The current server is blocked before download because both snapshots are absent,
the cache filesystem reports zero available bytes, and server-side access to
`huggingface.co:443` is unreachable. No cache was removed or relocated.
