# Phase 4 Med-PRISM Rank-1 Smoke Report

- Status: **PASS**
- Qwen3-VL revision: 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
- ms-swift: 4.4.1
- Task 1 / Task 2: 5 optimizer steps each, one GPU, no DDP.

## Checks

- PASS: cpu_tests_21_pass
- PASS: both_swift_runs_exit_zero
- PASS: both_runs_use_project_trainer
- PASS: targets_exactly_72_language_qv
- PASS: task1_orth_is_zero
- PASS: task2_orth_gradient_nonzero
- PASS: old_experts_unchanged_and_zero_grad
- PASS: loss_identity_holds
- PASS: real_qwen_reload_pass
