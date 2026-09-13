# Med-PRISM v2.2 MS-CRC finite-forward gate



Implemented: isolated boundary hook/fold/search/export/lock/panel. Trainer and A-only loss unchanged; TPM/RCWP OFF.

Mechanically verified: CPU tests plus separate real-model smoke. Completion PASS means engineering completion, NOT a method-performance PASS.

Gate status: **NO_SIGNAL**



Task3 binary correctness is exact normalized concept-set match (existing parser all_fp=all_fn=0). Official concept micro-F1 remains separately reported. Never interpret micro-F1 as accuracy.

Fixed Task3 train128+64 only; no data expansion; P2 requires32/16 witnesses. Shortlist uses only feasible, strictly improving single-OFF fit candidates; coordinate search uses at most2 sweeps. Holdout requires unchanged-or-better accuracy, CE<=full+.02, no lost N, and strictly improved lexicographic objective for nonidentity.

Teacher-forced KL uses H=base+S3+P1+P2 as reference, NOT the Task2 boundary teacher with S2. S2 was not loaded. KL is diagnostic only.

Same coefficient vectors in the fixed panel are exact aliases evaluated once per split; all distinct vectors use new full-network forward/generation. No linear sum of leave-one-out effects is used.



## Key gate quantities

{
  "status": "NO_SIGNAL",
  "selection_status": "NO_SIGNAL",
  "task2_gain": {
    "delta": 0.0,
    "CI95": [
      0.0,
      0.0
    ],
    "bootstrap": "paired examples, 2000 draws, seed2209; not a patient-level independence guarantee"
  },
  "task1_safety": {
    "delta": 0.0,
    "CI95": [
      0.0,
      0.0
    ],
    "bootstrap": "paired examples, 2000 draws, seed2209; not a patient-level independence guarantee"
  },
  "task3_safety": {
    "delta": 0.0,
    "CI95": [
      0.0,
      0.0
    ],
    "bootstrap": "paired examples, 2000 draws, seed2209; not a patient-level independence guarantee"
  },
  "matched_global": "global_1.0",
  "matched_current_close": true,
  "vs_matched_global": {
    "delta": 0.0,
    "CI95": [
      0.0,
      0.0
    ],
    "bootstrap": "paired examples, 2000 draws, seed2209; not a patient-level independence guarantee"
  },
  "vs_permutation": {
    "delta": 0.0,
    "CI95": [
      0.0,
      0.0
    ],
    "bootstrap": "paired examples, 2000 draws, seed2209; not a patient-level independence guarantee"
  },
  "current_positive_transfer_evaluable": false,
  "current_positive_transfer_lost": 0,
  "task3_micro_f1_safety": {
    "delta": 0.0,
    "CI95": [
      0.0,
      0.0
    ]
  },
  "permutation_current_matched": true,
  "old_positive_transfer_lost": 0,
  "old_forgotten_restored": 0,
  "oracle_panel_label": "off_07",
  "oracle_panel_task2_accuracy": 0.46484375,
  "oracle_is_NOT_MS_CRC_score": true,
  "cross_distribution_selection_signal_failed": true,
  "precision_note": "Diagnostic dev only; paired CIs do not prove unseen benchmark generalization"
}



## Answers

1. Current-only witness selection supports lower Task2 forgetting only if Task2 gain and its paired uncertainty meet the pre-registered checks; see gate_summary and finite_panel_task2.

2. Task3 exact-set accuracy, official micro-F1, answer-token CE and positive-transfer losses are reported independently; empty N is NOT evidence of preserved transfer.

3. Rank-group value is supported only if the selected vector beats a current-performance-matched global alpha and the fixed multiset permutation with adequate precision. NO_GO_SELECTIVITY/INCONCLUSIVE is not a success.

The fixed-panel oracle envelope is diagnostic only and never changes the exported vector. If a current/safety-feasible oracle is good but locked g is poor, the cross-distribution selection signal failed.

Paired bootstrap is by example (2000 draws, seed2209); correlated patients/groups and diagnostic-dev exposure limit generalization. No formal unseen benchmark or fresh continual-learning training was run.



Files: witness_summary.json, search_trace.jsonl, fit_summary.json, holdout_summary.json, selection_lock.json, selected_coefficients.json, exported/state.json, finite_panel_current.json, finite_panel_task2.json, finite_panel_task1.json, global_scaling_comparison.json, random_permutation_comparison.json, gate_summary.json, invariants.json.

Result root: `/remote-home/wangbomin/MedPRISM_v2_2_MS_CRC_gate_20260909_145121_3757`
