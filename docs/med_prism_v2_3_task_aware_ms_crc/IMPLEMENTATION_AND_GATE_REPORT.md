# Med-PRISM v2.3 Task-Aware MS-CRC

{
  "status": "UTILITY_REJECTED",
  "holdout_status": "UTILITY_REJECTED",
  "task2_gain": {
    "delta": 0.0,
    "CI95": [
      0.0,
      0.0
    ],
    "bootstrap": "paired examples, 2000 draws, seed2209; not a patient-level independence guarantee"
  },
  "selected_vs_global": {
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
  "nonuniform_selected": false,
  "validation_levels": {
    "implemented": true,
    "mechanically_verified": true,
    "atomic_observability": true,
    "current_only_nonidentity": true,
    "holdout_replication": false,
    "development_retention_direction": false,
    "selectivity_over_global": false,
    "formal_unseen_continual_learning_validation": "NOT RUN"
  }
}

## Atomic support and concentration

{
  "status": "SUPPORT_WITH_REPAIR_OPPORTUNITY",
  "q": 0.3044534412955466,
  "omega_gold": 1.6955465587044534,
  "omega_FP": 0.3044534412955466,
  "support_mass": 755.587044534413,
  "support_atoms_total": 2331,
  "samples_with_support": 127,
  "ESS": 76.81682936066704,
  "max_sample_support_share": 0.030449552590687438,
  "unique_supported_gold_concepts": 21,
  "unique_supported_FP_suppressions": 1199,
  "top_concepts_by_support_mass": [
    [
      "gold:heart",
      8.477732793522266
    ],
    [
      "FP:pathology",
      7.306882591093116
    ],
    [
      "gold:histopathology",
      6.7821862348178135
    ],
    [
      "FP:ct scan",
      5.784615384615384
    ],
    [
      "FP:biopsy",
      5.480161943319837
    ],
    [
      "FP:tumor",
      5.480161943319837
    ],
    [
      "FP:diagnosis",
      5.175708502024291
    ],
    [
      "FP:staining",
      5.175708502024291
    ],
    [
      "gold:abdomen",
      5.08663967611336
    ],
    [
      "FP:adenocarcinoma",
      4.871255060728744
    ],
    [
      "FP:cancer",
      4.871255060728744
    ],
    [
      "FP:tissue",
      4.566801619433198
    ],
    [
      "FP:medical imaging",
      4.566801619433198
    ],
    [
      "FP:histology",
      4.2623481781376515
    ],
    [
      "FP:inflammation",
      4.2623481781376515
    ],
    [
      "FP:metastasis",
      4.2623481781376515
    ],
    [
      "FP:lesion",
      4.2623481781376515
    ],
    [
      "FP:oncology",
      3.9578947368421047
    ],
    [
      "gold:dermoscopy",
      3.3910931174089067
    ],
    [
      "gold:lung",
      3.3910931174089067
    ]
  ],
  "overlap_between_banks": 114,
  "full": {
    "TP": 188,
    "FP": 436,
    "FN": 423,
    "utility": 0.3044534412955466,
    "parser_valid": true,
    "CE": 1.1231919594285833,
    "L_mem": 0.045574666452338874,
    "G": 971.1514170040559,
    "G_full": 971.1514170040559,
    "D_PT": 0.0,
    "U_PT": 0.0,
    "L_PT": 0.0,
    "PT_status": "EVALUABLE",
    "memory_supported_atoms_restored": 0,
    "full_positive_transfer_atoms_lost": 0,
    "new_positive_transfer_atoms_added": 0,
    "per_bank_candidate_mem_loss": {
      "1": 0.13209687210454008,
      "2": 0.03704495986394921
    }
  },
  "per_bank": {
    "1": {
      "positive_support_atoms": 28,
      "fp_suppression_atoms": 510,
      "support_mass": 202.74655870445346,
      "exposed_harm": 0.13209687210454008
    },
    "2": {
      "positive_support_atoms": 18,
      "fp_suppression_atoms": 1889,
      "support_mass": 605.6323886639676,
      "exposed_harm": 0.03704495986394921
    }
  }
}

## Fit selection and holdout

{
  "label": "off_07",
  "g_fit": [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0
  ],
  "status": "FIT_SELECTED",
  "current_only": true
}

{
  "status": "UTILITY_REJECTED",
  "g_fit": [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0
  ],
  "final_g": [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0
  ],
  "candidate": {
    "TP": 92,
    "FP": 231,
    "FN": 190,
    "utility": 0.30413223140495865,
    "parser_valid": true,
    "CE": 1.1034964515765509,
    "L_mem": 0.03200821021241988,
    "G": 538.5916398713667,
    "G_full": 550.1672025723308,
    "D_PT": 14.520900321543408,
    "U_PT": 2.945337620578778,
    "L_PT": 0.026393613166416145,
    "PT_status": "EVALUABLE",
    "memory_supported_atoms_restored": 3,
    "full_positive_transfer_atoms_lost": 20,
    "new_positive_transfer_atoms_added": 5,
    "per_bank_candidate_mem_loss": {
      "1": 0.11545166402535657,
      "2": 0.02449368380468555
    }
  },
  "baseline": {
    "TP": 98,
    "FP": 242,
    "FN": 184,
    "utility": 0.31511254019292606,
    "parser_valid": true,
    "CE": 1.099019412000974,
    "L_mem": 0.027883740293941094,
    "G": 550.1672025723308,
    "G_full": 550.1672025723308,
    "D_PT": 0.0,
    "U_PT": 0.0,
    "L_PT": 0.0,
    "PT_status": "EVALUABLE",
    "memory_supported_atoms_restored": 0,
    "full_positive_transfer_atoms_lost": 0,
    "new_positive_transfer_atoms_added": 0,
    "per_bank_candidate_mem_loss": {
      "1": 0.10245641838351822,
      "2": 0.02449368380468555
    }
  },
  "global_status": "IDENTITY",
  "global_candidate": {
    "TP": 98,
    "FP": 242,
    "FN": 184,
    "utility": 0.31511254019292606,
    "parser_valid": true,
    "CE": 1.099019412000974,
    "L_mem": 0.027883740293941094,
    "G": 550.1672025723308,
    "G_full": 550.1672025723308,
    "D_PT": 0.0,
    "U_PT": 0.0,
    "L_PT": 0.0,
    "PT_status": "EVALUABLE",
    "memory_supported_atoms_restored": 0,
    "full_positive_transfer_atoms_lost": 0,
    "new_positive_transfer_atoms_added": 0,
    "per_bank_candidate_mem_loss": {
      "1": 0.10245641838351822,
      "2": 0.02449368380468555
    }
  },
  "global_final": [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0
  ],
  "no_reranking": true
}

Task2 dev was already unblinded in v2.2. This is development/mechanism validation, not blind or formal benchmark evidence. CIs resample paired examples, not atoms; patient dependence and small samples limit inference. Current sample caches were removed before old dev. TPM/RCWP/training OFF.

Output: /remote-home/wangbomin/MedPRISM_v2_3_TaskAware_MS_CRC_gate_20260910_142249_12471
