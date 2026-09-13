# MedicalSkill-CL-v1

MedicalSkill-CL-v1 is a five-skill continual-learning dataset for Qwen3-VL.

1. `task_01_vqa`
2. `task_02_diagnosis_classification`
3. `task_03_concept_recognition`
4. `task_04_visual_grounding`
5. `task_05_reasoning_vqa`

Each task contains only `train.jsonl` and `test.jsonl`. No validation split is used. Test membership is preserved from the validated source tasks. Training records are closed against test lineage and cross-task training duplication before release.

The concept task combines both MedTrinity source groups and uses deterministic caption-text-only concept extraction with `Qwen/Qwen3-8B` at revision `b968826d9c46dd6066d109eabc6255188de91218`. Images are not passed to the extraction model.

MedSG training data is sampled by official source row, without replacement, with all turns from each selected source row retained. The eight official subtasks contribute approximately equal output-turn counts.

Formal training uses one epoch and no validation split:

```bash
CUDA_VISIBLE_DEVICES=0 /root/MedAgentCL_v4/scripts/skill_incremental_v1/train_one_epoch_task.sh TASK_ID
```

Evaluation is performed directly on each task's `test.jsonl`.
