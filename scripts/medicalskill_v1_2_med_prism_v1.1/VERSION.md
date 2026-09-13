# Med-PRISM v1.1: Shared-Fix

Method version 1.1 for the MedicalSkill-CL v1.2 dataset.

Locked changes relative to v1.0:

- shared/private AdamW parameter groups use learning rates 1e-5 and 1e-4;
- the legacy shared gradient hook is disabled;
- shared drift optimizes the globally normalized effective residual `scaling * B @ A`;
- Task 1 effective drift is exact differentiable zero;
- geometric private orthogonality remains RMS with lambda 0.1;
- key isolation is disabled;
- Primary remains base + latest shared once + every learned private bank.

Stage 0 and Oracle are disabled by default for formal evaluation. Pilot selection uses only locked training splits and seed 42.
