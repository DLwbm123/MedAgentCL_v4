# Med-PRISM v1.2: Shared-Fix + Rank-1 Key Isolation

Method version 1.2 for the MedicalSkill-CL v1.2 dataset. It inherits every v1.1 behavior and adds only all-history, same-layer RMS cosine isolation over rank-1 private A keys.

Pilot default `lambda_key=0.1` is provisional. Supported later pilot values are 0.03, 0.1, and 0.3. Geometric orthogonality remains fixed at 0.1 and effective shared drift remains 0.01 for the first controlled comparison.

No router, task ID, replay, feature memory, private normalization, expert selection, or inference-time module is introduced. Primary cumulative composition is unchanged. Stage 0 and Oracle are disabled by default for formal evaluation; Oracle is available only as an explicit pilot diagnostic.
