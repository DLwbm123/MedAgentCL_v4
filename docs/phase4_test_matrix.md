# Phase 4 Test Matrix

| Area | Verification | Expected |
|---|---|---|
| Rank-1 forward | Explicit base + sum of experts | Exact tensor equality |
| Cumulative tasks | Task 1 + Task 2 forward | Both contributions present |
| Scaling | E=1, E=4, E=16 | Per-task alpha/E, no later drift |
| Targets | Toy Qwen3VLTextModel, 36 layers | 36 q + 36 v, vision/merger 0 |
| Orthogonality | Fixed tensors | Squared/RMS match formula |
| Orth gradient | Task 2 backward | Current nonzero, old absent |
| Task 1 orth | No old bank | Exact zero |
| Freezing | Optimizer step | Current changes, old bitwise unchanged |
| Checkpoint | Save/reload | Explicit A/B tensors round-trip exactly |
| Rejection | Revision/target/schema/scaling | Fail closed |
| Swift plugin | med_prism_rank1 | Project tuner and trainer selected |
| Native LoRA | tuner_type=lora | Stock Swift trainer unchanged |
| Real smoke Task 1 | Qwen3-VL, 5 steps | Exit 0, checkpoint and audits |
| Real smoke Task 2 | Load Task 1, 5 steps | Old frozen, orth gradient nonzero |
| Real reload | Load Task 2 into fresh Qwen3-VL | 72 wrappers, exact tensor hashes |

CPU command:

    /root/anaconda3/envs/medagentcl_v4/bin/python -m unittest discover -s tests/phase4 -v

Full smoke command:

    bash /root/MedAgentCL_v4/scripts/phase4/run_phase4_rank1_smoke.sh
