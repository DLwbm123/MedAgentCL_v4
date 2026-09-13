# Phase 3 Native LoRA Smoke

This phase validates the fixed-revision Qwen3-VL and ms-swift 4.4.1 native PEFT LoRA path only.

## Boundaries

- Backbone: `Qwen/Qwen3-VL-8B-Instruct`
- Revision: `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- Targets: language-model `q_proj` and `v_proj` only
- LoRA: rank 48, alpha 96, dropout 0.05
- Single GPU, BF16, SDPA, 10 optimizer steps
- Vision tower and merger/aligner remain frozen
- No Med-PRISM, rank-1, shared/private, Octopus, DeepSpeed, vLLM, or flash-attn

Runtime artifacts are written only to `output/phase3_native_lora` and are not committed.

## Commands

Prepare data and target audits with `scripts/phase3/native_lora_tools.py`, then run:

```bash
bash scripts/phase3/train_phase3_native_lora.sh
```

After training, use the `adapter`, `infer`, `aggregate`, and `finalize` subcommands in
`native_lora_tools.py`. The inference subcommand must be invoked in separate Python
processes for the base model and each adapter reload.
