#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from huggingface_hub import snapshot_download
import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM


MODEL_ID = "Qwen/Qwen3-8B"


def nvidia_state() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free", "--format=csv,noheader"], text=True
    )


def first_stop(ids: list[int], eos_ids: set[int], pad_id: int | None) -> tuple[list[int], int | None, bool]:
    for index, token_id in enumerate(ids):
        if token_id in eos_ids:
            return ids[:index + 1], index, True
    if pad_id is not None:
        while ids and ids[-1] == pad_id:
            ids.pop()
    return ids, None, False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "huggingface" / "hub")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    before = nvidia_state()
    snapshot = snapshot_download(
        MODEL_ID, revision=args.revision, cache_dir=str(args.cache_dir), local_files_only=True
    )
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, revision=args.revision, local_files_only=True)
    tokenizer.padding_side = "left"
    model = Qwen3ForCausalLM.from_pretrained(
        snapshot,
        revision=args.revision,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device).eval()
    load_seconds = time.perf_counter() - load_start
    torch.cuda.reset_peak_memory_stats()

    cases = [
        {"id": "single", "prompt": "Return exactly one short sentence describing pulmonary edema."},
        {"id": "batch_a", "prompt": "Name one imaging feature of pleural effusion in one short sentence."},
        {"id": "batch_b", "prompt": "Name one imaging feature of pneumothorax in one short sentence."},
        {
            "id": "medical_concept_json",
            "prompt": (
                "Extract concepts from this caption and return only a strict JSON list of strings: "
                "Chest radiograph shows bilateral perihilar airspace opacities and a small right pleural effusion."
            ),
        },
    ]
    rendered = []
    for case in cases:
        messages = [{"role": "user", "content": case["prompt"]}]
        rendered.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        ))
    encoded = tokenizer(rendered, return_tensors="pt", padding=True).to(args.device)
    generation_config = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
    infer_start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**encoded, **generation_config)
    infer_seconds = time.perf_counter() - infer_start

    tokenizer_eos = tokenizer.eos_token_id
    model_eos = model.generation_config.eos_token_id
    eos_ids = set(model_eos if isinstance(model_eos, list) else [model_eos])
    eos_ids.discard(None)
    if tokenizer_eos is not None:
        eos_ids.add(tokenizer_eos)
    pad_id = model.generation_config.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.pad_token_id
    input_width = encoded.input_ids.shape[1]
    effective_input_lengths = encoded.attention_mask.sum(dim=1).tolist()
    rows = []
    raw_rows = []
    token_audit = []
    for index, case in enumerate(cases):
        all_new_ids = generated[index, input_width:].tolist()
        effective_ids, eos_index, eos_seen = first_stop(list(all_new_ids), eos_ids, pad_id)
        decoded_with_special = tokenizer.decode(effective_ids, skip_special_tokens=False)
        raw = tokenizer.decode(effective_ids, skip_special_tokens=True)
        parsed = None
        parse_error = None
        if case["id"] == "medical_concept_json":
            try:
                parsed = json.loads(raw.strip())
            except Exception as exc:
                parse_error = repr(exc)
        human_prefix = raw.lstrip().lower().startswith("human:")
        prompt_leaked = rendered[index] in raw or case["prompt"] in raw
        row = {
            "id": case["id"],
            "prompt": case["prompt"],
            "raw_output": raw,
            "output": raw.strip(),
            "effective_input_tokens": int(effective_input_lengths[index]),
            "padded_input_width": int(input_width),
            "output_tokens": len(effective_ids),
            "generated_tensor_width": len(all_new_ids),
            "eos_seen": eos_seen,
            "eos_generated_index": eos_index,
            "eos_within_64_tokens": bool(eos_seen and eos_index is not None and eos_index < 64),
            "padding_trimmed_from_output": len(all_new_ids) - len(effective_ids),
            "prompt_leaked_into_output": prompt_leaked,
            "human_prefix_present": human_prefix,
            "thinking_block_present": "<think>" in raw or "</think>" in raw,
            "parsed_json": parsed,
            "json_parse_error": parse_error,
        }
        rows.append(row)
        raw_rows.append({"id": case["id"], "prompt": case["prompt"], "raw_output": raw, "raw_decode_with_special_tokens": decoded_with_special})
        token_audit.append({
            "id": case["id"],
            "full_generated_suffix_ids": all_new_ids,
            "effective_generated_ids": effective_ids,
            "eos_token_ids": sorted(eos_ids),
            "pad_token_id": pad_id,
            "eos_seen": eos_seen,
            "eos_index": eos_index,
        })

    json_parseable = rows[-1]["parsed_json"] is not None
    unexpected_human = any(row["human_prefix_present"] for row in rows[:-1])
    prompt_leakage = any(row["prompt_leaked_into_output"] for row in rows)
    nonempty = all(row["output"] for row in rows)
    status = "PASS" if nonempty and json_parseable and not unexpected_human and not prompt_leakage else "PARTIAL"
    audit = {
        "status": status,
        "model_id": MODEL_ID,
        "revision": args.revision,
        "snapshot": snapshot,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "padding_side": tokenizer.padding_side,
        "decode_scope": "generated suffix only",
        "generation_config": generation_config,
        "special_tokens": {
            "tokenizer_eos_token_id": tokenizer.eos_token_id,
            "tokenizer_pad_token_id": tokenizer.pad_token_id,
            "tokenizer_bos_token_id": tokenizer.bos_token_id,
            "model_eos_token_id": model.generation_config.eos_token_id,
            "model_pad_token_id": model.generation_config.pad_token_id,
            "model_bos_token_id": model.generation_config.bos_token_id,
        },
        "effective_input_lengths": [int(value) for value in effective_input_lengths],
        "padded_input_width": int(input_width),
        "json_parseable": json_parseable,
        "json_eos_seen": rows[-1]["eos_seen"],
        "json_eos_within_64_tokens": rows[-1]["eos_within_64_tokens"],
        "unexpected_human_prefix": unexpected_human,
        "human_prefix_origin": "generated_suffix" if unexpected_human else "absent",
        "prompt_leakage": prompt_leakage,
        "all_output_token_counts_effective": all(row["output_tokens"] <= row["generated_tensor_width"] for row in rows),
        "load_seconds": load_seconds,
        "inference_seconds": infer_seconds,
    }
    for name, payload in (
        ("qwen3_text_generation_fixed.jsonl", rows),
        ("qwen3_text_smoke.jsonl", rows),
        ("qwen3_text_raw_outputs.jsonl", raw_rows),
    ):
        (args.output_dir / name).write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in payload), encoding="utf-8")
    write_templates = []
    for case, template in zip(cases, rendered):
        write_templates.extend([f"===== {case['id']} =====", template, ""])
    (args.output_dir / "qwen3_rendered_templates.txt").write_text("\n".join(write_templates), encoding="utf-8")
    (args.output_dir / "qwen3_generated_token_ids.json").write_text(json.dumps(token_audit, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "qwen3_text_generation_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    closure = f"""# Qwen3 Text Generation Closure

Status: **{status}**

- Chat templates were rendered with `add_generation_prompt=True` and `enable_thinking=False`.
- Batch padding side is `{tokenizer.padding_side}`; effective input lengths are `{audit['effective_input_lengths']}`.
- Decoding is restricted to the generated suffix. Output token counts stop at the first actual EOS and exclude later batch padding.
- Unexpected `Human:` prefix: `{unexpected_human}`; prompt leakage: `{prompt_leakage}`.
- Medical JSON parseable: `{json_parseable}`; EOS seen: `{audit['json_eos_seen']}`; EOS within 64 tokens: `{audit['json_eos_within_64_tokens']}`.
- If EOS is absent within 64 tokens, that fact is retained as model behavior rather than treated as an artificial stop.
"""
    (args.output_dir / "qwen3_text_generation_closure.md").write_text(closure, encoding="utf-8")
    memory = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": args.device,
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
        "nvidia_smi_before": before,
        "nvidia_smi_during": nvidia_state(),
        "cpu_offload": False,
        "oom": False,
    }
    (args.output_dir / "qwen3_text_memory.json").write_text(json.dumps(memory, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "qwen3_text_summary.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
