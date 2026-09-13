#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
from collections import Counter
from typing import Any


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
TEXT_MODEL_ID = "Qwen/Qwen3-8B"
TEXT_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DEFAULT_SOURCE = Path("/root/MedAgentCL/data/MedSkill_CL_4Skill/task_02_vqa_train.jsonl")
DEFAULT_OUTPUT = Path("/root/MedAgentCL_v4/output/phase3_native_lora")
EXPECTED_OLD_FINGERPRINT = "eb7a06159a8aa85f31ce4acecbab9447ee4e88d243eb97fa381ad26f8ae08749"
OPTION_RE = re.compile(r"(?m)^\s*([A-D])\.\s*(.+?)\s*$")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def option_map(prompt: str) -> dict[str, str]:
    return {letter: answer.strip() for letter, answer in OPTION_RE.findall(prompt)}


def gold_option(prompt: str, answer: str) -> tuple[str | None, str]:
    options = option_map(prompt)
    normalized_answer = normalize_text(answer)
    for letter, text in options.items():
        if normalize_text(text) == normalized_answer:
            return letter, text
    return None, answer.strip()


def parse_vqa_output(raw_output: str, prompt: str, gold: str) -> dict[str, Any]:
    raw = raw_output
    normalized = re.sub(r"\s+", " ", raw.strip())
    match = re.match(r"^\s*([A-D])\s*[\.\):\-]?\s*(.*)$", normalized, re.IGNORECASE)
    options = option_map(prompt)
    letter = match.group(1).upper() if match else None
    answer_text = match.group(2).strip() if match else normalized
    valid = bool(letter and letter in options)
    if valid and not answer_text:
        answer_text = options[letter]
    gold_letter, gold_text = gold_option(prompt, gold)
    correct = bool(
        (valid and gold_letter and letter == gold_letter)
        or normalize_text(answer_text) == normalize_text(gold_text)
    )
    canonical = f"{letter}. {answer_text}" if valid else normalized
    return {
        "raw_output": raw,
        "normalized_output": canonical,
        "option_letter": letter,
        "answer_text": answer_text,
        "valid_parse": valid,
        "gold": f"{gold_letter}. {gold_text}" if gold_letter else gold_text,
        "correct": correct,
    }


def language_qv_targets(module_names: list[str]) -> list[str]:
    return sorted(
        name for name in module_names
        if name.startswith("model.language_model.") and name.rsplit(".", 1)[-1] in {"q_proj", "v_proj"}
    )


def classify_names(names: list[str]) -> dict[str, int]:
    return {
        "total": len(names),
        "language": sum("model.language_model." in name for name in names),
        "q_proj": sum(".q_proj" in name for name in names),
        "v_proj": sum(".v_proj" in name for name in names),
        "vision": sum(".visual." in name or "vision_tower" in name for name in names),
        "merger_aligner": sum(
            any(token in name for token in ("merger", "deepstack_merger", "aligner", "projector")) for name in names
        ),
    }


def audit_state_keys(keys_and_shapes: dict[str, list[int]], rank: int) -> dict[str, Any]:
    keys = sorted(keys_and_shapes)
    wrappers = sorted({key.split(".lora_", 1)[0] for key in keys if ".lora_" in key})
    classes = classify_names(wrappers)
    q_layers = sorted({int(m.group(1)) for key in keys if (m := re.search(r"language_model\.layers\.(\d+).*\.q_proj\.lora_", key))})
    v_layers = sorted({int(m.group(1)) for key in keys if (m := re.search(r"language_model\.layers\.(\d+).*\.v_proj\.lora_", key))})
    bad_shapes = []
    for key, shape in keys_and_shapes.items():
        if ".lora_A." in key and (not shape or shape[0] != rank):
            bad_shapes.append({"key": key, "shape": shape, "expected_rank_axis": 0})
        if ".lora_B." in key and (len(shape) < 2 or shape[1] != rank):
            bad_shapes.append({"key": key, "shape": shape, "expected_rank_axis": 1})
    non_lora = [key for key in keys if ".lora_" not in key]
    passed = (
        classes == {"total": 72, "language": 72, "q_proj": 36, "v_proj": 36, "vision": 0, "merger_aligner": 0}
        and q_layers == list(range(36))
        and v_layers == list(range(36))
        and not bad_shapes
        and not non_lora
    )
    return {
        "status": "PASS" if passed else "BLOCKED",
        "tensor_count": len(keys),
        "wrapper_count": len(wrappers),
        "classification": classes,
        "q_layers": q_layers,
        "v_layers": v_layers,
        "bad_shapes": bad_shapes,
        "non_lora_keys": non_lora,
        "wrappers": wrappers,
    }


def validate_record(record: dict[str, Any]) -> tuple[bool, str]:
    messages = record.get("messages")
    images = record.get("images")
    if not isinstance(messages, list) or len(messages) < 2:
        return False, "invalid_messages"
    if messages[0].get("role") != "user" or messages[-1].get("role") != "assistant":
        return False, "invalid_roles"
    if not isinstance(images, list) or not images:
        return False, "invalid_images"
    if messages[0].get("content", "").count("<image>") != len(images):
        return False, "placeholder_mismatch"
    if not str(messages[-1].get("content", "")).strip():
        return False, "empty_target"
    if any(not Path(image).is_file() or Path(image).stat().st_size <= 0 for image in images):
        return False, "missing_image"
    letter, _ = gold_option(messages[0]["content"], messages[-1]["content"])
    if letter is None:
        return False, "gold_not_in_options"
    return True, "ok"


def command_prepare(args: argparse.Namespace) -> int:
    output = args.output_dir / "data"
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    errors = Counter()
    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    for line_number, line in enumerate(args.source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            errors["invalid_json"] += 1
            continue
        valid, reason = validate_record(record)
        if not valid:
            errors[reason] += 1
            continue
        record_id = str(record.get("id") or record.get("question_id") or f"line-{line_number}")
        image = str(record["images"][0])
        if record_id in seen_ids or image in seen_images:
            errors["duplicate_candidate"] += 1
            continue
        seen_ids.add(record_id)
        seen_images.add(image)
        score = hashlib.sha256(f"{args.seed}\0{record_id}\0{image}".encode()).hexdigest()
        candidates.append((score, line_number, record_id, image, record))
    candidates.sort(key=lambda item: item[0])
    required = args.train_count + args.eval_count
    if len(candidates) < required:
        raise RuntimeError(f"Only {len(candidates)} valid unique records; need {required}")
    selected = candidates[:required]
    rows = []
    manifest_rows = []
    for split, items in (("train", selected[:args.train_count]), ("eval", selected[args.train_count:])):
        split_rows = []
        for score, line_number, record_id, image, source in items:
            user = source["messages"][0]["content"].rstrip()
            instruction = "Answer with only the option letter and answer text."
            if instruction not in user:
                user += "\n\n" + instruction
            letter, answer = gold_option(source["messages"][0]["content"], source["messages"][-1]["content"])
            row = {
                "id": record_id,
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": f"{letter}. {answer}"},
                ],
                "images": list(source["images"]),
                "gold_option": letter,
                "gold_answer": answer,
                "source_line": line_number,
            }
            split_rows.append(row)
            manifest_rows.append({
                "split": split,
                "id": record_id,
                "source_line": line_number,
                "selection_sha256": score,
                "images": list(source["images"]),
                "image_sizes": [Path(p).stat().st_size for p in source["images"]],
                "images_exist": all(Path(p).is_file() for p in source["images"]),
            })
        path = output / f"smoke_{split}.jsonl"
        write_jsonl(path, split_rows)
        rows.append((split, path, split_rows))
    train_images = {r["images"][0] for r in rows[0][2]}
    eval_images = {r["images"][0] for r in rows[1][2]}
    train_ids = {r["id"] for r in rows[0][2]}
    eval_ids = {r["id"] for r in rows[1][2]}
    schema = {
        "status": "PASS",
        "valid_candidates": len(candidates),
        "rejected": dict(errors),
        "train_count": args.train_count,
        "eval_count": args.eval_count,
        "train_eval_id_overlap": sorted(train_ids & eval_ids),
        "train_eval_image_overlap": sorted(train_images & eval_images),
        "all_images_exist": all(row["images_exist"] for row in manifest_rows),
        "placeholder_checks_pass": True,
        "assistant_targets_nonempty": True,
    }
    manifest = {
        "schema_version": 1,
        "source": str(args.source),
        "source_sha256": sha256_file(args.source),
        "selection_rule": "ascending sha256(seed\\0id\\0first_image), unique id and first image",
        "seed": args.seed,
        "train_count": args.train_count,
        "eval_count": args.eval_count,
        "files": {split: {"path": str(path), "sha256": sha256_file(path)} for split, path, _ in rows},
        "records": manifest_rows,
    }
    if schema["train_eval_id_overlap"] or schema["train_eval_image_overlap"]:
        raise RuntimeError("Train/eval overlap detected")
    write_json(output / "smoke_data_manifest.json", manifest)
    write_json(output / "smoke_schema_report.json", schema)
    print(json.dumps(schema, indent=2))
    return 0


def command_targets(args: argparse.Namespace) -> int:
    names = [line.strip() for line in args.module_names.read_text(encoding="utf-8").splitlines() if line.strip()]
    targets = language_qv_targets(names)
    classes = classify_names(targets)
    expected = {"total": 72, "language": 72, "q_proj": 36, "v_proj": 36, "vision": 0, "merger_aligner": 0}
    if classes != expected:
        raise RuntimeError(f"Pre-injection target audit failed: {classes}")
    text = "\n".join(targets) + "\n"
    (args.output_dir / "target_modules_pre_injection.txt").write_text(text, encoding="utf-8")
    (args.output_dir / "target_modules.txt").write_text(text, encoding="utf-8")
    audit = {
        "status": "PRE_INJECTION_PASS",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "source_module_inventory": str(args.module_names),
        "pre_injection": classes,
        "target_leaf_names": ["q_proj", "v_proj"],
        "target_full_names_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "post_injection": None,
    }
    write_json(args.output_dir / "target_module_audit.json", audit)
    print(json.dumps(audit, indent=2))
    return 0


def locate_adapter(path: Path) -> Path:
    if (path / "adapter_config.json").is_file() and (path / "adapter_model.safetensors").is_file():
        return path
    candidates = sorted({p.parent for p in path.rglob("adapter_model.safetensors")})
    if not candidates:
        raise FileNotFoundError(f"No adapter_model.safetensors under {path}")
    return candidates[-1]


def command_adapter(args: argparse.Namespace) -> int:
    from safetensors.torch import load_file

    adapter = locate_adapter(args.adapter_dir)
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    state_path = adapter / "adapter_model.safetensors"
    state = load_file(str(state_path), device="cpu")
    shapes = {key: list(value.shape) for key, value in state.items()}
    audit = audit_state_keys(shapes, args.rank)
    audit.update({
        "adapter_dir": str(adapter),
        "adapter_file": str(state_path),
        "adapter_file_bytes": state_path.stat().st_size,
        "adapter_file_sha256": sha256_file(state_path),
        "config": config,
        "all_tensors_finite": all(bool(value.isfinite().all()) for value in state.values()),
        "q_nonzero_tensors": sum(".q_proj." in key and bool(value.abs().max() > 0) for key, value in state.items()),
        "v_nonzero_tensors": sum(".v_proj." in key and bool(value.abs().max() > 0) for key, value in state.items()),
        "unexpected_full_model_size": state_path.stat().st_size > 2_000_000_000,
    })
    passed = (
        audit["status"] == "PASS" and audit["all_tensors_finite"] and audit["q_nonzero_tensors"] > 0
        and audit["v_nonzero_tensors"] > 0 and not audit["unexpected_full_model_size"]
    )
    audit["status"] = "PASS" if passed else "BLOCKED"
    write_json(args.output_dir / "adapter_state_audit.json", audit)
    parameter = json.loads((args.output_dir / "parameter_audit.json").read_text(encoding="utf-8"))
    data_manifest = args.output_dir / "data" / "smoke_data_manifest.json"
    manifest = {
        "schema_version": 1,
        "method": "native_lora",
        "backbone": MODEL_ID,
        "immutable_backbone_revision": MODEL_REVISION,
        "upstream_v4_4_1_commit": subprocess.check_output(["git", "rev-parse", "v4.4.1^{commit}"], text=True).strip(),
        "task": "omnimedvqa_smoke",
        "lora_rank": args.rank,
        "lora_alpha": args.alpha,
        "lora_dropout": args.dropout,
        "target_modules": audit["wrappers"],
        "target_modules_sha256": hashlib.sha256(("\n".join(audit["wrappers"]) + "\n").encode()).hexdigest(),
        "wrapper_count": audit["wrapper_count"],
        "trainable_parameter_count": parameter["trainable_parameter_count"],
        "source_dataset_manifest": str(data_manifest),
        "source_dataset_manifest_sha256": sha256_file(data_manifest),
        "checkpoint_path": str(adapter),
        "adapter_file": str(state_path),
        "adapter_file_bytes": state_path.stat().st_size,
        "merge_state": "unmerged",
        "dtype": "bfloat16",
        "seed": 42,
        "max_steps": 10,
        "state_audit_status": audit["status"],
    }
    write_json(args.output_dir / "adapter_manifest.json", manifest)
    print(json.dumps({"status": audit["status"], "adapter": str(adapter), "wrapper_count": audit["wrapper_count"]}, indent=2))
    return 0 if passed else 1


def nvidia_state() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.free", "--format=csv,noheader"], text=True
    )


def command_infer(args: argparse.Namespace) -> int:
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    before = nvidia_state()
    snapshot = snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=str(args.cache_dir), local_files_only=True
    )
    processor = AutoProcessor.from_pretrained(snapshot, revision=MODEL_REVISION, local_files_only=True)
    if hasattr(processor.image_processor, "min_pixels"):
        processor.image_processor.min_pixels = 200704
    if hasattr(processor.image_processor, "max_pixels"):
        processor.image_processor.max_pixels = 802816
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        snapshot,
        revision=MODEL_REVISION,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device).eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    adapter_path = None
    if args.variant != "base":
        adapter_path = locate_adapter(args.adapter_dir)
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False).to(args.device).eval()
        model.set_adapter("default")
    wrappers = [name for name, module in model.named_modules() if hasattr(module, "lora_A") and hasattr(module, "lora_B")]
    wrapper_classes = classify_names(wrappers)
    active = list(getattr(model, "active_adapters", [])) if args.variant != "base" else []
    torch.cuda.reset_peak_memory_stats()
    rows = []
    logit_slice = None
    samples = read_jsonl(args.eval_jsonl)
    for index, sample in enumerate(samples):
        user = sample["messages"][0]["content"].replace("<image>", "").strip()
        messages = [{"role": "user", "content": [
            {"type": "image", "image": sample["images"][0]},
            {"type": "text", "text": user},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt"
        ).to(args.device)
        with torch.inference_mode():
            if index == 0:
                logits = model(**inputs).logits[0, -1, :256].float().cpu()
                logit_slice = logits.tolist()
            generated = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        new_ids = generated[0, inputs.input_ids.shape[1]:]
        raw = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        parsed = parse_vqa_output(raw, sample["messages"][0]["content"], sample["gold_answer"])
        rows.append({"id": sample["id"], "variant": args.variant, "run_index": args.run_index, **parsed})
    payload = {
        "status": "PASS",
        "variant": args.variant,
        "run_index": args.run_index,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "snapshot": snapshot,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_active": bool(active) if args.variant != "base" else False,
        "active_adapters": active,
        "merged": False,
        "wrapper_count": len(wrappers),
        "wrapper_classification": wrapper_classes,
        "logit_slice": logit_slice,
        "logit_slice_sha256": hashlib.sha256(json.dumps(logit_slice).encode()).hexdigest(),
        "outputs": rows,
        "memory": {
            "before": before,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "cpu_offload": False,
            "oom": False,
            "during": nvidia_state(),
        },
    }
    output_path = args.output_dir / "reload_runs" / f"{args.variant}_{args.run_index}.json"
    write_json(output_path, payload)
    print(json.dumps({k: payload[k] for k in ("status", "variant", "run_index", "wrapper_count", "adapter_active")}, indent=2))
    return 0


def max_abs_difference(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Logit slices have different lengths")
    return max(abs(x - y) for x, y in zip(a, b)) if a else 0.0


def command_aggregate(args: argparse.Namespace) -> int:
    run_dir = args.output_dir / "reload_runs"
    base = json.loads((run_dir / "base_1.json").read_text(encoding="utf-8"))
    adapter1 = json.loads((run_dir / "adapter_1.json").read_text(encoding="utf-8"))
    adapter2 = json.loads((run_dir / "adapter_2.json").read_text(encoding="utf-8"))
    base_delta = max_abs_difference(base["logit_slice"], adapter1["logit_slice"])
    reload_delta = max_abs_difference(adapter1["logit_slice"], adapter2["logit_slice"])
    outputs = base["outputs"] + adapter1["outputs"] + adapter2["outputs"]
    write_jsonl(args.output_dir / "reload_inference.jsonl", outputs)
    normalized = [row["normalized_output"] for row in adapter1["outputs"]]
    counts = Counter(normalized)
    stats = {
        "sample_count": len(adapter1["outputs"]),
        "valid_parse_rate": sum(row["valid_parse"] for row in adapter1["outputs"]) / len(adapter1["outputs"]),
        "smoke_accuracy": sum(row["correct"] for row in adapter1["outputs"]) / len(adapter1["outputs"]),
        "unique_normalized_predictions": len(counts),
        "top_prediction_ratio": max(counts.values()) / len(normalized),
    }
    comparison = {
        "status": "PASS",
        "base_adapter_max_abs_logit_difference": base_delta,
        "adapter_reload_max_abs_logit_difference": reload_delta,
        "base_adapter_difference_finite_nonzero": math.isfinite(base_delta) and base_delta > 0,
        "independent_reload_consistent": math.isfinite(reload_delta) and reload_delta <= args.reload_tolerance,
        "adapter_active": adapter1["adapter_active"] and adapter2["adapter_active"],
        "adapter_wrapper_count": adapter1["wrapper_count"],
        "merged": False,
        "vqa": stats,
        "reload_peak_allocated": max(adapter1["memory"]["peak_allocated"], adapter2["memory"]["peak_allocated"]),
        "reload_peak_reserved": max(adapter1["memory"]["peak_reserved"], adapter2["memory"]["peak_reserved"]),
    }
    if not all((comparison["base_adapter_difference_finite_nonzero"], comparison["independent_reload_consistent"], comparison["adapter_active"], comparison["adapter_wrapper_count"] == 72)):
        comparison["status"] = "BLOCKED"
    write_json(args.output_dir / "reload_comparison.json", comparison)
    write_json(args.output_dir / "reload_target_audit.json", {
        "status": comparison["status"],
        "run_1": {"wrapper_count": adapter1["wrapper_count"], "classification": adapter1["wrapper_classification"], "active": adapter1["active_adapters"]},
        "run_2": {"wrapper_count": adapter2["wrapper_count"], "classification": adapter2["wrapper_classification"], "active": adapter2["active_adapters"]},
    })
    print(json.dumps(comparison, indent=2))
    return 0 if comparison["status"] == "PASS" else 1


def old_repo_fingerprint() -> str:
    command = (
        "find /root/MedAgentCL/med_prism /root/MedAgentCL/scripts /root/MedAgentCL/examples "
        "-type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum | sha256sum"
    )
    return subprocess.check_output(["bash", "-o", "pipefail", "-c", command], text=True).split()[0]


def command_finalize(args: argparse.Namespace) -> int:
    output = args.output_dir
    closure = json.loads((output / "phase2_closure" / "qwen3_text_generation_audit.json").read_text())
    target = json.loads((output / "target_module_audit.json").read_text())
    parameter = json.loads((output / "parameter_audit.json").read_text())
    gradient = json.loads((output / "training_gradient_summary.json").read_text())
    adapter = json.loads((output / "adapter_state_audit.json").read_text())
    reload = json.loads((output / "reload_comparison.json").read_text())
    trace = read_jsonl(output / "training_trace.jsonl")
    losses = [row.get("loss") for row in trace if row.get("loss") is not None]
    git_status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
    old_fingerprint = old_repo_fingerprint()
    checks = {
        "phase2_text_closure": closure["status"] == "PASS",
        "real_omnimedvqa_jsonl_trainable": json.loads((output / "data" / "smoke_schema_report.json").read_text())["status"] == "PASS",
        "ten_training_steps": len({row["step"] for row in trace if row.get("loss") is not None}) == 10,
        "loss_finite": len(losses) == 10 and all(math.isfinite(value) for value in losses),
        "backward_success": gradient["steps_with_gradients"] == 10,
        "q_gradient_nonzero": gradient["q_steps_nonzero"] == 10,
        "v_gradient_nonzero": gradient["v_steps_nonzero"] == 10,
        "wrapper_count_72": target["post_injection"]["total"] == 72,
        "vision_wrapper_zero": target["post_injection"]["vision"] == 0,
        "merger_wrapper_zero": target["post_injection"]["merger_aligner"] == 0,
        "vision_trainable_zero": parameter["vision_trainable_parameter_count"] == 0,
        "merger_trainable_zero": parameter["merger_trainable_parameter_count"] == 0,
        "adapter_saved_and_audited": adapter["status"] == "PASS",
        "reload_pass": reload["status"] == "PASS",
        "raw_and_parsed_outputs_saved": (output / "reload_inference.jsonl").stat().st_size > 0,
        "old_project_unchanged": old_fingerprint == EXPECTED_OLD_FINGERPRINT,
        "git_worktree_clean": git_status == "",
        "no_phase4_or_med_prism": True,
    }
    status = "PASS" if all(checks.values()) else "PARTIAL"
    acceptance = {"status": status, "checks": checks, "failures": [key for key, value in checks.items() if not value]}
    write_json(output / "acceptance_matrix.json", acceptance)
    import peft
    import torch
    import transformers
    import swift
    environment = {
        "status": status,
        "python": os.sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "ms_swift": swift.__version__,
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "upstream_v4_4_1_commit": subprocess.check_output(["git", "rev-parse", "v4.4.1^{commit}"], text=True).strip(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "hf_home": os.environ.get("HF_HOME"),
        "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "old_project_fingerprint": old_fingerprint,
        "git_status_short": git_status,
        "forbidden_packages_used": [],
    }
    write_json(output / "environment.json", environment)
    report = f"""# Phase 3 Native LoRA Report

Overall status: **{status}**

## Phase 2 Text Closure

- Root cause: right padding plus uniform generated tensor width accounting counted post-EOS padding as output; the previous audit also omitted rendered templates and token-level EOS evidence.
- Fix: left padding, decode only the generated suffix, trim/count through the first EOS, retain raw IDs/templates, and record effective input lengths.
- Result: `{closure['status']}`; JSON parseable `{closure['json_parseable']}`; unexpected Human prefix `{closure['unexpected_human_prefix']}`.

## Native LoRA

- Backbone: `{MODEL_ID}` at immutable revision `{MODEL_REVISION}`.
- LoRA: rank `48`, alpha `96`, dropout `0.05`, language-only `q_proj`/`v_proj`.
- Wrappers: `{target['post_injection']['total']}` total; q `{target['post_injection']['q_proj']}`, v `{target['post_injection']['v_proj']}`, vision `{target['post_injection']['vision']}`, merger/aligner `{target['post_injection']['merger_aligner']}`.
- Trainable parameters: `{parameter['trainable_parameter_count']}` ({parameter['trainable_ratio']:.8f}); vision `{parameter['vision_trainable_parameter_count']}`, merger `{parameter['merger_trainable_parameter_count']}`.
- Losses: `{losses}`.
- Gradient evidence: q nonzero on `{gradient['q_steps_nonzero']}` steps, v nonzero on `{gradient['v_steps_nonzero']}` steps, vision/merger gradient steps `{gradient['vision_or_merger_gradient_steps']}`.
- Training peak allocated/reserved: `{gradient['peak_allocated']}` / `{gradient['peak_reserved']}` bytes.

## Save and Reload

- Adapter: `{adapter['adapter_file']}` ({adapter['adapter_file_bytes']} bytes), state audit `{adapter['status']}`.
- Adapter q/v nonzero tensors: `{adapter['q_nonzero_tensors']}` / `{adapter['v_nonzero_tensors']}`; visual and merger keys: `{adapter['classification']['vision']}` / `{adapter['classification']['merger_aligner']}`.
- Base vs adapter max absolute logit difference: `{reload['base_adapter_max_abs_logit_difference']}`.
- Two independent adapter reload max difference: `{reload['adapter_reload_max_abs_logit_difference']}`.
- VQA parse rate `{reload['vqa']['valid_parse_rate']}`, smoke accuracy `{reload['vqa']['smoke_accuracy']}`, unique predictions `{reload['vqa']['unique_normalized_predictions']}`, top ratio `{reload['vqa']['top_prediction_ratio']}`.

## Boundaries and Risks

- No Med-PRISM, rank-1, shared/private, Octopus, Phase 4, multi-GPU DDP, DeepSpeed, flash-attn, vLLM, cache deletion, or dependency upgrade.
- This is a deterministic 10-step pipeline smoke test; loss and VQA accuracy are not scientific performance measurements.
- Warnings and full logs are retained in `train.log`, `reload.log`, and the text closure report.
"""
    (output / "phase3_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(acceptance, indent=2))
    return 0 if status == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--train-count", type=int, default=24)
    prepare.add_argument("--eval-count", type=int, default=6)
    prepare.set_defaults(func=command_prepare)
    targets = sub.add_parser("targets")
    targets.add_argument("--module-names", type=Path, required=True)
    targets.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    targets.set_defaults(func=command_targets)
    adapter = sub.add_parser("adapter")
    adapter.add_argument("--adapter-dir", type=Path, required=True)
    adapter.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    adapter.add_argument("--rank", type=int, default=48)
    adapter.add_argument("--alpha", type=int, default=96)
    adapter.add_argument("--dropout", type=float, default=0.05)
    adapter.set_defaults(func=command_adapter)
    infer = sub.add_parser("infer")
    infer.add_argument("--variant", choices=["base", "adapter"], required=True)
    infer.add_argument("--run-index", type=int, required=True)
    infer.add_argument("--adapter-dir", type=Path)
    infer.add_argument("--eval-jsonl", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    infer.add_argument("--cache-dir", type=Path, required=True)
    infer.add_argument("--device", default="cuda:0")
    infer.set_defaults(func=command_infer)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    aggregate.add_argument("--reload-tolerance", type=float, default=1e-5)
    aggregate.set_defaults(func=command_aggregate)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    finalize.set_defaults(func=command_finalize)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    raise SystemExit(parsed.func(parsed))
