"""Checkpoint repair pilot. Fresh output only; never trains or generates."""
import argparse
import json
from pathlib import Path
import time
import torch
from .calibration import encode_current_rows
from .checkpoint import legacy_state, read_state, materialize_state, validate_transition, sha256_file
from .config import add_tpm_arguments, config_from_args
from .diagnostics import write_json

DEFAULT_NO_GEO = "/remote-home/wangbomin/med_prism_v1_2_ablation_no_geo_orth_medicalskill_v1_2_lite_10k1k_seed42"
DEFAULT_DATA = "/remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k"
TASKS = {1: "task_01_vqa", 2: "task_02_diagnosis_classification", 3: "task_03_concept_recognition"}


def load_calibration_model(student, max_length=1024):
    from swift.model import get_model_processor
    from swift.template import get_template
    from med_prism.config import MODEL_ID, MODEL_REVISION
    from med_prism.checkpoint.shared_private_checkpoint import compose_shared_private
    model, processor = get_model_processor(MODEL_ID, revision=MODEL_REVISION, use_hf=True,
                torch_dtype=torch.bfloat16, device_map="cuda:0", attn_impl="sdpa")
    compose_shared_private(model, shared_manifest=student.shared_manifest,
                           private_manifests=student.private_manifests)
    model.eval().requires_grad_(False)
    template = get_template(processor, template_type="qwen3_vl", max_length=max_length,
                            max_pixels=200704, padding_free=False, padding_side="right")
    template.model = model
    template.set_mode("train")  # Teacher-forced labels; model itself remains eval.
    return model, template


def repair(teacher, student, dataset, output, config, *, max_length=1024):
    output = Path(output).resolve()
    if "MedPRISM_v2_TPM" not in str(output):
        raise ValueError("Output must contain MedPRISM_v2_TPM, distinct from formal v1 roots")
    if output.exists():
        raise FileExistsError(f"Fresh repair output required: {output}")
    if config.mode != "off":
        validate_transition(teacher, student)
    dataset = Path(dataset).resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    output.mkdir(parents=True)
    started = time.perf_counter()
    write_json(output / "run_request.json", {"teacher": teacher.origin if teacher else None,
               "student": student.origin, "current_task": student.task_id, "dataset": str(dataset),
               "dataset_sha256": sha256_file(dataset), "method": "Med-PRISM v2.0 TPM",
               "tpm": config.to_dict(), "max_length": max_length,
               "scope": "current-task checkpoint repair; no training or generation"})
    pre = None
    try:
        pre = materialize_state(student, output / "pre_tpm", accepted=False)
        if config.mode == "off":
            # True no-op: no model load, no calibration data iteration or RNG changes.
            from .diagnostics import Diagnostics
            summary = Diagnostics(output / "diagnostics", config).finish({"status": "PASS", "mode": "off",
                "teacher_pass_seconds": 0, "student_calibration_seconds": 0,
                "invariants": "All adapter weights copied byte-for-byte; base not loaded or touched",
                "wall_seconds": time.perf_counter()-started})
            accepted = materialize_state(pre, output / "accepted", accepted=True, metadata=summary)
        else:
            model, template = load_calibration_model(pre, max_length)
            rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
            # Fail if input explicitly declares a different skill/task. No old task rows.
            for row in rows:
                value = row.get("task_id")
                if isinstance(value, int) and value != student.task_id:
                    raise ValueError("Calibration dataset contains another task")
            records, rejected = encode_current_rows(rows, template, config)
            write_json(output / "encoding_rejections.json", rejected)
            del rows
            from .engine import run_transport
            summary = run_transport(model, teacher, pre, records, config, output / "diagnostics")
            if summary["status"] != "PASS":
                raise RuntimeError("No accepted checkpoint after failed invariants")
            accepted = materialize_state(pre, output / "accepted", model=model, accepted=True, metadata=summary)
        write_json(output / "completion.json", {"status": "PASS", "accepted_state": accepted.origin,
                   "elapsed_seconds": time.perf_counter()-started})
        return accepted
    except Exception as exc:
        write_json(output / "failure.json", {"status": "FAIL", "error": str(exc),
                   "accepted_checkpoint": None, "pre_tpm_preserved": pre.origin if pre else None})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=DEFAULT_NO_GEO)
    parser.add_argument("--task", type=int, choices=[2, 3], default=3)
    parser.add_argument("--teacher-state")
    parser.add_argument("--student-state")
    parser.add_argument("--current-data")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--check-only", action="store_true")
    add_tpm_arguments(parser)
    args = parser.parse_args(argv)
    config = config_from_args(args)
    teacher = read_state(args.teacher_state, require_accepted=True) if args.teacher_state else legacy_state(args.source_root, args.task-1)
    student = read_state(args.student_state) if args.student_state else legacy_state(args.source_root, args.task)
    if student.task_id != args.task:
        raise ValueError("Student task differs from --task")
    validate_transition(teacher, student)
    dataset = Path(args.current_data) if args.current_data else Path(DEFAULT_DATA)/TASKS[args.task]/"train.jsonl"
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if args.check_only:
        print(json.dumps({"status": "PASS", "teacher": teacher.origin, "student": student.origin,
                          "data": str(dataset), "config": config.to_dict(), "no_model_loaded": True}, indent=2))
        return
    accepted = repair(teacher, student, dataset, args.output_root, config, max_length=args.max_length)
    print(json.dumps({"status": "PASS", "accepted_state": accepted.origin}, indent=2))


if __name__ == "__main__":
    main()
