"""Independent three-task orchestration using the unchanged v1 no-geo trainer.

Default prepares grouped data and a frozen plan only. --run explicitly starts
training. Completed stage/boundary indexes are validated and reused on restart.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid
from .checkpoint import AdapterState, materialize_state, read_state, sha256_file
from .cli import TASKS, DEFAULT_DATA, repair
from .config import TPMConfig, add_tpm_arguments, config_from_args
from .diagnostics import write_json
from .pilot_data import prepare_data

REPO = Path(__file__).resolve().parents[2]
SWIFT = "/root/anaconda3/envs/medagentcl_v4/bin/swift"


def training_config(stage, task, dataset, data_manifest, previous, seed):
    from med_prism.config import SharedPrivateConfig
    # Runtime trainer version deliberately remains v1.2: v2 is the independent
    # boundary state machine, not a renamed or duplicated gradient implementation.
    cfg = SharedPrivateConfig(current_task_id=task, private_rank=16, orth_lambda=0,
        method_version="1.2", method_variant="shared_fix_plus_key_isolation", ablation_name="no_geo_orth",
        shared_optimizer_mode="separate_lr_param_groups", shared_lr=1e-5, private_lr=1e-4,
        shared_gradient_hook_enabled=False, shared_drift_mode="effective_BA", shared_drift_lambda=.01,
        key_isolation_enabled=True, key_loss_mode="rms_cosine_A", key_lambda=.1,
        key_history_scope="all_previous_tasks_same_layer", trace_every_optimizer_steps=10,
        geometry_every_optimizer_steps=50, seed=seed,
        shared_source_manifest=previous.shared_manifest if previous else None,
        private_source_manifests=tuple(previous.private_manifests) if previous else (),
        shared_output_dir=str(stage/"shared"), private_output_dir=str(stage/"private"),
        dataset_manifest=str(data_manifest), dataset_sha256=sha256_file(dataset), output_dir=str(stage),
        artifact_root=str(stage/"runtime_audits"))
    cfg.validate()
    if previous and previous.task_id != task-1:
        raise ValueError("Training must load previous accepted state")
    return cfg


def training_command(stage, dataset, gpus, seed):
    from med_prism.config import MODEL_ID, MODEL_REVISION
    devices = gpus.split(",")
    if not 1 <= len(devices) <= 2 or len(set(devices)) != len(devices) or any(d not in {"0","1"} for d in devices):
        raise ValueError("Use GPU 0, 1, or 0,1")
    # Same locked options as run_versioned_train.sh, task 1..3. No new losses,
    # sampling flags, epochs, optimizer, teacher hooks, or scheduler settings.
    flags = {
        "model": MODEL_ID, "model_revision": MODEL_REVISION, "template": "qwen3_vl",
        "dataset": str(dataset), "tuner_type": "med_prism_shared_private",
        "freeze_llm": "false", "freeze_vit": "true", "freeze_aligner": "true",
        "torch_dtype": "bfloat16", "attn_impl": "sdpa", "max_length": "1024", "max_pixels": "200704",
        "per_device_train_batch_size": "1", "gradient_accumulation_steps": str(16//len(devices)),
        "learning_rate": "1e-4", "num_train_epochs": "1", "warmup_ratio": "0.03",
        "logging_steps": "10", "logging_first_step": "true", "save_strategy": "epoch",
        "save_total_limit": "1", "save_only_model": "false", "eval_strategy": "no",
        "gradient_checkpointing": "true", "vit_gradient_checkpointing": "false",
        "ddp_find_unused_parameters": "false", "dataloader_num_workers": "0", "dataset_num_proc": "1",
        "dataset_shuffle": "true", "split_dataset_ratio": "0", "packing": "false", "use_hf": "true",
        "seed": str(seed), "data_seed": str(seed), "report_to": "none", "add_version": "false",
        "create_checkpoint_symlink": "false", "logging_dir": str(stage/"train/logs"), "output_dir": str(stage/"train")}
    command = [SWIFT, "sft"]
    for key, value in flags.items():
        command += ["--"+key, value]
    return command + ["--external_plugins", str(REPO/"scripts/phase3/callback_compat.py"),
        str(REPO/"med_prism/swift_plugins/med_prism_rank1_plugin.py"),
        str(REPO/"med_prism/swift_plugins/med_prism_shared_private_plugin.py"),
        str(REPO/"scripts/med_prism_real_5skill/reload_probe_plugin.py")]


def validate_train_completion(stage, task):
    checks = {"checkpoint": any((stage/"train").glob("checkpoint-*/trainer_state.json"))}
    audit_paths = [stage/"target_module_audit.json"] + [stage/"runtime_audits"/name for name in (
        f"task{task}_optimizer_audit.json", f"task{task}_parameter_audit.json", "shared_hash_audit.json",
        "private_hash_audit.json", "orth_gradient_audit.json", "shared_drift_gradient_audit.json", "key_isolation_audit.json")]
    for path in audit_paths:
        checks[str(path.relative_to(stage))] = path.is_file() and json.loads(path.read_text()).get("status") == "PASS"
    if not all(checks.values()):
        raise RuntimeError(f"Gradient-training audits failed: {checks}")
    return checks


def train_stage(stage, task, data, previous, *, gpus, seed):
    stage = Path(stage)
    dataset = data/TASKS[task]/"train.jsonl"
    cfg = training_config(stage, task, dataset, data/"pilot_data_manifest.json", previous, seed)
    complete = stage/"gradient_completion.json"
    if complete.is_file():
        marker = json.loads(complete.read_text())
        if marker["status"] != "PASS" or json.loads((stage/"config.json").read_text()) != cfg.to_dict():
            raise ValueError("Completed stage configuration differs from requested accepted-state chain")
        validate_train_completion(stage,task)
        return read_state(stage/"pre_tpm")
    if stage.exists() and any(stage.iterdir()):
        raise RuntimeError(f"Incomplete gradient stage preserved: {stage}. Inspect logs; archive this exact stage to a new name before explicitly restarting. No automatic overwrite.")
    stage.mkdir(parents=True,exist_ok=True)
    (stage/"runtime_audits").mkdir()
    write_json(stage/"config.json",cfg.to_dict())
    command = training_command(stage,dataset,gpus,seed)
    write_json(stage/"command.json",command)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpus, NPROC_PER_NODE=str(len(gpus.split(","))),
               MED_PRISM_SHARED_PRIVATE_CONFIG=str(stage/"config.json"), MED_PRISM_RELOAD_PROBE="1",
               PYTHONPATH=str(REPO)+os.pathsep+env.get("PYTHONPATH",""))
    print(shlex.join(command),flush=True)
    with (stage/"train.log").open("w") as log:
        process = subprocess.Popen(command,cwd=REPO,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            print(line,end="",flush=True)
            log.write(line)
            log.flush()
        code=process.wait()
    if code:
        raise RuntimeError(f"swift training failed ({code}); incomplete stage preserved: {stage}")
    checks=validate_train_completion(stage,task)
    source=AdapterState(task,str(stage/"shared/shared_manifest.json"),
                [*(previous.private_manifests if previous else []),str(stage/"private/private_manifest.json")],str(stage)).validate()
    pre=materialize_state(source,stage/"pre_tpm",accepted=False)
    write_json(complete,{"status":"PASS","checks":checks,"pre_tpm":pre.origin})
    return pre


def boundary(stage, teacher, pre, dataset, config):
    marker=stage/"boundary_completion.json"
    if marker.is_file():
        value=json.loads(marker.read_text())
        if value["config"] != config.to_dict() or value["status"] != "PASS":
            raise ValueError("Existing accepted boundary has a different frozen TPM configuration")
        return read_state(value["accepted_state"], require_accepted=True)
    # Failed repair can restart from existing gradient output; no retraining needed.
    attempt=stage/("boundary_"+time.strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:6])
    cmd=[sys.executable,"-m","med_prism.transport.cli","--task",str(pre.task_id),
         "--student-state",pre.origin,"--teacher-state",teacher.origin,"--current-data",str(dataset),"--output-root",str(attempt)]
    for key,value in config.to_dict().items():
        if key in {"numerical_atol","numerical_rtol","edit_tolerance"}:
            continue
        flag="--tpm-"+key.replace("_","-")
        if isinstance(value,bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag,"none" if value is None else str(value)])
    # A subprocess frees the 8B model after each boundary, before subsequent training.
    subprocess.run(cmd,cwd=REPO,check=True)
    accepted=read_state(attempt/"accepted", require_accepted=True)
    write_json(marker,{"status":"PASS","config":config.to_dict(),"accepted_state":accepted.origin})
    return accepted


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",required=True)
    parser.add_argument("--source-data-root",default=DEFAULT_DATA)
    parser.add_argument("--gpus",default="0")
    parser.add_argument("--branch",choices=["both","baseline","tpm"],default="both")
    parser.add_argument("--run",action="store_true")
    add_tpm_arguments(parser)
    args=parser.parse_args(argv)
    config=config_from_args(args)
    root=Path(args.output_root).resolve()
    if "MedPRISM_v2_TPM" not in str(root):
        parser.error("New output root must contain MedPRISM_v2_TPM")
    # Pilot recipe has one fixed small experiment, not a grid-search interface.
    data=root/"data"
    prepare_data(args.source_data_root,data,seed=config.seed)
    contract={"method":"Med-PRISM v2.0 TPM","training_recipe":"v1.2 no_geo_orth unchanged",
        "task_sequence":list(TASKS.values()),"train_per_task":2000,"development_per_task":256,
        "shared_task1":True,"task2_and_task3_independent_per_branch":True,"tpm":config.to_dict(),
        "gpus":args.gpus,"data_manifest_sha256":sha256_file(data/"pilot_data_manifest.json"),
        "code_sha256":{str(p.relative_to(REPO)):sha256_file(p) for p in sorted((REPO/"med_prism").rglob("*.py"))}}
    contract_path=root/"pilot_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Pilot source/configuration changed; use a new run root, do not mix chains")
    write_json(contract_path,contract)
    training_command(root/"common/task_01",data/TASKS[1]/"train.jsonl",args.gpus,config.seed)
    if not args.run:
        print(json.dumps({"status":"PREPARED","root":str(root),"no_training_launched":True,
                          "start":"Rerun the same command with --run","gradient_samples_both_branches":10000},indent=2))
        return
    common=root/"common/task_01"
    pre=train_stage(common,1,data,None,gpus=args.gpus,seed=config.seed)
    if (common/"accepted/state.json").is_file():
        first=read_state(common/"accepted", require_accepted=True)
    else:
        first=materialize_state(pre,common/"accepted",accepted=True,metadata={"task1_no_history":True})
    branches=["baseline","tpm"] if args.branch=="both" else [args.branch]
    for branch in branches:
        previous=first
        mode=replace(config,mode="off") if branch=="baseline" else config
        for task in (2,3):
            stage=root/branch/f"task_{task:02d}"
            pre=train_stage(stage,task,data,previous,gpus=args.gpus,seed=config.seed)
            previous=boundary(stage,previous,pre,data/TASKS[task]/"train.jsonl",mode)
        write_json(root/branch/"completion.json",{"status":"PASS","final_accepted_state":previous.origin})
    print("Selected three-task pilot branches complete; no evaluation or five-task run was started.")


if __name__ == "__main__":
    main()
