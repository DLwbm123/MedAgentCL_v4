"""Explicit, gated v2.1 orchestration. Never invokes TPM or reads historical data.

--run is mandatory for training; by default only print the plan. Each task's
training completes before its train-only boundary; the next task requires GO.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from med_prism.transport.pilot import training_config,training_command,validate_train_completion,REPO
from med_prism.transport.checkpoint import AdapterState,materialize_state,read_state,sha256_file
from med_prism.transport.cli import TASKS
from med_prism.transport.diagnostics import write_json


def stage_train(stage,task,data,previous,summaries,gpus,branch):
    for path in summaries:
        marker=json.loads((Path(path).parent/"completion.json").read_text())
        if marker["status"]!="PASS" or marker["field_quality"]!="GO" or marker["summary_sha256"]!=sha256_file(path):
            raise ValueError("Historical summary integrity/quality mismatch")
    dataset=data/TASKS[task]/"train.jsonl"
    manifest=json.loads((data/"pilot_data_manifest.json").read_text())
    expected=[v for v in manifest["files"] if v.get("role")=="train" and v.get("task")==task]
    if (not manifest.get("global_group_disjoint") or len(expected)!=1 or
        expected[0]["sha256"]!=sha256_file(dataset)):
        raise ValueError("Current training file/grouped split differs from frozen data manifest")
    cfg=training_config(stage,task,dataset,data/"pilot_data_manifest.json",previous,42)
    command=training_command(stage,dataset,gpus,42)
    env=os.environ.copy()
    env.pop("MED_PRISM_RCWP_CONFIG_JSON",None)
    if branch=="rcwp":
        settings={"method_version":"2.1","method_name":"Med-PRISM-v2.1-RCWP",
            "rcwp_lambda":.1,"tpm_enabled":False,"summaries":summaries}
        command.append(str(REPO/"med_prism/rcwp/plugin.py"))
        env["MED_PRISM_RCWP_CONFIG_JSON"]=json.dumps(settings)
    contract={"method_version":"2.1" if branch=="rcwp" else "1.2",
        "method_name":"Med-PRISM-v2.1-RCWP" if branch=="rcwp" else "no-geo baseline",
        "baseline_config":cfg.to_dict(),"summary_hashes":{p:sha256_file(p) for p in summaries},
        "command":command,"tpm_enabled":False}
    if (stage/"gradient_completion.json").exists():
        if json.loads((stage/"run_contract.json").read_text())!=contract:
            raise ValueError("Resume contract mismatch; use new root")
        validate_train_completion(stage,task)
        return read_state(stage/"state")
    if stage.exists():raise FileExistsError(f"Incomplete stage preserved: {stage}; no implicit retraining")
    stage.mkdir(parents=True)
    write_json(stage/"config.json",cfg.to_dict())
    write_json(stage/"run_contract.json",contract)
    env.update(CUDA_VISIBLE_DEVICES=gpus,NPROC_PER_NODE=str(len(gpus.split(','))),
        MED_PRISM_SHARED_PRIVATE_CONFIG=str(stage/"config.json"),MED_PRISM_RELOAD_PROBE="1",
        PYTHONPATH=str(REPO)+os.pathsep+env.get("PYTHONPATH",""))
    with (stage/"train.log").open("w") as log:
        subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    checks=validate_train_completion(stage,task)
    source=AdapterState(task,str(stage/"shared/shared_manifest.json"),
        [*(previous.private_manifests if previous else []),str(stage/"private/private_manifest.json")],str(stage)).validate()
    # Existing immutable component serializer ONLY; never TPM repair/transport.
    state=materialize_state(source,stage/"state",accepted=True,metadata={"method_version":contract["method_version"],
        "method_name":contract["method_name"],"tpm_enabled":False})
    write_json(stage/"gradient_completion.json",{"status":"PASS","checks":checks,"state":state.origin})
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root",required=True)
    p.add_argument("--data-root",required=True,help="Existing grouped pilot data, not copied or modified")
    p.add_argument("--gpus",default="0")
    p.add_argument("--branch",choices=["baseline","rcwp"],default="rcwp")
    p.add_argument("--run",action="store_true")
    args=p.parse_args()
    root=Path(args.output_root).resolve();data=Path(args.data_root).resolve()
    if "MedPRISM_v2_1_RCWP" not in str(root):raise ValueError("Use independent MedPRISM_v2_1_RCWP root")
    if args.gpus not in {"0","1","0,1"}:raise ValueError("Unsupported GPU selection")
    manifest=data/"pilot_data_manifest.json"
    if not manifest.is_file():raise FileNotFoundError(manifest)
    for t in TASKS:
        if not (data/TASKS[t]/"train.jsonl").is_file():raise FileNotFoundError(TASKS[t])
    if not args.run:
        print(json.dumps({"status":"PLAN_ONLY","branch":args.branch,"tasks":[1,2,3],
            "output":str(root),"no_training_started":True,"boundary":"128 fit + 64 holdout, train only",
            "next_task_gate":"AUROC>=.65 and greater than constant/SRG; no automatic override"},indent=2))
        return
    previous=None;summaries=[]
    for task in (1,2,3):
        stage=root/args.branch/f"task_{task:02d}"
        previous=stage_train(stage,task,data,previous,summaries,args.gpus,args.branch)
        if args.branch=="rcwp":
            boundary=stage/"rcwp_boundary"
            if not (boundary/"completion.json").exists():
                env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=args.gpus.split(',')[0]
                subprocess.run([sys.executable,"-m","med_prism.rcwp.cli","--state",previous.origin,
                    "--current-task",str(task),"--current-train",str(data/TASKS[task]/"train.jsonl"),
                    "--output-root",str(boundary)],cwd=REPO,env=env,check=True)
            result=json.loads((boundary/"completion.json").read_text())
            if (result["status"]!="PASS" or result["source_state_sha256"]!=sha256_file(previous.origin)
                or result["summary_sha256"]!=sha256_file(boundary/"summary.pt")):
                raise RuntimeError("Boundary failed or changed after publication")
            summaries.append(str(boundary/"summary.pt"))
            write_json(stage/"rcwp_state.json",{"schema":"MedPRISM_RCWP_state_v1","method_version":"2.1",
                "method_name":"Med-PRISM-v2.1-RCWP","task_id":task,"state":previous.origin,
                "state_sha256":sha256_file(previous.origin),"shared_manifest":previous.shared_manifest,
                "private_manifests":previous.private_manifests,"summaries":list(summaries),
                "summary_sha256":{s:sha256_file(s) for s in summaries},"tpm_enabled":False,
                "field_quality":result["field_quality"],"deployment_requires_summaries":False})
            if result["field_quality"]!="GO":
                raise RuntimeError(f"Field gate {result['field_quality']}; STOP before next task. No full benchmark authorization inferred.")
    write_json(root/args.branch/"completion.json",{"status":"PASS","final_state":previous.origin})


if __name__=="__main__":main()
