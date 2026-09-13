#!/usr/bin/env python3
"""Shared failure-safe lifecycle controller for MedicalSkill-CL Wave 1."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any
from safetensors import safe_open
from scripts.medicalskill_v1_2_baselines.baseline_harness import (
    HistoricalMemory,TASK_ORDER,atomic_json,build_run_manifest,inspect_adapter,sha256_file,
    load_json,require_adapter_contract,trainer_state,validate_completion,
    validate_runtime_record,write_or_validate_run_manifest,
)
from med_prism.baselines.olora import inspect_olora_serialization
from med_prism.baselines.reglora import inspect_importance_mask

METHODS={
 "olora": {"id":"olora_qwen3vl_r16","family":"orthogonal_subspace_learning","rank":16,"alpha":32,
  "memory":HistoricalMemory(historical_parameters=True,notes="Frozen historical task LoRA parameters/A-subspaces; no historical raw examples."),
  "capacity":{"class":"task_growing","rank_per_task":16},
  "config":{"official_repository":"https://github.com/cmnfriend/O-LoRA","official_commit":"07117e1fc4a5f5ad9308a815a42cee8f46502dc8","lambda_1":0.5,"lambda_2":0.0,"orthogonality":"sum(abs(A_old @ A_current.T))","secondary_regularizer":"sum L2 norms of current A/B (coefficient 0 in official LLaMA scripts)","primary_inference":"base plus all task adapters in ascending order","oracle_task_id":False}},
 "reglora": {"id":"reglora_sefe_component_qwen3vl_r16","family":"importance_based_parameter_regularization","rank":16,"alpha":32,
  "memory":HistoricalMemory(statistics_or_masks=True,historical_parameters=True,notes="Cumulative top-2% |BA| index masks plus prior rank-16 adapters retained only to reconstruct the official merged evolving state; no ASD and no raw replay."),
  "capacity":{"class":"fixed_capacity_evolving","active_current_rank":16},
  "config":{"official_repository":"https://github.com/jinpeng0528/SEFE","official_commit":"2f61efe3a7da0407850399db6bc51f5ddc00f001","component":"RegLoRA only; ASD explicitly absent","importance_metric":"abs(B@A)","top_fraction":0.02,"mask_accumulation":"direct concatenation without deduplication","regularization":"mean across locked q/v linears of mean(abs((B@A)[protected indices])); official 224 divisor adapted to 72 q/v linears","coefficient":2500.0,"official_lifecycle":"fresh rank-16 LoRA on previous merged model, then merge","framework_adaptation":"base+sum of frozen prior adapters reconstructs merged forward exactly and avoids serializing a full 8B model per task","oracle_task_id":False}},
 "moelora": {"id":"coin_moelora_qwen3vl_total_r48_e4","family":"fixed_expert_token_routing","rank":48,"alpha":96,
  "memory":HistoricalMemory(notes="One evolving fixed-capacity expert/router model state; no frozen task-specific bank and no raw replay."),
  "capacity":{"class":"fixed_expert_pool","num_experts":4,"rank_per_expert":12,"total_expert_rank":48},
  "config":{"official_repository":"https://github.com/zackschen/CoIN","official_commit":"41411ab9ad77a520bc61f07a09fe9eba2f14cf6a","router_input":"token hidden state entering each q_proj/v_proj","routing":"token-level dense softmax over all four experts","active_experts_per_token":4,"auxiliary_router_loss":None,"lifecycle":"one persistent adapter/router continued across tasks","oracle_task_id":False}},
}
TARGETS=["q_proj","v_proj"]

def completion_path(root:Path,method:str,stage:int)->Path:
 return root/method/f"stage_{stage:02d}"/"completion.json"

def init_run(a):
 m=METHODS[a.method]; cap={**m["capacity"],"lora_alpha":m["alpha"],"target_modules":TARGETS,"parameter_equality_claim":False}
 payload=build_run_manifest(method=m["id"],method_family=m["family"],data_root=a.data_root,output_root=a.output_root,capacity_policy=cap,historical_memory=m["memory"],method_config={**m["config"],"lora_dropout":0.05,"learning_rate":1e-4,"epochs_per_task":1,"replay":False,"target_modules":TARGETS})
 if a.dry_run: print(json.dumps(payload,indent=2)); return
 a.output_root.mkdir(parents=True,exist_ok=True); write_or_validate_run_manifest(a.output_root/"run_manifest.json",payload); print(json.dumps(payload,indent=2))

def load_run(root:Path,method:str):
 v=load_json(root/"run_manifest.json"); expected=METHODS[method]["id"]
 if v.get("status")!="PASS" or v.get("method")!=expected: raise RuntimeError("Invalid Wave-1 run manifest")
 return v

def stage_ready(a):
 run=load_run(a.output_root,a.method)
 if a.stage>1: validate_completion(completion_path(a.output_root,a.method,a.stage-1),method=METHODS[a.method]["id"],stage=a.stage-1,immutable_config_sha256=run["immutable_config_sha256"])
 print("PASS")

def _router_count(info):
 count=0
 with safe_open(info["adapter_weights"],framework="pt",device="cpu") as h:
  for k in h.keys():
   if "lora_router" in k:
    n=1
    for d in h.get_slice(k).get_shape(): n*=int(d)
    count+=n
 return count

def finalize(a):
 run=load_run(a.output_root,a.method); m=METHODS[a.method]
 current=require_adapter_contract(a.checkpoint,rank=m["rank"],target_modules=TARGETS)
 histories=[require_adapter_contract(x,rank=m["rank"],target_modules=TARGETS) for x in a.inference_checkpoint]
 expected=a.stage if a.method in {"olora","reglora"} else 1
 if len(histories)!=expected or histories[-1]["adapter_weights_sha256"]!=current["adapter_weights_sha256"]: raise RuntimeError("Inference checkpoint lifecycle mismatch")
 prior=[]
 for task in range(1,a.stage): prior.append(validate_completion(completion_path(a.output_root,a.method,task),method=m["id"],stage=task,immutable_config_sha256=run["immutable_config_sha256"]))
 if a.method in {"olora","reglora"} and prior:
  expected_hashes=[x["adapter_weights_sha256"] for x in prior[-1]["inference_stage2_checkpoints"]]+[current["adapter_weights_sha256"]]
  if [x["adapter_weights_sha256"] for x in histories]!=expected_hashes: raise RuntimeError("Historical checkpoint/hash chain mismatch")
 if prior and current["adapter_weights_sha256"]==prior[-1]["adapter"]["adapter_weights_sha256"]: raise RuntimeError("Current stage adapter hash is unchanged; no effective training update")
 train=validate_runtime_record(a.train_runtime,"main_train"); evaluation=validate_runtime_record(a.eval_runtime,"evaluation")
 if train["status"]!="PASS" or evaluation["status"]!="PASS": raise RuntimeError("Training/evaluation runtime failed")
 if not a.evaluation_summary.is_file() or load_json(a.evaluation_summary).get("status")!="PASS": raise RuntimeError("Evaluator handoff is not PASS")
 mask_build=load_json(a.mask_build_record) if a.mask_build_record else None
 if a.mask and mask_build:
  if mask_build.get("status")!="PASS" or mask_build.get("mask_sha256")!=sha256_file(a.mask): raise RuntimeError("Mask build record mismatch")
  mask={key:value for key,value in mask_build.items() if key!="importance_mask_build_seconds"}
 else:
  mask=inspect_importance_mask(a.mask) if a.mask else None
 if (a.method=="reglora")!=(mask is not None): raise RuntimeError("RegLoRA alone requires a cumulative mask")
 if a.method=="olora": inspect_olora_serialization(a.checkpoint)
 router=_router_count(current) if a.method=="moelora" else 0
 if a.method=="moelora" and router<=0: raise RuntimeError("MoELoRA checkpoint lost router tensors")
 stored=sum(x["adapter_parameters"] for x in histories)
 active=stored if a.method in {"olora","reglora"} else current["adapter_parameters"]
 previous_aux=int(prior[-1]["parameter_accounting"]["accumulated_auxiliary_state_bytes"]) if prior else 0
 current_aux=int(mask["mask_storage_bytes"]) if mask else 0
 state=trainer_state(Path(current["checkpoint"]))
 peaks=[]
 for v in (train,evaluation): peaks.extend((v.get("peak_gpu_memory_mib") or {}).values())
 inference=[{"task":(a.stage if a.method=="moelora" else i+1),"checkpoint":x["checkpoint"],"adapter_weights_sha256":x["adapter_weights_sha256"],"adapter_config_sha256":x["adapter_config_sha256"],"parameters":x["adapter_parameters"]} for i,x in enumerate(histories)]
 payload={"status":"PASS","format_version":"medicalskill_cl_wave1_completion_v1","method":m["id"],"method_key":a.method,"stage":a.stage,"immutable_config_sha256":run["immutable_config_sha256"],"checkpoint":current["checkpoint"],"adapter":current,"inference_stage2_checkpoints":inference,"inference_semantics":("base_plus_cumulative_independent_task_adapters" if a.method=="olora" else "base_plus_reconstructed_merged_updates" if a.method=="reglora" else "single_fixed_dense_token_routed_state"),"oracle_task_id":False,"importance_mask":mask,"mask_build_record":mask_build,"parameter_accounting":{"trainable_parameters_current_stage":current["adapter_parameters"],"added_parameters_current_task":current["adapter_parameters"] if a.method in {"olora","reglora"} or a.stage==1 else 0,"accumulated_stored_adapter_parameters":stored,"active_adapter_parameters_at_inference":active,"official_materialized_merge_active_adapter_parameters":0 if a.method=="reglora" else None,"framework_reconstruction_active_adapter_parameters":stored if a.method=="reglora" else 0,"merged_update_source_parameters_at_inference":stored if a.method=="reglora" else 0,"router_parameters":router,"auxiliary_state_bytes":current_aux,"accumulated_auxiliary_state_bytes":previous_aux+current_aux,"checkpoint_bytes":sum(x["checkpoint_bytes"] for x in histories)},"runtime_accounting":{"main_train_seconds":train["wall_clock_seconds"],"auxiliary_seconds":float((mask_build or {}).get("importance_mask_build_seconds",0.0)),"evaluation_seconds":evaluation["wall_clock_seconds"],"total_stage_seconds":float(train["wall_clock_seconds"])+float(evaluation["wall_clock_seconds"])+float((mask_build or {}).get("importance_mask_build_seconds",0.0)),"training_gpu_hours":train["gpu_hours"],"peak_gpu_memory_mib":max(peaks) if peaks else None,"training_steps":state.get("training_steps"),"samples":train.get("number_of_samples"),"evaluation_gpu_hours":evaluation["gpu_hours"]},"evaluation_summary":str(a.evaluation_summary.resolve())}
 atomic_json(completion_path(a.output_root,a.method,a.stage),payload); print(json.dumps(payload,indent=2))

def completion_ok(a):
 run=load_run(a.output_root,a.method); v=validate_completion(completion_path(a.output_root,a.method,a.stage),method=METHODS[a.method]["id"],stage=a.stage,immutable_config_sha256=run["immutable_config_sha256"]); print(v["checkpoint"])

def main():
 p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
 q=sub.add_parser("init-run"); q.add_argument("--method",choices=METHODS,required=True);q.add_argument("--data-root",type=Path,required=True);q.add_argument("--output-root",type=Path,required=True);q.add_argument("--dry-run",action="store_true");q.set_defaults(func=init_run)
 for name,func in (("stage-ready",stage_ready),("completion-ok",completion_ok)):
  q=sub.add_parser(name);q.add_argument("--method",choices=METHODS,required=True);q.add_argument("--output-root",type=Path,required=True);q.add_argument("--stage",type=int,choices=range(1,6),required=True);q.set_defaults(func=func)
 q=sub.add_parser("finalize");q.add_argument("--method",choices=METHODS,required=True);q.add_argument("--output-root",type=Path,required=True);q.add_argument("--stage",type=int,choices=range(1,6),required=True);q.add_argument("--checkpoint",type=Path,required=True);q.add_argument("--inference-checkpoint",type=Path,action="append",default=[]);q.add_argument("--train-runtime",type=Path,required=True);q.add_argument("--eval-runtime",type=Path,required=True);q.add_argument("--evaluation-summary",type=Path,required=True);q.add_argument("--mask",type=Path);q.add_argument("--mask-build-record",type=Path);q.set_defaults(func=finalize)
 a=p.parse_args();a.output_root=a.output_root.resolve();return a.func(a)
if __name__=="__main__": main()
