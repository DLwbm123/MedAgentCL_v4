#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from med_prism.baselines.mrlora import assert_stage_router
from scripts.medicalskill_v1_2_baselines.baseline_harness import HistoricalMemory,TASK_ORDER,atomic_json,build_run_manifest,inspect_adapter,load_json,trainer_state,validate_runtime_record,write_or_validate_run_manifest
from scripts.medicalskill_v1_2_replay.replay_memory import ROUTER_ONLY,atomic_jsonl,build_memory,read_jsonl

METHOD="mrlora_mllm_cl_qwen3vl_r16"
OFFICIAL={"repository":"https://github.com/bjzhb666/MLLM-CL","commit":"a5a6da12a8bba6de453e5b30799cd450a28d6048","license":"Apache-2.0","files":["scripts/Train/Task1.sh","scripts/Train/train_DCL_router.sh","llava/eval/model_agent_select_lora_DCL.py","configs/model_configs_router/LLaVA/MLLM-DCL/train/task2.json"]}

def make_manifest(a):
 p=build_run_manifest(method=METHOD,method_family="independent_task_experts_plus_generative_router",data_root=a.data_root,output_root=a.output_root,
  capacity_policy={"kind":"task_growing_experts","expert_rank":a.rank,"expert_alpha":2*a.rank,"router_rank":a.router_rank,"router_alpha":2*a.router_rank,"target_modules":["q_proj","v_proj"]},
  historical_memory=HistoricalMemory(raw_examples=True,features_or_prototypes=False,statistics_or_masks=False,historical_parameters=True,notes="20 unlabeled image-question examples per learned task retained only for router training; expert and router parameters stored; no formal privacy guarantee."),
  method_config={"official_audit":OFFICIAL,"router_memory_examples_per_task":a.router_memory_per_task,"router_replay_usage":ROUTER_ONLY,"router_objective":"autoregressive generation of EXPERT_nn","router_epochs":a.router_epochs,"expert_initialization":"locked pretrained base independently for every task","historical_experts_frozen":True,"router_checkpoint_policy":"one independently trained cumulative-data router per stage","primary_inference":"router pass then one selected-expert answer pass","oracle_task_id":False,"invalid_output_fallback":"deterministic hash over available experts, never true task","training_data_limit":a.train_limit,"evaluation_data_limit":a.eval_limit,"requested_stage_count":a.max_stages})
 p["historical_memory"]["evolving_model_state"]="router_stage_specific"
 return p

def prepare_router(a):
 start=time.monotonic();stage_dir=a.output_root/"mrlora"/f"stage_{a.stage:02d}"
 m=build_memory(data_root=a.data_root,stage=a.stage,purpose=ROUTER_ONLY,output_dir=stage_dir/"router_memory",examples_per_task=a.router_memory_per_task,include_current=True,seed=a.seed,previous_manifest=(a.output_root/"mrlora"/f"stage_{a.stage-1:02d}"/"router_memory"/"manifest.json") if a.stage>1 else None)
 atomic_json(stage_dir/"router_memory_build_runtime.json",{"status":"PASS","wall_clock_seconds":time.monotonic()-start});return m

def prepare_expert(a):
 stage_dir=a.output_root/"mrlora"/f"stage_{a.stage:02d}";source=a.data_root/TASK_ORDER[a.stage-1]/"train.jsonl";rows=read_jsonl(source)
 if a.train_limit>0:rows=rows[:a.train_limit]
 target=stage_dir/"expert_train.jsonl";atomic_jsonl(target,rows)
 value={"status":"PASS","stage":a.stage,"task":TASK_ORDER[a.stage-1],"source":str(source.resolve()),"sample_count":len(rows),"dataset":str(target.resolve()),"initialization":"locked_pretrained_base_only","historical_experts_loaded":False};atomic_json(stage_dir/"expert_train_manifest.json",value);return value

def complete(a):
 run=load_json(a.output_root/"run_manifest.json");stage_dir=a.output_root/"mrlora"/f"stage_{a.stage:02d}";expert=inspect_adapter(a.expert_checkpoint);router=inspect_adapter(a.router_checkpoint)
 if expert["r"]!=a.rank or expert["target_modules"]!=["q_proj","v_proj"]:raise RuntimeError("Expert adapter contract mismatch")
 if router["r"]!=a.router_rank or router["target_modules"]!=["q_proj","v_proj"]:raise RuntimeError("Router adapter contract mismatch")
 expert_rt=validate_runtime_record(a.expert_runtime,"mrlora_expert_train");router_rt=validate_runtime_record(a.router_runtime,"mrlora_router_train")
 experts=[]
 for task in range(1,a.stage+1):
  c=stage_dir/"completion.json" if task==a.stage else a.output_root/"mrlora"/f"stage_{task:02d}"/"completion.json"
  if task==a.stage: info=expert
  else: info=load_json(c)["expert_adapter"]
  experts.append({"task":task,"checkpoint":info["checkpoint"],"adapter_weights_sha256":info["adapter_weights_sha256"]})
 assert_stage_router(a.stage,a.stage,[x["task"] for x in experts])
 memory=load_json(stage_dir/"router_memory"/"manifest.json");memory_rt=load_json(stage_dir/"router_memory_build_runtime.json")
 payload={"status":"PASS","method":METHOD,"stage":a.stage,"immutable_config_sha256":run["immutable_config_sha256"],"expert_checkpoint":expert["checkpoint"],"router_checkpoint":router["checkpoint"],"stage2_checkpoint":expert["checkpoint"],"expert_initialization":"locked_pretrained_base_only","previous_expert_loaded_for_training":False,"historical_experts_frozen":True,"expert_adapter":expert,"router_adapter":router,"inference_stage2_checkpoints":experts,"router_stage":a.stage,"router_memory_manifest_sha256":memory["buffer_manifest_sha256"],"parameter_accounting":{"task_expert_parameters":expert["adapter_parameters"],"accumulated_expert_parameters":sum(inspect_adapter(Path(x["checkpoint"]))["adapter_parameters"] for x in experts),"router_parameters":router["adapter_parameters"],"stored_parameters":sum(inspect_adapter(Path(x["checkpoint"]))["adapter_parameters"] for x in experts)+router["adapter_parameters"],"active_parameters_per_answer_pass":expert["adapter_parameters"],"active_parameters_router_pass":router["adapter_parameters"]},"runtime_accounting":{"expert_train_seconds":expert_rt["wall_clock_seconds"],"replay_memory_build_seconds":memory_rt["wall_clock_seconds"],"router_train_seconds":router_rt["wall_clock_seconds"],"total_training_seconds":expert_rt["wall_clock_seconds"]+router_rt["wall_clock_seconds"]+memory_rt["wall_clock_seconds"],"training_gpu_hours":expert_rt["gpu_hours"]+router_rt["gpu_hours"]},"replay_accounting":{"purpose":ROUTER_ONLY,"sample_count":memory["memory_count"],"count_per_task":memory["count_per_task"],"metadata_bytes":memory["physically_stored_metadata_bytes"],"copied_data_bytes":memory["copied_data_bytes"]},"expert_trainer_state":trainer_state(a.expert_checkpoint),"router_trainer_state":trainer_state(a.router_checkpoint)}
 atomic_json(stage_dir/"completion.json",payload);return payload

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest="cmd",required=True);common=argparse.ArgumentParser(add_help=False)
 for name,kwargs in [("data-root",{"type":Path,"required":True}),("output-root",{"type":Path,"required":True}),("rank",{"type":int,"default":16}),("router-rank",{"type":int,"default":32}),("router-memory-per-task",{"type":int,"default":20}),("router-epochs",{"type":float,"default":30.0}),("seed",{"type":int,"default":42}),("train-limit",{"type":int,"default":0}),("eval-limit",{"type":int,"default":0}),("max-stages",{"type":int,"default":5})]: common.add_argument("--"+name,**kwargs)
 c=sub.add_parser("check",parents=[common]);c.add_argument("--write",action="store_true")
 r=sub.add_parser("prepare-router",parents=[common]);r.add_argument("--stage",type=int,required=True)
 e=sub.add_parser("prepare-expert",parents=[common]);e.add_argument("--stage",type=int,required=True)
 d=sub.add_parser("complete",parents=[common]);d.add_argument("--stage",type=int,required=True);d.add_argument("--expert-checkpoint",type=Path,required=True);d.add_argument("--router-checkpoint",type=Path,required=True);d.add_argument("--expert-runtime",type=Path,required=True);d.add_argument("--router-runtime",type=Path,required=True)
 a=p.parse_args()
 if a.rank!=16:raise ValueError("MR-LoRA expert rank is frozen at 16")
 if a.router_memory_per_task!=20:raise ValueError("Official MR-LoRA router memory default is 20/task")
 if a.cmd=="check": value=make_manifest(a);write_or_validate_run_manifest(a.output_root/"run_manifest.json",value) if a.write else None
 elif a.cmd=="prepare-router":value=prepare_router(a)
 elif a.cmd=="prepare-expert":value=prepare_expert(a)
 else:value=complete(a)
 print(json.dumps(value,ensure_ascii=False,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
