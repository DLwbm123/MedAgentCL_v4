#!/usr/bin/env python3
"""Shared-scorer evaluator handoff for a single fixed CoIN MoELoRA state."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import torch
from peft import PeftModel
from med_prism.baselines.moelora import enable_moelora_patch, moelora_runtime_stats
from scripts.medicalskill_v1_2_baselines.baseline_harness import EVALUATION_PROTOCOL,MODEL_ID,MODEL_REVISION,SNAPSHOT,inspect_adapter
from scripts.medicalskill_v1_2_baselines.evaluate_cumulative_lora_v1_2 import atomic_json,append_prediction,read_raw_predictions
from scripts.medicalskill_v1_2_med_prism.evaluate_formal_v1_2 import TASK_DIRS,TASK_NAMES,evaluate_rows,generate,read_jsonl,sha256

def active_manifest(checkpoint:Path):
 info=inspect_adapter(checkpoint)
 if info["r"]!=48 or info["target_modules"]!=["q_proj","v_proj"]: raise RuntimeError("MoELoRA checkpoint contract mismatch")
 return {"method":"coin_moelora_qwen3vl_total_r48_e4","composition":"single_fixed_capacity_routed_state","checkpoint":info["checkpoint"],"adapter_weights_sha256":info["adapter_weights_sha256"],"adapter_config_sha256":info["adapter_config_sha256"],"active_adapter_parameters":info["adapter_parameters"],"num_experts":4,"rank_per_expert":12,"total_expert_rank":48,"routing":"token_level_dense_softmax_all_experts","oracle_task_id":False}

def load_model(checkpoint:Path):
 enable_moelora_patch()
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 processor=AutoProcessor.from_pretrained(str(SNAPSHOT),local_files_only=True)
 if hasattr(processor.image_processor,"min_pixels"): processor.image_processor.min_pixels=200704
 if hasattr(processor.image_processor,"max_pixels"): processor.image_processor.max_pixels=200704
 base=Qwen3VLForConditionalGeneration.from_pretrained(str(SNAPSHOT),local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa")
 model=PeftModel.from_pretrained(base,str(checkpoint),is_trainable=False)
 return model.to("cuda:0").eval(),processor

def main():
 p=argparse.ArgumentParser();p.add_argument("--stage",type=int,choices=range(1,6),required=True);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--eval-limit",type=int,default=0);a=p.parse_args()
 if a.eval_limit<0: raise ValueError("negative eval limit")
 checkpoint=a.checkpoint.resolve();data=a.data_root.resolve();root=a.output_root.resolve();active=active_manifest(checkpoint);stage_dir=root/"evaluation"/f"stage_{a.stage:02d}";summary=stage_dir/"stage_evaluation_summary.json"
 if summary.is_file():
  old=json.loads(summary.read_text())
  if old.get("status")=="PASS" and old.get("active_adapter")==active and old.get("eval_limit")==a.eval_limit: print("SKIP completed MoELoRA evaluation");return 0
 torch.manual_seed(42);started=time.time();model,processor=load_model(checkpoint);cells={}
 for task in range(1,a.stage+1):
  dataset=data/TASK_DIRS[task]/"test.jsonl";rows=read_jsonl(dataset,a.eval_limit)
  if not rows: raise RuntimeError(f"Empty evaluation dataset: {dataset}")
  stem=f"primary_routed_task_{task:02d}";raw_path=stage_dir/f"{stem}.raw_predictions.jsonl";values=read_raw_predictions(raw_path,rows,active)
  for i in range(len(values),len(rows)):
   prediction=generate(model,processor,rows[i],task);value={"id":rows[i]["id"],"prediction":prediction,"active_adapter":active};append_prediction(raw_path,value);values.append(value)
  metric,details=evaluate_rows(task,rows,[x["prediction"] for x in values]);detail=stage_dir/f"{stem}.predictions.jsonl";detail.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in details),encoding="utf-8")
  metric.update({"mode":"primary_fixed_routed","stage":a.stage,"eval_task":task,"task_name":TASK_NAMES[task],"method":active["method"],"evaluation_protocol":EVALUATION_PROTOCOL,"dataset":str(dataset),"dataset_sha256":sha256(dataset),"model_id":MODEL_ID,"model_revision":MODEL_REVISION,"active_adapter":active,"predictions":str(detail),"predictions_sha256":sha256(detail),"raw_predictions":str(raw_path),"raw_predictions_sha256":sha256(raw_path),"fresh_process_reload":True,"oracle_task_id":False});atomic_json(stage_dir/f"{stem}.summary.json",metric);cells[stem]=metric
 router=moelora_runtime_stats(model);status="PASS" if len(cells)==a.stage and all(x.get("status")=="PASS" for x in cells.values()) else "FAIL";final={"status":status,"stage":a.stage,"method":active["method"],"evaluation_protocol":EVALUATION_PROTOCOL,"matrix_scope":"seen_tasks_only","evaluated_task_ids":list(range(1,a.stage+1)),"cell_count":len(cells),"model_load_count":1,"elapsed_seconds":time.time()-started,"data_root":str(data),"eval_limit":a.eval_limit,"active_adapter":active,"router_runtime":router,"cells":cells};atomic_json(summary,final);print(json.dumps({"status":status,"stage":a.stage,"cells":len(cells)}));return 0 if status=="PASS" else 2
if __name__=="__main__": raise SystemExit(main())
