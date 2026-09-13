#!/usr/bin/env python3
"""Stage-correct two-pass MR-LoRA evaluator using shared MedicalSkill scorers."""
from __future__ import annotations
import argparse,gc,json,os,sys,tempfile,time
from pathlib import Path
from typing import Any
import torch
from peft import PeftModel

ROOT=Path("/root/MedAgentCL_v4")
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from med_prism.baselines.mrlora import assert_stage_router,parse_router_output,router_diagnostics
from scripts.medicalskill_v1_2_baselines.baseline_harness import EVALUATION_PROTOCOL,MODEL_ID,MODEL_REVISION,SNAPSHOT,inspect_adapter
from scripts.medicalskill_v1_2_med_prism.evaluate_formal_v1_2 import TASK_DIRS,TASK_NAMES,evaluate_rows,generate,prompt_messages,read_jsonl,sha256
from scripts.medicalskill_v1_2_replay.replay_memory import router_prompt
METHOD="mrlora_mllm_cl_qwen3vl_r16"

def atomic(path:Path,value:Any):
 path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
 try:
  with os.fdopen(fd,"w",encoding="utf-8") as h:json.dump(value,h,ensure_ascii=False,indent=2);h.write("\n");h.flush();os.fsync(h.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)

def load_adapter(path:Path):
 from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
 processor=AutoProcessor.from_pretrained(str(SNAPSHOT),local_files_only=True)
 if hasattr(processor.image_processor,"min_pixels"):processor.image_processor.min_pixels=200704;processor.image_processor.max_pixels=200704
 model=Qwen3VLForConditionalGeneration.from_pretrained(str(SNAPSHOT),local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa");model.requires_grad_(False)
 model=PeftModel.from_pretrained(model,str(path),is_trainable=False)
 return model.to("cuda:0").eval(),processor

def release(model):
 del model;gc.collect();torch.cuda.empty_cache()

def route_generate(model,processor,row,stage):
 from qwen_vl_utils import process_vision_info
 fake={"messages":[{"role":"user","content":router_prompt(row,stage)}],"images":row.get("images") or []}
 messages=prompt_messages(fake);text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);images,videos=process_vision_info(messages)
 inputs=processor(text=[text],images=images,videos=videos,return_tensors="pt").to("cuda:0")
 with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=16,do_sample=False,num_beams=1,temperature=None,top_p=None,top_k=None)
 return processor.tokenizer.decode(out[0,inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip()

def main():
 p=argparse.ArgumentParser();p.add_argument("--stage",type=int,choices=range(1,6),required=True);p.add_argument("--router",type=Path,required=True);p.add_argument("--expert",action="append",type=Path,required=True);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--eval-limit",type=int,default=0);a=p.parse_args()
 if len(a.expert)!=a.stage:raise ValueError("Exactly experts 1..stage are required")
 assert_stage_router(a.stage,a.stage,list(range(1,len(a.expert)+1)))
 completion_path=a.output_root.resolve()/"mrlora"/f"stage_{a.stage:02d}"/"completion.json"
 if not completion_path.is_file():raise FileNotFoundError(f"Missing stage-owned router completion: {completion_path}")
 completion=json.loads(completion_path.read_text(encoding="utf-8"))
 if Path(completion["router_checkpoint"]).resolve()!=a.router.resolve() or completion.get("router_stage")!=a.stage:raise RuntimeError("Evaluation must load router_t owned by row t")
 for task_id,path in enumerate(a.expert,1):
  owner=json.loads((a.output_root.resolve()/"mrlora"/f"stage_{task_id:02d}"/"completion.json").read_text(encoding="utf-8"))
  if Path(owner["expert_checkpoint"]).resolve()!=path.resolve():raise RuntimeError("Evaluation expert does not match its stage owner")
 router_info=inspect_adapter(a.router.resolve());experts=[inspect_adapter(x.resolve()) for x in a.expert]
 if router_info["r"]!=32 or any(x["r"]!=16 for x in experts):raise RuntimeError("MR-LoRA adapter rank mismatch")
 active={"method":METHOD,"router_stage":a.stage,"router":router_info,"experts":[{"task":i+1,**x} for i,x in enumerate(experts)],"stored_adapter_parameters":router_info["adapter_parameters"]+sum(x["adapter_parameters"] for x in experts),"active_router_parameters":router_info["adapter_parameters"],"active_expert_parameters_per_sample":experts[0]["adapter_parameters"],"oracle_task_id":False,"two_pass_inference":True}
 stage_dir=a.output_root.resolve()/"evaluation"/f"stage_{a.stage:02d}";summary_path=stage_dir/"stage_evaluation_summary.json"
 if summary_path.is_file():
  old=json.loads(summary_path.read_text());
  if old.get("status")=="PASS" and old.get("active_adapter")==active:print(f"SKIP completed MR-LoRA evaluation stage {a.stage}");return 0
 started=time.monotonic();all_rows=[]
 for true_task in range(1,a.stage+1):
  for row in read_jsonl(a.data_root.resolve()/TASK_DIRS[true_task]/"test.jsonl",a.eval_limit):all_rows.append((true_task,row))
 router_model,processor=load_adapter(a.router.resolve());routed=[];router_seconds=0.0
 for true_task,row in all_rows:
  tick=time.monotonic();raw=route_generate(router_model,processor,row,a.stage);elapsed=time.monotonic()-tick;router_seconds+=elapsed
  parsed=parse_router_output(raw,list(range(1,a.stage+1)),str(row["id"]),42);routed.append({"true_task_id":true_task,"id":row["id"],"router_latency_seconds":elapsed,**parsed})
 release(router_model);del router_model,processor
 answers=[None]*len(all_rows);answer_seconds=0.0;model_loads=1
 for expert_id in range(1,a.stage+1):
  indices=[i for i,r in enumerate(routed) if r["selected_expert"]==expert_id]
  if not indices:continue
  model,processor=load_adapter(a.expert[expert_id-1].resolve());model_loads+=1
  for i in indices:
   true_task,row=all_rows[i];tick=time.monotonic();prediction=generate(model,processor,row,true_task);elapsed=time.monotonic()-tick;answer_seconds+=elapsed
   answers[i]={**routed[i],"prediction":prediction,"answer_generation_latency_seconds":elapsed,"selected_expert_checkpoint":experts[expert_id-1]["checkpoint"],"selected_expert_weights_sha256":experts[expert_id-1]["adapter_weights_sha256"]}
  release(model);del model,processor
 diagnostics=router_diagnostics(routed,a.stage);atomic(stage_dir/"router_diagnostics.json",diagnostics)
 summaries={}
 for task_id in range(1,a.stage+1):
  selected=[answers[i] for i,(true,_) in enumerate(all_rows) if true==task_id];rows=[row for true,row in all_rows if true==task_id];pred=[x["prediction"] for x in selected]
  metric,details=evaluate_rows(task_id,rows,pred);stem=f"primary_cumulative_task_{task_id:02d}";raw_path=stage_dir/f"{stem}.raw_predictions.jsonl";detail_path=stage_dir/f"{stem}.predictions.jsonl"
  raw_path.parent.mkdir(parents=True,exist_ok=True);raw_path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in selected),encoding="utf-8");detail_path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in details),encoding="utf-8")
  metric.update({"mode":"primary_learned_router_two_pass","stage":a.stage,"eval_task":task_id,"task_name":TASK_NAMES[task_id],"method":METHOD,"evaluation_protocol":EVALUATION_PROTOCOL,"dataset":str(a.data_root.resolve()/TASK_DIRS[task_id]/"test.jsonl"),"dataset_sha256":sha256(a.data_root.resolve()/TASK_DIRS[task_id]/"test.jsonl"),"model_id":MODEL_ID,"model_revision":MODEL_REVISION,"active_adapter":active,"raw_predictions":str(raw_path),"raw_predictions_sha256":sha256(raw_path),"predictions":str(detail_path),"predictions_sha256":sha256(detail_path),"oracle_task_id":False,"fresh_process_reload":True});atomic(stage_dir/f"{stem}.summary.json",metric);summaries[stem]=metric
 status="PASS" if len(summaries)==a.stage and all(x["status"]=="PASS" for x in summaries.values()) else "FAIL"
 final={"status":status,"stage":a.stage,"method":METHOD,"evaluation_protocol":EVALUATION_PROTOCOL,"matrix_scope":"seen_tasks_only","evaluated_task_ids":list(range(1,a.stage+1)),"cell_count":len(summaries),"active_adapter":active,"model_load_count":model_loads,"inference_passes_per_sample":2,"router_inference_seconds":router_seconds,"expert_answer_inference_seconds":answer_seconds,"total_evaluation_seconds":time.monotonic()-started,"router_diagnostics":diagnostics,"cells":summaries};atomic(summary_path,final);print(json.dumps({"status":status,"stage":a.stage,"cells":len(summaries),"router_accuracy":diagnostics["router_accuracy_overall"]}));return 0 if status=="PASS" else 2
if __name__=="__main__":raise SystemExit(main())
