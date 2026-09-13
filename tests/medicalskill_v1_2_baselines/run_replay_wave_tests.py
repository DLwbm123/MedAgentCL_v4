#!/usr/bin/env python3
"""CPU/static contract tests for replay, MR-LoRA, and Replay+LoRA."""
from __future__ import annotations
import json,subprocess,sys,tempfile
from pathlib import Path
import torch
from torch import nn
from peft import LoraConfig,get_peft_model

ROOT=Path("/root/MedAgentCL_v4")
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from med_prism.baselines.mrlora import assert_stage_router,parse_router_output,router_diagnostics
from scripts.medicalskill_v1_2_replay.replay_memory import ROUTER_ONLY,TRAINING_REPLAY,atomic_jsonl,build_memory,concatenate_current_and_replay,read_jsonl,task_quotas
DATA=Path("/remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k")

class Toy(nn.Module):
 def __init__(self):super().__init__();self.q_proj=nn.Linear(16,16);self.v_proj=nn.Linear(16,16)
 def forward(self,x):return self.v_proj(torch.tanh(self.q_proj(x)))
def lora(rank):return get_peft_model(Toy(),LoraConfig(r=rank,lora_alpha=2*rank,target_modules=["q_proj","v_proj"],lora_dropout=0,bias="none"))

def test_memory(tmp):
 r1=build_memory(data_root=DATA,stage=2,purpose=ROUTER_ONLY,output_dir=tmp/"router1",examples_per_task=20,include_current=True)
 r2=build_memory(data_root=DATA,stage=2,purpose=ROUTER_ONLY,output_dir=tmp/"router2",examples_per_task=20,include_current=True)
 assert r1["memory_count"]==40 and r1["count_per_task"]=={"1":20,"2":20}
 assert r1["resulting_buffer_hash"]==r2["resulting_buffer_hash"] and r1["leakage_audit"]["status"]=="PASS"
 records=[json.loads(x) for x in (tmp/"router1/records.jsonl").read_text().splitlines()]
 assert all(x["answer_hash"] is None and not x["answer_retained"] and x["task_id"]<=2 for x in records)
 router_rows=read_jsonl(tmp/"router1/router_train.jsonl");assert all(len(x["messages"])==2 and x["messages"][-1]["content"].startswith("EXPERT_") for x in router_rows)
 pre=build_memory(data_root=DATA,stage=2,purpose=TRAINING_REPLAY,output_dir=tmp/"pre",total_budget=100,include_current=False)
 assert pre["memory_count"]==100 and pre["count_per_task"]=={"1":100}
 assert all(json.loads(x)["task_id"]==1 for x in (tmp/"pre/records.jsonl").read_text().splitlines())
 post=build_memory(data_root=DATA,stage=3,purpose=TRAINING_REPLAY,output_dir=tmp/"post",total_budget=101,include_current=True)
 assert post["count_per_task"]=={"1":34,"2":34,"3":33} and task_quotas(1000,[1,2,3])=={1:334,2:333,3:333}
 mix=concatenate_current_and_replay(DATA/"task_02_diagnosis_classification/train.jsonl",tmp/"pre/replay_train.jsonl",tmp/"mix.jsonl",current_limit=3)
 assert mix["current_sample_count"]==3 and mix["replay_sample_count"]==100 and mix["total_effective_training_examples"]==103
 mixed=read_jsonl(tmp/"mix.jsonl");assert all(x.get("messages") and isinstance(x.get("images"),list) for x in mixed)
 hetero=[]
 for task_name in ("task_01_vqa","task_02_diagnosis_classification","task_03_concept_recognition","task_04_visual_grounding","task_05_reasoning_vqa"):
  row=read_jsonl(DATA/task_name/"train.jsonl")[0]
  hetero.append({"id":row["id"],"messages":row["messages"],"images":row["images"]})
 hetero_path=tmp/"all_five_interfaces.jsonl";atomic_jsonl(hetero_path,hetero)
 from swift.dataset import load_dataset
 swift_rows,_=load_dataset([str(hetero_path)],split_dataset_ratio=0,use_hf=True)
 assert len(swift_rows)==5 and all(swift_rows[i]["messages"] and swift_rows[i]["images"] for i in range(5))
 bad=tmp/"bad";bad.mkdir();(bad/"manifest.json").write_text(json.dumps({**r1,"memory_budget":999}))
 try:build_memory(data_root=DATA,stage=2,purpose=ROUTER_ONLY,output_dir=bad,examples_per_task=20,include_current=True)
 except RuntimeError:pass
 else:raise AssertionError("Incompatible replay resume was accepted")

def test_mrlora_logic():
 assert_stage_router(2,2,[1,2])
 for bad in [(2,5,[1,2]),(2,2,[1,2,3])]:
  try:assert_stage_router(*bad)
  except RuntimeError:pass
  else:raise AssertionError("future router/expert state accepted")
 valid=parse_router_output("EXPERT_02",[1,2],"x");assert valid["selected_expert"]==2 and not valid["fallback_used"]
 invalid=parse_router_output("answer the question",[1,2],"x");assert invalid["fallback_used"] and invalid["selected_expert"] in [1,2]
 assert invalid==parse_router_output("answer the question",[1,2],"x")
 diag=router_diagnostics([{"true_task_id":1,**parse_router_output("EXPERT_01",[1,2],"a")},{"true_task_id":2,**parse_router_output("EXPERT_01",[1,2],"b")}],2)
 assert diag["router_accuracy_overall"]==.5 and diag["confusion_matrix"]["2"]["1"]==1
 expert=lora(16);router=lora(32);expert.requires_grad_(False);loss=router(torch.randn(2,3,16)).square().mean();loss.backward()
 assert any(p.grad is not None for n,p in router.named_parameters() if "lora_" in n) and all(p.grad is None for p in expert.parameters())

def test_static_contracts(tmp):
 mr=(ROOT/"scripts/medicalskill_v1_2_mrlora/run_mrlora_v1_2.sh").read_text();rp=(ROOT/"scripts/medicalskill_v1_2_replay_lora/run_replay_lora_v1_2.sh").read_text();ev=(ROOT/"scripts/medicalskill_v1_2_mrlora/evaluate_mrlora_v1_2.py").read_text()
 assert "--lora_rank \"$RANK\"" in mr and "--adapters" not in mr and "--lora_rank 48" in rp and "--adapters \"$PREVIOUS_CHECKPOINT\"" in rp
 assert "route_generate" in ev and "selected_expert" in ev and "true_task" not in ev.split("parse_router_output",1)[1].split(";",1)[0]
 for script,args in [("scripts/medicalskill_v1_2_mrlora/run_mrlora_v1_2.sh",["--rank","16","--router-rank","32","--router-memory-per-task","20","--router-epochs","30"]),("scripts/medicalskill_v1_2_replay_lora/run_replay_lora_v1_2.sh",["--rank","48","--replay-budget","1000","--buffer-policy","balanced_per_task_stratified_stable_sha256_v1"])]:
  result=subprocess.run([str(ROOT/script),"--data-root",str(DATA),"--output-root",str(tmp/Path(script).parent.name),*args,"--check-only"],cwd=ROOT,text=True,capture_output=True)
  assert result.returncode==0,(result.stdout,result.stderr)

def main():
 with tempfile.TemporaryDirectory(prefix="replay_wave_tests_") as value:
  tmp=Path(value);test_memory(tmp);print("PASS replay determinism/budget/balance/leakage/resume/mixture")
  test_mrlora_logic();print("PASS MR-LoRA routing/temporal/gradient/frozen-expert diagnostics")
  test_static_contracts(tmp);print("PASS rank/lifecycle/check-only contracts")
 print("REPLAY BASELINE WAVE CPU TESTS PASS")
if __name__=="__main__":main()
