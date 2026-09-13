#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import swift
def writej(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=Path(str(p)+".tmp");t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n");t.replace(p)
def main():
 a=argparse.ArgumentParser();a.add_argument("--data-root",default="/remote-home/wangbomin/MedicalSkill-CL-v1");a.add_argument("--output",default="/root/MedAgentCL_v4/artifacts/medicalskill_cl_v1_closure/swift_schema_smoke.json");o=a.parse_args();tasks={}
 for d in sorted(Path(o.data_root).glob("task_*")):
  source=d/"train.jsonl";ds,val=swift.load_dataset(str(source),split_dataset_ratio=0,seed=42,num_proc=1,shuffle=False,use_hf=True,strict=True,remove_unused_columns=False);samples=[]
  for row in ds.select(range(min(2,len(ds)))):
   images=[{"type":type(v).__name__,"keys":sorted(v) if isinstance(v,dict) else [],"path":v.get("path") if isinstance(v,dict) else str(v)} for v in row.get("images") or []]
   samples.append({"keys":sorted(row),"roles":[v.get("role") for v in row.get("messages") or []],"image_count":len(images),"images":images})
  tasks[d.name]={"status":"PASS","source":str(source),"rows_loaded":len(ds),"validation_dataset_is_none":val is None,"rows":samples}
 report={"status":"PASS","swift_version":swift.__version__,"offline_local_files_only":True,"model_loaded":False,"gpu_used":False,"strict":True,"use_hf":True,"split_dataset_ratio":0,"full_jsonl_parsed_before_two_row_selection":True,"tasks":tasks};writej(o.output,report);print(json.dumps({"status":"PASS","swift_version":swift.__version__,"tasks":tasks},indent=2))
if __name__=="__main__":main()
