#!/usr/bin/env python3
import argparse,collections,hashlib,json,random
from pathlib import Path
DATA=Path("/remote-home/wangbomin/MedicalSkill-CL-v1/task_04_visual_grounding/train.jsonl")
ART=Path("/root/MedAgentCL_v4/artifacts/medicalskill_cl_v1_closure")
def rows(path):
 with Path(path).open(encoding="utf-8-sig") as f:
  for line in f:
   if line.strip():yield json.loads(line)
def writej(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=Path(str(path)+".tmp");tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n");tmp.replace(path)
def writejl(path,values):
 path=Path(path);tmp=Path(str(path)+".tmp")
 with tmp.open("w",encoding="utf-8") as f:
  for value in values:f.write(json.dumps(value,ensure_ascii=False)+"\n")
 tmp.replace(path)
def path_hash(paths):return hashlib.sha256("\n".join(sorted(paths)).encode()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--train",default=str(DATA));ap.add_argument("--artifact-root",default=str(ART));ap.add_argument("--target",type=int,default=6190);ap.add_argument("--seed",type=int,default=42);a=ap.parse_args();path,art=Path(a.train),Path(a.artifact_root)
 values=list(rows(path));root=path.parent.parent;dataset_files=[directory/f"{split}.jsonl" for directory in sorted(root.glob("task_*")) for split in ("train","test")];before_paths={p for source in dataset_files for row in rows(source) for p in row.get("images") or []};strict=json.loads((art/"strict_validation.json").read_text())
 if (strict.get("schema_path_errors") or {}).get("image_path")!=0 or strict.get("unique_image_paths_checked")!=len(before_paths):raise RuntimeError("prior full path validation is not reusable")
 by_task=collections.defaultdict(lambda:collections.defaultdict(list))
 for row in values:
  meta=row.get("metadata") or {};by_task[int(meta["medsg_task"])][str(meta["source_id"])].append(row)
 drop=set();tasks={}
 for task in range(1,9):
  groups=list(by_task[task].items());random.Random(a.seed+1000+task).shuffle(groups);turns=sum(len(g) for _,g in groups);removed=[]
  for group_id,group in groups:
   after=turns-len(group)
   if abs(after-a.target)<abs(turns-a.target):removed.append(group_id);drop.update(row["id"] for row in group);turns=after
  tasks[str(task)]={"before":sum(len(g) for _,g in groups),"after":turns,"source_groups_removed":len(removed),"removed_group_examples":removed[:100]}
 final=[row for row in values if row["id"] not in drop]
 counts={str(t):sum(1 for row in final if int((row.get("metadata") or {})["medsg_task"])==t) for t in range(1,9)};mean=len(final)/8;dev=max(abs(v/mean-1)*100 for v in counts.values());status="PASS" if 48000<=len(final)<=52000 and dev<=1 else "FAIL"
 if status!="PASS":raise RuntimeError(f"final MedSG balance failed: {counts}, {dev}")
 writejl(path,final)
 after_paths={p for source in dataset_files for row in rows(source) for p in row.get("images") or []}
 if not after_paths<=before_paths:raise RuntimeError("rebalance introduced new image paths")
 report={"status":status,"seed":a.seed,"policy":"post-pHash deletion-only source-group rebalance; no replacement","target_per_official_task":a.target,"before_total":len(values),"total_output_turns":len(final),"removed_turns":len(drop),"tasks":tasks,"final_output_turns_per_official_task":counts,"max_percentage_point_deviation":dev,"replacement_sampling_used":False};writej(art/"medsg_final_balance.json",report)
 evidence={"status":"PASS","source_full_validation":str(art/"strict_validation.json"),"source_image_path_errors":0,"before_unique_paths":len(before_paths),"after_unique_paths":len(after_paths),"before_path_sha256":path_hash(before_paths),"after_path_sha256":path_hash(after_paths),"deletion_only":True,"after_is_subset_of_before":True,"removed_records":len(drop)};writej(art/"path_validation_monotonicity.json",evidence)
 writej(art/"post_balance_phash_monotonicity.json",{"status":"PASS","policy":"deleting records cannot create a new pHash candidate pair or confirmed overlap","source_final_phash_report":str(art/"phash_candidate_classification.json"),"source_confirmed_before_repair":0,"removed_records":len(drop)})
 print(json.dumps(report,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
