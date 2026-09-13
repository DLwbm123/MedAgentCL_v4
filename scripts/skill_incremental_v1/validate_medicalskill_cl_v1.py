#!/usr/bin/env python3
"""Strict validation for the five-skill MedicalSkill-CL-v1 release."""
from __future__ import annotations
import argparse,collections,concurrent.futures,hashlib,json,re
from pathlib import Path
ROOT=Path("/remote-home/wangbomin/MedicalSkill-CL-v1");ART=Path("/root/MedAgentCL_v4/artifacts/medicalskill_cl_v1_closure")
EXPECTED={1:("task_01_vqa","vqa"),2:("task_02_diagnosis_classification","diagnosis_classification"),3:("task_03_concept_recognition","concept_recognition"),4:("task_04_visual_grounding","visual_grounding"),5:("task_05_reasoning_vqa","reasoning_vqa")}
def rows(p):
 with open(p,encoding="utf-8-sig") as f:
  for n,line in enumerate(f,1):
   if line.strip():
    try:x=json.loads(line)
    except Exception as e:raise RuntimeError(f"{p}:{n}: {e}")
    yield x
def writej(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=Path(str(p)+".tmp");t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n");t.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--data-root",default=str(ROOT));ap.add_argument("--artifact-root",default=str(ART));ap.add_argument("--reuse-path-validation");a=ap.parse_args();root,art=Path(a.data_root),Path(a.artifact_root);errors=collections.Counter();ids=set();test_tokens=set();train_by_token=collections.defaultdict(set);counts={};image_paths=set()
 for task,(name,skill) in EXPECTED.items():
  directory=root/name;counts[name]={}
  for split in ("train","test"):
   count=0
   for x in rows(directory/f"{split}.jsonl"):
    count+=1
    if set(x)!={"id","task_id","skill","messages","images","metadata"}:errors["schema_keys"]+=1
    if x.get("task_id")!=task or x.get("skill")!=skill:errors["task_skill"]+=1
    if x.get("id") in ids:errors["global_duplicate_id"]+=1
    ids.add(x.get("id"));messages=x.get("messages") or []
    if len(messages)!=2 or [m.get("role") for m in messages]!=["user","assistant"]:errors["messages"]+=1
    if not str(messages[-1].get("content") or "").strip():errors["empty_target"]+=1
    if not x.get("images"):errors["empty_images"]+=1
    for image in x.get("images") or []:
     p=Path(image)
     if not p.is_absolute():errors["image_path"]+=1
     else:image_paths.add(str(p))
    tokens=set((x.get("metadata") or {}).get("canonical_lineage_tokens") or [])
    if not tokens:errors["empty_lineage"]+=1
    if split=="test":test_tokens.update(tokens)
    else:
     for token in tokens:train_by_token[token].add(task)
   counts[name][split]=count
 def image_ok(path):
  try:p=Path(path);return p.is_file() and p.stat().st_size>0
  except OSError:return False
 path_reuse={}
 if a.reuse_path_validation:
  evidence=json.loads(Path(a.reuse_path_validation).read_text());current_hash=hashlib.sha256("\n".join(sorted(image_paths)).encode()).hexdigest();reused=evidence.get("status")=="PASS" and evidence.get("source_image_path_errors")==0 and evidence.get("deletion_only") is True and evidence.get("after_is_subset_of_before") is True and evidence.get("after_unique_paths")==len(image_paths) and evidence.get("after_path_sha256")==current_hash
  if not reused:errors["path_validation_reuse"]+=1
  path_reuse={"reused":reused,"evidence":str(a.reuse_path_validation),"current_path_sha256":current_hash}
 else:
  with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
   errors["image_path"]+=sum(not ok for ok in pool.map(image_ok,sorted(image_paths)))
 train_test={token for token in train_by_token if token in test_tokens};cross_task={token:sorted(tasks) for token,tasks in train_by_token.items() if len(tasks)>1}
 if train_test:errors["train_test_lineage_groups"]+=len(train_test)
 if cross_task:errors["cross_task_train_groups"]+=len(cross_task)
 medsg=list(rows(root/"task_04_visual_grounding/train.jsonl"));dist=collections.Counter(int(x["metadata"]["medsg_task"]) for x in medsg);target=sum(dist.values())/8;deviation=max(abs(v/target-1)*100 for v in dist.values())
 if not 48000<=len(medsg)<=52000:errors["medsg_total"]+=1
 if deviation>1:errors["medsg_balance"]+=1
 forbidden=[]
 for path in [root/"configs/effective_config.json",root/"configs/metric_contracts.json",root/"manifests/medicalskill_cl_v1_manifest.json"]:
  text=path.read_text()
  if "caption_generation" in text or "task_06" in text:forbidden.append(str(path))
 if forbidden:errors["removed_skill_reference"]+=len(forbidden)
 report={"status":"PASS" if not errors else "FAIL","counts":counts,"global_unique_ids":len(ids),"unique_image_paths_checked":len(image_paths),"image_path_workers":0 if a.reuse_path_validation else 32,"path_validation_reuse":path_reuse,"schema_path_errors":dict(errors),"train_test_lineage_group_count":len(train_test),"cross_task_train_group_count":len(cross_task),"train_test_lineage_examples":sorted(train_test)[:100],"cross_task_train_examples":dict(list(sorted(cross_task.items()))[:100]),"medsg_output_turns_per_official_task":dict(sorted(dist.items())),"medsg_total":len(medsg),"medsg_max_percentage_point_deviation":deviation,"removed_sixth_skill_absent_from_v1_contracts":not forbidden}
 writej(art/"strict_validation.json",report);writej(root/"audits/strict_validation.json",report)
 md=["# MedicalSkill-CL-v1 validation","",f"Status: **{report['status']}**","",f"Global records: {len(ids)}",f"Train/test lineage groups: {len(train_test)}",f"Cross-task train lineage groups: {len(cross_task)}",f"MedSG train turns: {len(medsg)}",f"Schema/path errors: {dict(errors)}"]
 (art/"validation_report.md").write_text("\n".join(md)+"\n");print(json.dumps(report,indent=2));return 0 if report["status"]=="PASS" else 1
if __name__=="__main__":raise SystemExit(main())
