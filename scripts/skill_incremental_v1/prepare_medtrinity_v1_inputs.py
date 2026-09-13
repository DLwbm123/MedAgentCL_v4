#!/usr/bin/env python3
from __future__ import annotations
import argparse,collections,json,random
from pathlib import Path
V0=Path("/remote-home/wangbomin/MedicalSkill-CL");V1=Path("/remote-home/wangbomin/MedicalSkill-CL-v1");ART=Path("/root/MedAgentCL_v4/artifacts/medicalskill_cl_v1_closure")
def rows(p):
 with open(p,encoding="utf-8-sig") as f:
  for n,line in enumerate(f,1):
   if line.strip():
    x=json.loads(line)
    if not isinstance(x,dict):raise RuntimeError(f"non-object {p}:{n}")
    yield x
def writej(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=Path(str(p)+".tmp");t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");t.replace(p)
def writejl(p,xs):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=Path(str(p)+".tmp")
 with t.open("w",encoding="utf-8") as f:
  for x in xs:f.write(json.dumps(x,ensure_ascii=False)+"\n")
 t.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--v0-root",default=str(V0));ap.add_argument("--v1-root",default=str(V1));ap.add_argument("--artifact-root",default=str(ART));ap.add_argument("--pilot-size",type=int,default=500);ap.add_argument("--seed",type=int,default=42);a=ap.parse_args()
 v0,v1,art=map(Path,(a.v0_root,a.v1_root,a.artifact_root));old={x["id"]:x for x in rows(v0/"_concept/full.jsonl")};out=[];seen=set()
 for kind,d in (("concept",v0/"task_03_concept_recognition"),("caption",v0/"task_04_caption_generation")):
  for split in ("train","test"):
   for x in rows(d/f"{split}.jsonl"):
    i=str(x["id"]);m=x.get("metadata") or {}
    if i in seen:raise RuntimeError(f"duplicate id {i}")
    seen.add(i);cap=str((old.get(i) or {}).get("source_caption") if kind=="concept" else m.get("original_caption") or m.get("cleaned_caption") or "").strip()
    if not cap:raise RuntimeError(f"empty caption {i}")
    out.append({"id":i,"source_caption":cap,"split":split,"source_kind":kind,"modality":str(m.get("modality") or "unknown").casefold(),"group_key":str(m.get("group_key") or ""),"images":[str(v) for v in x.get("images") or []],"source_metadata":{k:m.get(k) for k in ("source_dataset","source_split","source_id","case_id","patient_id","image_sha256s")}})
 out.sort(key=lambda x:x["id"]);groups=collections.defaultdict(list)
 for x in out:groups[(x["source_kind"],x["split"],x["modality"])].append(x)
 rng=random.Random(a.seed)
 for g in groups.values():rng.shuffle(g)
 quotas={k:max(1,int(a.pilot_size*len(g)/len(out))) for k,g in groups.items()};left=min(a.pilot_size,len(out))-sum(quotas.values())
 ranked=sorted(groups,key=lambda k:(-(a.pilot_size*len(groups[k])/len(out)-quotas[k]),k))
 for k in ranked[:max(0,left)]:quotas[k]+=1
 while left<0:
  for k in reversed(ranked):
   if quotas[k]>1:
    quotas[k]-=1;left+=1
    if left==0:break
 pilot=[]
 for k in sorted(groups):pilot.extend(groups[k][:quotas[k]])
 rng.shuffle(pilot);base=v1/"_concept";writejl(base/"merged_input.jsonl",out);writejl(base/"pilot500_input.jsonl",pilot)
 report={"status":"PASS","seed":a.seed,"v0_root_read_only":str(v0),"v1_root":str(v1),"total":len(out),"source_kind_counts":dict(collections.Counter(x["source_kind"] for x in out)),"split_counts":dict(collections.Counter(x["split"] for x in out)),"pilot_count":len(pilot),"pilot_stratum_quotas":{"/".join(k):v for k,v in sorted(quotas.items())},"duplicate_ids":0,"empty_captions":0}
 writej(base/"input_manifest.json",report);writej(art/"medtrinity_merged_input_manifest.json",report);print(json.dumps(report,indent=2))
if __name__=="__main__":main()
