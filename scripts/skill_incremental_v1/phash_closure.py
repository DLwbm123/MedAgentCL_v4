#!/usr/bin/env python3
"""Classify and close confirmed perceptual duplicate groups in v1."""
from __future__ import annotations
import argparse,collections,concurrent.futures,functools,json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path("/remote-home/wangbomin/MedicalSkill-CL-v1");V0=Path("/remote-home/wangbomin/MedicalSkill-CL");ART=Path("/root/MedAgentCL_v4/artifacts/medicalskill_cl_v1_closure")
PRIORITY={5:0,3:1,2:2,1:3,4:4}
def rows(p):
 with open(p,encoding="utf-8-sig") as f:
  for line in f:
   if line.strip():yield json.loads(line)
def writej(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=Path(str(p)+".tmp");t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n");t.replace(p)
def writejl(p,xs):
 p=Path(p);t=Path(str(p)+".tmp")
 with t.open("w",encoding="utf-8") as f:
  for x in xs:f.write(json.dumps(x,ensure_ascii=False)+"\n")
 t.replace(p)
def cache_for(paths,cache):
 out={}
 for x in rows(cache):
  if x.get("path") in paths:out[x["path"]]=x
 return out
def boundaries(refs):
 splits={x[1] for x in refs};train_tasks={x[0] for x in refs if x[1]=="train"}
 return splits=={"train","test"} or len(train_tasks)>1
def candidates(percept,d=4):
 values={int(h,16) for h in percept};chunks=[];base=64//(d+1);extra=64%(d+1);shift=0
 for i in range(d+1):
  width=base+(i<extra);chunks.append((i,shift,(1<<width)-1));shift+=width
 buckets=collections.defaultdict(list)
 for value in values:
  for i,shift,mask in chunks:buckets[(i,(value>>shift)&mask)].append(value)
 seen=set()
 for value in values:
  possible=set()
  for i,shift,mask in chunks:possible.update(buckets[(i,(value>>shift)&mask)])
  for other in possible:
   key=(min(value,other),max(value,other))
   if key in seen or (value^other).bit_count()>d:continue
   seen.add(key);left,right=(f"{key[0]:016x}",f"{key[1]:016x}");refs=percept[left] if left==right else percept[left]+percept[right]
   if boundaries(refs):yield left,right,(value^other).bit_count(),refs
@functools.lru_cache(maxsize=None)
def pixels(path):
 with Image.open(path) as im:return np.asarray(im.convert("L").resize((128,128)),dtype=np.float32)
def warm_pixels(path):
 try:pixels(path);return None
 except Exception as e:return {"path":path,"error":f"{type(e).__name__}: {e}"}
def similarity(a,b):
 x,y=pixels(a),pixels(b);mx,my=float(x.mean()),float(y.mean());vx,vy=float(x.var()),float(y.var());cov=float(((x-mx)*(y-my)).mean());c1=6.5025;c2=58.5225
 return float(((2*mx*my+c1)*(2*cov+c2))/((mx*mx+my*my+c1)*(vx+vy+c2))),float(x.std()),float(y.std())
def classify(refs,left,right):
 grouped=collections.defaultdict(list)
 for ref in refs:
  if len(grouped[(ref[0],ref[1],ref[4])])<3:grouped[(ref[0],ref[1],ref[4])].append(ref)
 representatives=[x for group in grouped.values() for x in group]
 left_refs=[x for x in representatives if x[4]==left];right_refs=[x for x in representatives if x[4]==right]
 if left==right:left_refs,right_refs=representatives,representatives
 pairs=[]
 for a in left_refs:
  for b in right_refs:
   if a[3]==b[3]:continue
   if not ({a[1],b[1]}=={"train","test"} or (a[1]==b[1]=="train" and a[0]!=b[0])):continue
   try:s,sa,sb=similarity(a[3],b[3]);pairs.append((s,sa,sb,a,b))
   except Exception:continue
 if not pairs:return "unresolved",None
 source_pairs=[pair for pair in pairs if set(pair[3][5])&set(pair[4][5])]
 if source_pairs:return "confirmed_same_source_case",max(source_pairs,key=lambda x:x[0])
 best=max(pairs,key=lambda x:x[0]);s,sa,sb,a,b=best
 if s>=0.995:return "confirmed_same_image_transformation",best
 if min(sa,sb)<4 and s<0.95:return "low_information_false_positive",best
 if s>=0.98:return "likely_near_duplicate",best
 if s<0.80:return "low_information_false_positive",best
 return "unresolved",best
def collect(root,cache_path):
 records=[];paths=set()
 for d in sorted(root.glob("task_*")):
  task=int(d.name.split("_")[1])
  for split in ("train","test"):
   for row in rows(d/f"{split}.jsonl"):
    records.append((task,split,row));paths.update(row.get("images") or [])
 cache=cache_for(paths,cache_path);percept=collections.defaultdict(list)
 for task,split,row in records:
  tokens=(row.get("metadata") or {}).get("canonical_lineage_tokens") or []
  for path in row.get("images") or []:
   h=cache.get(path)
   if h and h.get("phash"):percept[h["phash"]].append((task,split,row["id"],path,h["phash"],tokens,h.get("sha256")))
 return records,percept
def repair(root,records,classifications):
 counts=collections.Counter(task for task,split,row in records if split=="train");drop=collections.defaultdict(set);decisions=[]
 for item in classifications:
  if not item["classification"].startswith("confirmed_"):continue
  refs=item["_all_refs"];has_test=any(x[1]=="test" for x in refs)
  if has_test:
   for x in refs:
    if x[1]=="train":drop[x[0]].add(x[2])
   policy="test_preserved"
  else:
   tasks={x[0] for x in refs if x[1]=="train"}
   if len(tasks)<2:continue
   owner=min(tasks,key=lambda t:(counts[t],PRIORITY[t],t))
   for x in refs:
    if x[1]=="train" and x[0]!=owner:drop[x[0]].add(x[2])
   policy=f"train_owner_task_{owner}"
  decisions.append({"candidate":item["candidate_id"],"classification":item["classification"],"policy":policy})
 for d in sorted(root.glob("task_*")):
  task=int(d.name.split("_")[1]);path=d/"train.jsonl";data=list(rows(path));writejl(path,[x for x in data if x["id"] not in drop[task]])
 return {"removed_per_task":{str(k):len(v) for k,v in sorted(drop.items())},"total_removed":sum(map(len,drop.values())),"decisions":decisions}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--data-root",default=str(ROOT));ap.add_argument("--cache",default=str(V0/"audits/image_hash_cache.jsonl"));ap.add_argument("--artifact-root",default=str(ART));ap.add_argument("--apply-confirmed-repair",action="store_true");ap.add_argument("--io-workers",type=int,default=16);a=ap.parse_args();root,art=Path(a.data_root),Path(a.artifact_root)
 records,percept=collect(root,Path(a.cache));groups=list(candidates(percept));candidate_paths=sorted({ref[3] for _,_,_,refs in groups for ref in refs});prefetch_failures=[]
 if a.io_workers>0:
  with concurrent.futures.ThreadPoolExecutor(max_workers=a.io_workers) as pool:
   for failure in pool.map(warm_pixels,candidate_paths):
    if failure:prefetch_failures.append(failure)
 classified=[];counts=collections.Counter()
 for index,(left,right,distance,refs) in enumerate(groups):
  label,best=classify(refs,left,right);counts[label]+=1
  detail=None if best is None else {"ssim":best[0],"std_left":best[1],"std_right":best[2],"left_path":best[3][3],"right_path":best[4][3]}
  classified.append({"candidate_id":f"phash_{index:07d}","left_phash":left,"right_phash":right,"distance":distance,"classification":label,"best_comparison":detail,"_all_refs":refs,"refs":[list(x) for x in refs[:100]],"refs_truncated":len(refs)>100})
 repair_report=repair(root,records,classified) if a.apply_confirmed_repair else {"total_removed":0,"removed_per_task":{},"decisions":[]}
 for item in classified:item.pop("_all_refs",None)
 report={"status":"REPAIRED" if repair_report["total_removed"] else "PASS","method":"64-bit DCT pHash radius 4 candidate generation; canonical lineage and normalized grayscale SSIM used only to classify, never to delete merely by pHash distance","thresholds":{"confirmed_ssim":0.995,"likely_ssim":0.98,"false_positive_ssim_below":0.80,"low_information_std_below":4},"candidate_group_count":len(classified),"pixel_prefetch":{"workers":a.io_workers,"unique_paths":len(candidate_paths),"failure_count":len(prefetch_failures),"failure_examples":prefetch_failures[:100]},"classification_counts":dict(counts),"unresolved_count":counts["unresolved"],"confirmed_before_repair":counts["confirmed_same_source_case"]+counts["confirmed_same_image_transformation"],"repair":repair_report,"candidates":classified}
 writej(art/"phash_candidate_classification.json",report);writej(root/"audits/phash_candidate_classification.json",report);print(json.dumps({k:report[k] for k in ("status","candidate_group_count","classification_counts","unresolved_count","confirmed_before_repair","repair")},indent=2))
if __name__=="__main__":main()
