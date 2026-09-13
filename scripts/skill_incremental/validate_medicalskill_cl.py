#!/usr/bin/env python3
"""Validate MedicalSkill-CL schema, paths, exact hashes and leakage."""
import argparse, collections, csv, hashlib, json, re, shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
ROOT=Path("/remote-home/wangbomin/MedicalSkill-CL"); ART=Path("/root/MedAgentCL_v4/artifacts/medicalskill_cl_build")
def cli():
 p=argparse.ArgumentParser(); p.add_argument("--data-root",default=str(ROOT)); p.add_argument("--artifact-root",default=str(ART)); p.add_argument("--tasks",default="all"); p.add_argument("--seed",type=int,default=42); p.add_argument("--dry-run",action="store_true"); p.add_argument("--resume",action="store_true"); p.add_argument("--verify-only",action="store_true"); p.add_argument("--skip-content-hash",action="store_true"); p.add_argument("--hash-workers",type=int,default=8); p.add_argument("--remove-train-exact-overlap",action="store_true",help="Remove train records sharing an exact image SHA256 with any test record; test is immutable"); return p.parse_args()
def rows(p):
 with open(p,encoding="utf-8-sig") as f:
  for n,l in enumerate(f,1):
   if l.strip(): yield json.loads(l)
def atomic(p,x):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=Path(str(p)+".tmp"); t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
 return h.hexdigest()
def phash(p):
 import numpy as np
 n=32
 with Image.open(p) as im: px=np.asarray(im.convert("L").resize((n,n)),dtype=np.float32)
 c=getattr(phash,"_dct",None)
 if c is None:
  k=np.arange(n)[:,None]; x=np.arange(n)[None,:]
  c=np.cos(np.pi*(2*x+1)*k/(2*n)); c[0]*=1/np.sqrt(2); c*=np.sqrt(2/n); phash._dct=c
 low=(c@px@c.T)[:8,:8].reshape(-1); median=float(np.median(low[1:])); bits="".join("1" if v>median else "0" for v in low)
 return f"{int(bits,2):016x}"
PHASH_METHOD="dct64_32x32_v1"
def inspect_image(item):
 path,cached=item; p=Path(path); size=p.stat().st_size
 if cached and cached.get("size")==size and cached.get("phash_method")==PHASH_METHOD and cached.get("sha256") and cached.get("phash"): return cached
 try:return {"path":path,"size":size,"sha256":sha(p),"phash":phash(p),"phash_method":PHASH_METHOD}
 except Exception as e:return {"path":path,"size":size,"error":f"{type(e).__name__}: {e}","phash_method":PHASH_METHOD}
def perceptual_cross_split(percept,max_distance=4,max_examples=1000):
 if not 0<=max_distance<64: raise ValueError("max_distance must be in [0, 63]")
 train={int(h,16) for h,rr in percept.items() if any(x[1]=="train" for x in rr)}
 test={int(h,16) for h,rr in percept.items() if any(x[1]=="test" for x in rr)}
 # With d+1 disjoint chunks, Hamming distance <= d guarantees one exact chunk.
 chunk_count=max_distance+1; base=64//chunk_count; extra=64%chunk_count
 chunks=[]; shift=0
 for index in range(chunk_count):
  width=base+(index<extra); chunks.append((index,shift,(1<<width)-1)); shift+=width
 buckets=collections.defaultdict(list)
 for value in train:
  for index,shift,mask in chunks: buckets[(index,(value>>shift)&mask)].append(value)
 seen=set(); count=0; examples=[]
 for value in test:
  candidates=set()
  for index,shift,mask in chunks: candidates.update(buckets.get((index,(value>>shift)&mask),()))
  for candidate in candidates:
   distance=(value^candidate).bit_count()
   if distance>max_distance: continue
   key=(min(value,candidate),max(value,candidate))
   if key in seen: continue
   seen.add(key); count+=1
   if len(examples)<max_examples:
    left=f"{candidate:016x}"; right=f"{value:016x}"
    refs=percept[left] if left==right else percept[left]+percept[right]
    examples.append({"train_phash":left,"test_phash":right,"distance":distance,"reference_count":len(refs),"refs":refs[:100],"refs_truncated":len(refs)>100})
 return count,examples
def selected(root,v):
 ds=sorted(p for p in root.glob("task_*") if p.is_dir())
 if v=="all": return ds
 ids={int(x) for x in v.split(",")}; return [p for p in ds if int(p.name.split("_")[1]) in ids]
def remove_train_exact_overlap(root,dirs,cross):
 drop=collections.defaultdict(set)
 for group in cross:
  for task,split,sample_id,_path in group["refs"]:
   if split=="train": drop[task].add(sample_id)
 report={"policy":"preserve every test record; remove a whole train record if any image SHA256 occurs in test","tasks":{},"total_removed":0,"test_records_removed":0}
 for d in dirs:
  src=d/"train.jsonl"; data=list(rows(src)); ids=drop.get(d.name,set()); kept=[x for x in data if str(x.get("id")) not in ids]
  if len(kept)!=len(data):
   tmp=Path(str(src)+".tmp")
   with tmp.open("w",encoding="utf-8") as f:
    for x in kept: f.write(json.dumps(x,ensure_ascii=False)+"\n")
   tmp.replace(src)
  removed=len(data)-len(kept); report["tasks"][d.name]={"before":len(data),"after":len(kept),"removed":removed,"removed_ids":sorted(ids)}
  report["total_removed"]+=removed
 report["status"]="REPAIR_APPLIED" if report["total_removed"] else "NO_CHANGES"
 atomic(root/"audits/global_exact_overlap_train_removal.json",report)
 return report
def main():
 a=cli(); root=Path(a.data_root); art=Path(a.artifact_root); dirs=selected(root,a.tasks)
 if a.dry_run: print(json.dumps({"status":"DRY_RUN","tasks":[str(p) for p in dirs]},indent=2)); return 0
 errors=collections.Counter(); refs=collections.defaultdict(list); ids=collections.defaultdict(set); groups=collections.defaultdict(set); global_ids=set(); total=0
 for d in dirs:
  for split in ("train","test"):
   p=d/f"{split}.jsonl"
   if not p.is_file(): errors["missing_jsonl"]+=1; continue
   for x in rows(p):
    total+=1; sample_id=x.get("id"); errors["schema"]+=set(x)!={"id","task_id","skill","messages","images","metadata"}; errors["duplicate_id"]+=sample_id in ids[(d.name,split)]; errors["global_duplicate_id"]+=sample_id in global_ids; ids[(d.name,split)].add(sample_id); global_ids.add(sample_id)
    m=x.get("messages") or []; errors["messages"]+=len(m)!=2 or [v.get("role") for v in m]!=["user","assistant"]; errors["empty_images"]+=not bool(x.get("images"))
    for im in x.get("images") or []:
     pth=Path(im)
     if not pth.is_absolute() or not pth.is_file() or pth.stat().st_size<=0: errors["missing_or_empty_image"]+=1
     else: refs[str(pth)].append((d.name,split,str(x.get("id"))))
    if d.name in ("task_03_concept_recognition","task_04_caption_generation"): groups[(d.name,split)].add(str((x.get("metadata") or {}).get("group_key") or ""))
    if d.name=="task_06_reasoning_vqa":
     prompt=m[0].get("content","").casefold() if m else ""
     forbidden_labels={"imaging_findings":r"(?im)^\s*imaging[_ ]findings\s*:","discussion":r"(?im)^\s*discussion\s*:","image_caption":r"(?im)^\s*image[_ ]caption\s*:","correct_answer":r"(?im)^\s*(?:correct|final)[_ ]answer\s*:"}
     for bad,pattern in forbidden_labels.items(): errors["medthink_prompt_"+bad]+=bool(re.search(pattern,prompt))
 exact=collections.defaultdict(list); percept=collections.defaultdict(list); cachep=root/"audits/image_hash_cache.jsonl"; cache={}
 if a.resume and cachep.is_file(): cache={x["path"]:x for x in rows(cachep)}
 if not a.skip_content_hash:
  cachep.parent.mkdir(parents=True,exist_ok=True); tmp=Path(str(cachep)+".tmp")
  ordered=sorted(refs.items()); n=0
  with tmp.open("w",encoding="utf-8") as f, ThreadPoolExecutor(max_workers=max(1,a.hash_workers)) as pool:
   for start in range(0,len(ordered),10000):
    batch=ordered[start:start+10000]; jobs=((path,cache.get(path)) for path,_rr in batch)
    for (path,rr),c in zip(batch,pool.map(inspect_image,jobs)):
     n+=1
     if c.get("error"): errors["undecodable_image"]+=1
     f.write(json.dumps(c)+"\n")
     if c.get("sha256"):
      for ref in rr: exact[c["sha256"]].append((*ref,path)); percept[c["phash"]].append((*ref,path))
    print(f"hashed={n}/{len(refs)}",flush=True)
  tmp.replace(cachep)
 cross=[]; duplicate=[]
 for h,rr in exact.items():
  if {x[1] for x in rr}=={"train","test"}: cross.append({"sha256":h,"refs":rr})
  if len({x[:3] for x in rr})>1: duplicate.append({"sha256":h,"refs":rr})
 if a.remove_train_exact_overlap and cross:
  repair=remove_train_exact_overlap(root,dirs,cross); atomic(art/"global_exact_overlap_train_removal.json",repair); print(json.dumps(repair,indent=2)); return 0
 near_count,near=perceptual_cross_split(percept,max_distance=4)
 cg=groups[("task_03_concept_recognition","train")]; pg=groups[("task_04_caption_generation","train")]; gover=sorted((cg&pg)-{""})
 leak={"status":"PASS" if not cross and not gover else "FAIL","exact_train_test_overlap_count":len(cross),"concept_caption_train_group_overlap_count":len(gover),"concept_caption_examples":gover[:50],"perceptual_candidate_group_count":near_count,"schema_path_errors":{k:v for k,v in errors.items() if v}}
 er={"status":"PASS" if not cross else "FAIL","unique_image_paths":len(refs),"unique_sha256":len(exact),"cross_split_duplicate_group_count":len(cross),"cross_split_duplicate_examples":cross[:1000],"cross_split_examples_truncated":len(cross)>1000,"all_duplicate_group_count":len(duplicate),"all_duplicate_examples":duplicate[:1000],"all_duplicate_examples_truncated":len(duplicate)>1000}
 pr={"status":"REVIEW" if near_count else "PASS","method":"64-bit DCT perceptual hash (32x32 input, 8x8 low frequencies) with Hamming distance <= 4; candidates require manual review","hamming_distance_threshold":4,"candidate_group_count":near_count,"candidate_examples":near,"examples_truncated":near_count>len(near)}
 atomic(root/"audits/global_exact_duplicate_audit.json",er); atomic(root/"audits/global_perceptual_duplicate_audit.json",pr); atomic(root/"audits/global_train_test_leakage_report.json",leak); atomic(art/"global_train_test_leakage_report.json",leak)
 names=[p.name for p in dirs]; sets={n:{h for h,rr in exact.items() if any(x[0]==n for x in rr)} for n in names} if exact else {n:{p for p,rr in refs.items() if any(x[0]==n for x in rr)} for n in names}; matrix=root/"audits/cross_skill_overlap_matrix.csv"; matrix.parent.mkdir(parents=True,exist_ok=True)
 with matrix.open("w",newline="",encoding="utf-8") as f:
  w=csv.writer(f); w.writerow(["task",*names])
  for left in names: w.writerow([left,*[len(sets[left]&sets[right]) for right in names]])
 art.mkdir(parents=True,exist_ok=True); (art/"cross_skill_overlap_matrix.csv").write_text(matrix.read_text(),encoding="utf-8")
 report=["# MedicalSkill-CL validation","",f"Status: **{leak['status']}**","",f"Records: {total}",f"Unique image paths: {len(refs)}",f"Exact train/test leakage groups: {len(cross)}",f"Concept/Caption train group overlap: {len(gover)}",f"Perceptual near-duplicate candidates (Hamming <= 4): {near_count}",f"Schema/path errors: {dict(errors)}"]
 (art/"validation_report.md").write_text("\n".join(report)+"\n",encoding="utf-8"); print(json.dumps(leak,indent=2)); return 0 if leak["status"]=="PASS" and sum(errors.values())==0 else 1
if __name__=="__main__": raise SystemExit(main())
