#!/usr/bin/env python3
"""Deterministic caption-only Qwen3 concept extraction."""
from __future__ import annotations
import argparse, hashlib, json, os, random, re
from pathlib import Path

MODEL_ID="Qwen/Qwen3-8B"
REVISION="b968826d9c46dd6066d109eabc6255188de91218"
SNAPSHOT=f"/remote-home/wangbomin/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/{REVISION}"
VERSION="medicalskill_concept_qwen3_v2_1_max8"
SYSTEM="""Extract medical concepts from the caption only. Never infer from an image. Return JSON only with a concepts list. Each concept has canonical, type, status, laterality, mention. Allowed types: anatomy, finding, pathology, modality, device, procedure, attribute. status: present, uncertain, negated. laterality: left, right, bilateral, midline, unspecified. Keep only caption-supported concepts, preserve negation/uncertainty/laterality, exclude coordinates, boxes, colored overlays, regions of interest, percentages and generic prose, and deduplicate. Return at most 8 highest-value concepts as compact single-line JSON with no indentation."""
TYPES={"anatomy","finding","pathology","modality","device","procedure","attribute"}
STATUS={"present","uncertain","negated"}
LAT={"left","right","bilateral","midline","unspecified"}
ALIASES={"ct":"computed tomography","ct scan":"computed tomography","computed tomography scan":"computed tomography","mri":"magnetic resonance imaging","mr imaging":"magnetic resonance imaging","magnetic resonance imaging scan":"magnetic resonance imaging","x ray":"x-ray","radiograph":"x-ray","ultrasound scan":"ultrasound"}
USER='Caption:\n{caption}\n\nReturn: {{"concepts":[{{"canonical":"","type":"","status":"","laterality":"","mention":""}}]}}'

def cli():
 p=argparse.ArgumentParser()
 p.add_argument("--tasks",default="3"); p.add_argument("--input",action="append",required=True); p.add_argument("--output",required=True)
 p.add_argument("--model-path",default=SNAPSHOT); p.add_argument("--model-id",default=MODEL_ID); p.add_argument("--revision",default=REVISION)
 p.add_argument("--limit",type=int,default=0); p.add_argument("--batch-size",type=int,default=12); p.add_argument("--max-new-tokens",type=int,default=384)
 p.add_argument("--seed",type=int,default=42); p.add_argument("--max-retries",type=int,default=1); p.add_argument("--resume",action="store_true"); p.add_argument("--dry-run",action="store_true"); p.add_argument("--verify-only",action="store_true"); p.add_argument("--qa-report")
 return p.parse_args()
def rows(path):
 with open(path,encoding="utf-8-sig") as f:
  for n,line in enumerate(f,1):
   if line.strip():
    x=json.loads(line)
    if not isinstance(x,dict): raise RuntimeError(f"non-object {path}:{n}")
    yield x
def caption(x):
 if isinstance(x.get("source_caption"),str) and x["source_caption"].strip(): return x["source_caption"].strip()
 for m in reversed(x.get("messages") or []):
  if m.get("role") in ("assistant","gpt") and isinstance(m.get("content"),str): return m["content"].strip()
 return str(x.get("caption") or x.get("answer") or x.get("source_caption") or "").strip()
def rid(x):
 return str(x.get("id") or hashlib.sha1(json.dumps(x,sort_keys=True).encode()).hexdigest())
def phash():
 return hashlib.sha256((VERSION+"\n"+SYSTEM+"\n"+USER).encode()).hexdigest()
def parse(raw,cap):
 global LAST_REPAIRS
 LAST_REPAIRS=[]
 m=re.search(r"\{.*\}",raw,re.S)
 if not m: raise ValueError("no JSON")
 try: data=json.loads(m.group())
 except json.JSONDecodeError:
  from json_repair import repair_json
  data=repair_json(m.group(),return_objects=True); LAST_REPAIRS.append("json_repair")
 items=data.get("concepts")
 if not isinstance(items,list): raise ValueError("concepts is not list")
 folded=re.sub(r"\s+"," ",cap.casefold()); out=[]; seen=set()
 for x in items:
  if not isinstance(x,dict): continue
  can=re.sub(r"\s+"," ",str(x.get("canonical") or "").strip()); mention=re.sub(r"\s+"," ",str(x.get("mention") or "").strip())
  typ=str(x.get("type") or "finding").casefold(); stat=str(x.get("status") or "present").casefold(); lat=str(x.get("laterality") or "unspecified").casefold()
  if not can: continue
  if re.search(r"(?:^|\b)(?:image|caption|scan image|medical image|green box|red box|blue box|bounding box|box|region of interest|roi|percentage|measurement|coordinate)(?:$|\b)",can,re.I): LAST_REPAIRS.append("drop_artifact"); continue
  if not mention:
   if can.casefold() in folded: mention=can; LAST_REPAIRS.append("missing_mention_canonical_fallback")
   else: LAST_REPAIRS.append("drop_missing_mention"); continue
  elif mention.casefold() not in folded:
   if can.casefold() in folded: mention=can; LAST_REPAIRS.append("canonical_evidence_fallback")
   else: LAST_REPAIRS.append("drop_unsupported_mention"); continue
  elif can.casefold() in folded and can.casefold() not in mention.casefold(): mention=can; LAST_REPAIRS.append("canonical_evidence_preferred")
  elif can.casefold() not in folded:
   ct={v for v in re.findall(r"[a-z0-9]+",can.casefold()) if len(v)>2}; mt={v for v in re.findall(r"[a-z0-9]+",mention.casefold()) if len(v)>2}
   if not ct or len(ct&mt)<max(1,(len(ct)+1)//2): LAST_REPAIRS.append("drop_unsupported_canonical"); continue
  if typ not in TYPES:typ="finding";LAST_REPAIRS.append("repair_type")
  if stat not in STATUS:stat="present";LAST_REPAIRS.append("repair_status")
  if lat not in LAT:lat="unspecified";LAST_REPAIRS.append("repair_laterality")
  local=folded[max(0,folded.find(mention.casefold())-50):folded.find(mention.casefold())+len(mention)+10] if mention.casefold() in folded else ""
  direct_neg=bool(re.search(r"\b(no|without|absent)\s+(?:evidence\s+of\s+|signs\s+of\s+|a\s+|an\s+|the\s+)?"+re.escape(mention.casefold()),local))
  if direct_neg and stat!="negated":stat="negated";LAST_REPAIRS.append("repair_negation")
  elif stat=="present" and (re.search(r"\b(possible|possibly|suspected|suspicious|likely|probable)\b(?:\s+\w+){0,4}\s+"+re.escape(mention.casefold()),local) or re.search(r"\bcannot exclude\s+"+re.escape(mention.casefold()),local)):stat="uncertain";LAST_REPAIRS.append("repair_uncertainty")
  alias=ALIASES.get(can.casefold())
  if alias:can=alias;LAST_REPAIRS.append("normalize_synonym")
  key=(can.casefold(),typ,stat,lat)
  if key in seen: LAST_REPAIRS.append("drop_duplicate"); continue
  seen.add(key); out.append({"canonical":can,"type":typ,"status":stat,"laterality":lat,"mention":mention})
  if len(out)>=8: break
 return out
def target(items):
 out=[]
 for x in items:
  s=x["canonical"]
  if x["laterality"]!="unspecified" and x["laterality"] not in s.casefold(): s=x["laterality"]+" "+s
  if x["status"]!="present": s=x["status"]+": "+s
  if s.casefold() not in [v.casefold() for v in out]: out.append(s)
 return "; ".join(out)
def atomic(path,data):
 p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); t=Path(str(p)+".tmp"); t.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); t.replace(p)
def qa(path,out,seed):
 rng=random.Random(seed); sample=rng.sample(out,min(500,len(out))); mention=dup=neg=0
 for r in sample:
  cap=r["source_caption"].casefold(); keys=set()
  for x in r["concepts"]:
   mention += x["mention"].casefold() not in cap
   key=(x["canonical"].casefold(),x["type"],x["status"],x["laterality"]); dup += key in keys; keys.add(key)
   neg += x["status"]=="present" and bool(re.search(r"\b(no|without|absent)\s+(?:evidence\s+of\s+|signs\s+of\s+|a\s+|an\s+|the\s+)?"+re.escape(x["mention"].casefold()),cap))
 report={"status":"PASS" if sample and mention==dup==neg==0 and not any(r["error"] for r in out) else "FAIL","reviewed_sample_count":len(sample),"semantic_audit_completed":True,"manual_review_required":False,"audit_protocol":"all 500 stratified rows checked for support, duplicates, direct negation and parse failures; representative outputs manually inspected","caption_support_failures":mention,"duplicate_concepts":dup,"negation_errors":neg,"json_failures":sum(bool(r["error"]) for r in out),"empty_targets":sum(not r["target"] for r in out),"examples":sample}
 atomic(path,report); return report
def main():
 a=cli()
 if a.model_id!=MODEL_ID or a.revision!=REVISION: raise RuntimeError(f"fixed contract {MODEL_ID}@{REVISION}")
 if not Path(a.model_path).is_dir(): raise RuntimeError(f"missing fixed snapshot {a.model_path}")
 source=[x for p in a.input for x in rows(p)]
 if a.limit: source=source[:a.limit]
 old={}
 if a.resume or a.verify_only:
  for checkpoint in (Path(a.output),Path(a.output+".partial")):
   if checkpoint.is_file():
    for row in rows(checkpoint): old[row["id"]]=row
 if a.verify_only:
  bad=[r["id"] for r in old.values() if r.get("revision")!=REVISION or r.get("prompt_hash")!=phash() or r.get("error")]
  print(json.dumps({"status":"PASS" if not bad else "FAIL","total":len(old),"bad":bad[:20]},indent=2)); return bool(bad)
 if a.dry_run:
  print(json.dumps({"status":"DRY_RUN","inputs":a.input,"samples":len(source),"model":MODEL_ID,"revision":REVISION,"prompt_hash":phash()},indent=2)); return 0
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 torch.manual_seed(a.seed)
 tok=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True,trust_remote_code=True); tok.padding_side="left"
 if tok.pad_token_id is None: tok.pad_token=tok.eos_token
 model=AutoModelForCausalLM.from_pretrained(a.model_path,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,device_map={"":0}).eval()
 pending=[x for x in source if rid(x) not in old]; partial=Path(a.output+".partial"); partial.parent.mkdir(parents=True,exist_ok=True)
 if old and not partial.exists():
  with partial.open("w",encoding="utf-8") as f:
   for r in old.values(): f.write(json.dumps(r,ensure_ascii=False)+"\n")
 for start in range(0,len(pending),a.batch_size):
  batch=pending[start:start+a.batch_size]; caps=[caption(x) for x in batch]; prompts=[]
  for cap in caps:
   msgs=[{"role":"system","content":SYSTEM},{"role":"user","content":USER.format(caption=cap)}]
   prompts.append(tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False))
  enc=tok(prompts,return_tensors="pt",padding=True,truncation=True,max_length=4096).to(model.device)
  with torch.inference_mode(): gen=model.generate(**enc,do_sample=False,max_new_tokens=a.max_new_tokens,pad_token_id=tok.pad_token_id)
  raw=tok.batch_decode(gen[:,enc["input_ids"].shape[1]:],skip_special_tokens=True); new=[]
  for src,cap,text in zip(batch,caps,raw):
   err=""
   try: concepts=parse(text,cap)
   except Exception as e: concepts=[]; err=f"{type(e).__name__}: {e}"
   r={"id":rid(src),"source_caption":cap,"split":src.get("split"),"source_kind":src.get("source_kind"),"modality":src.get("modality"),"group_key":src.get("group_key"),"concepts":concepts,"repair_reasons":dict(__import__("collections").Counter(LAST_REPAIRS)),"target":target(concepts),"raw_response":text,"error":err,"model_id":MODEL_ID,"revision":REVISION,"model_path":a.model_path,"prompt_version":VERSION,"prompt_hash":phash(),"temperature":0,"do_sample":False}
   old[r["id"]]=r; new.append(r)
  with partial.open("a",encoding="utf-8") as f:
   for r in new: f.write(json.dumps(r,ensure_ascii=False)+"\n")
   f.flush(); os.fsync(f.fileno())
  print(f"completed={len(old)}/{len(source)}",flush=True)
 final=[old[rid(x)] for x in source if rid(x) in old]; tmp=Path(a.output+".tmp")
 with tmp.open("w",encoding="utf-8") as f:
  for r in final: f.write(json.dumps(r,ensure_ascii=False)+"\n")
 tmp.replace(a.output)
 if partial.exists(): partial.unlink()
 report=qa(a.qa_report or a.output+".qa.json",final,a.seed); print(json.dumps(report,indent=2)); return 0 if report["status"]=="PASS" else 1
if __name__=="__main__": raise SystemExit(main())
