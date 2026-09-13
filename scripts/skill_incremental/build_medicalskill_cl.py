#!/usr/bin/env python3
"""Build six MedicalSkill-CL skills from locally validated sources."""
from __future__ import annotations
import argparse, collections, hashlib, json, os, random, re, shutil
from pathlib import Path
from PIL import Image

ROOT=Path("/root/MedAgentCL_v4"); OLD=Path("/root/MedAgentCL/data"); OUT=Path("/remote-home/wangbomin/MedicalSkill-CL")
ART=ROOT/"artifacts/medicalskill_cl_build"; MED=OLD/"MedTrinity_ConceptCaption_CL"; SG=Path("/remote-home/wangbomin/MedSG"); THINK=Path("/remote-home/wangbomin/MedThinkVQA")
TASKS={1:("task_01_vqa","vqa"),2:("task_02_diagnosis_classification","diagnosis_classification"),3:("task_03_concept_recognition","concept_recognition"),4:("task_04_caption_generation","caption_generation"),5:("task_05_visual_grounding","visual_grounding"),6:("task_06_reasoning_vqa","reasoning_vqa")}
BOX=re.compile(r"<\|box_start\|>\s*\(([-\d.]+),\s*([-\d.]+)\),\s*\(([-\d.]+),\s*([-\d.]+)\)\s*<\|box_end\|>")
ORD={"first":0,"1st":0,"second":1,"2nd":1,"third":2,"3rd":2,"fourth":3,"4th":3,"fifth":4,"5th":4,"sixth":5,"6th":5}

def cli():
 p=argparse.ArgumentParser(); p.add_argument("--output-root",default=str(OUT)); p.add_argument("--artifact-root",default=str(ART)); p.add_argument("--tasks",default="all")
 p.add_argument("--seed",type=int,default=42); p.add_argument("--max-images-per-case",type=int,default=8); p.add_argument("--concept-extractions",default=str(OUT/"_concept/full.jsonl"))
 p.add_argument("--concept-pilot-report",default=str(OUT/"_concept/pilot.qa.json")); p.add_argument("--dry-run",action="store_true"); p.add_argument("--resume",action="store_true"); p.add_argument("--verify-only",action="store_true")
 return p.parse_args()
def lines(p):
 with open(p,encoding="utf-8-sig") as f:
  for n,line in enumerate(f,1):
   if line.strip():
    x=json.loads(line)
    if not isinstance(x,dict): raise RuntimeError(f"non-object {p}:{n}")
    yield x
def atomic(p,x):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=Path(str(p)+".tmp"); t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); t.replace(p)
def writejl(p,rows):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=Path(str(p)+".tmp")
 with t.open("w",encoding="utf-8") as f:
  for x in rows: f.write(json.dumps(x,ensure_ascii=False)+"\n")
 t.replace(p)
def msgs(x):
 out=[]
 for m in x.get("messages") or []:
  role=m.get("role") or m.get("from"); role="user" if role in ("user","human") else "assistant" if role in ("assistant","gpt") else ""
  text=m.get("content") if "content" in m else m.get("value")
  if role and isinstance(text,str): out.append({"role":role,"content":text.strip()})
 return out
def images(x): return [str(v) for v in (x.get("images") or ([x.get("image")] if x.get("image") else [])) if v]
def base(x,dataset,split):
 m=x.get("metadata") or {}; sid=str(x.get("source_sample_id") or x.get("question_id") or x.get("id") or "")
 return {"source_dataset":dataset,"source_split":str(x.get("source_split") or split),"source_id":sid,"case_id":str(m.get("case_id") or m.get("study_id") or x.get("case_id") or x.get("study_id") or sid),"patient_id":str(m.get("patient_id") or x.get("patient_id") or "")}
def unified(i,t,skill,m,ims,meta): return {"id":i,"task_id":t,"skill":skill,"messages":m,"images":ims,"metadata":meta}
def unique(rows,key):
 seen=set(); out=[]; dup=0
 for x in rows:
  k=key(x)
  if k in seen: dup+=1
  else: seen.add(k); out.append(x)
 return out,dup
def task12(task,split):
 if task==1:
  src=OLD/"MedSkill_CL_4Skill"/f"task_02_vqa_{split}.jsonl"; raw=list(lines(src)); rows,dup=unique(raw,lambda x:(x.get("question_id") or x.get("id"),tuple(images(x))))
  out=[unified(str(x.get("id") or x.get("question_id")),1,"vqa",msgs(x),images(x),{**base(x,str(x.get("dataset") or "OmniMedVQA"),split),"modality":x.get("modality"),"question_type":x.get("question_type")}) for x in rows]
  return out,{"source":str(src),"input":len(raw),"output":len(out),"duplicates_removed":dup,"replacement_sampling_used":False}
 srcs=sorted((OLD/"MedIMeta_CL_Diag8").glob(f"task_*/{split}.jsonl")); raw=[x for p in srcs for x in lines(p)]
 rows,dup=unique(raw,lambda x:(x.get("dataset_id"),x.get("source_sample_id"),tuple(images(x)),x.get("answer"))); out=[]; invalid=0; dist=collections.Counter()
 for x in rows:
  opts=[str(v) for v in x.get("options") or []]; ans=str(x.get("answer") or "")
  if not opts or ans not in opts: invalid+=1; continue
  dist[len(opts)]+=1; meta={**base(x,"MedIMeta",split),"dataset_id":x.get("dataset_id"),"domain":x.get("domain"),"modality":x.get("modality"),"options":opts,"answer_option":x.get("answer_option"),"canonical_label":ans}
  out.append(unified(str(x.get("id")),2,"diagnosis_classification",msgs(x),images(x),meta))
 return out,{"sources":[str(p) for p in srcs],"input":len(raw),"output":len(out),"duplicates_removed":dup,"invalid_removed":invalid,"option_count_distribution":dict(dist),"replacement_sampling_used":False}
def clean_caption(s):
 original=s.strip(); rules=[]; x=re.sub(r"<[^>]+>"," ",original)
 if x!=original: rules.append("strip_html")
 y=re.sub(r"(?:[A-Za-z]:\\|/)(?:[^\s,;]+/)+[^\s,;]+"," ",x)
 if y!=x: rules.append("strip_export_path")
 y=re.sub(r"\s+"," ",y).strip(); ss=re.split(r"(?<=[.!?])\s+",y); z=list(dict.fromkeys(v for v in ss if v))
 if len(z)!=len(ss): rules.append("dedupe_sentence")
 return " ".join(z),rules
def medtrinity(task,split,extract_path):
 kind="concept" if task==3 else "caption"; src=MED/f"{kind}_{split}.jsonl"; raw=list(lines(src)); extracts={x["id"]:x for x in lines(extract_path)} if task==3 else {}
 out=[]; missing=empty=boxlang=0; rules=collections.Counter()
 for x in raw:
  i=str(x.get("id")); meta={**base(x,"MedTrinity-25M",split),"modality":x.get("modality") or x.get("source_modality"),"group_key":x.get("split_group_key") or (x.get("metadata") or {}).get("group_key"),"image_sha256s":x.get("image_sha256s") or []}
  if task==3:
   e=extracts.get(i)
   if not e: missing+=1; continue
   target=str(e.get("target") or "")
   if not target: empty+=1; continue
   meta.update({"concepts":e.get("concepts") or [],"model_id":e.get("model_id"),"revision":e.get("revision"),"prompt_hash":e.get("prompt_hash"),"all_target_concepts":[v.get("canonical") for v in e.get("concepts") or []],"core_target_concepts":[]})
   m=[{"role":"user","content":"<image>\nIdentify the caption-supported medical concepts. Return a concise semicolon-separated list."},{"role":"assistant","content":target}]; skill="concept_recognition"
  else:
   mm=msgs(x); original=mm[-1]["content"] if mm else ""; target,rr=clean_caption(original); rules.update(rr); empty+=not bool(target); boxlang+=bool(re.search(r"\bbox|bounding box|highlighted|region of interest|\bROI\b",original,re.I))
   if not target: continue
   meta.update({"original_caption":original,"cleaned_caption":target,"cleaning_rules":json.dumps(rr,ensure_ascii=False),"possible_pre_rendered_box":bool((x.get("metadata") or {}).get("boxed_image"))})
   m=[{"role":"user","content":"<image>\nDescribe the clinically relevant content of this medical image."},{"role":"assistant","content":target}]; skill="caption_generation"
  out.append(unified(i,task,skill,m,images(x),meta))
 return out,{"source":str(src),"input":len(raw),"output":len(out),"missing_extractions":missing,"empty_removed":empty,"cleaning_rule_counts":dict(rules),"possible_box_language":boxlang,"replacement_sampling_used":False}
def sgpath(v):
 s=str(v).replace("\\","/"); marker="MedSG-Bench/"
 if marker in s: s=s[s.index(marker):]
 return str(SG/s)
def ordinal(text,count,default):
 found=list(re.finditer(r"\b(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th) image\b",text,re.I))
 return min(ORD[found[-1].group(1).casefold()],count-1) if found else min(default,count-1)
def targetidx(task,prompt,answer,n):
 default={1:0,2:1,3:2,4:1,5:1,6:0,7:1,8:0}.get(task,0)
 return ordinal(answer+" "+prompt,n,default)
def normalize_prompt_boxes(prompt,ims):
 def repl(match):
  idx=ordinal(prompt[:match.start()],len(ims),0)
  try:
   with Image.open(ims[idx]) as im: width,height=im.size
   box=boxnorm(match.groups(),width,height)
   return "<box>"+",".join(map(str,box))+"</box>" if box else ""
  except Exception:return ""
 return BOX.sub(repl,prompt)
def redratio(path):
 import numpy as np
 with Image.open(path) as im: a=np.asarray(im.convert("RGB"))
 red=(a[:,:,0]>240)&(a[:,:,1]<40)&(a[:,:,2]<40)
 return float(red.mean())
def boxnorm(values,w,h):
 x1,y1,x2,y2=map(float,values); b=[round(x1*1000/w),round(y1*1000/h),round(x2*1000/w),round(y2*1000/h)]; b=[min(1000,max(0,v)) for v in b]
 return b if b[0]<b[2] and b[1]<b[3] else None
def dedupe_boxes(boxes):
 seen=set(); out=[]
 for box in boxes:
  key=tuple(box)
  if key not in seen: seen.add(key); out.append(box)
 return out
def grounding(split,seed):
 section="MedSG-Train" if split=="train" else "MedSG-Bench"; out=[]; audits=[]; filt=collections.Counter(); per={}; rng=random.Random(seed+(split=="test"))
 for task in range(1,9):
  source=json.loads((SG/section/f"Task{task}.json").read_text()); sample={id(x) for x in rng.sample(source,min(100,len(source)))}; count=0
  for rown,x in enumerate(source):
   ims=[sgpath(v) for v in x.get("images") or []]
   pairs=[]
   if "conversations" in x:
    prompt=""
    for m in x["conversations"]:
     role=m.get("from"); text=str(m.get("value") or "")
     if role=="human": prompt=text
     elif role=="gpt" and prompt: pairs.append((prompt,text)); prompt=""
   else:
    answer=x.get("answer"); text="<|box_start|>({0},{1}),({2},{3})<|box_end|>".format(*answer) if isinstance(answer,list) and len(answer)==4 else str(answer or "")
    pairs=[(str(x.get("question") or ""),str(x.get("additional_info") or "")+" "+text)]
   for turn,(prompt,answer) in enumerate(pairs):
    idx=targetidx(task,prompt,answer,len(ims))
    try:
     with Image.open(ims[idx]) as im: w,h=im.size
    except Exception: filt["missing_unreadable"]+=1; continue
    found=BOX.findall(answer); boxes=[boxnorm(v,w,h) for v in found]; boxes=dedupe_boxes([v for v in boxes if v])
    if not boxes: filt["invalid_box"]+=1; continue
    target_red=redratio(ims[idx]); ratios=[redratio(p) for p in ims] if id(x) in sample else []
    if ratios: audits.append({"split":split,"medsg_task":task,"source_id":x.get("id") or f"{task}_{rown}","image_red_ratios":ratios,"target_image_index":idx,"target_has_rendered_red":target_red>0.00005,"reference_has_rendered_red":any(v>0.00005 for j,v in enumerate(ratios) if j!=idx)})
    # Rendered red boxes are expected cues in reference images. A target red box is leakage.
    if target_red>0.00005: filt["target_gt_box_leakage_filtered"]+=1; continue
    target="; ".join("<box>"+",".join(map(str,b))+"</box>" for b in boxes); clean=normalize_prompt_boxes(prompt,ims)
    meta={"source_dataset":"MedSG","source_split":split,"source_id":str(x.get("id") or f"{task}_{rown}"),"case_id":str(x.get("id") or f"{task}_{rown}"),"patient_id":"","medsg_task":task,"medsg_task_name":x.get("task"),"target_image_index":idx,"original_target":answer,"boxes_0_1000":boxes,"target_image_size":[w,h],"coordinate_source":"official pixel coordinates normalized with target PIL dimensions","image_order_preserved":True,"reference_prompt_boxes_normalized_0_1000":True}
    out.append(unified(f"medsg_{split}_t{task}_{rown}_{turn}",5,"visual_grounding",[{"role":"user","content":clean},{"role":"assistant","content":target}],ims,meta)); count+=1
  per[str(task)]={"source_rows":len(source),"output_turns":count,"visual_audit_rows":min(100,len(source))}
 return out,{"section":section,"output":len(out),"task_statistics":per,"filtered":dict(filt),"role_audit":audits,"replacement_sampling_used":False}
def reasoning_text(s):
 s=re.sub(r"\[[0-9,\s-]+\]","",s).replace(";"," "); return re.sub(r"\s+"," ",s).strip()
def reasoning(split,max_images):
 src=THINK/f"{split}.jsonl"; out=[]; od=collections.Counter(); sd=collections.Counter(); reasoning_lengths=[]; longitudinal=0; forbidden=collections.Counter()
 for n,x in enumerate(lines(src)):
  paths=[]; ids=[]
  for j in range(1,100):
   v=str(x.get(f"image_{j:02d}_path") or "").strip()
   if v: paths.append(str(THINK/v)); ids.append(str(x.get(f"image_{j:02d}_id") or Path(v).stem))
  ims=paths[:max_images]; selected=ids[:max_images]; opts=x.get("options") or {}; options="\n".join(f"{k}. {v}" for k,v in opts.items()); history=str(x.get("CLINICAL_HISTORY") or "").strip()
  temporal=""
  if x.get("is_longitudinal"): longitudinal+=1; temporal=f"\nLongitudinal context: {x.get('interval_text') or 'Multiple timepoints are shown in native order.'}"
  prompt="\n".join("<image>" for _ in ims)+f"\nClinical history: {history}{temporal}\nQuestion: Choose the best diagnosis.\n{options}\nReturn the option letter and canonical text."
  letter=str(x.get("correct_answer") or "").strip().upper(); answer=str(x.get("correct_answer_text") or opts.get(letter) or "").strip(); source_reasoning=str(x.get("discussion") or ""); cleaned_reasoning=reasoning_text(source_reasoning); reasoning_lengths.append(len(cleaned_reasoning)); target=f"Reasoning:\n{cleaned_reasoning}\n\nFinal answer:\n{letter}. {answer}"
  forbidden_labels={"IMAGING_FINDINGS":r"(?im)^\s*imaging[_ ]findings\s*:","discussion":r"(?im)^\s*discussion\s*:","image_caption":r"(?im)^\s*image[_ ]caption\s*:","correct_answer":r"(?im)^\s*(?:correct|final)[_ ]answer\s*:"}
  for field,pattern in forbidden_labels.items(): forbidden[field]+=bool(re.search(pattern,prompt))
  case=re.sub(r"\D+","",str(x.get("title") or "")) or str(n); meta={"source_dataset":"MedThinkVQA","source_split":split,"source_id":str(x.get("title") or n),"case_id":"case"+case,"patient_id":"","is_longitudinal":bool(x.get("is_longitudinal")),"timepoint_count":x.get("timepoint_count"),"original_image_count":len(paths),"selected_image_count":len(ims),"dropped_image_count":len(paths)-len(ims),"selected_image_ids":selected,"selection_strategy":"first_k_native_order","max_images_per_case":max_images,"correct_answer":letter,"correct_answer_text":answer,"options":opts,"source_reasoning_char_count":len(source_reasoning),"cleaned_reasoning_char_count":len(cleaned_reasoning),"reasoning_truncated":False}
  out.append(unified(f"medthink_{split}_{case}",6,"reasoning_vqa",[{"role":"user","content":prompt},{"role":"assistant","content":target}],ims,meta)); od[len(paths)]+=1; sd[len(ims)]+=1
 return out,{"source":str(src),"output":len(out),"original_image_count_distribution":dict(od),"selected_image_count_distribution":dict(sd),"longitudinal_cases":longitudinal,"reasoning_cleaned_char_min":min(reasoning_lengths) if reasoning_lengths else 0,"reasoning_cleaned_char_median":sorted(reasoning_lengths)[len(reasoning_lengths)//2] if reasoning_lengths else 0,"reasoning_cleaned_char_max":max(reasoning_lengths) if reasoning_lengths else 0,"reasoning_truncated_count":0,"reasoning_empty_count":sum(v==0 for v in reasoning_lengths),"prompt_source_whitelist":["CLINICAL_HISTORY","interval_text","options","image paths"],"forbidden_source_fields_excluded":["IMAGING_FINDINGS","discussion","image_*_caption","correct_answer","correct_answer_text"],"forbidden_prompt_label_occurrences":dict(forbidden),"max_images_per_case":max_images,"selection_strategy":"first_k_native_order","untruncated_manifest":str(src),"replacement_sampling_used":False}
def caption_rule_names(value):
 if isinstance(value,str):
  try: value=json.loads(value)
  except json.JSONDecodeError: value=[] if not value else [value]
 return [str(v) for v in value] if isinstance(value,list) else []
def caption_audit(td):
 report={"policy":"Preserve the original caption; remove only HTML/export paths, collapse whitespace, and exact repeated sentences.","splits":{}}
 for split in ("train","test"):
  data=list(lines(td/f"{split}.jsonl")); changed=[]; rules=collections.Counter(); lengths=[]
  for x in data:
   meta=x.get("metadata") or {}; original=str(meta.get("original_caption") or ""); cleaned=str(meta.get("cleaned_caption") or "")
   applied=caption_rule_names(meta.get("cleaning_rules")); lengths.append(len(cleaned)); rules.update(applied)
   if original!=cleaned and len(changed)<50: changed.append({"id":x.get("id"),"original":original,"cleaned":cleaned,"rules":applied})
  modified=sum(str((x.get("metadata") or {}).get("original_caption") or "")!=str((x.get("metadata") or {}).get("cleaned_caption") or "") for x in data)
  ordered=sorted(lengths); report["splits"][split]={"records":len(data),"modified":modified,"unchanged":len(data)-modified,"modified_ratio":modified/len(data) if data else 0,"rule_counts":dict(rules),"cleaned_length_min":ordered[0] if ordered else 0,"cleaned_length_median":ordered[len(ordered)//2] if ordered else 0,"cleaned_length_max":ordered[-1] if ordered else 0,"changed_examples":changed}
 return report
def concept_vocab(td):
 train=list(lines(td/"train.jsonl")); folded=collections.Counter(); variants=collections.defaultdict(collections.Counter)
 for x in train:
  for concept in x["metadata"].get("concepts",[]):
   name=str(concept.get("canonical") or "").strip()
   if name: folded[name.casefold()]+=1; variants[name.casefold()][name]+=1
 display={key:sorted(counts.items(),key=lambda item:(-item[1],item[0]))[0][0] for key,counts in variants.items()}; core={key for key,count in folded.items() if count>=10}
 for split in ("train","test"):
  data=list(lines(td/f"{split}.jsonl"))
  for x in data:
   selected=[]; seen=set()
   for value in x["metadata"].get("all_target_concepts",[]):
    key=str(value or "").strip().casefold()
    if key in core and key not in seen: seen.add(key); selected.append(display[key])
   x["metadata"]["core_target_concepts"]=selected
  writejl(td/f"{split}.jsonl",data)
 frequencies={display[key]:folded[key] for key in sorted(folded,key=lambda v:(-folded[v],display[v]))}
 atomic(td/"concept_frequency_train.json",frequencies); atomic(td/"concept_vocabulary.json",{"all_train":[display[key] for key in sorted(display)],"core_frequency_ge_10":[display[key] for key in sorted(core)],"core_size":len(core),"normalization":"casefold; display form is most frequent train spelling","built_from_test":False}); atomic(td/"concept_synonyms.json",{display[key]:sorted(counts) for key,counts in variants.items()})
 return {"all_vocabulary_size":len(folded),"core_vocabulary_size":len(core),"minimum_core_frequency":10,"normalization":"train-only casefold aggregation"}
def validate(data):
 e=collections.Counter(); ims=set(); cases=set()
 for x in data:
  e["top_level_schema"]+=set(x)!={"id","task_id","skill","messages","images","metadata"}
  e["messages"]+=len(x["messages"])!=2 or [m.get("role") for m in x["messages"]]!=["user","assistant"]
  e["images"]+=not x["images"] or not all(Path(v).is_absolute() for v in x["images"])
  ims.update(x["images"]); cases.add(str(x["metadata"].get("case_id") or x["id"]))
 return {"status":"PASS" if sum(e.values())==0 else "FAIL","records":len(data),"unique_image_paths":len(ims),"unique_cases":len(cases),"errors":{k:v for k,v in e.items() if v}}
def contracts():
 common={"score_range":[0,100],"empty_target_behavior":"dataset validation error; excluded neither silently nor as a valid answer"}
 return {
  "task_01_vqa":{**common,"target_parser":"option letter when present, otherwise normalized answer text","prediction_parser":"option letter when present, otherwise normalized answer text","invalid_prediction_rule":"count as incorrect","primary_metric":"accuracy * 100","secondary_metrics":[]},
  "task_02_diagnosis_classification":{**common,"target_parser":"available option letter or normalized canonical diagnosis label","prediction_parser":"option letter when present, otherwise normalized diagnosis text","invalid_prediction_rule":"count as incorrect; nonexistent option cannot match","primary_metric":"accuracy * 100","secondary_metrics":[]},
  "task_03_concept_recognition":{**common,"target_parser":"semicolon-separated concepts; strip status prefix and normalize case/punctuation","prediction_parser":"same parser as target","invalid_prediction_rule":"empty/invalid output is an empty concept set","primary_metric":"all-concept micro-F1 * 100","secondary_metrics":["train-only core-vocabulary micro-F1 * 100"]},
  "task_04_caption_generation":{**common,"target_parser":"cleaned caption text","prediction_parser":"normalized text tokens","invalid_prediction_rule":"empty output receives zero","primary_metric":"ROUGE-L F1 * 100","secondary_metrics":[]},
  "task_05_visual_grounding":{**common,"target_parser":"all valid <box>x1,y1,x2,y2</box> coordinates in [0,1000]","prediction_parser":"same box parser; invalid or inverted boxes discarded","invalid_prediction_rule":"no valid predicted box receives zero","primary_metric":"Acc@matched-IoU>=0.5 * 100","secondary_metrics":["mean matched IoU * 100"],"multi_box_matching":"global greedy maximum-IoU one-to-one matching; unmatched target boxes score zero"},
  "task_06_reasoning_vqa":{**common,"target_parser":"case-insensitive Final answer section, then option letter/canonical text","prediction_parser":"same final-answer parser; preceding text is reasoning","invalid_prediction_rule":"unparseable final answer counts as incorrect","primary_metric":"final-answer accuracy * 100","secondary_metrics":["reasoning ROUGE-L F1 * 100"]}
 }
def inventory():
 paths=[OLD/"MedSkill_CL_4Skill",OLD/"MedIMeta_CL_Diag8",MED,SG,THINK]
 result={"sources":{str(p):p.exists() for p in paths},"root_space":dict(zip(("total","used","free"),shutil.disk_usage("/root"))),"remote_space":dict(zip(("total","used","free"),shutil.disk_usage("/remote-home/wangbomin"))),"no_images_copied":True,"absolute_paths_only":True}
 report=ROOT/"artifacts/dataset_download/medsg_verification.json"
 if report.is_file():
  source=json.loads(report.read_text()); verification=source.get("verification") or {}
  result["medsg_closure"]={"repo_id":source.get("repo_id"),"revision":source.get("revision"),"local_dir":source.get("local_dir"),"local_file_status":(source.get("local_file_audit") or {}).get("status"),"zip_full_test_count":len(source.get("zip_audits") or []),"zip_full_test_passed":sum(v.get("status")=="PASS" and v.get("full_test_performed") for v in source.get("zip_audits") or []),"extraction_requested_this_run":True,"extracted_payload_present":bool(source.get("extracted")),"extraction_complete":bool(source.get("extracted")) and all(v.get("status")=="PASS" for v in source.get("extraction_results") or []),"semantic_status":verification.get("status"),"missing_images":verification.get("missing_images"),"corrupt_images":verification.get("corrupt_images"),"coordinate_out_of_bounds":verification.get("coordinate_out_of_bounds")}
 return result
def main():
 a=cli(); out=Path(a.output_root); art=Path(a.artifact_root); selected=list(TASKS) if a.tasks=="all" else sorted({int(v) for v in a.tasks.split(",")}); inv=inventory()
 if not all(inv["sources"].values()): raise RuntimeError(inv)
 if a.dry_run: print(json.dumps({"status":"DRY_RUN","tasks":selected,"inventory":inv},indent=2)); return 0
 out.mkdir(parents=True,exist_ok=True); art.mkdir(parents=True,exist_ok=True); atomic(art/"source_inventory.json",inv); atomic(art/"build_config.json",vars(a)); atomic(art/"metric_contracts.json",contracts())
 pilot=json.loads(Path(a.concept_pilot_report).read_text()) if Path(a.concept_pilot_report).is_file() else {"status":"MISSING"}
 if 3 in selected and pilot.get("status")!="PASS": raise RuntimeError(f"concept pilot gate {pilot.get('status')}")
 builders={1:lambda s:task12(1,s),2:lambda s:task12(2,s),3:lambda s:medtrinity(3,s,Path(a.concept_extractions)),4:lambda s:medtrinity(4,s,Path(a.concept_extractions)),5:lambda s:grounding(s,a.seed),6:lambda s:reasoning(s,a.max_images_per_case)}
 allstats={}; manifests={}; role=[]
 for task in selected:
  d,skill=TASKS[task]; td=out/d; man={"task_id":task,"skill":skill,"splits":{}}
  for split in ("train","test"):
   if a.verify_only:
    data=list(lines(td/f"{split}.jsonl")); previous=td/f"{split}_manifest.json"; stats=dict((json.loads(previous.read_text()).get("statistics") or {}) if previous.is_file() else {})
    prior_output=stats.get("output")
    if isinstance(prior_output,int) and prior_output!=len(data):
     stats.setdefault("output_before_global_exact_leakage_repair",prior_output); stats["global_exact_leakage_records_removed"]=prior_output-len(data)
    stats["output"]=len(data)
   else: data,stats=builders[task](split); writejl(td/f"{split}.jsonl",data)
   check=validate(data)
   if check["status"]!="PASS": raise RuntimeError(f"validation {task} {split}: {check}")
   atomic(td/f"{split}_manifest.json",{"task_id":task,"skill":skill,"split":split,"statistics":stats,"validation":check}); man["splits"][split]={"path":str(td/f"{split}.jsonl"),"count":len(data),"validation":check}; allstats[f"task_{task:02d}_{split}"]={**stats,**check}
   if task==5: role.extend(stats.get("role_audit") or [])
  if task==3: man["concept_vocabulary"]=concept_vocab(td)
  if task==4:
   ca=caption_audit(td); atomic(out/"audits/medtrinity_caption_cleaning_audit.json",ca); atomic(art/"medtrinity_caption_cleaning_audit.json",ca)
  atomic(td/"statistics.json",man); atomic(td/"validation_report.json",{"status":"PASS","splits":man["splits"]}); manifests[d]=man
 splitman=json.loads((MED/"split_manifest.json").read_text()); overlap=json.loads((MED/"leakage_audit.json").read_text())
 atomic(out/"audits/medtrinity_group_split_audit.json",splitman); atomic(out/"audits/medtrinity_concept_caption_overlap.json",overlap); atomic(art/"medtrinity_group_split_audit.json",splitman)
 if role:
  leaks=[x for x in role if x["target_has_rendered_red"]]; filtered_total=sum((v.get("filtered") or {}).get("target_gt_box_leakage_filtered",0) for k,v in allstats.items() if k.startswith("task_05_"))
  download_report=ROOT/"artifacts/dataset_download/medsg_verification.json"; visual={}
  if download_report.is_file(): visual=((json.loads(download_report.read_text()).get("verification") or {}).get("bounding_box_rendering") or {})
  rr={"status":"FILTERED" if filtered_total else "PASS","audit_method":"automated strict rendered-red pixel role audit","automated_rows_per_task_per_split":100,"automated_audit_rows":len(role),"source_manual_visual_check":visual,"reference_red_box_allowed":True,"sampled_target_gt_box_leakage_count":len(leaks),"full_target_gt_box_rows_filtered":filtered_total,"test_records_preserved_except_explicit_target_leakage":True,"rows":role}
  atomic(out/"audits/medsg_rendered_box_role_audit.json",rr); atomic(out/"audits/medsg_leakage_filter_report.json",rr); atomic(art/"medsg_rendered_box_role_audit.json",rr)
 atomic(art/"task_statistics.json",allstats); manifest={"status":"PASS","seed":a.seed,"task_order":[TASKS[t][0] for t in TASKS],"tasks":manifests,"schema":["id","task_id","skill","messages","images","metadata"],"ms_swift_4_4_1_compatible":True,"splits":["train","test"],"validation_split_used":False,"replacement_sampling_used":False,"images_copied":False,"output_root":str(out)}
 atomic(out/"manifests/medicalskill_cl_manifest.json",manifest); atomic(art/"medicalskill_cl_manifest.json",manifest); print(json.dumps({"status":"PASS","tasks":{k:{s:v["count"] for s,v in m["splits"].items()} for k,m in manifests.items()}},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
