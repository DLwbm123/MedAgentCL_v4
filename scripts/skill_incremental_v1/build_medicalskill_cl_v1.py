#!/usr/bin/env python3
"""Build the five-skill MedicalSkill-CL-v1 dataset without modifying v0."""
from __future__ import annotations
import argparse,collections,csv,hashlib,json,random,re
from pathlib import Path

V0=Path("/remote-home/wangbomin/MedicalSkill-CL")
V1=Path("/remote-home/wangbomin/MedicalSkill-CL-v1")
ART=Path("/root/MedAgentCL_v4/artifacts/medicalskill_cl_v1_closure")
TASKS={1:("task_01_vqa","vqa"),2:("task_02_diagnosis_classification","diagnosis_classification"),3:("task_03_concept_recognition","concept_recognition"),4:("task_04_visual_grounding","visual_grounding"),5:("task_05_reasoning_vqa","reasoning_vqa")}
PRIORITY={5:0,3:1,2:2,1:3,4:4}
MODEL_ID="Qwen/Qwen3-8B";REVISION="b968826d9c46dd6066d109eabc6255188de91218"

def rows(path):
 with Path(path).open(encoding="utf-8-sig") as f:
  for n,line in enumerate(f,1):
   if line.strip():
    x=json.loads(line)
    if not isinstance(x,dict):raise RuntimeError(f"non-object {path}:{n}")
    yield x
def writej(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=Path(str(path)+".tmp");tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");tmp.replace(path)
def writejl(path,values):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=Path(str(path)+".tmp")
 with tmp.open("w",encoding="utf-8") as f:
  for value in values:f.write(json.dumps(value,ensure_ascii=False)+"\n")
 tmp.replace(path)
def normalize(value):
 return re.sub(r"[^a-z0-9]+","_",str(value).casefold()).strip("_")
def family(row):
 m=row.get("metadata") or {};ds=str(m.get("source_dataset") or "unknown");did=str(m.get("dataset_id") or "")
 raw=(ds+" "+did+" "+" ".join(row.get("images") or [])).casefold()
 rules=[("isic","isic"),("brats","brats"),("tcga","tcga"),("ihc4bc","ihc4bc"),("vindr","vindr"),("crc","crc"),("retinal oct","oct"),("oct ","oct"),("pneumonia","chest_pneumonia"),("coronahack","chest_pneumonia"),("medtrinity","medtrinity"),("medsg","medsg"),("medthink","medthink")]
 for needle,name in rules:
  if needle in raw:return name
 return normalize(did or ds)
def lineage_tokens(row,hashes):
 m=row.get("metadata") or {};fam=family(row);tokens=set()
 source_id=str(m.get("source_id") or "").strip();case_id=str(m.get("case_id") or "").strip();patient_id=str(m.get("patient_id") or "").strip();group=str(m.get("group_key") or "").strip()
 if group:tokens.add("group:"+fam+":"+normalize(group))
 if source_id and (fam in {"medsg","medthink"}):tokens.add("source:"+fam+":"+normalize(source_id))
 if case_id and case_id!=source_id:tokens.add("case:"+fam+":"+normalize(case_id))
 if patient_id:tokens.add("patient:"+fam+":"+normalize(patient_id))
 for image in row.get("images") or []:
  path=str(image).replace("\\","/");stem=normalize(Path(path).stem)
  if stem and len(stem)>=5:tokens.add("image:"+fam+":"+stem)
  low=path.casefold()
  for pattern,prefix in [
   (r"(isic[_-]\d+)","isic"),(r"(tcga-[a-z0-9]{2}-[a-z0-9]{4})","tcga"),
   (r"(brats(?:20)?\d*[_-]?[a-z0-9_-]{3,})","brats"),(r"(ihc4bc[_-]?[a-z0-9_-]+)","ihc4bc"),
   (r"(?:vindr|vindrcxr)[^/]*[/_-]([0-9a-f-]{16,})","vindr"),(r"(patient\d+)","patient"),(r"(case\d+)","case")]:
   match=re.search(pattern,low)
   if match:tokens.add(prefix+":"+normalize(match.group(1)))
  h=hashes.get(path)
  if h and h.get("sha256"):tokens.add("sha256:"+h["sha256"])
 if not tokens and source_id:tokens.add("source:"+fam+":"+normalize(source_id))
 return sorted(tokens)
def load_hashes(path,needed):
 found={}
 for x in rows(path):
  p=str(x.get("path") or "")
  if p in needed:found[p]={"sha256":x.get("sha256"),"phash":x.get("phash"),"width":x.get("width"),"height":x.get("height")}
 return found
def convert(row,task,skill,extracts=None):
 x={key:row[key] for key in ("id","messages","images","metadata")};x["task_id"]=task;x["skill"]=skill
 x={"id":str(x["id"]),"task_id":task,"skill":skill,"messages":x["messages"],"images":[str(v) for v in x["images"]],"metadata":dict(x["metadata"])}
 if task==3:
  e=(extracts or {}).get(x["id"])
  if not e or e.get("error") or not e.get("target"):raise RuntimeError(f"invalid concept extraction {x['id']}")
  x["messages"]=[{"role":"user","content":"<image>\nIdentify the caption-supported medical concepts. Return a concise semicolon-separated list."},{"role":"assistant","content":e["target"]}]
  x["metadata"].pop("original_caption",None);x["metadata"].pop("cleaned_caption",None);x["metadata"].pop("cleaning_rules",None);x["metadata"].pop("possible_pre_rendered_box",None)
  x["metadata"].update({"concepts":e["concepts"],"all_target_concepts":[v["canonical"] for v in e["concepts"]],"core_target_concepts":[normalize(v["canonical"]).replace("_"," ") for v in e["concepts"]],"model_id":e["model_id"],"revision":e["revision"],"prompt_hash":e["prompt_hash"],"prompt_version":e.get("prompt_version"),"max_concepts":8,"source_kind":e.get("source_kind")})
 return x
def balance_medsg(values,seed,target=6250):
 by_task=collections.defaultdict(lambda:collections.defaultdict(list))
 for row in values:by_task[int((row.get("metadata") or {})["medsg_task"])][str((row.get("metadata") or {})["source_id"])].append(row)
 selected=[];report={"seed":seed,"target_output_turns_per_official_task":target,"tasks":{}}
 for task in range(1,9):
  groups=list(by_task[task].items());random.Random(seed+task).shuffle(groups);kept=[];turns=0
  for group_id,group_rows in groups:
   before=abs(target-turns);after=abs(target-(turns+len(group_rows)))
   if turns<target or after<before:kept.append((group_id,group_rows));turns+=len(group_rows)
  selected.extend(row for _,group in kept for row in group)
  report["tasks"][str(task)]={"source_groups_available":len(groups),"source_groups_selected":len(kept),"output_turns_available":sum(len(g) for _,g in groups),"output_turns_selected":turns,"deviation_percentage_points":abs(turns/target-1)*100}
 report["total_output_turns"]=len(selected);report["within_required_range"]=0.96*target*8<=len(selected)<=1.04*target*8;report["max_percentage_point_deviation"]=max(v["deviation_percentage_points"] for v in report["tasks"].values());report["status"]="PASS" if report["within_required_range"] and report["max_percentage_point_deviation"]<=1 else "FAIL";report["sampling_unit"]="official source row; all rendered conversation turns retained";report["replacement_sampling_used"]=False
 if report["status"]!="PASS":raise RuntimeError(f"MedSG balance failed {report}")
 return selected,report
class DSU:
 def __init__(self,n):self.p=list(range(n))
 def find(self,x):
  while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
  return x
 def union(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.p[b]=a
def apply_lineage(data,hashes):
 test_tokens=collections.defaultdict(list);train=[]
 for task,splits in data.items():
  for split,values in splits.items():
   for row in values:
    tokens=lineage_tokens(row,hashes);row["metadata"]["canonical_lineage_tokens"]=tokens;row["metadata"]["lineage_group_id"]=hashlib.sha256("\n".join(tokens).encode()).hexdigest()[:20]
    if split=="test":
     for token in tokens:test_tokens[token].append((task,row["id"]))
    else:train.append((task,row))
 removed=collections.defaultdict(list);kept=[]
 for task,row in train:
  hits=sorted({ref for token in row["metadata"]["canonical_lineage_tokens"] for ref in test_tokens.get(token,[])})
  if hits:removed[task].append({"id":row["id"],"matched_test_refs":hits[:20],"matched_tokens":[t for t in row["metadata"]["canonical_lineage_tokens"] if t in test_tokens]})
  else:kept.append((task,row))
 for task in data:data[task]["train"]=[row for t,row in kept if t==task]
 test_report={"status":"PASS","policy":"preserve every test record; remove complete train record/case when any high-confidence canonical lineage token matches test","test_records_removed":0,"total_train_removed":sum(map(len,removed.values())),"per_task":{TASKS[t][0]:{"removed":len(removed[t]),"examples":removed[t][:100]} for t in TASKS}}
 counts={task:len(data[task]["train"]) for task in TASKS};flat=[(task,row) for task in TASKS for row in data[task]["train"]];dsu=DSU(len(flat));first={}
 for index,(task,row) in enumerate(flat):
  for token in row["metadata"]["canonical_lineage_tokens"]:
   if token in first:dsu.union(index,first[token])
   else:first[token]=index
 components=collections.defaultdict(list)
 for index,item in enumerate(flat):components[dsu.find(index)].append((index,*item))
 cross=[component for component in components.values() if len({x[1] for x in component})>1]
 remove_ids=collections.defaultdict(set);decisions=[]
 for component in cross:
  present=sorted({x[1] for x in component});owner=min(present,key=lambda task:(counts[task],PRIORITY[task],task))
  ids={task:[x[2]["id"] for x in component if x[1]==task] for task in present}
  for task in present:
   if task!=owner:remove_ids[task].update(ids[task])
  decisions.append({"component_id":hashlib.sha256("|".join(sorted(x[2]["id"] for x in component)).encode()).hexdigest()[:20],"present_tasks":present,"owner_task":owner,"task_train_sizes":{str(t):counts[t] for t in present},"record_ids":ids})
 for task in TASKS:data[task]["train"]=[row for row in data[task]["train"] if row["id"] not in remove_ids[task]]
 owner_report={"status":"PASS","policy":"owner is task with smaller pre-ownership train size; ties use reasoning > concept > diagnosis > vqa > grounding","cross_task_components":len(cross),"total_removed":sum(map(len,remove_ids.values())),"removed_per_task":{TASKS[t][0]:len(remove_ids[t]) for t in TASKS},"decisions":decisions}
 return test_report,owner_report
def contracts():
 common={"score_range":[0,100],"empty_target_behavior":"construction error; dataset closure rejects empty targets"}
 return {
 "task_01_vqa":{**common,"primary_metric":"answer accuracy * 100","target_parser":"multiple-choice letter or canonical answer","prediction_parser":"case-insensitive letter/text parser","invalid_prediction_rule":"unparseable output is incorrect"},
 "task_02_diagnosis_classification":{**common,"primary_metric":"classification accuracy * 100","target_parser":"canonical diagnosis label","prediction_parser":"option letter or canonical label","invalid_prediction_rule":"unparseable output is incorrect"},
 "task_03_concept_recognition":{**common,"primary_metric":"micro concept F1 * 100","target_parser":"semicolon-separated normalized concepts with status/laterality","prediction_parser":"same concept parser with synonym map","invalid_prediction_rule":"empty output has zero precision/recall"},
 "task_04_visual_grounding":{**common,"primary_metric":"Acc@matched-IoU>=0.5 * 100","target_parser":"valid <box>x1,y1,x2,y2</box> coordinates in [0,1000]","prediction_parser":"same box parser","invalid_prediction_rule":"no valid box receives zero"},
 "task_05_reasoning_vqa":{**common,"primary_metric":"final-answer accuracy * 100","secondary_metrics":["reasoning ROUGE-L F1 * 100"],"target_parser":"Final answer option and canonical text","prediction_parser":"case-insensitive final-answer parser","invalid_prediction_rule":"unparseable output is incorrect"}}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--v0-root",default=str(V0));ap.add_argument("--output-root",default=str(V1));ap.add_argument("--artifact-root",default=str(ART));ap.add_argument("--concept-extraction",action="append",required=True);ap.add_argument("--seed",type=int,default=42);ap.add_argument("--medsg-target-per-task",type=int,default=6250);a=ap.parse_args()
 v0,out,art=map(Path,(a.v0_root,a.output_root,a.artifact_root))
 if out.resolve()==v0.resolve():raise RuntimeError("v1 output must differ from v0")
 extracts={}
 for p in a.concept_extraction:
  for x in rows(p):
   if x["id"] in extracts:raise RuntimeError(f"duplicate extraction {x['id']}")
   extracts[x["id"]]=x
 expected=sum(1 for _ in rows(out/"_concept/merged_input.jsonl"))
 if len(extracts)!=expected:raise RuntimeError(f"extraction count {len(extracts)} != {expected}")
 data={}
 data[1]={"train":[convert(x,1,"vqa") for x in rows(v0/"task_01_vqa/train.jsonl")],"test":[convert(x,1,"vqa") for x in rows(v0/"task_01_vqa/test.jsonl")]}
 data[2]={"train":[convert(x,2,"diagnosis_classification") for x in rows(v0/"task_02_diagnosis_classification/train.jsonl")],"test":[convert(x,2,"diagnosis_classification") for x in rows(v0/"task_02_diagnosis_classification/test.jsonl")]}
 data[3]={}
 for split in ("train","test"):
  values=[*rows(v0/f"task_03_concept_recognition/{split}.jsonl"),*rows(v0/f"task_04_caption_generation/{split}.jsonl")]
  data[3][split]=[convert(x,3,"concept_recognition",extracts) for x in values]
 ground_train=[convert(x,4,"visual_grounding") for x in rows(v0/"task_05_visual_grounding/train.jsonl")];ground_train,balance=balance_medsg(ground_train,a.seed,a.medsg_target_per_task)
 data[4]={"train":ground_train,"test":[convert(x,4,"visual_grounding") for x in rows(v0/"task_05_visual_grounding/test.jsonl")]}
 data[5]={"train":[convert(x,5,"reasoning_vqa") for x in rows(v0/"task_06_reasoning_vqa/train.jsonl")],"test":[convert(x,5,"reasoning_vqa") for x in rows(v0/"task_06_reasoning_vqa/test.jsonl")]}
 needed={image for splits in data.values() for values in splits.values() for row in values for image in row["images"]};hashes=load_hashes(v0/"audits/image_hash_cache.jsonl",needed)
 missing=needed-set(hashes)
 if missing:raise RuntimeError(f"missing v0 image hashes: {len(missing)}")
 test_report,owner_report=apply_lineage(data,hashes)
 art.mkdir(parents=True,exist_ok=True);writej(art/"medsg_balanced_sampling.json",balance);writej(art/"train_test_lineage_removal.json",test_report);writej(art/"cross_task_train_ownership.json",owner_report)
 spec={"status":"PASS","normalization":"lowercase ASCII alphanumeric tokens","dataset_aware_namespaces":True,"parsers":{"ISIC":"ISIC numeric image id","BraTS":"BraTS subject token","TCGA":"TCGA case barcode","IHC4BC":"IHC4BC sample token","VinDr":"VinDr UUID","MedTrinity":"official split_group_key","MedSG":"official source row id","MedThinkVQA":"case id"},"exact_sha256_is_lineage_token":True,"basename_tokens_are_dataset_family_namespaced":True}
 writej(art/"lineage_parser_spec.json",spec)
 stats={};manifest={"version":"v1","seed":a.seed,"source_v0_read_only":str(v0),"tasks":[]}
 for task,(name,skill) in TASKS.items():
  directory=out/name
  for split in ("train","test"):writejl(directory/f"{split}.jsonl",data[task][split])
  s={"task_id":task,"task_name":name,"skill":skill,"train":len(data[task]["train"]),"test":len(data[task]["test"]),"replacement_sampling_used":False}
  stats[name]=s;writej(directory/"statistics.json",s);manifest["tasks"].append(s)
  if task==3:
   freq=collections.Counter(normalize(c["canonical"]).replace("_"," ") for row in data[task]["train"] for c in row["metadata"].get("concepts",[]))
   writej(directory/"concept_vocabulary.json",{"train_only":True,"frequency":dict(freq.most_common()),"core_frequency_ge_10":[k for k,v in sorted(freq.items()) if v>=10]})
   writej(directory/"concept_synonyms.json",{"computed tomography":["ct","ct scan"],"magnetic resonance imaging":["mri","mr imaging"],"x-ray":["x ray","radiograph"]})
  for split in ("train","test"):writej(directory/f"{split}_manifest.json",{"task_id":task,"skill":skill,"split":split,"records":len(data[task][split]),"file":str(directory/f"{split}.jsonl")})
 writej(out/"manifests/medicalskill_cl_v1_manifest.json",manifest);writej(art/"medicalskill_cl_v1_manifest.json",manifest);writej(art/"task_statistics.json",stats);writej(out/"configs/metric_contracts.json",contracts());writej(art/"metric_contracts.json",contracts())
 repair=collections.Counter()
 for x in extracts.values(): repair.update(x.get("repair_reasons") or {})
 errors=collections.Counter(x.get("error") for x in extracts.values() if x.get("error"))
 repair_report={"status":"PASS" if not errors and not any(not x.get("target") for x in extracts.values()) else "FAIL","records":len(extracts),"json_or_parse_errors":dict(errors),"repair_reason_counts":dict(repair),"empty_targets":sum(not x.get("target") for x in extracts.values()),"max_concepts_observed":max(len(x.get("concepts") or []) for x in extracts.values()),"raw_response_preserved":all("raw_response" in x for x in extracts.values()),"prompt_hashes":sorted({x.get("prompt_hash") for x in extracts.values()}),"model_id":MODEL_ID,"revision":REVISION}
 writej(art/"concept_extraction_repair_breakdown.json",repair_report)
 config={"status":"PASS","tasks":TASKS,"seed":a.seed,"epochs":1,"validation_split":None,"evaluate_on":"test","medsg_target_per_official_task":a.medsg_target_per_task,"five_skill_contract":True,"model_id":"Qwen/Qwen3-VL-8B-Instruct","model_revision":"0c351dd01ed87e9c1b53cbc748cba10e6187ff3b","concept_model_id":MODEL_ID,"concept_model_revision":REVISION}
 writej(art/"effective_config.json",config);writej(out/"configs/effective_config.json",config)
 print(json.dumps({"status":"PASS","tasks":stats,"lineage_train_removed":test_report["total_train_removed"],"cross_task_train_removed":owner_report["total_removed"],"medsg":balance["total_output_turns"]},indent=2))
if __name__=="__main__":main()
