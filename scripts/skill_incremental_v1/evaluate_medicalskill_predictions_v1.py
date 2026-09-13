#!/usr/bin/env python3
"""Evaluate MedicalSkill-CL-v1 predictions with fixed 0-100 metrics."""
import argparse,json,re
from pathlib import Path
BOX=re.compile(r"<box>\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\s*</box>")
def norm(x): return re.sub(r"[^a-z0-9]+"," ",str(x).casefold()).strip()
def rows(p):
 with open(p,encoding="utf-8-sig") as f:
  for l in f:
   if l.strip(): yield json.loads(l)
def text(x,k):
 if x.get(k) is not None:return str(x[k])
 m=x.get("messages") or []; return str(m[-1].get("content") or "") if m else ""
def option(s):
 m=re.search(r"(?:final answer\s*:\s*)?\b([A-Z])(?:[.)]|\b)",s,re.I); return m.group(1).upper() if m else norm(s)
def concepts(s): return {norm(x) for x in s.split(";") if norm(x)}
def lcs(a,b):
 old=[0]*(len(b)+1)
 for x in a:
  new=[0]
  for j,y in enumerate(b,1): new.append(old[j-1]+1 if x==y else max(old[j],new[-1]))
  old=new
 return old[-1]
def rouge(a,b):
 a=norm(a).split();b=norm(b).split()
 if not a or not b:return 0
 n=lcs(a,b);p=n/len(a);r=n/len(b);return 2*p*r/(p+r) if p+r else 0
def boxes(s):
 out=[]
 for m in BOX.finditer(s):
  b=tuple(map(float,m.groups()))
  if 0<=b[0]<b[2]<=1000 and 0<=b[1]<b[3]<=1000:out.append(b)
 return out
def iou(a,b):
 inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1])); union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter; return inter/union if union else 0
def ground(a,b):
 p=boxes(a);r=boxes(b); denominator=len(r)
 if not p or not r:return 0
 scores=[]
 while p and r:
  best=max(((iou(x,y),i,j) for i,x in enumerate(p) for j,y in enumerate(r)),key=lambda z:z[0])
  scores.append(best[0]); p.pop(best[1]); r.pop(best[2])
 return sum(scores)/denominator
def final_part(s):
 parts=re.split(r"final\s+answer\s*:",s,flags=re.I); return parts[-1]
def core_name(value,vocab):
 value=norm(re.sub(r"^(present|uncertain|negated)\s*:\s*","",str(value),flags=re.I))
 if value in vocab:return value
 stripped=re.sub(r"^(left|right|bilateral|midline)\s+","",value)
 return stripped if stripped in vocab else ""
def main():
 p=argparse.ArgumentParser();p.add_argument("--predictions",required=True);p.add_argument("--tasks",default="");p.add_argument("--seed",type=int,default=42);p.add_argument("--resume",action="store_true");p.add_argument("--task",type=int,choices=range(1,6),required=True);p.add_argument("--output");p.add_argument("--data-root",default="/remote-home/wangbomin/MedicalSkill-CL-v1");p.add_argument("--dry-run",action="store_true");p.add_argument("--verify-only",action="store_true");a=p.parse_args();data=list(rows(a.predictions))
 if a.dry_run:print(json.dumps({"status":"DRY_RUN","rows":len(data),"task":a.task}));return 0
 exact=[];rs=[];ious=[];tp=fp=fn=ctp=cfp=cfn=0;core_vocab=set()
 if a.task==3:
  vocab=Path(a.data_root)/"task_03_concept_recognition/concept_vocabulary.json"
  if not vocab.is_file():raise RuntimeError(f"missing train-only concept vocabulary: {vocab}")
  core_vocab={norm(v) for v in json.loads(vocab.read_text()).get("core_frequency_ge_10",[])}
 for x in data:
  pred,ref=text(x,"prediction"),text(x,"reference")
  if a.task in (1,2):exact.append(option(pred)==option(ref) or norm(pred)==norm(ref))
  elif a.task==3:
   pp,rr=concepts(pred),concepts(ref);tp+=len(pp&rr);fp+=len(pp-rr);fn+=len(rr-pp)
   pc={v for item in pp if (v:=core_name(item,core_vocab))}; meta=x.get("metadata") or {}; declared=meta.get("core_target_concepts")
   rc={norm(v) for v in declared} if isinstance(declared,list) else {v for item in rr if (v:=core_name(item,core_vocab))}
   ctp+=len(pc&rc);cfp+=len(pc-rc);cfn+=len(rc-pc)
  elif a.task==4:ious.append(ground(pred,ref))
  else:exact.append(option(final_part(pred))==option(final_part(ref)));rs.append(rouge(re.split(r"final\s+answer\s*:",pred,flags=re.I)[0],re.split(r"final\s+answer\s*:",ref,flags=re.I)[0]))
 if a.task in (1,2,5):
  m={"accuracy" if a.task!=6 else "final_answer_accuracy":100*sum(exact)/len(data) if data else 0}
  if a.task==5:m["reasoning_rouge_l"]=100*sum(rs)/len(rs) if rs else 0
 elif a.task==3:m={"micro_f1_all":100*(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0),"micro_f1_core":100*(2*ctp/(2*ctp+cfp+cfn) if 2*ctp+cfp+cfn else 0),"all_counts":{"tp":tp,"fp":fp,"fn":fn},"core_counts":{"tp":ctp,"fp":cfp,"fn":cfn},"core_vocabulary_size":len(core_vocab)}
 elif a.task==4:m={"accuracy_iou_0.5":100*sum(v>=.5 for v in ious)/len(ious) if ious else 0,"mean_iou":100*sum(ious)/len(ious) if ious else 0}
 report={"status":"PASS","task":a.task,"total":len(data),"metrics":m,"range":[0,100]}
 if a.output:Path(a.output).write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps(report,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
