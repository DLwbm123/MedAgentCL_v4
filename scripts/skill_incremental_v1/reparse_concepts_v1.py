#!/usr/bin/env python3
import argparse,collections,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("concept_extract_v1",HERE/"extract_medtrinity_concepts_v1.py");mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def rows(p):
 with open(p,encoding="utf-8-sig") as f:
  for line in f:
   if line.strip():yield json.loads(line)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--input",action="append",required=True);ap.add_argument("--artifact-root",default="/root/MedAgentCL_v4/artifacts/medicalskill_cl_v1_closure");a=ap.parse_args();art=Path(a.artifact_root);all_rows=[];reasons=collections.Counter();errors=[]
 for name in a.input:
  path=Path(name);out=[]
  for row in rows(path):
   try:
    concepts=mod.parse(row["raw_response"],row["source_caption"]);row["concepts"]=concepts;row["target"]=mod.target(concepts);row["repair_reasons"]=dict(collections.Counter(mod.LAST_REPAIRS));row["parser_version"]="medicalskill_concept_parser_v2_2"
    reasons.update(row["repair_reasons"])
    if not row["target"]:errors.append({"id":row["id"],"error":"empty_after_reparse"})
   except Exception as e:row["error"]=f"{type(e).__name__}: {e}";errors.append({"id":row["id"],"error":row["error"]})
   out.append(row)
  tmp=Path(str(path)+".tmp")
  with tmp.open("w",encoding="utf-8") as f:
   for row in out:f.write(json.dumps(row,ensure_ascii=False)+"\n")
  tmp.replace(path);all_rows.extend(out)
  import re
  match=re.search(r"part_(\d+)",path.stem)
  if match:mod.qa(art/f"concept_full_part_{match.group(1)}_qa.json",out,42+int(match.group(1)))
 report={"status":"PASS" if not errors else "FAIL","records":len(all_rows),"parser_version":"medicalskill_concept_parser_v2_2","repair_reason_counts":dict(reasons),"errors":errors[:100],"errors_truncated":len(errors)>100,"raw_responses_preserved":all("raw_response" in x for x in all_rows),"max_concepts":max(len(x.get("concepts") or []) for x in all_rows)}
 mod.atomic(art/"concept_reparse_report.json",report);print(json.dumps(report,indent=2));return 0 if report["status"]=="PASS" else 1
if __name__=="__main__":raise SystemExit(main())
