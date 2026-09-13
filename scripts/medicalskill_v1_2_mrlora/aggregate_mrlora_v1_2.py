#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from scripts.medicalskill_v1_2_baselines.baseline_harness import atomic_json
METHOD="mrlora_mllm_cl_qwen3vl_r16";TASKS=["VQA","Diagnosis","Concept","Grounding","Reasoning"]
def load(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument("--output-root",type=Path,required=True);a=p.parse_args();r=a.output_root.resolve();v=[]
 for s in range(1,6):
  summary=load(r/"evaluation"/f"stage_{s:02d}"/"stage_evaluation_summary.json");assert summary["method"]==METHOD and summary["active_adapter"]["router_stage"]==s and summary["evaluated_task_ids"]==list(range(1,s+1));v.append([float(load(r/"evaluation"/f"stage_{s:02d}"/f"primary_cumulative_task_{t:02d}.summary.json")["primary_score"]) if t<=s else None for t in range(1,6)])
 with (r/"primary_cumulative_matrix.csv").open("w",newline="",encoding="utf-8") as h:
  w=csv.writer(h);w.writerow(["after_stage",*TASKS,"average"])
  for s,row in enumerate(v,1):seen=[x for x in row if x is not None];w.writerow([s,*["" if x is None else f"{x:.8f}" for x in row],f"{sum(seen)/len(seen):.8f}"])
 final=[float(x) for x in v[-1]];diag=[v[i][i] for i in range(5)];best=[max(v[s][t] for s in range(t,5)) for t in range(5)];forget=[best[i]-final[i] for i in range(5)];bwt=[final[i]-diag[i] for i in range(4)];m={"status":"PASS","method":METHOD,"matrix_cells":15,"final_average_score":sum(final)/5,"average_forgetting":sum(forget)/5,"per_task_forgetting":forget,"BWT":sum(bwt)/4,"BWT_terms":bwt,"FWT":None};atomic_json(r/"cl_metrics.json",m);print(json.dumps(m,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
