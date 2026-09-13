#!/usr/bin/env python3
"""Aggregate the formal Replay+LoRA lower triangle."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from scripts.medicalskill_v1_2_baselines.baseline_harness import atomic_json

METHOD="replay_lora_r48_balanced_total_v1"
TASKS=["VQA","Diagnosis","Concept","Grounding","Reasoning"]
def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def main():
 p=argparse.ArgumentParser();p.add_argument("--data-root",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--eval-limit",type=int,default=0);a=p.parse_args();root=a.output_root.resolve()
 values=[]
 for stage in range(1,6):
  row=[]
  summary=load(root/"evaluation"/f"stage_{stage:02d}"/"stage_evaluation_summary.json")
  assert summary["method"]==METHOD and summary["evaluated_task_ids"]==list(range(1,stage+1))
  for task in range(1,6): row.append(float(load(root/"evaluation"/f"stage_{stage:02d}"/f"primary_cumulative_task_{task:02d}.summary.json")["primary_score"]) if task<=stage else None)
  values.append(row)
 with (root/"primary_cumulative_matrix.csv").open("w",newline="",encoding="utf-8") as h:
  w=csv.writer(h);w.writerow(["after_stage",*TASKS,"average"])
  for i,row in enumerate(values,1): seen=[v for v in row if v is not None];w.writerow([i,*["" if v is None else f"{v:.8f}" for v in row],f"{sum(seen)/len(seen):.8f}"])
 final=[float(v) for v in values[-1]];diag=[values[i][i] for i in range(5)];best=[max(values[s][t] for s in range(t,5)) for t in range(5)];forget=[best[i]-final[i] for i in range(5)];bwt=[final[i]-diag[i] for i in range(4)]
 metrics={"status":"PASS","method":METHOD,"matrix_cells":15,"final_average_score":sum(final)/5,"per_task_best_score":best,"per_task_final_score":final,"per_task_forgetting":forget,"average_forgetting":sum(forget)/5,"BWT":sum(bwt)/4,"BWT_terms":bwt,"FWT":None}
 atomic_json(root/"cl_metrics.json",metrics);print(json.dumps(metrics,indent=2));return 0
if __name__=="__main__": raise SystemExit(main())
