#!/usr/bin/env python3
"""Aggregate Wave-1 lower-triangular outputs with the frozen native scorers."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from scripts.medicalskill_v1_2_baselines.baseline_harness import EVALUATION_PROTOCOL,atomic_json,load_json,validate_completion,validate_dataset_lock
IDS={"olora":"olora_qwen3vl_r16","reglora":"reglora_sefe_component_qwen3vl_r16","moelora":"coin_moelora_qwen3vl_total_r48_e4"}
STEMS={"olora":"primary_cumulative_task_","reglora":"primary_cumulative_task_","moelora":"primary_routed_task_"}
TASKS=["VQA","Diagnosis","Concept","Grounding","Reasoning"]
def main():
 p=argparse.ArgumentParser();p.add_argument("--method",choices=IDS,required=True);p.add_argument("--data-root",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True);p.add_argument("--eval-limit",type=int,default=0);a=p.parse_args();root=a.output_root.resolve();dataset=validate_dataset_lock(a.data_root.resolve());run=load_json(root/"run_manifest.json");mid=IDS[a.method]
 if run.get("method")!=mid: raise RuntimeError("Run method mismatch")
 matrix=[];stages={};total_main=total_aux=total_eval=total_gpu=0.0
 for stage in range(1,6):
  row=[]
  for task in range(1,6):
   path=root/"evaluation"/f"stage_{stage:02d}"/f"{STEMS[a.method]}{task:02d}.summary.json";row.append(float(load_json(path)["primary_score"]) if task<=stage else None)
  matrix.append(row);completion_path=root/a.method/f"stage_{stage:02d}"/"completion.json";c=validate_completion(completion_path,method=mid,stage=stage,immutable_config_sha256=run["immutable_config_sha256"]);e=load_json(root/"evaluation"/f"stage_{stage:02d}"/"stage_evaluation_summary.json")
  if e.get("status")!="PASS" or e.get("cell_count")!=stage or e.get("evaluation_protocol")!=EVALUATION_PROTOCOL or e.get("active_adapter",{}).get("oracle_task_id") is not False: raise RuntimeError(f"Evaluation contract stage {stage}")
  rt=c["runtime_accounting"];total_main+=rt["main_train_seconds"];total_aux+=rt["auxiliary_seconds"];total_eval+=rt["evaluation_seconds"];total_gpu+=rt["training_gpu_hours"];stages[str(stage)]={"completion":str(completion_path),"checkpoint":c["checkpoint"],"parameter_accounting":c["parameter_accounting"],"runtime_accounting":rt}
 with (root/"primary_cumulative_matrix.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.writer(f);w.writerow(["after_stage",*TASKS,"average"])
  for i,row in enumerate(matrix,1): seen=[x for x in row if x is not None];w.writerow([i,*["" if x is None else f"{x:.8f}" for x in row],f"{sum(seen)/len(seen):.8f}"])
 final=[float(x) for x in matrix[-1]];diag=[matrix[i][i] for i in range(5)];best=[max(matrix[s][t] for s in range(t,5)) for t in range(5)];forget=[best[i]-final[i] for i in range(5)];bwt=[final[i]-diag[i] for i in range(4)];metrics={"status":"PASS","method":mid,"scale":"0..1","evaluation_protocol":EVALUATION_PROTOCOL,"matrix_cells":15,"final_average_score":sum(final)/5,"per_task_best_score":best,"per_task_final_score":final,"per_task_forgetting":forget,"average_forgetting":sum(forget)/5,"BWT":sum(bwt)/4,"BWT_terms":bwt,"FWT":None};atomic_json(root/"cl_metrics.json",metrics)
 summary={"status":"PASS","format_version":"medicalskill_cl_wave1_summary_v1","method":mid,"data_root":str(a.data_root.resolve()),"output_root":str(root),"eval_limit_per_task":a.eval_limit,"dataset_lock_sha256":dataset["dataset_lock_sha256"],"selected_ids_sha256":dataset["selected_ids_sha256"],"stages":stages,"metrics":metrics,"runtime":{"main_train_seconds_t1_t5":total_main,"auxiliary_seconds_t1_t5":total_aux,"evaluation_seconds_t1_t5":total_eval,"training_gpu_hours_t1_t5":total_gpu,"evaluation_reported_separately":True},"historical_memory":run["historical_memory"]};atomic_json(root/"baseline_experiment_summary.json",summary);print(json.dumps(summary,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
