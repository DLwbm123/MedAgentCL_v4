#!/usr/bin/env python3
import argparse,json,math
from pathlib import Path
def read(p):return json.loads(Path(p).read_text()) if Path(p).is_file() else {}
def main():
 a=argparse.ArgumentParser();a.add_argument("--root",required=True);a.add_argument("--output",required=True);o=a.parse_args();root=Path(o.root);tasks={};ok=True
 for d in sorted(root.glob("task_*")):
  exit_info=read(d/"exit_code.json");grad=read(d/"training_gradient_summary.json");params=read(d/"parameter_audit.json");trace=[json.loads(x) for x in (d/"training_trace.jsonl").read_text().splitlines() if x.strip()] if (d/"training_trace.jsonl").is_file() else []
  finite_loss=bool(trace) and all(math.isfinite(float(x.get("loss",float("nan")))) for x in trace);finite_grad=bool(trace) and all(math.isfinite(float(x.get("q_gradient_norm",float("nan")))) and math.isfinite(float(x.get("v_gradient_norm",float("nan")))) for x in trace);passed=exit_info.get("exit_code")==0 and grad.get("status")=="PASS" and params.get("status")=="PASS" and finite_loss and finite_grad
  tasks[d.name]={"status":"PASS" if passed else "FAIL","exit_code":exit_info.get("exit_code"),"model_loaded":params.get("status")=="PASS","gpu_used":bool(grad.get("peak_allocated",0)),"loss_finite":finite_loss,"gradients_finite":finite_grad,"optimizer_steps":grad.get("steps_with_gradients"),"trainable_parameter_count":params.get("trainable_parameter_count"),"peak_gpu_allocated_bytes":grad.get("peak_allocated"),"peak_gpu_reserved_bytes":grad.get("peak_reserved"),"trace":trace};ok &= passed
 report={"status":"PASS" if ok and len(tasks)==5 else "FAIL","model_id":"Qwen/Qwen3-VL-8B-Instruct","model_revision":"0c351dd01ed87e9c1b53cbc748cba10e6187ff3b","local_files_only":True,"tasks":tasks};p=Path(o.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2));return 0 if report["status"]=="PASS" else 1
if __name__=="__main__":raise SystemExit(main())
