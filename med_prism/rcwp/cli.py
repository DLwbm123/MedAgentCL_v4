"""Boundary summary / tiny real smoke. Explicit current training data only."""
import argparse
import json
from pathlib import Path
import time
import traceback
import torch
from med_prism.transport.checkpoint import read_state, sha256_file
from med_prism.transport.cli import load_calibration_model
from .data import encode_boundary_rows
from med_prism.transport.banks import fingerprint,check_invariants
from med_prism.transport.diagnostics import write_json
from .boundary import BoundaryField,field_quality,model_inputs
from .runtime import load_summary,save_summary,wrappers,WriteProtection


def run(args):
    root = Path(args.output_root).resolve()
    if root.exists(): raise FileExistsError(root)
    data = Path(args.current_train).resolve()
    if data.name != "train.jsonl":
        raise ValueError("Only explicitly current train.jsonl allowed; never old dev/test")
    state = read_state(args.state)
    if state.task_id != args.current_task:
        raise ValueError("Boundary current task/state mismatch")
    if args.smoke:
        fit,holdout = 4,2
    else:
        fit,holdout = 128,64
    root.mkdir(parents=True)
    started = time.time()
    config = {"ridge":1e-3,"lambda_rcwp":.1,"fit_samples":fit,"holdout_samples":holdout,
        "token_selection":"all valid nonpadding","coordinate_std_floor":1e-6,
        "negative_budget":0,"seed":42,"smoke_only":args.smoke,"tpm_enabled":False,
        "fit_holdout_group_disjoint":True}
    files = [Path(args.state).resolve(),data,*map(Path,state.components())]
    sources = {str(p):sha256_file(p) for p in files}
    source = {"state":str(Path(args.state).resolve()),"hashes":sources,
        "shared_manifest":state.shared_manifest,"private_manifests":state.private_manifests,
        "protocol":"explicit current-task boundary reconstruction; train only; temporary records discarded"}
    code_hashes={str(p):sha256_file(p) for p in Path(__file__).parent.glob("*.py")}
    write_json(root/"request.json",{"config":config,"source":source,"code_hashes":code_hashes})
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model,template = load_calibration_model(state,1024)
    template.processor.image_processor.min_pixels=200704
    template.processor.image_processor.max_pixels=200704
    before = fingerprint(model)
    torch.cuda.reset_peak_memory_stats()
    rows = [json.loads(l) for l in data.read_text().splitlines() if l.strip()]
    for row in rows:
        if isinstance(row.get("task_id"),int) and row["task_id"] != args.current_task:
            raise ValueError("Another task in current training data")
    records,rejected = encode_boundary_rows(rows,template,fit,holdout)
    del rows
    # No raw rows, IDs, per-token features, gradients or episodic logits persisted.
    print(f"READY: current task {state.task_id}; {fit}+{holdout} samples; all valid tokens",flush=True)
    field = BoundaryField(model,state.task_id)
    for record in records:
        field.observe(record["batch"],record["split"])
    summary = field.finish(source,config)
    quality = field_quality(model,[r for r in records if r["split"]=="holdout"],summary,field.constant)
    write_json(root/"field_quality.json",quality)
    save_summary(root/"summary.pt",summary)
    reloaded = load_summary(root/"summary.pt",wrappers(model))
    for name in summary["fields"]:
        for k in ("M","mu","scale"):
            if not torch.equal(summary["fields"][name][k],reloaded["fields"][name][k]):
                raise RuntimeError("Summary reload mismatch")
    smoke = None
    if args.smoke:
        # Ephemeral next bank only: never save a new model checkpoint or optimizer step.
        original = fingerprint(model)
        old_keys = {n:set(m.experts.keys()) for n,m in wrappers(model).items()}
        active = {n:list(m._active_tasks) for n,m in wrappers(model).items()}
        try:
            for m in wrappers(model).values():
                alpha=float(m.task_experts(state.task_id)[0].alpha)
                m.add_private_task(state.task_id+1,private_rank=16,experts_per_task=16,alpha=alpha,trainable=True)
                # Seeded nonzero toy current B exercises both A/B gradients, no training.
                for e in m.task_experts(state.task_id+1):
                    with torch.no_grad(): e.B.fill_(.001)
            # Smoke protects exactly this legally reconstructed boundary bank. It does
            # not fabricate summaries for earlier tasks. Direct low-rank API tests the
            # next-task gradient ownership; full history lifecycle covered by CPU tests.
            from .core import factors,signed_score,net_penalty
            terms=[]
            handles=[]
            batch=records[0]["batch"]
            for name,m in wrappers(model).items():
                def capture(module,inputs,name=name):
                    h=inputs[0]
                    f=reloaded["fields"][name]
                    with torch.enable_grad(),torch.autocast(h.device.type,enabled=False):
                        a,b=factors(module,state.task_id+1,False)
                        ai,bi=factors(module,state.task_id)
                        terms.append(signed_score(h,a,b,ai,bi,*[f[k].to(h.device) for k in ("M","mu","scale")]).sum())
                handles.append(m.register_forward_pre_hook(capture))
            try:
                with torch.no_grad():
                    output=model(**model_inputs(batch,next(model.parameters()).device),use_cache=False,return_dict=True)
                del output
            finally:
                for handle in handles: handle.remove()
            score=torch.stack(terms).sum().reshape(1,1)
            loss=net_penalty(score,score.new_tensor([summary["nu"]]))
            # Signed score gradient verifies ownership even if the legitimate penalty is zero.
            score.sum().backward()
            current_marker=f"task_{state.task_id+1:04d}__"
            current=[(n,p) for n,p in model.named_parameters() if current_marker in n]
            forbidden=[n for n,p in model.named_parameters() if current_marker not in n and p.grad is not None]
            grad_a=sum(float(p.grad.float().norm()) for n,p in current if n.endswith(".A") and p.grad is not None)
            grad_b=sum(float(p.grad.float().norm()) for n,p in current if n.endswith(".B") and p.grad is not None)
            if forbidden or not grad_a>0 or not grad_b>0: raise RuntimeError("Smoke gradient ownership failure")
            smoke={"status":"PASS","next_task":state.task_id+1,"protected_task":state.task_id,
                "rcwp_loss":float(loss.detach()),"signed_score":float(score.detach()),
                "A_gradient_norm_sum":grad_a,"B_gradient_norm_sum":grad_b,"forbidden_gradients":forbidden,
                "optimizer_steps":0,"checkpoint_reload":"exact summary plus source component load; no model checkpoint saved"}
        finally:
            for name,m in wrappers(model).items():
                for key in set(m.experts.keys())-old_keys[name]: del m.experts[key]
                m.set_active_tasks(active[name])
            model.zero_grad(set_to_none=True)
            check_invariants(original,fingerprint(model),set())
    del records,field
    invariants=check_invariants(before,fingerprint(model),set())
    state.validate()
    if any(sha256_file(p)!=v for p,v in sources.items()): raise RuntimeError("Source changed")
    if any(sha256_file(p)!=v for p,v in code_hashes.items()): raise RuntimeError("Code changed during smoke; use fresh run")
    write_json(root/"invariants.json",invariants)
    write_json(root/"completion.json",{"status":"PASS","field_quality":quality["status"],"smoke":smoke,
        "summary_sha256":sha256_file(root/"summary.pt"),"source_state_sha256":sha256_file(args.state),
        "elapsed_seconds":time.time()-started,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        "summary_bytes":(root/"summary.pt").stat().st_size,"summary_tensor_bytes":sum(f[k].numel()*4 for f in summary["fields"].values() for k in ("M","mu","scale")),
        "encoding_rejections":rejected,"source_hashes_unchanged":True,"model_tensors_unchanged":True})
    write_json(root/"summary_diagnostics.json",{n:{"effective_B_rank":f["effective_B_rank"],**f["diagnostics"]} for n,f in summary["fields"].items()})
    print("COMPLETE "+str(root),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("state","current-train","output-root"): p.add_argument("--"+name,required=True)
    p.add_argument("--current-task",type=int,required=True)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    try: run(args)
    except Exception as exc:
        root=Path(args.output_root)
        if root.exists() and not (root/"completion.json").exists():
            write_json(root/"failure.json",{"error":str(exc),"traceback":traceback.format_exc()})
        raise


if __name__=="__main__": main()
