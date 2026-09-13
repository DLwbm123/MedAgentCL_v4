#!/usr/bin/env python3
"""GPU fidelity probe of RegLoRA loss/gradients on a real adapter checkpoint."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import torch
from med_prism.baselines.reglora import _checkpoint_pairs,load_importance_mask,inspect_importance_mask
from scripts.medicalskill_v1_2_baselines.baseline_harness import atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--mask',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();started=time.perf_counter();pairs,info=_checkpoint_pairs(a.checkpoint.resolve());m=load_importance_mask(a.mask.resolve())['masks']
 if set(pairs)!=set(m):raise RuntimeError('checkpoint/mask layer mismatch')
 total=0.0;grad_a=grad_b=0;nonzero_layers=0
 for name,(a_cpu,b_cpu) in sorted(pairs.items()):
  aa=a_cpu.float().to(a.device).requires_grad_(True);bb=b_cpu.float().to(a.device).requires_grad_(True);idx=m[name].to(a.device)
  selected=(bb[idx[:,0]]*aa.transpose(0,1)[idx[:,1]]).sum(-1);term=selected.abs().mean()/len(pairs);value=float(term.detach().cpu());total+=value
  if value>0:nonzero_layers+=1
  term.backward();grad_a+=int(aa.grad is not None and torch.count_nonzero(aa.grad).item()>0);grad_b+=int(bb.grad is not None and torch.count_nonzero(bb.grad).item()>0)
 value={'status':'PASS' if total>0 and grad_a==len(pairs) and grad_b==len(pairs) else 'FAIL','semantics':'actual Task2 adapter BA at actual Task1 protected indices; mean across 72 q/v linears','checkpoint':info['checkpoint'],'checkpoint_sha256':info['adapter_weights_sha256'],'mask':inspect_importance_mask(a.mask),'raw_reglora_loss':total,'weighted_reglora_loss':2500.0*total,'layer_count':len(pairs),'nonzero_loss_layer_count':nonzero_layers,'lora_a_layers_with_nonzero_gradient':grad_a,'lora_b_layers_with_nonzero_gradient':grad_b,'device':a.device,'elapsed_seconds':time.perf_counter()-started};atomic_json(a.output.resolve(),value);print(json.dumps(value,indent=2));return 0 if value['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
