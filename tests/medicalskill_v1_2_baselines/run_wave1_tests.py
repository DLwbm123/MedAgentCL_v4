#!/usr/bin/env python3
from __future__ import annotations
import json,subprocess,tempfile,sys
from pathlib import Path
ROOT=Path("/root/MedAgentCL_v4")
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from peft import LoraConfig,PeftModel,get_peft_model
from safetensors.torch import load_file
from med_prism.baselines.octopus import attach_historical_stage2_extras,historical_extra_runtime_stats,lora_pairs
from med_prism.baselines.olora import olora_regularizers,inspect_olora_serialization
from med_prism.baselines.reglora import build_importance_mask,inspect_importance_mask,load_importance_mask,reglora_regularizer,selected_entry_abs_mean

DATA=Path('/remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k')
class Toy(nn.Module):
 def __init__(self,width=64): super().__init__();self.q_proj=nn.Linear(width,width);self.v_proj=nn.Linear(width,width)
 def forward(self,x): return self.v_proj(torch.tanh(self.q_proj(x)))
def adapter(rank): return get_peft_model(Toy(),LoraConfig(r=rank,lora_alpha=2*rank,lora_dropout=0.0,bias='none',target_modules=['q_proj','v_proj']))
def fill_b(model,value=0.1):
 with torch.no_grad():
  for n,p in model.named_parameters():
   if '.lora_B.' in n:p.fill_(value)
def assert_grad(model,pattern):
 values=[p.grad for n,p in model.named_parameters() if pattern in n]
 assert values and all(x is not None and torch.isfinite(x).all() for x in values),pattern

def test_olora(tmp):
 old=adapter(16);fill_b(old,.2);old_dir=tmp/'old';old.save_pretrained(old_dir)
 current=adapter(16);fill_b(current,.1);meta=attach_historical_stage2_extras(current,[old_dir]);assert meta['historical_adapter_count']==1
 orth,l2,details=olora_regularizers(current);assert orth.item()>0 and l2.item()>0 and details['historical_a_block_count']==2
 out=current(torch.randn(2,3,64));loss=out.square().mean()+.5*orth;loss.backward();assert_grad(current,'.lora_A.');assert_grad(current,'.lora_B.')
 extras=[p for n,p in current.named_parameters() if '.extras_' in n];assert extras and all(not p.requires_grad and p.grad is None for p in extras)
 stats=historical_extra_runtime_stats(current);assert stats['all_historical_extras_active_in_forward'] and stats['trainable_extra_tensor_count']==0
 cur_dir=tmp/'current';current.save_pretrained(cur_dir);serial=inspect_olora_serialization(cur_dir);assert serial['serialized_historical_extra_tensor_count']==0
 keys=load_file(str(cur_dir/'adapter_model.safetensors'));assert all('extras_' not in k for k in keys)

def test_reglora(tmp):
 a=torch.randn(5,9,requires_grad=True);b=torch.randn(7,5,requires_grad=True);index=torch.tensor([[0,0],[2,4],[2,4],[6,8]],dtype=torch.int32)
 bounded=selected_entry_abs_mean(a,b,index,2);dense=(b@a)[index[:,0],index[:,1]].abs().mean()
 bounded_grad=torch.autograd.grad(bounded,(a,b),retain_graph=True);dense_grad=torch.autograd.grad(dense,(a,b))
 assert torch.allclose(bounded,dense,rtol=1e-6,atol=1e-7)
 assert all(torch.allclose(x,y,rtol=1e-5,atol=1e-6) for x,y in zip(bounded_grad,dense_grad))
 t1=adapter(16);fill_b(t1,.15);d1=tmp/'reg1';t1.save_pretrained(d1);m1=tmp/'m1.pt';v1=build_importance_mask(d1,m1);assert v1['protected_position_fraction']>0
 raw1=load_importance_mask(m1);assert all(x.shape[0]==int(.02*64*64) for x in raw1['masks'].values())
 t2=adapter(16);fill_b(t2,.11);attach_historical_stage2_extras(t2,[d1]);reg,meta=reglora_regularizer(t2,m1);assert reg.item()>0 and meta['protected_position_count']>0
 dense=t2.q_proj.weight.new_zeros(())
 for name,a,b in lora_pairs(t2):
  index=raw1['masks'][name];dense=dense+torch.abs((b@a)[index[:,0],index[:,1]]).mean()
 dense=dense/len(lora_pairs(t2));assert torch.allclose(reg,dense,rtol=1e-6,atol=1e-7)
 (t2(torch.randn(2,4,64)).square().mean()+2500*reg).backward();assert_grad(t2,'.lora_A.');assert_grad(t2,'.lora_B.');assert all(p.grad is None for n,p in t2.named_parameters() if '.extras_' in n)
 d2=tmp/'reg2';t2.save_pretrained(d2);m2=tmp/'m2.pt';build_importance_mask(d2,m2,m1);raw2=load_importance_mask(m2)
 for name in raw1['masks']: assert raw2['masks'][name].shape[0]==2*raw1['masks'][name].shape[0]
 i2=inspect_importance_mask(m2);assert i2['mask_sha256']!=inspect_importance_mask(m1)['mask_sha256'] and i2['mask_storage_bytes']>0

def test_moelora(tmp):
 from med_prism.baselines.moelora import enable_moelora_patch,moelora_runtime_stats
 enable_moelora_patch();m=adapter(48);fill_b(m,.12);x=torch.randn(2,5,64);y=m(x);stats=moelora_runtime_stats(m)
 assert stats['num_experts']==4 and stats['rank_per_expert']==12 and stats['total_expert_rank']==48
 assert stats['max_probability_sum_error']<1e-5 and all(v>0 for v in stats['mean_expert_utilization'])
 y.square().mean().backward();assert_grad(m,'lora_router');assert_grad(m,'.lora_A.');assert_grad(m,'.lora_B.')
 d=tmp/'moe';m.save_pretrained(d);state=load_file(str(d/'adapter_model.safetensors'));router=[k for k in state if 'lora_router' in k];assert len(router)==2
 restored=PeftModel.from_pretrained(Toy(),d,is_trainable=False);restored(torch.randn(1,2,64));restats=moelora_runtime_stats(restored);assert restats['min_forward_calls']>0

def test_contracts_and_static():
 forbidden=['HiDe-LLaVA','PASs-MoE','ModalPrompt','D-MoLE','Sparse Spectral']
 wave_files=list((ROOT/'med_prism/baselines').glob('*lora.py'))+list((ROOT/'scripts/medicalskill_v1_2_wave1').glob('*'))
 text='\n'.join(p.read_text(errors='ignore') for p in wave_files if p.is_file())
 assert not any(value in text for value in forbidden)
 for method in ('olora','reglora','moelora'):
  result=subprocess.run([str(ROOT/'scripts/medicalskill_v1_2_wave1/run_wave1_v1_2.sh'),'--method',method,'--output-root',f'/tmp/wave1_check_{method}','--check-only'],cwd=ROOT,text=True,capture_output=True)
  assert result.returncode==0,(method,result.stdout,result.stderr);payload=json.loads(result.stdout[:result.stdout.rfind('}')+1]);assert payload['status']=='PASS' and payload['backbone']['model_revision']=='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b' and payload['dataset']['selected_ids_sha256']

def main():
 with tempfile.TemporaryDirectory(prefix='wave1_unit_') as value:
  tmp=Path(value);test_olora(tmp);print('PASS O-LoRA unit/serialization/gradient');test_reglora(tmp);print('PASS RegLoRA mask/lifecycle/gradient');test_moelora(tmp);print('PASS MoELoRA router/gradient/reload')
 test_contracts_and_static();print('PASS immutable contracts / forbidden scope');print('WAVE1 UNIT TESTS PASS')
if __name__=='__main__':main()
