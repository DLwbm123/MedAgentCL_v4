import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch import nn
import torch.nn.functional as F
from med_prism.adapters.shared_private import SharedPrivateLinear
from med_prism.transport.banks import fingerprint

spec=importlib.util.spec_from_file_location('p3_audit',Path(__file__).resolve().parents[2]/'scripts/medicalskill_v2_tpm/audit_p3_activation.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class ActivationTests(unittest.TestCase):
    @torch.no_grad()
    def test_actual_bf16_forward_scaling_order_and_p3_only(self):
        for dtype in (torch.float32,torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(42);torch.set_num_threads(1)
                wrapper=SharedPrivateLinear(nn.Linear(3,5,bias=False).to(dtype),module_identity='q_proj',
                    shared_rank=2,shared_alpha=2,shared_lr_scale=1,dropout=.3,shared_trainable=False)
                for task in (1,2,3):
                    wrapper.add_private_task(task,private_rank=2,experts_per_task=2,alpha=4,trainable=False)
                    for e in wrapper.task_experts(task):e.B.normal_(std=100 if task<3 else .5)
                wrapper.eval()
                class Text(nn.Module):
                    def __init__(self):super().__init__();self.q_proj=wrapper
                    def forward(self,input_ids,visual_pos_masks=None):
                        return SimpleNamespace(last_hidden_state=self.q_proj(input_ids))
                text=Text().eval()
                def backbone(input_ids,attention_mask=None,**kwargs):
                    return text(input_ids,visual_pos_masks=torch.zeros(1,4,dtype=torch.bool))
                bridge=SimpleNamespace(text=text,targets={'q_proj':wrapper},backbone=backbone,
                    model_inputs=lambda batch:{k:v for k,v in batch.items() if k!='labels'})
                x=torch.randn(1,4,3,dtype=dtype)
                record={'id':'fixed','batch':{'input_ids':x,'attention_mask':torch.ones(1,4,dtype=torch.long),
                                             'labels':torch.tensor([[-100,-100,1,1]])}}
                before=fingerprint(text)
                collector=audit.CaptureP3(bridge,count=3)
                measured=collector.collect(record,verify=True)['q_proj'];positions=record['positions']
                expected=wrapper.task_contribution(x,3)[0,positions].double().norm(dim=-1)
                keys=torch.cat([F.linear(x,e.A)[0,positions] for e in wrapper.task_experts(3)],dim=-1).double().abs()
                self.assertEqual(measured['functional_response'],expected.tolist())
                self.assertEqual(measured['key_abs_per_expert'],keys.tolist())
                self.assertEqual(measured['key_activation'],keys.mean(dim=-1).tolist())
                self.assertEqual(before,fingerprint(text));self.assertEqual(collector.verified,{'q_proj'})
                self.assertEqual(wrapper.active_tasks,(1,2,3))

    def test_statistics_and_behavior_effect_direction(self):
        s=audit.statistics([1,2,3,4,5]);self.assertEqual(s['mean'],3);self.assertEqual(s['median'],3)
        self.assertEqual(s['max'],5);self.assertAlmostEqual(s['p90'],4.6)
        rows=[{'task':2,'teacher_correct':True,'pre_correct':i<3,
               'behavior_group':'retained' if i<3 else 'forgotten','functional_response':float(i),
               'key_activation':float(i),'normalized_response':float(i)} for i in range(6)]
        result=audit.correlation(rows)['functional_response']
        self.assertEqual(result['rank_biserial_forgotten_minus_retained'],1)
        self.assertGreater(result['spearman_teacher_correct'],.8)
