import unittest
from types import SimpleNamespace
import torch
from torch import nn
from scripts.medicalskill_v2_1_rcwp.audit_derivatives import derivatives,observations,aggregate,metrics
from med_prism.adapters.shared_private import SharedPrivateLinear
from med_prism.rcwp.runtime import wrappers


class DerivativeAuditTests(unittest.TestCase):
    def test_projection_and_offspan_quantization(self):
        Q=torch.tensor([[1.],[0.]])
        g=torch.tensor([[2.,7.]])
        inside=torch.tensor([[3.,0.]])
        a=derivatives(g,Q,g,inside)
        self.assertEqual(a['d_exact'],a['d_Q'])
        self.assertAlmostEqual(a['coverage_g'],4/53)
        rounded=torch.tensor([[3.,1.]])
        b=derivatives(g,Q,g,rounded)
        self.assertEqual(b['d_exact'],13.)
        self.assertEqual(b['d_Q'],6.)

    def test_metric_orientation_and_ties(self):
        rows=[{'actual':-1.,'prediction':-2.},{'actual':1.,'prediction':2.}]
        m=metrics(rows,'prediction')
        self.assertEqual(m['harmful_auroc'],1.)
        self.assertAlmostEqual(m['spearman'],1.)

    def test_real_hooks_fd_units_and_no_parameter_mutation(self):
        torch.manual_seed(3)
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__();self.embed=nn.Embedding(4,4)
                self.model=nn.Module();self.model.layers=nn.ModuleList([nn.Module()])
                attn=nn.Module();self.model.layers[0].self_attn=attn
                for name in ('q_proj','v_proj'):
                    w=SharedPrivateLinear(nn.Linear(4,4,bias=False),module_identity=name,
                        shared_rank=2,shared_alpha=2,shared_lr_scale=1,dropout=0,shared_trainable=False)
                    w.add_private_task(1,private_rank=2,experts_per_task=2,alpha=2,trainable=False)
                    for e in w.task_experts(1):e.B.normal_()
                    setattr(attn,name,w)
            def forward(self,input_ids,**kwargs):
                h=self.embed(input_ids);a=self.model.layers[0].self_attn
                return SimpleNamespace(logits=(a.q_proj(h)+a.v_proj(h)).cumsum(1))
        model=Tiny().eval().requires_grad_(False);modules=wrappers(model)
        summary={'task_id':1,'fields':{n:{'M':torch.randn(2,3),'mu':torch.zeros(2),'scale':torch.ones(2)} for n in modules}}
        original={n:t.clone() for n,t in model.state_dict().items()}
        batch={'input_ids':torch.tensor([[0,1,2]]),'labels':torch.tensor([[-100,1,2]]),'attention_mask':torch.ones(1,3,dtype=torch.long)}
        rows,fd=observations(model,batch,summary,modules,0)
        self.assertEqual(len(rows),12);self.assertEqual(len(fd),6)
        for r in fd:
            if r['amplitude']==.01:self.assertAlmostEqual(r['fd_slope'],r['d_exact_direction'],delta=.005)
        result,coverage,finite=aggregate(rows,fd)
        self.assertEqual(set(result),{'0.01','0.05','0.1'})
        self.assertTrue(all(torch.equal(t,model.state_dict()[n]) for n,t in original.items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        model.to(torch.bfloat16)
        bf16_before={n:t.clone() for n,t in model.state_dict().items()}
        bf16_rows,_=observations(model,batch,summary,wrappers(model),0)
        self.assertEqual(len(bf16_rows),12)
        self.assertTrue(all(torch.equal(t,model.state_dict()[n]) for n,t in bf16_before.items()))


if __name__=='__main__':unittest.main()
