import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
from med_prism.rcwp.core import (Moments,fit_field,phi,output_basis,signed_score,net_penalty,
                                 gold_margin,factors,bank_binding)
from med_prism.rcwp.runtime import (SCHEMA,save_summary,load_summary,validate_summary,
                                    WriteProtection,wrappers)
from med_prism.adapters.shared_private import SharedPrivateLinear


def model_fixture(tasks=2,dtype=torch.float32):
    torch.manual_seed(42)
    m=nn.Sequential(SharedPrivateLinear(nn.Linear(4,5,bias=False,dtype=dtype),
        module_identity="toy.q_proj",shared_rank=2,shared_alpha=2,shared_lr_scale=1,
        dropout=0,shared_trainable=True))
    for t in range(1,tasks+1):
        m[0].add_private_task(t,private_rank=2,experts_per_task=2,alpha=2,trainable=t==tasks)
        for e in m[0].task_experts(t):
            with torch.no_grad(): e.B.normal_()
    return m


def summary(m,task=1):
    return {"schema":SCHEMA,"method_version":"2.1","task_id":task,"nu":1.,
        "fields":{n:{"M":torch.randn(2,3),"mu":torch.zeros(2),"scale":torch.ones(2),
            "binding":bank_binding(w,task)} for n,w in wrappers(m).items()}}


class RCWPTests(unittest.TestCase):
    def test_auc_ties_and_single_class(self):
        from med_prism.rcwp.boundary import harmful_auc
        self.assertEqual(harmful_auc([True,False],[1.,0.]),1.)
        self.assertEqual(harmful_auc([True,False],[0.,1.]),0.)
        self.assertEqual(harmful_auc([True,False],[1.,1.]),.5)
        self.assertIsNone(harmful_auc([True,True],[1.,2.]))

    def test_affine_recovery_and_fold(self):
        torch.manual_seed(1)
        z=torch.randn(4000,3)
        known=torch.randn(2,4)
        c=phi(z,torch.zeros(3),torch.ones(3))@known.T
        f,h=Moments(3,2),Moments(3,2)
        f.add(z[:3000],c[:3000]);h.add(z[3000:],c[3000:])
        K,mu,s,d,_=fit_field(f,h)
        self.assertLess(d["holdout_error"],1e-4)
        self.assertTrue(torch.allclose(phi(z,mu,s)@K.T,c,atol=.02))
        B=torch.randn(7,2);Q,lift,r=output_basis(B)
        self.assertEqual(r,2)
        self.assertTrue(torch.allclose(B@lift@K,Q@K,atol=2e-6))

    def test_rank_deficient_fold(self):
        B=torch.tensor([[1.,2.],[2.,4.],[3.,6.]])
        Q,lift,r=output_basis(B)
        self.assertEqual(r,1)
        self.assertTrue(torch.allclose(B@lift,Q,atol=1e-6))

    def test_positive_negative_equal_norm(self):
        nu=torch.ones(1)
        self.assertEqual(net_penalty(torch.tensor([[2.]]),nu).item(),0.)
        self.assertEqual(net_penalty(torch.tensor([[-2.]]),nu).item(),4.)

    def test_net_sum_before_penalty_multiple_history(self):
        terms=torch.tensor([[[3.],[-4.]],[[-2.],[2.]]])
        scores=terms.sum(0)
        self.assertEqual(net_penalty(scores,torch.ones(2)).item(),2.)

    def test_task1_exact_zero(self):
        m=model_fixture(1);run=WriteProtection(m,1,[])
        run.begin(torch.ones(1,3));m(torch.randn(1,3,4));loss,_=run.finish()
        self.assertEqual(loss.item(),0.)
        self.assertFalse(loss.requires_grad)

    def test_gradient_ownership_and_mask(self):
        m=model_fixture();s=summary(m);run=WriteProtection(m,2,[s])
        x=torch.randn(1,4,4,requires_grad=True)
        # A deliberately live base/shared cannot receive auxiliary gradients.
        m[0].base_layer.weight.requires_grad_(True)
        run.begin(torch.tensor([[1,1,1,0]]));out=m(x);loss,diag=run.finish()
        # Use score to exercise both ownership paths even when penalty is zero.
        a,b=factors(m[0],2,False);ai,bi=factors(m[0],1)
        f=s["fields"]["0"]
        score=signed_score(x,a,b,ai,bi,f["M"],f["mu"],f["scale"])
        score.sum().backward()
        self.assertIsNone(x.grad)
        for name,p in m.named_parameters():
            if "task_0002__" in name: self.assertIsNotNone(p.grad)
            else: self.assertIsNone(p.grad)

    def test_detach_all_historical_metadata(self):
        values=[torch.randn(*shape,requires_grad=True) for shape in
                [(1,3,4),(2,4),(5,2),(2,4),(5,2),(2,3),(2,),(2,)]]
        values[-1]=torch.ones(2,requires_grad=True)
        signed_score(*values).sum().backward()
        for i,v in enumerate(values):
            if i in (1,2): self.assertIsNotNone(v.grad)
            else: self.assertIsNone(v.grad)

    def test_negative_penalty_gradient_only_current_factors(self):
        h=torch.ones(1,2,1,requires_grad=True)
        a=torch.ones(1,1,requires_grad=True)
        b=torch.tensor([[-1.]],requires_grad=True)
        ai=torch.ones(1,1,requires_grad=True)
        bi=torch.ones(1,1,requires_grad=True)
        M=torch.tensor([[0.,1.]],requires_grad=True)
        mu=torch.zeros(1,requires_grad=True);scale=torch.ones(1,requires_grad=True)
        score=signed_score(h,a,b,ai,bi,M,mu,scale).sum().reshape(1,1)
        loss=net_penalty(score,torch.ones(1))
        self.assertEqual(loss.item(),4.)
        loss.backward()
        self.assertGreater(a.grad.abs().sum().item(),0.)
        self.assertGreater(b.grad.abs().sum().item(),0.)
        for t in (h,ai,bi,M,mu,scale):self.assertIsNone(t.grad)

    def test_multiple_history_runtime_matches_manual(self):
        m=model_fixture(3);histories=[summary(m,1),summary(m,2)]
        runtime=WriteProtection(m,3,histories);x=torch.randn(1,4,4)
        runtime.begin(torch.ones(1,4));m(x);actual,_=runtime.finish()
        a,b=factors(m[0],3,False);scores=[]
        for s in histories:
            ai,bi=factors(m[0],s["task_id"]);f=s["fields"]["0"]
            scores.append(signed_score(x,a,b,ai,bi,f["M"],f["mu"],f["scale"]).sum(-1))
        self.assertTrue(torch.equal(actual,net_penalty(torch.stack(scores),torch.ones(2))))

    def test_save_reload_and_hash_mismatch(self):
        m=model_fixture();s=summary(m)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"summary.pt";save_summary(p,s)
            loaded=load_summary(p,wrappers(m))
            self.assertTrue(torch.equal(s["fields"]["0"]["M"],loaded["fields"]["0"]["M"]))
            with self.assertRaises(FileExistsError):save_summary(p,s)
            with torch.no_grad():m[0].task_experts(1)[0].A.add_(.1)
            with self.assertRaisesRegex(ValueError,"hash mismatch"):load_summary(p,wrappers(m))

    def test_schema_and_scaling_mismatch(self):
        m=model_fixture();s=summary(m)
        s["fields"]["0"]["M"]=s["fields"]["0"]["M"].double()
        with self.assertRaises(ValueError):validate_summary(s,wrappers(m))
        s=summary(m)
        m[0].task_experts(1)[0].scaling.add_(1)
        with self.assertRaises(ValueError):validate_summary(s,wrappers(m))

    def test_bf16_fp32_auxiliary(self):
        m=model_fixture(dtype=torch.bfloat16);run=WriteProtection(m,2,[summary(m)])
        run.begin(torch.ones(1,3));out=m(torch.randn(1,3,4,dtype=torch.bfloat16));loss,_=run.finish()
        self.assertEqual(out.dtype,torch.bfloat16)
        self.assertEqual(loss.dtype,torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for p in m.parameters():
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())

    def test_forward_unchanged_off_and_on(self):
        m=model_fixture();x=torch.randn(1,3,4)
        before=m(x)
        run=WriteProtection(m,2,[summary(m)])
        run.begin(torch.ones(1,3));after=m(x);run.finish()
        self.assertTrue(torch.equal(before,after))
        self.assertTrue(torch.equal(before,m(x)))

    def test_training_dropout_rng_not_changed(self):
        m=model_fixture().train();m[0].dropout.p=.05
        x=torch.randn(1,3,4);s=summary(m)
        torch.manual_seed(12);before=m(x);rng=torch.random.get_rng_state()
        run=WriteProtection(m,2,[s]);run.begin(torch.ones(1,3))
        torch.manual_seed(12);after=m(x);run.finish()
        self.assertTrue(torch.equal(before,after))
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))

    def test_reentrant_checkpoint_first_forward(self):
        from torch.utils.checkpoint import checkpoint
        m=model_fixture();x=torch.randn(1,3,4,requires_grad=True)
        run=WriteProtection(m,2,[summary(m)])
        run.begin(torch.ones(1,3))
        out=checkpoint(m,x,use_reentrant=True)
        loss,_=run.finish()
        (out.sum()+loss).backward()
        self.assertFalse(run.handles)

    def test_margin_causal_shift(self):
        logits=torch.tensor([[[0.,2.,1.],[3.,1.,0.],[8.,0.,0.]]],requires_grad=True)
        labels=torch.tensor([[-100,1,-100]])
        actual=gold_margin(logits,labels)
        expected=2-torch.logsumexp(torch.tensor([0.,1.]),0)
        self.assertTrue(torch.equal(actual,expected))
        actual.backward()
        self.assertEqual(logits.grad[:,1:].abs().sum().item(),0.)

    def test_boundary_all_tokens_single_backward(self):
        from types import SimpleNamespace
        from med_prism.rcwp.boundary import BoundaryField
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj=model_fixture(1)[0]
                self.embed=nn.Embedding(5,4)
            def forward(self,input_ids,**kwargs):
                # Earlier prompt tokens affect a later answer margin.
                return SimpleNamespace(logits=self.proj(self.embed(input_ids)).cumsum(1))
        m=Tiny().eval().requires_grad_(False)
        field=BoundaryField(m,1)
        batch={"input_ids":torch.tensor([[1,2,3,0]]),"attention_mask":torch.tensor([[1,1,1,0]]),
               "labels":torch.tensor([[-100,-100,2,-100]])}
        field.observe(batch,"fit");field.observe(batch,"holdout")
        s=field.finish({}, {})
        self.assertEqual(field.fields["proj"]["fit"].n,3)
        self.assertTrue(torch.isfinite(s["fields"]["proj"]["M"]).all())
        self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_plugin_adds_only_auxiliary_once(self):
        import os
        import runpy
        # Same compatibility entrypoint used by the unchanged official launcher.
        runpy.run_path(str(Path(__file__).resolve().parents[2]/"scripts/phase3/callback_compat.py"))
        from unittest.mock import patch
        from med_prism.rcwp.plugin import RCWPTrainer,get_trainer
        from med_prism.swift_plugins.shared_private_trainer import MedPrismSharedPrivateTrainer
        m=model_fixture();s=summary(m)
        trainer=object.__new__(RCWPTrainer)
        trainer.rcwp=WriteProtection(m,2,[s]);trainer.rcwp_weight=.1
        def baseline(obj,model,inputs,**kwargs):
            outputs=model(inputs["x"])
            obj.last_loss={"raw_key_loss":.2,"shared_drift_raw_loss":.3}
            return (outputs.sum(),outputs) if kwargs["return_outputs"] else outputs.sum()
        inputs={"x":torch.randn(1,3,4),"attention_mask":torch.ones(1,3)}
        expected=m(inputs["x"]).sum()
        with patch.object(MedPrismSharedPrivateTrainer,"compute_loss",baseline):
            actual,out=trainer.compute_loss(m,inputs,return_outputs=True)
        self.assertAlmostEqual(float(actual-expected),.1*trainer.last_loss["rcwp_loss"],places=4)
        self.assertTrue(torch.equal(out,m(inputs["x"])))

    def test_full_component_checkpoint_roundtrip(self):
        import io
        m=model_fixture();s=summary(m);x=torch.randn(1,3,4)
        before=m(x);stream=io.BytesIO();torch.save(m.state_dict(),stream);stream.seek(0)
        clone=model_fixture();clone.load_state_dict(torch.load(stream,weights_only=True))
        self.assertTrue(torch.equal(before,clone(x)))
        validate_summary(s,wrappers(clone))

    def test_boundary_groups_disjoint(self):
        from med_prism.rcwp.data import encode_boundary_rows
        class Template:
            def encode(self,row):return row
            def data_collator(self,rows):
                group=int(rows[0]["images"][0].split("image")[-1])
                return {"input_ids":torch.tensor([[group,1]]),"labels":torch.tensor([[-100,1]])}
        rows=[{"id":str(i),"messages":[],"images":[f"/image{i//2}"]} for i in range(24)]
        records,_=encode_boundary_rows(rows,Template(),4,4)
        fit={int(r["batch"]["input_ids"][0,0]) for r in records if r["split"]=="fit"}
        holdout={int(r["batch"]["input_ids"][0,0]) for r in records if r["split"]=="holdout"}
        self.assertFalse(fit&holdout)


if __name__=="__main__":unittest.main()
