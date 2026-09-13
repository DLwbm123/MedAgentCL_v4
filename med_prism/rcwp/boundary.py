"""Single scalar gold-margin backward for all wrappers; streaming rank moments."""
import math
import torch
from .core import factors, output_basis, Moments, fit_field, phi, gold_margin, bank_binding
from .runtime import wrappers, SCHEMA


def harmful_auc(harmful, scores):
    """Mann-Whitney AUROC, average ranks for ties; no sklearn dependency."""
    from scipy.stats import rankdata
    positives=sum(harmful);negatives=len(harmful)-positives
    if not positives or not negatives:return None
    ranks=rankdata(scores,method="average")
    return float((sum(r for r,y in zip(ranks,harmful) if y)-positives*(positives+1)/2)/(positives*negatives))


def model_inputs(batch, device):
    from med_prism.transport.hooks import tree_to
    removed = {"labels","loss_scale","text_position_ids","channel","compute_loss_func"}
    return {k:tree_to(v,device) for k,v in batch.items() if k not in removed}


def forward_margin(model,batch):
    device = next(model.parameters()).device
    out = model(**model_inputs(batch,device),use_cache=False,return_dict=True)
    return gold_margin(out.logits,batch["labels"].to(device))


class BoundaryField:
    def __init__(self, model, task):
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("Boundary requires eval mode and all model parameters frozen")
        self.model,self.task,self.modules = model,task,wrappers(model)
        self.fields = {}
        for name,m in self.modules.items():
            A,B = factors(m,task)
            Q,lift,rank = output_basis(B)
            self.fields[name] = dict(A=A,B=B,Q=Q,lift=lift,rank=rank,
                fit=Moments(len(A),rank),holdout=Moments(len(A),rank))
        self.rms_sum = 0.
        self.rms_n = 0

    def observe(self,batch,partition):
        if partition not in {"fit","holdout"}:
            raise ValueError(partition)
        handles,seen = [],set()
        own = []
        mask = batch["attention_mask"].bool()
        if mask.shape[0] != 1:
            raise ValueError("Boundary first version uses batch size one")
        for name,m in self.modules.items():
            def hook(module,args,out,name=name):
                f = self.fields[name]
                h = args[0].detach()
                if h.shape[:2] != mask.shape:
                    raise ValueError("Boundary token mask mismatch")
                valid = mask.to(h.device)
                z = h.float()[valid] @ f["A"].T
                # Enabling gradients on outputs leaves all model parameters frozen.
                out.requires_grad_(True)
                def sensitivity(grad):
                    if name in seen:
                        raise RuntimeError("Repeated boundary gradient")
                    seen.add(name)
                    g = grad.detach().float()[valid]
                    c = g @ f["Q"]
                    f[partition].add(z,c)
                    own.append(float((c * (z @ (f["Q"].T @ f["B"]).T)).sum()))
                out.register_hook(sensitivity)
            handles.append(m.register_forward_hook(hook))
        try:
            with torch.enable_grad():
                margin = forward_margin(self.model,batch)
                margin.backward()  # Exactly one scalar backward; not per expert.
            if seen != set(self.modules):
                raise RuntimeError("Missing wrapper sensitivity")
            if any(p.grad is not None for p in self.model.parameters()):
                raise RuntimeError("Boundary unexpectedly populated parameter gradients")
            if partition == "fit":
                self.rms_sum += sum(own)**2
                self.rms_n += 1
            return float(margin.detach())
        finally:
            for h in handles: h.remove()

    def finish(self,source,config):
        result = {"schema":SCHEMA,"method_version":"2.1","method_name":"Med-PRISM-v2.1-RCWP",
            "task_id":self.task,"nu":max(math.sqrt(self.rms_sum/self.rms_n),1.),
            "score_rms":math.sqrt(self.rms_sum/self.rms_n),"source":source,"config":config,"fields":{}}
        self.constant = {}
        for name,f in self.fields.items():
            K,mu,scale,diag,constant = fit_field(f["fit"],f["holdout"])
            lift = f["lift"].cpu()
            self.constant[name] = lift@constant
            result["fields"][name] = {"M":lift@K,"mu":mu,"scale":scale,
                "effective_B_rank":f["rank"],"diagnostics":diag,
                "binding":bank_binding(self.modules[name],self.task)}
        return result


def field_quality(model,records,summary,constants):
    """Predeclared small factor-write perturbations, NEVER fitting on holdout.

    Inject +/- eps B_eff C A h at one deep q/v wrapper, C fixed seeded random
    unit-Frobenius matrix. A/B tensors never changed. Both measured signs and all
    four predictors use identical holdout inputs. Signed scores are first order;
    labels are actual perturbed-minus-unperturbed scalar gold margins.
    """
    from scipy.stats import spearmanr
    modules = wrappers(model)
    names = list(modules)
    chosen = names[-2:]
    observations = []
    for record in records:
        batch = record["batch"]
        with torch.no_grad():
            before = float(forward_margin(model,batch))
        for name in chosen:
            m = modules[name]
            A,B = factors(m,summary["task_id"])
            f = summary["fields"][name]
            M,mu,scale = [f[k].to(A.device) for k in ("M","mu","scale")]
            generator = torch.Generator().manual_seed(2100+names.index(name))
            C = torch.randn(len(A),len(A),generator=generator).to(A.device)
            C = C/C.norm()
            for sign in (-1,1):
                predicted = {}
                def inject(module,args,out):
                    h = args[0].detach().float()
                    z = h@A.T
                    delta = (z@C.T@B.T)*(.01*sign)
                    valid = batch["attention_mask"].bool().to(h.device)
                    delta = delta.masked_fill(~valid[...,None],0)
                    # Measure actual BF16-deployed intervention, not an uncast ideal.
                    edited = out+delta.to(out.dtype)
                    actual = edited.float()-out.float()
                    feature = phi(z,mu,scale)
                    for key,field in (("conditional",M),("constant",constants[name].to(A.device))):
                        predicted[key] = float(((feature@field.T@B.T)*actual).sum())
                    response = z@B.T
                    predicted["srg_like"] = float((response*actual).sum())
                    # Magnitude has no beneficial direction: higher means more harmful.
                    predicted["magnitude"] = -float(actual.norm())
                    return edited
                handle = m.register_forward_hook(inject)
                try:
                    with torch.no_grad(): after = float(forward_margin(model,batch))
                finally: handle.remove()
                observations.append({"actual":after-before,**predicted})
    metrics = {}
    actual = [x["actual"] for x in observations]
    harmful = [x<0 for x in actual]
    for key in ("conditional","constant","srg_like","magnitude"):
        score = [x[key] for x in observations]
        corr = float(spearmanr(score,actual).statistic)
        metrics[key] = {"sign_accuracy":sum((a<0)==(b<0) for a,b in zip(score,actual))/len(actual),
            "spearman":corr if math.isfinite(corr) else None,
            "harmful_auroc":harmful_auc(harmful,[-s for s in score])}
    auc = metrics["conditional"]["harmful_auroc"]
    baseline = [metrics[k]["harmful_auroc"] for k in ("constant","srg_like")]
    adequate = len(records)>=64
    go = adequate and auc is not None and auc>=.65 and all(v is not None and auc>v for v in baseline)
    return {"status":"GO" if go else ("INCONCLUSIVE_SMALL_SMOKE" if not adequate else "NO_GO"),
        "holdout_samples":len(records),"perturbations":len(observations),"metrics":metrics,
        "protocol":"last/deep q and v; fixed seeded rank mix; +/-0.01; actual BF16 injection; no weights changed",
        "threshold":.65,"strictly_beats_constant_and_srg":auc is not None and all(v is not None and auc>v for v in baseline),
        "limitation":"Preset perturbation family only; threshold is a research screen, not a retention guarantee"}
