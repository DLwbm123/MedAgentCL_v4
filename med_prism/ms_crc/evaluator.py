"""Existing MedicalSkill generation/parser plus full teacher-forced margin/CE."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from .data import encode
from .search import IDENTITY

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/medicalskill_v1_2_med_prism'))
import evaluate_formal_v1_2 as legacy
import evaluate_formal_v1_2_cell as cell
import evaluation_contract_v1_2 as contract


def decoded_correct(task,detail):
    if task in (1,2):return bool(detail['correct'])
    if task==3:return detail['all_fp']==0 and detail['all_fn']==0
    raise ValueError('Gate only supports Tasks1..3')


def inputs(batch,device):
    from med_prism.transport.hooks import tree_to
    return {k:tree_to(v,device) for k,v in batch.items() if k not in
        ('labels','loss_scale','text_position_ids','channel','compute_loss_func')}


@torch.no_grad()
def token_scores(model,batch):
    device=next(model.parameters()).device
    output=model(**inputs(batch,device),use_cache=False,return_dict=True)
    labels=batch['labels'][:,1:].to(device);valid=labels!=-100
    logits=output.logits[:,:-1][valid].float();gold=labels[valid]
    if len(gold)==0:raise ValueError('Missing gold answer tokens')
    target=logits.gather(1,gold[:,None]).squeeze(1)
    other=logits.scatter(1,gold[:,None],-torch.inf).logsumexp(1)
    stats={'margin':float((target-other).mean()),'nll_sum':float(F.cross_entropy(logits,gold,reduction='sum')),
        'answer_tokens':len(gold)}
    if not all(torch.isfinite(torch.tensor(v)) for v in stats.values()):raise ValueError('Nonfinite scores')
    return stats,logits


class Evaluator:
    def __init__(self,model,template,composition):
        self.model=model;self.template=template;self.composition=composition;self.processor=template.processor
        self.processor.image_processor.min_pixels=200704;self.processor.image_processor.max_pixels=200704

    def records(self,rows):return [{'row':row,'batch':encode(row,self.template)} for row in rows]

    @torch.no_grad()
    def score(self,records,g=IDENTITY,active=(1,2,3),task=3,kl=False):
        rows=[r['row'] for r in records]
        torch.manual_seed(42)
        with self.composition.use(g,active):
            predictions=cell.batched_generate(self.model,self.processor,rows,task,4)
            metric,details=legacy.evaluate_rows(task,rows,predictions)
            if metric['status']!='PASS':raise RuntimeError('Existing evaluator failure')
            result=[]
            for record,detail in zip(records,details):
                scalar,actual=token_scores(self.model,record['batch'])
                if kl:
                    # H uses SAME S3, NOT the true Task2/S2 boundary teacher.
                    with self.composition.use(IDENTITY,(1,2)):
                        _,reference=token_scores(self.model,record['batch'])
                    a=reference.double().log_softmax(-1);b=actual.double().log_softmax(-1)
                    scalar['kl_H_same_S3_to_candidate']=float((a.exp()*(a-b)).sum(-1).mean().clamp_min(0))
                    del reference,a,b
                result.append({'sample_id':str(record['row']['id']),'prediction':detail['prediction'],
                    'correct':decoded_correct(task,detail),**scalar,
                    **{k:detail[k] for k in ('all_tp','all_fp','all_fn') if k in detail}})
                del actual
        return result,metric


def equivalent(a,b,tolerance):
    if len(a)!=len(b):raise RuntimeError('Reload sample count differs')
    for x,y in zip(a,b):
        for key in ('sample_id','prediction','correct','answer_tokens'):
            if x[key]!=y[key]:raise RuntimeError('Search/deployment mismatch: '+key)
        for key in ('margin','nll_sum'):
            if abs(x[key]-y[key])>tolerance:raise RuntimeError('Search/deployment numerical mismatch: '+key)
    return True
