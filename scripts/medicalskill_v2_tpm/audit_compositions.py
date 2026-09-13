"""Frozen A-E composition diagnostics on grouped dev only. No checkpoint writes."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_retention as retention
from med_prism.transport.checkpoint import legacy_state, apply_tensors, sha256_file
from med_prism.transport.cli import DEFAULT_NO_GEO, TASKS
from med_prism.transport.banks import fingerprint, check_invariants
from med_prism.adapters.injection import iter_shared_private_wrappers
from med_prism.transport.diagnostics import write_json


def compositions(pre, teacher, post):
    # A and D differ ONLY in activity of P3; C and B likewise.
    old_shared = dict(pre)
    old_shared.update(teacher)
    return {'A': (old_shared, [1,2]), 'B': (pre, [1,2,3]),
            'C': (pre, [1,2]), 'D': (old_shared, [1,2,3]), 'E': (post, [1,2,3])}


def logit_metrics(reference, actual):
    a, b = reference.double(), actual.double()
    la, lb = F.log_softmax(a,dim=-1), F.log_softmax(b,dim=-1)
    return {'kl_teacher_to_variant': float((la.exp()*(la-lb)).sum().clamp_min(0)),
            'logits_mse':float((a-b).square().mean()),
            'logits_cosine':float(F.cosine_similarity(a,b,dim=0)),
            'centered_logits_mse':float(((a-a.mean())-(b-b.mean())).square().mean()),
            'teacher_top1_agrees':int(a.argmax()==b.argmax())}


def interaction_metrics(a,b,c,d):
    # Center logits to remove irrelevant scalar shifts; no softmax linearity assumption.
    a,b,c,d=[x.double()-x.double().mean() for x in (a,b,c,d)]
    shared=c-a; private=d-a; full=b-a; interaction=b-c-d+a
    return {'interaction_squared_norm':float(interaction.square().sum()),
            'full_drift_squared_norm':float(full.square().sum()),
            'shared_only_squared_norm':float(shared.square().sum()),
            'private_only_squared_norm':float(private.square().sum()),
            'shared_private_delta_cosine':float(F.cosine_similarity(shared,private,dim=0))}


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--pilot-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--task',type=int,choices=(1,2),required=True)
    args=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(42)
    out=args.output_root.resolve()
    if out.exists():raise FileExistsError(out)
    states, pair_audit=retention.audit_pair(args.run_root.resolve())
    teacher=legacy_state(DEFAULT_NO_GEO,2)
    req=json.loads((args.run_root/'run_request.json').read_text())
    if teacher.origin!=req['teacher']:raise ValueError('Wrong Task2 teacher')
    pre,post=states['pre_tpm'],states['post_tpm']
    pre_tensors,teacher_tensors,post_tensors=pre.tensors(),teacher.tensors(),post.tensors()
    for key,value in teacher_tensors.items():
        if '.experts.' in key and not torch.equal(value,pre_tensors[key]):
            raise ValueError('Historical pre banks differ from Task2 teacher')
    variants=compositions(pre_tensors,teacher_tensors,post_tensors)
    data=args.pilot_root.resolve()/'data'/TASKS[args.task]/'development.jsonl'
    rows=retention.evaluator.read_jsonl(data,0)
    if len(rows)!=256:raise ValueError('Expected existing complete 256-row grouped dev')
    manifests={
      'A':{'shared':teacher.shared_manifest,'private':teacher.private_manifests},
      'B':{'shared':pre.shared_manifest,'private':pre.private_manifests},
      'C':{'shared':pre.shared_manifest,'private':pre.private_manifests[:2]},
      'D':{'shared':teacher.shared_manifest,'private':pre.private_manifests},
      'E':{'shared':post.shared_manifest,'private':post.private_manifests}}
    files=[data,args.pilot_root/'data/pilot_data_manifest.json',Path(__file__)]
    for state in (teacher,pre,post):
        if Path(state.origin).is_file():files.append(Path(state.origin))
        for path in state.components():
            files.extend([Path(path),Path(path).parent/json.loads(Path(path).read_text())['weights_file']])
    for name in ('evaluate_formal_v1_2.py','evaluate_formal_v1_2_cell.py'):
        files.append(retention.LEGACY/name)
    hashes={str(path):sha256_file(path) for path in files}
    provenance={'task':args.task,'data':str(data),'samples':len(rows),
        'ordered_ids':[row['id'] for row in rows],'source_sha256':hashes,'pair_audit':pair_audit,
        'compositions':manifests,'base':str(retention.evaluator.SNAPSHOT),
        'batch_size':4,'seed':42,'generation':retention.contract.GENERATION_CONFIG,
        'image_preprocessing':retention.contract.IMAGE_PREPROCESSING,
        'KL':'KL(teacher||variant), full vocabulary, raw first next-token logits at identical generation prompt',
        'interaction':'centered logits B-C-D+A; A-D form shared(S2/S3) x P3(off/on) factorial',
        'dev_caveat':'Grouped held out for future pilot, but present in original no-geo training; not formal test',
        'no_training_or_fitting_or_checkpoint_write':True}
    out.mkdir(parents=True)
    write_json(out/'provenance.json',provenance)
    model,processor,_=retention.load_evaluation_state(pre)
    baseline=fingerprint(model)
    params=dict(model.named_parameters())
    logits={};details_by_variant={}; summaries={}
    started=time.time()
    for variant,(tensors,active) in variants.items():
        apply_tensors(model,tensors)
        retention.evaluator.set_active(model,active)
        for key,expected in tensors.items():
            actual=params[key].detach().cpu()
            if actual.dtype!=expected.dtype or not torch.equal(actual,expected):
                raise RuntimeError('Composition tensor mismatch: '+key)
        actual_activity={name:list(w.active_tasks) for name,w in iter_shared_private_wrappers(model)}
        if not actual_activity or any(v!=active for v in actual_activity.values()):
            raise RuntimeError('Active bank mismatch')
        write_json(out/f'{variant}.loaded.json',{'status':'PASS',**manifests[variant],
            'active_banks':active,'wrapper_count':len(actual_activity),'tensor_count_verified':len(tensors),
            'dormant_P3_allocated_but_inactive':variant in ('A','C')})
        captured=[]
        def capture(module, positional, kwargs, output):
            ids=kwargs.get('input_ids')
            if ids is None and positional:ids=positional[0]
            # Existing batched_generate does one multi-token prefill, then cached 1-token steps.
            if ids is not None and ids.shape[1]>1:
                captured.append(output.logits[:,-1,:].detach().float().cpu())
        handle=model.register_forward_hook(capture,with_kwargs=True)
        predictions=[]
        torch.manual_seed(42)
        try:
            for offset in range(0,len(rows),16):
                predictions.extend(retention.cell.batched_generate(model,processor,rows[offset:offset+16],args.task,4))
                print(f'Task{args.task} variant {variant} {min(offset+16,len(rows))}/{len(rows)} elapsed={time.time()-started:.1f}s',flush=True)
        finally:handle.remove()
        values=torch.cat(captured)
        if len(predictions)!=len(rows) or values.shape[0]!=len(rows) or not torch.isfinite(values).all():
            raise RuntimeError('Generation/prefill-logit count or finiteness mismatch')
        logits[variant]=values
        metric,details=retention.evaluator.evaluate_rows(args.task,rows,predictions)
        if metric['status']!='PASS':raise RuntimeError('Original evaluator rejected output')
        metrics=[]
        for i,detail in enumerate(details):
            deviation=logit_metrics(logits['A'][i],values[i]);metrics.append(deviation)
            detail.update(deviation)
        metric.update({key:sum(r[key] for r in metrics)/len(metrics) for key in metrics[0]})
        metric.update(variant=variant,task=args.task,active_banks=active)
        summaries[variant]=metric;details_by_variant[variant]=details
        retention.cell.atomic_write(out/f'{variant}.predictions.jsonl',''.join(json.dumps(x)+'\n' for x in details))
        write_json(out/f'{variant}.summary.json',metric)
        print('RESULT '+json.dumps(metric),flush=True)
    interactions=[]
    for i,row in enumerate(rows):
        item={'id':row['id'],**interaction_metrics(*[logits[v][i] for v in 'ABCD'])}
        for variant in 'ABCDE':item[variant+'_correct']=details_by_variant[variant][i]['correct']
        interactions.append(item)
    totals={k:sum(r[k] for r in interactions) for k in ('interaction_squared_norm','full_drift_squared_norm','shared_only_squared_norm','private_only_squared_norm')}
    totals['relative_interaction_l2']=(totals['interaction_squared_norm']/totals['full_drift_squared_norm'])**.5
    totals['accuracy_interaction_B_minus_C_minus_D_plus_A']=summaries['B']['accuracy']-summaries['C']['accuracy']-summaries['D']['accuracy']+summaries['A']['accuracy']
    retention.cell.atomic_write(out/'interactions.jsonl',''.join(json.dumps(r)+'\n' for r in interactions))
    apply_tensors(model,pre_tensors);retention.evaluator.set_active(model,[1,2,3])
    invariants=check_invariants(baseline,fingerprint(model),set())
    if any(sha256_file(path)!=digest for path,digest in hashes.items()):raise RuntimeError('Input file changed')
    write_json(out/'summary.json',{'status':'PASS','task':args.task,'variants':summaries,
        'interaction':totals,'invariants':invariants,'source_hashes_unchanged':True,'elapsed_seconds':time.time()-started})
    print('COMPLETE Task'+str(args.task),flush=True)


if __name__=='__main__':main()
