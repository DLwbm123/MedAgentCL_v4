"""Read-only P3 functional activation audit; detached execution, self-contained report."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, mannwhitneyu

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_retention as retention
from med_prism.transport.checkpoint import read_state, sha256_file
from med_prism.transport.calibration import select_tokens, stable_id
from med_prism.transport.hooks import QwenBlockBridge, cleanup_hooks
from med_prism.transport.banks import fingerprint, check_invariants
from med_prism.transport.diagnostics import write_json
from med_prism.transport.cli import load_calibration_model

METRICS=('key_activation','functional_response','normalized_response')


def statistics(values):
    a=np.asarray(values,dtype=np.float64)
    if not len(a):return {'count':0,'mean':None,'median':None,'p90':None,'p95':None,'max':None}
    if not np.isfinite(a).all():raise ValueError('Nonfinite metric')
    return {'count':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
            'p90':float(np.quantile(a,.9)),'p95':float(np.quantile(a,.95)),'max':float(a.max())}


class CaptureP3:
    def __init__(self,bridge,count=8,seed=42):
        self.bridge=bridge;self.count=count;self.seed=seed;self.suspended=False
        self.verified=set()

    @torch.no_grad()
    def collect(self,record,verify=False):
        bridge=self.bridge;batch=record['batch'];pending={};finished={}
        def select(module,args,kwargs):
            visual=kwargs.get('visual_pos_masks')
            positions,selection=select_tokens(batch['attention_mask'][0],batch['labels'][0],
                visual[0] if visual is not None else None,count=self.count,
                seed=self.seed+int(hashlib.sha256(record['id'].encode()).hexdigest()[:8],16))
            record['positions']=positions;record['selection']=selection
        def expert_hook(name,index,wrapper):
            def hook(expert,args,output):
                if self.suspended:return
                if index!=len(pending.setdefault(name,{'keys':[],'sum':None})['keys']):
                    raise RuntimeError('Expert order differs from original task_contribution')
                x,dropout=args
                if dropout.training:raise RuntimeError('Audit requires eval dropout')
                pos=record['positions'].to(x.device)
                # Same full input, dtype and F.linear as Rank1Expert.forward.
                key=F.linear(dropout(x),expert.A)[0,pos].detach()
                effective=output[0,pos].detach()  # Actual B projection AND scaling, not recomputed.
                state=pending[name];state['keys'].append(key)
                if state['sum'] is None:state['sum']=torch.zeros_like(effective)
                state['sum']=state['sum']+effective  # Preserve forward's BF16 accumulation order.
                state['input_norm']=x[0,pos].double().norm(dim=-1)
            return hook
        def wrapper_hook(name):
            def hook(wrapper,args,output):
                state=pending[name]
                if len(state['keys'])!=len(wrapper.task_experts(3)):raise RuntimeError('Incomplete P3')
                pos=record['positions'].to(args[0].device)
                if verify:
                    self.suspended=True
                    try:expected=wrapper.task_contribution(args[0],3)[0,pos]
                    finally:self.suspended=False
                    if not torch.equal(state['sum'],expected):raise RuntimeError('P3 forward not bitwise equal')
                    self.verified.add(name)
                keys=torch.cat(state['keys'],dim=-1).double().abs()
                response=state['sum'].double().norm(dim=-1)
                normalized=response/(state['input_norm']+1e-12)
                if not all(torch.isfinite(x).all() for x in (keys,response,normalized)):raise RuntimeError('Nonfinite response')
                finished[name]={'key_activation':keys.mean(dim=-1).cpu().tolist(),
                    'key_abs_per_expert':keys.cpu().tolist(),
                    'functional_response':response.cpu().tolist(),
                    'normalized_response':normalized.cpu().tolist()}
                del pending[name]
            return hook
        with cleanup_hooks() as handles:
            handles.append(bridge.text.register_forward_pre_hook(select,with_kwargs=True))
            for name,wrapper in bridge.targets.items():
                if 3 not in wrapper.active_tasks:raise RuntimeError('P3 is inactive')
                for index,expert in enumerate(wrapper.task_experts(3)):
                    handles.append(expert.register_forward_hook(expert_hook(name,index,wrapper)))
                handles.append(wrapper.register_forward_hook(wrapper_hook(name)))
            output=bridge.backbone(**bridge.model_inputs(batch),use_cache=False,return_dict=True)
            if not torch.isfinite(output.last_hidden_state).all():raise RuntimeError('Nonfinite hidden')
        if set(finished)!=set(bridge.targets):raise RuntimeError('Missing q/v captures')
        return finished


def correlation(samples):
    task2=[r for r in samples if r['task']==2]
    groups={g:[r for r in task2 if r['behavior_group']==g] for g in sorted({r['behavior_group'] for r in task2})}
    result={'groups':{g:{k:statistics([r[k] for r in rows]) for k in METRICS} for g,rows in groups.items()}}
    correct=[r for r in task2 if r['teacher_correct']]
    result['teacher_correct_n']=len(correct)
    for metric in ('key_activation','functional_response'):
        forgotten=[r[metric] for r in correct if not r['pre_correct']]
        retained=[r[metric] for r in correct if r['pre_correct']]
        values=[r[metric] for r in correct];labels=[int(not r['pre_correct']) for r in correct]
        rho,pvalue=spearmanr(values,labels) if len(set(labels))>1 and len(set(values))>1 else (np.nan,np.nan)
        effect=None
        if forgotten and retained:
            u=mannwhitneyu(forgotten,retained,alternative='two-sided').statistic
            effect=float(2*u/(len(forgotten)*len(retained))-1)
        result[metric]={'forgotten':statistics(forgotten),'retained':statistics(retained),
            'spearman_teacher_correct':float(rho) if np.isfinite(rho) else None,
            'spearman_pvalue_exploratory':float(pvalue) if np.isfinite(pvalue) else None,
            'rank_biserial_forgotten_minus_retained':effect}
    return result


def render_report(summary,out):
    lines=['# P3 functional activation audit','', 'Read-only Task3 pre-TPM, P1/P2/P3 active. All original tensors and inputs unchanged.',
      '', '## Definition and limits','',
      'Teacher-forced forward uses the original train-mode Qwen template, max_length1024, 8 selected tokens/sample, seed42 and native visual masks. No new generation or labels are fitted. Accuracy groups reuse composition A/B correctness.',
      'Key activation is the mean of the 16 actual |A_e x| values per selected token. Raw expert absolute values are in per_bank.jsonl. Functional response is the L2 norm of the actual scaled expert outputs summed in original BF16 order; it is not a sum of expert norms or a merged FP32 approximation.',
      'Normalized response divides by the same projection input norm plus1e-12; it is supplementary. Summary distributions pool selected token-bank observations, while behavioral statistics use per-sample means across the same72 banks/tokens.',
      'These grouped dev rows came from original training data: diagnostic, not an unseen benchmark. Teacher-forced token responses and prior free-generation forgetting measure different aspects; associations are descriptive, not causal proof.',
      '', '## Task distributions','', '|Task|Metric|Mean|Median|P90|P95|Max|','|---|---|---:|---:|---:|---:|---:|']
    for task in ('1','2','3'):
        for metric in METRICS:
            s=summary['tasks'][task]['all'][metric]
            lines.append(f"|{task}|{metric}|"+'|'.join(f'{s[k]:.6g}' for k in ('mean','median','p90','p95','max'))+'|')
    lines+=['','## Old-task / Task3 mean response ratios','']
    for task in ('1','2'):
        ratios=summary['task3_ratios'][task]
        lines.append(f"- Task{task}/Task3: key={ratios['key_activation']:.4f}, effective response={ratios['functional_response']:.4f}. These ratios quantify separation; no universal functional-isolation threshold is assumed.")
    lines+=['','## Projection / depth','', '|Task|Group|Key mean|Response mean|Response P95|','|---|---|---:|---:|---:|']
    for task in ('1','2','3'):
        for group,stats in summary['tasks'][task]['breakdown'].items():
            lines.append(f"|{task}|{group}|{stats['key_activation']['mean']:.6g}|{stats['functional_response']['mean']:.6g}|{stats['functional_response']['p95']:.6g}|")
    lines+=['','Compare q/v with care: q output width4096 and v width1024 differ; raw norms are not width-normalized. Largest response need not be the most causally important layer.','', '## Task2 behavior groups','',
            '|Group|N|Key mean|Key median|Response mean|Response median|','|---|---:|---:|---:|---:|---:|']
    corr=summary['behavioral_correlation']
    for group,stats in corr['groups'].items():
        k=stats['key_activation'];r=stats['functional_response']
        lines.append(f"|{group}|{k['count']}|{k['mean']:.6g}|{k['median']:.6g}|{r['mean']:.6g}|{r['median']:.6g}|")
    for metric in ('key_activation','functional_response'):
        s=corr[metric]
        lines+=['',f"{metric}: teacher-correct-only Spearman={s['spearman_teacher_correct']}, exploratory p={s['spearman_pvalue_exploratory']}; rank-biserial (forgotten minus retained)={s['rank_biserial_forgotten_minus_retained']}."]
    s=corr['functional_response'];a=s['forgotten']['mean'];b=s['retained']['mean']
    lines+=['','## Interpretation','',f"Forgotten mean response is {'higher' if a>b else 'not higher'} than retained ({a:.6g} vs {b:.6g}). Interpret with the effect size above: a mean difference alone is not evidence of a strong mechanism association.",
       'Task specificity must be judged from the reported old/current ratios and their overlapping quantiles, not parameter-space orthogonality alone. This audit cannot certify functional isolation from orthogonality; it measures actual responses directly.',
       'The earlier P3 removal intervention and these activation statistics are complementary. They can motivate investigating functional/activation-subspace isolation if old-task responses remain substantial, but do not establish that any new method will work. No method change, training, hyperparameter search or next experiment was performed.',
       '', '## Files','', '`summary.json`, `per_sample.jsonl`, `per_bank.jsonl`, `provenance.json`, `invariants.json`, `completion.json` in '+str(out)+'.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


@torch.no_grad()
def run(args):
    out=args.output_root.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.time();torch.set_num_threads(1);torch.manual_seed(42)
    state=read_state(args.run_root/'pre_tpm/state.json')
    if state.task_id!=3:raise ValueError('Task3 required')
    request=json.loads((args.run_root/'run_request.json').read_text())
    if (request['max_length'],request['tpm']['tokens_per_sample'],request['tpm']['seed'])!=(1024,8,42):
        raise ValueError('Diagnostic contract changed')
    files=[Path(__file__),args.run_root/'run_request.json',Path(state.origin)]
    for path in state.components():
        files.extend([Path(path),Path(path).parent/json.loads(Path(path).read_text())['weights_file']])
    sources={};datasets={}
    for task in (1,2,3):
        directory=retention.evaluator.TASK_DIRS[task]
        path=args.pilot_root/'data'/directory/'development.jsonl';files.append(path)
        rows=retention.evaluator.read_jsonl(path,0)
        if len(rows)!=256:raise ValueError('Expected all256 dev rows')
        datasets[task]=rows;sources[str(task)]={'path':str(path),'ordered_ids':[stable_id(r) for r in rows]}
    predictions=[]
    for variant in ('A','B'):
        path=args.composition_root/'task2'/f'{variant}.predictions.jsonl';files.append(path)
        rows=retention.evaluator.read_jsonl(path,0)
        if [str(r['id']) for r in rows]!=sources['2']['ordered_ids']:raise ValueError('Composition sample order differs')
        predictions.append({str(r['id']):r for r in rows})
    cp=args.composition_root/'task2/provenance.json';files.append(cp)
    previous=json.loads(cp.read_text())
    devpath=sources['2']['path']
    if previous['source_sha256'][devpath]!=sha256_file(devpath):raise ValueError('Composition dev changed')
    if previous['source_sha256'][str(Path(state.origin))]!=sha256_file(state.origin):raise ValueError('Composition pre state changed')
    for path in ('med_prism/adapters/rank1_lora.py','med_prism/adapters/shared_private.py',
                 'med_prism/transport/calibration.py','med_prism/transport/hooks.py','med_prism/transport/cli.py'):
        files.append(ROOT/path)
    hashes={str(p):sha256_file(p) for p in files}
    # Swift attaches model_info/model_meta required by its teacher-forced template.
    # The original calibration loader supplies these; plain eval AutoProcessor does not.
    model,template=load_calibration_model(state,1024)
    processor=template.processor
    processor.image_processor.min_pixels=200704
    processor.image_processor.max_pixels=200704
    params=dict(model.named_parameters())
    for name,expected in state.tensors().items():
        actual=params[name].detach().cpu()
        if actual.dtype!=expected.dtype or not torch.equal(actual,expected):raise RuntimeError('Loaded pre tensor mismatch: '+name)
    bridge=QwenBlockBridge(model);capture=CaptureP3(bridge)
    if len(bridge.targets)!=72:raise ValueError('Expected72 q/v banks')
    for wrapper in bridge.targets.values():
        if tuple(wrapper.active_tasks)!=(1,2,3) or len(wrapper.task_experts(3))!=16:raise ValueError('P3 configuration mismatch')
    scaling={name:[float(e.scaling) for e in w.task_experts(3)] for name,w in bridge.targets.items()}
    write_json(out/'provenance.json',{'state':state.origin,'shared_manifest':state.shared_manifest,
        'private_manifests':state.private_manifests,'active_banks':[1,2,3],'scaling':scaling,
        'datasets':sources,'source_sha256':hashes,'processor':'original Swift calibration loader; same Qwen3VL processor/revision and composition image settings',
        'image_min_pixels':processor.image_processor.min_pixels,'image_max_pixels':processor.image_processor.max_pixels,
        'dtype':'bfloat16','forward':'teacher forced, train-mode template; model.eval; no generation',
        'token_rule':'existing attention-valid stratified select_tokens;8tokens;seed42+sha256(id)',
        'key_definition':'mean absolute actual rank1 A projection over16experts per token; individual absolutes retained',
        'functional_definition':'L2 of actual scaled expert outputs accumulated in original BF16 order',
        'normalization':'functional_response/(norm of true projection input+1e-12)',
        'behavioral_groups':'original Task2 composition A/B correct fields; no correctness redefinition'})
    print('READY: model/template loaded; P3 active on all72 wrappers; frozen diagnostic starting',flush=True)
    before=fingerprint(model);samples=[];bank_rows=[]
    for task,rows in datasets.items():
        for index,row in enumerate(rows):
            encoded=template.encode(copy.deepcopy({k:row[k] for k in ('messages','images','videos') if k in row}))
            batch=template.data_collator([encoded])
            if 'attention_mask' not in batch:batch['attention_mask']=torch.ones_like(batch['input_ids'])
            if 'labels' not in batch or batch['input_ids'].shape[0]!=1:raise ValueError('Teacher-forced batch mismatch')
            record={'id':stable_id(row),'batch':batch}
            observations=capture.collect(record,verify=task==1 and index==0)
            sample={'task':task,'id':record['id'],'token_selection':record['selection']}
            for metric in METRICS:sample[metric]=float(np.mean([v for obs in observations.values() for v in obs[metric]]))
            if task==2:
                a,b=[p[record['id']]['correct'] for p in predictions]
                sample.update(teacher_correct=bool(a),pre_correct=bool(b),
                    behavior_group=('teacher_correct_to_' if a else 'teacher_wrong_to_')+('correct' if b else 'wrong'))
            with (out/'per_bank.jsonl').open('a') as f:
                for name,obs in observations.items():
                    block=next(i for i,names in enumerate(bridge.block_targets) if name in names)
                    entry={'task':task,'id':record['id'],'module':name,'block':block,
                        'projection':name.rsplit('.',1)[-1],'depth':('shallow','middle','deep')[min(2,block*3//len(bridge.blocks))],**obs}
                    f.write(json.dumps(entry)+'\n');bank_rows.append({k:v for k,v in entry.items() if k!='key_abs_per_expert'})
            with (out/'per_sample.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n')
            samples.append(sample)
            if (index+1)%32==0:print(f'Task{task} {index+1}/256 elapsed={time.time()-started:.1f}s',flush=True)
    if len(capture.verified)!=72:raise RuntimeError('Incomplete exact-forward verification')
    invariants=check_invariants(before,fingerprint(model),set())
    if any(sha256_file(p)!=h for p,h in hashes.items()):raise RuntimeError('Source file changed')
    if any(tuple(w.active_tasks)!=(1,2,3) for w in bridge.targets.values()):raise RuntimeError('Active banks changed')
    write_json(out/'invariants.json',invariants)
    summary={'status':'PASS','samples':len(samples),'exact_forward_verified_banks':len(capture.verified),'tasks':{},'invariants':invariants}
    def stats(rows):return {m:statistics([v for r in rows for v in r[m]]) for m in METRICS}
    for task in (1,2,3):
        selected=[r for r in bank_rows if r['task']==task]
        summary['tasks'][str(task)]={'all':stats(selected),'breakdown':{
            f'{key}:{value}':stats([r for r in selected if r[key]==value])
            for key in ('projection','depth') for value in sorted({r[key] for r in selected})}}
    summary['task3_ratios']={str(t):{m:summary['tasks'][str(t)]['all'][m]['mean']/summary['tasks']['3']['all'][m]['mean'] for m in METRICS} for t in (1,2)}
    summary['behavioral_correlation']=correlation(samples)
    write_json(out/'summary.json',summary);render_report(summary,out)
    write_json(out/'completion.json',{'status':'PASS','elapsed_seconds':time.time()-started,'samples':768,
        'report':str(out/'report.md'),'source_hashes_unchanged':True,'persistent_tensors_unchanged':True})
    print('COMPLETE '+str(out/'report.md'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('run-root','pilot-root','composition-root','output-root'):p.add_argument('--'+arg,type=Path,required=True)
    args=p.parse_args()
    if args.output_root.exists():p.error('Refusing existing output directory')
    try:run(args)
    except Exception as exc:
        if args.output_root.exists() and not (args.output_root/'completion.json').exists():
            write_json(args.output_root/'failure.json',{'status':'FAIL','error':str(exc),'traceback':traceback.format_exc()})
        raise
