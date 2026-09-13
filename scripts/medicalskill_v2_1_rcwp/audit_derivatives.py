"""Frozen full-gate derivative-chain audit. Never fits, trains or saves weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import torch
from med_prism.rcwp.boundary import forward_margin,harmful_auc
from med_prism.rcwp.core import factors,output_basis,phi
from med_prism.rcwp.runtime import wrappers,load_summary
from med_prism.rcwp.data import encode_boundary_rows
from med_prism.transport.cli import load_calibration_model
from med_prism.transport.checkpoint import read_state,sha256_file
from med_prism.transport.banks import fingerprint,check_invariants
from med_prism.transport.diagnostics import write_json

SCALES=(.01,.05,.1)
EPS=1e-20


def dot(a,b):return float((a.double()*b.double()).sum())
def sign(x):return (x>0)-(x<0)


def stats(values):
    x=torch.tensor(values,dtype=torch.float64)
    return {"count":len(values),"mean":float(x.mean()),"median":float(x.median()),
        "p90":float(torch.quantile(x,.9)),"min":float(x.min()),"max":float(x.max())}


def correlation(a,b):
    from scipy.stats import spearmanr
    if len(set(a))<2 or len(set(b))<2:return None
    v=float(spearmanr(a,b).statistic)
    return v if math.isfinite(v) else None


def metrics(rows,key):
    actual=[r['actual'] for r in rows];pred=[r[key] for r in rows]
    return {"n":len(rows),"sign_accuracy":sum((x<0)==(y<0) for x,y in zip(actual,pred))/len(rows),
        "three_way_sign_agreement":sum(sign(x)==sign(y) for x,y in zip(actual,pred))/len(rows),
        "spearman":correlation(pred,actual),"harmful_auroc":harmful_auc([x<0 for x in actual],[-x for x in pred])}


def derivatives(g,Q,predicted,delta):
    projected=(g@Q)@Q.T
    exact=dot(g,delta);oracle=dot(projected,delta);field=float((predicted*delta).sum())
    return {"d_exact":exact,"d_Q":oracle,"d_RCWP":field,
        "coverage_g":dot(g@Q,g@Q)/(dot(g,g)+EPS),
        "coverage_delta":dot(delta@Q,delta@Q)/(dot(delta,delta)+EPS),
        "projected_derivative_ratio":abs(oracle)/(abs(exact)+EPS),
        "sign_Q_exact":sign(oracle)==sign(exact),"sign_RCWP_Q":sign(field)==sign(oracle),
        "sign_RCWP_exact":sign(field)==sign(exact)}


def baseline(model,batch,chosen):
    capture={};handles=[]
    for name,module in chosen.items():
        def hook(m,args,out,name=name):
            capture[name]={"h":args[0].detach(),"out":out.detach().clone(),"live":out.requires_grad_(True)}
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.enable_grad():
            margin=forward_margin(model,batch)
            gradients=torch.autograd.grad(margin,[capture[n]['live'] for n in chosen])
        for name,g in zip(chosen,gradients):
            capture[name].pop('live');capture[name]['g']=g.detach().float()
        value=float(margin.detach())
    finally:
        for handle in handles:handle.remove()
    if any(p.grad is not None for p in model.parameters()):raise RuntimeError('Parameter gradient populated')
    return value,capture


def observations(model,batch,summary,modules,ordinal):
    names=list(modules);chosen={n:modules[n] for n in names[-2:]}
    grad_margin,cache=baseline(model,batch,chosen)
    with torch.no_grad():before=float(forward_margin(model,batch))
    rows=[];fd=[]
    for name,module in chosen.items():
        A,B=factors(module,summary['task_id']);Q,_,_=output_basis(B)
        f=summary['fields'][name]
        M,mu,scale=[f[k].to(A.device) for k in ('M','mu','scale')]
        C=torch.randn(len(A),len(A),generator=torch.Generator().manual_seed(2100+names.index(name))).to(A.device)
        C=C/C.norm()
        saved=cache[name];h=saved['h'].float();g=saved['g']
        valid=batch['attention_mask'].bool().to(h.device)
        g=g.masked_fill(~valid[...,None],0)
        direction=(h@A.T@C.T@B.T).masked_fill(~valid[...,None],0)
        predicted=phi(h@A.T,mu,scale)@M.T@B.T
        ideal=derivatives(g,Q,predicted,direction)
        meta={'holdout_index':ordinal,'wrapper':name,'projection':name.rsplit('.',1)[-1],
            'layer':int(name.split('.layers.')[1].split('.')[0]),
            'baseline_margin':before,'grad_baseline_margin_difference':grad_margin-before,
            'd_exact_direction':ideal['d_exact'],'d_Q_direction':ideal['d_Q'],
            'ideal_direction_B_coverage':ideal['coverage_delta']}
        for amplitude in SCALES:
            pair={};deployed={}
            for polarity in (-1,1):
                scalar={}
                def inject(m,args,out):
                    # Upstream inputs and original output must match the gradient point.
                    if not torch.equal(args[0],saved['h']) or not torch.equal(out,saved['out']):
                        raise RuntimeError('Gradient point differs from unperturbed injection point')
                    # EXACT expression/order from original gate, including BF16 cast/add.
                    z=args[0].detach().float()@A.T
                    requested=(z@C.T@B.T)*(amplitude*polarity)
                    requested=requested.masked_fill(~valid[...,None],0)
                    edited=out+requested.to(out.dtype)
                    actual_delta=edited.float()-out.float()
                    scalar.update(derivatives(g,Q,predicted,actual_delta))
                    scalar.update(delta_norm=float(actual_delta.norm()),
                        intended_delta_norm=float(requested.norm()),
                        zero_edit=bool(torch.equal(edited,out)),
                        rounded_component_fraction=float((actual_delta[valid]==0).float().mean()),
                        quantization_error_norm=float((actual_delta-requested).norm()))
                    deployed[polarity]=actual_delta.detach()
                    return edited
                handle=module.register_forward_hook(inject)
                try:
                    with torch.no_grad():after=float(forward_margin(model,batch))
                finally:handle.remove()
                pair[polarity]=after
                rows.append({**meta,'amplitude':amplitude,'polarity':polarity,'actual':after-before,
                    'absolute_margin_delta':abs(after-before),'identical_margin':after==before,**scalar})
            central=(pair[1]-pair[-1])/(2*amplitude)
            fd.append({**meta,'amplitude':amplitude,'fd_slope':central,
                'fd_exact_sign_agreement':sign(central)==sign(ideal['d_exact']),
                'pair_identical_margin':pair[1]==pair[-1],
                'deployed_chord_derivative':dot(g,(deployed[1]-deployed[-1])/(2*amplitude))})
        del saved,direction,predicted,g
    return rows,fd


def aggregate(rows,fd):
    result={};coverage={};finite={}
    for a in SCALES:
        selected=[r for r in rows if r['amplitude']==a]
        groups={'all':selected}
        for field in ('projection','wrapper'):
            for value in sorted({r[field] for r in selected}):groups[field+':'+value]=[r for r in selected if r[field]==value]
        result[str(a)]={group:{key:metrics(rs,key) for key in ('d_exact','d_Q','d_RCWP')} for group,rs in groups.items()}
        coverage[str(a)]={group:{**{k:stats([r[k] for r in rs]) for k in
            ('coverage_g','coverage_delta','projected_derivative_ratio','ideal_direction_B_coverage')},
            **{k:sum(r[k] for r in rs)/len(rs) for k in ('sign_Q_exact','sign_RCWP_Q','sign_RCWP_exact')},
            'Q_exact_spearman':correlation([r['d_Q'] for r in rs],[r['d_exact'] for r in rs]),
            'RCWP_Q_spearman':correlation([r['d_RCWP'] for r in rs],[r['d_Q'] for r in rs])}
            for group,rs in groups.items()}
        pairs=[r for r in fd if r['amplitude']==a]
        finite[str(a)]={'exact_FD_sign_agreement':sum(r['fd_exact_sign_agreement'] for r in pairs)/len(pairs),
            'exact_FD_spearman':correlation([r['d_exact_direction'] for r in pairs],[r['fd_slope'] for r in pairs]),
            'deployed_chord_FD_spearman':correlation([r['deployed_chord_derivative'] for r in pairs],[r['fd_slope'] for r in pairs]),
            'pair_tie_fraction':sum(r['pair_identical_margin'] for r in pairs)/len(pairs),
            'baseline_identical_margin_fraction':sum(r['identical_margin'] for r in selected)/len(selected),
            'zero_edit_fraction':sum(r['zero_edit'] for r in selected)/len(selected),
            'grad_vs_nograd_baseline_margin_abs_difference':stats([abs(r['grad_baseline_margin_difference']) for r in selected]),
            'absolute_margin_delta':stats([r['absolute_margin_delta'] for r in selected])}
    return result,coverage,{'summary':finite,'per_direction':fd}


def report(root,summary,coverage,finite):
    lines=['# Frozen RCWP derivative-chain audit','',
        'No refit, optimizer step, checkpoint edit or method change. Oracle denotes a local derivative, not task-ID oracle evaluation.','',
        '|Amplitude|Predictor|Sign accuracy|Spearman|Harmful AUROC|','|---|---|---:|---:|---:|']
    for a,groups in summary['predictors'].items():
        for k,v in groups['all'].items():lines.append(f"|{a}|{k}|{v['sign_accuracy']:.4f}|{v['spearman']}|{v['harmful_auroc']}|")
    lines+=['','## Answers and limitations','']
    base=summary['predictors']['0.01']['all'];c=coverage['0.01']['all']
    lines+=[f"1. Exact derivative vs actual: {base['d_exact']}. This is the autograd derivative at the frozen BF16 operating point, not an RCWP prediction.",
        f"2. Historical B gradient-energy coverage: {c['coverage_g']}. See coverage.json for q/v and wrapper/layer breakdown.",
        f"3. Projected oracle vs actual: {base['d_Q']}; Q/exact sign agreement={c['sign_Q_exact']:.4f}, derivative ratio={c['projected_derivative_ratio']}.",
        f"4. Conditional vs actual: {base['d_RCWP']}; conditional/Q sign agreement={c['sign_RCWP_Q']:.4f}, Spearman={c['RCWP_Q_spearman']}.",
        f"5. Fixed-resolution audit (no amplitude selected): {finite['summary']}."]
    auc=[base[k]['harmful_auroc'] for k in ('d_exact','d_Q','d_RCWP')]
    if any(x is None for x in auc):interpret='Insufficient harmful/nonharmful classes; attribution unresolved.'
    elif auc[0]<.65:interpret='Case 3 is the first limitation: exact local derivative itself does not clear the existing .65 research screen under this deployed protocol. Do not attribute all failure to the learned field.'
    elif auc[1]<.65:interpret='Case 1 pattern for DEPLOYED edits: projection loses predictive information. Check off-span BF16 error before calling this an inadequate basis for ideal writes.'
    elif auc[2]<.65:interpret='Case 2 pattern: the projected oracle remains predictive but the conditional field does not; the conditional approximation is the main measured bottleneck.'
    else:interpret='All three clear the descriptive .65 screen; compare their gaps and scale dependence, without asserting a unique failure cause.'
    lines+=['6. '+interpret,'',
        'The .65 screen is descriptive, not a significance test. Observations share examples/directions and are not independent.',
        'Crucial: the prescribed ideal direction B C A h lies inside col(B) by construction. Low total gradient coverage cannot by itself prove insufficient coverage for these ideal writes. BF16 rounding can push the deployed edit outside that span.',
        'Predictor tables use the actual deployed displacement; central differences compare against the UN-SCALED ideal directional derivative. Their units are not mixed.',
        'No new method design, amplitude recommendation or pilot follows this audit.']
    (root/'report.md').write_text('\n\n'.join(lines)+'\n')


def run(args):
    gate=Path(args.gate_root).resolve();root=Path(args.output_root).resolve()
    if root.exists():raise FileExistsError(root)
    request=json.loads((gate/'request.json').read_text());complete=json.loads((gate/'completion.json').read_text())
    if complete['status']!='PASS' or request['config']['fit_samples']!=128 or request['config']['holdout_samples']!=64:
        raise ValueError('Expected completed full128+64 gate')
    hashes={**request['source']['hashes'],**request['code_hashes'],
        str(gate/'summary.pt'):complete['summary_sha256']}
    for p,v in hashes.items():
        if sha256_file(p)!=v:raise ValueError('Full-gate source/code/summary changed: '+p)
    for p in ('request.json','completion.json','field_quality.json'):hashes[str(gate/p)]=sha256_file(gate/p)
    hashes[str(Path(__file__).resolve())]=sha256_file(__file__)
    state=read_state(request['source']['state'])
    data=next(Path(p) for p in request['source']['hashes'] if Path(p).name=='train.jsonl')
    root.mkdir(parents=True);started=time.time();torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    write_json(root/'provenance.json',{'gate':str(gate),'hashes':hashes,'scales':SCALES,'refit':False,
        'source':request['source'],'split':'same hash-verified data/encoder/seed, deterministic 128+64 reconstruction',
        'scope':'all 64 holdout; original last/deep q and v only; scalar output only'})
    print('PRECHECK PASS: original full-gate source/code/summary hashes verified; loading frozen model',flush=True)
    model,template=load_calibration_model(state,1024)
    template.processor.image_processor.min_pixels=200704;template.processor.image_processor.max_pixels=200704
    before=fingerprint(model);modules=wrappers(model);summary=load_summary(gate/'summary.pt',modules)
    records,rejected=encode_boundary_rows([json.loads(l) for l in data.read_text().splitlines() if l.strip()],template,128,64)
    if rejected!=complete['encoding_rejections']:raise ValueError('Split encoding changed')
    holdout=[r for r in records if r['split']=='holdout'];del records
    if args.smoke:holdout=holdout[:1]
    split_hash=hashlib.sha256()
    for r in holdout:
        for k in ('input_ids','attention_mask','labels'):
            split_hash.update(r['batch'][k].contiguous().numpy().tobytes())
    print('READY: frozen summary loaded, original split rebuilt, derivative audit running',flush=True)
    rows=[];fds=[]
    with (root/'per_observation.jsonl').open('w') as stream:
        for index,r in enumerate(holdout):
            observations_,fd=observations(model,r['batch'],summary,modules,index)
            for observation in observations_:stream.write(json.dumps(observation,allow_nan=False)+'\n')
            stream.flush();rows+=observations_;fds+=fd
    results,coverage,finite=aggregate(rows,fds)
    reproduction={}
    if not args.smoke:
        reference=json.loads((gate/'field_quality.json').read_text())['metrics']['conditional']
        measured=results['0.01']['all']['d_RCWP']
        for key in ('sign_accuracy','spearman','harmful_auroc'):
            reproduction[key]={'original':reference[key],'reproduced':measured[key]}
            if abs(reference[key]-measured[key])>1e-9:raise RuntimeError('Original gate metric reproduction mismatch: '+key)
    invariants=check_invariants(before,fingerprint(model),set());state.validate()
    for p,v in hashes.items():
        if sha256_file(p)!=v:raise RuntimeError('Source changed during diagnostic: '+p)
    result={'status':'PASS','holdout':len(holdout),'observations':len(rows),'predictors':results,
        'original_gate_reproduction':reproduction,'encoded_holdout_sha256':split_hash.hexdigest(),
        'elapsed_seconds':time.time()-started,'smoke':args.smoke}
    write_json(root/'summary.json',result);write_json(root/'coverage.json',coverage)
    write_json(root/'finite_difference.json',finite);write_json(root/'invariants.json',invariants)
    report(root,result,coverage,finite)
    write_json(root/'completion.json',{'status':'PASS','report':str(root/'report.md'),'invariants':'PASS'})
    print('COMPLETE '+str(root),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gate-root',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    try:run(args)
    except Exception as e:
        root=Path(args.output_root)
        if root.exists() and not (root/'completion.json').exists():write_json(root/'failure.json',{'error':str(e),'traceback':traceback.format_exc()})
        raise
