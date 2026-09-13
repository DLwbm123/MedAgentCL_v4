"""Two independent GPU workers; immutable manifests and per-composition outputs."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import torch
from med_prism.ms_crc.composition import Composition
from med_prism.ms_crc.data import current_split
from med_prism.ms_crc.evaluator import Evaluator,legacy,cell,equivalent,token_scores
from med_prism.ms_crc.search import IDENTITY
from med_prism.ms_crc.fold import export as export_v22
from med_prism.ms_crc.gate import assert_fingerprints
from med_prism.ms_crc.analysis import describe,paired_ci
from med_prism.transport.cli import load_calibration_model,TASKS
from med_prism.transport.checkpoint import read_state,AdapterState,sha256_file
from med_prism.transport.banks import fingerprint
from .behavior import ConceptRecognitionAdapter


def read(path):return json.loads(Path(path).read_text())
def write(path,value):cell.write_json(Path(path),value)
def verify(root,locked=False):
    req=read(Path(root)/'request.json')
    for p,h in req['input_hashes'].items():
        if sha256_file(p)!=h:raise RuntimeError('Input/code modified: '+p)
    if locked:
        lock=read(Path(root)/'selection_lock.json')
        if lock['SELECTION_LOCKED'] is not True:raise PermissionError('Old dev remains locked')
        for p,h in lock['hashes'].items():
            if sha256_file(p)!=h:raise RuntimeError('Locked artifact changed: '+p)
    return req


@torch.no_grad()
def concept_score(scorer,records,g=IDENTITY,active=(1,2,3)):
    adapter=ConceptRecognitionAdapter();scored=[];rows=[r['row'] for r in records]
    torch.manual_seed(42)
    with scorer.composition.use(g,active):
        predictions=cell.batched_generate(scorer.model,scorer.processor,rows,3,4)
        if any(not isinstance(p,str) for p in predictions):raise RuntimeError('PARSER_FAILURE')
        official,details=legacy.evaluate_rows(3,rows,predictions)
        # Its status flag only requires nonempty strings; the official counts
        # still correctly count empty predictions as FN. Keep that distinction.
        if len(details)!=len(records):raise RuntimeError('Evaluator sample count differs')
        for detail,record in zip(details,records):
            parsed=adapter.parse(record['row'],detail['prediction'])
            if not parsed['valid']:raise RuntimeError('PARSER_FAILURE')
            for k in ('all_tp','all_fp','all_fn'):
                if parsed[k]!=detail[k]:raise RuntimeError('METRIC_MISMATCH atom counts vs official')
            scalar,logits=token_scores(scorer.model,record['batch']);del logits
            scored.append({'sample_id':str(record['row']['id']),'prediction':detail['prediction'],
                'correct':parsed['all_fp']==parsed['all_fn']==0,**scalar,**parsed})
    aggregate=adapter.metric(scored)
    if abs(aggregate['utility']-official['all_concept_micro_f1'])>1e-12:raise RuntimeError('METRIC_MISMATCH micro-F1')
    return scored,{**aggregate,'CE':sum(r['nll_sum'] for r in scored)/sum(r['answer_tokens'] for r in scored)}


def folded_export(source,comp,g,destination):
    # Reuse v2.2 export schema/invariant checks. Its search-space restriction is
    # not used: v2.3 admits exactly the locked finite-panel global coefficients.
    original=source.tensors();values={k:v.clone() for k,v in original.items()}
    names={id(p):n for n,p in comp.model.named_parameters()}
    for expert,_,B,gid in comp.entries:
        name=names[id(expert.B)]
        if not torch.equal(original[name],B.cpu()):raise RuntimeError('Not immutable original B')
        values[name]=(B*g[gid]).cpu().contiguous()
    deployed=export_v22(source,values,destination,g)
    manifest=Path(deployed.private_manifests[-1]);header=read(manifest)
    header['ms_crc'].update(method_version='2.3',selector='Task-Aware MS-CRC');write(manifest,header)
    state=read(deployed.origin);state.update(method_version='2.3',method_name='Med-PRISM-v2.3-Task-Aware-MS-CRC')
    state['format']='MedPRISM_TaskAware_MS_CRC_state_v1'
    state['manifest_sha256']={p:sha256_file(p) for p in deployed.components()};write(deployed.origin,state)
    return AdapterState(3,deployed.shared_manifest,deployed.private_manifests,deployed.origin).validate(),values


def worker(root,phase,index):
    root=Path(root);req=verify(root,phase=='development');torch.manual_seed(42);torch.set_num_threads(1)
    jobs=read(root/f'{phase}_jobs.json')[index::2]
    if not jobs:write(root/f'{phase}_worker{index}_done.json',{'status':'PASS','jobs':0});return
    source=read_state(req['source_state']);model,template=load_calibration_model(source,1024)
    comp=Composition(model);scorer=Evaluator(model,template,comp);before=fingerprint(model)
    if comp.group_map!=read(root/'group_map.json'):raise RuntimeError('Group map differs from v2.2')
    fit=holdout=dev=None
    if phase in ('fit','holdout','export','smoke'):
        fit,holdout,split=current_split(Path(req['data_root'])/TASKS[3]/'train.jsonl',template,
            2 if req['smoke'] else 128,1 if req['smoke'] else 64)
        if not req['smoke'] and split!=read(root/'v22_current_split.json'):raise RuntimeError('v2.2 split differs')
        write(root/f'{phase}_worker{index}_split.json',split)
    if phase=='metric':
        path=Path(req['data_root'])/TASKS[3]/'development.jsonl'
        if sha256_file(path)!=req['current_dev_hash']:raise RuntimeError('Current dev changed')
        dev=scorer.records(legacy.read_jsonl(path,0))
    for job in jobs:
        verify(root,phase=='development')
        dest=root/job['output']
        if dest.exists():
            # Only resumable within identical frozen request; verify output sidecar.
            if sha256_file(dest)!=read(str(dest)+'.sha256.json')['sha256']:raise RuntimeError('Resume artifact changed')
            continue
        print(f'START {phase} GPU{index} {job["label"]}',flush=True)
        g=tuple(job.get('g',IDENTITY));active=tuple(job.get('active',(1,2,3)))
        if phase=='development':
            task=job['task'];path=Path(req['data_root'])/TASKS[task]/'development.jsonl'
            # First old-data access is AFTER lock verification above.
            if sha256_file(path)!=req['development_hashes'][str(task)]:raise RuntimeError('Old dev hash mismatch')
            records=scorer.records(legacy.read_jsonl(path,0));allrows={}
            vectors=read(root/'deployment_vectors.json');cache={}
            for label,vector in vectors.items():
                vector=tuple(vector)
                if vector not in cache:cache[vector]=scorer.score(records,vector,task=task,kl=True)[0]
                allrows[label]=cache[vector]
                # Persist aggregates immediately; no old per-sample cache.
                if 'full' in allrows and 'H' in allrows:
                    write(root/f'task{task}_development_progress.json',{k:describe(v,allrows['full'],allrows['H']) for k,v in allrows.items()})
            result={'task':task,'scope':'ALREADY_UNBLINDED_DEVELOPMENT_MECHANISM_ONLY','dataset_sha256':sha256_file(path),
                'variants':{k:describe(v,allrows['full'],allrows['H']) for k,v in allrows.items()},
                'selected_vs_global':paired_ci(allrows['selected'],allrows['global']),
                'KL_reference':'same-S3 H, NOT Task2 boundary teacher'}
            del allrows,cache,records
        elif phase=='export':
            expected,_=concept_score(scorer,holdout,g)
            repeat,_=concept_score(scorer,holdout,g);equivalent(expected,repeat,0.)
            deployment,values=folded_export(source,comp,g,root/'exported')
            allowed=[k for k,v in source.tensors().items() if not torch.equal(v,values[k])]
            assert_fingerprints(before,fingerprint(model))
            del scorer,comp,model,template;gc.collect();torch.cuda.empty_cache()
            model,template=load_calibration_model(deployment,1024);comp=Composition(model);scorer=Evaluator(model,template,comp)
            actual,_=concept_score(scorer,holdout);equivalent(expected,actual,0.)
            loaded=model.state_dict()
            for k,v in values.items():
                if not torch.equal(loaded[k].cpu(),v):raise RuntimeError('Reload tensor mismatch '+k)
            result={'status':'PASS','strict_numerical_tolerance':0.,'exported_state':deployment.origin,
                'persistent_invariants':assert_fingerprints(before,fingerprint(model),allowed)}
            # Establish the newly deployed tensor baseline for final read-only check.
            before=fingerprint(model)
        else:
            records=dev if phase=='metric' else (holdout if phase=='holdout' else fit)
            rows,metric=concept_score(scorer,records,g,active)
            result={'label':job['label'],'g':g,'active':active,'metric':metric}
            if phase=='metric':
                target=read(root/'v22_current_panel.json')['variants'][job['label']]['existing_all_concept_micro_f1']
                result['v22_F1']=target;result['status']='PASS' if abs(metric['utility']-target)<=1e-12 else 'METRIC_MISMATCH'
                # v2.2 did NOT persist counts or predictions: verify new TP/FP/FN
                # against official evaluator, never reconstruct atoms from old F1.
                result['old_TP_FP_FN_available']=False
                if result['status']!='PASS':write(dest,result);raise RuntimeError('METRIC_MISMATCH '+job['label'])
            else:result['rows']=rows # current-only temporary, removed before old dev.
        write(dest,result);write(str(dest)+'.sha256.json',{'sha256':sha256_file(dest)})
        print(f'DONE {phase} GPU{index} {job["label"]}',flush=True)
    verify(root,phase=='development')
    write(root/f'{phase}_worker{index}_done.json',{'status':'PASS','jobs':len(jobs),
        'invariants':assert_fingerprints(before,fingerprint(model))})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase',required=True);p.add_argument('--index',type=int,required=True)
    a=p.parse_args()
    try:worker(a.root,a.phase,a.index)
    except Exception as exc:
        import traceback
        write(Path(a.root)/f'{a.phase}_worker{a.index}_failure.json',{'error':str(exc),'traceback':traceback.format_exc()});raise
