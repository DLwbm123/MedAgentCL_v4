"""Frozen v2.3 gate orchestration. No training; two GPU composition workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from med_prism.ms_crc.search import IDENTITY,panel
from med_prism.ms_crc.gate import validate_lock as validate_v22
from med_prism.transport.checkpoint import read_state,sha256_file
from .runtime import read,write,verify
from .behavior import AtomicSupport,select,accept

REPO=Path(__file__).resolve().parents[3]
DEFAULT_V22='/remote-home/wangbomin/MedPRISM_v2_2_MS_CRC_gate_20260909_145121_3757'


def prepare(root,old,smoke):
    if root.exists():raise FileExistsError('Fresh output root required')
    validate_v22(old)
    provenance=read(old/'provenance.json');config=read(old/'request.json');source=read_state(provenance['source_state'])
    if source.task_id!=3:raise ValueError('Task3 source required')
    if read(source.origin)['status']!='PRE_TPM':raise ValueError('Only PRE_TPM')
    for p,h in provenance['source_hashes'].items():
        if sha256_file(p)!=h:raise RuntimeError('v2.2 source changed '+p)
    for p,h in provenance['code_hashes'].items():
        if sha256_file(p)!=h:raise RuntimeError('v2.2 evaluator/implementation changed '+p)
    current_paths=[p for p in provenance['source_hashes'] if p.endswith('/task_03_concept_recognition/train.jsonl')]
    if len(current_paths)!=1:raise ValueError('Ambiguous current dataset')
    data=Path(current_paths[0]).parent.parent
    old_panel=read(old/'finite_panel_current.json');expected=panel(IDENTITY)
    if {k:tuple(v) for k,v in old_panel['vectors'].items()}!=expected:raise ValueError('Fixed v2.2 panel mismatch')
    # Preserve documented order; collapse aliases only, never rank by dev scores.
    vectors={};seen=set()
    for k,g in expected.items():
        if g not in seen:vectors[k]=list(g);seen.add(g)
    groups=read(old/'group_map.json')
    from collections import Counter
    if len(groups)!=1152 or Counter(g['group_id'] for g in groups)!={i:72 for i in range(16)}:raise ValueError('Group cardinality')
    if len({g['wrapper'] for g in groups})!=72:raise ValueError('Wrapper cardinality')
    for item in groups:
        if item['group_id']!=4*(item['layer']//9)+item['rank_idx_zero_based']//4:raise ValueError('Group geometry')
    hashes={**provenance['source_hashes'],**provenance['code_hashes']}
    for p in [*Path(__file__).parent.glob('*.py'),REPO/'scripts/medicalskill_v1_2_med_prism/evaluation_contract_v1_2.py',
              REPO/'docs/med_prism_v2_3_task_aware_ms_crc/Med_PRISM_v2_3_Task_Aware_MS_CRC_Design.md']:
        hashes[str(p)]=sha256_file(p)
    for name in ('provenance.json','request.json','current_split.json','group_map.json','finite_panel_current.json'):
        hashes[str(old/name)]=sha256_file(old/name)
    devhash=provenance['dev_hashes_after_lock']
    root.mkdir(parents=True);(root/'cache').mkdir()
    write(root/'group_map.json',groups);write(root/'candidate_manifest.json',vectors)
    write(root/'v22_current_split.json',read(old/'current_split.json'));write(root/'v22_current_panel.json',old_panel)
    write(root/'request.json',{'method':'Med-PRISM v2.3 Task-Aware MS-CRC','source_state':source.origin,'data_root':str(data),
        'v22_root':str(old),'input_hashes':hashes,'current_dev_hash':devhash[str(data/'task_03_concept_recognition/development.jsonl')],
        'development_hashes':{str(t):devhash[str(data/n/'development.jsonl')] for t,n in ((1,'task_01_vqa'),(2,'task_02_diagnosis_classification'))},
        'smoke':smoke,'fit':2 if smoke else 128,'holdout':1 if smoke else 64,'expansion':False,
        'CE_tolerance':.02,'metric_tolerance':1e-12,'generation':config['generation'],'image_settings':config['image_settings'],
        'tpm_enabled':False,'rcwp_enabled':False,'training':False,'coordinate_search':False,'gpus':[0,1],
        'selector':'lex L_mem,L_PT,L1; fixed pooled max-bank atoms; q=full official microF1',
        'holdout_rule':'one g_fit and independently selected scalar baseline vs identity; no reranking',
        'development_scope':'already unblinded development/mechanism ONLY','formal_unseen_validation':'NOT RUN'})
    write(root/'provenance.json',{'source_manifests':source.components(),'source_hashes':provenance['source_hashes'],
        'candidate_manifest_sha256':sha256_file(root/'candidate_manifest.json'),'group_map_sha256':sha256_file(root/'group_map.json'),
        'references':{'H':'base+S3+P1+P2','H_without_P1':'base+S3+P2','H_without_P2':'base+S3+P1'},
        'metric_counts_note':'v2.2 persisted F1, not TP/FP/FN or raw outputs; re-run official parser and verify counts against fresh evaluator'})
    print('AUDIT PASS: immutable v2.2 source, grouping, panel, parser and settings verified; GPU0+1',flush=True)


def phase(root,name,jobs):
    path=root/f'{name}_jobs.json'
    if path.exists() and read(path)!=jobs:raise RuntimeError('Resume job manifest changed')
    if all((root/f'{name}_worker{i}_done.json').exists() for i in (0,1)):
        verify(root,name=='development')
        for j in jobs:
            dest=root/j['output']
            if sha256_file(dest)!=read(str(dest)+'.sha256.json')['sha256']:raise RuntimeError('Completed output changed')
        return
    write(path,jobs);children=[]
    for index in (0,1):
        log=(root/f'{name}_gpu{index}.log').open('a')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(index),'PYTHONPATH':str(REPO)}
        p=subprocess.Popen([sys.executable,'-u','-m','med_prism.ms_crc.task_aware.runtime','--root',str(root),
            '--phase',name,'--index',str(index)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append((p,log));write(root/f'{name}_gpu{index}_pid.json',{'pid':p.pid,'gpu':index})
    failed=[]
    for p,log in children:
        code=p.wait();log.close()
        if code:failed.append((p.pid,code))
    if failed:raise RuntimeError(f'{name} workers failed: {failed}; see per-GPU logs')
    for i in (0,1):
        if read(root/f'{name}_worker{i}_done.json')['status']!='PASS':raise RuntimeError('Missing worker invariant PASS')


def job(label,g=IDENTITY,active=(1,2,3),prefix='cache',tag='fit'):
    return {'label':label,'g':list(g),'active':list(active),'output':f'{prefix}/{tag}_{label}.json'}


def references(tag):return [job('H',active=(1,2),tag=tag),job('minus1',active=(2,),tag=tag),job('minus2',active=(1,),tag=tag),job('full',tag=tag)]
def rows(root,tag,label):return read(root/f'cache/{tag}_{label}.json')['rows']
def support(root,tag):return AtomicSupport(rows(root,tag,'H'),rows(root,tag,'full'),{1:rows(root,tag,'minus1'),2:rows(root,tag,'minus2')})
def save_support(root,tag,obj):
    diag=obj.diagnostics();write(root/f'{tag}_atomic_support.json',diag);write(root/f'{tag}_per_bank_support.json',diag['per_bank'])


def lock(root,vectors):
    write(root/'selected_coefficients.json',{'method_version':'2.3','g':vectors['selected']})
    write(root/'deployment_vectors.json',vectors)
    export=read(root/'exported/state.json');manifest=Path(export['private_manifests'][-1]);weights=manifest.parent/read(manifest)['weights_file']
    names=['request.json','candidate_manifest.json','group_map.json','fit_selection.json','holdout_result.json',
        'global_alpha_current_selection.json','selected_coefficients.json','deployment_vectors.json','exported/state.json']
    files=[*(root/n for n in names),manifest,weights]
    write(root/'selection_lock.json',{'SELECTION_LOCKED':True,'hashes':{str(p):sha256_file(p) for p in files},
        'locked_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    verify(root,True)


def clear_cache(root):
    if (root/'current_cache_removed.json').exists():return
    cache=(root/'cache').resolve()
    if cache.parent!=root.resolve() or cache.name!='cache':raise RuntimeError('Unsafe cache path')
    for p in cache.iterdir():
        if p.is_file() and p.suffix=='.json':p.unlink()
        else:raise RuntimeError('Unexpected cache entry')
    cache.rmdir()
    write(root/'current_cache_removed.json',{'removed':str(cache),'no_cross_task_sample_memory':True})


def finish(root,status,started):
    req=verify(root,True)
    inv={p.name:read(p).get('invariants',{}) for p in root.glob('*_worker*_done.json')}
    write(root/'invariants.json',{'status':'PASS','phases':inv,'source_files_unchanged':True})
    smoke=req['smoke']
    if not smoke:
        t2=read(root/'task2_development_metrics.json');t1=read(root/'task1_development_metrics.json')
        gain=t2['variants']['selected']['accuracy_delta_vs_full'];nonuniform=len(set(read(root/'selected_coefficients.json')['g']))>1
        outcome=(status if status!='ACCEPTED_CURRENT_HOLDOUT' else 'GLOBAL_ONLY' if not nonuniform else
            'CROSS_DISTRIBUTION_SELECTION_FAILURE' if gain['delta']<=0 else
            'INCONCLUSIVE' if gain['delta']<.03 or gain['CI95'][0]<=0 or t2['selected_vs_global']['CI95'][0]<=0 else
            'UTILITY_REJECTED' if t1['variants']['selected']['accuracy_delta_vs_full']['delta']<-.01 else 'DEVELOPMENT_PASS')
        summary={'status':outcome,'holdout_status':status,'task2_gain':gain,'selected_vs_global':t2['selected_vs_global'],
            'task1_safety':t1['variants']['selected']['accuracy_delta_vs_full'],'nonuniform_selected':nonuniform,
            'validation_levels':{'implemented':True,'mechanically_verified':True,
                'atomic_observability':read(root/'fit_atomic_support.json')['support_mass']>0,
                'current_only_nonidentity':read(root/'fit_selection.json')['status']=='FIT_SELECTED',
                'holdout_replication':status=='ACCEPTED_CURRENT_HOLDOUT','development_retention_direction':gain['delta']>0,
                'selectivity_over_global':nonuniform and t2['selected_vs_global']['CI95'][0]>0,
                'formal_unseen_continual_learning_validation':'NOT RUN'}}
        write(root/'global_alpha_development_comparison.json',{'task2':t2['selected_vs_global'],'selection_source':'CURRENT FIT/HOLDOUT ONLY'})
    else:summary={'status':'MECHANICAL_SMOKE_PASS','formal_unseen_continual_learning_validation':'NOT RUN'}
    write(root/'gate_summary.json',summary)
    report='# Med-PRISM v2.3 Task-Aware MS-CRC\n\n'+json.dumps(summary,indent=2)+'\n\n'
    report+='## Atomic support and concentration\n\n'+json.dumps(read(root/'fit_atomic_support.json'),indent=2)+'\n\n'
    report+='## Fit selection and holdout\n\n'+json.dumps(read(root/'fit_selection.json'),indent=2)+'\n\n'+json.dumps(read(root/'holdout_result.json'),indent=2)
    report+='\n\nTask2 dev was already unblinded in v2.2. This is development/mechanism validation, not blind or formal benchmark evidence. CIs resample paired examples, not atoms; patient dependence and small samples limit inference. Current sample caches were removed before old dev. TPM/RCWP/training OFF.\n\nOutput: '+str(root)+'\n'
    (root/'report.md').write_text(report)
    if not smoke:
        (REPO/'docs/med_prism_v2_3_task_aware_ms_crc/IMPLEMENTATION_AND_GATE_REPORT.md').write_text(report)
    write(root/'completion.json',{'status':'PASS','gate_status':summary['status'],'elapsed_seconds':time.time()-started,'report':str(root/'report.md')})
    print('COMPLETE '+str(root),flush=True)


def run(root,old,smoke,resume=False):
    started=time.time()
    if not resume:prepare(root,old,smoke)
    else:
        req=verify(root)
        if req['smoke']!=smoke or req['v22_root']!=str(old):raise ValueError('Resume request mismatch')
        if (root/'completion.json').exists():print('Already complete '+str(root));return
        if (root/'selection_lock.json').exists():
            verify(root,True);clear_cache(root)
            if not smoke:
                phase(root,'development',[{'label':'task2','task':2,'output':'task2_development_metrics.json'},
                    {'label':'task1','task':1,'output':'task1_development_metrics.json'}])
            finish(root,read(root/'holdout_result.json')['status'],started);return
    vectors=read(root/'candidate_manifest.json')
    if not smoke:
        phase(root,'metric',[job(k,g,prefix='metric',tag='dev') for k,g in vectors.items()])
        checks=[read(root/f'metric/dev_{k}.json') for k in vectors]
        write(root/'metric_consistency.json',{'status':'PASS','all_v22_unique_vectors_reproduced':True,'checks':checks,
            'TP_FP_FN_verified_against_fresh_official_evaluator':True,'v22_counts_not_saved':True})
    else:write(root/'metric_consistency.json',{'status':'SMOKE_ONLY','full_v22_panel_reproduction':'PENDING_FORMAL_GATE0'})
    jobs=references('fit')
    if smoke:jobs.append(job('probe',(.5,)+(1.,)*15))
    else:jobs.extend(job(k,g) for k,g in vectors.items() if k!='full')
    phase(root,'fit',jobs)
    S=support(root,'fit');save_support(root,'fit',S)
    if smoke:
        selected='probe';g=(.5,)+(1.,)*15;selection_status='SMOKE_ONLY';scalar='full';scalar_g=IDENTITY
        metrics={k:S.evaluate(rows(root,'fit',k)) for k in ('full','probe')}
    else:
        metrics={k:S.evaluate(rows(root,'fit',k)) for k in vectors}
        selected,selection_status=select(S,metrics,vectors);g=vectors[selected]
        global_vectors={k:v for k,v in vectors.items() if len(set(v))==1}
        scalar,_=select(S,metrics,global_vectors);scalar_g=vectors[scalar]
    write(root/'fit_candidates.json',metrics)
    write(root/'fit_selection.json',{'label':selected,'g_fit':g,'status':selection_status,'current_only':True})
    write(root/'global_alpha_current_selection.json',{'label':scalar,'g_fit':scalar_g,'selection':'same current-only lex objective and feasibility'})
    # Only locked selected/scalar hypotheses; never evaluate second place.
    phase(root,'holdout',references('holdout')+[job('selected',g,tag='holdout'),job('global',scalar_g,tag='holdout')])
    T=support(root,'holdout');save_support(root,'holdout',T)
    candidate=T.evaluate(rows(root,'holdout','selected'));glob=T.evaluate(rows(root,'holdout','global'))
    accepted,status=accept(T,candidate,g);scalar_ok,scalar_status=accept(T,glob,scalar_g)
    if tuple(g)==IDENTITY:status=selection_status
    final=list(g) if accepted or smoke else list(IDENTITY);global_final=list(scalar_g) if scalar_ok else list(IDENTITY)
    write(root/'holdout_result.json',{'status':status,'g_fit':g,'final_g':final,'candidate':candidate,'baseline':T.baseline,
        'global_status':scalar_status,'global_candidate':glob,'global_final':global_final,'no_reranking':True})
    phase(root,'export',[job('selected',final,prefix='.',tag='reload')])
    # Atomic metrics remain split-specific; no inappropriate dev metric used for selection.
    write(root/'task3_current_metrics.json',{'fit':metrics[selected],'holdout_candidate':candidate,'holdout_full':T.baseline,
        'holdout_deployed':candidate if final==list(g) else T.baseline,'metric':'official concept micro-F1',
        'status':'SMOKE_ONLY' if smoke else status})
    lock(root,{'full':list(IDENTITY),'H':[0.]*16,'selected':final,'global':global_final})
    clear_cache(root)
    del S,T  # discard current per-sample atoms before old-development workers
    if not smoke:
        write(root/'old_dev_unlock.json',{'SELECTION_LOCKED':True,'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        phase(root,'development',[{'label':'task2','task':2,'output':'task2_development_metrics.json'},
            {'label':'task1','task':1,'output':'task1_development_metrics.json'}])
    finish(root,status,started)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);p.add_argument('--v22-root',default=DEFAULT_V22);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();root=Path(a.output_root).resolve()
    if root.exists() and not a.resume:p.error('Fresh output required, or explicit --resume after stopping prior workers')
    try:run(root,Path(a.v22_root).resolve(),a.smoke,a.resume)
    except Exception as exc:
        import traceback
        if root.exists():write(root/'failure.json',{'status':'FAIL','error':str(exc),'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
