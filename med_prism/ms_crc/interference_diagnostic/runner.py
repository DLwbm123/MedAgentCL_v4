"""Isolated v2.3 audit. prepare -> CPU tests -> dual-GPU cache -> offline splits.

No training, repair, checkpoint export, or writes to source result directories.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .statistics import read,write,csv_write,save_historical,sign
from .splits import grouped_split,select_on_fit,evaluate_holdout,summarize_repeats

DEFAULT_SOURCE='/remote-home/wangbomin/MedPRISM_v2_3_TaskAware_MS_CRC_gate_20260910_142249_12471'
REPO=Path(__file__).resolve().parents[3]
PYTHON=sys.executable


def sha(path):
    from med_prism.transport.checkpoint import sha256_file
    return sha256_file(path)


def check_hashes(hashes):
    for p,h in hashes.items():
        if sha(p)!=h:raise RuntimeError('Immutable input changed: '+p)


def verify(root):
    prov=read(root/'provenance.json')
    check_hashes(prov['input_hashes'])
    check_hashes(prov['locked_artifacts_before'])
    check_hashes(prov['diagnostic_code_hashes'])
    for name,h in prov['diagnostic_plan_hashes'].items():
        if sha(root/name)!=h:raise RuntimeError('Diagnostic plan changed: '+name)
    return read(root/'request.json')


def prepare(root, source):
    from med_prism.ms_crc.task_aware.runtime import verify as verify_v23
    from med_prism.ms_crc.gate import validate_lock
    from med_prism.ms_crc.evaluator import contract,legacy
    from med_prism.ms_crc.search import panel,IDENTITY
    from med_prism.transport.checkpoint import read_state
    q=verify_v23(source,True);old=Path(q['v22_root']);validate_lock(old)
    p=read(old/'provenance.json');oldq=read(old/'request.json')
    check_hashes(p['source_hashes']);check_hashes(p['code_hashes']);check_hashes(p['dev_hashes_after_lock'])
    assert q['source_state']==p['source_state']
    # JSON round-trip canonicalizes integer task keys in the Python contract.
    assert q['generation']==oldq['generation']==json.loads(json.dumps(contract.GENERATION_CONFIG))
    assert q['image_settings']==oldq['image_settings']==contract.IMAGE_PREPROCESSING
    assert q['fit']==128 and q['holdout']==64 and not q['smoke']
    assert read(source/'completion.json')['status']=='PASS'
    assert read(source/'gate_summary.json')['status']=='UTILITY_REJECTED'
    assert read(source/'selected_coefficients.json')['g']==list(IDENTITY)
    assert read(source/'invariants.json')['status']=='PASS'
    assert read(source/'metric_consistency.json')['status']=='PASS'
    state=read_state(q['source_state']);assert read(state.origin)['status']=='PRE_TPM'
    vectors=read(source/'candidate_manifest.json');expected={};seen=set()
    for label,g in panel(IDENTITY).items():
        if g not in seen:expected[label]=list(g);seen.add(g)
    assert list(vectors.items())==list(expected.items()) and len(vectors)==25
    for t in (1,2):
        a=read(old/f'finite_panel_task{t}.json');b=read(source/f'task{t}_development_metrics.json')
        assert a['dataset_sha256']==b['dataset_sha256']==q['development_hashes'][str(t)]
        assert a['variants']['full']==b['variants']['full']
        assert a['variants']['P3_off']==b['variants']['H']
        assert {k:tuple(v) for k,v in a['vectors'].items()}==panel(IDENTITY)
        assert all(a['variants'][k]['n']==256 for k in vectors)
    data=Path(q['data_root'])/'task_03_concept_recognition/development.jsonl'
    assert sha(data)==q['current_dev_hash'];rows=legacy.read_jsonl(data,0);assert len(rows)==256
    plans=[grouped_split(rows,seed) for seed in range(2209,2219)]
    # Do not alter original generation batching based on a resample's membership.
    request={'source_v23':str(source),'source_v22':str(old),'source_state':q['source_state'],'current_dev':str(data),
        'current_dev_sha256':sha(data),'current_pool_size':256,'seeds':list(range(2209,2219)),
        'fit':128,'holdout':64,'generation':q['generation'],'image_settings':q['image_settings'],
        'dtype':'BF16','attention':'sdpa','generation_batch_size':4,'max_length':1024,
        'sample_order':'Original development.jsonl order for every composition; original seed-hash order inside split scoring',
        'cache_semantics':'Inference independent of selection; all fixed candidates on fixed dev batches once. Offline group-disjoint resampling. Not independent repeated inference trials.',
        'selector':'Unmodified task_aware.behavior.select; same 25 candidates/order and scalar subset',
        'off07':'Prespecified holdout probe only, never a fallback candidate or reranking input',
        'historical_reuse':'v2.2 16 off + full/H, source/code/data/config hashes verified; v2.3 full/H summaries exactly equal',
        'training':False,'TPM':False,'RCWP':False,'checkpoint_export':False,'formal_unseen_validation':'NOT RUN',
        'gpu_min_free_MiB':22000,'gpus':[0,1],
        'decision_guidance':'CASE B if material cross-task association contrast/task-conditioned vectors; CASE A requires strong signal AND stable selection/generalization; CASE C if no useful association and unstable selection. Exploratory, not a statistical test.'}
    locked={str(x):sha(x) for x in sorted(source.rglob('*')) if x.is_file()}
    input_hashes={**q['input_hashes'],**p['code_hashes'],**p['dev_hashes_after_lock']}
    for name in ('finite_panel_task1.json','finite_panel_task2.json','provenance.json','selection_lock.json','request.json'):
        input_hashes[str(old/name)]=sha(old/name)
    input_hashes.update(read(source/'selection_lock.json')['hashes'])
    root.mkdir(parents=True,exist_ok=False)
    write(root/'request.json',request);write(root/'split_plan.json',plans);write(root/'candidate_manifest.json',vectors)
    jobs=[{'label':k,'g':g,'active':[1,2,3]} for k,g in vectors.items()]
    # H is measured with actual P3 disabled (not guessed from zero-gating equivalence).
    jobs += [{'label':'H','g':list(IDENTITY),'active':[1,2]},
             {'label':'minus1','g':list(IDENTITY),'active':[2]},
             {'label':'minus2','g':list(IDENTITY),'active':[1]}]
    write(root/'jobs.json',jobs)
    import torch,transformers,scipy,platform
    provenance={'created_utc':datetime.now(timezone.utc).isoformat(),'input_hashes':input_hashes,
        'locked_artifacts_before':locked,'source_manifests':state.components(),
        'diagnostic_code_hashes':{str(x):sha(x) for x in sorted(Path(__file__).parent.glob('*.py'))},
        'diagnostic_plan_hashes':{name:sha(root/name) for name in ('request.json','split_plan.json','candidate_manifest.json','jobs.json')},
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,
            'cuda':torch.version.cuda,'scipy':scipy.__version__},
        'raw_historical_data_read':False,'formal_test_read':False,'no_source_writes':True}
    write(root/'provenance.json',provenance)
    save_historical(root,source,old)
    render_report(root,complete=False)
    print('PREPARED '+str(root),flush=True)


def worker(root,index):
    import torch
    from med_prism.ms_crc.task_aware.runtime import concept_score
    from med_prism.ms_crc.evaluator import Evaluator,legacy
    from med_prism.ms_crc.composition import Composition
    from med_prism.ms_crc.gate import assert_fingerprints
    from med_prism.transport.cli import load_calibration_model
    from med_prism.transport.checkpoint import read_state
    from med_prism.transport.banks import fingerprint
    req=verify(root);jobs=read(root/'jobs.json')[index::2]
    torch.manual_seed(42);torch.set_num_threads(1)
    source=read_state(req['source_state']);model,template=load_calibration_model(source,1024)
    before=fingerprint(model);comp=Composition(model);scorer=Evaluator(model,template,comp)
    if comp.group_map!=read(Path(req['source_v23'])/'group_map.json'):raise RuntimeError('Loaded group geometry differs')
    rows=legacy.read_jsonl(Path(req['current_dev']),0);records=scorer.records(rows)
    if len(records)!=256:raise RuntimeError('No data-size reduction allowed')
    # Full-pool encode must pass; do not silently omit long examples or change masks.
    import hashlib
    encoded=hashlib.sha256()
    for r in records:
        encoded.update(str(r['row']['id']).encode())
        for key in ('input_ids','labels','attention_mask'):encoded.update(r['batch'][key].contiguous().numpy().tobytes())
    write(root/f'worker{index}_loaded.json',{'status':'RUNNING','pid':os.getpid(),'gpu':index,
        'source_manifests':source.components(),'default_active_banks':{n:list(w.active_tasks) for n,w in comp.modules.items()},
        'group_count':16,'P3_experts':len(comp.entries),'persistent_tensors':len(before),
        'encoded_pool_sha256':encoded.hexdigest(),'samples':256,'jobs':[j['label'] for j in jobs]})
    targets={v['label']:v['metric'] for v in read(Path(req['source_v23'])/'metric_consistency.json')['checks']}
    for job in jobs:
        label=job['label'];dest=root/'cache'/f'{label}.json'
        if dest.exists():
            if sha(dest)!=read(str(dest)+'.sha256.json')['sha256']:raise RuntimeError('Cache changed '+label)
            print('REUSE '+label,flush=True);continue
        started=time.time();print(f'START GPU{index} {label} samples=256',flush=True)
        scored,metric=concept_score(scorer,records,tuple(job['g']),tuple(job['active']))
        # Independent aggregate reproduction against v2.3 metric gate on same dev.
        if label in targets:
            for k in ('TP','FP','FN'):assert metric[k]==targets[label][k],(label,k,metric,targets[label])
            assert abs(metric['utility']-targets[label]['utility'])<=1e-12
            assert abs(metric['CE']-targets[label]['CE'])<=1e-10
        result={**job,'rows':scored,'metric':metric,'elapsed_seconds':time.time()-started,
            'current_dev_sha256':req['current_dev_sha256'],'original_metric_reproduced':label in targets}
        write(dest,result);write(str(dest)+'.sha256.json',{'sha256':sha(dest)})
        print(f'DONE GPU{index} {label} seconds={result["elapsed_seconds"]:.1f}',flush=True)
    invariant=assert_fingerprints(before,fingerprint(model))
    verify(root)
    write(root/f'worker{index}_done.json',{'status':'PASS','jobs':len(jobs),'invariants':invariant})


def aggregate(root):
    req=verify(root)
    for i in (0,1):assert read(root/f'worker{i}_done.json')['status']=='PASS'
    assert read(root/'worker0_loaded.json')['encoded_pool_sha256']==read(root/'worker1_loaded.json')['encoded_pool_sha256']
    vectors=read(root/'candidate_manifest.json');cache={};checks=[]
    for job in read(root/'jobs.json'):
        path=root/'cache'/f'{job["label"]}.json'
        assert sha(path)==read(str(path)+'.sha256.json')['sha256']
        value=read(path);assert len(value['rows'])==256
        cache[job['label']]={v['sample_id']:v for v in value['rows']}
        assert len(cache[job['label']])==256
        checks.append({k:value[k] for k in ('label','metric','original_metric_reproduced','elapsed_seconds')})
    assert all(list(v)==list(cache['full']) for v in cache.values())
    write(root/'pool_metric_consistency.json',{'status':'PASS','compositions':checks})
    records=[]
    for plan in read(root/'split_plan.json'):
        seed=plan['seed'];dest=root/'splits'/str(seed)
        fit=select_on_fit(cache,plan['fit_ids'],vectors)
        # Persist/hash fit-only decision before evaluating any holdout candidates.
        write(dest/'fit_selection.json',fit);fit_hash=sha(dest/'fit_selection.json')
        hold=evaluate_holdout(cache,plan['holdout_ids'],fit,vectors)
        assert sha(dest/'fit_selection.json')==fit_hash
        write(dest/'holdout.json',hold)
        f=fit['scores'];chosen,scalar=fit['selected'],fit['global']
        fg=f[chosen]['utility']-f['full']['utility'];hg=hold['selected_score']['utility']-hold['full_score']['utility']
        records.append({'seed':seed,'selected':chosen,'fit_status':fit['status'],'global':scalar,
            'fit_full_F1':f['full']['utility'],'fit_selected_F1':f[chosen]['utility'],'fit_gain':fg,
            'holdout_full_F1':hold['full_score']['utility'],'holdout_selected_F1':hold['selected_score']['utility'],'holdout_gain':hg,
            'global_fit_gain':f[scalar]['utility']-f['full']['utility'],
            'global_holdout_gain':hold['global_score']['utility']-hold['full_score']['utility'],
            'selected_minus_global_holdout':hold['selected_score']['utility']-hold['global_score']['utility'],
            'sign_consistent':bool(sign(fg)==sign(hg)),'accepted':hold['accepted'],'accept_status':hold['accept_status'],
            'off_07_fit_gain':f['off_07']['utility']-f['full']['utility'],
            'off_07_holdout_gain':hold['off_07_prespecified_probe']['utility']-hold['full_score']['utility'],
            'fit_selection_sha256':fit_hash,'fit_group_disjoint':plan['group_disjoint'],
            'post_gate_holdout_gain':hg if hold['accepted'] else 0.})
    summary=summarize_repeats(records,vectors)
    write(root/'repeated_split_selection.json',{'summary':summary,'splits':records})
    csv_write(root/'repeated_split_selection.csv',records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,4));counts=summary['selection_counts']
    ax.bar(list(counts),list(counts.values()));ax.tick_params(axis='x',rotation=60)
    ax.set(ylabel='Selections across 10 overlapping dev splits',ylim=(0,10))
    fig.tight_layout();fig.savefig(root/'selection_frequency.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter([r['fit_gain']*100 for r in records],[r['holdout_gain']*100 for r in records])
    for r in records:ax.annotate(f'{r["seed"]}: {r["selected"]}',(r['fit_gain']*100,r['holdout_gain']*100),fontsize=7,xytext=(3,3),textcoords='offset points')
    ax.axhline(0,color='grey');ax.axvline(0,color='grey');ax.set(xlabel='Fit micro-F1 gain (pp)',ylabel='Holdout micro-F1 gain (pp)')
    fig.tight_layout();fig.savefig(root/'fit_vs_holdout_gain.png',dpi=160);plt.close(fig)
    verify(root)
    write(root/'invariants.json',{'status':'PASS','all_v23_artifacts_unchanged':True,'source_and_code_unchanged':True,
        'workers':{str(i):read(root/f'worker{i}_done.json')['invariants'] for i in (0,1)},
        'no_checkpoint_saved':True,'formal_unseen_validation':'NOT RUN'})
    render_report(root,complete=True)
    write(root/'completion.json',{'status':'PASS','completed_utc':datetime.now(timezone.utc).isoformat(),
        'report':str(root/'DIAGNOSTIC_REPORT.md'),'case':read(root/'decision.json')['case'],
        'meaning':'Diagnostic engineering completion, not a method performance PASS'})


def render_report(root,complete):
    req=read(root/'request.json');stats=read(root/'signal_correlations.json');repair=read(root/'task2_repair_decomposition.json')
    matrix=read(root/'historical_interference_matrix.json')['rows'];source=Path(req['source_v23'])
    r1=stats['task1']['memory_repair_signal']['spearman'];r2=stats['task2']['memory_repair_signal']['spearman']
    cross=stats['task1_vs_task2']['spearman'];repeats=read(root/'repeated_split_selection.json') if complete else None
    if not complete:case='PENDING_REPEATED_SPLITS';direction='Do not finalize a v2.4 decision before repeated-split results.'
    elif abs(r2-r1)>=.3 or abs(cross)<.3:
        case='CASE B';direction='Historical-task-aware minimal stored summaries are worth studying; current-only global scores cannot represent task-conditioned interference. No implementation in this audit.'
    elif r2>=.6 and repeats['summary']['top_candidate_frequency']>=.7 and (repeats['summary']['nonidentity_sign_consistency'] or 0)>=.7 and repeats['summary']['mean_holdout_gain']>0:
        case='CASE A';direction='Study cross-fit/shrinkage/continuous gating in a future v2.4, without interpreting development reuse as unseen validation.'
    else:
        case='CASE C';direction='Do not expand current-only binary/pair/triple search. Investigate explicit historical conditioning or stored geometry. No method change now.'
    if complete:write(root/'decision.json',{'case':case,'recommendation':direction,
        'caution':'Exploratory operational categorization, not a significance test. See all effect sizes and split results; thresholds are descriptive report rules, not method hyperparameters.'})
    lines=['# Med-PRISM v2.3 interference diagnostic','',
        '## 1. Executive conclusion','',f'Status: {"COMPLETE" if complete else "PARTIAL: repeated-split GPU inference pending"}. {case}.',
        f'Current fit memory-repair Spearman: Task1 {r1:.4f}; Task2 {r2:.4f}. Task1 vs Task2 repair-vector Spearman {cross:.4f}.',
        'The locked v2.3 result remains UTILITY_REJECTED and deployed identity. Historical diagnostic ablations are not a reranking or deployment decision.',
        '', '## 2. Environment and source provenance','',f'Source v2.3: `{source}`',f'Historical panel: `{req["source_v22"]}`',
        f'Source state: `{req["source_state"]}`',f'Current dev: `{req["current_dev"]}`',
        'See provenance.json for exact hashes, manifests and environment. BF16/SDPA, greedy generation, batch4, original image pixels and parser; answer-token CE unchanged.',
        'Historical full/H aggregates match v2.3 exactly. All historical single-group entries reuse source-equivalent v2.2 dev panel; no new historical inference or raw historical sample cache.',
        '', '## 3. Locked artifacts and invariants','',
        'Every existing v2.3 output file plus locked code/model/data hashes was snapshotted before the run. No source writes, training, TPM, RCWP, checkpoint export, commits, push or remote changes.',
        'Final persistent tensor and file checks: '+('PASS (invariants.json).' if complete else 'Pending worker completion; preflight hashes PASS.'),
        '', '## 4. 16-group historical interference matrix','',
        '| Off group | Task1 delta (pp) | Task2 delta (pp) |','|---|---:|---:|']
    lines += [f'| {r["label"]} | {r["task1_delta"]*100:+.4f} | {r["task2_delta"]*100:+.4f} |' for r in matrix]
    lines += ['', 'JSON/CSV include full/H, CE, prediction transitions, gold margin and KL to SAME-S3 H (not an S2 teacher).',
        '', '## 5. Task2 repair decomposition','',
        f'Full {repair["full"]*100:.4f}%; H {repair["H"]*100:.4f}%; gap {repair["H_gain"]*100:.4f} pp. Max single-group repair {repair["max_single_group_repair"]*100:.4f} pp.',
        f'Positive/negative/zero groups: {repair["positive_groups"]}/{repair["negative_groups"]}/{repair["zero_groups"]}. Positive HHI {repair["positive_repair_HHI"]:.4f}; effective groups {repair["effective_number_of_positive_groups"]:.2f}.',
        f'Top3 groups {[v["group"] for v in repair["ranked_groups"][:3]]}; descriptive top3 sum {repair["top3_sum_descriptive_NOT_joint"]*100:.4f} pp. This is NOT their joint effect. No new combination search.',
        '', '## 6. Current selector signals','',
        'group_signal_vs_historical_interference.csv uses the LOCKED ORIGINAL Task3 fit, not the new diagnostic dev pool. Includes utility/CE deltas, L_mem, bank losses, G/G_full, D/U/L_PT and restored/lost/added atoms.',
        'Atomic support mass and positive/FP support counts are H-vs-historical-bank reference quantities, not P3-group quantities: per-group attribution is unavailable. Original pooled/per-bank values remain in signal_provenance.json.',
        '', '## 7. Signal versus historical interference','',
        '| Signal (higher = more predicted repair) | Task1 Spearman | Task2 Spearman |','|---|---:|---:|']
    for key in stats['task1']:
        a,b=stats['task1'][key]['spearman'],stats['task2'][key]['spearman']
        lines.append(f'| {key} | {a:.4f} | {b:.4f} |')
    lines += ['', 'Loss/CE signals use negative direction or improvement from full; raw correlations also saved. Sign agreement uses identity-referenced deltas. Selector order has no sign interpretation.',
        'Top3/top5 overlap uses fractional cutoff ties (expected overlap under independent random tie resolution). Complete Pearson, Spearman, exploratory p-values, zero-inclusive/nonzero sign agreement and overlaps: signal_correlations.json.',
        f'Original selector off_07 historical ranks (average ties): {stats["locked_selected_group"]}.',
        'Only 16 related ablations, same examples and multiple signals: correlations and p-values are exploratory. Task1 deltas are strongly discretized, so a correlation is not evidence of a large Task1 effect.',
        '', '## 8. Task1 versus Task2 similarity','',
        f'Spearman {cross:.4f}; Pearson {stats["task1_vs_task2"]["pearson"]:.4f}. Task1 suppression has tiny accuracy changes whereas Task2 has substantial recoverable harm. Correlation alone does not establish equality or inequality of task mechanisms.',
        '', '## 9. Repeated split stability','',
        '10 seeds 2209..2218; each fit128/holdout64, transitive grouping aliases and original seed-hash ordering. Same fixed dev256 pool; same original 25 candidates and three H references. No candidate search expansion.',
        'Generate every composition once in original file order and batch4, then resample cached per-example official atoms/CE. Predictions are fixed independently of split selection; this is cached-forward resampling, not a claim of exact numerical equivalence to rebatched generation.',
        'Splits overlap: not independent trials or a fresh unseen benchmark. Fit atoms/q/support are rebuilt using FIT ONLY. Holdout support is built separately after the fit decision is saved; only selected, scalar, identity and prespecified off_07 are scored on holdout.',
        json.dumps(repeats['summary'],indent=2) if complete else 'PENDING. No stability or CASE conclusion is claimed yet.',
        '', '## 10. Fit-to-holdout generalization','',
        'repeated_split_selection.csv separates pre-gate selected gain, scalar gain, and post-gate fallback-to-identity gain. Acceptance failure is never hidden by reporting only the fallback.',
        '', '## 11. off_07 reproducibility','',
        'Original fit F1 0.304453 -> 0.327600; original holdout 0.315113 -> 0.304132. That original reversal is unchanged.',
        (f'New diagnostic splits select off_07 {repeats["summary"]["off_07_selected_count"]}/10 times; its prespecified holdout probe improves in {repeats["summary"]["off_07_positive_holdout_probe_count"]}/10. Mean probe gain {repeats["summary"]["off_07_mean_holdout_gain"]*100:+.4f} pp.' if complete else 'New off_07 repeated-split reproducibility pending.'),
        '', '## 12. CASE decision','',case,
        'The report uses transparent descriptive decision rules recorded in runner.py/request.json, not a fitted classifier or a formal hypothesis test. CASE B takes priority for material task-specific contrasts; repeated-split weaknesses still limit current-only use.',
        '', '## 13. Recommended next direction','',direction,
        '', '## 14. Scope','', 'formal unseen continual-learning validation was NOT run',
        'No v2.4 implementation, new training, new repair, formal test access, or accepted-checkpoint changes.',
        '', '## Execution and artifacts','',
        f'Command: `{PYTHON} -u -m med_prism.ms_crc.interference_diagnostic.runner run --output-root {root}`',
        'Key outputs: historical_interference_matrix.json/.csv, group_signal_vs_historical_interference.csv, signal_correlations.json, task2_repair_decomposition.json, repeated_split_selection.json/.csv, six PNG plots, provenance.json, invariants.json, completion.json.']
    (root/'DIAGNOSTIC_REPORT.md').write_text('\n'.join(lines)+'\n')


def run(root):
    req=verify(root)
    if (root/'completion.json').exists():print('Already complete');return
    # One controller per output directory. Existing user GPU jobs are never killed.
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        children=[]
        values=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
        pending=[i for i in (0,1) if not (root/f'worker{i}_done.json').exists()]
        for index in pending:
            if int(values[index])<req['gpu_min_free_MiB']:
                raise RuntimeError(f'GPU{index} lacks safe free memory; resume after it frees. No worker started; other jobs untouched.')
        for index in (0,1):
            if (root/f'worker{index}_done.json').exists():continue
            log=(root/f'gpu{index}.log').open('a')
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(index),'PYTHONPATH':str(REPO),'PYTHONDONTWRITEBYTECODE':'1'}
            process=subprocess.Popen([PYTHON,'-u','-m','med_prism.ms_crc.interference_diagnostic.runner','worker',
                '--output-root',str(root),'--index',str(index)],env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append((index,process,log));write(root/f'gpu{index}_pid.json',{'pid':process.pid,'gpu':index})
        failures=[]
        for index,process,log in children:
            code=process.wait();log.close()
            if code:failures.append({'gpu':index,'exit_code':code})
        if failures:raise RuntimeError(f'Workers failed {failures}; valid caches retained for explicit resume')
        aggregate(root);print('COMPLETE '+str(root),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','run','worker','aggregate'))
    p.add_argument('--output-root',required=True);p.add_argument('--source',default=DEFAULT_SOURCE);p.add_argument('--index',type=int,choices=(0,1))
    a=p.parse_args();root=Path(a.output_root).resolve();source=Path(a.source).resolve()
    if root.parent!=Path('/remote-home/wangbomin') or not root.name.startswith('MedPRISM_v2_3_interference_diagnostic_'):
        p.error('Only a new isolated diagnostic directory under /remote-home/wangbomin is allowed')
    if root==source or source in root.parents:p.error('Source output writes forbidden')
    try:
        if a.mode=='prepare':prepare(root,source)
        elif a.mode=='run':run(root)
        elif a.mode=='worker':
            if a.index is None:p.error('--index required')
            worker(root,a.index)
        else:aggregate(root)
    except Exception as exc:
        if root.exists():
            name=f'worker{a.index}_failure.json' if a.mode=='worker' else 'failure.json'
            write(root/name,{'status':'FAIL','mode':a.mode,'error':str(exc),'traceback':traceback.format_exc(),
                'utc':datetime.now(timezone.utc).isoformat()})
        raise


if __name__=='__main__':main()
