"""Task3-only selection -> one holdout decision -> locked fixed dev panel."""
import gc
import json
from pathlib import Path
import time
import torch
from .composition import Composition
from .data import current_split
from .evaluator import Evaluator,equivalent,legacy,contract
from .search import IDENTITY,witness,search,accept_holdout,summarize,panel
from .fold import export
from .analysis import describe,conclude
from med_prism.transport.cli import load_calibration_model,TASKS
from med_prism.transport.checkpoint import read_state,sha256_file
from med_prism.transport.banks import fingerprint
from med_prism.transport.diagnostics import write_json


def assert_fingerprints(before,after,allowed=()):
    if before.keys()!=after.keys():raise RuntimeError('Persistent tensor schema changed')
    changed=[]
    for k,v in before.items():
        if v!=after[k]:
            if k not in allowed:raise RuntimeError('Forbidden persistent tensor change: '+k)
            changed.append(k)
    return {'status':'PASS','changed_current_B':changed,'tensor_count':len(before),
        'base_shared_all_A_historical_banks_unchanged':True}


def lock(root,source,deployed,g,config_hashes,split):
    selected=root/'selected_coefficients.json'
    if selected.exists():raise FileExistsError('Coefficient lock cannot be overwritten')
    write_json(selected,{'method_version':'2.2','g':list(g)})
    private=Path(deployed.private_manifests[-1]);header=json.loads(private.read_text())
    hashes={str(selected):sha256_file(selected),source.origin:sha256_file(source.origin),
        deployed.origin:sha256_file(deployed.origin),str(private):sha256_file(private),
        str(private.parent/header['weights_file']):sha256_file(private.parent/header['weights_file']),**config_hashes}
    value={'SELECTION_LOCKED':True,'locked_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'hashes':hashes,'selected_coefficients_sha256':hashes[str(selected)],
        'source_checkpoint_sha256':hashes[source.origin],'exported_checkpoint_sha256':hashes[deployed.origin],
        'algorithm_config_sha256':sha256_file(root/'request.json'),'current_split_sha256':split}
    write_json(root/'selection_lock.json',value)
    return value


def validate_lock(root):
    value=json.loads((Path(root)/'selection_lock.json').read_text())
    if value.get('SELECTION_LOCKED') is not True:raise PermissionError('Old evaluation not unlocked')
    for p,h in value['hashes'].items():
        if sha256_file(p)!=h:raise RuntimeError('Locked artifact changed: '+p)
    return value


def locked_dev(root,path):
    validate_lock(root)  # MUST execute before path read or dataset hashing.
    path=Path(path)
    if path.name!='development.jsonl' or path.parent.name not in TASKS.values():raise ValueError('Only fixed diagnostic dev panel allowed')
    return legacy.read_jsonl(path,0)


def witness_stats(W,n):
    sets=[set(v['indices']) for v in W.values()]
    return {'banks':{k:{'count':v['count'],'strict_count':v['strict_count'],'fraction':v['count']/n} for k,v in W.items()},
        'overlap':len(sets[0]&sets[1]),'union_fraction':len(set.union(*sets))/n,'samples':n}


def render(root,result):
    lines=['# Med-PRISM v2.2 MS-CRC finite-forward gate','',
        'Implemented: isolated boundary hook/fold/search/export/lock/panel. Trainer and A-only loss unchanged; TPM/RCWP OFF.',
        'Mechanically verified: CPU tests plus separate real-model smoke. Completion PASS means engineering completion, NOT a method-performance PASS.',
        f"Gate status: **{result['status']}**",'',
        'Task3 binary correctness is exact normalized concept-set match (existing parser all_fp=all_fn=0). Official concept micro-F1 remains separately reported. Never interpret micro-F1 as accuracy.',
        'Fixed Task3 train128+64 only; no data expansion; P2 requires32/16 witnesses. Shortlist uses only feasible, strictly improving single-OFF fit candidates; coordinate search uses at most2 sweeps. Holdout requires unchanged-or-better accuracy, CE<=full+.02, no lost N, and strictly improved lexicographic objective for nonidentity.',
        'Teacher-forced KL uses H=base+S3+P1+P2 as reference, NOT the Task2 boundary teacher with S2. S2 was not loaded. KL is diagnostic only.',
        'Same coefficient vectors in the fixed panel are exact aliases evaluated once per split; all distinct vectors use new full-network forward/generation. No linear sum of leave-one-out effects is used.',
        '', '## Key gate quantities',json.dumps(result,indent=2),
        '', '## Answers',
        '1. Current-only witness selection supports lower Task2 forgetting only if Task2 gain and its paired uncertainty meet the pre-registered checks; see gate_summary and finite_panel_task2.',
        '2. Task3 exact-set accuracy, official micro-F1, answer-token CE and positive-transfer losses are reported independently; empty N is NOT evidence of preserved transfer.',
        '3. Rank-group value is supported only if the selected vector beats a current-performance-matched global alpha and the fixed multiset permutation with adequate precision. NO_GO_SELECTIVITY/INCONCLUSIVE is not a success.',
        'The fixed-panel oracle envelope is diagnostic only and never changes the exported vector. If a current/safety-feasible oracle is good but locked g is poor, the cross-distribution selection signal failed.',
        'Paired bootstrap is by example (2000 draws, seed2209); correlated patients/groups and diagnostic-dev exposure limit generalization. No formal unseen benchmark or fresh continual-learning training was run.',
        '', 'Files: witness_summary.json, search_trace.jsonl, fit_summary.json, holdout_summary.json, selection_lock.json, selected_coefficients.json, exported/state.json, finite_panel_current.json, finite_panel_task2.json, finite_panel_task1.json, global_scaling_comparison.json, random_permutation_comparison.json, gate_summary.json, invariants.json.']
    (root/'report.md').write_text('\n\n'.join(lines)+'\n')
    report_path=Path(__file__).resolve().parents[2]/'docs/med_prism_v2_2_ms_crc/IMPLEMENTATION_AND_GATE_REPORT.md'
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text('\n\n'.join(lines)+f'\n\nResult root: `{root}`\n')


def run(args):
    root=Path(args.output_root).resolve()
    if root.exists():raise FileExistsError(root)
    source=read_state(args.state)
    if source.task_id!=3 or json.loads(Path(source.origin).read_text())['status']!='PRE_TPM':raise ValueError('Only Task3 PRE_TPM state')
    data=Path(args.data_root).resolve();current=data/TASKS[3]/'train.jsonl'
    original=source.tensors()
    files=[Path(source.origin),current,*map(Path,source.components())]
    for p in source.components():
        header=json.loads(Path(p).read_text());files.append(Path(p).parent/header['weights_file'])
    code=[*Path(__file__).parent.glob('*.py'),Path(legacy.__file__),Path(__file__).resolve().parents[2]/'scripts/medicalskill_v1_2_med_prism/evaluate_formal_v1_2_cell.py']
    source_hashes={str(p):sha256_file(p) for p in files};code_hashes={str(p):sha256_file(p) for p in code}
    root.mkdir(parents=True);started=time.time();torch.manual_seed(42);torch.set_num_threads(1)
    config={'method_version':'2.2','tpm_enabled':False,'rcwp_enabled':False,'optimizer_steps':0,
        'fit':2 if args.smoke else 128,'holdout':1 if args.smoke else 64,'max_changed_groups':4,
        'coefficient_values':[0,.5,1],'CE_tolerance':.02,'bootstrap_draws':2000,'bootstrap_seed':2209,
        'grouping':'4*(layer_zero_based//9)+(local_rank_zero_based//4)','eta':'max repeated H margin difference',
        'Task3_C':'existing concept parser, all_fp==0 and all_fn==0; official micro-F1 separate',
        'shortlist':'up to4 feasible strictly improving single-OFF; lexicographic then group id',
        'holdout_rule':'single strict lexicographic acceptance with fit plasticity constraints; otherwise identity',
        'witness_coverage':{'fit':32,'holdout':16,'P2_required':True},
        'gate_thresholds':{'task2_gain':.03,'task1_max_drop':.01,'task3_exact_and_micro_f1_max_drop':.01,
            'current_positive_transfer_lost':0,'matched_current_accuracy_and_f1_distance':.01,
            'matched_current_CE_distance':.02,'confidence_level':.95},
        'generation':contract.GENERATION_CONFIG,'image_settings':contract.IMAGE_PREPROCESSING,'smoke':args.smoke}
    write_json(root/'request.json',config)
    write_json(root/'provenance.json',{'source_state':source.origin,'manifests':source.components(),
        'source_hashes':source_hashes,'code_hashes':code_hashes,'old_data_read_before_lock':False})
    print('PRECHECK PASS: source verified; loading frozen model; no old data opened',flush=True)
    model,template=load_calibration_model(source,1024)
    before=fingerprint(model);comp=Composition(model);scorer=Evaluator(model,template,comp)
    write_json(root/'group_map.json',comp.group_map)
    fit,holdout,split=current_split(current,template,config['fit'],config['holdout']);write_json(root/'current_split.json',split)
    def score(records,g=IDENTITY,active=(1,2,3)):
        return scorer.score(records,g,active)[0]
    H=score(fit,active=(1,2));repeat=score(fit,active=(1,2))
    eta=max(abs(a['margin']-b['margin']) for a,b in zip(H,repeat))
    reload_tol=max(eta,max(abs(a['nll_sum']-b['nll_sum']) for a,b in zip(H,repeat)))
    equivalent(H,repeat,reload_tol)
    ablations={1:score(fit,active=(2,)),2:score(fit,active=(1,))}
    full=score(fit);W=witness(H,ablations,eta)
    write_json(root/'witness_summary.json',{'eta_num':eta,'fit':witness_stats(W,len(fit))})
    with (root/'search_trace.jsonl').open('w') as stream:
        def trace(value):stream.write(json.dumps(value)+'\n');stream.flush()
        if args.smoke:
            chosen=(.5,)+(1.,)*15;info={'status':'SMOKE_ONLY','protected_tasks':[],'shortlist':[0]}
            probe=score(holdout,chosen);trace({'stage':'SMOKE_ONLY','g':chosen,'full_forward':True})
        else:chosen,info=search(lambda g:score(fit,g),H,full,W,trace)
    fitted=full if tuple(chosen)==IDENTITY else score(fit,chosen)
    write_json(root/'fit_summary.json',{'g_fit':chosen,**info,'full':summarize(full,full,H,W,IDENTITY,info['protected_tasks']),
        'selected':summarize(fitted,full,H,W,chosen,info['protected_tasks'])})
    # g_fit is fixed before any holdout forward. No candidate adaptation follows.
    if args.smoke:
        selected=chosen;status='SMOKE_ONLY';expected=probe
        write_json(root/'holdout_summary.json',{'status':status,'not_an_empirical_gate':True})
    else:
        HH=score(holdout,active=(1,2));aa={1:score(holdout,active=(2,)),2:score(holdout,active=(1,))}
        ff=score(holdout);WW=witness(HH,aa,eta);ss=score(holdout,chosen)
        selected,status=accept_holdout(chosen,ss,ff,HH,WW,info['protected_tasks'])
        if info['status']=='NO_SIGNAL':selected,status=IDENTITY,'NO_SIGNAL'
        expected=ss if selected==tuple(chosen) else ff
        write_json(root/'holdout_summary.json',{'status':status,'g_fit':chosen,'final_g':selected,
            'candidate':summarize(ss,ff,HH,WW,chosen,[t for t in info['protected_tasks'] if WW[str(t)]['count']>0]),
            'witnesses':witness_stats(WW,len(holdout))})
        write_json(root/'witness_summary.json',{'eta_num':eta,'fit':witness_stats(W,len(fit)),
            'holdout':witness_stats(WW,len(holdout))})
    candidate_invariants=assert_fingerprints(before,fingerprint(model))
    values=comp.exported_tensors(original,selected)
    deployed=export(source,values,root/'exported',selected)
    # Release the complete old model before real checkpoint reload; no dual 8B copy.
    del scorer,comp,template,model;gc.collect();torch.cuda.empty_cache()
    model,template=load_calibration_model(deployed,1024)
    loaded=model.state_dict()
    for k,v in values.items():
        if not torch.equal(loaded[k].cpu(),v):raise RuntimeError('Reloaded adapter differs '+k)
    del loaded
    native_comp=Composition(model);scorer=Evaluator(model,template,native_comp)
    actual=scorer.score(holdout,IDENTITY)[0]
    equivalent(expected,actual,reload_tol)
    write_json(root/'reload_equivalence.json',{'status':'PASS','numerical_tolerance':reload_tol,
        'predictions_correctness_CE_margin_and_adapter_tensors':True})
    allowed=[k for k in original if not torch.equal(original[k],values[k])]
    deployment_invariants=assert_fingerprints(before,fingerprint(model),allowed)
    locked=lock(root,source,deployed,selected,{**code_hashes,**source_hashes,
        **{str(root/name):sha256_file(root/name) for name in ('request.json','current_split.json','group_map.json',
            'fit_summary.json','holdout_summary.json','witness_summary.json','search_trace.jsonl')}},split)
    validate_lock(root)
    if args.smoke:
        write_json(root/'invariants.json',{'candidate_forwards':candidate_invariants,'export_only_current_B':deployment_invariants})
        write_json(root/'completion.json',{'status':'PASS','scope':'MECHANICAL_SMOKE_ONLY','no_old_data_read':True,
            'elapsed_seconds':time.time()-started,'export_reload_equivalence':'PASS'})
        print('SMOKE PASS '+str(root),flush=True);return
    # Recomputed hooks remain based on IMMUTABLE ORIGINAL B even on reloaded g*.
    comp=Composition(model,original_state=original);scorer.composition=comp
    native_baseline=fingerprint(model);vectors=panel(selected);panels={};dev_hashes={}
    validate_lock(root)
    write_json(root/'old_eval_unlock.json',{'SELECTION_LOCKED':True,'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    for task in (3,2,1):
        path=data/TASKS[task]/'development.jsonl';rows=locked_dev(root,path)
        if len(rows)!=256:raise ValueError('Fixed panel requires complete256 grouped dev rows per task')
        dev_hashes[str(path)]=sha256_file(path);records=scorer.records(rows);cache={};evaluated={}
        for label,g in vectors.items():
            validate_lock(root)
            if g not in cache:
                scored,metric=scorer.score(records,g,task=task,kl=task in (1,2))
                cache[g]=scored
            evaluated[label]=cache[g]
            print(f'LOCKED PANEL task{task} {label}',flush=True)
        panels[task]=evaluated
        summary_={k:describe(v,evaluated['full'],evaluated['P3_off']) for k,v in evaluated.items()}
        name='current' if task==3 else 'task'+str(task)
        write_json(root/f'finite_panel_{name}.json',{'task':task,'samples':256,'dataset_sha256':dev_hashes[str(path)],
            'KL_reference':'H same S3, NOT S2 teacher','variants':summary_,
            'vectors':vectors,'unique_forward_vectors':len(cache)})
        # No raw old rows, references, full-vocabulary logits or features on disk.
        del rows,records,cache
    result=conclude(panels,status)
    validate_lock(root)
    final=assert_fingerprints(native_baseline,fingerprint(model))
    source.validate()
    for p,h in {**source_hashes,**dev_hashes,**code_hashes}.items():
        if sha256_file(p)!=h:raise RuntimeError('Input/code changed '+p)
    write_json(root/'global_scaling_comparison.json',{k:result[k] for k in ('matched_global','matched_current_close','vs_matched_global')})
    write_json(root/'random_permutation_comparison.json',{'seed':2209,'g':vectors['permutation'],'vs_selected':result['vs_permutation'],
        'uniform_multiset_not_evaluable':len(set(selected))==1})
    write_json(root/'gate_summary.json',result)
    write_json(root/'invariants.json',{'candidate_forwards':candidate_invariants,'export_only_current_B':deployment_invariants,
        'locked_panel_no_parameter_mutations':final,'source_files_unchanged':True})
    write_json(root/'provenance.json',{'source_state':source.origin,'manifests':source.components(),'source_hashes':source_hashes,
        'code_hashes':code_hashes,'dev_hashes_after_lock':dev_hashes,'selection_lock':locked,'old_data_read_before_lock':False})
    render(root,result)
    write_json(root/'completion.json',{'status':'PASS','gate_status':result['status'],'elapsed_seconds':time.time()-started,
        'report':str(root/'report.md'),'no_training_started':True})
    print('COMPLETE '+str(root),flush=True)
