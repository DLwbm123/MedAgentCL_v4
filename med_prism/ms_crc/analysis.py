"""Locked-panel, paired descriptive inference; no coefficient-selection API."""
import numpy as np
from .search import accuracy,ce,positive_set


def concept_f1(rows):
    a=np.array([[r['all_tp'],r['all_fp'],r['all_fn']] for r in rows]).sum(0)
    return float(2*a[0]/(2*a[0]+a[1]+a[2])) if a.sum() else 0.


def concept_ci(a,b):
    left=np.array([[r['all_tp'],r['all_fp'],r['all_fn']] for r in a]);right=np.array([[r['all_tp'],r['all_fp'],r['all_fn']] for r in b])
    indices=np.random.default_rng(2209).integers(0,len(a),size=(2000,len(a)))
    def f(x):
        total=x[indices].sum(1);den=2*total[:,0]+total[:,1]+total[:,2]
        return np.divide(2*total[:,0],den,out=np.zeros(len(den)),where=den!=0)
    return {'delta':concept_f1(a)-concept_f1(b),'CI95':[float(v) for v in np.quantile(f(left)-f(right),[.025,.975])]}


def paired_ci(a,b):
    d=np.array([int(x['correct'])-int(y['correct']) for x,y in zip(a,b)],dtype=float)
    if [x['sample_id'] for x in a]!=[x['sample_id'] for x in b]:raise ValueError('Bootstrap pairing mismatch')
    rng=np.random.default_rng(2209)
    boot=d[rng.integers(0,len(d),size=(2000,len(d)))].mean(1)
    return {'delta':float(d.mean()),'CI95':[float(v) for v in np.quantile(boot,[.025,.975])],
        'bootstrap':'paired examples, 2000 draws, seed2209; not a patient-level independence guarantee'}


def transitions(reference,actual):
    return {'correct_to_wrong':sum(a['correct'] and not b['correct'] for a,b in zip(reference,actual)),
        'wrong_to_correct':sum(not a['correct'] and b['correct'] for a,b in zip(reference,actual))}


def describe(rows,full,H):
    N=positive_set(H,full)
    result={'n':len(rows),'accuracy':accuracy(rows),'CE_nats_per_answer_token':ce(rows),
        'mean_gold_margin':sum(x['margin'] for x in rows)/len(rows),
        'vs_full':transitions(full,rows),'vs_H_same_S3':transitions(H,rows),
        'accuracy_delta_vs_full':paired_ci(rows,full),
        'positive_transfer_set_size':len(N),'lost_full_positive_transfer':sum(not rows[j]['correct'] for j in N),
        'positive_transfer_evaluable':bool(N)}
    if 'kl_H_same_S3_to_candidate' in rows[0]:
        result['teacher_forced_KL_reference_H_same_S3']=sum(x['kl_H_same_S3_to_candidate'] for x in rows)/len(rows)
    if 'all_tp' in rows[0]:
        tp=sum(x['all_tp'] for x in rows);fp=sum(x['all_fp'] for x in rows);fn=sum(x['all_fn'] for x in rows)
        result['existing_all_concept_micro_f1']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    return result


def conclude(panels,selection_status):
    current,old,safety=panels[3],panels[2],panels[1]
    selected=current['selected']
    globals_=[k for k in current if k.startswith('global_')]
    # Match ONLY using current performance, never old-task accuracy.
    def close_current(k):return (abs(accuracy(current[k])-accuracy(selected))<=.01 and
        abs(ce(current[k])-ce(selected))<=.02 and abs(concept_f1(current[k])-concept_f1(selected))<=.01)
    pool=[k for k in globals_ if close_current(k)]
    match=min(pool or globals_,key=lambda k:(abs(accuracy(current[k])-accuracy(selected)),abs(ce(current[k])-ce(selected)),
        abs(concept_f1(current[k])-concept_f1(selected)),float(k.split('_')[1])))
    close=bool(pool)
    gain=paired_ci(old['selected'],old['full']);task1=paired_ci(safety['selected'],safety['full']);task3=paired_ci(selected,current['full'])
    global_ci=paired_ci(old['selected'],old[match]);random_ci=paired_ci(old['selected'],old['permutation'])
    N=positive_set(current['P3_off'],current['full'])
    old_N=positive_set(old['P3_off'],old['full'])
    lost=sum(not old['selected'][j]['correct'] for j in old_N)
    restored=sum(h['correct'] and not f['correct'] and s['correct'] for h,f,s in zip(old['P3_off'],old['full'],old['selected']))
    micro_f1=concept_ci(selected,current['full'])
    current_N_lost=sum(not selected[j]['correct'] for j in N)
    oracle_eligible=[k for k in old if accuracy(current[k])>=accuracy(current['full'])-.01 and
        concept_f1(current[k])>=concept_f1(current['full'])-.01 and ce(current[k])<=ce(current['full'])+.02 and
        accuracy(safety[k])>=accuracy(safety['full'])-.01]
    oracle=max(oracle_eligible,key=lambda k:(accuracy(old[k]),-abs(accuracy(current[k])-accuracy(current['full']))))
    precision=(gain['CI95'][0]>0 and task1['CI95'][0]>=-.01 and task3['CI95'][0]>=-.01 and micro_f1['CI95'][0]>=-.01)
    if selection_status=='NO_SIGNAL':status='NO_SIGNAL'
    elif selection_status in ('IDENTITY','REJECTED_CURRENT_HOLDOUT'):status=selection_status
    elif gain['delta']<.03 or task1['delta']<-.01 or task3['delta']<-.01 or micro_f1['delta']<-.01 or current_N_lost or (lost and restored<=lost):status='NO_GO_BEHAVIOR'
    elif close and global_ci['delta']<=0:status='NO_GO_SELECTIVITY'
    elif not close or not close_current('permutation') or not precision or global_ci['CI95'][0]<=0 or random_ci['CI95'][0]<=0 or not N:status='INCONCLUSIVE'
    else:status='PASS'
    return {'status':status,'selection_status':selection_status,'task2_gain':gain,'task1_safety':task1,'task3_safety':task3,
        'matched_global':match,'matched_current_close':close,'vs_matched_global':global_ci,'vs_permutation':random_ci,
        'current_positive_transfer_evaluable':bool(N),'current_positive_transfer_lost':current_N_lost,
        'task3_micro_f1_safety':micro_f1,'permutation_current_matched':close_current('permutation'),
        'old_positive_transfer_lost':lost,'old_forgotten_restored':restored,
        'oracle_panel_label':oracle,'oracle_panel_task2_accuracy':accuracy(old[oracle]),
        'oracle_is_NOT_MS_CRC_score':True,'cross_distribution_selection_signal_failed':
            accuracy(old[oracle])>=accuracy(old['full'])+.03 and gain['delta']<.03,
        'precision_note':'Diagnostic dev only; paired CIs do not prove unseen benchmark generalization'}
