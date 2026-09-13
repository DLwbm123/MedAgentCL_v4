"""Group-disjoint development resampling with the original v2.3 selector."""
from collections import Counter
import hashlib
import math
import numpy as np
from med_prism.transport.calibration import ordered_rows, stable_id
from med_prism.transport.pilot_data import group_aliases
from med_prism.ms_crc.task_aware.behavior import AtomicSupport, select, accept


def grouped_split(rows, seed, fit_count=128, holdout_count=64):
    # Same ordered_rows and transitive alias union as ms_crc.data.current_split.
    # Encoding eligibility is checked on the whole fixed dev pool by the worker.
    rows=ordered_rows(rows,seed);parent=list(range(len(rows)));seen={}
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(rows):
        for a in group_aliases(row):
            if a in seen:parent[root(i)]=root(seen[a])
            seen[a]=i
    fit=[];holdout=[];groups=set()
    for i,row in enumerate(rows):
        if len(fit)<fit_count:fit.append(stable_id(row));groups.add(root(i))
        elif root(i) not in groups:holdout.append(stable_id(row))
        if len(holdout)==holdout_count:break
    if len(fit)!=fit_count or len(holdout)!=holdout_count:raise ValueError('Insufficient group-disjoint dev pool')
    return {'seed':seed,'fit_ids':fit,'holdout_ids':holdout,'fit_count':len(fit),'holdout_count':len(holdout),
        'connected_groups':len({root(i) for i in range(len(rows))}), 'group_disjoint':True,
        'fit_id_sha256':hashlib.sha256('\n'.join(fit).encode()).hexdigest(),
        'holdout_id_sha256':hashlib.sha256('\n'.join(holdout).encode()).hexdigest()}


def select_on_fit(cache, ids, vectors):
    rows={k:[v[i] for i in ids] for k,v in cache.items()}
    support=AtomicSupport(rows['H'],rows['full'],{1:rows['minus1'],2:rows['minus2']})
    scored={k:support.evaluate(rows[k]) for k in vectors}
    chosen,status=select(support,scored,vectors)
    global_vectors={k:v for k,v in vectors.items() if len(set(v))==1}
    scalar,scalar_status=select(support,scored,global_vectors)
    return {'selected':chosen,'status':status,'global':scalar,'global_status':scalar_status,
            'scores':scored,'support':support.diagnostics()}


def evaluate_holdout(cache, ids, locked_fit, vectors):
    # No candidate pool or select() call on holdout. off_07 is a prespecified probe.
    labels=list(dict.fromkeys(('H','full','minus1','minus2',locked_fit['selected'],locked_fit['global'],'off_07')))
    rows={k:[cache[k][i] for i in ids] for k in labels}
    support=AtomicSupport(rows['H'],rows['full'],{1:rows['minus1'],2:rows['minus2']})
    scored={k:support.evaluate(rows[k]) for k in labels if k not in ('H','minus1','minus2')}
    chosen,scalar=locked_fit['selected'],locked_fit['global']
    accepted,status=accept(support,scored[chosen],vectors[chosen])
    ga,gs=accept(support,scored[scalar],vectors[scalar])
    return {'selected':chosen,'selected_score':scored[chosen],'global':scalar,'global_score':scored[scalar],
        'full_score':scored['full'],'off_07_prespecified_probe':scored['off_07'],
        'accepted':accepted,'accept_status':status,'global_accepted':ga,'global_accept_status':gs,
        'support':support.diagnostics(),'no_holdout_reranking':True}


def summarize_repeats(rows, vectors):
    counts=Counter(r['selected'] for r in rows);n=len(rows)
    p=np.array(list(counts.values()),float)/n
    nonid=[r for r in rows if r['selected']!='full']
    gains=np.array([r['holdout_gain'] for r in rows])
    return {'splits':n,'selection_counts':{k:counts[k] for k in vectors},
        'group_suppression_counts':{str(j):sum(vectors[r['selected']][j]<1 for r in rows) for j in range(16)},
        'identity_frequency':counts['full']/n,'selection_entropy_nats':float(-(p*np.log(p)).sum()),
        'normalized_selection_entropy':float(-(p*np.log(p)).sum()/math.log(len(vectors))),
        'top_candidate_frequency':max(counts.values())/n,'top_candidates':sorted(k for k,v in counts.items() if v==max(counts.values())),
        'mean_fit_gain':float(np.mean([r['fit_gain'] for r in rows])),
        'mean_holdout_gain':float(gains.mean()),'holdout_gain_std_descriptive':float(gains.std(ddof=1)),
        'positive_holdout_splits':int((gains>1e-12).sum()),'negative_holdout_splits':int((gains< -1e-12).sum()),
        'sign_consistency':float(np.mean([r['sign_consistent'] for r in rows])),
        'nonidentity_count':len(nonid),
        'nonidentity_sign_consistency':float(np.mean([r['sign_consistent'] for r in nonid])) if nonid else None,
        'mean_holdout_gain_nonidentity':float(np.mean([r['holdout_gain'] for r in nonid])) if nonid else None,
        'mean_selected_minus_global_holdout':float(np.mean([r['selected_minus_global_holdout'] for r in rows])),
        'accepted_count':sum(r['accepted'] for r in rows),
        'off_07_selected_count':counts['off_07'],
        'off_07_positive_holdout_probe_count':sum(r['off_07_holdout_gain']>1e-12 for r in rows),
        'off_07_mean_holdout_gain':float(np.mean([r['off_07_holdout_gain'] for r in rows])),
        'warning':'Development-only cached deterministic forwards. Splits overlap on one 256-example pool, NOT independent replications or a fresh unseen holdout. No holdout reranking.'}
