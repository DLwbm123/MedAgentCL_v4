"""Descriptive 16-group statistics; no model/selector changes."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def csv_write(path, rows):
    # Machine-readable diagnostic output, not an authored spreadsheet workbook.
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: 'unavailable' if v is None else v for k,v in row.items()} for row in rows)


def sign(x):
    return np.where(np.abs(x) <= 1e-12, 0, np.sign(x))


def top_weights(values, k):
    """Fractionally allocate a tied cutoff, avoiding arbitrary top-k claims."""
    a = np.asarray(values, dtype=float)
    cutoff = sorted(a, reverse=True)[k-1]
    above, tied = a > cutoff, a == cutoff
    return above.astype(float) + tied * ((k-int(above.sum())) / int(tied.sum()))


def association(x, y):
    x,y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or len(x) < 3 or not np.isfinite([x,y]).all():
        raise ValueError('Invalid paired vectors')
    constant = np.ptp(x)==0 or np.ptp(y)==0
    p = None if constant else pearsonr(x,y)
    s = None if constant else spearmanr(x,y)
    nonzero = (sign(x)!=0)&(sign(y)!=0)
    return {'n_groups':len(x), 'pearson':None if p is None else float(p.statistic),
            'pearson_p_exploratory':None if p is None else float(p.pvalue),
            'spearman':None if s is None else float(s.statistic),
            'spearman_p_exploratory':None if s is None else float(s.pvalue),
            'sign_agreement_including_zero':float(np.mean(sign(x)==sign(y))),
            'sign_agreement_nonzero':float(np.mean(sign(x[nonzero])==sign(y[nonzero]))) if nonzero.any() else None,
            'nonzero_pairs':int(nonzero.sum()),
            'top_k':{str(k):{'expected_overlap_count_with_random_cutoff_ties':float(top_weights(x,k)@top_weights(y,k)),
                            'chance_expected_count':k*k/len(x)} for k in (3,5) if len(x)>=k},
            'unique_x':len(set(x)), 'unique_y':len(set(y)),
            'warning':'16 dependent ablations on shared examples; ties, multiple exploratory signals, no confirmatory p-values'}


def historical_tables(source, old):
    from med_prism.ms_crc.task_aware.behavior import feasible
    source,old = Path(source),Path(old)
    fit=read(source/'fit_candidates.json');base=fit['full']
    labels=[f'off_{j:02d}' for j in range(16)]
    panels={t:read(old/f'finite_panel_task{t}.json') for t in (1,2)}
    matrix=[];signals=[]
    # This is the actual fit ordering among groups, NOT an invented scalar selector.
    ordered=sorted(labels,key=lambda k:(not feasible(fit[k],base),fit[k]['L_mem'],fit[k]['L_PT'],labels.index(k)))
    for j,k in enumerate(labels):
        row={'group':j,'label':k}
        for t,p in panels.items():
            f,v=p['variants']['full'],p['variants'][k]
            row.update({f'task{t}_full_metric':f['accuracy'],f'task{t}_off_metric':v['accuracy'],
                f'task{t}_delta':v['accuracy']-f['accuracy'],
                f'task{t}_CE_delta':v['CE_nats_per_answer_token']-f['CE_nats_per_answer_token'],
                f'task{t}_gold_margin_delta':v['mean_gold_margin']-f['mean_gold_margin'],
                f'task{t}_KL_to_H_same_S3':v['teacher_forced_KL_reference_H_same_S3'],
                f'task{t}_KL_delta':v['teacher_forced_KL_reference_H_same_S3']-f['teacher_forced_KL_reference_H_same_S3'],
                f'task{t}_correct_to_wrong':v['vs_full']['correct_to_wrong'],
                f'task{t}_wrong_to_correct':v['vs_full']['wrong_to_correct']})
        matrix.append(row)
        v=fit[k]
        sr={**row,'current_task_utility_delta':v['utility']-base['utility'],
            'current_task_CE_delta':v['CE']-base['CE'],'L_mem':v['L_mem'],
            'memory_repair_signal':base['L_mem']-v['L_mem'],
            'G':v['G'],'G_full':v['G_full'],'G_delta':v['G']-v['G_full'],
            'G_over_G_full':v['G']/v['G_full'] if v['G_full'] else None,
            **{key:v[key] for key in ('L_PT','D_PT','U_PT','memory_supported_atoms_restored',
                'full_positive_transfer_atoms_lost','new_positive_transfer_atoms_added')},
            'fit_feasible':feasible(v,base),'selector_group_rank':ordered.index(k)+1,
            'group_atomic_support_mass':None,'group_positive_support_atoms':None,'group_FP_suppression_atoms':None}
        for t in ('1','2'):
            sr[f'bank{t}_memory_loss']=v['per_bank_candidate_mem_loss'][t]
            sr[f'bank{t}_memory_repair']=base['per_bank_candidate_mem_loss'][t]-v['per_bank_candidate_mem_loss'][t]
        signals.append(sr)
    # All directional signals have zero=identity; rank has no meaningful sign.
    directions={'current_task_utility_delta':1,'current_task_CE_delta':-1,
        'memory_repair_signal':1,'bank1_memory_repair':1,'bank2_memory_repair':1,
        'G_delta':1,'D_PT':-1,'U_PT':1,'L_PT':-1,'memory_supported_atoms_restored':1,
        'full_positive_transfer_atoms_lost':-1,'new_positive_transfer_atoms_added':1}
    stats={}
    for t in (1,2):
        y=[r[f'task{t}_delta'] for r in matrix]
        stats[f'task{t}']={key:{**association([r[key]*d for r in signals],y),
            'orientation':d,'raw_pearson':association([r[key] for r in signals],y)['pearson'],
            'raw_spearman':association([r[key] for r in signals],y)['spearman']} for key,d in directions.items()}
        rank=association([-r['selector_group_rank'] for r in signals],y)
        for key in ('sign_agreement_including_zero','sign_agreement_nonzero','nonzero_pairs'):rank[key]=None
        rank['meaning']='Feasible groups first, then original lex L_mem/L_PT/manifest order; no sign semantics'
        stats[f'task{t}']['selector_order']=rank
    stats['task1_vs_task2']=association([r['task1_delta'] for r in matrix],[r['task2_delta'] for r in matrix])
    chosen=read(source/'fit_selection.json')['label'];ci=labels.index(chosen)
    stats['locked_selected_group']={'label':chosen,**{f'task{t}_historical_repair_rank_average_ties':
        float(rankdata([-r[f'task{t}_delta'] for r in matrix],method='average')[ci]) for t in (1,2)}}
    R=np.array([r['task2_delta'] for r in matrix]);pos=np.maximum(R,0);p=pos/pos.sum()
    top=sorted(range(16),key=lambda j:(-R[j],j))
    hhi=float(p@p)
    repair={'full':panels[2]['variants']['full']['accuracy'],'H':panels[2]['variants']['P3_off']['accuracy'],
        'H_gain':panels[2]['variants']['P3_off']['accuracy']-panels[2]['variants']['full']['accuracy'],
        'max_single_group_repair':float(max(R)), 'top3_sum_descriptive_NOT_joint':float(R[top[:3]].sum()),
        'positive_groups':int((R>0).sum()),'negative_groups':int((R<0).sum()),'zero_groups':int((R==0).sum()),
        'positive_repair_HHI':hhi,'HHI_normalized_16':(hhi-1/16)/(1-1/16),
        'effective_number_of_positive_groups':1/hhi,'top3_share_of_positive_sum':float(p[top[:3]].sum()),
        'ranked_groups':[{'group':j,'repair':float(R[j])} for j in top],
        'warning':'Single-group effects overlap and are non-additive; neither top3 sum nor HHI decomposes the joint causal effect.'}
    return matrix,signals,stats,repair,panels


def save_historical(root, source, old):
    root=Path(root)
    matrix,signals,stats,repair,panels=historical_tables(source,old)
    write(root/'historical_interference_matrix.json',{'rows':matrix,
        'baselines':{str(t):{k:p['variants'][k] for k in ('full','P3_off')} for t,p in panels.items()},
        'source_panels':{str(t):str(Path(old)/f'finite_panel_task{t}.json') for t in panels}})
    csv_write(root/'historical_interference_matrix.csv',matrix)
    csv_write(root/'group_signal_vs_historical_interference.csv',signals)
    write(root/'signal_correlations.json',stats);write(root/'task2_repair_decomposition.json',repair)
    write(root/'signal_provenance.json',{'fit_source':str(Path(source)/'fit_candidates.json'),
        'atomic_support_global_reference':read(Path(source)/'fit_atomic_support.json'),
        'unavailable':'Support mass / positive support / FP suppression counts describe H vs H-minus-bank, not P3 groups. No per-group attribution is inferred.',
        'selector':'Original feasibility + lex L_mem,L_PT,L1; rank orders infeasible groups last only for descriptive correlation.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def scatter(x,y,xlabel,ylabel,name):
        fig,ax=plt.subplots(figsize=(7,5))
        ax.scatter(x,y,c=range(16),cmap='viridis')
        for j,(a,b) in enumerate(zip(x,y)):ax.annotate(f'{j:02d}',(a,b),xytext=(3,4),textcoords='offset points',fontsize=8)
        ax.axhline(0,color='grey',lw=.6);ax.axvline(0,color='grey',lw=.6)
        ax.set(xlabel=xlabel,ylabel=ylabel);fig.tight_layout();fig.savefig(root/name,dpi=160);plt.close(fig)
    for t in (1,2):
        scatter([r['memory_repair_signal'] for r in signals],[100*r[f'task{t}_delta'] for r in matrix],
            'Current fit memory repair (L_mem full - off)',f'Task{t} accuracy change (pp)',f'selector_vs_task{t}_interference.png')
    scatter([r['task1_delta']*100 for r in matrix],[r['task2_delta']*100 for r in matrix],
        'Task1 accuracy change (pp)','Task2 accuracy change (pp)','task1_vs_task2_interference.png')
    a=np.array([[r[f'task{t}_delta']*100 for r in matrix] for t in (1,2)])
    fig,ax=plt.subplots(figsize=(12,3));im=ax.imshow(a,cmap='RdBu_r',vmin=-abs(a).max(),vmax=abs(a).max(),aspect='auto')
    for t in range(2):
        for j in range(16):ax.text(j,t,f'{a[t,j]:.2f}',ha='center',va='center',fontsize=8)
    ax.set_xticks(range(16),[f'{j:02d}' for j in range(16)]);ax.set_yticks((0,1),('Task1','Task2'))
    ax.set_xlabel('Suppressed P3 group');fig.colorbar(im,ax=ax,label='Accuracy change (pp)')
    fig.tight_layout();fig.savefig(root/'historical_interference_heatmap.png',dpi=160);plt.close(fig)
    return stats,repair
