"""Current-only lexicographic search. No dataset paths, teachers or old-dev API."""
import random

IDENTITY=(1.,)*16


def witness(H,ablations,eta):
    if eta<0:raise ValueError('Negative numerical tolerance')
    result={}
    for task,other in ablations.items():
        if [r['sample_id'] for r in H]!=[r['sample_id'] for r in other]:raise ValueError('Sample order mismatch')
        indices=tuple(j for j,(a,b) in enumerate(zip(H,other)) if a['correct'] and a['margin']-b['margin']>eta)
        strict=tuple(j for j,(a,b) in enumerate(zip(H,other)) if a['correct'] and not b['correct'])
        result[str(task)]={'indices':indices,'strict_indices':strict,'count':len(indices),'strict_count':len(strict)}
    return result


def positive_set(H,full):return tuple(j for j,(a,b) in enumerate(zip(H,full)) if not a['correct'] and b['correct'])
def accuracy(rows):return sum(r['correct'] for r in rows)/len(rows)
def ce(rows):return sum(r['nll_sum'] for r in rows)/sum(r['answer_tokens'] for r in rows)


def objective(rows,H,W,g,tasks):
    error=[];regret=[]
    for task in tasks:
        indices=W[str(task)]['indices']
        if not indices:raise ValueError('Empty protected witness set')
        error.append(sum(not rows[j]['correct'] for j in indices)/len(indices))
        regret.append(sum(max(H[j]['margin']-rows[j]['margin'],0) for j in indices)/len(indices))
    return (sum(error)/len(error),sum(regret)/len(regret),sum(abs(1-v) for v in g))


def feasible(rows,full,N,g):
    if tuple(g)==IDENTITY:return True
    return ce(rows)<=ce(full)+.02 and accuracy(rows)>=accuracy(full) and not any(not rows[j]['correct'] for j in N)


def summarize(rows,full,H,W,g,tasks):
    N=positive_set(H,full)
    return {'accuracy':accuracy(rows),'CE_nats_per_answer_token':ce(rows),
        'positive_transfer_count':len(N),'lost_positive_transfer':sum(not rows[j]['correct'] for j in N),
        'positive_transfer_evaluable':bool(N),'objective':objective(rows,H,W,g,tasks) if tasks else None,
        'feasible':feasible(rows,full,N,g)}


def search(score,H,full,W,trace):
    """No holdout inputs allowed. score(g) always runs the current actual network."""
    tasks=tuple(int(t) for t,v in W.items() if v['count']>=32)
    if 2 not in tasks:return IDENTITY,{'status':'NO_SIGNAL','protected_tasks':tasks,'shortlist':[]}
    N=positive_set(H,full);current=IDENTITY;best=objective(full,H,W,current,tasks)
    ranked=[]
    for gid in range(16):
        g=list(IDENTITY);g[gid]=0.;g=tuple(g)
        rows=score(g);obj=objective(rows,H,W,g,tasks);ok=feasible(rows,full,N,g)
        trace({'stage':'A','group':gid,'g':g,**summarize(rows,full,H,W,g,tasks)})
        if ok and obj<best:ranked.append((obj,gid))
    shortlist=[gid for _,gid in sorted(ranked)[:4]]
    for sweep in range(2):
        changed=False
        for gid in shortlist:
            selected=current;selected_obj=best
            for value in (0.,.5,1.):
                candidate=list(current);candidate[gid]=value;candidate=tuple(candidate)
                rows=score(candidate);obj=objective(rows,H,W,candidate,tasks)
                trace({'stage':'B','sweep':sweep,'group':gid,'g':candidate,**summarize(rows,full,H,W,candidate,tasks)})
                if feasible(rows,full,N,candidate) and obj<selected_obj:selected,selected_obj=candidate,obj
            if selected!=current:changed=True
            current,best=selected,selected_obj
        if not changed:break
    return current,{'status':'IDENTITY' if current==IDENTITY else 'FIT_SELECTED','protected_tasks':tasks,'shortlist':shortlist}


def accept_holdout(g,selected,full,H,W,tasks):
    if W['2']['count']<16 or any(W[str(t)]['count']<16 for t in tasks):return IDENTITY,'NO_SIGNAL'
    if tuple(g)==IDENTITY:return IDENTITY,'IDENTITY'
    if feasible(selected,full,positive_set(H,full),g) and objective(selected,H,W,g,tasks)<objective(full,H,W,IDENTITY,tasks):
        return tuple(g),'ACCEPTED_CURRENT_HOLDOUT'
    return IDENTITY,'REJECTED_CURRENT_HOLDOUT'


def panel(g):
    result={'full':IDENTITY,'P3_off':(0.,)*16}
    for j in range(16):
        v=list(IDENTITY);v[j]=0.;result[f'off_{j:02d}']=tuple(v)
    for a,b in ((0,1),(4,5),(8,9),(12,13)):
        v=list(IDENTITY);v[a]=v[b]=0.;result[f'pair_{a}_{b}']=tuple(v)
    for a in (0.,.25,.5,.75,1.):result[f'global_{a}']=(a,)*16
    result['selected']=tuple(g)
    permuted=list(g);random.Random(2209).shuffle(permuted);result['permutation']=tuple(permuted)
    return result
