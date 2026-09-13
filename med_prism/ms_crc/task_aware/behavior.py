"""Official behavior adapters and frozen, pooled atomic support. No data paths."""
from collections import Counter
from math import isclose
from med_prism.ms_crc.evaluator import legacy
from med_prism.ms_crc.search import ce,IDENTITY


class BehaviorMetricAdapter:
    def parse(self,row,prediction):raise NotImplementedError
    def metric(self,rows):raise NotImplementedError


class ConceptRecognitionAdapter(BehaviorMetricAdapter):
    def parse(self,row,prediction):
        # The official parser accepts every string, including empty/punctuation.
        # Only transport/type/parser exceptions are failures; invent no syntax rules.
        if not isinstance(prediction,str):
            try:gold=sorted(legacy.concept_sets(row,'')[1])
            except (TypeError,ValueError,KeyError,IndexError):gold=[]
            return {'valid':False,'gold':gold,'predicted':[]}
        try:
            predicted,gold,_,_=legacy.concept_sets(row,prediction)
        except (TypeError,ValueError,KeyError,IndexError):
            return {'valid':False,'gold':[],'predicted':[]}
        return {'valid':True,'gold':sorted(gold),'predicted':sorted(predicted),
            'all_tp':len(gold&predicted),'all_fp':len(predicted-gold),'all_fn':len(gold-predicted)}

    def metric(self,rows):
        if not all(r['valid'] for r in rows):raise ValueError('PARSER_FAILURE')
        tp=sum(r['all_tp'] for r in rows);fp=sum(r['all_fp'] for r in rows);fn=sum(r['all_fn'] for r in rows)
        return {'TP':tp,'FP':fp,'FN':fn,'utility':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


class SingleAnswerAdapter(BehaviorMetricAdapter):
    def __init__(self,task):
        if task not in (1,2,5):raise NotImplementedError('Unverified task, including grounding')
        self.task=task
    def parse(self,row,prediction):
        if not isinstance(prediction,str):return {'valid':False,'correct':False}
        # Reasoning uses the existing final-answer correctness, not chain similarity.
        return {'valid':True,'correct':legacy.answer_match(row,prediction)}
    def metric(self,rows):
        if not all(r['valid'] for r in rows):raise ValueError('PARSER_FAILURE')
        return {'utility':sum(r['correct'] for r in rows)/len(rows)}


class AtomicSupport:
    def __init__(self,H,full,ablations,adapter=None):
        self.adapter=adapter or ConceptRecognitionAdapter();self.H=H;self.full=full
        self.q=self.adapter.metric(full)['utility'];self.atoms=[];self.bank_ids=sorted(ablations)
        for rows in [full,*ablations.values()]:
            if [r['sample_id'] for r in H]!=[r['sample_id'] for r in rows]:raise ValueError('Order mismatch')
        for j,h in enumerate(H):
            refs=[h,full[j],*[v[j] for v in ablations.values()]]
            if isinstance(self.adapter,ConceptRecognitionAdapter):
                gold=set(h['gold'])
                if any(set(r['gold'])!=gold for r in refs if r['valid']):raise ValueError('Gold changed')
                negative=set().union(*(set(r['predicted']) for r in refs if r['valid']))-gold
                keys=[(c,True,2-self.q) for c in sorted(gold)]+[(c,False,self.q) for c in sorted(negative)]
            else:keys=[('answer',True,1.)]
            for concept,positive,weight in keys:
                atom={'sample':j,'concept':concept,'positive':positive,'weight':weight}
                uh=self.u(h,atom)
                banks=[i for i in self.bank_ids if h['valid'] and ablations[i][j]['valid'] and uh>self.u(ablations[i][j],atom)]
                atom.update(h=uh,banks=banks,supported=bool(banks));self.atoms.append(atom)
        self.Z=sum(a['weight'] for a in self.atoms if a['supported'])
        self.baseline=self.evaluate(full)
        self.status=('METRIC_DEGENERATE' if isinstance(self.adapter,ConceptRecognitionAdapter) and self.q==0 else
            'NO_SUPPORT' if self.Z==0 else 'NO_EXPOSED_HARM' if self.baseline['L_mem']==0 else 'SUPPORT_WITH_REPAIR_OPPORTUNITY')

    @staticmethod
    def u(row,atom):
        if not row['valid']:return 0 # failure NEVER earns negative suppression.
        if 'predicted' not in row:return float(row['correct'])
        present=atom['concept'] in row['predicted']
        return float(present if atom['positive'] else not present)

    def evaluate(self,rows):
        if len(rows)!=len(self.H) or [r['sample_id'] for r in rows]!=[r['sample_id'] for r in self.H]:raise ValueError('Order mismatch')
        loss=G=Gfull=D=U=0.;restored=lost=added=0;bankloss={i:0. for i in self.bank_ids}
        for a in self.atoms:
            j=a['sample'];h=a['h'];actual=self.u(rows[j],a);full=self.u(self.full[j],a);w=a['weight']
            damage=w*max(h-actual,0)
            if a['supported']:
                loss+=damage;restored+=int(h>full and actual>full)
            for i in a['banks']:bankloss[i]+=damage
            # Invalid H produces neither support nor a positive-transfer ledger.
            v=max(actual-h,0) if self.H[j]['valid'] else 0
            vf=max(full-h,0) if self.H[j]['valid'] else 0
            G+=w*v;Gfull+=w*vf;D+=w*max(vf-v,0);U+=w*max(v-vf,0)
            lost+=int(vf>v);added+=int(v>vf)
        if not isclose(G-Gfull,U-D,abs_tol=1e-8,rel_tol=1e-10):raise AssertionError('PT ledger identity')
        valid=all(r['valid'] for r in rows)
        return {**(self.adapter.metric(rows) if valid else {'utility':None}),'parser_valid':valid,
            'CE':ce(rows),'L_mem':loss/self.Z if self.Z else None,'G':G,'G_full':Gfull,'D_PT':D,'U_PT':U,
            'L_PT':D/Gfull if Gfull else 0.,'PT_status':'EVALUABLE' if Gfull else 'PT_NOT_EVALUABLE',
            'memory_supported_atoms_restored':restored,'full_positive_transfer_atoms_lost':lost,
            'new_positive_transfer_atoms_added':added,'per_bank_candidate_mem_loss':{
                i:bankloss[i]/z if (z:=sum(a['weight'] for a in self.atoms if i in a['banks'])) else None for i in self.bank_ids}}

    def diagnostics(self):
        supported=[a for a in self.atoms if a['supported']];mass=Counter();concepts=Counter()
        for a in supported:
            mass[a['sample']]+=a['weight'];concepts[('gold:' if a['positive'] else 'FP:')+a['concept']]+=a['weight']
        return {'status':self.status,'q':self.q,'omega_gold':2-self.q,'omega_FP':self.q,'support_mass':self.Z,
            'support_atoms_total':len(supported),'samples_with_support':len(mass),
            'ESS':self.Z**2/sum(v*v for v in mass.values()) if self.Z else 0.,
            'max_sample_support_share':max(mass.values())/self.Z if self.Z else 0.,
            'unique_supported_gold_concepts':len({a['concept'] for a in supported if a['positive']}),
            'unique_supported_FP_suppressions':len({a['concept'] for a in supported if not a['positive']}),
            'top_concepts_by_support_mass':concepts.most_common(20),'overlap_between_banks':sum(len(a['banks'])>1 for a in supported),
            'full':self.baseline,'per_bank':{i:{
                'positive_support_atoms':sum(i in a['banks'] and a['positive'] for a in self.atoms),
                'fp_suppression_atoms':sum(i in a['banks'] and not a['positive'] for a in self.atoms),
                'support_mass':sum(a['weight'] for a in self.atoms if i in a['banks']),
                'exposed_harm':self.baseline['per_bank_candidate_mem_loss'][i]} for i in self.bank_ids}}


def feasible(value,baseline):
    return value['parser_valid'] and value['utility']>=baseline['utility'] and value['G']>=baseline['G']-1e-9 and value['CE']<=baseline['CE']+.02


def select(support,scored,vectors,control=False):
    baseline=support.baseline if support is not None else scored['full']
    if not control and support.status!='SUPPORT_WITH_REPAIR_OPPORTUNITY':return 'full',support.status
    def ok(v):return (v['parser_valid'] and v['utility']>=baseline['utility'] and v['CE']<=baseline['CE']+.02) if control else feasible(v,baseline)
    order=[k for k in vectors if k in scored and (k=='full' or ok(scored[k]))]
    def key(k):
        v=scored[k];edit=sum(abs(1-g) for g in vectors[k])
        return (-v['utility'],v['CE'],edit) if control else (v['L_mem'],v['L_PT'],edit)
    chosen=min(order,key=key)
    return chosen,'IDENTITY' if tuple(vectors[chosen])==IDENTITY else 'FIT_SELECTED'


def accept(support,candidate,g):
    if tuple(g)==IDENTITY:return False,'IDENTITY'
    if support.Z==0 or support.q==0:return False,'INCONCLUSIVE_SUPPORT'
    if not feasible(candidate,support.baseline):return False,'UTILITY_REJECTED'
    if candidate['L_mem']>=support.baseline['L_mem']:return False,'NO_REPLICATED_MEMORY_EFFECT'
    return True,'ACCEPTED_CURRENT_HOLDOUT'
