import copy
import inspect
import unittest
from unittest.mock import patch
from med_prism.ms_crc.task_aware.behavior import (ConceptRecognitionAdapter,SingleAnswerAdapter,AtomicSupport,select,accept,feasible)
from med_prism.ms_crc.search import IDENTITY

AD=ConceptRecognitionAdapter()
def row(gold,pred,nll=1.,identity='x'):
    value=AD.parse({'messages':[{'content':gold}]},pred)
    return {**value,'sample_id':identity,'nll_sum':nll,'answer_tokens':1}
def example():
    H=[row('a;b;c','a;b')];full=[row('a;b;c','b;c;d')];minus=[row('a;b;c','b;d')]
    return AtomicSupport(H,full,{1:minus,2:minus})


class TaskAwareTests(unittest.TestCase):
    def test_exact_zero_atomic_nonzero(self):
        s=example();self.assertNotEqual(s.H[0]['gold'],s.H[0]['predicted']);self.assertGreater(s.Z,0)
    def test_gold_support(self):self.assertEqual(example().diagnostics()['per_bank'][1]['positive_support_atoms'],1)
    def test_fp_suppression(self):self.assertEqual(example().diagnostics()['per_bank'][1]['fp_suppression_atoms'],1)
    def test_pool_deduplicates(self):
        s=example();self.assertEqual(len([a for a in s.atoms if a['supported']]),2);self.assertAlmostEqual(s.Z,2.)
    def test_per_bank_overlap(self):self.assertEqual(example().diagnostics()['overlap_between_banks'],2)
    def test_candidate_new_fp_counted_without_atom_expansion(self):
        s=example();before=copy.deepcopy(s.atoms);a=s.evaluate([row('a;b;c','a;b;c')]);b=s.evaluate([row('a;b;c','a;b;c;new')])
        self.assertLess(b['utility'],a['utility']);self.assertEqual(before,s.atoms)
    def test_parser_failure_no_reward(self):
        s=example();bad=row('a;b;c',None);value=s.evaluate([bad]);self.assertFalse(value['parser_valid']);self.assertFalse(feasible(value,s.baseline))
        neg=next(a for a in s.atoms if not a['positive']);self.assertEqual(s.u(bad,neg),0)
    def test_bad_reference_no_support(self):
        s=AtomicSupport([row('a',None)],[row('a','a')],{1:[row('a','x')]});self.assertEqual(s.Z,0)
    def test_empty_prediction(self):self.assertEqual(row('a;b','')['all_fn'],2)
    def test_empty_gold(self):self.assertEqual(row('','a;b')['all_fp'],2)
    def test_both_empty_no_tn(self):
        s=AtomicSupport([row('','')],[row('','')],{1:[row('','')]});self.assertEqual(s.atoms,[])
    def test_micro_not_sample_average(self):
        values=[row('a','a'),row('b;c;d','')];self.assertEqual(AD.metric(values)['utility'],.4)
    def test_finite_micro_identity(self):
        f=[row('a;b;c','b;c;d')];g=[row('a;b;c','a;b;c;x')];q=AD.metric(f)['utility'];m=AD.metric(g);b=AD.metric(f)
        rhs=((2-q)*(m['TP']-b['TP'])-q*(m['FP']-b['FP']))/(3+4)
        self.assertAlmostEqual(m['utility']-q,rhs)
    def test_ledger_identity(self):
        v=example().evaluate([row('a;b;c','a;b;c')]);self.assertAlmostEqual(v['G']-v['G_full'],v['U_PT']-v['D_PT'])
    def test_repair_not_PT(self):
        s=example();v=s.evaluate([row('a;b;c','a;b;c')]);self.assertEqual(v['new_positive_transfer_atoms_added'],0)
        self.assertEqual(v['memory_supported_atoms_restored'],2)
    def test_gain_can_be_replaced(self):
        s=AtomicSupport([row('a;b;c','a')],[row('a;b;c','b')],{1:[row('a;b;c','')]})
        v=s.evaluate([row('a;b;c','a;c')]);self.assertEqual(v['full_positive_transfer_atoms_lost'],1)
        self.assertEqual(v['new_positive_transfer_atoms_added'],1);self.assertTrue(feasible(v,s.baseline))
    def test_utility_constraint(self):
        s=example();v=s.evaluate([row('a;b;c','')]);self.assertFalse(feasible(v,s.baseline))
    def test_CE_constraint(self):
        s=example();v=s.evaluate([row('a;b;c','a;b;c',1.021)]);self.assertFalse(feasible(v,s.baseline))
    def test_no_support(self):
        r=[row('a','a')];self.assertEqual(AtomicSupport(r,r,{1:r}).status,'NO_SUPPORT')
    def test_no_exposed_harm(self):
        r=[row('a','a')];self.assertEqual(AtomicSupport(r,r,{1:[row('a','')]}).status,'NO_EXPOSED_HARM')
    def test_repair_opportunity(self):self.assertEqual(example().status,'SUPPORT_WITH_REPAIR_OPPORTUNITY')
    def test_degenerate(self):
        self.assertEqual(AtomicSupport([row('a','a')],[row('a','x')],{1:[row('a','')]}).status,'METRIC_DEGENERATE')
    def test_fit_lexicographic_and_manifest_tie(self):
        s=example();v=s.evaluate([row('a;b;c','a;b;c')]);vectors={'full':IDENTITY,'a':(0.,)+(1.,)*15,'b':(1.,0.)+(1.,)*14}
        self.assertEqual(select(s,{'full':s.baseline,'a':v,'b':v},vectors)[0],'a')
    def test_identity_fallback(self):
        s=example();self.assertEqual(select(s,{'full':s.baseline,'a':s.baseline},{'full':IDENTITY,'a':(0.,)+(1.,)*15})[0],'full')
    def test_holdout_accept(self):
        s=example();self.assertTrue(accept(s,s.evaluate([row('a;b;c','a;b;c')]),(0.,)+(1.,)*15)[0])
    def test_holdout_cannot_choose_second(self):
        s=example();self.assertEqual(accept(s,s.baseline,(0.,)+(1.,)*15),(False,'NO_REPLICATED_MEMORY_EFFECT'))
        self.assertEqual(list(inspect.signature(accept).parameters),['support','candidate','g'])
    def test_holdout_utility_reject(self):
        s=example();self.assertEqual(accept(s,s.evaluate([row('a;b;c','a;b;c',2.)]),(0.,)+(1.,)*15)[1],'UTILITY_REJECTED')
    def test_no_old_data_API(self):
        import med_prism.ms_crc.task_aware.behavior as behavior
        source=inspect.getsource(behavior)
        for key in ('open(','Path(','development.jsonl','train.jsonl'):self.assertNotIn(key,source)
    def test_single_answer_adapter_and_grounding_unsupported(self):
        for task in (1,2,5):self.assertTrue(SingleAnswerAdapter(task).parse({'messages':[{'content':'yes'}]},'yes')['correct'])
        with self.assertRaises(NotImplementedError):SingleAnswerAdapter(4)
    def test_parser_normalizer_duplicates(self):
        a=row('A;B','a, A\nb!');self.assertEqual(a['all_tp'],2);self.assertEqual(a['all_fp'],0)
    def test_ESS_by_sample_not_atoms(self):
        s=example();self.assertEqual(s.diagnostics()['ESS'],1.)
    def test_control_has_no_support_requirement(self):
        baseline={'utility':.3,'CE':1.,'parser_valid':True};candidate={'utility':.4,'CE':1.,'parser_valid':True}
        self.assertEqual(select(None,{'full':baseline,'a':candidate},{'full':IDENTITY,'a':(0.,)+(1.,)*15},control=True)[0],'a')
    def test_PT_not_evaluable(self):
        r=[row('a','a')];s=AtomicSupport(r,r,{1:[row('a','')]});self.assertEqual(s.baseline['PT_status'],'PT_NOT_EVALUABLE')

    def test_empty_real_evaluator_glue_keeps_FN(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from med_prism.ms_crc.task_aware.runtime import concept_score
        scorer=SimpleNamespace(composition=SimpleNamespace(use=lambda *args:nullcontext()),model=None,processor=None)
        records=[{'row':{'id':'x','messages':[{'content':'a;b'}]},'batch':{}}]
        with patch('med_prism.ms_crc.task_aware.runtime.cell.batched_generate',return_value=['']),patch(
            'med_prism.ms_crc.task_aware.runtime.token_scores',return_value=({'nll_sum':1.,'answer_tokens':1,'margin':0.},None)):
            values,metric=concept_score(scorer,records)
        self.assertTrue(values[0]['valid']);self.assertEqual(metric['FN'],2)

    def test_v23_global_fold_metadata_and_A_history(self):
        import tempfile
        from pathlib import Path
        import torch
        from test_ms_crc import toy,source
        from med_prism.ms_crc.composition import Composition
        from med_prism.ms_crc.task_aware.runtime import folded_export,read
        m=toy();comp=Composition(m)
        with tempfile.TemporaryDirectory() as d:
            src=source(m,Path(d)/'source');original=src.tensors()
            deployed,values=folded_export(src,comp,(.25,)*16,Path(d)/'export')
            self.assertEqual(read(deployed.origin)['method_version'],'2.3')
            reloaded=deployed.tensors()
            for k,v in original.items():
                target=v*.25 if '.task_0003__' in k and k.endswith('.B') else v
                self.assertTrue(torch.equal(target,values[k]))
                self.assertTrue(torch.equal(values[k],reloaded[k]))


if __name__=='__main__':unittest.main()
