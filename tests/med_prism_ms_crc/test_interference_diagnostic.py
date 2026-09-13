"""CPU-only regression checks for orchestration and diagnostic statistics."""
import copy
import inspect
from pathlib import Path
import tempfile
import unittest
import numpy as np
from med_prism.ms_crc.interference_diagnostic.statistics import association, top_weights, sign, csv_write, read, write
from med_prism.ms_crc.interference_diagnostic.splits import grouped_split,select_on_fit,evaluate_holdout,summarize_repeats
from med_prism.ms_crc.task_aware.behavior import ConceptRecognitionAdapter


def sample(i,aliases=None):
    return {'id':str(i),'images':[f'/images/{i}.png'],'metadata':aliases or {}}


def parsed(i,text):
    return {'sample_id':str(i),'nll_sum':1.,'answer_tokens':1,
        **ConceptRecognitionAdapter().parse({'messages':[{'content':'a;b;c'}]},text)}


def fixture():
    vectors={'full':[1.]*16,'off_07':[1.]*7+[0.]+[1.]*8,'P3_off':[0.]*16}
    strings={'H':'a;b','full':'b;c;d','minus1':'b;d','minus2':'b;d','off_07':'a;b;c','P3_off':'a;b'}
    cache={k:{str(i):parsed(i,s) for i in range(8)} for k,s in strings.items()}
    return cache,vectors


class DiagnosticTests(unittest.TestCase):
    def test_existing_reader_requires_path(self):
        from med_prism.ms_crc.evaluator import legacy
        from med_prism.ms_crc.interference_diagnostic.runner import worker
        import json
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'development.jsonl'
            path.write_text(json.dumps(sample(1))+'\n')
            self.assertEqual(legacy.read_jsonl(path,0),[sample(1)])
        self.assertIn("legacy.read_jsonl(Path(req['current_dev']),0)",inspect.getsource(worker))

    def test_sizes_seeds_disjoint(self):
        rows=[sample(i) for i in range(256)]
        a=grouped_split(rows,2209);b=grouped_split(rows,2210)
        self.assertEqual((len(a['fit_ids']),len(a['holdout_ids'])),(128,64))
        self.assertFalse(set(a['fit_ids'])&set(a['holdout_ids']))
        self.assertNotEqual(a['fit_ids'],b['fit_ids'])
        self.assertEqual(a,grouped_split(list(reversed(rows)),2209))
    def test_transitive_groups(self):
        rows=[sample(i,{'patient_id':'one'} if i<3 else {}) for i in range(8)]
        rows[2]['metadata']['study_id']='linked';rows[3]['metadata']['study_id']='linked'
        p=grouped_split(rows,2209,2,2)
        family=set(map(str,range(4)))
        if set(p['fit_ids'])&family:self.assertFalse(set(p['holdout_ids'])&family)
    def test_insufficient_and_duplicates_fail(self):
        with self.assertRaises(ValueError):grouped_split([sample(i) for i in range(10)],42)
        with self.assertRaises(ValueError):grouped_split([sample(1),sample(1)],42,1,1)
    def test_fit_cannot_see_holdout(self):
        c,v=fixture();f=select_on_fit(c,['0','1','2'],v)
        self.assertEqual(f['selected'],'off_07')
        other=copy.deepcopy(c)
        for rows in other.values():
            for i in ('3','4','5'):rows[i]=parsed(i,'garbage')
        self.assertEqual(f,select_on_fit(other,['0','1','2'],v))
    def test_no_reranking_or_checkpoint_API(self):
        src=inspect.getsource(evaluate_holdout)
        self.assertNotIn('=select(',src)
        self.assertNotIn('min(',src)
        c,v=fixture();f=select_on_fit(c,['0','1','2'],v)
        for i in ('3','4'):c['off_07'][i]=parsed(i,'garbage')
        h=evaluate_holdout(c,['3','4'],f,v)
        self.assertEqual(h['selected'],'off_07');self.assertFalse(h['accepted'])
        self.assertTrue(h['no_holdout_reranking'])
    def test_holdout_cache_immutable(self):
        c,v=fixture();original=copy.deepcopy(c);f=select_on_fit(c,['0','1'],v)
        evaluate_holdout(c,['2','3'],f,v);self.assertEqual(c,original)
    def test_tie_aware_top_k(self):
        self.assertAlmostEqual(sum(top_weights([1]*16,3)),3.)
        self.assertAlmostEqual(float(top_weights([1]*16,3)@top_weights([1]*16,3)),9/16)
        self.assertTrue(np.array_equal(top_weights(range(16),3),[0]*13+[1]*3))
    def test_correlations_zero_variance(self):
        self.assertIsNone(association([1]*16,list(range(16)))['spearman'])
        self.assertAlmostEqual(association(range(16),range(16))['spearman'],1.)
        self.assertAlmostEqual(association(range(16),range(15,-1,-1))['spearman'],-1.)
    def test_sign_zero_not_positive(self):
        self.assertEqual(sign(1e-16),0);self.assertEqual(sign(-.1),-1)
    def test_json_csv_roundtrip_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            write(Path(d)/'a.json',{'a':1});self.assertEqual(read(Path(d)/'a.json'),{'a':1})
            csv_write(Path(d)/'a.csv',[{'a':1.,'b':None}])
            self.assertIn('unavailable',(Path(d)/'a.csv').read_text())
    def test_summary_identity_not_stability_success(self):
        _,v=fixture();r={'selected':'full','holdout_gain':0.,'fit_gain':0.,'sign_consistent':True,
            'selected_minus_global_holdout':0.,'accepted':False,'off_07_holdout_gain':-.1}
        s=summarize_repeats([r]*10,v)
        self.assertEqual(s['identity_frequency'],1.);self.assertIsNone(s['nonidentity_sign_consistency'])
        self.assertEqual(s['accepted_count'],0)


if __name__=='__main__':unittest.main()
