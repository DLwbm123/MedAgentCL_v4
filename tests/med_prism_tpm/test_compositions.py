import importlib.util
from pathlib import Path
import unittest
import torch

spec=importlib.util.spec_from_file_location('compositions_audit',Path(__file__).resolve().parents[2]/'scripts/medicalskill_v2_tpm/audit_compositions.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class CompositionTests(unittest.TestCase):
    def test_exact_factorial_compositions_and_no_input_mutation(self):
        pre={'S':3,'P1':1,'P2':2,'P3':3};teacher={'S':2,'P1':1,'P2':2};post={'S':3,'P1':11,'P2':22,'P3':3}
        result=audit.compositions(pre,teacher,post)
        self.assertEqual(result['A'],({'S':2,'P1':1,'P2':2,'P3':3},[1,2]))
        self.assertEqual(result['D'][0],result['A'][0]);self.assertEqual(result['D'][1],[1,2,3])
        self.assertEqual(result['C'],(pre,[1,2]));self.assertEqual(result['B'],(pre,[1,2,3]))
        self.assertEqual(result['E'],(post,[1,2,3]));self.assertEqual(pre['S'],3)

    def test_kl_and_centered_interaction(self):
        a=torch.tensor([1.,2.,4.]);c=a+torch.tensor([.1,-.1,.2]);d=a+torch.tensor([-.2,.3,.1]);b=c+d-a
        self.assertAlmostEqual(audit.logit_metrics(a,a)['kl_teacher_to_variant'],0,places=12)
        self.assertGreater(audit.logit_metrics(a,c)['kl_teacher_to_variant'],0)
        self.assertAlmostEqual(audit.logit_metrics(a,a+10)['kl_teacher_to_variant'],0,places=12)
        result=audit.interaction_metrics(a,b,c,d)
        self.assertLess(result['interaction_squared_norm'],1e-12)
        result=audit.interaction_metrics(a,b+torch.tensor([0.,0.,1.]),c,d)
        self.assertGreater(result['interaction_squared_norm'],.1)
