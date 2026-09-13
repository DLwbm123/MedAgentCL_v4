from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
from med_prism.rgpa import separated as s


class SeparatedTests(TestCase):
    def test_environment_overrides_wrong_inherited_data(self):
        data = '/remote-home/wangbomin/MedicalSkill-CL-v1.2-lite-10k1k'
        def read(path):
            return {'data':{'dataset_root':data}} if path.name=='run_manifest.json' else {'slack_enabled':True}
        with patch.object(s, 'read', read), patch.dict('os.environ', {'MED_PRISM_DATA_ROOT':'WRONG'}):
            e = s.eval_env(Path('/run'), Path('/run/evaluation_lite_new'), 1)
        self.assertEqual(e['MED_PRISM_DATA_ROOT'], data)
        self.assertEqual(e['CUDA_VISIBLE_DEVICES'], '1')
        self.assertEqual(e['MED_PRISM_EVALUATION_ROOT'], '/run/evaluation_lite_new')

    def test_full_dataset_result_rejected(self):
        spec = dict(test='/lite/test.jsonl', test_sha256='litehash', test_count=1000)
        result = dict(status='PASS', stage=2, eval_task=1, mode='primary_cumulative',
            dataset='/full/test.jsonl', dataset_sha256='other', total=17796)
        def read(path):
            return {'data':{'tasks':{'1':spec}}} if path.name=='run_manifest.json' else result
        with patch.object(s, 'read', read), self.assertRaises(ValueError):
            s.validate_cell(Path('/run'), Path('/new'), 2, 1)

    def test_no_work_on_occupied_gpu(self):
        with patch.object(s.subprocess, 'check_output', return_value='34941, 99\n'):
            self.assertFalse(s.gpu_available(1))
        with patch.object(s.subprocess, 'check_output', return_value='1, 0\n'):
            self.assertTrue(s.gpu_available(0))

    def test_exact_six_cells_no_stage_zero_or_oracle(self):
        self.assertEqual(set(s.CELLS), {(i,j) for i in range(1,4) for j in range(1,i+1)})
