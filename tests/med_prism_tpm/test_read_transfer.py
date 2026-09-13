import importlib.util
from pathlib import Path
import unittest
import torch
from test_transport import make_model, records
from med_prism.transport.hooks import QwenBlockBridge, cleanup_hooks
from med_prism.transport.config import TPMConfig
from med_prism.transport.banks import fingerprint

spec=importlib.util.spec_from_file_location('read_audit',Path(__file__).resolve().parents[2]/'scripts/medicalskill_v2_tpm/audit_read_transfer.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class ReadAuditTests(unittest.TestCase):
    def test_native_capture_matches_dense_effective_error_and_is_read_only(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        model=make_model();bridge=QwenBlockBridge(model);record=records()[0]
        for w in bridge.targets.values():w.set_active_tasks([1,2])
        bridge.collect_teacher(record,3,TPMConfig(tokens_per_sample=8))
        for w in bridge.targets.values():
            w.shared.B.add_(.01);w.set_active_tasks([1,2,3])
        before=fingerprint(model)
        rms={name:.7 for name in bridge.targets}
        actual=audit.student_reads(bridge,record,rms)
        expected={}
        def hook(name):
            def capture(module,args):
                x=args[0][0,record['positions']].float().double()/.7
                for bank in bridge.banks[name]:
                    if bank.task_id>=3:continue
                    a,b=bank.matrices(device=bridge.device)
                    z=record['teacher'][name]['targets'][bank.task_id].double()/.7
                    dense=(x@a.double().T-z)@b.double().T
                    expected[name,bank.task_id]=float(dense.square().sum()/len(x))
            return capture
        with cleanup_hooks() as handles:
            for name,w in bridge.targets.items():handles.append(w.register_forward_pre_hook(hook(name)))
            bridge.backbone(**bridge.model_inputs(record['batch']),use_cache=False,return_dict=True)
        self.assertEqual(len(actual),12)
        for key in actual:self.assertAlmostEqual(actual[key],expected[key],places=12)
        self.assertEqual(before,fingerprint(model))
