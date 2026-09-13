import importlib.util
from pathlib import Path
import unittest

import torch


REPO = Path("/root/MedAgentCL_v4")
PLUGIN = REPO / "scripts/med_prism_real_5skill/reload_probe_plugin.py"


class ReloadProbeTensorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compat_path = REPO / "scripts/phase3/callback_compat.py"
        compat_spec = importlib.util.spec_from_file_location("callback_compat_test", compat_path)
        compat_module = importlib.util.module_from_spec(compat_spec)
        compat_spec.loader.exec_module(compat_module)
        spec = importlib.util.spec_from_file_location("reload_probe_plugin_test", PLUGIN)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_scalar_tensor_is_cloned_without_indexing(self):
        source = torch.tensor(7)
        result = self.module.MedPrismReloadProbeTrainer._clone_probe_tensor(source)
        self.assertEqual(result.ndim, 0)
        self.assertEqual(result.item(), 7)
        self.assertEqual(result.device.type, "cpu")

    def test_batched_tensor_keeps_first_example(self):
        source = torch.arange(12).reshape(3, 4)
        result = self.module.MedPrismReloadProbeTrainer._clone_probe_tensor(source)
        self.assertEqual(tuple(result.shape), (3, 4))
        self.assertTrue(torch.equal(result, source))


if __name__ == "__main__":
    unittest.main()