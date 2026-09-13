import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


REPO = Path("/root/MedAgentCL_v4")


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DdpRankZeroIoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_path("callback_compat_ddp_test", REPO / "scripts/phase3/callback_compat.py")
        cls.trainer = load_path("shared_private_trainer_ddp_test", REPO / "med_prism/swift_plugins/shared_private_trainer.py")
        cls.plugin = load_path("shared_private_plugin_ddp_test", REPO / "med_prism/swift_plugins/med_prism_shared_private_plugin.py")
        cls.probe = load_path("reload_probe_ddp_test", REPO / "scripts/med_prism_real_5skill/reload_probe_plugin.py")

    def assert_rank(self, module, rank, expected):
        with mock.patch.object(module.dist, "is_available", return_value=True), \
             mock.patch.object(module.dist, "is_initialized", return_value=True), \
             mock.patch.object(module.dist, "get_rank", return_value=rank):
            self.assertEqual(module._is_main_process(), expected)

    def test_rank_detection(self):
        for module in (self.trainer, self.plugin, self.probe):
            self.assert_rank(module, 0, True)
            self.assert_rank(module, 1, False)

    def test_rank_one_does_not_write_shared_audits(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, module in enumerate((self.trainer, self.plugin)):
                path = Path(directory) / f"audit_{index}.json"
                with mock.patch.object(module, "_is_main_process", return_value=False):
                    module._write_json(path, {"status": "PASS"})
                self.assertFalse(path.exists())

    def test_rank_zero_writes_shared_audits(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, module in enumerate((self.trainer, self.plugin)):
                path = Path(directory) / f"audit_{index}.json"
                with mock.patch.object(module, "_is_main_process", return_value=True):
                    module._write_json(path, {"status": "PASS"})
                self.assertEqual(json.loads(path.read_text())["status"], "PASS")


if __name__ == "__main__":
    unittest.main()