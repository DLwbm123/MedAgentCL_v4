"""Guard the existing reload probe when selecting the RGPA trainer."""
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest

runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/phase3/callback_compat.py'))
import med_prism.swift_plugins.med_prism_rank1_plugin
import med_prism.swift_plugins.med_prism_shared_private_plugin
from med_prism.rgpa.plugin import RGPATrainer
from scripts.med_prism_real_5skill.reload_probe_plugin import MedPrismReloadProbeTrainer
from swift.trainers import TrainerFactory


class PluginTest(unittest.TestCase):
    def test_existing_reload_probe_is_retained(self):
        self.assertTrue(issubclass(RGPATrainer, MedPrismReloadProbeTrainer))
        self.assertIs(TrainerFactory.get_trainer_cls(SimpleNamespace(tuner_type='med_prism_shared_private')), RGPATrainer)


if __name__ == '__main__':
    unittest.main()
