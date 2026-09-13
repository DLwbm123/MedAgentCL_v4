import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace


class SwiftPluginTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compatibility = (
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "phase3"
            / "callback_compat.py"
        )
        runpy.run_path(str(compatibility))
        from med_prism.swift_plugins import med_prism_rank1_plugin as plugin

        cls.plugin = plugin

    def test_custom_tuner_and_trainer_are_registered(self):
        from swift.tuner_plugin import tuners_map

        self.assertIs(
            tuners_map["med_prism_rank1"],
            self.plugin.MedPrismRank1Tuner,
        )
        args = SimpleNamespace(tuner_type="med_prism_rank1", task_type="causal_lm")
        self.assertIs(
            self.plugin.TrainerFactory.get_trainer_cls(args),
            self.plugin.MedPrismRank1Trainer,
        )

    def test_native_lora_still_uses_stock_swift_trainer(self):
        args = SimpleNamespace(tuner_type="lora", task_type="causal_lm")
        trainer = self.plugin.TrainerFactory.get_trainer_cls(args)
        self.assertIsNot(trainer, self.plugin.MedPrismRank1Trainer)
        self.assertEqual(trainer.__module__, "swift.trainers.seq2seq_trainer")


if __name__ == "__main__":
    unittest.main()
