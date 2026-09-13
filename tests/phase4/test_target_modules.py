import unittest

from med_prism.adapters.injection import (
    classify_target_names,
    collect_language_targets,
    inject_rank1_wrappers,
    iter_rank1_wrappers,
)
from med_prism.adapters.rank1_lora import Rank1ExpertLinear
from tests.phase4.helpers import ToyQwen3VL


class TargetModuleTest(unittest.TestCase):
    def test_qwen_language_inventory_is_exactly_36_q_and_36_v(self):
        inventory = collect_language_targets(ToyQwen3VL())
        self.assertEqual(inventory.language, 72)
        self.assertEqual(inventory.q_proj, 36)
        self.assertEqual(inventory.v_proj, 36)
        self.assertEqual(inventory.vision, 0)
        self.assertEqual(inventory.merger, 0)
        self.assertEqual(len(inventory.sha256), 64)

    def test_name_classifier_excludes_vision_merger_k_and_o(self):
        names = [
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.self_attn.v_proj",
            "model.language_model.layers.0.self_attn.k_proj",
            "model.language_model.layers.0.self_attn.o_proj",
            "visual.blocks.0.attn.q_proj",
            "model.language_model.merger.q_proj",
        ]
        inventory = classify_target_names(names)
        self.assertEqual(
            inventory.names,
            (
                "model.language_model.layers.0.self_attn.q_proj",
                "model.language_model.layers.0.self_attn.v_proj",
            ),
        )

    def test_injection_wraps_only_language_q_and_v(self):
        model = ToyQwen3VL()
        inventory = inject_rank1_wrappers(model, dropout=0.0)
        wrappers = list(iter_rank1_wrappers(model))
        self.assertEqual(len(wrappers), 72)
        self.assertEqual({name for name, _ in wrappers}, set(inventory.names))
        self.assertNotIsInstance(model.visual.q_proj, Rank1ExpertLinear)
        self.assertNotIsInstance(model.visual.v_proj, Rank1ExpertLinear)
        self.assertNotIsInstance(
            model.model.language_model.layers[0].self_attn.k_proj,
            Rank1ExpertLinear,
        )


if __name__ == "__main__":
    unittest.main()
