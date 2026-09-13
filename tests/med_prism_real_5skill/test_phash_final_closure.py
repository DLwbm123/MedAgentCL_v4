import importlib.util
import pathlib
import unittest


HERE = pathlib.Path(__file__).parent
SCRIPT_DIR = HERE if (HERE / "phash_final_closure.py").is_file() else HERE.parents[1] / "scripts" / "med_prism_real_5skill"
spec = importlib.util.spec_from_file_location("phash_final", SCRIPT_DIR / "phash_final_closure.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PHashFinalTest(unittest.TestCase):
    def test_per_image_tokens_do_not_retain_foreign_tokens(self):
        ref = {"path": "/x/MedSG/a/ISIC_0001131.png", "sha256": "a" * 64, "lineage_tokens": ["isic:isic_9999999", "sha256:" + "b" * 64, "source:record"]}
        clean = module.clean_ref(ref)
        self.assertIn("isic:isic_0001131", clean["image_level_tokens"])
        self.assertNotIn("isic:isic_9999999", clean["image_level_tokens"])
        self.assertEqual(clean["group_level_tokens"], ["source:record"])

    def test_cross_split_match(self):
        left = {"task": 1, "split": "train", "id": "a", "path": "/x/MedIMeta/oct/images/1.png", "sha256": "a" * 64, "phash": "1", "lineage_tokens": []}
        right = {**left, "split": "test", "id": "b"}
        refs = module.dedup_refs([left, right])
        self.assertTrue(module.cross_split(refs))
        self.assertTrue(module.evidence_matches(refs))


    def test_compact_ref(self):
        clean = module.clean_ref([1, "train", "x", "/x/MedIMeta/oct/images/1.png", "abc", [], "a" * 64])
        self.assertEqual(clean["task"], 1)
        self.assertEqual(clean["split"], "train")

if __name__ == "__main__":
    unittest.main()
