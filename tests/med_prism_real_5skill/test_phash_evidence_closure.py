import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).with_name("phash_evidence_closure.py")
if not PATH.is_file():
    PATH = Path("/root/MedAgentCL_v4/scripts/med_prism_real_5skill/phash_evidence_closure.py")
SPEC = importlib.util.spec_from_file_location("phash_evidence_closure", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PhashEvidenceTest(unittest.TestCase):
    def test_classification_buckets(self):
        self.assertEqual(MODULE.classification_bucket("likely_near_duplicate"), "likely")
        self.assertEqual(MODULE.classification_bucket("low_information_false_positive"), "false_positive")
        self.assertEqual(MODULE.classification_bucket("unresolved"), "unresolved")

    def test_source_dataset(self):
        self.assertEqual(MODULE.source_dataset("/remote-home/x/MedSG/a.png"), "MedSG")
        self.assertEqual(MODULE.source_dataset("/remote-home/x/MedTrinity-25M/a.png"), "MedTrinity-25M")

    def test_canonical_match_requires_train_test(self):
        refs = [
            {"task": 1, "split": "train", "id": "a", "lineage_tokens": ["isic:x"]},
            {"task": 2, "split": "test", "id": "b", "lineage_tokens": ["isic:x"]},
        ]
        self.assertEqual(len(MODULE.canonical_train_test_matches(refs)), 1)
        refs[1]["split"] = "train"
        self.assertEqual(MODULE.canonical_train_test_matches(refs), [])


if __name__ == "__main__":
    unittest.main()
