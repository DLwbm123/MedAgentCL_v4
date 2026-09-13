import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("concept_v2_closure.py")
if not MODULE_PATH.is_file():
    MODULE_PATH = Path('/root/MedAgentCL_v4/scripts/med_prism_real_5skill/concept_v2_closure.py')
SPEC = importlib.util.spec_from_file_location("concept_v2_closure", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ConceptV2Test(unittest.TestCase):
    @staticmethod
    def concept(canonical, mention=None, kind="finding", status="present"):
        return {"canonical": canonical, "mention": mention or canonical,
                "type": kind, "status": status, "laterality": "unspecified"}

    def test_required_artifacts_are_failures(self):
        values = ("central position", "upper", "cross-sectional view",
                  "anatomical association", "region of interest", "bounding box",
                  "area ratio", "highlighted region", "disease process",
                  "generic abnormality")
        for value in values:
            with self.subTest(value=value):
                self.assertIsNotNone(MODULE.artifact_reason(self.concept(value)))

    def test_medical_directional_anatomy_is_retained(self):
        self.assertIsNone(MODULE.artifact_reason(self.concept("left kidney", kind="anatomy")))
        self.assertIsNone(MODULE.artifact_reason(self.concept("upper lobe", kind="anatomy")))

    def test_uncertainty_cues(self):
        cues = ("possible pneumonia", "possibly pneumonia", "potential pneumonia",
                "potentially pneumonia", "this may represent pneumonia",
                "this might represent pneumonia", "this could represent pneumonia",
                "findings suggest pneumonia", "findings suggesting pneumonia",
                "likely pneumonia", "findings are indicative of pneumonia")
        for caption in cues:
            with self.subTest(caption=caption):
                self.assertTrue(MODULE.should_be_uncertain(caption, self.concept("pneumonia")))

    def test_clear_affirmation_overrides_separate_uncertainty(self):
        caption = "Possible pneumonia was considered. CT definitively confirms pneumonia."
        self.assertFalse(MODULE.should_be_uncertain(caption, self.concept("pneumonia")))

    def test_unrelated_distant_cue_does_not_taint_finding(self):
        caption = "Possible pneumonia is discussed with many unrelated descriptive words before pleural effusion."
        self.assertFalse(MODULE.should_be_uncertain(caption, self.concept("pleural effusion")))

    def test_uncertainty_does_not_change_anatomy_or_negation(self):
        self.assertFalse(MODULE.should_be_uncertain(
            "Possible left kidney finding.", self.concept("left kidney", kind="anatomy")))
        self.assertFalse(MODULE.should_be_uncertain(
            "Possible pneumonia.", self.concept("pneumonia", status="negated")))


if __name__ == "__main__":
    unittest.main()
