import importlib.util
import pathlib
import unittest


HERE = pathlib.Path(__file__).parent
SCRIPT_DIR = HERE if (HERE / "concept_v3_closure.py").is_file() else HERE.parents[1] / "scripts" / "med_prism_real_5skill"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


repair = load("concept_v3_closure")
audit = load("concept_v3_auditor")


class ConceptV3Test(unittest.TestCase):
    def test_position_templates_are_removed(self):
        for value in ("quadrant", "upper right quadrant", "lower left region", "central position", "peripheral area"):
            self.assertEqual(repair.drop_reason({"canonical": value, "mention": value}), "drop_position_template_v3")

    def test_specific_medical_phrases_survive(self):
        for value in ("pathological fracture", "fibrotic change", "brain tissue", "epithelial tissue"):
            self.assertIsNone(repair.drop_reason({"canonical": value, "mention": value}))

    def test_semantics_precede_dedup(self):
        row = {"source_caption": "A possible left lesion is seen.", "concepts": [
            {"canonical": "lesion", "type": "finding", "status": "present", "laterality": "unspecified", "mention": "lesion"},
            {"canonical": "Lesion", "type": "finding", "status": "present", "laterality": "left", "mention": "lesion"},
        ]}
        result = repair.repair_row(row)
        self.assertEqual(len(result["concepts"]), 1)
        self.assertEqual(result["concepts"][0]["status"], "uncertain")
        self.assertEqual(result["concepts"][0]["laterality"], "left")

    def test_independent_auditor_detects_failures(self):
        row = {"source_caption": "There may be a lesion in the upper right quadrant.", "modality": "ct"}
        result = audit.audit_row(row, [
            {"canonical": "upper right quadrant", "type": "attribute", "status": "present", "mention": "upper right quadrant"},
            {"canonical": "lesion", "type": "finding", "status": "present", "mention": "lesion"},
            {"canonical": "Lesion", "type": "finding", "status": "present", "mention": "lesion"},
        ])
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("pure_positional_concept", result["reason_counts"])
        self.assertIn("exact_duplicate", result["reason_counts"])
        self.assertIn("uncertainty_cue_inconsistency", result["reason_counts"])


if __name__ == "__main__":
    unittest.main()
