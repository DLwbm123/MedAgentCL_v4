import importlib.util
import pathlib
import unittest


HERE = pathlib.Path(__file__).parent
SCRIPT_DIR = HERE if (HERE / "evaluate_real_5stage.py").is_file() else HERE.parents[1] / "scripts" / "med_prism_real_5skill"
spec = importlib.util.spec_from_file_location("evaluation", SCRIPT_DIR / "evaluate_real_5stage.py")
evaluation = importlib.util.module_from_spec(spec)


class RealFiveStageTest(unittest.TestCase):
    def test_hungarian_beats_greedy_conflict(self):
        predicted = [[0, 0, 10, 10], [0, 0, 8, 10]]
        reference = [[0, 0, 8, 10], [2, 0, 10, 10]]
        # Import lazily on the server where torch/transformers/qwen utilities exist.
        spec.loader.exec_module(evaluation)
        acc, mean, matches = evaluation.hungarian_scores(predicted, reference)
        self.assertEqual(len(matches), 2)
        self.assertGreater(mean, 0.7)
        self.assertEqual(acc, 1.0)


if __name__ == "__main__":
    unittest.main()
