from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluation_scope_is_lower_triangular() -> None:
    evaluator = load_module(
        "formal_evaluator",
        ROOT / "scripts/medicalskill_v1_2_med_prism/evaluate_formal_v1_2.py",
    )
    assert evaluator.evaluation_task_ids(0) == [1, 2, 3, 4, 5]
    assert [evaluator.evaluation_task_ids(stage) for stage in range(1, 6)] == [
        [1],
        [1, 2],
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
    ]


def test_matrix_leaves_future_tasks_empty(tmp_path: Path) -> None:
    aggregator = load_module(
        "formal_aggregator",
        ROOT / "scripts/medicalskill_v1_2_med_prism/aggregate_formal_v1_2.py",
    )
    for stage in range(1, 6):
        stage_dir = tmp_path / "evaluation" / f"stage_{stage:02d}"
        stage_dir.mkdir(parents=True)
        for task in range(1, stage + 1):
            score = stage / 10 + task / 100
            path = stage_dir / f"primary_cumulative_task_{task:02d}.summary.json"
            path.write_text(json.dumps({"primary_score": score}), encoding="utf-8")

    values = aggregator.matrix(tmp_path, "primary_cumulative")
    assert [sum(value is not None for value in row) for row in values] == [1, 2, 3, 4, 5]
    assert values[0][1:] == [None, None, None, None]
    assert all(value is not None for value in values[-1])

    output = tmp_path / "matrix.csv"
    aggregator.write_matrix(output, values)
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[1][1:6] == ["0.11000000", "", "", "", ""]
    assert rows[1][6] == "0.11000000"
    assert rows[5][1:6] == ["0.51000000", "0.52000000", "0.53000000", "0.54000000", "0.55000000"]
    assert rows[5][6] == "0.53000000"


if __name__ == "__main__":
    test_evaluation_scope_is_lower_triangular()
    with tempfile.TemporaryDirectory() as directory:
        test_matrix_leaves_future_tasks_empty(Path(directory))
    print("lower-triangular evaluation tests: PASS")
