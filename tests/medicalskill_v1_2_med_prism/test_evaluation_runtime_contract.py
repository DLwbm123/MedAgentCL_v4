from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts/medicalskill_v1_2_med_prism"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))

from evaluation_contract_v1_2 import required_generation_cells


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generation_cell_contract() -> None:
    expected = {
        (False, False): 15,
        (False, True): 30,
        (True, False): 20,
        (True, True): 35,
    }
    for (with_stage0, with_oracle), count in expected.items():
        cells = required_generation_cells(
            with_stage0=with_stage0, with_oracle=with_oracle
        )
        assert len(cells) == count
        assert all(mode != "task_free" for _, mode, _ in cells)


def test_reasoning_priority_and_stage_affinity() -> None:
    runner = load_module("parallel_runner", SCRIPT_DIR / "run_parallel_formal_v1_2.py")
    jobs = [
        (5, "primary_cumulative", 5),
        (5, "oracle_skill_aware", 5),
        (4, "primary_cumulative", 5),
        (5, "primary_cumulative", 4),
    ]
    assert runner.select_job(jobs.copy(), None) == (5, "primary_cumulative", 5)
    assert runner.select_job(jobs.copy(), 4) == (4, "primary_cumulative", 5)


def test_oracle_off_aggregation_config_does_not_require_oracle(tmp_path: Path) -> None:
    aggregator = load_module("formal_aggregator_v2", SCRIPT_DIR / "aggregate_formal_v1_2.py")
    config = {
        "status": "PASS",
        "oracle_evaluation_enabled": False,
        "stage0_evaluation_enabled": False,
        "base_zero_shot_reused": True,
        "base_evaluation_root": "/canonical/base",
        "generation_cell_count": 15,
    }
    (tmp_path / "evaluation_run_config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    loaded = aggregator.evaluation_config(tmp_path)
    assert loaded["oracle_evaluation_enabled"] is False
    assert loaded["generation_cell_count"] == 15


def test_legacy_output_defaults_to_full_35_contract(tmp_path: Path) -> None:
    aggregator = load_module("formal_aggregator_legacy", SCRIPT_DIR / "aggregate_formal_v1_2.py")
    loaded = aggregator.evaluation_config(tmp_path)
    assert loaded["format_version"] == "legacy_full_35_cell"
    assert loaded["oracle_evaluation_enabled"] is True
    assert loaded["stage0_evaluation_enabled"] is True
    assert loaded["generation_cell_count"] == 35


def test_sequential_contract_remains_primary_only() -> None:
    seq = (ROOT / "scripts/medicalskill_v1_2_sequential/evaluate_sequential_v1_2.py").read_text()
    aggregate = (ROOT / "scripts/medicalskill_v1_2_sequential/aggregate_sequential_v1_2.py").read_text()
    assert 'stem = f"primary_cumulative_task_{task_id:02d}"' in seq
    assert "oracle_skill_aware" not in seq
    assert "base_zero_shot" not in seq
    assert "task_free" not in seq
    assert '"matrix_cells": 15' in aggregate
    assert "no base/oracle/task-free matrix" in aggregate
