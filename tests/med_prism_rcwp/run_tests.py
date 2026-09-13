"""Run RCWP and relevant existing suites without adding pytest dependencies."""
import inspect
from pathlib import Path
import runpy
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"scripts/medicalskill_v1_2_med_prism"))
runpy.run_path(str(ROOT/"scripts/phase3/callback_compat.py"))
success=True
for folder in ("med_prism_rcwp","med_prism_tpm","medicalskill_v1_2_med_prism_versions"):
    suite=unittest.TestLoader().discover(str(ROOT/"tests"/folder))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    success=result.wasSuccessful() and success
module=runpy.run_path(str(ROOT/"tests/medicalskill_v1_2_med_prism/test_evaluation_runtime_contract.py"))
count=0
for name,fn in module.items():
    if name.startswith("test_") and callable(fn):
        parameters=list(inspect.signature(fn).parameters)
        if parameters==["tmp_path"]:
            with tempfile.TemporaryDirectory() as directory:fn(Path(directory))
        elif not parameters:fn()
        else:raise RuntimeError("Unexpected fixture requirement")
        count+=1
print(f"Existing evaluation runtime contract: {count} PASS")
runpy.run_path(str(ROOT/"tests/medicalskill_v1_2_med_prism/test_lower_triangular_evaluation.py"),run_name="__main__")
if not success:raise SystemExit(1)
print("ALL SELECTED RCWP / MED-PRISM REGRESSIONS PASS")
