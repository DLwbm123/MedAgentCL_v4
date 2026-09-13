#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import traceback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {"status": "PASS", "imports": {}, "checks": {}, "error": None}
    modules = [
        "numpy", "scipy", "pandas", "pyarrow", "datasets", "PIL",
        "torchvision", "decord", "transformers", "swift",
    ]
    try:
        for name in modules:
            module = importlib.import_module(name)
            result["imports"][name] = {
                "version": getattr(module, "__version__", None),
                "file": getattr(module, "__file__", None),
            }

        import numpy as np
        import pandas as pd
        import scipy.linalg
        import torch
        from datasets import Dataset
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor

        array = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = torch.from_numpy(array)
        result["checks"]["numpy_to_torch"] = {
            "passed": tensor.shape == (3, 4) and tensor.dtype == torch.float32,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }

        round_trip = tensor.numpy()
        result["checks"]["torch_to_numpy"] = {
            "passed": np.array_equal(array, round_trip),
            "shape": list(round_trip.shape),
            "dtype": str(round_trip.dtype),
        }

        inverse = scipy.linalg.inv(np.array([[2.0, 0.0], [0.0, 4.0]]))
        result["checks"]["scipy_numeric"] = {
            "passed": np.allclose(inverse, [[0.5, 0.0], [0.0, 0.25]]),
            "value": inverse.tolist(),
        }

        frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result["checks"]["pandas_dataframe"] = {
            "passed": frame.shape == (2, 2) and int(frame["a"].sum()) == 3,
            "shape": list(frame.shape),
        }

        dataset = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})
        result["checks"]["datasets_from_dict"] = {
            "passed": len(dataset) == 2 and dataset.column_names == ["text", "label"],
            "rows": len(dataset),
            "columns": dataset.column_names,
        }

        image = Image.new("RGB", (19, 11), color=(12, 34, 56))
        image_tensor = pil_to_tensor(image)
        result["checks"]["pil_to_torchvision"] = {
            "passed": image_tensor.shape == (3, 11, 19) and image_tensor.dtype == torch.uint8,
            "shape": list(image_tensor.shape),
            "dtype": str(image_tensor.dtype),
        }

        failed = [name for name, check in result["checks"].items() if not check["passed"]]
        if failed:
            raise AssertionError(f"Failed mixed-stack checks: {failed}")
    except Exception:
        result["status"] = "BLOCKED"
        result["error"] = traceback.format_exc()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
