import csv
import json
from pathlib import Path
import statistics


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class Diagnostics:
    def __init__(self, directory, config):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows = []
        write_json(self.directory / "tpm_config.json", config.to_dict())

    def append(self, row):
        self.rows.append(row)
        with (self.directory / "tpm_bank_diagnostics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
            f.flush()

    def finish(self, metadata):
        def aggregate(rows):
            valid = [r for r in rows if "post_cast" in r]
            edits = [r["post_cast"]["effective_relative_edit"] if r["commit"] else 0.0 for r in valid]
            stats = {"banks": len(rows), "committed": sum(r["commit"] for r in rows),
                     "rejected": sum(not r["commit"] for r in rows),
                     "commit_rate": sum(r["commit"] for r in rows)/len(rows) if rows else None,
                     "mean_accepted_relative_edit": statistics.mean(edits) if edits else None,
                     "median_accepted_relative_edit": statistics.median(edits) if edits else None}
            for split, key in (("fit", "post_cast"), ("holdout", "holdout")):
                before = sum(r[key]["read_error_before"] for r in valid)
                after = sum(r[key]["read_error_after"] if r["commit"] else r[key]["read_error_before"] for r in valid)
                stats.update({split+"_error_before": before, split+"_error_after_accepted": after,
                              split+"_relative_improvement": (before-after)/before if before > 1e-12 else None})
            energies = [r["captured_repair_spectral_energy"] for r in valid
                        if r.get("captured_repair_spectral_energy") is not None]
            stats["mean_captured_spectral_energy"] = statistics.mean(energies) if energies else None
            return stats
        summary = {**metadata, "all": aggregate(self.rows), "groups": {}}
        for key in ("projection", "depth", "block"):
            for value in sorted({r[key] for r in self.rows}):
                summary["groups"][f"{key}:{value}"] = aggregate([r for r in self.rows if r[key] == value])
        write_json(self.directory / "tpm_summary.json", summary)
        rows = [{"group": "all", **summary["all"]}] + [{"group": k, **v} for k, v in summary["groups"].items()]
        with (self.directory / "tpm_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return summary
