"""Train-derived 3-skill pilot with global grouped-development exclusion."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from .calibration import stable_id
from .checkpoint import sha256_file
from .cli import DEFAULT_DATA, TASKS
from .diagnostics import write_json


def hash_key(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def group_aliases(row):
    meta = row.get("metadata") or {}
    source = str(meta.get("source_dataset", "unknown_source"))
    invalid = {"", "none", "null", "unknown", "nan", "n/a", "-1"}
    keys = []
    for field in ("patient_id", "study_id", "case_id", "group_key", "lineage_group_id"):
        value = meta.get(field, row.get(field))
        if value is not None and str(value).lower() not in invalid:
            keys.append(f"{source}:{field}:{value}")
    for value in meta.get("image_sha256s", []):
        if value:
            keys.append(f"image_sha256:{value}")
    # Shared images link groups even when patient IDs are absent/inconsistent.
    for value in row.get("images", []):
        if not isinstance(value, str):
            raise ValueError("Pilot source requires filesystem image references")
        keys.append("image_path:" + str(Path(value).resolve()))
    if not keys:
        raise ValueError(f"No reliable grouping metadata or image reference: {stable_id(row)}")
    return keys


def build_split(task_rows, *, train_count=2000, dev_count=256, seed=42):
    entries = [(task, row) for task, rows in sorted(task_rows.items()) for row in rows]
    parent = list(range(len(entries)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    aliases = {}
    seen = set()
    for i,(task,row) in enumerate(entries):
        identity = (task, stable_id(row))
        if identity in seen:
            raise ValueError(f"Duplicate sample identity: {identity}")
        seen.add(identity)
        for alias in group_aliases(row):
            if alias in aliases:
                parent[find(i)] = find(aliases[alias])
            aliases[alias] = i
    groups = defaultdict(list)
    for i, entry in enumerate(entries):
        groups[find(i)].append(entry)
    def signature(group):
        return min(f"{t}:{stable_id(r)}" for t,r in group)
    ordered = sorted(groups.values(), key=lambda g: hash_key(seed, "group:"+signature(g)))
    dev_pool, train_pool = defaultdict(list), defaultdict(list)
    selected_group_ids = []
    for group in ordered:
        counts = Counter(t for t,_ in group)
        if any(len(dev_pool[t]) < dev_count for t in counts):
            selected_group_ids.append(hash_key(seed,signature(group)))
            for task,row in group:
                dev_pool[task].append(row)
        else:
            for task,row in group:
                train_pool[task].append(row)
    splits, report = {}, {}
    for task in task_rows:
        if len(train_pool[task]) < train_count or len(dev_pool[task]) < dev_count:
            raise ValueError(f"Insufficient grouped data for task {task}: train={len(train_pool[task])}, dev={len(dev_pool[task])}")
        train = sorted(train_pool[task],key=lambda r: hash_key(seed,"train:"+stable_id(r)))[:train_count]
        dev = sorted(dev_pool[task],key=lambda r: hash_key(seed,"dev:"+stable_id(r)))[:dev_count]
        splits[task] = (train,dev)
        report[str(task)] = {"train_ids": [stable_id(r) for r in train], "development_ids": [stable_id(r) for r in dev],
            "train_count": len(train), "development_count": len(dev),
            "quarantined_extra_development_group_rows": len(dev_pool[task])-len(dev)}
    train_aliases = {k for train,_ in splits.values() for row in train for k in group_aliases(row)}
    dev_aliases = {k for _,dev in splits.values() for row in dev for k in group_aliases(row)}
    if train_aliases & dev_aliases:
        raise RuntimeError("Global train/development grouping overlap")
    return splits, {"tasks": report, "global_group_disjoint": True,
                    "grouping": "connected components of available patient/study/case/lineage and shared image identifiers across all 3 tasks",
                    "development_group_ids_sha256": selected_group_ids}


def prepare_data(source, output, *, train_count=2000, dev_count=256, seed=42):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or "MedPRISM_v2_TPM" not in str(output):
        raise ValueError("A separate MedPRISM_v2_TPM output root is required")
    if output.exists():
        manifest = json.loads((output/"pilot_data_manifest.json").read_text())
        if manifest["config"] != {"source": str(source), "train_count": train_count, "development_count": dev_count, "seed": seed}:
            raise ValueError("Existing pilot data has another frozen selection configuration")
        for item in manifest["files"]:
            if sha256_file(item["path"]) != item["sha256"]:
                raise ValueError("Pilot input/output data checksum changed")
        return manifest
    paths = {task: source/name/"train.jsonl" for task,name in TASKS.items()}
    rows = {task: [json.loads(l) for l in path.read_text().splitlines() if l.strip()] for task,path in paths.items()}
    splits, report = build_split(rows,train_count=train_count,dev_count=dev_count,seed=seed)
    output.mkdir(parents=True)
    files = [{"path": str(p), "sha256": sha256_file(p), "role": "source_training"} for p in paths.values()]
    for task,(train,dev) in splits.items():
        folder = output/TASKS[task]
        folder.mkdir()
        for name,values in (("train",train),("development",dev)):
            path = folder/(name+".jsonl")
            path.write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in values),encoding="utf-8")
            files.append({"path": str(path), "sha256": sha256_file(path), "role": name, "task": task})
    report.update(status="PASS", config={"source": str(source), "train_count": train_count,
                  "development_count": dev_count, "seed": seed}, files=files,
                  calibration="Only selected current-task TRAIN rows; development is reserved before any pilot training")
    write_json(output/"pilot_data_manifest.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-root",default=DEFAULT_DATA)
    parser.add_argument("--output-data-root",required=True)
    parser.add_argument("--train-count",type=int,default=2000)
    parser.add_argument("--development-count",type=int,default=256)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    if min(args.train_count,args.development_count)<1:
        parser.error("Counts must be positive")
    report=prepare_data(args.source_data_root,args.output_data_root,train_count=args.train_count,
                        dev_count=args.development_count,seed=args.seed)
    print(json.dumps({"status":report["status"],"output":args.output_data_root,"global_group_disjoint":True},indent=2))


if __name__ == "__main__":
    main()
