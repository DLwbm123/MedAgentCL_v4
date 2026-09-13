"""Full cumulative accepted-state index over unchanged v1 component format.

Every boundary saves ALL private banks, including already transported historical
keys. Existing v1 compose/load APIs can consume the resulting manifest paths.
No full base checkpoint or optimizer state is duplicated by TPM.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import uuid
import torch
from safetensors.torch import load_file, save_file
from .diagnostics import write_json


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_component(path):
    path = Path(path).resolve()
    header = json.loads(path.read_text())
    if header.get("format_version") != "med_prism_shared_private_v1" or header.get("merged") is not False:
        raise ValueError(f"Unsupported component: {path}")
    if header.get("private_adapter_type", "rank1_expert_bank") != "rank1_expert_bank":
        raise ValueError("TPM v2 requires no-geo rank-1 banks")
    weights = path.parent / header["weights_file"]
    if sha256_file(weights) != header["weights_sha256"]:
        raise ValueError(f"Component checksum mismatch: {weights}")
    tensors = load_file(str(weights))
    schema = {v["key"]: v for v in header["tensor_schema"]}
    if tensors.keys() != schema.keys() or len(tensors) != header["tensor_count"]:
        raise ValueError("Component key schema mismatch")
    for k, v in tensors.items():
        if list(v.shape) != schema[k]["shape"] or str(v.dtype).removeprefix("torch.") != schema[k]["dtype"]:
            raise ValueError(f"Component tensor schema mismatch: {k}")
    return header, tensors


@dataclass
class AdapterState:
    task_id: int
    shared_manifest: str
    private_manifests: list[str]
    origin: str

    def components(self):
        return [self.shared_manifest, *self.private_manifests]

    def tensors(self):
        result = {}
        for path in self.components():
            _, state = read_component(path)
            if result.keys() & state.keys():
                raise ValueError("Duplicate component tensors")
            result.update(state)
        return result

    def validate(self):
        headers = [read_component(p)[0] for p in self.components()]
        if headers[0]["component_type"] != "shared" or headers[0]["task_id"] != self.task_id:
            raise ValueError("Wrong shared stage")
        if [h["task_id"] for h in headers[1:]] != list(range(1, self.task_id + 1)):
            raise ValueError("Accepted state must contain every task exactly once, in order")
        for key in ("model_id", "model_revision", "target_module_hash", "target_modules"):
            if any(h[key] != headers[0][key] for h in headers):
                raise ValueError(f"Component disagreement: {key}")
        return self


def read_state(path, *, require_accepted=False):
    path = Path(path).resolve()
    if path.is_dir():
        path /= "state.json"
    value = json.loads(path.read_text())
    if value["format"] != "MedPRISM_v2_TPM_state_v1" or value["status"] not in {"ACCEPTED", "PRE_TPM"}:
        raise ValueError("State index is not an accepted or pre-TPM snapshot")
    if require_accepted and value["status"] != "ACCEPTED":
        raise ValueError("Teacher/previous state must be ACCEPTED, not merely PRE_TPM")
    def resolve(value):
        return str((path.parent / value).resolve())
    state = AdapterState(value["task_id"], resolve(value["shared_manifest"]),
                         [resolve(p) for p in value["private_manifests"]], str(path)).validate()
    for component in state.components():
        rel = str(Path(component).relative_to(path.parent))
        if sha256_file(component) != value["manifest_sha256"][rel]:
            raise ValueError("Accepted component manifest changed after publication")
    return state


def legacy_state(root, task):
    root = Path(root).resolve()
    stage = root / "med_prism" / f"stage_{task:02d}"
    completion = json.loads((stage / "completion.json").read_text())
    if completion.get("status") != "PASS":
        raise ValueError("Legacy training stage is not PASS")
    config = json.loads((root / "configs" / f"stage_{task:02d}.json").read_text())
    if config["orth_lambda"] != 0 or config["key_lambda"] != .1 or config["shared_drift_lambda"] != .01:
        raise ValueError("Checkpoint repair pilot requires the no-geo recipe")
    return AdapterState(task, str(stage / "shared/shared_manifest.json"),
                         [*config["private_source_manifests"], str(stage / "private/private_manifest.json")],
                         str(stage)).validate()


def validate_transition(teacher, student):
    if teacher.task_id != student.task_id - 1:
        raise ValueError("Teacher must be exactly the previous accepted stage")
    previous = teacher.tensors()
    current = student.tensors()
    for key, value in previous.items():
        if ".experts." in key and (key not in current or not torch.equal(value, current[key])):
            raise ValueError(f"Historical bank changed during gradient training or stale teacher: {key}")
    for old, new in zip(teacher.private_manifests, student.private_manifests):
        a, _ = read_component(old)
        b, _ = read_component(new)
        for field in ("alpha", "scaling", "rank", "experts_per_task", "target_module_hash"):
            if a[field] != b[field]:
                raise ValueError(f"Historical metadata mismatch: {field}")


@torch.no_grad()
def apply_tensors(model, state):
    params = dict(model.named_parameters())
    for key, value in state.items():
        if key not in params or params[key].shape != value.shape:
            raise ValueError(f"Adapter swap schema mismatch: {key}")
        params[key].copy_(value.to(params[key]))


def materialize_state(source, destination, *, model=None, accepted=False, metadata=None):
    """Publish a new directory atomically; refuse overwrite, even incomplete outputs.

    If model is absent, copies only adapter components from source (exact no-op).
    A crash leaves a uniquely named .building directory; never treated as accepted.
    """
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing checkpoint overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    building = destination.with_name(destination.name + ".building_" + uuid.uuid4().hex)
    building.mkdir()
    params = dict(model.named_parameters()) if model is not None else None
    shared, private, hashes = None, [], {}
    try:
        for path in source.components():
            header, original = read_component(path)
            relative = Path("shared") if header["component_type"] == "shared" else Path("private") / f"task_{header['task_id']:04d}"
            folder = building / relative
            folder.mkdir(parents=True)
            weights = folder / header["weights_file"]
            if params is None:
                # Network storage may reject extended-attribute copying. Only
                # checkpoint bytes (verified below), not filesystem xattrs, matter.
                shutil.copyfile(Path(path).parent / header["weights_file"], weights)
                values = original
            else:
                values = {}
                for key, old in original.items():
                    if params[key].shape != old.shape:
                        raise ValueError("Refusing changed checkpoint shapes")
                    values[key] = params[key].detach().cpu().contiguous()
                save_file(values, str(weights), metadata={"format": "pt", "method": "med_prism_shared_private"})
            header.update(weights_sha256=sha256_file(weights), source_checkpoint=str(path),
                          tensor_schema=[{"key": k, "shape": list(v.shape), "dtype": str(v.dtype).removeprefix("torch.")}
                                         for k, v in sorted(values.items())],
                          dtype=sorted({str(v.dtype).removeprefix("torch.") for v in values.values()}),
                          tpm_state={"version": "2.0", "boundary_task": source.task_id,
                                     "historical_keys_boundary_editable": True, "historical_values_fixed": True})
            rel_manifest = relative / Path(path).name
            write_json(building / rel_manifest, header)
            hashes[str(rel_manifest)] = sha256_file(building / rel_manifest)
            if header["component_type"] == "shared":
                shared = str(rel_manifest)
            else:
                private.append(str(rel_manifest))
            loaded = load_file(str(weights))
            if loaded.keys() != values.keys() or any(not torch.equal(loaded[k], v) for k, v in values.items()):
                raise RuntimeError("Checkpoint write/reload was not exact")
        write_json(building / "state.json", {
            "format": "MedPRISM_v2_TPM_state_v1", "method": "Med-PRISM v2.0 TPM",
            "status": "ACCEPTED" if accepted else "PRE_TPM", "task_id": source.task_id,
            "shared_manifest": shared, "private_manifests": private, "manifest_sha256": hashes,
            "source_state": source.origin, "metadata": metadata or {},
            "gradient_recipe": "unchanged v1.2 no_geo_orth", "inference": "unchanged v1 component compose API"})
        building.rename(destination)
    except Exception:
        # Preserve partial files for diagnosis; never publish state at destination.
        raise
    return read_state(destination)
