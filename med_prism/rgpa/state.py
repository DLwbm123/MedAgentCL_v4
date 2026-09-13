"""Small, atomic RGPA state; checkpoint-local trainable tensors for resume."""
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def save_resume(model, protection, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tensors = {n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad}
    save_file(tensors, str(path / 'rgpa_trainable.safetensors'))
    write_json(path / 'rgpa_state.json', protection.state())


@torch.no_grad()
def load_resume(model, path):
    path = Path(path)
    state = json.loads((path / 'rgpa_state.json').read_text())
    tensors = load_file(str(path / 'rgpa_trainable.safetensors'))
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if parameters.keys() != tensors.keys():
        raise ValueError('Resume trainable schema mismatch')
    for name, parameter in parameters.items():
        if parameter.shape != tensors[name].shape:
            raise ValueError('Resume tensor shape mismatch')
        parameter.copy_(tensors[name].to(parameter))
    return state
