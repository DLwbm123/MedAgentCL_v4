"""Publish ONLY a new current private component; original manifests stay immutable."""
import json
from pathlib import Path
import torch
from safetensors.torch import save_file
from med_prism.transport.checkpoint import AdapterState,read_component,sha256_file
from med_prism.transport.diagnostics import write_json


def read_export(path):
    path=Path(path);value=json.loads(path.read_text())
    if value['format']!='MedPRISM_MS_CRC_state_v1' or value['method_version']!='2.2':raise ValueError('Wrong exported schema')
    for p,h in value['manifest_sha256'].items():
        if sha256_file(p)!=h:raise ValueError('Export manifest changed')
    return AdapterState(3,value['shared_manifest'],value['private_manifests'],str(path)).validate()


def export(source,values,destination,g):
    destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    original=source.tensors()
    if values.keys()!=original.keys():raise ValueError('Export schema changed')
    for name,t in original.items():
        if t.dtype!=values[name].dtype or t.shape!=values[name].shape:raise ValueError('Export dtype/shape changed')
        if not ('.task_0003__' in name and name.endswith('.B')) and not torch.equal(t,values[name]):
            raise ValueError('Forbidden export edit '+name)
    header,current=read_component(source.private_manifests[-1])
    destination.mkdir(parents=True)
    weights=destination/header['weights_file']
    save_file({k:values[k].contiguous() for k in current},str(weights),metadata={'format':'pt','method':'Med-PRISM-v2.2-MS-CRC'})
    header.pop('tpm_state',None)
    header.update(weights_sha256=sha256_file(weights),source_checkpoint=source.private_manifests[-1],
        ms_crc={'method_version':'2.2','coefficients':list(g),'A_unchanged':True,'rank_scaling_unchanged':True})
    manifest=destination/'private_manifest.json';write_json(manifest,header)
    paths=[source.shared_manifest,*source.private_manifests[:-1],str(manifest)]
    write_json(destination/'state.json',{'format':'MedPRISM_MS_CRC_state_v1','method_version':'2.2',
        'method_name':'Med-PRISM-v2.2-MS-CRC','task_id':3,'shared_manifest':paths[0],
        'private_manifests':paths[1:],'manifest_sha256':{p:sha256_file(p) for p in paths},
        'source_state':source.origin,'tpm_enabled':False,'rcwp_enabled':False})
    deployed=read_export(destination/'state.json')
    reloaded=deployed.tensors()
    if any(not torch.equal(t,reloaded[k]) for k,t in values.items()):raise RuntimeError('Export tensor reload mismatch')
    return deployed
