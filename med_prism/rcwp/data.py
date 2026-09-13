"""Current training rows only, group-disjoint temporary fit/holdout partitions."""
import copy
import torch
from med_prism.transport.calibration import ordered_rows,stable_id
from med_prism.transport.pilot_data import group_aliases


def encode_boundary_rows(rows,template,fit_count=128,holdout_count=64):
    from swift.template import MaxLengthError
    rows=ordered_rows(rows,42)
    parent=list(range(len(rows)));aliases={}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(rows):
        for key in group_aliases(row):
            if key in aliases: parent[find(i)]=find(aliases[key])
            aliases[key]=i
    records=[];fit_groups=set();rejected=0
    for i,row in enumerate(rows):
        group=find(i)
        partition="fit" if len(records)<fit_count else "holdout"
        if partition=="holdout" and group in fit_groups:continue
        try:
            encoded=template.encode(copy.deepcopy({k:row[k] for k in ("messages","images","videos") if k in row}))
        except MaxLengthError:
            rejected+=1;continue
        batch=template.data_collator([encoded])
        if "input_ids" not in batch or batch["input_ids"].shape[0]!=1 or "labels" not in batch:
            raise ValueError("Boundary requires unpacked teacher-forced batch size one")
        if "attention_mask" not in batch:batch["attention_mask"]=torch.ones_like(batch["input_ids"])
        if partition=="fit":fit_groups.add(group)
        records.append({"batch":batch,"split":partition})
        if len(records)==fit_count+holdout_count:break
    if len(records)!=fit_count+holdout_count:
        raise ValueError("Insufficient group-disjoint encodable current training rows")
    # IDs/group aliases/features exist only in this temporary function scope.
    return records,rejected
