"""Train-only deterministic group split; no old-data interface during selection."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from med_prism.transport.calibration import ordered_rows,stable_id
from med_prism.transport.pilot_data import group_aliases


def encode(row,template):
    encoded=template.encode(copy.deepcopy({k:row[k] for k in ('messages','images','videos') if k in row}))
    batch=template.data_collator([encoded])
    if 'attention_mask' not in batch:batch['attention_mask']=torch.ones_like(batch['input_ids'])
    if batch['input_ids'].shape[0]!=1 or 'labels' not in batch:raise ValueError('Expected teacher-forced batch one')
    if not (batch['labels'][:,1:]!=-100).any():raise ValueError('No answer tokens')
    return batch


def current_split(path,template,fit_count=128,holdout_count=64):
    if type(fit_count) is not int or type(holdout_count) is not int or min(fit_count,holdout_count)<1:
        raise ValueError('Fit/holdout counts must be positive integers')
    from swift.template import MaxLengthError
    path=Path(path)
    if path.name!='train.jsonl' or path.parent.name!='task_03_concept_recognition':
        raise ValueError('Selection can only access current Task3 train.jsonl')
    rows=ordered_rows([json.loads(l) for l in path.read_text().splitlines() if l.strip()],42)
    parent=list(range(len(rows)));seen={}
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,row in enumerate(rows):
        for a in group_aliases(row):
            if a in seen:parent[root(i)]=root(seen[a])
            seen[a]=i
    groups=set();fit=[];holdout=[];rejected=0
    for i,row in enumerate(rows):
        if len(fit)>=fit_count and root(i) in groups:continue
        try:batch=encode(row,template)
        except MaxLengthError:rejected+=1;continue
        record={'row':row,'batch':batch}
        if len(fit)<fit_count:fit.append(record);groups.add(root(i))
        else:holdout.append(record)
        if len(holdout)==holdout_count:break
    if len(fit)!=fit_count or len(holdout)!=holdout_count:raise ValueError('Insufficient group-disjoint current data')
    def digest(records):
        h=hashlib.sha256()
        for r in records:
            h.update(stable_id(r['row']).encode())
            for k in ('input_ids','labels','attention_mask'):h.update(r['batch'][k].contiguous().numpy().tobytes())
        return h.hexdigest()
    return fit,holdout,{'fit_sha256':digest(fit),'holdout_sha256':digest(holdout),
        'fit_samples':len(fit),'holdout_samples':len(holdout),'group_disjoint':True,'rejections':rejected,
        'expansion':'disabled; fixed128+64, no post-unblinding expansion'}
