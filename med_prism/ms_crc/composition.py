"""Read-only candidate forwards and immutable-origin, BF16 B-column folding."""
from contextlib import contextmanager
from collections import Counter
import re
import torch
import torch.nn.functional as F
from med_prism.adapters.injection import iter_shared_private_wrappers
from med_prism.transport.banks import tensor_hash


def group_id(layer,rank):
    if not 0<=layer<36 or not 0<=rank<16:raise ValueError('Invalid zero-based layer/rank')
    return 4*(layer//9)+rank//4


def coefficients(g,search=False):
    g=tuple(float(v) for v in g)
    if len(g)!=16 or any(v<0 or v>1 for v in g):raise ValueError('Invalid static coefficient vector')
    if search and (any(v not in (0,.5,1) for v in g) or sum(v!=1 for v in g)>4):
        raise ValueError('Outside pre-registered search space')
    return g


class Composition:
    def __init__(self,model,task=3,require_full=True,original_state=None):
        if task!=3 or model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Task3 frozen eval model required')
        self.model=model;self.task=task;self.modules=dict(iter_shared_private_wrappers(model))
        self.entries=[];self.group_map=[];self.native_original={}
        named={id(p):n for n,p in model.named_parameters()}
        for name,w in self.modules.items():
            match=re.search(r'\.layers\.(\d+)\.',name)
            if not match or name.rsplit('.',1)[-1] not in ('q_proj','v_proj'):
                raise ValueError('Unexpected wrapper '+name)
            if w.private_adapter_type!='rank1_expert_bank' or list(w.task_ids)!=[1,2,3]:
                raise ValueError('Expected cumulative rank1 P1/P2/P3')
            experts=w.task_experts(task)
            if len(experts)!=16:raise ValueError('P3 nominal rank must remain16')
            for index,e in enumerate(experts):
                gid=group_id(int(match.group(1)),index)
                original=e.B.detach().clone() if original_state is None else original_state[named[id(e.B)]].to(e.B).clone()
                self.native_original[id(e)]=torch.equal(original,e.B)
                self.entries.append((e,w,original,gid))
                self.group_map.append({'wrapper':name,'layer':int(match.group(1)),
                    'rank_idx_zero_based':index,'expert_id':e.expert_id,'group_id':gid,
                    'A_sha256':tensor_hash(e.A),'B_sha256':tensor_hash(original),'scaling_sha256':tensor_hash(e.scaling)})
        counts=Counter(v['group_id'] for v in self.group_map)
        if require_full and (len(self.modules)!=72 or len(self.entries)!=1152 or counts!={i:72 for i in range(16)}):
            raise ValueError('Group coverage is not16 x72 =1152')

    @contextmanager
    def use(self,g,active=(1,2,3)):
        g=coefficients(g);handles=[];prior={n:list(w.active_tasks) for n,w in self.modules.items()}
        if not set(active)<=set((1,2,3)):raise ValueError('Unknown active task')
        try:
            for w in self.modules.values():w.set_active_tasks(active)
            for e,w,original,gid in self.entries:
                value=g[gid]
                if value==1 and self.native_original[id(e)]:continue  # Exact native identity.
                B=original*value  # Native deployment dtype. NEVER cumulative scaling.
                def replace(module,args,out,B=B,w=w):
                    x=args[0]
                    return F.linear(F.linear(w.dropout(x),module.A),B)*module.scaling.to(x)
                handles.append(e.register_forward_hook(replace))
            yield
        finally:
            for h in handles:h.remove()
            for n,w in self.modules.items():w.set_active_tasks(prior[n])

    def exported_tensors(self,original_state,g):
        g=coefficients(g,search=True);result={k:v.clone() for k,v in original_state.items()}
        named={id(p):n for n,p in self.model.named_parameters()}
        for e,_,B,gid in self.entries:
            key=named[id(e.B)]
            if not torch.equal(original_state[key],B.cpu()):raise ValueError('Immutable B differs from source checkpoint')
            result[key]=(B*g[gid]).cpu().contiguous()
        for k,v in original_state.items():
            if not ('.task_0003__' in k and k.endswith('.B')) and not torch.equal(v,result[k]):
                raise RuntimeError('Forbidden export mutation '+k)
        return result
