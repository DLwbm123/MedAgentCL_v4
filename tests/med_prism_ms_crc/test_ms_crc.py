import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from safetensors.torch import save_file
from med_prism.adapters.shared_private import SharedPrivateLinear
from med_prism.ms_crc.composition import Composition,group_id,coefficients
from med_prism.ms_crc.search import (IDENTITY,witness,positive_set,objective,feasible,search,accept_holdout,panel)
from med_prism.ms_crc.fold import export,read_export
from med_prism.ms_crc.gate import lock,validate_lock,locked_dev
from med_prism.ms_crc.evaluator import decoded_correct,equivalent
from med_prism.transport.checkpoint import AdapterState,sha256_file,apply_tensors


def toy():
    torch.manual_seed(4);m=nn.Module();m.model=nn.Module();m.model.layers=nn.ModuleList()
    for i in range(36):
        layer=nn.Module();layer.self_attn=nn.Module()
        for key in ('q_proj','v_proj'):
            w=SharedPrivateLinear(nn.Linear(3,3,bias=False),module_identity=f'model.layers.{i}.self_attn.{key}',
                shared_rank=2,shared_alpha=2,shared_lr_scale=1,dropout=0,shared_trainable=False)
            for t in (1,2,3):
                w.add_private_task(t,private_rank=16,experts_per_task=16,alpha=16,trainable=False)
                for e in w.task_experts(t):
                    with torch.no_grad():e.B.normal_(std=.1)
            setattr(layer.self_attn,key,w)
        m.model.layers.append(layer)
    return m.eval().requires_grad_(False)


def source(m,root):
    paths=[];params=dict(m.named_parameters());targets=list(Composition(m).modules)
    for kind,t in [('shared',3),('private',1),('private',2),('private',3)]:
        folder=root/f'{kind}{t}';folder.mkdir(parents=True)
        marker='.shared.' if kind=='shared' else f'.experts.task_{t:04d}__'
        values={k:p.detach().clone() for k,p in params.items() if marker in k}
        weights=folder/'weights.safetensors';save_file(values,str(weights))
        header={'format_version':'med_prism_shared_private_v1','merged':False,'component_type':kind,
            'task_id':t,'rank':16 if kind=='private' else 2,'alpha':16 if kind=='private' else 2,
            'scaling':1,'experts_per_task':16,'private_adapter_type':'rank1_expert_bank',
            'weights_file':weights.name,'weights_sha256':sha256_file(weights),'model_id':'toy','model_revision':'toy',
            'target_modules':targets,'target_module_hash':'toy','tensor_count':len(values),
            'tensor_schema':[{'key':k,'shape':list(v.shape),'dtype':str(v.dtype).split('.')[-1]} for k,v in values.items()]}
        p=folder/f'{kind}_manifest.json';p.write_text(json.dumps(header));paths.append(str(p))
    origin=root/'state.json';origin.write_text('{}')
    return AdapterState(3,paths[0],paths[1:],str(origin)).validate()


def rows(n=64,correct=True,margin=1.,nll=1.):
    return [{'sample_id':str(i),'prediction':'x','correct':correct,'margin':margin,'nll_sum':nll,'answer_tokens':1} for i in range(n)]


class MSCRCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m=toy()

    def test_zero_based_mapping(self):
        self.assertEqual([group_id(i,0) for i in (0,8,9,17,18,26,27,35)],[0,0,4,4,8,8,12,12])
        self.assertEqual([group_id(0,i) for i in range(16)],[i//4 for i in range(16)])
        with self.assertRaises(ValueError):group_id(36,0)

    def test_group_coverage(self):
        c=Composition(self.m)
        self.assertEqual(Counter(v['group_id'] for v in c.group_map),{i:72 for i in range(16)})

    def test_identity_and_immutable_candidates(self):
        c=Composition(self.m);w=next(iter(c.modules.values()));x=torch.randn(1,5,3)
        before={k:v.clone() for k,v in self.m.state_dict().items()};original=w(x)
        g=(.5,)+(1.,)*15
        with c.use(g):first=w(x)
        with c.use(g):second=w(x)
        with c.use(IDENTITY):identity=w(x)
        self.assertTrue(torch.equal(first,second));self.assertTrue(torch.equal(original,identity))
        after=self.m.state_dict()
        self.assertTrue(all(torch.equal(v,after[k]) for k,v in before.items()))

    def test_B_fold_A_history_unchanged_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);s=source(self.m,root/'source');original=s.tensors();c=Composition(self.m)
            g=(.5,)+(1.,)*15;values=c.exported_tensors(original,g)
            deployed=export(s,values,root/'exported',g)
            actual=read_export(deployed.origin).tensors()
            self.assertTrue(all(torch.equal(v,actual[k]) for k,v in values.items()))
            self.assertTrue(all(torch.equal(v,actual[k]) for k,v in original.items() if not ('.task_0003__' in k and k.endswith('.B'))))
            m2=toy();apply_tensors(m2,actual);native=Composition(m2);w2=next(iter(native.modules.values()))
            w=next(iter(c.modules.values()));x=torch.randn(1,4,3)
            with c.use(g):expected=w(x)
            self.assertTrue(torch.equal(expected,w2(x)))
            based_on_original=Composition(m2,original_state=original)
            with based_on_original.use(IDENTITY):self.assertTrue(torch.equal(w(x),w2(x)))

    def test_bf16_native_folding(self):
        m=toy().to(torch.bfloat16);c=Composition(m);w=next(iter(c.modules.values()));x=torch.randn(1,2,3,dtype=torch.bfloat16)
        orig={k:v.clone() for k,v in m.state_dict().items()};g=(.5,)+(1.,)*15
        with c.use(g):a=w(x)
        # A second independently constructed folded model, same scaling/rank.
        folded=toy().to(torch.bfloat16);folded.load_state_dict(orig)
        c2=Composition(folded);params={k:v.detach().cpu().clone() for k,v in folded.named_parameters() if '.shared.' in k or '.experts.' in k}
        apply_tensors(folded,c2.exported_tensors(params,g));b=next(iter(c2.modules.values()))(x)
        self.assertTrue(torch.equal(a,b))

    def test_lexicographic_behavior_first(self):
        H=rows();W=witness(H,{1:rows(margin=0),2:rows(margin=0)},0)
        right=rows(margin=-100);wrong=rows(correct=False,margin=100)
        self.assertLess(objective(right,H,W,IDENTITY,(1,2)),objective(wrong,H,W,IDENTITY,(1,2)))

    def test_identity_always_feasible(self):
        self.assertTrue(feasible(rows(correct=False,nll=100),rows(),(0,),IDENTITY))

    def test_fixed_witness_and_N_constraints(self):
        H=rows();H[0]['correct']=False;full=rows();W=witness(H,{1:rows(margin=0),2:rows(margin=0)},0)
        snapshot=copy.deepcopy(W);N=positive_set(H,full);self.assertEqual(N,(0,))
        bad=rows();bad[0]['correct']=False
        self.assertFalse(feasible(bad,full,N,(0.,)+(1.,)*15));self.assertEqual(W,snapshot)

    def test_holdout_rejection_identity(self):
        H=rows();full=rows();W=witness(H,{1:rows(margin=0),2:rows(margin=0)},0)
        g=(0.,)+(1.,)*15
        selected,status=accept_holdout(g,rows(correct=False),full,H,W,(1,2))
        self.assertEqual(selected,IDENTITY);self.assertEqual(status,'REJECTED_CURRENT_HOLDOUT')

    def test_no_signal_no_search(self):
        H=rows();W=witness(H,{1:H,2:H},0)
        def forbidden(g):raise AssertionError('No-signal must not search')
        g,info=search(forbidden,H,H,W,lambda x:None)
        self.assertEqual(g,IDENTITY);self.assertEqual(info['status'],'NO_SIGNAL')

    def test_nonadditive_search_recomputes(self):
        H=rows(margin=10);full=rows(margin=0);W=witness(H,{1:rows(margin=0),2:rows(margin=0)},0);calls=[]
        def score(g):
            calls.append(g)
            # Both alone help, but together reverse the benefit.
            a,b=g[0]==0,g[1]==0
            return rows(margin=3 if a!=b else (-5 if a and b else 0))
        g,info=search(score,H,full,W,lambda x:None)
        self.assertTrue(any(v[0]==0 and v[1]==0 for v in calls))
        self.assertFalse(g[0]==0 and g[1]==0)

    def test_lock_required_before_old_path_access(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):locked_dev(Path(d),'/never/read/task_02_diagnosis_classification/development.jsonl')

    def test_selection_rejects_old_training_and_dev(self):
        from med_prism.ms_crc.data import current_split
        for path in ('/missing/task_02_diagnosis_classification/train.jsonl','/missing/task_03_concept_recognition/development.jsonl'):
            with self.assertRaises(ValueError):current_split(path,None)

    def test_split_count_types_and_no_duplicate_config_keys(self):
        import ast,inspect
        import med_prism.ms_crc.gate as gate
        from med_prism.ms_crc.data import current_split
        with self.assertRaises(ValueError):current_split('/missing',None,2,'holdout rule')
        for node in ast.walk(ast.parse(inspect.getsource(gate))):
            if isinstance(node,ast.Dict):
                keys=[k.value for k in node.keys if isinstance(k,ast.Constant)]
                self.assertEqual(len(keys),len(set(keys)))

    def test_current_micro_f1_preserved_as_distinct_metric(self):
        from med_prism.ms_crc.analysis import concept_f1,concept_ci
        a=[{'all_tp':2,'all_fp':0,'all_fn':0}]*4
        b=[{'all_tp':1,'all_fp':1,'all_fn':1}]*4
        self.assertEqual(concept_f1(a),1.)
        self.assertEqual(concept_ci(a,b)['delta'],.5)

    def test_locked_panel_no_signal_cannot_pass(self):
        from med_prism.ms_crc.analysis import conclude
        vectors=panel(IDENTITY)
        current=[{**r,'all_tp':1,'all_fp':0,'all_fn':0} for r in rows(8)]
        panels={3:{k:copy.deepcopy(current) for k in vectors},
                2:{k:rows(8) for k in vectors},1:{k:rows(8) for k in vectors}}
        result=conclude(panels,'NO_SIGNAL')
        self.assertEqual(result['status'],'NO_SIGNAL')
        self.assertEqual(result['task2_gain']['delta'],0.)
        self.assertFalse(result['current_positive_transfer_evaluable'])

    def test_lock_hash_tampering_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);s=source(self.m,root/'source');values=s.tensors();deployed=export(s,values,root/'exported',IDENTITY)
            (root/'request.json').write_text('{}');lock(root,s,deployed,IDENTITY,{}, {'x':'hash'})
            self.assertTrue(validate_lock(root)['SELECTION_LOCKED'])
            (root/'selected_coefficients.json').write_text('{}')
            with self.assertRaises(RuntimeError):validate_lock(root)

    def test_panel_permutation_and_global_alpha(self):
        g=(0.,.5)+(1.,)*14;a=panel(g);b=panel(g)
        self.assertEqual(a,b);self.assertEqual(sorted(a['permutation']),sorted(g))
        for alpha in (0.,.25,.5,.75,1.):self.assertEqual(a[f'global_{alpha}'],(alpha,)*16)

    def test_coefficient_limits(self):
        with self.assertRaises(ValueError):coefficients((-1.,)+(1.,)*15)
        with self.assertRaises(ValueError):coefficients((0.,)*5+(1.,)*11,True)

    def test_existing_parser_and_reload_gate(self):
        self.assertTrue(decoded_correct(3,{'all_fp':0,'all_fn':0}))
        self.assertFalse(decoded_correct(3,{'all_fp':0,'all_fn':1}))
        a=rows(2);equivalent(a,copy.deepcopy(a),0.)
        b=copy.deepcopy(a);b[0]['margin']+=.00001
        with self.assertRaises(RuntimeError):equivalent(a,b,0.)

    def test_no_trainer_RCWP_TPM_search_dependencies(self):
        import inspect
        import med_prism.ms_crc.search as module
        source_=inspect.getsource(module)
        for text in ('development.jsonl','train.jsonl','teacher','optimizer.step','rcwp','transport_keys'):
            self.assertNotIn(text,source_.replace('teachers',''))

    def test_algebra_selectivity_telescoping_and_interaction(self):
        A=torch.eye(2);B=torch.eye(2);g=torch.tensor([0.,1.]);x=torch.tensor([1.,1.])
        self.assertTrue(torch.equal((B*g)@A@x,B@torch.diag(g)@A@x))
        self.assertFalse(torch.equal((B*g)@x,.5*B@x))
        f=lambda a,b:a+b-.5*a*b
        self.assertAlmostEqual((f(1,0)-f(0,0))+(f(1,1)-f(1,0)),f(1,1)-f(0,0))
        self.assertLess(f(1,1)-f(1,0)-f(0,1)+f(0,0),0)
        self.assertGreater(f(1,1)-f(0,0),0)  # Negative interaction need not be harmful.


if __name__=='__main__':unittest.main()
