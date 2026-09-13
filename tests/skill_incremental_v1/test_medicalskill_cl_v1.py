import importlib.util,json,tempfile,unittest
from unittest import mock
from pathlib import Path
ROOT=Path("/root/MedAgentCL_v4")
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
build=load("v1_build","scripts/skill_incremental_v1/build_medicalskill_cl_v1.py")
extract=load("v1_extract","scripts/skill_incremental_v1/extract_medtrinity_concepts_v1.py")
evaluate=load("v1_eval","scripts/skill_incremental_v1/evaluate_medicalskill_predictions_v1.py")
phash=load("v1_phash","scripts/skill_incremental_v1/phash_closure.py")
class V1ContractTest(unittest.TestCase):
 def row(self,task,skill,image="/tmp/ISIC_0001234.jpg",source="ISIC2019_1",case="ISIC_0001234"):
  return {"id":source,"task_id":task,"skill":skill,"messages":[{"role":"user","content":"<image>"},{"role":"assistant","content":"A"}],"images":[image],"metadata":{"source_dataset":"ISIC2019","source_id":source,"case_id":case}}
 def test_exactly_five_skills_no_caption_contract(self):
  self.assertEqual(list(build.TASKS),[1,2,3,4,5]);self.assertNotIn("caption"+"_"+"generation",json.dumps(build.TASKS))
 def test_dataset_aware_isic_lineage(self):
  tokens=build.lineage_tokens(self.row(1,"vqa"),{})
  self.assertIn("isic:isic_0001234",tokens);self.assertTrue(any(x.startswith("image:isic:") for x in tokens))
 def test_group_lineage_is_dataset_namespaced(self):
  row=self.row(1,"vqa");row["metadata"]["group_key"]="shared-group"
  self.assertIn("group:isic:shared_group",build.lineage_tokens(row,{}))
  self.assertNotIn("group:medtrinity:shared_group",build.lineage_tokens(row,{}))
 def test_sha_is_lineage(self):
  row=self.row(1,"vqa",image="/tmp/x.jpg")
  self.assertIn("sha256:abc",build.lineage_tokens(row,{"/tmp/x.jpg":{"sha256":"abc"}}))
 def test_concept_parser_drops_overlay_artifact(self):
  raw='{"concepts":[{"canonical":"green box","type":"device","status":"present","laterality":"unspecified","mention":"green box"},{"canonical":"lung nodule","type":"finding","status":"present","laterality":"right","mention":"right lung nodule"}]}'
  got=extract.parse(raw,"A green box surrounds a right lung nodule.")
  self.assertEqual([x["canonical"] for x in got],["lung nodule"])
  self.assertIn("drop_artifact",extract.LAST_REPAIRS)
 def test_concept_parser_drops_unsupported_canonical(self):
  raw='{"concepts":[{"canonical":"pneumothorax","type":"finding","status":"present","laterality":"right","mention":"right-center"}]}'
  self.assertEqual(extract.parse(raw,"A marker is at the right-center."),[])
 def test_concept_synonym_and_negation_repairs(self):
  raw='{"concepts":[{"canonical":"CT scan","type":"modality","status":"present","laterality":"unspecified","mention":"CT scan"},{"canonical":"effusion","type":"finding","status":"present","laterality":"left","mention":"effusion"}]}'
  got=extract.parse(raw,"CT scan shows no effusion.")
  self.assertEqual(got[0]["canonical"],"computed tomography")
  self.assertEqual(got[1]["status"],"negated")
  self.assertIn("normalize_synonym",extract.LAST_REPAIRS);self.assertIn("repair_negation",extract.LAST_REPAIRS)
 def test_concept_max_eight(self):
  items=",".join('{"canonical":"finding %d","type":"finding","status":"present","laterality":"unspecified","mention":"finding %d"}'%(i,i) for i in range(12))
  got=extract.parse('{"concepts":['+items+']}'," ".join("finding %d"%i for i in range(12)))
  self.assertEqual(len(got),8)
 def test_medsg_group_sampling_preserves_turns(self):
  data=[]
  for task in range(1,9):
   for group in range(4):
    for turn in range(2):
     x=self.row(4,"visual_grounding",image=f"/tmp/{task}_{group}.png",source=f"{task}_{group}_{turn}");x["metadata"].update({"medsg_task":task,"source_id":f"g{group}"});data.append(x)
  selected,report=build.balance_medsg(data,42,target=4)
  self.assertEqual(len(selected),32);self.assertEqual(report["status"],"PASS")
  counts={}
  for x in selected:counts.setdefault((x["metadata"]["medsg_task"],x["metadata"]["source_id"]),0);counts[(x["metadata"]["medsg_task"],x["metadata"]["source_id"])]+=1
  self.assertTrue(all(v==2 for v in counts.values()))
 def test_lineage_test_preservation_and_train_removal(self):
  test=self.row(1,"vqa",source="official");train=self.row(2,"diagnosis_classification",source="train")
  data={i:{"train":[],"test":[]} for i in build.TASKS};data[1]["test"]=[test];data[2]["train"]=[train]
  report,_=build.apply_lineage(data,{})
  self.assertEqual(len(data[1]["test"]),1);self.assertEqual(len(data[2]["train"]),0);self.assertEqual(report["test_records_removed"],0)
 def test_concept_metric_preserves_status(self):
  self.assertNotEqual(evaluate.concepts("negated: effusion"),evaluate.concepts("effusion"))
 def test_grounding_metric_is_task_four_contract(self):
  self.assertEqual(evaluate.ground("<box>0,0,10,10</box>","<box>0,0,10,10</box>"),1)
 def test_phash_classification_checks_all_task_split_representatives(self):
  h="0000000000000000";refs=[(1,"train",f"a{i}",f"/a{i}",h,["case:shared"],None) for i in range(3)]+[(2,"test","b","/b",h,["case:shared"],None)]
  with mock.patch.object(phash,"similarity",return_value=(1.0,10.0,10.0)):
   label,best=phash.classify(refs,h,h)
  self.assertEqual(label,"confirmed_same_source_case");self.assertIsNotNone(best)
 def test_phash_radius_boundary(self):
  near=f"{((1<<0)|(1<<13)|(1<<26)|(1<<39)):016x}";percept={"0000000000000000":[(1,"train","a","/a","0000000000000000",[],None)],near:[(2,"test","b","/b",near,[],None)]}
  got=list(phash.candidates(percept,4));self.assertEqual(len(got),1);self.assertEqual(got[0][2],4)
if __name__=="__main__":unittest.main()
