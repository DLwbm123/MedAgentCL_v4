import importlib.util, json, tempfile, unittest
from pathlib import Path

ROOT=Path("/root/MedAgentCL_v4")
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
build=load("medicalskill_build","scripts/skill_incremental/build_medicalskill_cl.py")
extract=load("medicalskill_extract","scripts/skill_incremental/extract_medtrinity_concepts.py")
evaluate=load("medicalskill_eval","scripts/skill_incremental/evaluate_medicalskill_predictions.py")
validator=load("medicalskill_validator","scripts/skill_incremental/validate_medicalskill_cl.py")

class ContractTest(unittest.TestCase):
 def test_unified_schema(self):
  row=build.unified("x",1,"vqa",[{"role":"user","content":"<image>"},{"role":"assistant","content":"A"}],["/tmp/a.png"],{"source_dataset":"x"})
  self.assertEqual(list(row),["id","task_id","skill","messages","images","metadata"])
  self.assertEqual(build.validate([row])["status"],"PASS")
 def test_box_normalization(self):
  self.assertEqual(build.boxnorm([10,20,110,220],200,400),[50,50,550,550])
  self.assertIsNone(build.boxnorm([10,20,10,220],200,400))
 def test_grounding_box_dedup_preserves_distinct_boxes(self):
  self.assertEqual(build.dedupe_boxes([[1,2,3,4],[1,2,3,4],[5,6,7,8]]),[[1,2,3,4],[5,6,7,8]])
 def test_target_image_ordinals(self):
  self.assertEqual(build.targetidx(4,"locate it in the third image","",5),2)
  self.assertEqual(build.targetidx(6,"locate crop in source image","",4),0)
 def test_caption_cleaning(self):
  text,rules=build.clean_caption("<p>Finding.</p> Finding.")
  self.assertNotIn("<p>",text); self.assertIn("strip_html",rules)
 def test_caption_rules_have_arrow_stable_storage(self):
  self.assertEqual(build.caption_rule_names("[]"),[])
  self.assertEqual(build.caption_rule_names('["strip_html"]'),["strip_html"])
 def test_concept_parser_requires_support(self):
  raw='{"concepts":[{"canonical":"liver","type":"anatomy","status":"present","laterality":"unspecified","mention":"liver"},{"canonical":"kidney","type":"anatomy","status":"present","laterality":"unspecified","mention":"kidney"}]}'
  got=extract.parse(raw,"CT of the liver")
  self.assertEqual([x["canonical"] for x in got],["liver"])
 def test_concept_parser_canonical_evidence_fallback(self):
  raw='{"concepts":[{"canonical":"pulmonary artery","type":"anatomy","status":"present","laterality":"unspecified","mention":"lower central vessel"}]}'
  got=extract.parse(raw,"MRI demonstrates the pulmonary artery.")
  self.assertEqual(got[0]["mention"],"pulmonary artery")
 def test_concept_parser_missing_mention_uses_exact_canonical_evidence(self):
  raw='{"concepts":[{"canonical":"mass effect","type":"finding","status":"present","laterality":"unspecified"},{"canonical":"invented finding","type":"finding","status":"present","laterality":"unspecified"}]}'
  got=extract.parse(raw,"There is a mass effect in the left lung.")
  self.assertEqual([(x["canonical"],x["mention"]) for x in got],[("mass effect","mass effect")])
 def test_concept_target_status(self):
  got=extract.target([{"canonical":"effusion","type":"finding","status":"negated","laterality":"left","mention":"left effusion"}])
  self.assertEqual(got,"negated: left effusion")
 def test_metric_contracts_are_complete(self):
  required={"target_parser","prediction_parser","invalid_prediction_rule","primary_metric","secondary_metrics","score_range","empty_target_behavior"}
  contracts=build.contracts()
  self.assertEqual(len(contracts),6)
  self.assertTrue(all(required <= set(contract) for contract in contracts.values()))
 def test_iou(self):
  self.assertEqual(evaluate.iou((0,0,10,10),(0,0,10,10)),1)
  self.assertEqual(evaluate.ground("<box>0,0,10,10</box>","<box>0,0,10,10</box>"),1)
 def test_perceptual_bk_tree_cross_split(self):
  percept={"0000000000000000":[("task_01_vqa","train","a","/a")],"0000000000000001":[("task_02_diagnosis_classification","test","b","/b")]}
  count,examples=validator.perceptual_cross_split(percept,max_distance=4)
  self.assertEqual(count,1)
  self.assertEqual(examples[0]["distance"],1)
 def test_perceptual_multi_index_exact_radius_boundary(self):
  near=(1<<0)|(1<<13)|(1<<26)|(1<<39)
  far=near|(1<<52)
  percept={"0000000000000000":[("task_01_vqa","train","a","/a")],f"{near:016x}":[("task_02_diagnosis_classification","test","b","/b")],f"{far:016x}":[("task_02_diagnosis_classification","test","c","/c")]}
  count,examples=validator.perceptual_cross_split(percept,max_distance=4)
  self.assertEqual(count,1)
  self.assertEqual(examples[0]["distance"],4)
 def test_exact_overlap_repair_preserves_test_policy(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); task=root/"task_01_vqa"; task.mkdir()
   data=[{"id":"keep"},{"id":"drop"}]
   (task/"train.jsonl").write_text("\n".join(json.dumps(x) for x in data)+"\n")
   cross=[{"refs":[("task_01_vqa","train","drop","/a"),("task_01_vqa","test","official","/b")]}]
   report=validator.remove_train_exact_overlap(root,[task],cross)
   remaining=[json.loads(x)["id"] for x in (task/"train.jsonl").read_text().splitlines()]
   self.assertEqual(remaining,["keep"])
   self.assertEqual(report["total_removed"],1)
   self.assertEqual(report["test_records_removed"],0)



if __name__=="__main__": unittest.main()
