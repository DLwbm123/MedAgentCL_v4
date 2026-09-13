#!/usr/bin/env python3
import argparse,collections,csv,hashlib,json,subprocess
from pathlib import Path
ROOT=Path("/root/MedAgentCL_v4");DATA=Path("/remote-home/wangbomin/MedicalSkill-CL-v1");ART=ROOT/"artifacts/medicalskill_cl_v1_closure"
def rows(p):
 with open(p,encoding="utf-8-sig") as f:
  for line in f:
   if line.strip():yield json.loads(line)
def read(p):return json.loads(Path(p).read_text()) if Path(p).is_file() else {}
def writej(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=Path(str(p)+".tmp");t.write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n");t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--data-root",default=str(DATA));ap.add_argument("--artifact-root",default=str(ART));a=ap.parse_args();data,art=Path(a.data_root),Path(a.artifact_root)
 task_dirs=sorted(data.glob("task_*"));stats={};sources={};tokens={}
 for d in task_dirs:
  stats[d.name]={};source_counts=collections.Counter();token_set=set()
  for split in ("train","test"):
   values=list(rows(d/f"{split}.jsonl"));stats[d.name][split]=len(values)
   source_counts.update(str((x.get("metadata") or {}).get("source_dataset") or "") for x in values)
   if split=="train":
    for x in values:token_set.update((x.get("metadata") or {}).get("canonical_lineage_tokens") or [])
  sources[d.name]=dict(source_counts);tokens[d.name]=token_set
  current=read(d/"statistics.json");current.update(stats[d.name]);writej(d/"statistics.json",current)
  for split in ("train","test"):
   sm=read(d/f"{split}_manifest.json");sm["records"]=stats[d.name][split];writej(d/f"{split}_manifest.json",sm)
 manifest=read(data/"manifests/medicalskill_cl_v1_manifest.json")
 for item in manifest.get("tasks",[]):item.update(stats.get(item["task_name"],{}))
 writej(data/"manifests/medicalskill_cl_v1_manifest.json",manifest);writej(art/"medicalskill_cl_v1_manifest.json",manifest)
 writej(art/"source_inventory.json",{"status":"PASS","tasks":sources,"v0_read_only":"/remote-home/wangbomin/MedicalSkill-CL"})
 with (art/"cross_skill_overlap_matrix.csv").open("w",newline="") as f:
  w=csv.writer(f);names=list(tokens);w.writerow(["task",*names])
  for left in names:w.writerow([left,*[len(tokens[left]&tokens[right]) for right in names]])
 preserved={"task_01_vqa":17796,"task_02_diagnosis_classification":6804,"task_03_concept_recognition":7038,"task_04_visual_grounding":9623,"task_05_reasoning_vqa":720}
 test_report={"status":"PASS" if all(stats[k]["test"]==v for k,v in preserved.items()) else "FAIL","policy":"all v0 official test memberships preserved; concept test is the union of the two disjoint MedTrinity v0 test groups","expected":preserved,"actual":{k:v["test"] for k,v in stats.items()},"test_records_removed":0};writej(art/"test_preservation_report.json",test_report)
 strict=read(art/"strict_validation.json");swift=read(art/"swift_schema_smoke.json");phash=read(art/"phash_candidate_classification.json");smoke=read(art/"model_smoke_summary.json");pilot=read(art/"concept_semantic_audit_500.json");q0=read(art/"concept_full_part_00_qa.json");q1=read(art/"concept_full_part_01_qa.json");balance=read(art/"medsg_final_balance.json") or read(art/"medsg_balanced_sampling.json")
 lineage_train_test=read(art/"train_test_lineage_removal.json");lineage_ownership=read(art/"cross_task_train_ownership.json")
 lineage_audit={"status":"PASS" if strict.get("train_test_lineage_group_count")==0 and strict.get("cross_task_train_group_count")==0 and test_report.get("status")=="PASS" else "FAIL","parser_spec_file":str(art/"lineage_parser_spec.json"),"train_test":{"test_records_removed":lineage_train_test.get("test_records_removed"),"train_records_removed":lineage_train_test.get("total_train_removed"),"remaining_groups":strict.get("train_test_lineage_group_count")},"cross_task_train":{"components_before_ownership":lineage_ownership.get("cross_task_components"),"train_records_removed":lineage_ownership.get("total_removed"),"remaining_groups":strict.get("cross_task_train_group_count")},"test_preservation":test_report}
 writej(art/"canonical_lineage_audit.json",lineage_audit)
 before=read(art/"v0_read_only_fingerprint_before.json")
 current={str(Path(p)):{"bytes":Path(p).stat().st_size,"mtime_ns":Path(p).stat().st_mtime_ns} for p in before.get("files",{})}
 v0_unchanged=current==before.get("files",{})
 writej(art/"v0_read_only_verification.json",{"status":"PASS" if v0_unchanged else "FAIL","unchanged":v0_unchanged,"files_checked":len(current)})
 checks={
  "v0_read_only_unchanged":v0_unchanged,
  "canonical_lineage_audit_pass":lineage_audit.get("status")=="PASS",
  "five_skills_only":len(task_dirs)==5,
  "five_skill_contract_only":strict.get("removed_sixth_skill_absent_from_v1_contracts") is True,
  "tests_preserved":test_report["status"]=="PASS",
  "train_test_lineage_zero":strict.get("train_test_lineage_group_count")==0,
  "cross_task_train_lineage_zero":strict.get("cross_task_train_group_count")==0,
  "confirmed_phash_groups_zero":phash.get("confirmed_before_repair")==0,
  "phash_candidates_classified":phash.get("candidate_group_count",0)==sum(phash.get("classification_counts",{}).values()),
  "medsg_balanced_48k_52k":balance.get("status")=="PASS",
  "concept_pilot_500_pass":pilot.get("status")=="PASS" and pilot.get("reviewed_sample_count")==500,
  "concept_full_pass":q0.get("status")=="PASS" and q1.get("status")=="PASS",
  "swift_4_4_1_strict_pass":swift.get("status")=="PASS" and swift.get("swift_version")=="4.4.1",
  "real_qwen3vl_five_skill_smoke_pass":smoke.get("status")=="PASS" and len(smoke.get("tasks",{}))==5,
  "no_validation_split":read(art/"effective_config.json").get("validation_split") is None,
  "one_epoch_formal_commands":read(art/"effective_config.json").get("epochs")==1,
 }
 acceptance={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks};writej(art/"acceptance_matrix.json",acceptance)
 model_manifest={"concept_extraction":{"model_id":"Qwen/Qwen3-8B","revision":"b968826d9c46dd6066d109eabc6255188de91218","snapshot":"/remote-home/wangbomin/huggingface_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218","temperature":0,"do_sample":False},"multimodal_validation":{"model_id":"Qwen/Qwen3-VL-8B-Instruct","revision":"0c351dd01ed87e9c1b53cbc748cba10e6187ff3b","snapshot":"/remote-home/wangbomin/huggingface_cache/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b","offline":True},"swift_version":"4.4.1"};writej(art/"model_revision_manifest.json",model_manifest)
 commands=["# Actual closure commands",f"{ROOT}/scripts/skill_incremental_v1/build_medicalskill_cl_v1.sh prepare",f"{ROOT}/scripts/skill_incremental_v1/build_medicalskill_cl_v1.sh pilot",f"{ROOT}/scripts/skill_incremental_v1/build_medicalskill_cl_v1.sh extract",f"{ROOT}/scripts/skill_incremental_v1/build_medicalskill_cl_v1.sh reparse",f"{ROOT}/scripts/skill_incremental_v1/build_medicalskill_cl_v1.sh build",f"{ROOT}/scripts/skill_incremental_v1/build_medicalskill_cl_v1.sh phash",f"{ROOT}/scripts/skill_incremental_v1/validate_medicalskill_cl_v1.sh",f"{ROOT}/scripts/skill_incremental_v1/run_model_smoke_v1.sh","","# Formal one-epoch commands (not executed)"]+[f"CUDA_VISIBLE_DEVICES=0 {ROOT}/scripts/skill_incremental_v1/train_one_epoch_task.sh {i}" for i in range(1,6)]
 (art/"commands.sh").write_text("\n".join(commands)+"\n");(art/"commands.sh").chmod(0o755)
 try:git=subprocess.run(["git","status","--short"],cwd=ROOT,text=True,capture_output=True,check=True).stdout
 except Exception as e:git=str(e)
 (art/"git_state.txt").write_text(git)
 summary=["# MedicalSkill-CL-v1 closure","",f"Status: **{acceptance['status']}**","",f"Dataset: {data}",f"Tasks: {len(task_dirs)}",f"Train/test counts: {stats}",f"MedSG train turns: {balance.get('total_output_turns')}",f"pHash unresolved candidates: {phash.get('unresolved_count')}",f"Strict Swift: {swift.get('status')}",f"Real Qwen3-VL smoke: {smoke.get('status')}","","No formal one-epoch training was executed."]
 (art/"closure_summary.md").write_text("\n".join(summary)+"\n")
 files=sorted({p for base in (data,art,ROOT/"scripts/skill_incremental_v1",ROOT/"tests/skill_incremental_v1") for p in base.rglob("*") if p.is_file() and p!=art/"file_hashes.json" and p.suffix!=".log"} | {ROOT/"docs/medicalskill_cl_v1.md"},key=str)
 hashes={str(p):{"sha256":sha(p),"bytes":p.stat().st_size} for p in files};writej(art/"file_hashes.json",{"status":"PASS","algorithm":"sha256","files":hashes})
 print(json.dumps({"status":acceptance["status"],"stats":stats,"checks":checks},indent=2));return 0 if acceptance["status"]=="PASS" else 1
if __name__=="__main__":raise SystemExit(main())
