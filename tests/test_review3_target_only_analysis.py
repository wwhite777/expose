import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_analyzer():
    spec = importlib.util.spec_from_file_location("target_analysis", ROOT / "scripts/analyze_review3_target_only.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, fields, values, zipped=False):
    opener = gzip.open if zipped else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(values)


def target_output(root, analyzer, label, people):
    root.mkdir(); bindings=[]; predictions=[]; scores=[]
    for target in range(1, people + 1):
        for draw in range(10):
            for h in analyzer.HISTORY:
                op=f"target_only__{label}_{target}_{draw}_{h}"; anchor=f"{label}_{target}_{draw}_{h}"
                personal, evaluation=f"p-{target}-{draw}-{h}",f"e-{target}"
                bindings.append({"operation_id":op,"anchor_operation_id":anchor,"personal_membership":personal,"evaluation_membership":evaluation,"h_total":h,"target":target,"draw":draw})
                for trial, truth in enumerate((0,1)):
                    predictions.append({"operation_id":op,"anchor_operation_id":anchor,"study_id":label,"draw":draw,"target":target,"source_size":"all","h_total":h,"C":1.0,"trial_id":f"{op}-{trial}","y_true":truth,"y_pred":truth,"p0":1-truth,"p1":truth,"personal_membership":personal,"evaluation_membership":evaluation})
                scores.append({"operation_id":op,"anchor_operation_id":anchor,"study_id":label,"draw":draw,"target":target,"source_size":"all","h_total":h,"C":1.0,"balanced_accuracy":1.0,"evaluation_n":2,"personal_membership":personal,"evaluation_membership":evaluation})
    write_csv(root/"predictions.csv.gz", analyzer.PREDICTION_FIELDS, predictions, True)
    write_csv(root/"operation_scores.csv", analyzer.SCORE_FIELDS, scores)
    binding={"schema":"review3-target-only-v1","source_covariances_used_for_fitting":False,"source_array_validation_note":"DatasetStore validates all indexed arrays; source covariances are never passed to target-only fitting.","operations":bindings}
    (root/"PLAN_BINDING.json").write_text(json.dumps(binding))
    receipt={"schema":"review3-target-only-v1","status":"completed","outputs":{name:sha(root/name) for name in ("predictions.csv.gz","operation_scores.csv","PLAN_BINDING.json")}}
    (root/"receipt.json").write_text(json.dumps(receipt))
    (root/"COMPLETED.json").write_text(json.dumps({"status":"completed","receipt_sha256":sha(root/"receipt.json")}))


def test_full_target_only_inventory_reconstructs_and_pairs(tmp_path):
    analyzer=load_analyzer(); core=tmp_path/"core"; core.mkdir(); core_rows=[]
    for label, people, full in (("openbmi8",12,"1800"),("bnci",9,"all")):
        for target in range(1,people+1):
            for draw in range(10):
                for h in analyzer.HISTORY:
                    for source in ("100",full):
                        core_rows.append({"dataset_variant":label,"operation_id":f"core-{label}-{source}-{target}-{draw}-{h}","study_id":analyzer.MAIN_CORE_STUDY[label],"method":"plain_ts","target":target,"draw":draw,"source_size":source,"source_n":1,"donor_count":"all","h_total":h,"evaluation_n":2,"balanced_accuracy":.5})
    # Same comparison coordinate retained for the composition/diversity study:
    # it must not masquerade as a duplicate main-study estimate.
    diversity=dict(core_rows[0]); diversity.update({"operation_id":"diversity-same-coordinate","study_id":"openbmi_diversity","balanced_accuracy":.75})
    core_rows.append(diversity)
    write_csv(core/"reconstructed_scores.csv", analyzer.CORE_FIELDS, core_rows); (core/"summary.json").write_text("{}")
    openbmi,bnci=tmp_path/"openbmi",tmp_path/"bnci"; target_output(openbmi,analyzer,"openbmi8",12); target_output(bnci,analyzer,"bnci",9)
    inputs={"schema":analyzer.SCHEMA,"core_analysis":{"directory":str(core),"summary_sha256":sha(core/"summary.json"),"reconstructed_scores_sha256":sha(core/"reconstructed_scores.csv")},"target_only":[{"label":"openbmi8","directory":str(openbmi)},{"label":"bnci","directory":str(bnci)}]}
    path=tmp_path/"inputs.json"; path.write_text(json.dumps(inputs)); out=tmp_path/"out"
    import sys
    previous=sys.argv; sys.argv=["x","--inputs",str(path),"--out-dir",str(out)]
    try: analyzer.main()
    finally: sys.argv=previous
    contrasts=list(csv.DictReader((out/"paired_contrasts.csv").open()))
    assert len(contrasts)==(12+9)*6*2 and {r["difference_pp"] for r in contrasts}=={"50.0"}
    assert json.loads((out/"COMPLETED.json").read_text())["status"]=="completed"


def test_core_rows_rejects_duplicate_within_main_study(tmp_path):
    analyzer=load_analyzer(); core=tmp_path/"core"; core.mkdir()
    row={"dataset_variant":"openbmi8","operation_id":"main-a","study_id":"openbmi_main","method":"plain_ts","target":1,"draw":0,"source_size":"100","source_n":100,"donor_count":"all","h_total":4,"evaluation_n":100,"balanced_accuracy":.5}
    duplicate=dict(row); duplicate["operation_id"]="main-b"
    write_csv(core/"reconstructed_scores.csv", analyzer.CORE_FIELDS, [row,duplicate])
    (core/"summary.json").write_text("{}")
    binding={"directory":str(core),"summary_sha256":sha(core/"summary.json"),"reconstructed_scores_sha256":sha(core/"reconstructed_scores.csv")}
    import pytest
    with pytest.raises(ValueError, match="duplicate core coordinate"):
        analyzer.core_rows(binding)
