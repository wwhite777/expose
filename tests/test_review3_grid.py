import importlib.util
import csv
import io
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_review3_grid", ROOT / "scripts/run_review3_grid.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_fixture(tmp_path, runner, empirical=False, count=100,
                  source_sizes=(100, 200), history_totals=(0, 60, 100), donor_counts=(3,)):
    runner.DATA_ROOT = tmp_path
    records = []
    labels = np.tile(np.array([0, 1], dtype=np.int64), count // 2)
    identity = np.eye(2, dtype=np.float64)
    for role, subjects, sessions in (
            ("source", range(1, 19), (1,)),
            ("development", range(101, 113), (1, 2))):
        for subject in subjects:
            for session in sessions:
                trial_ids = [f"p{subject}-s{session}-t{trial}" for trial in range(count)]
                scale = 1 + subject / 1000 + session / 100
                cov = np.stack([identity * (scale + trial / 1000) for trial in range(count)])
                relative = f"arrays/p{subject}_s{session}.npz"
                path = tmp_path / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = {"cov": cov, "ea_cov": cov.copy(), "y": labels}
                if empirical and subject == 1:
                    payload["empirical_cov"] = cov.copy()
                np.savez(path, **payload)
                records.append({"subject": subject, "session": session, "role": role,
                                "npz_path": relative, "sha256": runner.sha256(path),
                                "trial_ids": trial_ids})
    index = {"dataset": "openbmi", "montage": "8ch", "records": records}
    index_path = tmp_path / "GRID_INDEX.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    config = {
        "schema": runner.SCHEMA, "seed": 20260923, "draws": 1,
        "C_grid": [0.1, 1.0, 10.0], "C_tie_tolerance": 1e-12,
        "methods": list(runner.METHODS), "mdwm_lambda": 0.5,
        "index_sha256": runner.sha256(index_path),
        "limits": {"cpu_seconds": 120, "wall_seconds": 180, "memory_gib": 4},
        "studies": [{"study_id": "openbmi_main", "dataset": "openbmi",
                     "montage": "8ch", "mode": "frozen_source",
                     "source_sizes": list(source_sizes), "history_totals": list(history_totals),
                     "donor_counts": list(donor_counts), "reference_trials_per_person": count}],
    }
    config_path = tmp_path / "CONFIG.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return index_path, config_path


def test_plan_has_nested_paired_memberships_and_source_only_donor_cv(tmp_path):
    runner = load_runner()
    index_path, config_path = write_fixture(tmp_path, runner, empirical=True)
    config = runner.validate_config(runner.load_json(config_path), runner.sha256(index_path))
    store = runner.DatasetStore(index_path, runner.sha256(index_path))
    operations, sets, tuning = runner.make_plan(config, store)

    def operation(target, source_size, h_total, method="plain_ts"):
        found = [value for value in operations
                 if value["target"] == target and value["source_size"] == source_size
                 and value["h_total"] == h_total and value["method"] == method]
        assert len(found) == 1
        return found[0]

    small, large = operation(101, 100, 60), operation(101, 200, 100)
    assert set(sets[small["memberships"]["source"]]) <= set(sets[large["memberships"]["source"]])
    assert set(sets[small["memberships"]["personal"]]) <= set(sets[large["memberships"]["personal"]])
    paired_target = operation(102, 200, 100)
    assert large["memberships"]["source"] == paired_target["memberships"]["source"]
    assert large["donor_subjects"] == paired_target["donor_subjects"]
    assert not set(sets[large["memberships"]["evaluation"]]) & (
        set(sets[large["memberships"]["source"]]) |
        set(sets[large["memberships"]["personal"]]))

    context = next(value for value in tuning
                   if value["tuning_id"] == large["tuning_id"])
    assert len(context["folds"]) == 3
    for fold in context["folds"]:
        train_subjects = set(store.subjects(sets[fold["train"]]).tolist())
        assert fold["heldout_subject"] not in train_subjects
        assert len(sets[fold["evaluation"]]) == 100
        assert set(store.subjects(sets[fold["evaluation"]]).tolist()) == {fold["heldout_subject"]}


def test_plan_hash_guard_rejects_tampering_and_c_ties_choose_smallest(tmp_path):
    runner = load_runner()
    index_path, config_path = write_fixture(tmp_path, runner)
    plan_dir = tmp_path / "plan"
    receipt = runner.plan(config_path, index_path, plan_dir)
    assert receipt["status"] == "planned_no_fits"
    runner.validate_plan(config_path, index_path, plan_dir)
    assert runner.choose_c({.1: .7, 1.: .7 + 5e-13, 10.: .6}, (.1, 1., 10.), 1e-12) == .1
    with (plan_dir / "operations.json.gz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="changed"):
        runner.validate_plan(config_path, index_path, plan_dir)


def test_store_rejects_non_spd_saved_covariance(tmp_path):
    runner = load_runner()
    index_path, _ = write_fixture(tmp_path, runner)
    index = runner.load_json(index_path)
    record = index["records"][0]
    path = tmp_path / record["npz_path"]
    with np.load(path, allow_pickle=False) as archive:
        cov, ea_cov, labels = archive["cov"], archive["ea_cov"], archive["y"]
    cov[0, 0, 0] = -1
    np.savez(path, cov=cov, ea_cov=ea_cov, y=labels)
    record["sha256"] = runner.sha256(path)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="covariance"):
        runner.DatasetStore(index_path)


def test_tiny_subprocess_runs_all_methods_cv_streaming_and_historical_scores(tmp_path):
    runner = load_runner()
    index_path, config_path = write_fixture(
        tmp_path, runner, count=100, source_sizes=(100,),
        history_totals=(0, 60), donor_counts=("all",))
    plan_dir, run_dir = tmp_path / "plan", tmp_path / "run"
    donor_plan_dir, donor_run_dir = tmp_path / "donor_plan", tmp_path / "donor_run"
    code = """
import importlib.util, pathlib, sys
spec=importlib.util.spec_from_file_location('r', sys.argv[1]); r=importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
r.DATA_ROOT=pathlib.Path(sys.argv[2]); r.plan(sys.argv[3],sys.argv[4],sys.argv[5]); r.run(sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6])
spec=importlib.util.spec_from_file_location('d', sys.argv[7]); d=importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
d.grid.DATA_ROOT=pathlib.Path(sys.argv[2]); d.plan(sys.argv[3],sys.argv[4],sys.argv[5],'openbmi_main',sys.argv[8]); d.run(sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[8],pathlib.Path(sys.argv[6])/'selected_c.csv',sys.argv[9])
"""
    environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    completed = subprocess.run(
        [sys.executable, "-c", code, str(ROOT / "scripts/run_review3_grid.py"),
         str(tmp_path), str(config_path), str(index_path), str(plan_dir), str(run_dir),
         str(ROOT / "scripts/run_review3_donors.py"), str(donor_plan_dir), str(donor_run_dir)],
        cwd=ROOT, env=environment, check=False, text=True, capture_output=True, timeout=90)
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads((run_dir / "receipt.json").read_text())
    assert receipt["status"] == "completed" and receipt["counts"]["operations"] == 120
    assert receipt["counts"]["predictions"] == 12000
    assert receipt["counts"]["historical_decodability_rows"] == 24
    assert not list(run_dir.glob("*.partial"))
    with gzip.open(run_dir / "predictions.csv.gz", "rt", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["method"] for row in rows} == set(runner.METHODS)
    assert all(abs(float(row["p0"]) + float(row["p1"]) - 1) <= 1e-10 for row in rows)
    with (run_dir / "timing.csv").open(newline="", encoding="utf-8") as handle:
        timing = list(csv.DictReader(handle))
    assert any(row["reused_h0_fit"] == "True" for row in timing)
    assert all(row["reused_h0_fit"] == "False" for row in timing
               if "__recenter_mdm__" in row["operation_id"])
    _, operations, sets, _ = runner.validate_plan(config_path, index_path, plan_dir)
    for operation in operations:
        assert not set(sets[operation["memberships"]["source"]]) & set(
            sets[operation["memberships"]["evaluation"]])
    donor_receipt = json.loads((donor_run_dir / "receipt.json").read_text())
    assert donor_receipt["status"] == "completed"
    assert donor_receipt["counts"] == {"operations": 144, "predictions": 14400}
    with (donor_plan_dir / "donor_identity.csv").open(newline="", encoding="utf-8") as handle:
        identities = list(csv.DictReader(handle))
    assert {row["condition"] for row in identities} == {
        "base", "own", "pooled", "single_0", "single_1", "single_2"}
    assert all(row["added_trial_overlap_with_base"] == "0" for row in identities)


def test_prediction_gzip_hash_is_final_only_after_all_stream_layers_close(tmp_path):
    """The grid must hash its gzip only after TextIO, gzip, and raw layers close."""
    runner = load_runner()
    path = tmp_path / "predictions.csv.gz"
    raw = path.open("wb")
    compressed = gzip.GzipFile(fileobj=raw, mode="wb", mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    writer = csv.DictWriter(text, fieldnames=["operation_id", "p0", "p1"])
    writer.writeheader(); writer.writerow({"operation_id": "fixture", "p0": 1.0, "p1": 0.0})
    runner.close_prediction_stream(raw, text)
    recorded_hash = runner.sha256(path)
    assert raw.closed and text.closed
    del raw, text, compressed
    assert recorded_hash == runner.sha256(path)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"operation_id": "fixture", "p0": "1.0", "p1": "0.0"}]
