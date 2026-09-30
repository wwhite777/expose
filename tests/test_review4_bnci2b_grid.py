import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_review4_bnci2b_grid", ROOT / "scripts/run_review4_bnci2b_grid.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_fixture(tmp_path, runner, count_override=None):
    runner.DATA_ROOT = tmp_path
    count_override = count_override or {}
    records = []
    identity = np.eye(2, dtype=np.float64)
    for subject, expected_pair in runner.EXPECTED_SESSION_TRIAL_COUNTS.items():
        for session, expected_count in enumerate(expected_pair, start=1):
            count = count_override.get((subject, session), expected_count)
            labels = np.tile(np.array([0, 1], dtype=np.int64), count // 2)
            cov = np.stack(
                [identity * (1 + subject / 100 + session / 1000 + trial / 10000)
                 for trial in range(count)]
            )
            relative = f"arrays/s{subject:02d}_session{session}.npz"
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, cov=cov, ea_cov=cov.copy(), empirical_cov=cov.copy(), y=labels)
            records.append({
                "subject": subject,
                "session": session,
                "role": "external",
                "npz_path": relative,
                "sha256": runner.sha256(path),
                "trial_ids": [
                    f"bnci2014-004:s{subject:02d}:session{session:02d}:trial{trial:03d}"
                    for trial in range(1, count + 1)
                ],
            })
    index = {"dataset": "bnci2014b", "montage": "3ch", "records": records}
    index_path = tmp_path / "dataset_index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    config = {
        "schema": runner.SCHEMA,
        "seed": 2026092403,
        "draws": 10,
        "C_grid": [0.1, 1.0, 10.0],
        "C_tie_tolerance": 1e-12,
        "methods": list(runner.METHODS),
        "mdwm_lambda": 0.5,
        "index_sha256": runner.sha256(index_path),
        "limits": {"cpu_seconds": 1800, "wall_seconds": 14400, "memory_gib": 16},
        "studies": [{
            "study_id": "bnci2b_main",
            "dataset": "bnci2014b",
            "montage": "3ch",
            "mode": "loso",
            "source_sizes": [100, "all"],
            "history_totals": [0, 60],
            "donor_counts": ["all"],
            "reference_trials_per_person": "all_available",
        }],
    }
    return index_path, config


def test_exact_observed_matrix_includes_all_three_special_cells(tmp_path):
    runner = load_runner()
    index_path, config = write_fixture(tmp_path, runner)
    runner.validate_config(config, runner.sha256(index_path))
    store = runner.DatasetStore(index_path, runner.sha256(index_path))
    assert runner.cell_records(store, config["studies"][0]) == (list(range(1, 10)),) * 2
    assert len(store.by_cell[(4, 2)]["trial_ids"]) == 140
    assert len(store.by_cell[(5, 2)]["trial_ids"]) == 140
    assert len(store.by_cell[(8, 1)]["trial_ids"]) == 160


def test_unknown_subject_session_count_is_rejected(tmp_path):
    runner = load_runner()
    index_path, config = write_fixture(tmp_path, runner, {(4, 2): 142})
    store = runner.DatasetStore(index_path, runner.sha256(index_path))
    with pytest.raises(ValueError, match="observed BNCI2B matrix"):
        runner.cell_records(store, config["studies"][0])


def test_full_source_generation_uses_variable_loso_pool_and_own_reference(tmp_path):
    runner = load_runner()
    index_path, config = write_fixture(tmp_path, runner)
    store = runner.DatasetStore(index_path, runner.sha256(index_path))
    operations, sets, tuning = runner.make_plan(config, store)
    assert len(operations) == 1800
    assert len(tuning) == 360

    full = [
        operation for operation in operations
        if operation["draw"] == 0
        and operation["source_size"] == "all"
        and operation["h_total"] == 0
        and operation["method"] == "plain_ts"
    ]
    assert len(full) == 9
    by_target = {operation["target"]: operation for operation in full}
    assert by_target[8]["source_n"] == 960
    assert all(by_target[target]["source_n"] == 1000 for target in range(1, 10) if target != 8)
    assert by_target[8]["reference_trials_per_person"] == 160
    assert all(
        by_target[target]["reference_trials_per_person"] == 120
        for target in range(1, 10) if target != 8
    )
    assert len(sets[by_target[4]["memberships"]["evaluation"]]) == 140
    assert len(sets[by_target[5]["memberships"]["evaluation"]]) == 140
    assert len(sets[by_target[8]["memberships"]["evaluation"]]) == 120

