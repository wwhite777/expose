import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_day5_raw_reference.py"
SPEC = importlib.util.spec_from_file_location("day5_raw_reference", SCRIPT)
day5 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(day5)


@pytest.fixture(scope="module")
def raw_struct():
    # Exactly 100 isolated trials. smt follows the published file's Python-t
    # convention, one sample after the MATLAB-t-1 support used for epochs.
    events = 2 + np.arange(100, dtype=np.int64) * 4001
    samples = int(events[-1] + 4001)
    x = np.arange(samples, dtype=np.float64)[:, None] * 1e-3 + np.arange(8, dtype=np.float64)[None, :]
    smt = np.stack([x[event:event + 4000] for event in events], axis=1)
    labels = np.tile(np.array([1, 2], dtype=np.uint8), 50)
    return {
        "x": x, "t": events, "fs": 1000, "y_dec": labels,
        "chan": np.array(day5.CHANNELS, dtype=object),
        "class": np.array([["1", "right"], ["2", "left"]], dtype=object),
        "y_class": np.array(["right" if value == 1 else "left" for value in labels], dtype=object),
        "y_logic": np.vstack([labels == 1, labels == 2]).astype(np.uint8),
        "smt": smt,
    }


def test_source_is_syntactically_isolated():
    tree = ast.parse(SCRIPT.read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(name.startswith("expose") for name in imported)
    assert "EEG_MI_test" not in SCRIPT.read_text()
    assert "NotImplementedError" not in SCRIPT.read_text()


def test_valid_raw_parser_reconstructs_exact_recipe(raw_struct):
    result = day5.reconstruct_raw_struct(raw_struct, {}, build_epochs=True)
    assert result["X"].shape == (100, 8, 750)
    assert result["X"].dtype == np.dtype("float64")
    assert np.bincount(result["y"]).tolist() == [50, 50]
    assert result["smt_matches"] == {"matlab_t_minus_1": 0, "python_t": 100}
    assert len(result["raw_support_sha256"]) == len(result["epoch_sha256"]) == 100


def test_reconstructed_split_and_history_use_raw_labels(raw_struct):
    first = day5.reconstruct_raw_struct(raw_struct, {}, build_epochs=False)
    first = {**first, "epoch_sha256": ["0" * 64] * 100}
    rebuilt = {(5, 1): first, (2, 1): first, (2, 2): first}
    cfg = {"source_subjects": [5], "development_subjects": [2]}
    split = day5.reconstructed_split(cfg, rebuilt)
    selected = day5.history_ids(split, 2, 20260915)
    assert len(split) == 300
    assert len(selected) == 60
    assert [sum(split[tid]["label"] == label for tid in selected) for label in (0, 1)] == [30, 30]


def test_class_code_name_swap_is_rejected(raw_struct):
    changed = dict(raw_struct)
    changed["class"] = np.array([["1", "left"], ["2", "right"]], dtype=object)
    with pytest.raises(ValueError, match="class code/name mapping"):
        day5.reconstruct_raw_struct(changed, {}, build_epochs=False)


@pytest.mark.parametrize("field", ["y_class", "y_logic"])
def test_trial_level_class_crosschecks_are_enforced(raw_struct, field):
    changed = dict(raw_struct)
    changed[field] = raw_struct[field].copy()
    if field == "y_class":
        changed[field][0] = "left"
    else:
        changed[field][:, 0] = changed[field][::-1, 0]
    with pytest.raises(ValueError, match=field):
        day5.reconstruct_raw_struct(changed, {}, build_epochs=False)


def test_fractional_event_is_rejected_before_index_coercion(raw_struct):
    changed = dict(raw_struct)
    changed["t"] = raw_struct["t"].astype(float)
    changed["t"][3] += 0.5
    with pytest.raises(ValueError, match="finite nonnegative integers"):
        day5.reconstruct_raw_struct(changed, {}, build_epochs=False)


def test_actual_one_sample_smt_discrepancy_is_enforced(raw_struct):
    changed = dict(raw_struct)
    changed["smt"] = np.stack([raw_struct["x"][event - 1:event - 1 + 4000]
                                for event in raw_struct["t"]], axis=1)
    with pytest.raises(ValueError, match="one-sample"):
        day5.reconstruct_raw_struct(changed, {}, build_epochs=False)


def test_in_bounds_one_sample_event_shift_is_rejected(raw_struct):
    changed = dict(raw_struct)
    changed["t"] = raw_struct["t"].copy()
    changed["t"][3] += 1  # Remains increasing, nonoverlapping and in bounds.
    with pytest.raises(ValueError, match="one-sample"):
        day5.reconstruct_raw_struct(changed, {}, build_epochs=False)


def test_failed_producer_rejected_before_other_artifacts_are_opened(tmp_path):
    run = tmp_path / "producer"
    run.mkdir()
    (run / "supervisor_receipt.json").write_text(json.dumps({"status": "failed", "exit_code": 1}))
    # No receipt, completion marker, manifest, raw file, or cache exists. The
    # production guard must reject from the supervisor receipt itself.
    with pytest.raises(ValueError, match="supervisor did not complete"):
        day5.validate_stage(tmp_path, run, set(), "scripts/producer.py")


def test_confirmation_data_row_rejected_by_production_manifest_guard(tmp_path):
    cfg = {"source_subjects": list(range(5, 23)), "development_subjects": list(range(30, 42))}
    roles = {subject: "source" for subject in cfg["source_subjects"]}
    roles.update({subject: "development" for subject in cfg["development_subjects"]})
    roles[1] = "confirmation"
    rows = [{"subject_id": "1", "session": "1", "role": "confirmation"}] + [{} for _ in range(41)]
    with pytest.raises(ValueError, match="confirmation"):
        day5.validate_prepared_manifests(tmp_path, cfg, roles, rows, [{} for _ in range(4200)], {"records": []})


def prediction(operation, draw, p0="0.6", p1="0.4", *, tid="trial-1", truth="0", predicted="0"):
    return {"operation_id": operation, "arm": "practical", "model": "ts_lr", "draw": draw,
            "dose_per_class": "0", "prior_weight": "", "subject_id": "2", "trial_id": tid,
            "y_true": truth, "y_pred": predicted, "p_class0_right": p0, "p_class1_left": p1}


def test_shared_n0_duplicate_collapse_agrees():
    operation = "practical__ts_lr__rshared_n0"
    collapsed = day5.collapse_original_predictions([prediction(operation, "0"), prediction(operation, "1")],
                                                   operation, {("0", ""), ("1", "")})
    assert list(collapsed) == ["trial-1"]


def test_shared_n0_duplicate_disagreement_is_rejected():
    operation = "practical__ts_lr__rshared_n0"
    rows = [prediction(operation, "0"), prediction(operation, "1", "0.61", "0.39")]
    with pytest.raises(ValueError, match="duplicates disagree"):
        day5.collapse_original_predictions(rows, operation, {("0", ""), ("1", "")})


def test_prediction_comparison_success_exercises_both_class_recalls():
    operation = "practical__ts_lr__r0_n30_s002"
    reference = [prediction(operation, 0),
                 prediction(operation, 0, "0.2", "0.8", tid="trial-2", truth="1", predicted="1")]
    original = [prediction(operation, "0"),
                prediction(operation, "0", "0.2", "0.8", tid="trial-2", truth="1", predicted="1")]
    result = day5.compare_predictions(reference, original, operation, {("0", "")})
    assert result["physical_rows"] == 2
    assert result["balanced_accuracy"] == 1.0
    assert result["max_abs_probability_difference"] == 0.0


def test_changed_saved_prediction_is_rejected():
    operation = "practical__ts_lr__r0_n30_s002"
    reference = [prediction(operation, 0)]
    original = [prediction(operation, "0", "0.7", "0.3")]
    with pytest.raises(ValueError, match="probability tolerance"):
        day5.compare_predictions(reference, original, operation, {("0", "")})


def test_completion_marker_binds_exact_owned_outputs(tmp_path):
    out = tmp_path / "raw_reference_r001"
    out.mkdir()
    (out / "receipt.json").write_text(json.dumps({"run_id": out.name, "status": "completed", "exit_code": 0}))
    for name in ("INPUT_MANIFEST.json", "raw_checks.csv", "array_checks.csv", "predictions.csv", "membership.csv",
                 "timing.csv", "PREDICTION_LOCK.json", "comparison.json", "checks.json"):
        (out / name).write_text("{}\n" if name.endswith(".json") else "header\n")
    marker = day5.write_completion(out)
    assert marker["receipt_sha256"] == day5.sha256(out / "receipt.json")
    assert set(marker["outputs"]) == {"receipt.json", "INPUT_MANIFEST.json", "raw_checks.csv", "array_checks.csv",
                                      "predictions.csv", "membership.csv", "timing.csv", "PREDICTION_LOCK.json",
                                      "comparison.json", "checks.json"}
