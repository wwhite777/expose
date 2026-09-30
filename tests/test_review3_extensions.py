import csv
import gzip
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def load_analyzer(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "analyze_review3_extensions", REPO / "scripts/analyze_review3_extensions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    return module


def write_csv(path, rows, compressed=False):
    opener = gzip.open if compressed else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def predicted_rows(operation_id, score, variant=None, prefix="t"):
    truth = [0, 0, 1, 1]
    correct = {0.0: [1, 1, 0, 0], 0.25: [0, 1, 0, 0],
               0.5: [0, 0, 0, 0], 0.75: [0, 0, 1, 0], 1.0: truth}[score]
    rows = []
    for index, (actual, predicted) in enumerate(zip(truth, correct)):
        row = {"operation_id": operation_id, "trial_id": f"{prefix}{index}",
               "y_true": actual, "y_pred": predicted,
               "p0": 1.0 if predicted == 0 else 0.0,
               "p1": 1.0 if predicted == 1 else 0.0}
        if variant is not None:
            row["variant"] = variant
        rows.append(row)
    return rows


def complete(directory, module, outputs, neural=False):
    receipt = {"status": "completed", "outputs": {name: module.sha256(directory / name)
                                                    for name in outputs}}
    (directory / "receipt.json").write_text(json.dumps(receipt))
    completed = {"status": "completed", "receipt_sha256": module.sha256(directory / "receipt.json")}
    if neural:
        completed.update(predictions_sha256=module.sha256(directory / "predictions.csv.gz"),
                         scores_sha256=module.sha256(directory / "scores.csv"))
        receipt["outputs"] = {}
        (directory / "receipt.json").write_text(json.dumps(receipt))
        completed["receipt_sha256"] = module.sha256(directory / "receipt.json")
    (directory / "COMPLETED.json").write_text(json.dumps(completed))


def test_donor_averages_three_singles_within_draw_and_preserves_both_settings(tmp_path):
    module = load_analyzer(tmp_path)
    directory = tmp_path / "donor"; directory.mkdir()
    predictions, scores = [], []
    condition_score = {"base": .5, "own": 1., "pooled": .75,
                       "single_0": .5, "single_1": .75, "single_2": 1.}
    for setting in module.DONOR_SETTINGS:
        for target in (1, 2):
            for draw in range(10):
                for condition, score in condition_score.items():
                    operation = f"{setting}-{target}-{draw}-{condition}"
                    predictions.extend(predicted_rows(operation, score, prefix=operation))
                    scores.append({"operation_id": operation, "study_id": "study",
                                   "setting": setting, "condition": condition,
                                   "draw": draw, "target": target, "donor_subject": "",
                                   "C": 1, "balanced_accuracy": score, "evaluation_n": 4})
    write_csv(directory / "predictions.csv.gz", predictions, compressed=True)
    write_csv(directory / "operation_scores.csv", scores)
    (directory / "timing.csv").write_text("operation_id\n")
    complete(directory, module, ["predictions.csv.gz", "operation_scores.csv", "timing.csv"])
    participants, contrasts, reconstructed = [], [], []
    endpoint, verification = module.analyze_donor(
        {"label": "bnci", "directory": "donor", "required": True},
        participants, contrasts, reconstructed)
    selected = [row for row in contrasts if row["contrast"] == "own_minus_mean_single"]
    assert len(selected) == 4 and all(row["value_pp"] == pytest.approx(25) for row in selected)
    assert {row["method_or_setting"] for row in selected} == set(module.DONOR_SETTINGS)
    assert endpoint["values_pp"] == [25.0, 25.0]
    assert verification["operations"] == 240


def test_neural_pairs_exactly_classical_draws_zero_to_two(tmp_path):
    module = load_analyzer(tmp_path)
    directory = tmp_path / "neural"; directory.mkdir()
    predictions, scores, core = [], [], {}
    for draw in range(3):
        classical = f"plain-{draw}"; operation = "eegnet__" + classical
        for row in predicted_rows(operation, 1., prefix=operation):
            predictions.append({name: value for name, value in row.items() if name not in ("p0", "p1")}
                               | {"p_right": row["p0"], "p_left": row["p1"]})
        scores.append({"operation_id": operation, "method": "eegnet", "target": 7,
                       "draw": draw, "source_size": 100, "h_total": 60,
                       "balanced_accuracy": 1., "source_checkpoint": "x",
                       "fine_seed": 1, "fine_losses": "[]",
                       "source_membership": "s", "history_membership": "h"})
        core[("openbmi8", classical)] = {
            "dataset_variant": "openbmi8", "method": "plain_ts",
            "target": 7, "draw": draw, "source_size": "100",
            "h_total": 60, "balanced_accuracy": .5}
    write_csv(directory / "predictions.csv.gz", predictions, compressed=True)
    write_csv(directory / "scores.csv", scores)
    complete(directory, module, ["predictions.csv.gz", "scores.csv"], neural=True)
    participants, contrasts, reconstructed = [], [], []
    verification = module.analyze_neural(
        {"label": "openbmi8", "directory": "neural", "required": True}, core,
        participants, contrasts, reconstructed)
    assert participants[0]["draw_count"] == 3
    assert contrasts[0]["draw_count"] == 3 and contrasts[0]["value_pp"] == pytest.approx(50)
    assert verification["paired_draws"] == [0, 1, 2]


def test_alignment_rejects_different_trial_ids_for_same_scope_pair(tmp_path):
    module = load_analyzer(tmp_path)
    directory = tmp_path / "alignment"; directory.mkdir()
    predictions, scores, core = [], [], {}
    variants = [value for pair in module.ALIGNMENT_PAIRS for value in pair[:2]]
    for draw in range(10):
        operation = f"ea-{draw}"
        core[("bnci", operation)] = {
            "dataset_variant": "bnci", "method": "ea_ts", "target": 3,
            "draw": draw, "source_size": "all", "h_total": 60,
            "balanced_accuracy": .5}
        for variant in variants:
            prefix = operation
            if draw == 0 and variant == "R2_current_prefix20_tail":
                prefix = operation + "-different"
            predictions.extend(predicted_rows(operation, .5, variant=variant, prefix=prefix))
            scores.append({"operation_id": operation, "variant": variant, "draw": draw,
                           "target": 3, "source_size": "all", "h_total": 60, "C": 1,
                           "evaluation_n": 4, "balanced_accuracy": .5})
    write_csv(directory / "trial_predictions.csv.gz", predictions, compressed=True)
    write_csv(directory / "scores.csv", scores)
    (directory / "REFERENCE_AND_EVALUATION_IDS.json").write_text("[]")
    (directory / "EMPIRICAL_COVARIANCE_GUARDS.json").write_text("{}")
    complete(directory, module, ["trial_predictions.csv.gz", "scores.csv",
                                 "REFERENCE_AND_EVALUATION_IDS.json",
                                 "EMPIRICAL_COVARIANCE_GUARDS.json"])
    with pytest.raises(ValueError, match="different evaluation IDs"):
        module.analyze_alignment(
            {"label": "bnci", "directory": "alignment", "required": True}, core,
            [], [], [])


def test_alignment_label_gains_pair_people_and_reject_missing_h_pair(tmp_path, monkeypatch):
    module = load_analyzer(tmp_path)
    monkeypatch.setattr(module, "effect", lambda values: {"values_pp": list(values)})
    means, cells, trial_sets = {}, {}, {}
    for variant, _ in module.ALIGNMENT_LABEL_GAIN_REGIMES:
        for target, low, high in ((1, .50, .70), (2, .60, .50)):
            for history, value in ((0, low), (60, high)):
                key = ("bnci", variant, target, "all", history)
                means[key] = value
                cells[key] = [(draw, value, f"{variant}-{target}-{draw}-{history}") for draw in range(10)]
                for draw, _, operation in cells[key]:
                    trial_sets[(operation, variant)] = tuple(f"t{target}-trial{i}" for i in range(4))
    rows = []
    result = module.alignment_label_gains("bnci", means, cells, rows, trial_sets)
    assert result["current_label_gain_prefix20_tail__Sall__h60"]["values_pp"] == pytest.approx([20., -10.])
    assert sorted(row["value_pp"] for row in rows if row["contrast"] == "current_label_gain_prefix20_tail") == pytest.approx([-10., 20.])
    broken_means, broken_cells = dict(means), dict(cells)
    del broken_means[("bnci", "R2_current_prefix20_tail", 2, "all", 60)]
    with pytest.raises(ValueError, match="participant sets differ"):
        module.alignment_label_gains("bnci", broken_means, broken_cells, [], trial_sets)
    changed_ids = dict(trial_sets)
    changed_ids[("R2_current_prefix20_tail-1-0-60", "R2_current_prefix20_tail")] = ("different",)
    with pytest.raises(ValueError, match="different evaluation IDs"):
        module.alignment_label_gains("bnci", means, cells, [], changed_ids)


def test_bnci_family_excludes_descriptive_plain_sall_gain_and_holm_adjusts_six():
    module = load_analyzer(Path("/tmp"))
    contrasts = {name: {"signflip_p": (index + 1) / 100}
                 for index, name in enumerate(module.BNCI_CORE_FAMILY)}
    contrasts["plain_label_gain_Sall"] = {"signflip_p": .0001}
    family = module.build_bnci_family(
        {"endpoint_results": {"bnci": {"contrasts": contrasts}}},
        {"bnci": {"signflip_p": .06}})
    assert family["status"] == "complete" and len(family["holm_adjusted_p"]) == 6
    assert "plain_label_gain_Sall" not in family["raw_p"]


def test_core_loader_allows_matched_operation_ids_across_dataset_variants(tmp_path):
    module = load_analyzer(tmp_path)
    directory = tmp_path / "core"; directory.mkdir()
    summary = directory / "summary.json"; summary.write_text("{}")
    base = {"operation_id": "shared-op", "study_id": "openbmi_main",
            "method": "plain_ts", "target": 2, "draw": 0, "source_size": 100,
            "source_n": 100, "donor_count": "all", "h_total": 0,
            "evaluation_n": 100, "balanced_accuracy": .5}
    score_path = directory / "reconstructed_scores.csv"
    write_csv(score_path, [dict(base, dataset_variant="openbmi8"),
                           dict(base, dataset_variant="openbmi20")])
    _, _, operations, _ = module.load_core({
        "directory": "core", "summary_sha256": module.sha256(summary),
        "reconstructed_scores_sha256": module.sha256(score_path)})
    assert set(operations) == {("openbmi8", "shared-op"), ("openbmi20", "shared-op")}


def test_neural_float32_probability_tolerance_keeps_bounds_and_rejects_material_sum_error(tmp_path):
    module = load_analyzer(tmp_path)
    valid = []
    for index, truth in enumerate((0, 0, 1, 1)):
        if truth == 0:
            right, left = .7000000596, .3
        else:
            right, left = .3, .7000000596
        valid.append({"operation_id": "op", "trial_id": f"t{index}", "y_true": truth,
                      "y_pred": truth, "p_right": right, "p_left": left})
    path = tmp_path / "valid.csv.gz"; write_csv(path, valid, compressed=True)
    scores, _ = module.reconstruct_scores(
        path, ("operation_id",), ("p_right", "p_left"), sum_tolerance=1e-6)
    assert scores[("op",)] == 1
    invalid = [dict(row) for row in valid]
    invalid[0]["p_right"], invalid[0]["p_left"] = .701, .3
    bad = tmp_path / "invalid.csv.gz"; write_csv(bad, invalid, compressed=True)
    with pytest.raises(ValueError, match="probabilities"):
        module.reconstruct_scores(
            bad, ("operation_id",), ("p_right", "p_left"), sum_tolerance=1e-6)
