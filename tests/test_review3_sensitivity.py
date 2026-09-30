import csv
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def load_analyzer(tmp_path=None):
    spec = importlib.util.spec_from_file_location(
        "analyze_review3_sensitivity", REPO / "scripts/analyze_review3_sensitivity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tmp_path is not None:
        module.ROOT = tmp_path
    return module


def score(variant, target, draw, value, method="plain_ts", source="100",
          history="60", study="openbmi_main", donor="all", operation=None):
    return {"dataset_variant": variant, "operation_id": operation or f"{variant}-{target}-{draw}",
            "study_id": study, "method": method, "target": str(target), "draw": str(draw),
            "source_size": str(source), "source_n": str(source), "donor_count": str(donor),
            "h_total": str(history), "evaluation_n": "100", "balanced_accuracy": str(value)}


def paired_rows():
    result = []
    for target in (2, 7):
        for draw in range(10):
            result.extend([score("openbmi8", target, draw, .5),
                           score("openbmi20", target, draw, .6),
                           score("openbmi_offset", target, draw, .45)])
    return result


def test_paired_montage_and_offset_deltas_are_draw_averaged_with_participant_ids():
    module = load_analyzer()
    vectors, summaries = module.paired_sensitivities(paired_rows())
    assert len(vectors) == 4
    montage = [row for row in vectors if row["variant"] == "openbmi20"]
    offset = [row for row in vectors if row["variant"] == "openbmi_offset"]
    assert [row["target"] for row in montage] == ["2", "7"]
    assert all(row["difference_pp"] == pytest.approx(10) for row in montage)
    assert all(row["difference_pp"] == pytest.approx(-5) for row in offset)
    assert summaries["openbmi20/plain_ts/100/60"]["targets"] == [2, 7]


@pytest.mark.parametrize("mutation", ["missing_pair", "missing_draw"])
def test_paired_sensitivity_rejects_missing_pair_or_draw(mutation):
    module = load_analyzer()
    values = paired_rows()
    if mutation == "missing_pair":
        values = [row for row in values
                  if not (row["dataset_variant"] == "openbmi8"
                          and row["target"] == "2" and row["draw"] == "0")]
    else:
        values = [row for row in values
                  if not (row["dataset_variant"] == "openbmi20"
                          and row["target"] == "2" and row["draw"] == "0")]
    with pytest.raises(ValueError, match="unpaired|draw"):
        module.paired_sensitivities(values)


def test_training_fold_winners_preserve_ties_and_report_changed_set():
    module = load_analyzer()
    people = []
    for target in (1, 2, 3):
        for method in ("a", "b"):
            for history in (0, 60):
                value = .5 if history == 0 else (.7 if method == "a" else .6)
                people.append({"dataset_variant": "openbmi8", "study_id": "openbmi_main",
                               "source_size": "100", "donor_count": "all", "target": str(target),
                               "method": method, "h_total": str(history),
                               "balanced_accuracy": str(value)})
    output = module.winners(people)
    assert len(output) == 12
    assert all(row["h0_winners"] == "a|b" for row in output)
    h0 = [row for row in output if row["h_total"] == 0]
    h60 = [row for row in output if row["h_total"] == 60]
    assert all(row["winners"] == "a|b" and row["winner_set_changed"] is False for row in h0)
    assert all(row["winners"] == "a" and row["winner_set_changed"] is True for row in h60)
    assert all(row["margin_to_best_pp"] == pytest.approx(0) for row in h60 if row["method"] == "a")


def test_source_diversity_compares_three_and_six_donors_with_eighteen_at_s300():
    module = load_analyzer()
    people = []
    for target in (1, 2, 3):
        for method in ("plain_ts", "mdwm"):
            for history in (0, 60):
                for donors, offset in (("3", -.04), ("6", -.01), ("18", 0.0)):
                    people.append({
                        "dataset_variant": "openbmi8", "study_id": "openbmi_diversity",
                        "source_size": "300", "donor_count": donors, "target": str(target),
                        "method": method, "h_total": str(history),
                        "balanced_accuracy": str(.6 + offset)})
    result = module.diversity(people)
    assert len(result) == 8
    assert result["plain_ts/h0/donors3_minus18"]["values_pp"] == pytest.approx([-4, -4, -4])
    assert result["mdwm/h60/donors6_minus18"]["values_pp"] == pytest.approx([-1, -1, -1])


def write_csv(path, values):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader(); writer.writerows(values)


def historical_fixture(tmp_path, module, inconsistent=False):
    directory = tmp_path / "grid"; directory.mkdir()
    historical, scores = [], []
    for target in (1, 2, 3):
        for draw in range(10):
            operation = f"op-{target}-{draw}"
            observed = .75 if inconsistent and target == 1 and draw == 0 else .5
            historical.append({"operation_id": operation, "method": "plain_ts",
                               "target": target, "draw": draw, "source_size": 100,
                               "h_total": 4, "correct_count": 2, "history_n": 4,
                               "balanced_accuracy": observed, "history_membership": "h",
                               "h0_fit_signature": "fit"})
            scores.extend([score("bnci", target, draw, .55 + target / 100,
                                 study="bnci_main", history="4", operation=operation),
                           score("bnci", target, draw, .5, study="bnci_main",
                                 history="0", operation=f"h0-{target}-{draw}")])
    path = directory / "historical_decodability.csv"
    write_csv(path, historical)
    receipt = {"status": "completed", "outputs": {path.name: module.sha(path)}}
    (directory / "receipt.json").write_text(json.dumps(receipt))
    return {"historical_grids": [{"directory": "grid", "label": "bnci"}]}, scores


def test_historical_decodability_checks_charged_budget_and_reports_each_draw(tmp_path):
    module = load_analyzer(tmp_path)
    spec, scores = historical_fixture(tmp_path, module)
    raw, aggregate, correlations = module.decodability(spec, scores)
    assert len(raw) == 30 and len(aggregate) == 3 and len(correlations) == 1
    assert all(row["historical_ba"] == pytest.approx(50) for row in aggregate)
    assert len(correlations[0]["individual_draw_spearman"]) == 10


def test_historical_decodability_rejects_correct_count_inconsistent_with_saved_ba(tmp_path):
    module = load_analyzer(tmp_path)
    spec, scores = historical_fixture(tmp_path, module, inconsistent=True)
    with pytest.raises(ValueError, match="charged balanced historical score is inconsistent"):
        module.decodability(spec, scores)
