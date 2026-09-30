import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/analyze_review4_confirmation.py"
SPEC = importlib.util.spec_from_file_location("review4_confirmation", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


PEOPLE = list(range(1, 25))


def operations():
    result = []
    for participant in PEOPLE:
        for draw in range(10):
            for condition in (*MOD.PRIMARY, *MOD.OPTIONAL):
                result.append({"operation_id": f"d{draw}_p{participant}_{condition}",
                               "participant_id": participant, "draw": draw,
                               "condition": condition, "setting": MOD.SETTING,
                               "source_n": 100, "added_n": 0 if condition == "base" else 60,
                               "h_total": 0 if condition == "base" else 60, "C": 1.0})
    return result


def primary_scores(effect_pp=5.0, people=PEOPLE):
    rows = []
    for participant in people:
        for draw in range(10):
            for condition in MOD.PRIMARY:
                ba = 0.65 + effect_pp / 100 if condition == "own" else 0.65
                rows.append({"operation_id": f"d{draw}_p{participant}_{condition}",
                             "participant_id": str(participant), "draw": str(draw),
                             "condition": condition, "setting": MOD.SETTING,
                             "source_n": "100", "added_n": "60", "h_total": "60",
                             "C": "1.0", "balanced_accuracy": str(ba), "evaluation_n": "100"})
    return rows


@pytest.mark.parametrize("effect,expected", [(5.0, "supportive"),
                                               (1.0, "positive_below_threshold"),
                                               (0.0, "uncertain"),
                                               (-5.0, "adverse")])
def test_primary_classifications(effect, expected):
    result, effects, _, statuses = MOD.analyze_rows(
        primary_scores(effect), operations(), PEOPLE, bootstrap_draws=100, signflip_draws=100)
    assert result["primary"]["classification"] == expected
    assert len(effects) == 24 and all(row["status"] == "complete" for row in statuses)


def test_fewer_than_twenty_complete_is_operationally_incomplete_and_bounded():
    result, effects, _, statuses = MOD.analyze_rows(
        primary_scores(5.0, PEOPLE[:19]), operations(), PEOPLE,
        bootstrap_draws=100, signflip_draws=100)
    assert result["primary"]["classification"] == "operationally_incomplete"
    assert len(effects) == 19
    assert sum(row["status"] == "incomplete" for row in statuses) == 5
    assert result["primary"]["full24_missing_bound_low_pp"] < 0
    assert result["primary"]["full24_missing_bound_high_pp"] > 0


def test_optional_rows_may_match_mechanically_complete_people():
    complete = PEOPLE[:23]
    rows = primary_scores(5.0, complete)
    for participant in complete:
        for draw in range(10):
            for condition in MOD.OPTIONAL:
                rows.append({"operation_id": f"d{draw}_p{participant}_{condition}",
                             "participant_id": str(participant), "draw": str(draw),
                             "condition": condition, "setting": MOD.SETTING,
                             "source_n": "100", "added_n": "0" if condition == "base" else "60",
                             "h_total": "0" if condition == "base" else "60", "C": "1.0",
                             "balanced_accuracy": "0.65", "evaluation_n": "100"})
    result, effects, _, statuses = MOD.analyze_rows(
        rows, operations(), PEOPLE, bootstrap_draws=10, signflip_draws=10)
    assert len(effects) == 23 and result["optional"]["base_absolute_ba_percent"]["n"] == 23
    assert sum(row["status"] == "incomplete" for row in statuses) == 1


def test_duplicate_and_extra_rows_are_rejected():
    rows = primary_scores()
    with pytest.raises(ValueError, match="duplicate"):
        MOD.analyze_rows(rows + [dict(rows[0])], operations(), PEOPLE,
                         bootstrap_draws=10, signflip_draws=10)
    extra = [dict(row) for row in rows]; extra[0]["operation_id"] = "unknown"
    with pytest.raises(ValueError, match="extra or unknown"):
        MOD.analyze_rows(extra, operations(), PEOPLE, bootstrap_draws=10, signflip_draws=10)


@pytest.mark.parametrize("field,value,match", [
    ("h_total", "40", "dose"),
    ("setting", "legacy_C1_pooled_refit", "setting"),
    ("balanced_accuracy", "1.001", "outside"),
])
def test_wrong_dose_setting_and_range_are_rejected(field, value, match):
    rows = primary_scores(); rows[0][field] = value
    with pytest.raises(ValueError, match=match):
        MOD.analyze_rows(rows, operations(), PEOPLE, bootstrap_draws=10, signflip_draws=10)
