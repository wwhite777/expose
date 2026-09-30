"""Synthetic guard tests for review3 preparation; never touches project MAT/cache arrays."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_review3_openbmi.py"
spec = importlib.util.spec_from_file_location("review3_preparation", SCRIPT)
review3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review3)


def test_proposed_names_are_exactly_20_and_present():
    assert set(review3.LEGACY_CHANNELS) <= set(review3.PROPOSED_20_MOTOR_CHANNELS)
    names = list(review3.PROPOSED_20_MOTOR_CHANNELS) + ["unused"]
    assert review3.validate_channel_names(review3.PROPOSED_20_MOTOR_CHANNELS, names) == list(range(20))
    with pytest.raises(ValueError, match="absent"):
        review3.validate_channel_names(review3.PROPOSED_20_MOTOR_CHANNELS, names[:-2])


def test_extreme_mapping_keeps_boundary_membership_without_exclusion():
    rows = [{"trial_id": "t0", "trial_index_zero_based": "0", "event_sample_zero_based": "1000"}]
    assert review3.intervals_for_sample(1500, rows)[0]["in_legacy_0p5_to_3p5_window"]
    assert not review3.intervals_for_sample(4500, rows)[0]["in_legacy_0p5_to_3p5_window"]
    assert review3.intervals_for_sample(4500, rows)[0]["in_raw_4s_support"]


def test_segmented_field_distinguishes_t_minus_one_from_offset_plus_one():
    # Supplied field begins at MATLAB t, as documented; this does not alter the legacy anchor.
    signal = np.arange(5000 * 20, dtype=np.int64).reshape(5000, 20)
    rows = [{"trial_index_zero_based": "0", "event_sample_matlab": "10"}]
    supplied = signal[10:4010, :20][:, None, :]
    result = review3.audit_segmented_field({"smt": supplied}, signal, list(range(20)), rows)
    assert result["exact_matches_legacy_t_minus_1"] == 0
    assert result["exact_matches_alternate_t"] == 1


def test_confirmation_role_set_is_rejected_before_raw_open():
    cfg = {"source_subjects": [1], "development_subjects": [2]}
    expected = review3.expected_roles(cfg)
    records = [{"subject": 1, "session": 1, "role": "source"},
               {"subject": 2, "session": 1, "role": "development"},
               {"subject": 2, "session": 2, "role": "development"},
               {"subject": 3, "session": 1, "role": "confirmation"}]
    with pytest.raises(ValueError, match="frozen source/development"):
        review3.validate_record_roles(records, expected)
