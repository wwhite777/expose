"""Run the full v1 fixture contract against v2, plus the r004-F10 regression."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
V2_PATH = ROOT / "scripts/verify_day5_raw_reference_v2.py"
V2_SPEC = importlib.util.spec_from_file_location("day5_raw_reference_v2", V2_PATH)
day5_v2 = importlib.util.module_from_spec(V2_SPEC)
V2_SPEC.loader.exec_module(day5_v2)

V1_TEST_PATH = ROOT / "tests/test_day5_raw_reference.py"
V1_SPEC = importlib.util.spec_from_file_location("day5_raw_reference_v1_suite", V1_TEST_PATH)
v1_suite = importlib.util.module_from_spec(V1_SPEC)
V1_SPEC.loader.exec_module(v1_suite)
v1_suite.day5 = day5_v2
v1_suite.SCRIPT = V2_PATH

# Reuse all 16 production-helper fixtures/assertions with their module globals
# rebound to the successor implementation. Parametrization marks are preserved.
raw_struct = v1_suite.raw_struct
test_source_is_syntactically_isolated = v1_suite.test_source_is_syntactically_isolated
test_valid_raw_parser_reconstructs_exact_recipe = v1_suite.test_valid_raw_parser_reconstructs_exact_recipe
test_reconstructed_split_and_history_use_raw_labels = v1_suite.test_reconstructed_split_and_history_use_raw_labels
test_class_code_name_swap_is_rejected = v1_suite.test_class_code_name_swap_is_rejected
test_trial_level_class_crosschecks_are_enforced = v1_suite.test_trial_level_class_crosschecks_are_enforced
test_fractional_event_is_rejected_before_index_coercion = v1_suite.test_fractional_event_is_rejected_before_index_coercion
test_actual_one_sample_smt_discrepancy_is_enforced = v1_suite.test_actual_one_sample_smt_discrepancy_is_enforced
test_in_bounds_one_sample_event_shift_is_rejected = v1_suite.test_in_bounds_one_sample_event_shift_is_rejected
test_failed_producer_rejected_before_other_artifacts_are_opened = v1_suite.test_failed_producer_rejected_before_other_artifacts_are_opened
test_confirmation_data_row_rejected_by_production_manifest_guard = v1_suite.test_confirmation_data_row_rejected_by_production_manifest_guard
test_shared_n0_duplicate_collapse_agrees = v1_suite.test_shared_n0_duplicate_collapse_agrees
test_shared_n0_duplicate_disagreement_is_rejected = v1_suite.test_shared_n0_duplicate_disagreement_is_rejected
test_prediction_comparison_success_exercises_both_class_recalls = v1_suite.test_prediction_comparison_success_exercises_both_class_recalls
test_changed_saved_prediction_is_rejected = v1_suite.test_changed_saved_prediction_is_rejected
test_completion_marker_binds_exact_owned_outputs = v1_suite.test_completion_marker_binds_exact_owned_outputs


def test_nonfinite_used_epoch_support_fails_before_smt_diagnostic(raw_struct):
    changed = dict(raw_struct)
    changed["x"] = raw_struct["x"].copy()
    onset = int(raw_struct["t"][7]) - 1
    changed["x"][onset + 25, 0] = float("nan")
    with pytest.raises(ValueError, match="selected raw epoch support contains nonfinite values") as caught:
        day5_v2.reconstruct_raw_struct(changed, {}, build_epochs=False)
    assert "smt one-sample discrepancy" not in str(caught.value)
