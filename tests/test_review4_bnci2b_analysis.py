import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("review4_bnci2b_analysis", Path(__file__).parents[1] / "scripts/analyze_review4_bnci2b.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_constant_participant_effect_has_exact_interval_and_counts():
    result = module.summary([2.5] * 9)
    assert result["mean_pp"] == result["ci_low_pp"] == result["ci_high_pp"] == 2.5
    assert (result["positive"], result["zero"], result["negative"]) == (9, 0, 0)


def test_reversed_effect_preserves_interval_pairing_and_signs():
    x = np.array([-5., -2., -1., 0., 2., 3., 4., 5., 6.])
    a, b = module.summary(x), module.summary(-x)
    assert a["mean_pp"] == -b["mean_pp"]
    assert np.isclose(a["ci_low_pp"], -b["ci_high_pp"])
    assert np.isclose(a["ci_high_pp"], -b["ci_low_pp"])
    assert a["positive"] == b["negative"]


@pytest.mark.parametrize("bad", [[1.] * 8, [1.] * 10, [1.] * 8 + [np.nan]])
def test_incomplete_extra_or_nonfinite_participants_rejected(bad):
    with pytest.raises(ValueError, match="nine complete"):
        module.summary(bad)
