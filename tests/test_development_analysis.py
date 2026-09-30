from pathlib import Path
import numpy as np
import pytest

from expose.development import load_config
from expose.development_analysis import ba, choose, disposition, policies


def test_recall_balancing_known_unequal_case():
    assert ba([0, 0, 0, 1], [0, 0, 0, 0]) == .5
    assert ba([0, 0, 0, 1], [1, 1, 1, 1]) == .5
    assert ba([0, 1], [0, 1]) == 1
    assert ba([0, 1], [1, 0]) == 0
    with pytest.raises(ValueError):
        ba([0, 0], [0, 1])


def test_ties_use_cost_then_order_and_prior_order():
    assert choose([.6, .6, .5], [.3, .2, .1]) == 1
    assert choose([.6, .6, .5], [.2, .2, .1]) == 0
    assert choose([.6, .6, .6, .6], order=[3, 2, 1, 0]) == 3
    with pytest.raises(ValueError):
        choose([.6, np.nan])


def test_lopo_never_selects_using_heldout_scores():
    cfg = load_config(Path(__file__).resolve().parents[1])
    rng = np.random.default_rng(12)
    classical = rng.uniform(.5, .9, (12, 3, 4))
    calibration = rng.uniform(.5, .9, (12, 4, 4))
    _, original, _ = policies(classical, calibration, [1, 2, 3], cfg)
    for i in range(12):
        a, b = classical.copy(), calibration.copy()
        a[i], b[i] = rng.uniform(0, 1, (3, 4)), rng.uniform(0, 1, (4, 4))
        _, altered, _ = policies(a, b, [1, 2, 3], cfg)
        assert original[i] == altered[i]
        assert original[i]['subject_id'] not in original[i]['training_subjects']


def test_policy_identity_and_dose_specific_challenge():
    cfg = load_config(Path(__file__).resolve().parents[1])
    a = np.tile([[.9, .5, .5, .5], [.6, .8, .8, .8], [.4, .4, .4, .4]], (12, 1, 1))
    b = np.full((12, 4, 4), .55)
    selected, choices, pooled = policies(a, b, [1, 2, 3], cfg)
    assert choices[0]['a0'] == 'csp_lda'
    assert choices[0]['a_pi'] == choices[0]['a_uniform'] == 'ts_lr'
    assert choices[0]['dose_specific'] == ['csp_lda', 'ts_lr', 'ts_lr', 'ts_lr']
    assert np.array_equal(selected['a_pi'], selected['a_uniform'])
    assert pooled['calibration_prior'] == 100


@pytest.mark.parametrize('g,cal,excess,control,wanted', [
    ([.02, .02, .02], .1, .1, True, 'SELECTION_SIGNAL'),
    ([.03, 0, .04], .03, .025, True, 'CALIBRATION_ONLY'),
    ([.03, 0, .04], .03, .0199, True, 'INCONCLUSIVE'),
    ([0, 0, 0], .01, .1, True, 'INCONCLUSIVE'),
    ([.05, .05, .05], .1, .1, False, 'ERROR'),
    ([.05, np.nan, .05], .1, .1, True, 'ERROR'),
])
def test_frozen_routing_requires_all_three_and_errors_take_priority(g, cal, excess, control, wanted):
    assert disposition(g, cal, excess, control) == wanted
