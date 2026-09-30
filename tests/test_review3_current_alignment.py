import numpy as np
from sklearn.covariance import OAS

from expose.review2_controls import fit_ea_whitener
from scripts.run_review3_current_alignment import current_ea_oas, current_whitener, empirical_ml


def test_each_dataset_plan_is_accepted_separately_with_mixed_source_size_types():
    from scripts.run_review3_current_alignment import STUDIES, applicable_operations
    for name, setting in STUDIES.items():
        operations = [dict(study_id=name, dataset=setting['dataset'], montage=setting['montage'],
                           mode=setting['mode'], method='ea_ts', donor_count='all',
                           source_size=source, h_total=h, draw=draw, target=target, tuning_id='source-only')
                      for source in setting['source_sizes'] for h in (0, 60)
                      for draw in range(10) for target in range(setting['targets'])]
        selected = applicable_operations(operations, setting['dataset'], setting['montage'])
        assert list(selected) == [name]
        assert len(selected[name]) == len(operations)


def test_analytic_ml_ea_then_oas_matches_sklearn_on_transformed_raw_epochs():
    raw = np.random.default_rng(71).normal(size=(100, 8, 750))
    empirical = empirical_ml(raw, 8)
    W = current_whitener(empirical, 8, 100)
    np.testing.assert_allclose(W, fit_ea_whitener(raw), rtol=0, atol=2e-14)
    analytic = current_ea_oas(empirical, W)
    transformed = np.einsum("ij,njt->nit", W, raw)
    sklearn = np.stack([OAS(store_precision=False).fit(epoch.T).covariance_ for epoch in transformed])
    np.testing.assert_allclose(analytic, sklearn, rtol=0, atol=2e-14)


def test_prefix_transform_does_not_depend_on_tail_values():
    epochs = np.random.default_rng(72).normal(size=(100, 8, 750))
    empirical = empirical_ml(epochs, 8)
    W = current_whitener(empirical[:20], 8, 100)
    changed = empirical.copy(); changed[20:] *= 3
    np.testing.assert_allclose(W, current_whitener(changed[:20], 8, 100), rtol=0, atol=0)


def test_bnci_channel_and_session_count_contract():
    raw = np.random.default_rng(73).normal(size=(144, 22, 750))
    values = empirical_ml(raw, 22)
    assert current_ea_oas(values[:2], current_whitener(values[:20], 22, 144)).shape == (2, 22, 22)
