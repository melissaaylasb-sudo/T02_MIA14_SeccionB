import numpy as np
from src.eda import adjust_pvalues, permutation_rank_test, rank_association, ols_influence


def test_family_determined_feature_is_not_identifiable_conditionally():
    groups = np.repeat([0, 1, 2], 4)
    x = groups * 10
    y = x + np.tile([0, 1, 2, 3], 3)
    effect, p = permutation_rank_test(x, y, groups, resamples=99)
    assert np.isnan(effect) and np.isnan(p)


def test_conditioning_detects_reversal_and_permutation_is_reproducible():
    groups = np.repeat([0, 1, 2], 4)
    x = 10 * groups + np.tile([0, 1, 2, 3], 3)
    y = 20 * groups - np.tile([0, 1, 2, 3], 3)
    assert rank_association(x, y) > 0
    assert np.isclose(rank_association(x, y, groups), -1)
    a = permutation_rank_test(x, y, groups, resamples=199, seed=3)
    assert a == permutation_rank_test(x, y, groups, resamples=199, seed=3)
    assert 0 < a[1] <= 1


def test_fdr_keeps_unidentified_tests_absent():
    values = adjust_pvalues([.01, np.nan, .04])
    np.testing.assert_allclose(values[[0, 2]], [.02, .04])
    assert np.isnan(values[1])


def test_ols_residual_orthogonality_and_leverage_trace():
    x = np.arange(12, dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    response = 3 + 2 * x + np.sin(x)
    diagnostics, _ = ols_influence(design, response)
    np.testing.assert_allclose(design.T @ diagnostics.residuo_N, 0, atol=1e-10)
    assert np.isclose(diagnostics.leverage.sum(), design.shape[1])
    assert (diagnostics.Cook >= 0).all()
