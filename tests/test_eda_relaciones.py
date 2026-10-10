"""Invariantes del análisis por parejas y de las derivaciones experimentales."""
import numpy as np
import pandas as pd
from scipy import stats

from src.eda_relaciones import (
    TARGET, INPUTS, CYCLE_COLUMNS, association_diagnostics, build_analysis_frame,
    univariate_summary, _matrix_pairwise, _algebraic_warning,
)


def test_association_uses_only_pairwise_finite_cases_and_preserves_case_ids():
    x = np.array([1, 2, np.nan, 4, 5, 6, np.inf])
    y = np.array([1, 3, 0, 2, 15, 4, 10])
    case_ids = np.arange(101, 108)
    result = association_diagnostics(x, y, case_ids=case_ids)
    keep = np.isfinite(x) & np.isfinite(y)
    assert result["n_efectivo"] == 5
    assert np.isclose(result["Pearson"], stats.pearsonr(x[keep], y[keep]).statistic)
    assert np.isclose(result["Spearman"], stats.spearmanr(x[keep], y[keep]).statistic)
    assert np.isclose(result["Kendall_tau_b"], stats.kendalltau(x[keep], y[keep]).statistic)
    assert result["Pearson_LOO_caso_influyente"] in case_ids[keep]
    omitted = [stats.spearmanr(np.delete(x[keep], i), np.delete(y[keep], i)).statistic for i in range(5)]
    assert np.isclose(result["Spearman_LOO_min"], min(omitted))
    assert np.isclose(result["Spearman_LOO_max"], max(omitted))


def test_constant_or_too_sparse_association_is_unidentified_not_zero():
    for x, y in [(np.ones(5), np.arange(5)), ([1, np.nan, 2], [4, 2, 6])]:
        result = association_diagnostics(x, y)
        assert np.isnan(result["Pearson"])
        assert np.isnan(result["Spearman"])
        assert np.isnan(result["Spearman_LOO_signo_pct"])
        assert result["Spearman_LOO_evaluaciones_validas"] == 0


def test_rank_invariance_and_loo_are_exact_for_monotonic_data_with_ties():
    x = np.array([1, 1, 2, 3, 4, 4, 5], dtype=float)
    y = x**3
    result = association_diagnostics(x, y)
    assert result["Pearson"] < result["Spearman"]
    assert np.isclose(result["Spearman"], 1)
    assert np.isclose(result["Kendall_tau_b"], 1)
    assert np.isclose(result["Spearman_LOO_min"], 1)
    assert np.isclose(result["Spearman_LOO_max"], 1)
    assert result["Spearman_LOO_signo_pct"] == 100


def test_matrix_effective_size_is_per_pair_and_matches_independent_scipy():
    df = pd.DataFrame({"a": [1., 2, 3, 4, 5], "b": [1., np.nan, 2, 3, 4], "c": [1., 1, 1, 1, 1]})
    values, sizes = _matrix_pairwise(df, ["a", "c"], ["a", "b"], "spearman")
    assert sizes.loc["a", "a"] == 5 and sizes.loc["a", "b"] == 4
    assert np.isclose(values.loc["a", "b"], stats.spearmanr(df.a, df.b, nan_policy="omit").statistic)
    assert np.isnan(values.loc["c", "a"])


def _synthetic_source():
    df = pd.DataFrame({"Case_n": [10, 20, 30], **{c:[1., 2., 3.] for c in INPUTS},
        TARGET:[10., np.nan, 30.], "Ultimate_Load":[5., np.nan, 0.],
        "Maximum_Displacement":[65., np.nan, 75.],
        **{c:[float(i*2), np.nan, float(i*4)] for i,c in enumerate(CYCLE_COLUMNS,1)}})
    energies = {
        "Energy_Dissipated":pd.DataFrame({"cycle_1":[-1.,np.nan,2.]},index=pd.Index([10,20,30],name="Case_n")),
        "Energy_Fracture":pd.DataFrame({"event_1":[10.,np.nan,12.],"event_2":[2.,np.nan,np.nan]},index=pd.Index([10,20,30],name="Case_n"))}
    return df, energies


def test_derivation_preserves_absence_and_sign_and_does_not_impute_or_mutate_source():
    df, energies = _synthetic_source()
    original = df.copy(deep=True)
    result, metadata = build_analysis_frame(df, energies)
    pd.testing.assert_frame_equal(df, original)
    assert pd.isna(result.loc[20, "Fracture_observed_sum"])
    assert pd.isna(result.loc[20, "Fracture_registered_events"])
    assert result.loc[10, "Dissipated_cycle_1"] == -1
    assert result.loc[10, "Fracture_observed_sum"] == 12
    assert result.loc[10, "Fracture_registered_events"] == 2
    assert np.isclose(result.loc[10, "Fracture_initial_share"], 10/12)
    assert pd.isna(result.loc[30, "Load_ratio_first_over_ultimate"])
    assert np.isclose(result.loc[10, "Recovery_complement_cycle6"] + result.loc[10, "Residual_fraction_cycle6"], 1)
    assert metadata["Dissipated_cycle_1"]["unidad"] == "mJ"
    assert metadata["Fiber_area_proxy"]["rol"] == "INPUT derivada"
    assert "directa" in _algebraic_warning(TARGET,"Load_per_pivot",metadata)
    assert "Complementos" in _algebraic_warning("Residual_fraction_cycle6","Recovery_complement_cycle6",metadata)


def test_univariate_mad_cv_and_sample_std_definitions():
    frame=pd.DataFrame({"x":[1., 2, 3, 4, np.nan], "zero":[-2., -1, 1, 2, np.nan]})
    metadata={c:{"rol":"INPUT","unidad":"mm","formula":"fuente","limitacion":"descriptivo"} for c in frame}
    summary=univariate_summary(frame,list(frame),metadata).set_index("variable")
    assert summary.loc["x","n_efectivo"] == 4
    assert summary.loc["x","MAD"] == 1
    assert np.isclose(summary.loc["x","desviacion"],np.std([1,2,3,4],ddof=1))
    assert summary.loc["x","IQR"] == 1.5
    assert pd.isna(summary.loc["zero","CV_pct"])
