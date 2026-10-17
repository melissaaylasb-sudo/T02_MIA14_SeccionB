"""Pruebas de frontera temporal, ajuste aislado y dependencia por configuración."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from src.experimentos_fe import build_splits, load_dataset, metrics
from src.feature_engineering import GEOMETRY, LOG_RATIO, MOMENTS, REDUNDANT, TARGET, make_pipeline
from src.verificar_fe import auditar_preprocesamiento

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def data():
    config = json.loads((ROOT / "config/feature_engineering.json").read_text(encoding="utf-8"))
    return load_dataset(ROOT, config), config


def test_responses_and_identifiers_are_rejected(data):
    frame, _ = data
    for forbidden in [TARGET, "Ultimate_Load", "Maximum_Displacement", "Energy_Fracture", "Case_n"]:
        x = frame[GEOMETRY].copy()
        x[forbidden] = np.arange(len(x))
        with pytest.raises(ValueError, match="exclusivamente"):
            make_pipeline().fit(x, frame[TARGET])


def test_validation_values_do_not_change_learned_statistics(data):
    frame, _ = data
    train = frame[GEOMETRY].iloc[:15].copy()
    train.loc[train.index[0], "Fiber_base"] = np.nan
    pipe = make_pipeline("C").fit(train, frame[TARGET].iloc[:15])
    expected = train.median().to_numpy()
    np.testing.assert_allclose(pipe.named_steps["imputer"].statistics_, expected)
    before_mean = pipe.named_steps["scale"].mean_.copy()
    before_scale = pipe.named_steps["scale"].scale_.copy()
    held_out = frame[GEOMETRY].iloc[15:].copy()
    held_out["Fiber_base"] = 999.0  # Prueba adversaria de aislamiento, no dato experimental añadido.
    held_out.loc[held_out.index[0], "Fiber_height"] = np.nan
    assert np.isfinite(pipe.predict(held_out)).all()
    np.testing.assert_array_equal(pipe.named_steps["scale"].mean_, before_mean)
    np.testing.assert_array_equal(pipe.named_steps["scale"].scale_, before_scale)
    np.testing.assert_array_equal(pipe.named_steps["imputer"].statistics_, expected)
    assert pipe.named_steps["scale"].n_samples_seen_ == len(train)


def test_constant_filter_learns_only_from_train(data):
    frame, _ = data
    train = frame.loc[frame.n_cells_Y == 4]
    pipe = make_pipeline().fit(train[GEOMETRY], train[TARGET])
    cols = pipe.named_steps["features"].get_feature_names_out()
    assert not pipe.named_steps["variance"].get_support()[list(cols).index("n_cells_Y")]
    assert np.isfinite(pipe.predict(frame[GEOMETRY])).all()


def test_nominal_groups_are_kept_together(data):
    frame, config = data
    splits, groups, records = build_splits(frame, config)
    for protocol, partitions in splits.items():
        seen = []
        for train, test in partitions:
            assert set(train).isdisjoint(test)
            assert set(groups[protocol].iloc[train]).isdisjoint(groups[protocol].iloc[test])
            seen.extend(test)
        assert sorted(seen) == list(range(len(frame)))
    by_case = frame.set_index("Case_n").index
    for a, b in [(19,25),(24,27)]:
        for record in records:
            if record["protocolo"] == "configuracion":
                assert (a in record["test_cases"]) == (b in record["test_cases"])


def test_formulas_and_single_interventions(data):
    frame, _ = data
    x, y = frame[GEOMETRY], frame[TARGET]
    represented = {}
    for exp in "ABCD":
        pipe = clone(make_pipeline(exp)).fit(x, y)
        represented[exp] = pipe[:3].transform(x)
    b, h = x.Fiber_base, x.Fiber_height
    np.testing.assert_allclose(represented["B"][MOMENTS[0]], b*h**3/12)
    np.testing.assert_allclose(represented["B"][MOMENTS[1]], h*b**3/12)
    np.testing.assert_allclose(represented["C"][LOG_RATIO], np.log(x.Pivot_total_volume/x.Fiber_total_volume))
    pd.testing.assert_frame_equal(represented["B"][GEOMETRY], represented["A"])
    pd.testing.assert_frame_equal(represented["C"].drop(columns=LOG_RATIO), represented["B"])
    pd.testing.assert_frame_equal(represented["D"], represented["C"].drop(columns=REDUNDANT))


@pytest.mark.parametrize("bad", [0, -1, np.inf])
def test_invalid_domains_fail(data, bad):
    frame, _ = data
    x = frame[GEOMETRY].copy()
    x.loc[0,"Fiber_height"] = bad
    with pytest.raises(ValueError, match="positiva"):
        make_pipeline("C").fit(x, frame[TARGET])


def test_completely_missing_training_column_fails(data):
    frame, _ = data
    x = frame[GEOMETRY].copy()
    x["Fiber_base"] = np.nan
    with pytest.raises(ValueError, match="carece"):
        make_pipeline().fit(x, frame[TARGET])


def test_target_never_creates_features(data):
    frame, _ = data
    x, y = frame[GEOMETRY], frame[TARGET]
    a = make_pipeline("C").fit(x, y)
    b = make_pipeline("C").fit(x, y.iloc[::-1].to_numpy())
    np.testing.assert_allclose(a[:-1].transform(x), b[:-1].transform(x))


def test_metrics_use_original_units():
    observed = np.array([10.,20.,30.])
    predicted = np.array([12.,18.,33.])
    scores = metrics(observed, predicted)
    assert scores["MAE_N"] == mean_absolute_error(observed, predicted)
    assert scores["RMSE_N"] == pytest.approx(np.sqrt(17/3))


@pytest.mark.parametrize("field", ["scaler_mean", "scaler_scale"])
def test_preprocessing_audit_detects_corrupted_statistics_in_last_fold(data, field):
    frame, _ = data
    records = json.loads((ROOT / "reports/feature_engineering/auditoria_transformaciones.json").read_text(encoding="utf-8"))
    checked = auditar_preprocesamiento(frame, records)
    assert len(checked) == len(records)
    # Corromper la última partición exige recorrer la auditoría completa.
    records[-1][field][0] += 1.0
    with pytest.raises(AssertionError, match="entrenamiento incorrect"):
        auditar_preprocesamiento(frame, records)


def test_preprocessing_audit_ignores_reserved_values_and_target(data):
    frame, _ = data
    records = json.loads((ROOT / "reports/feature_engineering/auditoria_transformaciones.json").read_text(encoding="utf-8"))
    record = next(r for r in records if r["experimento"] == "C")
    expected = auditar_preprocesamiento(frame, [record])
    altered = frame.copy()
    altered.loc[altered.Case_n.isin(record["test_cases"]), GEOMETRY] = 999.0
    altered[TARGET] = -999.0
    pd.testing.assert_frame_equal(auditar_preprocesamiento(altered, [record]), expected)
