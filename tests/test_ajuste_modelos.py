"""Aislamiento del ajuste anidado, referencia histórica y selección interna."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_limits

from src.feature_engineering import GEOMETRY, TARGET, make_pipeline, nominal_groups
from src.ajuste_modelos import (
    inner_splits,
    load_config,
    load_frozen_data,
    load_outer_splits,
    model_specs,
    select_inner_winner,
)
from src.experimentos_fe import metrics
from src.verificar_ajuste import (
    BASELINE,
    MODEL_NAMES,
    SELECTOR,
    audit_outer,
    audit_inner,
    audit_predictions,
    audit_search,
    audit_train,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_SHA = "2d0a17f54bdc6a6361aaa6e25a57cf67775c7d47a5e2adefb408f114defa21f8"


@pytest.fixture(scope="module")
def entrada():
    data = load_frozen_data(ROOT)
    splits, groups, records = load_outer_splits(data, ROOT)
    return data, splits, groups, records


def test_snapshot_y_particiones_reproducen_semana5(entrada):
    data, _, _, records = entrada
    path = ROOT / "data/processed/preliminary/model_table.csv"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == DATA_SHA
    historical = json.loads((ROOT / "reports/feature_engineering/particiones.json").read_text(encoding="utf-8"))
    assert records == historical
    assert {"Case_n", TARGET, *GEOMETRY}.issubset(data.columns)
    assert set(data.columns) - {"Case_n", TARGET, *GEOMETRY} <= {"__source_record"}
    assert not data.Case_n.duplicated().any()
    assert np.isfinite(data[TARGET]).all()


def test_ambos_niveles_respetan_grupos_y_reserva_externa(entrada):
    data, splits, groups, _ = entrada
    for protocol, partitions in splits.items():
        outer_seen = []
        for outer_train, outer_test in partitions:
            assert set(outer_train).isdisjoint(outer_test)
            assert set(groups[protocol].iloc[outer_train]).isdisjoint(groups[protocol].iloc[outer_test])
            inside = inner_splits(data, outer_train, protocol)
            seen = []
            for train, valid in inside:
                train, valid = np.asarray(train), np.asarray(valid)
                assert set(train).isdisjoint(valid)
                assert set(train) | set(valid) == set(range(len(outer_train)))
                global_train, global_valid = np.asarray(outer_train)[train], np.asarray(outer_train)[valid]
                assert set(global_train).isdisjoint(outer_test)
                assert set(global_valid).isdisjoint(outer_test)
                assert set(groups[protocol].iloc[global_train]).isdisjoint(groups[protocol].iloc[global_valid])
                seen.extend(valid.tolist())
            assert sorted(seen) == list(range(len(outer_train)))
            outer_seen.extend(np.asarray(outer_test).tolist())
        assert sorted(outer_seen) == list(range(len(data)))


def test_busqueda_ajusta_imputacion_y_escala_solo_en_train(entrada, monkeypatch):
    """Registra cada fit real del contrato durante GridSearchCV, incluido refit."""
    data, splits, _, _ = entrada
    outer_train, outer_test = splits["configuracion"][0]
    x = data[GEOMETRY].copy()
    x.loc[x.index[outer_test], "Fiber_base"] = 999999.0
    x.loc[x.index[outer_train[0]], "Fiber_height"] = np.nan
    from src.feature_engineering import GeometryContract

    calls = []
    original_fit = GeometryContract.fit

    def tracked_fit(self, X, y=None):
        calls.append(tuple(X.index))
        return original_fit(self, X, y)

    monkeypatch.setattr(GeometryContract, "fit", tracked_fit)
    inside = inner_splits(data, outer_train, "configuracion")
    search = GridSearchCV(make_pipeline("A"), {"model__alpha": [0.1, 1.0]},
                          cv=inside, scoring="neg_mean_absolute_error", n_jobs=1,
                          refit=True, error_score="raise")
    with threadpool_limits(limits=1):
        search.fit(x.iloc[outer_train], data[TARGET].iloc[outer_train])
    expected = {tuple(x.index[np.asarray(outer_train)[train]]) for train, _ in inside}
    expected.add(tuple(x.index[outer_train]))
    assert set(calls) == expected
    assert all(set(indices).isdisjoint(x.index[outer_test]) for indices in calls)
    model = search.best_estimator_
    np.testing.assert_allclose(model.named_steps["imputer"].statistics_,
                               x.iloc[outer_train].median().to_numpy())
    assert model.named_steps["scale"].n_samples_seen_ == len(outer_train)
    original_median = model.named_steps["imputer"].statistics_.copy()
    original_mean = model.named_steps["scale"].mean_.copy()
    original_scale = model.named_steps["scale"].scale_.copy()
    assert np.isfinite(model.predict(x.iloc[outer_test])).all()
    np.testing.assert_array_equal(model.named_steps["imputer"].statistics_, original_median)
    np.testing.assert_array_equal(model.named_steps["scale"].mean_, original_mean)
    np.testing.assert_array_equal(model.named_steps["scale"].scale_, original_scale)


def test_target_reservado_no_cambia_busqueda_ni_preprocesamiento(entrada):
    data, splits, _, _ = entrada
    train, test = splits["configuracion"][0]
    altered = data[TARGET].copy()
    altered.iloc[test] = -999999.0
    inside = inner_splits(data, train, "configuracion")
    results = []
    for target in [data[TARGET], altered]:
        search = GridSearchCV(make_pipeline("A"), {"model__alpha": [0.1, 1.0]},
                              cv=inside, scoring="neg_mean_absolute_error", n_jobs=1,
                              refit=True, error_score="raise")
        with threadpool_limits(limits=1):
            search.fit(data.iloc[train][GEOMETRY], target.iloc[train])
        results.append(search)
    assert results[0].best_params_ == results[1].best_params_
    np.testing.assert_array_equal(results[0].cv_results_["mean_test_score"],
                                  results[1].cv_results_["mean_test_score"])
    np.testing.assert_array_equal(results[0].predict(data.iloc[test][GEOMETRY]),
                                  results[1].predict(data.iloc[test][GEOMETRY]))


def test_target_e_identificadores_no_entran_en_pipeline(entrada):
    data, *_ = entrada
    for field in [TARGET, "Case_n", "__source_record", "Ultimate_Load", "Energy_Fracture"]:
        bad = data[GEOMETRY].copy()
        bad[field] = np.arange(len(bad))
        with pytest.raises(ValueError, match="exclusivamente"):
            make_pipeline("A").fit(bad, data[TARGET])


def test_selector_usa_mae_interno_y_no_resultado_externo():
    # Los valores son scores negativos de validación interna, como en sklearn.
    scores = {"Ridge": -12.0, "ElasticNet": -10.0, "SVR_RBF": -14.0}
    assert select_inner_winner(scores) == "ElasticNet"
    assert scores == {"Ridge": -12.0, "ElasticNet": -10.0, "SVR_RBF": -14.0}
    assert select_inner_winner({"Ridge": -10.0, "ElasticNet": -10.0}) == "Ridge"
    with pytest.raises(ValueError):
        select_inner_winner({"Ridge": np.nan})


def test_modelos_conservan_pipeline_y_semillas_declaradas(entrada):
    data, splits, _, _ = entrada
    config = load_config(ROOT)
    specs = model_specs(config)
    assert tuple(specs) == MODEL_NAMES
    train, test = splits["familia"][0]
    for name, (pipeline, grid) in specs.items():
        assert list(pipeline.named_steps) == ["contract", "imputer", "features", "variance", "scale", "model"]
        assert pipeline.named_steps["features"].variant == "A"
        model = pipeline.named_steps["model"]
        params = model.get_params()
        if "random_state" in params and name != "Ridge":
            assert params["random_state"] == 42
        if name == "Ridge":
            assert params["solver"] == "svd"  # Solver determinista, sin muestreo.
        assert all(key.startswith("model__") for key in grid)
        with threadpool_limits(limits=1):
            pipeline.fit(data.iloc[train][GEOMETRY], data.iloc[train][TARGET])
        assert np.isfinite(pipeline.predict(data.iloc[test][GEOMETRY])).all()
        assert pipeline.named_steps["scale"].n_samples_seen_ == len(train)


def test_loader_rechaza_datos_modificados(tmp_path):
    config = load_config(ROOT)
    target_config = tmp_path / "config/ajuste_modelos.json"
    target_config.parent.mkdir(parents=True)
    target_config.write_text(json.dumps(config), encoding="utf-8")
    dataset = tmp_path / config["dataset"]
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes((ROOT / config["dataset"]).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="snapshot"):
        load_frozen_data(tmp_path)


def test_auditor_interno_detecta_contaminacion_y_cobertura_incompleta(entrada):
    data, splits, _, outer = entrada
    inside = []
    for protocol, partitions in splits.items():
        for fold, (train, test) in enumerate(partitions, 1):
            for inner_fold, (itrain, valid) in enumerate(inner_splits(data, train, protocol), 1):
                inside.append({"protocolo": protocol, "fold": fold, "inner_fold": inner_fold,
                    "train_cases": data.iloc[np.asarray(train)[itrain]].Case_n.astype(int).tolist(),
                    "validation_cases": data.iloc[np.asarray(train)[valid]].Case_n.astype(int).tolist(),
                    "outer_test_cases": data.iloc[test].Case_n.astype(int).tolist()})
    assert audit_inner(data, outer, inside).labels
    bad = copy.deepcopy(inside)
    bad[-1]["train_cases"].append(bad[-1]["outer_test_cases"][0])
    with pytest.raises(AssertionError, match="dentro del train externo"):
        audit_inner(data, outer, bad)
    with pytest.raises(AssertionError, match="número y orden"):
        audit_inner(data, outer, inside[:-1])


def test_reajuste_baseline_reproduce_todas_predicciones_historicas(entrada):
    data, splits, _, _ = entrada
    path = ROOT / "reports/feature_engineering/tables/predicciones_oof.csv"
    historical = pd.read_csv(path)
    historical = historical[historical.experimento == "A"].set_index(["protocolo", "Case_n"])
    with threadpool_limits(limits=1):
        for protocol, partitions in splits.items():
            for fold, (train, test) in enumerate(partitions, 1):
                model = make_pipeline("A", alpha=1.0).fit(data.iloc[train][GEOMETRY], data.iloc[train][TARGET])
                expected = historical.loc[[(protocol, int(case)) for case in data.iloc[test].Case_n]]
                np.testing.assert_allclose(model.predict(data.iloc[test][GEOMETRY]), expected.predicho_N,
                                           rtol=1e-10, atol=1e-10)
                assert expected.fold.eq(fold).all()


@pytest.fixture(scope="module")
def evidencia_control(entrada):
    """Copias de la referencia para probar el auditor; no son experimentos nuevos."""
    data, _, _, outer = entrada
    base = ROOT / "reports/feature_engineering"
    reference = pd.read_csv(base / "tables/predicciones_oof.csv")
    p0 = reference[reference.experimento == "A"].copy()
    f0 = pd.read_csv(base / "tables/metricas_folds.csv")
    f0 = f0[f0.experimento == "A"].copy()
    a0 = json.loads((base / "auditoria_transformaciones.json").read_text(encoding="utf-8"))
    a0 = [a for a in a0 if a["experimento"] == "A"]
    predictions, folds, audits, summaries = [], [], [], []
    for model in [BASELINE, *MODEL_NAMES, SELECTOR]:
        p = p0.assign(modelo=model)
        predictions.append(p)
        folds.append(f0.assign(modelo=model))
        if model != SELECTOR:
            for record in a0:
                audit = {**copy.deepcopy(record), "modelo": model}
                if model == "GaussianProcess":
                    ytrain = data.set_index("Case_n").loc[record["train_cases"], TARGET]
                    audit["target_train_mean"] = float(ytrain.mean())
                    audit["target_train_std"] = float(ytrain.std(ddof=0)) or 1.0
                audits.append(audit)
        for protocol, group in p.groupby("protocolo"):
            summaries.append({"protocolo": protocol, "modelo": model,
                              **metrics(group.observado_N, group.predicho_N)})
    return data, outer, pd.concat(predictions, ignore_index=True), pd.concat(folds, ignore_index=True), pd.DataFrame(summaries), reference, audits


def test_verificador_recalcula_metricas_y_detecta_manipulacion(evidencia_control):
    data, outer, pred, folds, summary, reference, _ = evidencia_control
    assert audit_predictions(data, outer, pred, folds, summary, reference).labels
    bad = summary.copy()
    bad.loc[bad.index[-1], "MAE_N"] += 1.0
    with pytest.raises(AssertionError, match="métrica OOF MAE_N"):
        audit_predictions(data, outer, pred, folds, bad, reference)
    bad = pred.copy()
    row = bad.index[(bad.modelo == BASELINE) & (bad.protocolo == "familia")][-1]
    bad.loc[row, "observado_N"] += 0.5
    with pytest.raises(AssertionError, match="objetivo original"):
        audit_predictions(data, outer, bad, folds, summary, reference)


def test_verificador_rechaza_fuga_externa_y_estadisticas_alteradas(evidencia_control):
    data, outer, _, _, _, _, audits = evidencia_control
    assert audit_outer(data, outer).labels
    assert audit_train(data, audits, outer).labels
    bad_outer = copy.deepcopy(outer)
    bad_outer[-1]["train_cases"].append(bad_outer[-1]["test_cases"][0])
    with pytest.raises(AssertionError, match="completa y disjunta"):
        audit_outer(data, bad_outer)
    bad_audit = copy.deepcopy(audits)
    bad_audit[-1]["scaler_mean"][0] += 1.0
    with pytest.raises(AssertionError, match="Media de entrenamiento incorrecta"):
        audit_train(data, bad_audit, outer)


def test_auditor_rechaza_selector_distinto_del_ganador_interno(evidencia_control):
    """Control fabricado del verificador; no registra resultados de modelos reales."""
    data, outer, pred, folds, _, _, _ = evidencia_control
    pred, folds = pred.copy(), folds.copy()
    pred["modelo_seleccionado"] = pred.modelo.where(pred.modelo.ne(SELECTOR), "Ridge")
    folds["inner_MAE_N"] = np.nan
    folds["params_json"] = '{}'
    inner, searches, selections = [], [], []
    for split in outer:
        protocol, fold = split["protocolo"], split["fold"]
        n_inner = 5 if protocol == "configuracion" else 2
        inner.extend({"protocolo": protocol, "fold": fold, "inner_fold": i + 1} for i in range(n_inner))
        scores = {}
        for model_index, model in enumerate(MODEL_NAMES):
            score = float(10 + model_index)
            scores[model] = -score
            mask = folds.protocolo.eq(protocol) & folds.fold.eq(fold) & folds.modelo.eq(model)
            folds.loc[mask, "inner_MAE_N"] = score
            for candidate in range(2):
                searches.append({"protocolo": protocol, "modelo": model, "fold": fold,
                    "candidate_id": candidate, "params_json": '{}',
                    "mean_inner_MAE_N": score + candidate, "std_inner_MAE_N": 0.0,
                    "rank": candidate + 1, "selected": candidate == 0,
                    **{f"split{i}_MAE_N": score + candidate for i in range(n_inner)}})
        selections.append({"protocolo": protocol, "fold": fold, "modelo_seleccionado": "Ridge",
                           "scores_internos_neg_MAE": scores, "test_cases": split["test_cases"]})
    table = pd.DataFrame(searches)
    assert audit_search(table, selections, inner, pred, folds).labels
    bad = copy.deepcopy(selections)
    bad[-1]["modelo_seleccionado"] = "ElasticNet"
    with pytest.raises(AssertionError, match="solo por validación interna"):
        audit_search(table, bad, inner, pred, folds)
    altered = table.copy()
    altered.loc[altered.index[-1], "mean_inner_MAE_N"] += 0.2
    with pytest.raises(AssertionError, match="media interna recalculada"):
        audit_search(altered, selections, inner, pred, folds)
