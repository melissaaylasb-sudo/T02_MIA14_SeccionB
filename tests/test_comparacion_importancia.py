"""Controles del contraste de cuatro variables frente a la geometría completa."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.ajuste_modelos import load_frozen_data, load_outer_splits
from src.comparacion_importancia import IMPORTANT, fixed_pipeline, run
from src.feature_engineering import TARGET
from src.transformaciones_eda import GEOMETRY

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def comparison(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, pd.DataFrame], Path]:
    """Ejecuta el contraste una vez sin sobrescribir los reportes publicados."""
    output = tmp_path_factory.mktemp("comparacion_importancia")
    return run(ROOT, output), output


def test_mismas_particiones_y_metricas_verificables(
        comparison: tuple[dict[str, pd.DataFrame], Path]) -> None:
    """Las dos representaciones deben predecir los mismos casos en cada fold."""
    tables, output = comparison
    pred = tables["predicciones_variables_importantes"]
    summary = tables["metricas_variables_importantes"]
    folds = tables["metricas_folds_variables_importantes"]
    frame = load_frozen_data(ROOT)
    outer, _, _ = load_outer_splits(frame, ROOT)
    assert len(summary) == 8
    assert len(pred) == len(frame) * 8
    assert len(folds) == 4 * sum(map(len, outer.values()))
    for (protocol, family, feature_set), rows in pred.groupby(["protocolo", "modelo", "conjunto"]):
        assert not rows.Case_n.duplicated().any()
        assert set(rows.Case_n) == set(frame.Case_n)
        record = summary[(summary.protocolo == protocol) & (summary.modelo == family)
                         & (summary.conjunto == feature_set)].iloc[0]
        y, p = rows.observado_N.to_numpy(), rows.predicho_N.to_numpy()
        assert record.MAE_N == pytest.approx(mean_absolute_error(y, p), abs=1e-10)
        assert record.MSE_N2 == pytest.approx(mean_squared_error(y, p), abs=1e-10)
        assert record.RMSE_N == pytest.approx(np.sqrt(mean_squared_error(y, p)), abs=1e-10)
        assert record.R2_OOF == pytest.approx(r2_score(y, p), abs=1e-10)
        assert record.n_variables_entrada == (10 if feature_set == "Todas" else 4)
        assert record.n_variables_modelo == (9 if feature_set == "Todas" else 4)
        for fold, (_, test) in enumerate(outer[protocol], 1):
            actual = set(rows.loc[rows.fold == fold, "Case_n"])
            assert actual == set(frame.iloc[test].Case_n)
    for (_, case), rows in pred.groupby(["protocolo", "Case_n"]):
        assert len(rows) == 4 and rows.fold.nunique() == 1 and rows.observado_N.nunique() == 1
    metadata = json.loads((output / "logs/comparacion_variables_importantes.json").read_text(encoding="utf-8"))
    assert metadata["hyperparameter_search"] is False
    assert tuple(metadata["selected_features"]) == IMPORTANT
    assert "Retrospectiva" in metadata["selection"]
    assert all((output / "tables" / f"{name}.csv").is_file() for name in tables)


def test_referencia_fija_y_variables_excluidas(
        comparison: tuple[dict[str, pd.DataFrame], Path]) -> None:
    """El modelo completo reproduce E1/S1; el reducido ignora las otras entradas."""
    tables, _ = comparison
    pred = tables["predicciones_variables_importantes"]
    previous = pd.read_csv(ROOT / "reports/feature_importance/tables/predicciones_oof.csv")
    for family, experiment in [("Ridge", "E1"), ("ElasticNet", "S1")]:
        now = pred[(pred.modelo == family) & (pred.conjunto == "Todas")]
        before = previous[previous.experimento == experiment]
        paired = now.merge(before, on=["protocolo", "fold", "Case_n"],
                           suffixes=("_nuevo", "_anterior"), validate="one_to_one")
        assert len(paired) == len(now) == len(before)
        assert np.allclose(paired.predicho_N_nuevo, paired.predicho_N_anterior, atol=1e-9)

    frame = load_frozen_data(ROOT)
    outer, _, _ = load_outer_splits(frame, ROOT)
    train, test = outer["configuracion"][0]
    X = frame[GEOMETRY]
    y = frame[TARGET]
    for family in ("Ridge", "ElasticNet"):
        model = fixed_pipeline(family, "Importantes")
        model.fit(X.iloc[train], y.iloc[train])
        estimator = model.named_steps["model"]
        assert isinstance(estimator, Ridge if family == "Ridge" else ElasticNet)
        assert estimator.alpha == (1.0 if family == "Ridge" else 0.1)
        assert model.named_steps["scale"].n_samples_seen_ == len(train)
        assert np.allclose(model.named_steps["imputer"].statistics_,
                           np.nanmedian(X.iloc[train].to_numpy(dtype=float), axis=0))
        changed = X.iloc[test].copy()
        for column in set(GEOMETRY) - set(IMPORTANT):
            changed[column] = changed[column] * 1.1
        assert np.allclose(model.predict(X.iloc[test]), model.predict(changed), atol=1e-10)
