"""Auditoría independiente de los artefactos OOF y las explicaciones del sprint."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

from src.ajuste_modelos import load_frozen_data, load_outer_splits


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "feature_importance"


@pytest.fixture(scope="module")
def artifacts() -> dict[str, object]:
    """Carga la ejecución publicada; su ausencia significa que el sprint está incompleto."""
    tables = OUT / "tables"
    logs = OUT / "logs"
    required = [
        "metricas_resumen", "metricas_folds", "predicciones_oof", "busquedas",
        "shap_oof", "shap_global", "shap_estabilidad", "permutacion", "ablaciones",
        "correlaciones", "comparacion_pareada", "coeficientes_folds",
    ]
    for name in required:
        assert (tables / f"{name}.csv").is_file(), f"Falta {name}.csv"
    assert (logs / "auditoria_folds.json").is_file()
    assert (logs / "manifest.json").is_file()
    result: dict[str, object] = {
        name: pd.read_csv(tables / f"{name}.csv") for name in required
    }
    result["auditoria"] = json.loads(
        (logs / "auditoria_folds.json").read_text(encoding="utf-8")
    )
    result["manifest"] = json.loads(
        (logs / "manifest.json").read_text(encoding="utf-8")
    )
    return result


def _audit_rows(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("auditoria", "folds", "records", "audits"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise AssertionError("La auditoría debe contener registros de folds")


def _bool(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().eq("true")


def test_mismas_particiones_y_cobertura_historica(artifacts: dict[str, object]) -> None:
    """E0–E4 y S1–S4 deben predecir cada caso en el mismo fold externo."""
    pred = artifacts["predicciones_oof"]
    assert isinstance(pred, pd.DataFrame)
    assert {"protocolo", "experimento", "fold", "Case_n", "observado_N", "predicho_N"} <= set(pred)
    frame = load_frozen_data(ROOT)
    outer, _, _ = load_outer_splits(frame, ROOT)
    expected_cases = set(frame.Case_n.astype(int))
    assert set(pred.experimento) >= {
        "E0", "E1", "E2", "E3", "E4", "S1", "S2", "S3", "S4"
    }
    assert set(pred.protocolo) == set(outer)
    for protocol, splits in outer.items():
        for experiment in sorted(set(pred.experimento)):
            subset = pred[(pred.protocolo == protocol) & (pred.experimento == experiment)]
            assert len(subset) == len(expected_cases)
            assert not subset.Case_n.duplicated().any()
            assert set(subset.Case_n.astype(int)) == expected_cases
            for fold, (_, test) in enumerate(splits, 1):
                actual = set(subset.loc[subset.fold == fold, "Case_n"].astype(int))
                expected = set(frame.iloc[test].Case_n.astype(int))
                assert actual == expected, (protocol, experiment, fold)
    for (_, case), rows in pred.groupby(["protocolo", "Case_n"]):
        assert rows.fold.nunique() == 1
        assert rows.observado_N.nunique() == 1


def test_metricas_recalculadas_desde_predicciones(artifacts: dict[str, object]) -> None:
    """Las métricas publicadas deben usar predicciones OOF sin redondeos intermedios."""
    pred = artifacts["predicciones_oof"]
    summary = artifacts["metricas_resumen"]
    assert isinstance(pred, pd.DataFrame) and isinstance(summary, pd.DataFrame)
    assert not summary.duplicated(["protocolo", "experimento"]).any()
    assert np.isfinite(pred[["observado_N", "predicho_N"]].to_numpy()).all()
    for (protocol, experiment), rows in pred.groupby(["protocolo", "experimento"]):
        record = summary[(summary.protocolo == protocol) & (summary.experimento == experiment)]
        assert len(record) == 1
        row = record.iloc[0]
        y, estimate = rows.observado_N.to_numpy(), rows.predicho_N.to_numpy()
        assert np.isclose(row.MAE_N, mean_absolute_error(y, estimate), atol=1e-9)
        assert np.isclose(row.RMSE_N, np.sqrt(mean_squared_error(y, estimate)), atol=1e-9)
        if "MedAE_N" in summary:
            assert np.isclose(row.MedAE_N, median_absolute_error(y, estimate), atol=1e-9)
        if "R2_OOF" in summary:
            assert np.isclose(row.R2_OOF, r2_score(y, estimate), atol=1e-9)
        if "sesgo_N" in summary:
            assert np.isclose(row.sesgo_N, np.mean(estimate - y), atol=1e-9)


def test_tuning_elegido_por_mae_interno(artifacts: dict[str, object]) -> None:
    """La selección registrada debe minimizar el MAE interno de su fold."""
    searches = artifacts["busquedas"]
    assert isinstance(searches, pd.DataFrame)
    assert {"protocolo", "experimento", "fold", "selected", "mean_inner_MAE_N"} <= set(searches)
    assert set(searches.experimento) >= {"E3", "E4", "S3", "S4"}
    assert np.isfinite(searches.mean_inner_MAE_N).all()
    for _, rows in searches.groupby(["protocolo", "experimento", "fold"]):
        selected = rows[_bool(rows.selected)]
        assert len(selected) == 1
        assert np.isclose(
            selected.mean_inner_MAE_N.iloc[0],
            rows.mean_inner_MAE_N.min(),
            atol=1e-9,
        )


def test_background_shap_y_nombres_pertenecen_al_fold(
    artifacts: dict[str, object],
) -> None:
    """El background y la explicación externa no pueden incluir entrenamiento ajeno."""
    explanation = artifacts["shap_oof"]
    pred = artifacts["predicciones_oof"]
    audit = _audit_rows(artifacts["auditoria"])
    assert isinstance(explanation, pd.DataFrame) and isinstance(pred, pd.DataFrame)
    keys = ["protocolo", "experimento", "fold", "Case_n"]
    required = set(keys + ["feature", "shap_N", "base_N", "predicho_N",
                           "background_train_cases_json"])
    assert required <= set(explanation)
    assert not explanation.duplicated(keys + ["feature"]).any()
    audit_lookup = {
        (row["protocolo"], row["experimento"], int(row["fold"])): row
        for row in audit
    }
    assert len(audit_lookup) == len(audit)
    for (protocol, experiment, fold, case), rows in explanation.groupby(keys):
        record = audit_lookup[(protocol, experiment, int(fold))]
        train_cases = set(map(int, record["train_cases"]))
        test_cases = set(map(int, record["test_cases"]))
        background = set(map(int, json.loads(rows.background_train_cases_json.iloc[0])))
        assert background
        assert background == set(map(int, record["background_cases"]))
        assert background <= train_cases
        assert not background & test_cases
        assert int(case) in test_cases
        assert list(rows.feature) == list(record["feature_names"])
        assert rows.base_N.nunique() == 1
        assert rows.predicho_N.nunique() == 1
        expected = pred[
            (pred.protocolo == protocol) & (pred.experimento == experiment)
            & (pred.fold == fold) & (pred.Case_n == case)
        ]
        assert len(expected) == 1
        assert np.isclose(rows.predicho_N.iloc[0], expected.predicho_N.iloc[0], atol=1e-8)
        assert np.isclose(
            rows.base_N.iloc[0] + rows.shap_N.sum(),
            rows.predicho_N.iloc[0],
            atol=1e-5,
        )


def test_auditoria_respeta_entrenamiento_externo(artifacts: dict[str, object]) -> None:
    """Cada background y ajuste debe quedar aislado de su test externo."""
    frame = load_frozen_data(ROOT)
    outer, _, _ = load_outer_splits(frame, ROOT)
    audit = _audit_rows(artifacts["auditoria"])
    for record in audit:
        protocol, fold = record["protocolo"], int(record["fold"])
        train, test = outer[protocol][fold - 1]
        expected_train = set(frame.iloc[train].Case_n.astype(int))
        expected_test = set(frame.iloc[test].Case_n.astype(int))
        assert set(map(int, record["train_cases"])) == expected_train
        assert set(map(int, record["test_cases"])) == expected_test
        assert set(map(int, record["background_cases"])) <= expected_train
        assert not set(map(int, record["background_cases"])) & expected_test


def test_contrastes_aislan_feature_set_y_tuning(artifacts: dict[str, object]) -> None:
    """Los contrastes Ridge y ElasticNet mantienen parámetro o representación."""
    audit = _audit_rows(artifacts["auditoria"])
    lookup = {
        (row["protocolo"], int(row["fold"]), row["experimento"]): row
        for row in audit
    }
    for protocol, fold in {(r["protocolo"], int(r["fold"])) for r in audit}:
        e1, e2, e3, e4 = (
            lookup[(protocol, fold, experiment)] for experiment in ("E1", "E2", "E3", "E4")
        )
        assert e1["selected_params"] == e2["selected_params"]
        assert e1["feature_names"] == e4["feature_names"]
        assert e2["feature_names"] == e3["feature_names"]
        assert len(e2["feature_names"]) >= len(e1["feature_names"])
        s1, s2, s3, s4 = (
            lookup[(protocol, fold, experiment)] for experiment in ("S1", "S2", "S3", "S4")
        )
        assert s1["selected_params"] == s2["selected_params"]
        assert s1["feature_names"] == s4["feature_names"]
        assert s2["feature_names"] == s3["feature_names"]
        assert e1["feature_names"] == s1["feature_names"]
        assert e2["feature_names"] == s2["feature_names"]


def test_matriz_conserva_variables_fisicas_y_hipotesis_de_trabajo() -> None:
    """Las cuatro filas, sus vistas y los criterios operativos deben concordar."""
    import csv
    import hashlib

    csv_path = ROOT / "docs/matriz_consistencia.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)
    expected_columns = [
        "Problema de investigación", "Objetivo", "Hipótesis",
        "Variables independientes y variable dependiente", "Indicadores",
    ]
    assert columns == expected_columns
    assert len(rows) == 4
    assert all(all(row[column].strip() for column in columns) for row in rows)
    assert [row["Problema de investigación"].split(".", 1)[0] for row in rows] == [
        "General", "Específico 1", "Específico 2", "Específico 3",
    ]
    assert [row["Hipótesis"].split(".", 1)[0] for row in rows] == ["HG", "H1", "H2", "H3"]
    assert all("para " in row["Objetivo"] for row in rows)
    assert all("<br>" in row["Hipótesis"] for row in rows)
    for row in rows:
        variables = row["Variables independientes y variable dependiente"]
        assert "carga experimental de primera falla [N]" in variables
        assert all(term in variables for term in [
            "Número de celdas en Y", "número total de pivotes",
            "Altura, base y longitud total de fibras", "Altura y radio de pivote",
            "Volúmenes declarados de fibras, pivotes y espécimen",
        ])
        assert "radio es constante" in variables
    markdown = (ROOT / "docs/matriz_consistencia.md").read_text(encoding="utf-8")
    table = markdown.split("<!-- MATRIZ_CSV_INICIO -->", 1)[1].split("<!-- MATRIZ_CSV_FIN -->", 1)[0].strip()
    expected = "\n".join([
        "| " + " | ".join(columns) + " |",
        "|---|---|---|---|---|",
        *["| " + " | ".join(row[column].replace("|", "&#124;") for column in columns) + " |" for row in rows],
    ])
    assert table == expected
    assert "SHAP" not in markdown and "SHAP" not in csv_path.read_text(encoding="utf-8")
    criteria = json.loads((ROOT / "config/criterios_entrega_permutacion.json").read_text(encoding="utf-8"))
    assert criteria["fecha_congelacion"] == "2026-10-30"
    assert criteria["fuente_unica_matriz"] == "docs/matriz_consistencia.csv"
    assert criteria["matriz_csv_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    model_path = ROOT / criteria["configuracion_modelos"]
    assert criteria["configuracion_modelos_sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()
    models = json.loads(model_path.read_text(encoding="utf-8"))["modelos"]
    assert criteria["modelos_fijos"] == list(models) and criteria["n_modelos"] == 5
    assert criteria["busqueda_hiperparametros"] is False
    assert len(criteria["entradas_originales"]) == 10
    assert len({entry["columna"] for entry in criteria["entradas_originales"]}) == 10
    assert criteria["constante_documentada"]["columna"] == "Pivot_radius"
    assert criteria["permutacion_externa"]["repeticiones"] == 100
    assert criteria["seleccion_interna"]["repeticiones_permutacion"] == 20
    assert criteria["seleccion_interna"]["n_splits"] == 5
    assert criteria["seleccion_interna"]["n_variables"] == 4
    assert criteria["seleccion_interna"]["permitir_importancia_no_positiva"] is True
    assert criteria["bloque_hipotesis_H3"]["columnas"] == [
        "n_cells_Y", "Fiber_total_length", "Pivot_total_number",
        "Fiber_total_volume", "Pivot_total_volume", "Sample_total_volume",
    ]
    hypotheses = criteria["hipotesis"]
    assert hypotheses["HG"]["protocolos"] == ["configuracion"]
    assert hypotheses["H1"]["protocolos"] == ["configuracion", "familia"]
    assert hypotheses["H1"]["referencias"] == ["mediana", "media_por_nivel"]
    assert hypotheses["H1"]["conjunto"] == "completo"
    assert hypotheses["H2"]["protocolos"] == ["configuracion", "familia"]
    assert hypotheses["H3"]["protocolos"] == ["configuracion"]
    assert hypotheses["H3"]["min_modelos"] == 3 and hypotheses["H3"]["total_modelos"] == 5
    assert len(hypotheses["H3"]["operadores"]) == 2
    assert "intersección" in hypotheses["H3"]["cumplimiento"]
    assert criteria["umbral_cientifico_N"] == 0.0
    assert criteria["tolerancia_numerica_N"] == 1e-10
