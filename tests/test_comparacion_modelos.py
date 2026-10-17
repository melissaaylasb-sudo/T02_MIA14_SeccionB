"""Contratos compartidos, aislamiento y regresión de referencia en cinco modelos."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import is_regressor
from threadpoolctl import threadpool_limits

from src.comparacion_modelos import MODELOS, crear_modelo, evaluar_modelos, resumir_modelos, validar_particiones
from src.experimentos_fe import build_splits, load_dataset, metrics
from src.feature_engineering import TARGET, make_pipeline
from src.transformaciones_eda import GEOMETRY
from src.verificar_fe import auditar_preprocesamiento

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def entrada():
    configuracion_fe = json.loads((ROOT / "config/feature_engineering.json").read_text(encoding="utf-8"))
    configuracion = json.loads((ROOT / "config/comparacion_modelos.json").read_text(encoding="utf-8"))
    datos = load_dataset(ROOT, configuracion_fe)
    splits, grupos, registros = build_splits(datos, configuracion_fe)
    return datos, configuracion, splits, grupos, registros


@pytest.fixture(scope="module")
def evaluacion(entrada):
    with threadpool_limits(limits=1):
        return evaluar_modelos(*entrada)


@pytest.mark.parametrize("nombre", MODELOS)
def test_cada_modelo_es_regresor_y_rechaza_objetivo_como_entrada(entrada, nombre):
    datos, configuracion, *_ = entrada
    pipeline = crear_modelo(nombre, configuracion)
    assert is_regressor(pipeline)
    contaminadas = datos[GEOMETRY].copy()
    contaminadas[TARGET] = datos[TARGET]
    with pytest.raises(ValueError, match="exclusivamente"):
        pipeline.fit(contaminadas, datos[TARGET])


@pytest.mark.parametrize("nombre", MODELOS)
def test_prediccion_no_reajusta_estadisticas_de_train(entrada, nombre):
    datos, configuracion, splits, *_ = entrada
    train, test = splits["familia"][0]
    pipeline = crear_modelo(nombre, configuracion).fit(datos.iloc[train][GEOMETRY], datos.iloc[train][TARGET])
    mediana = pipeline.named_steps["imputer"].statistics_.copy()
    media = pipeline.named_steps["scale"].mean_.copy()
    escala = pipeline.named_steps["scale"].scale_.copy()
    reservadas = datos.iloc[test][GEOMETRY].copy()
    reservadas["Fiber_base"] = 999.0  # Adversario de aislamiento; no dato de la campaña.
    reservadas.loc[reservadas.index[0], "Fiber_height"] = np.nan
    assert np.isfinite(pipeline.predict(reservadas)).all()
    np.testing.assert_array_equal(pipeline.named_steps["imputer"].statistics_, mediana)
    np.testing.assert_array_equal(pipeline.named_steps["scale"].mean_, media)
    np.testing.assert_array_equal(pipeline.named_steps["scale"].scale_, escala)
    if nombre == "GaussianProcess":
        regresor = pipeline.named_steps["model"]
        assert regresor.optimizer is None
        assert regresor._y_train_mean == pytest.approx(datos.iloc[train][TARGET].mean())
        assert regresor._y_train_std == pytest.approx(datos.iloc[train][TARGET].std(ddof=0))


def test_los_cinco_comparten_reserva_y_estadisticas_por_particion(entrada, evaluacion):
    datos, configuracion, splits, grupos, registros = entrada
    predicciones, partes, auditorias = evaluacion
    assert set(predicciones.modelo) == set(MODELOS)
    assert len(auditorias) == len(MODELOS) * len(registros)
    for nombre in MODELOS:
        actuales = [a for a in auditorias if a["modelo"] == nombre]
        assert len(auditar_preprocesamiento(datos, actuales)) == len(registros)
        for protocolo in splits:
            resultados = predicciones[(predicciones.modelo == nombre) & (predicciones.protocolo == protocolo)]
            assert sorted(resultados.Case_n) == sorted(datos.Case_n)
            assert not resultados.Case_n.duplicated().any()
    for protocolo, agrupacion in partes.groupby("protocolo"):
        for numero, particion in agrupacion.groupby("fold"):
            assert particion.grupo.nunique() == 1
            assert particion.n_train.nunique() == 1
            assert particion.n_test.nunique() == 1
            assert particion.n_features.nunique() == 1


def test_ridge_reproduce_linea_base_y_metricas_se_recalculan(entrada, evaluacion):
    datos, configuracion, splits, *_ = entrada
    predicciones, particiones, _ = evaluacion
    resumen, familias = resumir_modelos(predicciones, particiones, configuracion)
    for protocolo, partes in splits.items():
        for numero, (train, test) in enumerate(partes, 1):
            referencia = make_pipeline("A").fit(datos.iloc[train][GEOMETRY], datos.iloc[train][TARGET])
            esperadas = referencia.predict(datos.iloc[test][GEOMETRY])
            observadas = predicciones[(predicciones.modelo == "Ridge") & (predicciones.protocolo == protocolo) & (predicciones.fold == numero)]
            np.testing.assert_allclose(observadas.predicho_N, esperadas, rtol=1e-12, atol=1e-12)
    for fila in resumen.itertuples(index=False):
        valores = predicciones[(predicciones.modelo == fila.modelo) & (predicciones.protocolo == fila.protocolo)]
        for metrica, valor in metrics(valores.observado_N, valores.predicho_N).items():
            assert getattr(fila, metrica) == pytest.approx(valor)
    assert len(familias) == len(MODELOS) * 2 * datos.n_cells_Y.nunique()


def test_particiones_con_fuga_o_registro_modificado_se_rechazan(entrada):
    datos, _, splits, grupos, registros = entrada
    alteradas = copy.deepcopy(splits)
    train, test = alteradas["configuracion"][0]
    alteradas["configuracion"][0] = (np.append(train, test[0]), test)
    with pytest.raises(ValueError, match="separa"):
        validar_particiones(datos, alteradas, grupos, registros)
    alterados = copy.deepcopy(registros)
    alterados[-1]["test_cases"][0] = -123
    with pytest.raises(ValueError, match="difieren"):
        validar_particiones(datos, splits, grupos, alterados)


def test_fuga_de_grupos_no_se_oculta_con_registro_coherente(entrada):
    datos, _, splits, grupos, registros = entrada
    alterados = {k: v.copy() for k, v in grupos.items()}
    train, test = splits["configuracion"][0]
    alterados["configuracion"].iloc[train[0]] = alterados["configuracion"].iloc[test[0]]
    with pytest.raises(ValueError, match="fuga"):
        validar_particiones(datos, splits, alterados, registros)
