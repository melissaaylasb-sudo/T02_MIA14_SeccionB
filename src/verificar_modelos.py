"""Contrasta la comparación de regresores con sus predicciones y registros train.

La auditoría comprueba el protocolo ejecutado; no convierte la selección exploratoria
del menor error en una estimación independiente de su rendimiento futuro.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .experimentos_fe import metrics
from .feature_engineering import GEOMETRY, nominal_groups

MODELOS = {"Ridge", "ElasticNet", "SVR_RBF", "RandomForest", "GaussianProcess"}


def auditar_comparacion(data, config, partitions, predictions, folds, summary,
                       family, audits, reference, model_config):
    """Recalcula métricas y verifica aislamiento para los cinco modelos y dos CV."""
    # Importación diferida: verificar_fe invoca este módulo después de definir
    # la auditoría común, reutilizada por las cuatro representaciones y modelos.
    from .verificar_fe import auditar_preprocesamiento

    checks = []

    def check(condition, label):
        if not bool(condition):
            raise AssertionError("Comparación de modelos: " + label)
        checks.append("Modelos: " + label)

    def close(actual, expected, label):
        check(np.allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=False), label)

    check(set(model_config["modelos"]) == MODELOS, "cinco regresores previstos")
    check(model_config["representacion"] == "A", "representación común fijada en A")
    check(model_config["sin_busqueda_hiperparametros"] is True,
          "comparación declarada sin búsqueda de hiperparámetros")
    check(model_config["objetivo"] == config["target"] and model_config["unidad_objetivo"] == "N",
          "objetivo y unidad comunes a la ingeniería de características")
    check(not data.Case_n.duplicated().any(), "identificadores fuente únicos")
    by_case = data.set_index("Case_n")
    all_cases = set(by_case.index)
    partition_index = {(r["protocolo"], r["fold"]): r for r in partitions}
    check(len(partition_index) == len(partitions), "particiones sin claves duplicadas")
    protocols = {k[0] for k in partition_index}
    check(protocols == {"configuracion", "familia"}, "ambos protocolos presentes")
    grouping = {
        "configuracion": nominal_groups(by_case, config["group_columns"]),
        "familia": by_case.n_cells_Y.astype(int).astype(str),
    }
    for (protocol, fold), split in partition_index.items():
        train, test = split["train_cases"], split["test_cases"]
        label = f"{protocol}, partición {fold}"
        check(bool(train) and bool(test) and len(train) == len(set(train)) and
              len(test) == len(set(test)), "casos train/test únicos y no vacíos: " + label)
        check(set(train).isdisjoint(test) and set(train) | set(test) == all_cases,
              "partición completa y disjunta: " + label)
        check(set(grouping[protocol].loc[train]).isdisjoint(grouping[protocol].loc[test]),
              "grupo excluido del entrenamiento: " + label)
        check(len(set(grouping[protocol].loc[test])) == 1,
              "un grupo completo por partición: " + label)
        check(split["n_train"] == len(train) and split["n_test"] == len(test),
              "tamaños de partición consistentes: " + label)

    expected_pairs = {(p, m) for p in protocols for m in MODELOS}
    expected_keys = {(p, m, fold) for p, fold in partition_index for m in MODELOS}
    check(set(predictions.modelo) == MODELOS, "predicciones de todos los modelos")
    check(predictions.representacion.eq("A").all(), "predicciones con geometría original")
    check(not predictions.duplicated(["protocolo", "modelo", "Case_n"]).any(),
          "una predicción OOF por caso, modelo y protocolo")
    check(set(predictions[["protocolo", "modelo"]].itertuples(index=False, name=None)) == expected_pairs,
          "cobertura completa de protocolos y modelos")
    check(set(predictions[["protocolo", "modelo", "fold"]].itertuples(index=False, name=None)) == expected_keys,
          "predicciones en todas las particiones compartidas")
    check(np.isfinite(predictions[["observado_N", "predicho_N", "residuo_N", "error_absoluto_N"]]).all().all(),
          "predicciones y errores finitos")
    check(not folds.duplicated(["protocolo", "modelo", "fold"]).any(), "métricas por partición únicas")
    check(set(folds[["protocolo", "modelo", "fold"]].itertuples(index=False, name=None)) == expected_keys,
          "métricas de todas las particiones")
    check(not summary.duplicated(["protocolo", "modelo"]).any() and
          set(summary[["protocolo", "modelo"]].itertuples(index=False, name=None)) == expected_pairs,
          "resumen completo y único")
    check(summary.representacion.eq("A").all(), "resumen con representación común A")
    check(np.isfinite(folds.tiempo_s).all() and folds.tiempo_s.gt(0).all(),
          "tiempos de ajuste y predicción positivos y finitos")
    fold_index = folds.set_index(["protocolo", "modelo", "fold"])
    summary_index = summary.set_index(["protocolo", "modelo"])
    audit_keys = [(r["protocolo"], r["modelo"], r["fold"]) for r in audits]
    check(len(audit_keys) == len(set(audit_keys)) and set(audit_keys) == expected_keys,
          "auditoría train de todos los ajustes, sin duplicados")
    audit_index = dict(zip(audit_keys, audits))

    for (protocol, model), pred in predictions.groupby(["protocolo", "modelo"], sort=False):
        label = f"{protocol} / {model}"
        check(set(pred.Case_n) == all_cases, "todos los casos evaluados: " + label)
        close(pred.observado_N, by_case.loc[pred.Case_n, config["target"]], "objetivo observado correcto: " + label)
        close(pred.residuo_N, pred.observado_N - pred.predicho_N, "residuos correctos: " + label)
        close(pred.error_absoluto_N, pred.residuo_N.abs(), "errores absolutos correctos: " + label)
        close(pred.familia, by_case.loc[pred.Case_n, "n_cells_Y"], "familia fuente correcta: " + label)
        check(pred.configuracion.astype(str).tolist() == grouping["configuracion"].loc[pred.Case_n].tolist(),
              "configuración fuente correcta: " + label)
        row = summary_index.loc[(protocol, model)]
        for key, value in metrics(pred.observado_N, pred.predicho_N).items():
            close(row[key], value, f"métrica OOF {key}: " + label)
        ff = folds[(folds.protocolo == protocol) & (folds.modelo == model)]
        close(row.tiempo_s, ff.tiempo_s.sum(), "tiempo total coincide con ajustes: " + label)
        check(row.n_folds == len(ff), "particiones contadas correctamente: " + label)
        check(row.n_features_min == ff.n_features.min() and row.n_features_max == ff.n_features.max(),
              "dimensión de entradas correcta: " + label)
        check(row.predicciones_negativas == pred.predicho_N.lt(0).sum(),
              "predicciones negativas contabilizadas sin recortar: " + label)
        baseline = summary_index.loc[(protocol, "Ridge")]
        close(row.delta_MAE_vs_Ridge_N, row.MAE_N - baseline.MAE_N,
              "diferencia de MAE frente a Ridge: " + label)
        for fold, group in pred.groupby("fold"):
            split = partition_index[(protocol, fold)]
            item = fold_index.loc[(protocol, model, fold)]
            audit = audit_index[(protocol, model, fold)]
            detail = label + f", partición {fold}"
            check(set(group.Case_n) == set(split["test_cases"]), "casos OOF del grupo reservado: " + detail)
            check(audit["experimento"] == "A", "auditoría de geometría original: " + detail)
            check(audit["train_cases"] == split["train_cases"] and audit["test_cases"] == split["test_cases"],
                  "auditoría y partición coinciden: " + detail)
            check(item.n_train == len(split["train_cases"]) and item.n_test == len(split["test_cases"]),
                  "tamaños efectivos train/test: " + detail)
            check(str(item.grupo) == str(split["grupo_excluido"]), "grupo de evaluación registrado: " + detail)
            check(item.n_features == len(audit["feature_names"]), "dimensión efectiva por ajuste: " + detail)
            check(item.avisos == len(audit["avisos"]), "advertencias del ajuste registradas: " + detail)
            train_geometry = by_case.loc[split["train_cases"], GEOMETRY]
            held_geometry = by_case.loc[group.Case_n, GEOMETRY]
            outside = ((held_geometry < train_geometry.min()) |
                       (held_geometry > train_geometry.max())).sum(axis=1)
            close(group.fuera_rango_train, outside,
                  "extrapolación de entradas respecto al entrenamiento: " + detail)
            if model == "GaussianProcess":
                target_train = by_case.loc[split["train_cases"], config["target"]]
                close(audit["target_train_mean"], target_train.mean(),
                      "media del objetivo GPR aprendida en train: " + detail)
                # sklearn utiliza escala unitaria cuando el objetivo train es
                # constante, igual que en su normalización interna.
                scale = target_train.std(ddof=0)
                if scale == 0:
                    scale = 1.0
                close(audit["target_train_std"], scale,
                      "escala del objetivo GPR aprendida en train: " + detail)
            for key in ["MAE_N", "RMSE_N"]:
                # R² no se calcula por fold: una configuración puede tener una sola respuesta.
                value = (group.error_absoluto_N.mean() if key == "MAE_N"
                         else np.sqrt(np.mean(group.residuo_N.to_numpy() ** 2)))
                close(item[key], value, f"métrica por partición {key}: " + detail)

    for model in sorted(MODELOS):
        records = [r for r in audits if r["modelo"] == model]
        checked = auditar_preprocesamiento(data, records)
        check(len(checked) == len(partition_index), "medianas, filtro y escalador exclusivos de train: " + model)

    # La repetición de Ridge debe reproducir el A anterior; solo cambia el eje
    # de comparación (algoritmo), no sus entradas, particiones ni parámetros.
    ridge = predictions[predictions.modelo == "Ridge"].set_index(["protocolo", "Case_n"]).sort_index()
    original = reference[reference.experimento == "A"].set_index(["protocolo", "Case_n"]).sort_index()
    check(ridge.index.equals(original.index), "Ridge y A tienen las mismas predicciones reservadas")
    close(ridge.predicho_N, original.predicho_N, "Ridge reproduce exactamente la línea base A")
    close(ridge.fold, original.fold, "Ridge y A usan las mismas particiones")

    expected_families = {(p, m, f) for p, m in expected_pairs for f in data.n_cells_Y.unique()}
    check(not family.duplicated(["protocolo", "modelo", "familia"]).any() and
          set(family[["protocolo", "modelo", "familia"]].itertuples(index=False, name=None)) == expected_families,
          "desglose por familia completo y único")
    family_index = family.set_index(["protocolo", "modelo", "familia"])
    for key, pred in predictions.groupby(["protocolo", "modelo", "familia"]):
        row = family_index.loc[key]
        label = " / ".join(map(str, key))
        for metric, value in metrics(pred.observado_N, pred.predicho_N).items():
            close(row[metric], value, f"métrica por familia {metric}: " + label)
        close(row.sesgo_N, pred.residuo_N.mean(), "sesgo por familia: " + label)
        close(row.error_maximo_N, pred.error_absoluto_N.max(), "máximo error por familia: " + label)
        check(row.n_casos == len(pred), "soporte por familia: " + label)
    return checks


def verificar_modelos(root, data, config, partitions, reference):
    """Carga únicamente artefactos locales ejecutados, sin entrenar otros modelos."""
    root = Path(root)
    output = root / "reports/feature_engineering"
    read = lambda name: pd.read_csv(output / "tables" / f"{name}.csv")
    return auditar_comparacion(
        data, config, partitions,
        read("predicciones_modelos_oof"), read("metricas_modelos_folds"),
        read("comparacion_modelos"), read("errores_modelos_familia"),
        json.loads((output / "auditoria_modelos.json").read_text(encoding="utf-8")),
        reference, json.loads((root / "config/comparacion_modelos.json").read_text(encoding="utf-8")),
    )


def verificar_comparacion(root, frame):
    """Auditoría invocable desde el notebook antes de guardar sus salidas finales."""
    root = Path(root)
    out = root / "reports/feature_engineering"
    config = json.loads((root / "config/feature_engineering.json").read_text(encoding="utf-8"))
    partitions = json.loads((out / "particiones.json").read_text(encoding="utf-8"))
    reference = pd.read_csv(out / "tables/predicciones_oof.csv")
    return verificar_modelos(root, frame, config, partitions, reference)
