"""Contraste exploratorio de variables originales completas frente a cuatro importantes.

Las cuatro variables se eligieron retrospectivamente por el SHAP global de Ridge
E1 sobre estos mismos 26 casos. Por ello, esta comparación describe sensibilidad
del desempeño al conjunto de entradas; no constituye validación independiente
de la selección. Se reutilizan particiones externas congeladas y parámetros fijos.

Ejecutar desde la raíz: ``python -m src.comparacion_importancia``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .ajuste_modelos import ROOT, load_frozen_data, load_outer_splits
from .feature_engineering import TARGET, make_pipeline
from .transformaciones_eda import GEOMETRY

IMPORTANT = ("Pivot_total_number", "Fiber_total_length", "Sample_total_volume", "n_cells_Y")
FAMILIES = ("Ridge", "ElasticNet")
SETS = ("Todas", "Importantes")


class SelectGeometry(TransformerMixin, BaseEstimator):
    """Conserva columnas declaradas después de representar la geometría."""

    def __init__(self, columns: tuple[str, ...]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: object = None) -> SelectGeometry:
        if not isinstance(X, pd.DataFrame) or X.columns.has_duplicates:
            raise ValueError("La selección requiere un DataFrame con nombres únicos.")
        if len(set(self.columns)) != len(self.columns) or not set(self.columns) <= set(X.columns):
            raise ValueError("Las columnas importantes deben existir y ser únicas.")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        if not isinstance(X, pd.DataFrame) or list(X.columns) != list(self.feature_names_in_):
            raise ValueError("El orden de columnas cambió entre entrenamiento y evaluación.")
        return X.loc[:, list(self.columns)].copy()

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        check_is_fitted(self)
        return np.asarray(self.columns, dtype=object)


def fixed_pipeline(family: str, feature_set: str) -> Pipeline:
    """Construye la misma familia y parámetros para ambos conjuntos de entradas."""
    if family not in FAMILIES or feature_set not in SETS:
        raise ValueError("Modelo o conjunto de variables no reconocido.")
    pipeline = make_pipeline("A")
    if family == "ElasticNet":
        pipeline.set_params(model=ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000,
                                             tol=0.0001, selection="cyclic", random_state=42))
    if feature_set == "Importantes":
        pipeline = Pipeline(pipeline.steps[:3] + [("selector", SelectGeometry(IMPORTANT))]
                            + pipeline.steps[3:])
    return pipeline


def _scores(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calcula las cuatro métricas a partir de predicciones emparejadas."""
    y = np.asarray(observed, dtype=float)
    estimate = np.asarray(predicted, dtype=float)
    if y.shape != estimate.shape or y.size == 0 or not np.isfinite(estimate).all():
        raise ValueError("Las predicciones deben ser finitas y emparejadas.")
    mse = float(mean_squared_error(y, estimate))
    return {"MAE_N": float(mean_absolute_error(y, estimate)), "MSE_N2": mse,
            "RMSE_N": float(np.sqrt(mse)),
            "R2_OOF": float(r2_score(y, estimate)) if y.size > 1 else np.nan}


def run(root: Path | str = ROOT, output: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Evalúa dos familias × dos conjuntos en 24+3 folds y exporta tablas OOF.

    ``output`` señala la carpeta que contendrá ``tables`` y ``logs``. Si se
    omite, se utilizan los reportes del proyecto. La imputación, la selección
    por varianza y la escala se ajustan de nuevo en cada entrenamiento externo.
    """
    root = Path(root).resolve()
    destination = Path(output).resolve() if output is not None else root / "reports/feature_importance"
    source = root / "reports/feature_importance/tables/shap_global.csv"
    importance = pd.read_csv(source)
    ranked = (importance[importance.protocolo.eq("configuracion")
                         & importance.experimento.eq("E1")]
              .sort_values("mean_abs_shap_N", ascending=False))
    if tuple(ranked.feature.head(len(IMPORTANT))) != IMPORTANT:
        raise ValueError("Las cuatro variables declaradas ya no coinciden con SHAP E1.")
    frame = load_frozen_data(root)
    splits, _, _ = load_outer_splits(frame, root)
    X = frame[GEOMETRY]
    y = frame[TARGET].to_numpy(dtype=float)
    predictions: list[dict] = []
    fold_metrics: list[dict] = []
    for protocol, outer in splits.items():
        for family in FAMILIES:
            for feature_set in SETS:
                for fold, (train, test) in enumerate(outer, 1):
                    if set(train) & set(test) or len(train) + len(test) != len(frame):
                        raise ValueError("El fold externo tiene solapamiento o cobertura incompleta.")
                    model = fixed_pipeline(family, feature_set)
                    model.fit(X.iloc[train], y[train])
                    estimate = np.asarray(model.predict(X.iloc[test]), dtype=float)
                    before_variance = (model.named_steps["selector"].get_feature_names_out()
                                       if feature_set == "Importantes"
                                       else model.named_steps["features"].get_feature_names_out())
                    names = before_variance[model.named_steps["variance"].get_support()]
                    if feature_set == "Importantes" and tuple(names) != IMPORTANT:
                        raise ValueError("El modelo reducido no conserva exactamente las cuatro variables.")
                    row = {"protocolo": protocol, "modelo": family, "conjunto": feature_set,
                           "fold": fold, "n_train": len(train), "n_test": len(test),
                           "n_variables_entrada": len(GEOMETRY) if feature_set == "Todas" else len(IMPORTANT),
                           "n_variables_modelo": len(names), **_scores(y[test], estimate)}
                    fold_metrics.append(row)
                    for pos, predicted in zip(test, estimate, strict=True):
                        predictions.append({"protocolo": protocol, "modelo": family,
                                            "conjunto": feature_set, "fold": fold,
                                            "Case_n": int(frame.iloc[pos].Case_n),
                                            "observado_N": float(y[pos]), "predicho_N": float(predicted)})
    pred = pd.DataFrame(predictions)
    folds = pd.DataFrame(fold_metrics)
    summary_rows: list[dict] = []
    for (protocol, family, feature_set), cases in pred.groupby(
            ["protocolo", "modelo", "conjunto"], sort=False):
        if len(cases) != len(frame) or cases.Case_n.duplicated().any():
            raise ValueError("Cada comparación debe predecir cada caso exactamente una vez.")
        sizes = folds[(folds.protocolo == protocol) & (folds.modelo == family)
                      & (folds.conjunto == feature_set)]
        if sizes.n_variables_modelo.nunique() != 1:
            raise ValueError("El número de columnas modeladas cambió entre folds.")
        summary_rows.append({"protocolo": protocol, "modelo": family, "conjunto": feature_set,
                             "n_casos": len(cases), "n_folds": len(sizes),
                             "n_variables_entrada": int(sizes.n_variables_entrada.iloc[0]),
                             "n_variables_modelo": int(sizes.n_variables_modelo.iloc[0]),
                             **_scores(cases.observado_N.to_numpy(), cases.predicho_N.to_numpy())})
    summary = pd.DataFrame(summary_rows)
    for (protocol, family), group in summary.groupby(["protocolo", "modelo"]):
        full = float(group[group.conjunto == "Todas"].MAE_N.iloc[0])
        mask = (summary.protocolo == protocol) & (summary.modelo == family)
        summary.loc[mask, "delta_MAE_vs_todas_N"] = summary.loc[mask, "MAE_N"] - full
    for (protocol, case), group in pred.groupby(["protocolo", "Case_n"]):
        if len(group) != len(FAMILIES) * len(SETS) or group.fold.nunique() != 1 \
                or group.observado_N.nunique() != 1:
            raise ValueError(f"Comparación no emparejada: {protocol}, caso {case}.")
    tables = {"metricas_variables_importantes": summary,
              "predicciones_variables_importantes": pred,
              "metricas_folds_variables_importantes": folds}
    table_dir = destination / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(table_dir / f"{name}.csv", index=False, encoding="utf-8")
    log_dir = destination / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"selection": "Retrospectiva exploratoria: cuatro mayores SHAP globales de Ridge E1 sobre estos mismos casos; sin validación independiente de la selección.",
                "selected_features": list(IMPORTANT),
                "source": "reports/feature_importance/tables/shap_global.csv",
                "fixed_parameters": {"Ridge": {"alpha": 1.0, "solver": "svd"},
                                     "ElasticNet": {"alpha": 0.1, "l1_ratio": 0.5,
                                                    "max_iter": 20000, "tol": 0.0001}},
                "hyperparameter_search": False,
                "outer_splits": {protocol: len(outer) for protocol, outer in splits.items()},
                "note": "Todas: diez entradas originales; el filtro de varianza elimina el radio constante. Importantes: cuatro entradas originales."}
    (log_dir / "comparacion_variables_importantes.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return tables


if __name__ == "__main__":
    result = run()
    print(result["metricas_variables_importantes"].to_string(index=False))
