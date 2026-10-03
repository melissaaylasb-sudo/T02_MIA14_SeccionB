"""
Baseline preliminar para regresión.

Implementa la referencia usada en la corrida preliminar de validación anidada.
Importar este módulo no ajusta estimadores; el experimento aprende una mediana
por entrenamiento externo con el protocolo de config/model_config.yml.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    medae: float
    r2: float | None


def build_baseline() -> DummyRegressor:
    """Predice la mediana del target aprendida solo del conjunto de entrenamiento."""
    return DummyRegressor(strategy="median")


def evaluate_regression(y_true, y_pred) -> RegressionMetrics:
    """Evalúa predicciones suministradas; el llamador debe asegurar que sean fuera de muestra.

    Devuelve r2=None con menos de dos observaciones o respuesta constante, para
    no sustituir una métrica indefinida por un resultado aparentemente válido.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.shape != y_pred.shape or not y_true.size:
        raise ValueError("Se requieren dos vectores unidimensionales no vacíos del mismo tamaño.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Las respuestas y predicciones deben ser numéricas y finitas.")
    r2 = None
    if y_true.size >= 2 and np.any(y_true != y_true[0]):
        r2 = float(r2_score(y_true, y_pred, force_finite=False))
    return RegressionMetrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        medae=float(median_absolute_error(y_true, y_pred)),
        r2=r2,
    )


if __name__ == "__main__":
    print(
        "Para ejecutar baseline y modelos en las mismas particiones: "
        "python -m src.experimento --run --fit-final"
    )
