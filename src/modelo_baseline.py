from __future__ import annotations

"""
Baseline preliminar para regresión.

Este archivo deja implementada la referencia metodológica sin ejecutar
entrenamiento automáticamente. El protocolo definitivo de partición deberá
definirse después de verificar tamaño, réplicas y estructura del dataset.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    medae: float
    r2: float


def build_baseline() -> DummyRegressor:
    """Predice la mediana del target aprendida solo del conjunto de entrenamiento."""
    return DummyRegressor(strategy="median")


def evaluate_regression(y_true, y_pred) -> RegressionMetrics:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return RegressionMetrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        medae=float(median_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
    )


if __name__ == "__main__":
    print(
        "Baseline preparado. Defina primero el protocolo de partición y el nombre "
        "canónico del target antes de ejecutar entrenamiento."
    )
