"""Diagnósticos posteriores a la selección; no prueban causalidad física."""

from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance


def linear_coefficients(pipeline, original_features) -> pd.DataFrame:
    """Coeficientes en la escala del pipeline, sin interpretarlos como fuerzas causales."""
    model = pipeline.named_steps["model"]
    columns = ["feature", "coefficient", "scale"]
    if not hasattr(model, "coef_"):
        return pd.DataFrame(columns=columns)
    features = list(pd.Index(original_features)[pipeline.named_steps["variance"].get_support()])
    return pd.DataFrame({
        "feature": features,
        "coefficient": model.coef_,
        "scale": "entrada estandarizada dentro del entrenamiento; salida en unidad del target",
    })


def heldout_importance(pipeline, X_test, y_test, repeats: int, seed: int) -> pd.DataFrame:
    """Permuta entradas del conjunto externo tras selección; mayor valor = mayor aumento de MAE."""
    result = permutation_importance(
        pipeline, X_test, y_test, scoring="neg_mean_absolute_error",
        n_repeats=repeats, random_state=seed, n_jobs=1,
    )
    return pd.DataFrame({
        "feature": X_test.columns,
        "mae_increase_mean": result.importances_mean,
        "mae_increase_std": result.importances_std,
    })
