"""Ajuste interno por MAE; la evaluación externa no decide hiperparámetros."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV


def tune_candidates(X_train, y_train, candidates: dict, inner_splits: list, n_jobs: int = 1):
    """Ajusta SOLO el entrenamiento recibido y selecciona familia usando CV interna.

    Esta función sí entrena si se invoca. Los imports y constructores no lo hacen.
    La lista explícita de folds debe estar referida a X_train, nunca al dataset global.
    """
    searches, histories, notices = {}, [], []
    for name, (pipeline, grid) in candidates.items():
        search = GridSearchCV(
            clone(pipeline), grid, cv=inner_splits, scoring="neg_mean_absolute_error",
            refit=True, n_jobs=n_jobs, error_score="raise", return_train_score=False,
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            search.fit(X_train, y_train)
        if not np.isfinite(search.best_score_):
            raise ValueError(f"La búsqueda interna de {name} no produjo un criterio finito.")
        searches[name] = search
        history = pd.DataFrame(search.cv_results_)
        history.insert(0, "family", name)
        histories.append(history)
        notices.extend(f"{name}: {w.category.__name__}: {w.message}" for w in captured)
    # Los empates se resuelven por el orden predeclarado del YAML, no por el fold externo.
    selected = max(searches, key=lambda name: searches[name].best_score_)
    return searches, selected, pd.concat(histories, ignore_index=True), notices
