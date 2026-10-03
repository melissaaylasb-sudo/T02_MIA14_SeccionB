"""Particiones externas e internas auditables, sin ajustar ningún modelo."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold


def make_splits(n_rows: int, n_splits: int, strategy: str, seed: int, groups=None) -> list:
    """Devuelve índices relativos a la entrada; mantiene grupos completos."""
    if type(n_splits) is not int or n_splits < 2:
        raise ValueError("Cada nivel de CV requiere al menos dos particiones definidas.")
    if strategy == "group_kfold":
        if groups is None or len(groups) != n_rows or len(np.unique(groups)) < n_splits:
            raise ValueError("No hay suficientes grupos completos para esta partición.")
        splitter = GroupKFold(n_splits=n_splits) # Compatible con scikit-learn >=1.5
        splits = list(splitter.split(np.zeros(n_rows), groups=groups))
    elif strategy == "kfold":
        if groups is not None or n_rows < n_splits:
            raise ValueError("KFold exige observaciones independientes y filas suficientes.")
        splits = list(KFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(n_rows)))
    else:
        raise ValueError("Defina strategy como group_kfold o kfold.")
    for train, test in splits:
        if len(train) < 2 or len(test) < 1 or np.intersect1d(train, test).size:
            raise ValueError("Partición vacía, solapada o con menos de dos filas para entrenar.")
        if groups is not None and set(groups[train]) & set(groups[test]):
            raise ValueError("Un grupo aparece en entrenamiento y evaluación.")
    return splits


def check_training_features(X: pd.DataFrame) -> None:
    """No permite aprender medianas de columnas completamente ausentes."""
    empty = X.columns[X.isna().all()].tolist()
    if empty:
        raise ValueError(f"Predictores sin observaciones en un entrenamiento: {empty}")
    if (X.nunique(dropna=True) <= 1).all():
        raise ValueError("Todos los predictores son constantes en un entrenamiento.")


def nested_plan(X: pd.DataFrame, protocol: dict, groups=None) -> list[dict]:
    """Materializa la CV completa y revisa viabilidad antes de cualquier fit."""
    if protocol.get("reviewed") is not True or not str(protocol.get("rationale") or "").strip():
        raise ValueError("Revise y justifique el protocolo antes de construir particiones.")
    outer = make_splits(
        len(X), protocol["outer_splits"], protocol["strategy"], protocol["seed"], groups
    )
    plan = []
    for fold, (train, test) in enumerate(outer, start=1):
        check_training_features(X.iloc[train])
        inner = make_splits(
            len(train), protocol["inner_splits"], protocol["strategy"],
            protocol["seed"] + fold, None if groups is None else groups[train],
        )
        for inner_train, _ in inner:
            check_training_features(X.iloc[train[inner_train]])
        plan.append({"fold": fold, "train": train, "test": test, "inner": inner})
    return plan
