"""Estadística exploratoria reproducible; no modifica datos ni ajusta modelos de producción."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def safe_correlation(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 1e-12 else float("nan")


def ranked_residuals(values, groups=None):
    """Rangos globales residualizados contra indicadores categóricos de familia."""
    ranks = stats.rankdata(np.asarray(values, dtype=float))
    if groups is None:
        return ranks - ranks.mean()
    frame = pd.DataFrame({"r": ranks, "g": np.asarray(groups)})
    return (frame.r - frame.groupby("g").r.transform("mean")).to_numpy()


def rank_association(x, y, groups=None) -> float:
    return safe_correlation(ranked_residuals(x, groups), ranked_residuals(y, groups))


def permutation_rank_test(x, y, groups=None, *, resamples=4999, seed=42):
    """Bilateral; con grupos permuta dentro de familia. Corrección Monte Carlo +1."""
    a, b = ranked_residuals(x, groups), ranked_residuals(y, groups)
    observed = safe_correlation(a, b)
    if not np.isfinite(observed):
        return observed, float("nan")
    rng = np.random.default_rng(seed)
    indexes = [np.arange(len(a))] if groups is None else [
        np.flatnonzero(np.asarray(groups) == g) for g in np.unique(groups)
    ]
    simulated = np.tile(b, (resamples, 1))
    for index in indexes:
        order = np.argsort(rng.random((resamples, len(index))), axis=1)
        simulated[:, index] = b[index][order]
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    correlations = simulated @ a / denominator
    p = (1 + np.sum(np.abs(correlations) >= abs(observed) - 1e-12)) / (resamples + 1)
    return observed, float(p)


def bootstrap_rank_interval(x, y, groups=None, *, resamples=1499, seed=42):
    """Intervalo percentil; estratificado si groups se especifica."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    group_array = None if groups is None else np.asarray(groups)
    blocks = [np.arange(len(x))] if groups is None else [
        np.flatnonzero(group_array == g) for g in np.unique(group_array)
    ]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        index = np.concatenate([rng.choice(block, size=len(block), replace=True) for block in blocks])
        value = rank_association(x[index], y[index], None if groups is None else group_array[index])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return tuple(np.quantile(values, [.025, .975]).astype(float))


def adjust_pvalues(values, method="bh"):
    """FDR solo sobre p finitos; conserva valores no identificables como NaN."""
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    adjusted = np.full(len(values), np.nan)
    if mask.any():
        adjusted[mask] = stats.false_discovery_control(values[mask], method=method)
    return adjusted


def outlier_flags(values):
    values = pd.Series(values, dtype=float)
    q1, q3 = values.quantile([.25, .75])
    iqr = q3 - q1
    median = values.median()
    mad = np.median(np.abs(values - median))
    iqr_flag = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
    robust_z = .67448975 * (values - median) / mad if mad > 1e-12 else pd.Series(np.nan, index=values.index)
    return iqr_flag, robust_z.abs() > 3.5, robust_z


def eta_squared(values, groups):
    frame = pd.DataFrame({"y": values, "g": groups})
    total = ((frame.y - frame.y.mean()) ** 2).sum()
    between = sum(len(s) * (s.mean() - frame.y.mean()) ** 2 for _, s in frame.groupby("g").y)
    return float(between / total) if total else float("nan")


def cliffs_delta(x, y):
    """Dominancia de x frente a y; no presupone normalidad."""
    return float(np.sign(np.asarray(x)[:, None] - np.asarray(y)[None, :]).mean())


def distance_correlation(x, y):
    """Correlación de distancias empírica sesgada, únicamente descriptiva."""
    def centered(v):
        d = np.abs(np.asarray(v)[:, None] - np.asarray(v)[None, :])
        return d - d.mean(axis=0) - d.mean(axis=1)[:, None] + d.mean()
    a, b = centered(x), centered(y)
    den = np.sqrt(np.mean(a * a) * np.mean(b * b))
    return float(np.sqrt(max(0, np.mean(a * b)) / den)) if den > 1e-14 else float("nan")


def ols_influence(design, response):
    """OLS descriptivo de diseño predeclarado; residuos internos studentizados y Cook."""
    a, y = np.asarray(design, dtype=float), np.asarray(response, dtype=float)
    rank = np.linalg.matrix_rank(a)
    if rank != a.shape[1] or len(y) <= rank:
        raise ValueError("El diseño OLS debe tener rango completo y grados residuales positivos.")
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    fitted = a @ coef
    residual = y - fitted
    leverage = np.sum(a * np.linalg.pinv(a).T, axis=1)
    mse = (residual @ residual) / (len(y) - rank)
    if mse <= 0 or np.any(leverage >= 1 - 1e-10):
        raise ValueError("No se pueden calcular diagnósticos de influencia para este diseño.")
    student = residual / np.sqrt(mse * (1 - leverage))
    cook = residual ** 2 * leverage / (rank * mse * (1 - leverage) ** 2)
    return pd.DataFrame({"ajustado_N": fitted, "residuo_N": residual, "leverage": leverage,
                         "residuo_studentizado": student, "Cook": cook}), coef
