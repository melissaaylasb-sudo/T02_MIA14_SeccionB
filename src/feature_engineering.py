"""Pipelines geométricos: contrato temporal explícito y ajuste solo en entrenamiento."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .transformaciones_eda import GEOMETRY, geometry_features

TARGET = "First_Failure_Load_N"
MOMENTS = ["I_fibra_bh3_mm4", "I_fibra_hb3_mm4"]
LOG_RATIO = "log_volumen_pivotes_sobre_fibras"
REDUNDANT = ["Fiber_total_length", "Pivot_total_number", "Fiber_total_volume"]


class GeometryContract(TransformerMixin, BaseEstimator):
    """Rechaza respuestas, identificadores, infinitos y geometrías no positivas.

    Se admiten NaN en transform para imputación con estadísticas de entrenamiento.
    Una columna completamente ausente en entrenamiento no permite estimar mediana.
    """
    def _validate(self, X):
        if not isinstance(X, pd.DataFrame) or set(X.columns) != set(GEOMETRY) or X.columns.duplicated().any():
            raise ValueError("Se requieren exclusivamente las columnas geométricas autorizadas; sin target ni identificadores.")
        x = X[GEOMETRY].apply(pd.to_numeric, errors="raise").astype(float)
        a = x.to_numpy()
        if np.isinf(a).any() or (a <= 0).any():
            raise ValueError("La geometría debe ser positiva y finita, o NaN para imputación.")
        for col in ["n_cells_Y", "Pivot_total_number"]:
            v = x[col].dropna().to_numpy()
            if not np.equal(v, np.floor(v)).all():
                raise ValueError("Los conteos geométricos deben ser enteros.")
        return x

    def fit(self, X, y=None):
        x = self._validate(X)
        if x.isna().all().any():
            raise ValueError("Una entrada carece de observaciones en entrenamiento.")
        self.feature_names_in_ = np.array(GEOMETRY, dtype=object)
        self.n_features_in_ = len(GEOMETRY)
        return self

    def transform(self, X):
        check_is_fitted(self)
        return self._validate(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return self.feature_names_in_.copy()


class GeometryRepresentation(TransformerMixin, BaseEstimator):
    """Fórmulas por fila reutilizadas del cuaderno 03; ninguna lee el target."""
    def __init__(self, variant="A"):
        self.variant = variant

    def fit(self, X, y=None):
        if self.variant not in {"A", "B", "C", "D"}:
            raise ValueError("Variante desconocida.")
        self.feature_names_in_ = np.array(GEOMETRY, dtype=object)
        self.n_features_in_ = len(GEOMETRY)
        self.output_names_ = np.array(self._build(X).columns, dtype=object)
        return self

    def _build(self, X):
        x = X[GEOMETRY].copy()
        if self.variant != "A":
            derived = geometry_features(x)
            x[MOMENTS] = derived[MOMENTS]
        if self.variant in {"C", "D"}:
            # Cociente adimensional; logaritmos separados evitan desbordar la razón.
            x[LOG_RATIO] = np.log(x.Pivot_total_volume) - np.log(x.Fiber_total_volume)
        if self.variant == "D":
            x = x.drop(columns=REDUNDANT)
        if not np.isfinite(x.to_numpy()).all():
            raise ValueError("Representación no finita; revisar dominio e imputación.")
        return x

    def transform(self, X):
        check_is_fitted(self)
        return self._build(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return self.output_names_.copy()


def make_pipeline(variant="A", *, alpha=1.0, dummy=False):
    """Homogéneo numérico: ColumnTransformer no es necesario en esta comparación."""
    return Pipeline([
        ("contract", GeometryContract()),
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True).set_output(transform="pandas")),
        ("features", GeometryRepresentation(variant)),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scale", StandardScaler()),
        ("model", DummyRegressor(strategy="mean") if dummy else Ridge(alpha=alpha, solver="svd")),
    ])


def feature_names(pipeline):
    return pipeline.named_steps["features"].get_feature_names_out()[pipeline.named_steps["variance"].get_support()]


def nominal_groups(frame, columns):
    if frame[columns].isna().any().any():
        raise ValueError("No se pueden verificar grupos nominales con claves ausentes.")
    return frame[columns].apply(lambda row: "|".join(format(float(v), ".12g") for v in row), axis=1)
