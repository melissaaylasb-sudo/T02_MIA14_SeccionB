"""Regresores y reglas de selección congeladas para el benchmark de primera falla."""
from __future__ import annotations

from itertools import product
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from .feature_engineering import GEOMETRY, GeometryContract, make_pipeline

Params = dict[str, Any]
INITIAL = ("SVR_RBF","XGBoost","CatBoost","Lineal_multiple","KNN","Arbol_decision")
REFS = ("Dummy_mediana","Media_por_nivel","Ridge_historico")
ADAPT = ("Tuning_rango_1","Tuning_rango_2","Selector_tuning")
UNITS = dict(zip(GEOMETRY,["conteo","mm","mm","mm","mm³","mm","mm","conteo","mm³","mm³"]))


def config(root: Path) -> Params:
    """Carga la configuración y rechaza cambios en el contrato de datos y familias."""
    cfg=json.loads((root/"config/benchmark_regresion.json").read_text(encoding="utf-8"))
    if (tuple(cfg["familias"])!=INITIAL or tuple(cfg["referencias"])!=REFS or cfg["target"]!="First_Failure_Load_N"
            or cfg["unidad"]!="N" or cfg["feature_set"]!="A" or set(cfg["espacios"])!=set(INITIAL)-{"Lineal_multiple"}):
        raise ValueError("La configuración no cumple el contrato de seis regresores para F1.")
    return cfg


class MeanByLevel(RegressorMixin, BaseEstimator):
    """Media Y de train; si falta el nivel usa la media global del mismo train."""
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.global_mean_=float(np.mean(y))
        self.means_=pd.Series(np.asarray(y)).groupby(X.n_cells_Y.to_numpy()).mean().to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self,"means_")
        return X.n_cells_Y.map(self.means_).fillna(self.global_mean_).to_numpy(float)


class FullRankBasis(TransformerMixin, BaseEstimator):
    """Base no supervisada de la matriz de diseño centrada, aprendida solo en fit."""
    def __init__(self, rtol: float = 1e-10):
        self.rtol=rtol

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        a=np.asarray(X,dtype=float);a=a-a.mean(axis=0)
        self.singular_values_=np.linalg.svd(a,compute_uv=False)
        self.tol_=float(self.rtol*self.singular_values_[0]); selected=[]
        for i in range(a.shape[1]):
            if np.linalg.matrix_rank(a[:,selected+[i]],tol=self.tol_)>len(selected):selected.append(i)
        if not selected:raise ValueError("Diseño OLS sin columnas informativas.")
        self.indices_=np.asarray(selected,dtype=int);self.n_features_in_=a.shape[1]
        self.rank_=int(np.linalg.matrix_rank(a,tol=self.tol_))
        if self.rank_!=len(selected):raise ValueError("Base secuencial numéricamente ambigua; revisar la tolerancia.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self,"indices_")
        return np.asarray(X)[:,self.indices_]


def pipeline(family: str, params: Params, cfg: Params) -> Pipeline:
    """Misma geometría A; transformaciones estadísticas ajustadas en cada train."""
    if family in ("Dummy_mediana","Media_por_nivel"):
        model=DummyRegressor(**params) if family=="Dummy_mediana" else MeanByLevel()
        return Pipeline([("contract",GeometryContract()),("model",model)])
    base=make_pipeline("A")
    if family=="Ridge_historico":return base.set_params(**{"model__"+k:v for k,v in params.items()})
    constructors={"SVR_RBF":SVR,"XGBoost":XGBRegressor,"CatBoost":CatBoostRegressor,
                  "Lineal_multiple":LinearRegression,"KNN":KNeighborsRegressor,"Arbol_decision":DecisionTreeRegressor}
    steps=base.steps[:-1]
    if family in ("XGBoost","CatBoost","Arbol_decision"):steps=steps[:-1]
    if family=="Lineal_multiple":steps.append(("basis",FullRankBasis(cfg["ols"]["tolerancia_svd_relativa"])))
    return Pipeline([*steps,("model",constructors[family](**params))])


def freeze_candidates(cfg: Params) -> dict[str,list[Params]]:
    """Muestrea una vez cada pequeño espacio, sin y, folds ni resultados."""
    result={}
    for i,family in enumerate(INITIAL):
        if family not in cfg["espacios"]:continue
        fixed=cfg["parametros_iniciales"][family];space=cfg["espacios"][family];keys=list(space)
        pool=[dict(fixed,**dict(zip(keys,row))) for row in product(*(space[k] for k in keys))]
        pool=[p for p in pool if p!=fixed];rng=np.random.default_rng(cfg["semilla"]+i)
        inds=rng.choice(len(pool),size=cfg["presupuesto_por_candidata"]-1,replace=False)
        result[family]=[fixed.copy(),*[pool[j] for j in inds]]
    return result


def inner_metrics(y: np.ndarray, pred: np.ndarray, groups: np.ndarray) -> tuple[float,float]:
    """MAE pooled por caso y MAD no escalada de las MAE por grupo."""
    errors=np.abs(np.asarray(y)-np.asarray(pred));means=pd.Series(errors).groupby(np.asarray(groups)).mean()
    return float(errors.mean()),float(np.median(np.abs(means-np.median(means))))


def rank_families(scores: dict[str,tuple[float,float]], cfg: Params) -> list[str]:
    """Ordena por banda de MAE, estabilidad interna y orden congelado."""
    remaining=dict(scores);order=[]
    while remaining:
        limit=min(v[0] for v in remaining.values())+cfg["banda_cribado_N"]
        eligible=[f for f,v in remaining.items() if v[0]<=limit+1e-12]
        chosen=min(eligible,key=lambda f:(remaining[f][1],remaining[f][0],cfg["orden_complejidad"].index(f),f))
        order.append(chosen);del remaining[chosen]
    return order


def best_index(losses: list[tuple[float,float]]) -> int:
    """Menor MAE; empate numérico por MAD y orden del plan."""
    minimum=min(x[0] for x in losses)
    return min((i for i,v in enumerate(losses) if v[0]<=minimum+1e-10),
               key=lambda i:(losses[i][1],i))
