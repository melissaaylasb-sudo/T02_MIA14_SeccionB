"""Construcción sin ajuste de candidatos y preprocesamiento dentro de cada fold."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def build_pipeline(name: str, seed: int) -> Pipeline:
    """Crea estimadores nuevos: imputa, elimina constantes y escala solo al hacer fit."""
    models = {
        "ridge": Ridge(),
        "elastic_net": ElasticNet(max_iter=20000, random_state=seed),
        "support_vector": SVR(kernel="rbf"),
        "decision_tree": DecisionTreeRegressor(random_state=seed),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1),
        "gaussian_process": GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3)),
            normalize_y=True, random_state=seed,
        ),
        "neural_network": MLPRegressor(
            hidden_layer_sizes=(16,), solver="lbfgs", max_iter=2000, random_state=seed,
        ),
    }
    if name not in models:
        raise ValueError(f"Familia desconocida: {name}")
    scale = "passthrough" if name in {"decision_tree", "random_forest"} else StandardScaler()
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold(threshold=0.0)),
        ("scale", scale),
        ("model", models[name]),
    ])


def candidate_specs(config: dict) -> dict:
    """Valida nombres de parámetros y devuelve pipelines/rejillas sin entrenar."""
    candidates = {}
    for name, spec in config["models"].items():
        if not isinstance(spec, dict) or type(spec.get("enabled")) is not bool:
            raise ValueError(f"Declare enabled como booleano para {name}.")
        if not spec["enabled"]:
            continue
        pipeline = build_pipeline(name, config["protocol"]["seed"])
        grid = spec.get("grid")
        if not isinstance(grid, dict) or any(not key.startswith("model__") for key in grid):
            raise ValueError("Las rejillas solo deben cambiar hiperparámetros model__*.")
        for params in ParameterGrid(grid):
            pipeline.set_params(**params)
        candidates[name] = (build_pipeline(name, config["protocol"]["seed"]), grid)
    if not candidates:
        raise ValueError("Debe habilitar al menos una familia candidata.")
    return candidates
