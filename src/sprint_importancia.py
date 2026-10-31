"""Comparación emparejada A/B, ajuste anidado e interpretabilidad fuera de muestra.

Ejecutar desde la raíz: ``python -m src.sprint_importancia``. El snapshot y las
particiones se validan con las huellas publicadas en el sprint anterior.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

from .ajuste_modelos import ROOT, inner_splits, load_frozen_data, load_outer_splits
from .feature_engineering import MOMENTS, TARGET, feature_names, make_pipeline
from .ingesta import sha256_file
from .transformaciones_eda import GEOMETRY, geometry_features

OUT = Path("reports/feature_importance")
MODEL_CONFIG = json.loads((ROOT / "config/ajuste_modelos.json").read_text(encoding="utf-8"))
ALPHAS = tuple(MODEL_CONFIG["modelos"]["Ridge"]["grid"]["model__alpha"])
ELASTIC_GRID = MODEL_CONFIG["modelos"]["ElasticNet"]["grid"]
EXPERIMENTS = {
    "E0": ("A", False, "Dummy"),
    "E1": ("A", False, "Ridge"),
    "E2": ("B", False, "Ridge"),
    "E3": ("B", True, "Ridge"),
    "E4": ("A", True, "Ridge"),
    "S1": ("A", False, "ElasticNet"),
    "S2": ("B", False, "ElasticNet"),
    "S3": ("B", True, "ElasticNet"),
    "S4": ("A", True, "ElasticNet"),
}
UNITS = {"n_cells_Y": "conteo", "Fiber_height": "mm", "Fiber_base": "mm",
         "Fiber_total_length": "mm", "Fiber_total_volume": "mm³", "Pivot_height": "mm",
         "Pivot_radius": "mm", "Pivot_total_number": "conteo", "Pivot_total_volume": "mm³",
         "Sample_total_volume": "mm³", **{name: "mm⁴" for name in MOMENTS}}
FEATURE_FAMILIES = {
    "arquitectura": ["n_cells_Y", "Fiber_total_length", "Pivot_total_number"],
    "seccion_fibra": ["Fiber_height", "Fiber_base", *MOMENTS],
    "pivotes": ["Pivot_height", "Pivot_total_volume"],
    "volumen_global_y_fibras": ["Sample_total_volume", "Fiber_total_volume"],
}


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calcula métricas en N a partir de pares fuera de entrenamiento."""
    y = np.asarray(observed, dtype=float).reshape(-1)
    p = np.asarray(predicted, dtype=float).reshape(-1)
    if y.size == 0 or y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Observaciones y predicciones deben ser finitas y emparejadas.")
    residual = y - p
    return {"MAE_N": float(mean_absolute_error(y, p)),
            "RMSE_N": float(np.sqrt(mean_squared_error(y, p))),
            "MedAE_N": float(median_absolute_error(y, p)),
            "R2_OOF": float(r2_score(y, p)) if len(y) > 1 else np.nan,
            "sesgo_residuo_N": float(residual.mean()),
            "sd_residuo_N": float(residual.std(ddof=1)) if len(y) > 1 else np.nan}


def validate_shap_reconstruction(base: np.ndarray, values: np.ndarray,
                                 predictions: np.ndarray, *, atol: float = 1e-6) -> None:
    """Exige aditividad de SHAP en la escala física de la carga [N]."""
    actual = np.asarray(base, dtype=float).reshape(-1) + np.asarray(values, dtype=float).sum(axis=1)
    expected = np.asarray(predictions, dtype=float).reshape(-1)
    if actual.shape != expected.shape or not np.allclose(actual, expected, atol=atol, rtol=1e-8):
        raise ValueError("Los valores SHAP no reconstruyen las predicciones en N.")


def compare_paired(predictions: pd.DataFrame, protocol: str, left: str, right: str) -> dict:
    """Contrasta errores absolutos de los mismos casos: derecha menos izquierda."""
    subset = predictions[predictions.protocolo == protocol]
    a = subset[subset.experimento == left].set_index("Case_n")
    b = subset[subset.experimento == right].set_index("Case_n")
    if set(a.index) != set(b.index) or a.index.has_duplicates or b.index.has_duplicates:
        raise ValueError("La comparación requiere exactamente los mismos casos.")
    a, b = a.sort_index(), b.sort_index()
    if not np.allclose(a.observado_N, b.observado_N):
        raise ValueError("La respuesta observada cambió entre experimentos.")
    delta = b.error_absoluto_N.to_numpy() - a.error_absoluto_N.to_numpy()
    return {"protocolo": protocol, "referencia": left, "intervencion": right,
            "n_casos": len(delta), "delta_MAE_N": float(delta.mean()),
            "mediana_delta_error_absoluto_N": float(np.median(delta)),
            "fraccion_casos_mejora": float((delta < 0).mean()),
            "sd_delta_error_absoluto_N": float(delta.std(ddof=1))}


def _explain(model, train_x: pd.DataFrame, test_x: pd.DataFrame,
             predictions: np.ndarray, *, train_cases: list[int]) -> tuple[list[dict], dict]:
    """LinearExplainer interventional con background transformado del train externo."""
    import shap

    transformed_train = np.asarray(model[:-1].transform(train_x), dtype=float)
    transformed_test = np.asarray(model[:-1].transform(test_x), dtype=float)
    names = feature_names(model).tolist()
    if transformed_train.shape[1] != len(names) or transformed_test.shape[1] != len(names):
        raise ValueError("Nombres y orden de características transformadas no coinciden.")
    explainer = shap.LinearExplainer(model.named_steps["model"],
                                    shap.maskers.Independent(transformed_train,
                                                              max_samples=len(transformed_train)))
    explanation = explainer(transformed_test)
    values = np.asarray(explanation.values, dtype=float)
    base = np.broadcast_to(np.asarray(explanation.base_values, dtype=float), (len(test_x),))
    validate_shap_reconstruction(base, values, predictions)
    physical = model.named_steps["features"].transform(
        model.named_steps["imputer"].transform(model.named_steps["contract"].transform(test_x)))
    background_json = json.dumps(train_cases)
    rows = []
    for i, case in enumerate(test_x.index):
        for j, name in enumerate(names):
            rows.append({"Case_n": int(case), "feature": name, "feature_unit": UNITS[name],
                         "feature_value": float(physical.iloc[i][name]),
                         "feature_value_scaled": float(transformed_test[i, j]),
                         "shap_N": float(values[i, j]), "base_N": float(base[i]),
                         "predicho_N": float(predictions[i]),
                         "background_train_cases_json": background_json})
    audit = {"explainer": "shap.LinearExplainer", "masker": "shap.maskers.Independent",
             "output_unit": "N", "background_cases": train_cases,
             "max_reconstruction_error_N": float(np.max(np.abs(base + values.sum(axis=1) - predictions)))}
    return rows, audit


def _summarize_shap(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alinea atributos por nombre y separa resumen OOF de diagnóstico por fold."""
    if rows.empty:
        raise ValueError("Se requieren explicaciones fuera de muestra.")
    stability = (rows.assign(abs_shap_N=rows.shap_N.abs())
                 .groupby(["protocolo", "experimento", "modelo", "fold", "feature"], as_index=False)
                 .agg(mean_abs_shap_N=("abs_shap_N", "mean"),
                      mean_signed_shap_N=("shap_N", "mean"), n_casos=("Case_n", "nunique")))
    stability["rank_fold"] = stability.groupby(["protocolo", "experimento", "fold"])[
        "mean_abs_shap_N"].rank(method="min", ascending=False).astype(int)
    global_rows = (rows.assign(abs_shap_N=rows.shap_N.abs())
                   .groupby(["protocolo", "experimento", "modelo", "feature"], as_index=False)
                   .agg(mean_abs_shap_N=("abs_shap_N", "mean"),
                        median_abs_shap_N=("abs_shap_N", "median"),
                        mean_signed_shap_N=("shap_N", "mean"), n_casos=("Case_n", "nunique")))
    folds = stability.groupby(["protocolo", "experimento", "feature"], as_index=False).agg(
        sd_fold_mean_abs_shap_N=("mean_abs_shap_N", "std"),
        median_rank_fold=("rank_fold", "median"),
        top3_fold_fraction=("rank_fold", lambda s: float((s <= 3).mean())),
        n_folds=("fold", "nunique"))
    global_rows = global_rows.merge(folds, on=["protocolo", "experimento", "feature"], validate="one_to_one")
    global_rows["rank_global"] = global_rows.groupby(["protocolo", "experimento"])[
        "mean_abs_shap_N"].rank(method="min", ascending=False).astype(int)
    return global_rows, stability


def aggregate_shap_families(rows: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Suma contribuciones firmadas de fuentes disjuntas; no estima SHAP grupal formal."""
    mapping = {name: family for family, names in FEATURE_FAMILIES.items() for name in names}
    if set(rows.feature) - set(mapping):
        raise ValueError("Hay características SHAP sin familia geométrica.")
    x = rows.copy()
    x["familia_geometrica"] = x.feature.map(mapping)
    grouped = x.groupby(["protocolo", "experimento", "modelo", "fold", "Case_n",
                         "familia_geometrica"], as_index=False).agg(
        suma_shap_N=("shap_N", "sum"), n_features=("feature", "nunique"))
    grouped["abs_suma_shap_N"] = grouped.suma_shap_N.abs()
    means = grouped.groupby(["protocolo", "experimento", "familia_geometrica"],
                            as_index=False).agg(mean_abs_sum_shap_N=("abs_suma_shap_N", "mean"))
    grouped = grouped.merge(means, on=["protocolo", "experimento", "familia_geometrica"],
                            validate="many_to_one")
    family_by_case = frame.set_index("Case_n").n_cells_Y.astype(int)
    grouped["familia_celdas"] = grouped.Case_n.map(family_by_case)
    if grouped.familia_celdas.isna().any():
        raise ValueError("Hay casos SHAP sin familia experimental.")
    grouped["metodo"] = "suma descriptiva de SHAP por fuentes; no SHAP grupal formal"
    return grouped


def _permutation_oof(frame: pd.DataFrame, fitted: list[tuple[np.ndarray, object]],
                     protocol: str, *, repetitions: int = 12, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Permuta entradas geométricas en el conjunto OOF emparejado.

    Muchos folds contienen un solo caso; por ello la permutación dentro de cada
    fold sería nula. Se permuta sobre los 26 casos OOF, conservando juntos los
    grupos predeclarados de fuentes. Puede crear combinaciones fuera del soporte.
    """
    X = frame[GEOMETRY].copy()
    y = frame[TARGET].to_numpy(dtype=float)
    original = np.empty(len(frame), dtype=float)
    for test, model in fitted:
        original[test] = model.predict(X.iloc[test])
    baseline_mae = float(mean_absolute_error(y, original))
    groups = {name: [name] for name in GEOMETRY}
    groups["seccion_fibra_fuentes"] = ["Fiber_height", "Fiber_base"]
    groups["volumenes_componentes"] = ["Fiber_total_volume", "Pivot_total_volume"]
    rng = np.random.default_rng(seed)
    rows = []
    for name, columns in groups.items():
        for repeat in range(repetitions):
            permutation = rng.permutation(len(frame))
            predicted = np.empty(len(frame), dtype=float)
            for test, model in fitted:
                x_test = X.iloc[test].copy()
                for column in columns:
                    x_test[column] = X.iloc[permutation[test]][column].to_numpy()
                predicted[test] = model.predict(x_test)
            permuted_mae = float(mean_absolute_error(y, predicted))
            rows.append({"protocolo": protocol, "experimento": "E3", "feature_or_group": name,
                         "source_columns_json": json.dumps(columns), "repeticion": repeat + 1,
                         "MAE_original_N": baseline_mae, "MAE_permutado_N": permuted_mae,
                         "importancia_N": permuted_mae - baseline_mae})
    replicates = pd.DataFrame(rows)
    summary = replicates.groupby(["protocolo", "experimento", "feature_or_group", "source_columns_json"],
                                 as_index=False).agg(
        MAE_original_N=("MAE_original_N", "first"),
        importancia_media_N=("importancia_N", "mean"),
        importancia_sd_N=("importancia_N", "std"),
        importancia_min_N=("importancia_N", "min"),
        importancia_max_N=("importancia_N", "max"), n_repeticiones=("repeticion", "count"))
    return summary, replicates


def _correlations(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame[GEOMETRY].copy()
    x[MOMENTS] = geometry_features(x)[MOMENTS]
    x = x.loc[:, x.nunique() > 1]
    rows = []
    for i, left in enumerate(x.columns):
        for right in x.columns[i + 1:]:
            rho, _ = spearmanr(x[left], x[right])
            rows.append({"feature_1": left, "feature_2": right,
                         "pearson_r": float(x[left].corr(x[right])),
                         "spearman_rho": float(rho), "n_casos": len(x)})
    return pd.DataFrame(rows)


def _save_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def run(root: Path | str = ROOT) -> dict:
    """Ejecuta Ridge E0–E4 y ElasticNet S1–S4 con las mismas particiones."""
    root = Path(root).resolve()
    frame = load_frozen_data(root)
    splits, groups, historical = load_outer_splits(frame, root)
    if not {"E0", "E1", "E2", "E3", "E4"}.issubset(EXPERIMENTS):
        raise ValueError("Contrato experimental E0–E4 incompleto.")
    X = frame[GEOMETRY].copy()
    X.index = frame.Case_n.astype(int)
    y = frame[TARGET].astype(float)
    y.index = X.index
    prediction_rows, fold_rows, search_rows, shap_rows, audits, coefficient_rows = [], [], [], [], [], []
    fitted_e3 = {protocol: [] for protocol in splits}
    for protocol, outer in splits.items():
        for fold, (train, test) in enumerate(outer, 1):
            train_cases = X.iloc[train].index.astype(int).tolist()
            test_cases = X.iloc[test].index.astype(int).tolist()
            for experiment, (variant, tuned, model_name) in EXPERIMENTS.items():
                started = perf_counter()
                dummy = model_name == "Dummy"
                model = make_pipeline(variant, dummy=dummy)
                if model_name == "Dummy":
                    model.set_params(model__strategy="median")
                    selected_params = {"model__strategy": "median"}
                elif model_name == "Ridge":
                    selected_params = {"model__alpha": 1.0, "model__solver": "svd"}
                elif model_name == "ElasticNet":
                    model.set_params(model=ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000,
                                                      tol=0.0001, selection="cyclic", random_state=42))
                    selected_params = {"model__alpha": 0.1, "model__l1_ratio": 0.5,
                                       "model__max_iter": 20000, "model__tol": 0.0001,
                                       "model__selection": "cyclic"}
                    if tuned:
                        model.set_params(model__max_iter=100000)
                else:
                    raise ValueError(f"Familia de modelo no reconocida: {model_name}")
                if tuned:
                    cv = inner_splits(frame, train, protocol)
                    grid = {"model__alpha": list(ALPHAS)} if model_name == "Ridge" else ELASTIC_GRID
                    search = GridSearchCV(model, grid, cv=cv,
                                          scoring="neg_mean_absolute_error", refit=True,
                                          n_jobs=1, error_score="raise", return_train_score=False)
                    search.fit(X.iloc[train], y.iloc[train])
                    model = search.best_estimator_
                    selected_params = {key: float(value) for key, value in search.best_params_.items()}
                    selected_params.update({"model__solver": "svd"} if model_name == "Ridge"
                                           else {"model__max_iter": 100000, "model__tol": 0.0001,
                                                 "model__selection": "cyclic"})
                    for candidate, params in enumerate(search.cv_results_["params"]):
                        search_rows.append({"protocolo": protocol, "experimento": experiment,
                                            "fold": fold, "candidate_id": candidate,
                                            "params_json": json.dumps(params, sort_keys=True),
                                            "mean_inner_MAE_N": -float(search.cv_results_["mean_test_score"][candidate]),
                                            "std_inner_MAE_N": float(search.cv_results_["std_test_score"][candidate]),
                                            "rank": int(search.cv_results_["rank_test_score"][candidate]),
                                            "selected": candidate == search.best_index_,
                                            "inner_folds": len(cv)})
                else:
                    model.fit(X.iloc[train], y.iloc[train])
                predictions = np.asarray(model.predict(X.iloc[test]), dtype=float)
                if not np.isfinite(predictions).all():
                    raise ValueError("Predicción externa no finita.")
                elapsed = perf_counter() - started
                audit = {"protocolo": protocol, "experimento": experiment, "modelo": model_name,
                         "fold": fold, "train_cases": train_cases, "test_cases": test_cases,
                         "background_cases": train_cases if not dummy else [],
                         "feature_names": feature_names(model).tolist(), "selected_params": selected_params,
                         "imputer_median": model.named_steps["imputer"].statistics_.tolist(),
                         "scaler_mean": model.named_steps["scale"].mean_.tolist(),
                         "fit_predict_time_s": elapsed}
                if not dummy:
                    estimator = model.named_steps["model"]
                    names = feature_names(model).tolist()
                    if len(names) != len(estimator.coef_):
                        raise ValueError("Coeficientes y nombres transformados no coinciden.")
                    for name, coefficient in zip(names, estimator.coef_):
                        coefficient_rows.append({"protocolo": protocol, "experimento": experiment,
                                                 "modelo": model_name, "fold": fold, "feature": name,
                                                 "coeficiente_por_z_N": float(coefficient),
                                                 "intercept_N": float(estimator.intercept_),
                                                 "alpha_selected": float(estimator.alpha),
                                                 "l1_ratio_selected": (float(estimator.l1_ratio)
                                                                       if model_name == "ElasticNet" else np.nan)})
                    explained, explanation_audit = _explain(model, X.iloc[train], X.iloc[test], predictions,
                                                             train_cases=train_cases)
                    shap_rows.extend({"protocolo": protocol, "experimento": experiment, "modelo": model_name,
                                      "fold": fold, **row} for row in explained)
                    audit.update(explanation_audit)
                audits.append(audit)
                fold_metric = metrics(y.iloc[test].to_numpy(), predictions)
                fold_rows.append({"protocolo": protocol, "experimento": experiment,
                                  "modelo": model_name, "feature_set": variant,
                                  "tuning": tuned, "fold": fold, "n_train": len(train), "n_test": len(test),
                                  "n_features": len(feature_names(model)), "tiempo_s": elapsed, **fold_metric})
                for pos, estimate in zip(test, predictions):
                    observed = float(frame.iloc[pos][TARGET])
                    prediction_rows.append({"protocolo": protocol, "experimento": experiment,
                                            "modelo": model_name,
                                            "feature_set": variant, "tuning": tuned, "fold": fold,
                                            "Case_n": int(frame.iloc[pos].Case_n),
                                            "familia": int(frame.iloc[pos].n_cells_Y),
                                            "grupo": str(groups["configuracion"].iloc[pos]),
                                            "observado_N": observed, "predicho_N": float(estimate),
                                            "residuo_N": observed - float(estimate),
                                            "error_absoluto_N": abs(observed - float(estimate))})
                if experiment == "E3":
                    fitted_e3[protocol].append((test, model))
    predictions = pd.DataFrame(prediction_rows)
    folds = pd.DataFrame(fold_rows)
    searches = pd.DataFrame(search_rows)
    shap_frame = pd.DataFrame(shap_rows)
    summaries = []
    for (protocol, experiment), subset in predictions.groupby(["protocolo", "experimento"], sort=False):
        if len(subset) != len(frame) or subset.Case_n.duplicated().any():
            raise ValueError("Cada experimento debe predecir cada caso exactamente una vez.")
        part_folds = folds[(folds.protocolo == protocol) & (folds.experimento == experiment)]
        summaries.append({"protocolo": protocol, "experimento": experiment,
                          "modelo": subset.modelo.iloc[0], "feature_set": subset.feature_set.iloc[0],
                          "tuning": bool(subset.tuning.iloc[0]), "n_casos": len(subset),
                          "n_folds": len(part_folds), "tiempo_s": float(part_folds.tiempo_s.sum()),
                          "MAE_fold_media_N": float(part_folds.MAE_N.mean()),
                          "MAE_fold_sd_N": float(part_folds.MAE_N.std(ddof=1)),
                          **metrics(subset.observado_N.to_numpy(), subset.predicho_N.to_numpy())})
    summary = pd.DataFrame(summaries)
    for protocol in splits:
        ref = float(summary[(summary.protocolo == protocol) & (summary.experimento == "E0")].MAE_N.iloc[0])
        summary.loc[summary.protocolo == protocol, "delta_MAE_vs_E0_N"] = (
            summary.loc[summary.protocolo == protocol, "MAE_N"] - ref)
    paired = pd.DataFrame([compare_paired(predictions, protocol, a, b)
                           for protocol in splits
                           for a, b in [("E1", "E2"), ("E2", "E3"), ("E1", "E4"), ("E4", "E3"),
                                        ("S1", "S2"), ("S2", "S3"), ("S1", "S4"), ("S4", "S3")]])
    ablations = pd.DataFrame([compare_paired(predictions, protocol, a, b)
                              for protocol in splits
                              for a, b in [("E2", "E1"), ("E3", "E4"), ("S2", "S1"), ("S3", "S4")]])
    ablations["ablacion"] = "Retirar la familia de momentos de sección de B: evaluar A"
    shap_global, shap_stability = _summarize_shap(shap_frame)
    perm_summaries, perm_replicates = [], []
    for protocol in splits:
        ps, pr = _permutation_oof(frame, fitted_e3[protocol], protocol)
        perm_summaries.append(ps)
        perm_replicates.append(pr)
    tables = {
        "metricas_resumen": summary, "metricas_folds": folds, "predicciones_oof": predictions,
        "coeficientes_folds": pd.DataFrame(coefficient_rows),
        "comparacion_pareada": paired, "busquedas": searches, "shap_oof": shap_frame,
        "shap_global": shap_global, "shap_estabilidad": shap_stability,
        "shap_familias": aggregate_shap_families(shap_frame, frame),
        "permutacion": pd.concat(perm_summaries, ignore_index=True),
        "permutacion_replicas": pd.concat(perm_replicates, ignore_index=True),
        "ablaciones": ablations, "correlaciones": _correlations(frame),
    }
    output = root / OUT
    for name, table in tables.items():
        _save_csv(output / "tables" / f"{name}.csv", table)
    logdir = output / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    (logdir / "auditoria_folds.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": "data/processed/preliminary/model_table.csv",
                "dataset_sha256": sha256_file(root / "data/processed/preliminary/model_table.csv"),
                "particiones": "reports/feature_engineering/particiones.json",
                "particiones_sha256": sha256_file(root / "reports/feature_engineering/particiones.json"),
                "n_casos": len(frame), "target": TARGET, "target_unit": "N", "seed": 42,
                "feature_set_best_fe": "B (A + dos momentos de sección)",
                "feature_set_selection": "retrospectiva exploratoria; no validación independiente",
                "model_families": ["Ridge", "ElasticNet"],
                "fixed_parameters": {"Ridge": {"alpha": 1.0, "solver": "svd"},
                                     "ElasticNet": {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 20000}},
                "search_grids": {"Ridge": {"model__alpha": list(ALPHAS)}, "ElasticNet": ELASTIC_GRID},
                "inner_selection": "neg_mean_absolute_error",
                "shap_explainer": "LinearExplainer con masker Independent y background train por fold",
                "shap_output": "carga de primera falla [N]",
                "shap_family_aggregation": "suma descriptiva firmada de fuentes disjuntas; no SHAP grupal formal",
                "permutation": "12 permutaciones OOF globales por fuente; pueden romper dependencia",
                "ablation": "familia de dos momentos, A frente a B; fixed y tuned",
                "versions": {package: version(package) for package in
                             ("numpy", "pandas", "scipy", "scikit-learn", "shap")},
                "python": platform.python_version(),
                "tables": {name: f"tables/{name}.csv" for name in tables},
                "historical_outer_records": len(historical)}
    (logdir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def write_shap_family_postprocess(root: Path | str = ROOT) -> Path:
    """Actualiza la agregación descriptiva cuando las tablas OOF ya están calculadas."""
    root = Path(root).resolve()
    output = root / OUT
    source = output / "tables/shap_oof.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = load_frozen_data(root)
    result = aggregate_shap_families(pd.read_csv(source), frame)
    destination = output / "tables/shap_familias.csv"
    _save_csv(destination, result)
    manifest_path = output / "logs/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tables"]["shap_familias"] = "tables/shap_familias.csv"
    manifest["shap_family_aggregation"] = "suma descriptiva firmada de fuentes disjuntas; no SHAP grupal formal"
    manifest["family_postprocess_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--postprocess-families", action="store_true")
    args = parser.parse_args()
    if args.postprocess_families:
        print(write_shap_family_postprocess(args.root))
    else:
        print(json.dumps(run(args.root), ensure_ascii=False, indent=2))
