"""Ajuste anidado: particiones históricas externas y búsqueda solo dentro de train.

Ejecutar: python -m src.ajuste_modelos --output reports/modelos_ajustados
La salida debe estar vacía. Los resultados no se utilizan para ampliar la rejilla.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import traceback
import uuid
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from threadpoolctl import threadpool_info, threadpool_limits

from .comparacion_modelos import validar_particiones
from .experimentos_fe import save_csv, save_json, source_digest
from .feature_engineering import TARGET, feature_names, make_pipeline, nominal_groups
from .ingesta import sha256_file
from .transformaciones_eda import GEOMETRY

ROOT = Path(__file__).resolve().parents[1]
DATASET_SHA256 = "2d0a17f54bdc6a6361aaa6e25a57cf67775c7d47a5e2adefb408f114defa21f8"
SPLITS_SHA256 = "763f20193cd8c58beeb471bc2024015588068a48a1cc0540c0b10ee25ff61e59"
MODELS = ("Ridge", "ElasticNet", "SVR_RBF", "RandomForest", "ExtraTrees", "GaussianProcess", "GradientBoosting")
BASELINE = "Baseline_Ridge"
SELECTOR = "Selector_interno"
GROUP_COLUMNS = ["n_cells_Y", "Fiber_height", "Pivot_height"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_value(value):
    """Conversión estable de parámetros sklearn y escalares NumPy para los logs."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


def append_jsonl(path, event):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(json_value(event), ensure_ascii=False, allow_nan=False) + "\n")


class TrackedPipeline(Pipeline):
    """Pipeline clonable que registra cada ajuste interno, incluidos avisos.

    Cada proceso escribe su propio JSONL para evitar colisiones entre workers.
    Los índices de X son identificadores de casos; no entran como predictores.
    """
    def __init__(self, steps, *, memory=None, verbose=False, audit_dir=None, context=None):
        super().__init__(steps, memory=memory, verbose=verbose)
        self.audit_dir = audit_dir
        self.context = context

    def _record(self, event):
        if self.audit_dir is not None:
            append_jsonl(Path(self.audit_dir) / f"worker_{os.getpid()}.jsonl", event)

    def fit(self, X, y=None, **params):
        self.fit_id_ = uuid.uuid4().hex
        self.fit_started_utc_ = utc_now()
        started = perf_counter()
        error = None
        caught = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                super().fit(X, y, **params)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.fit_warnings_ = [{"categoria": w.category.__name__, "mensaje": str(w.message)} for w in caught]
            self.fit_finished_utc_ = utc_now()
            self.fit_seconds_ = perf_counter() - started
            self._record({"evento": "fit", "fit_id": self.fit_id_, **(self.context or {}),
                          "inicio_utc": self.fit_started_utc_, "fin_utc": self.fit_finished_utc_,
                          "tiempo_s": self.fit_seconds_, "train_cases": X.index.astype(int).tolist(),
                          "parametros_modelo": self.named_steps["model"].get_params(deep=False),
                          "avisos": self.fit_warnings_, "error": error})
        return self


def score_inner(estimator, X, y):
    prediction = estimator.predict(X)
    if not np.isfinite(prediction).all():
        raise ValueError("Predicción interna no finita.")
    mae = float(mean_absolute_error(y, prediction))
    estimator._record({"evento": "score_interno", "fit_id": estimator.fit_id_,
                       **(estimator.context or {}), "timestamp_utc": utc_now(),
                       "validation_cases": X.index.astype(int).tolist(), "MAE_N": mae})
    return {"neg_MAE": -mae, "avisos": float(len(estimator.fit_warnings_))}


def load_config(root=ROOT):
    config = json.loads((Path(root) / "config/ajuste_modelos.json").read_text(encoding="utf-8"))
    if (config["dataset_sha256"] != DATASET_SHA256 or config["particiones_sha256"] != SPLITS_SHA256
            or config["target"] != TARGET or config["target_unit"] != "N" or config["representacion"] != "A"
            or config["seed"] != 42 or config["group_columns"] != GROUP_COLUMNS
            or tuple(config["modelos"]) != MODELS or config["protocolos"] != ["configuracion", "familia"]):
        raise ValueError("La configuración no cumple el contrato del experimento congelado.")
    return config


def load_frozen_data(root=ROOT):
    root = Path(root)
    config = load_config(root)
    path = root / config["dataset"]
    if sha256_file(path) != DATASET_SHA256:
        raise ValueError("El dataset no coincide con el snapshot histórico; no se ejecuta.")
    frame = pd.read_csv(path).sort_values("Case_n").reset_index(drop=True)
    if frame.Case_n.isna().any() or frame.Case_n.duplicated().any():
        raise ValueError("Los casos deben tener identificadores únicos.")
    if frame[TARGET].isna().any() or not np.isfinite(frame[TARGET]).all() or (frame[TARGET] <= 0).any():
        raise ValueError("El target debe ser observado, finito y positivo; no se imputa ni filtra.")
    make_pipeline("A").named_steps["contract"].fit(frame[GEOMETRY])
    return frame


def load_outer_splits(frame, root=ROOT):
    root = Path(root)
    config = load_config(root)
    path = root / config["particiones"]
    if sha256_file(path) != SPLITS_SHA256:
        raise ValueError("Las particiones históricas han cambiado.")
    records = json.loads(path.read_text(encoding="utf-8"))
    groups = {"configuracion": nominal_groups(frame, GROUP_COLUMNS),
              "familia": frame.n_cells_Y.astype(int).astype(str)}
    lookup = {int(case): i for i, case in enumerate(frame.Case_n)}
    splits = {p: [] for p in config["protocolos"]}
    for protocol in splits:
        selected = sorted((r for r in records if r["protocolo"] == protocol), key=lambda r: r["fold"])
        for record in selected:
            splits[protocol].append((np.array([lookup[c] for c in record["train_cases"]], dtype=int),
                                     np.array([lookup[c] for c in record["test_cases"]], dtype=int)))
    validar_particiones(frame, splits, groups, records)
    return splits, groups, records


def inner_splits(frame, outer_train, protocol):
    """Devuelve índices relativos a outer_train; nunca observa el target."""
    subset = frame.iloc[np.asarray(outer_train, dtype=int)]
    if protocol == "configuracion":
        grouping = nominal_groups(subset, GROUP_COLUMNS)
        if grouping.nunique() < 5:
            raise ValueError("Se requieren al menos cinco configuraciones en train externo.")
        splitter = GroupKFold(n_splits=5)
    elif protocol == "familia":
        grouping = subset.n_cells_Y.astype(int).astype(str)
        if grouping.nunique() < 2:
            raise ValueError("Se requieren al menos dos familias en train externo.")
        splitter = LeaveOneGroupOut()
    else:
        raise ValueError(f"Protocolo no reconocido: {protocol}")
    result = list(splitter.split(subset, groups=grouping))
    seen = []
    for train, validation in result:
        if set(grouping.iloc[train]) & set(grouping.iloc[validation]):
            raise ValueError("Solapamiento de grupos internos.")
        if set(train) & set(validation) or set(train) | set(validation) != set(range(len(subset))):
            raise ValueError("Partición interna incompleta.")
        seen.extend(validation.tolist())
    if sorted(seen) != list(range(len(subset))):
        raise ValueError("Cada caso de train externo debe validarse internamente una vez.")
    return result


def model_specs(config):
    """Construye estimadores nuevos; cada rejilla queda declarada antes de evaluar."""
    classes = {"Ridge": Ridge, "ElasticNet": ElasticNet, "SVR_RBF": SVR,
               "RandomForest": RandomForestRegressor, "ExtraTrees": ExtraTreesRegressor,
               "GaussianProcess": GaussianProcessRegressor, "GradientBoosting": GradientBoostingRegressor}
    specs = {}
    for name, declaration in config["modelos"].items():
        estimator = classes[name](**declaration["parametros"])
        grid = dict(declaration["grid"])
        if name == "GaussianProcess":
            grid["model__kernel"] = [Matern(length_scale=value, length_scale_bounds="fixed", nu=declaration["matern_nu"])
                                     for value in declaration["matern_length_scales"]]
        if not 1 <= len(ParameterGrid(grid)) <= 20:
            raise ValueError(f"Rejilla fuera del límite declarado: {name}")
        pipeline = make_pipeline("A").set_params(model=estimator)
        specs[name] = (pipeline, grid)
    return specs


def select_inner_winner(scores):
    """Selección basada exclusivamente en el score interno negativo de MAE."""
    if not scores or any(not np.isfinite(value) for value in scores.values()):
        raise ValueError("Se requieren scores internos finitos.")
    if any(name not in MODELS for name in scores):
        raise ValueError("Modelo fuera del conjunto predeclarado.")
    return max((name for name in MODELS if name in scores), key=lambda name: scores[name])


def _audit(pipeline, frame, train, test, protocol, name, fold, params):
    audit = {"protocolo": protocol, "modelo": name, "experimento": "A", "fold": fold,
             "train_cases": frame.iloc[train].Case_n.astype(int).tolist(),
             "test_cases": frame.iloc[test].Case_n.astype(int).tolist(),
             "imputer_columns": GEOMETRY, "imputer_median": pipeline.named_steps["imputer"].statistics_.tolist(),
             "feature_names": feature_names(pipeline).tolist(),
             "variance_support": pipeline.named_steps["variance"].get_support().tolist(),
             "variance_values": pipeline.named_steps["variance"].variances_.tolist(),
             "scaler_mean": pipeline.named_steps["scale"].mean_.tolist(),
             "scaler_scale": pipeline.named_steps["scale"].scale_.tolist(),
             "scaler_n_samples_seen": int(pipeline.named_steps["scale"].n_samples_seen_),
             "best_params": json_value(params), "avisos": pipeline.fit_warnings_,
             "inicio_utc": pipeline.fit_started_utc_, "fin_utc": pipeline.fit_finished_utc_}
    if isinstance(pipeline.named_steps["model"], GaussianProcessRegressor):
        audit["target_train_mean"] = float(pipeline.named_steps["model"]._y_train_mean)
        audit["target_train_std"] = float(pipeline.named_steps["model"]._y_train_std)
    return audit


def _metrics(observed, predicted):
    return {"MAE_N": float(mean_absolute_error(observed, predicted)),
            "RMSE_N": float(np.sqrt(mean_squared_error(observed, predicted))),
            "MedAE_N": float(median_absolute_error(observed, predicted)),
            "R2_OOF": float(r2_score(observed, predicted)) if len(observed) > 1 else None}


def _predictions(frame, groups, test, predicted, protocol, name, fold, selected_model=None):
    rows = []
    for pos, estimate in zip(test, predicted):
        observed = float(frame.iloc[pos][TARGET])
        residual = observed - float(estimate)
        rows.append({"protocolo": protocol, "modelo": name, "fold": fold, "Case_n": int(frame.iloc[pos].Case_n),
                     "familia": int(frame.iloc[pos].n_cells_Y), "grupo": str(groups["configuracion"].iloc[pos]),
                     "observado_N": observed, "predicho_N": float(estimate), "residuo_N": residual,
                     "error_absoluto_N": abs(residual), "modelo_seleccionado": selected_model or name})
    return rows


def _search_rows(search, protocol, name, fold, started, ended):
    result = search.cv_results_
    rows = []
    for i, params in enumerate(result["params"]):
        row = {"protocolo": protocol, "modelo": name, "fold": fold, "candidate_id": i,
               "params_json": json.dumps(json_value(params), ensure_ascii=False, sort_keys=True),
               "mean_inner_MAE_N": -float(result["mean_test_neg_MAE"][i]),
               "std_inner_MAE_N": float(result["std_test_neg_MAE"][i]),
               "rank": int(result["rank_test_neg_MAE"][i]), "selected": i == search.best_index_,
               "mean_fit_time_s": float(result["mean_fit_time"][i]), "std_fit_time_s": float(result["std_fit_time"][i]),
               "mean_score_time_s": float(result["mean_score_time"][i]), "std_score_time_s": float(result["std_score_time"][i]),
               "mean_fit_warning_count": float(result["mean_test_avisos"][i]),
               "inicio_utc": started, "fin_utc": ended}
        for inner_fold in range(search.n_splits_):
            row[f"split{inner_fold}_MAE_N"] = -float(result[f"split{inner_fold}_test_neg_MAE"][i])
            row[f"split{inner_fold}_fit_warning_count"] = float(result[f"split{inner_fold}_test_avisos"][i])
        rows.append(row)
    return rows


def _environment():
    packages = {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl")}
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "threadpool_info": threadpool_info(),
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}


def run(root=ROOT, output=None, n_jobs=2):
    root = Path(root).resolve()
    out = root / "reports/modelos_ajustados" if output is None else Path(output)
    if not out.is_absolute():
        out = root / out
    out = out.resolve()
    if n_jobs < 1:
        raise ValueError("n_jobs debe ser positivo y explícito.")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"No se sobrescriben resultados: la carpeta debe estar vacía: {out}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"estado": "preparando", "ejecutado": False, "inicio_utc": utc_now(),
                "seed": 42, "n_jobs_busqueda": n_jobs, "n_jobs_estimadores": 1, "salida": str(out)}
    log_path = out / "ejecucion.jsonl"
    save_json(out / "manifest.json", manifest)
    append_jsonl(log_path, {"evento": "inicio", **manifest})
    try:
        config = load_config(root)
        frame = load_frozen_data(root)
        splits, groups, records = load_outer_splits(frame, root)
        before = {config[key]: sha256_file(root / config[key]) for key in ("dataset", "particiones")}
        source_paths = ["config/ajuste_modelos.json", "src/ajuste_modelos.py", "src/feature_engineering.py",
                        "src/comparacion_modelos.py", "src/experimentos_fe.py", "src/transformaciones_eda.py"]
        source_hashes = {path: source_digest(root / path) for path in source_paths}
        specs = model_specs(config)
        nested, internal_records = {}, []
        for protocol, partitions in splits.items():
            for fold, (train, test) in enumerate(partitions, 1):
                nested[(protocol, fold)] = inner_splits(frame, train, protocol)
                for inner_fold, (inner_train, validation) in enumerate(nested[(protocol, fold)], 1):
                    internal_records.append({"protocolo": protocol, "fold": fold, "inner_fold": inner_fold,
                        "train_cases": frame.iloc[train[inner_train]].Case_n.astype(int).tolist(),
                        "validation_cases": frame.iloc[train[validation]].Case_n.astype(int).tolist(),
                        "outer_test_cases": frame.iloc[test].Case_n.astype(int).tolist()})
        save_json(out / "config_ejecutada.json", config)
        save_json(out / "particiones_externas.json", records)
        save_json(out / "particiones_internas.json", internal_records)
        manifest.update({"estado": "ejecutando", "data_sha256_antes": before,
                         "fuentes_sha256": source_hashes, "entorno": _environment(),
                         "fuentes_hash_normalizacion": "CRLF a LF; SHA-256 de bytes UTF-8",
                         "target": TARGET, "target_unit": "N", "entradas": GEOMETRY,
                         "representacion": "A", "modelos": [BASELINE, *MODELS, SELECTOR],
                         "candidatos_por_modelo": {name: len(ParameterGrid(grid)) for name, (_, grid) in specs.items()},
                         "criterio_busqueda": config["criterio_busqueda"],
                         "criterio_selector": config["criterio_selector"],
                         "particiones_predeclaradas_utc": utc_now(),
                         "sin_holdout_independiente": True,
                         "nota": "Evaluación anidada de un snapshot ya explorado. No equivale a validación externa prospectiva."})
        try:
            manifest["git_commit_inicio"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            manifest["git_commit_inicio"] = None
        save_json(out / "manifest.json", manifest)
        append_jsonl(log_path, {"evento": "protocolo_congelado", "timestamp_utc": utc_now(),
                               "config": config, "data_sha256": before, "fuentes_sha256": source_hashes})
        x = frame[GEOMETRY].copy()
        x.index = pd.Index(frame.Case_n.astype(int), name="Case_n")
        y = pd.Series(frame[TARGET].to_numpy(), index=x.index, name=TARGET)
        predictions, fold_rows, searches, audits, selections = [], [], [], [], []
        fit_log_dir = out / "logs_ajustes"
        fit_log_dir.mkdir()
        np.random.seed(config["seed"])
        experiment_started = perf_counter()
        with threadpool_limits(limits=1), parallel_backend("loky", inner_max_num_threads=1):
            for protocol, partitions in splits.items():
                for fold, (train, test) in enumerate(partitions, 1):
                    print(f"[{utc_now()}] {protocol}: fold {fold}/{len(partitions)}", flush=True)
                    append_jsonl(log_path, {"evento": "fold_inicio", "timestamp_utc": utc_now(),
                                           "protocolo": protocol, "fold": fold})
                    inner_scores, heldout_predictions, fit_costs, feature_counts = {}, {}, {}, {}
                    for name in [BASELINE, *MODELS]:
                        context = {"protocolo": protocol, "modelo": name, "fold": fold}
                        base = make_pipeline("A") if name == BASELINE else specs[name][0]
                        model = TrackedPipeline(steps=base.steps, audit_dir=str(fit_log_dir), context=context)
                        started_utc, started = utc_now(), perf_counter()
                        if name == BASELINE:
                            model.fit(x.iloc[train], y.iloc[train])
                            best_params = {"model__alpha": 1.0, "model__solver": "svd"}
                            inner_mae = None
                        else:
                            search = GridSearchCV(model, specs[name][1], scoring=score_inner, refit="neg_MAE",
                                cv=nested[(protocol, fold)], n_jobs=n_jobs, pre_dispatch=n_jobs,
                                error_score="raise", return_train_score=False)
                            with warnings.catch_warnings(record=True) as search_warnings:
                                warnings.simplefilter("always")
                                search.fit(x.iloc[train], y.iloc[train])
                            for warning in search_warnings:
                                append_jsonl(log_path, {"evento": "aviso_busqueda", **context,
                                    "timestamp_utc": utc_now(), "categoria": warning.category.__name__,
                                    "mensaje": str(warning.message)})
                            model = search.best_estimator_
                            best_params = search.best_params_
                            inner_scores[name] = float(search.best_score_)
                            inner_mae = -float(search.best_score_)
                            searches.extend(_search_rows(search, protocol, name, fold, started_utc, utc_now()))
                        predicted = model.predict(x.iloc[test])
                        elapsed, ended_utc = perf_counter() - started, utc_now()
                        if not np.isfinite(predicted).all():
                            raise ValueError(f"Predicción externa no finita: {context}")
                        heldout_predictions[name] = predicted
                        fit_costs[name] = elapsed
                        feature_counts[name] = len(feature_names(model))
                        fold_rows.append({**context, **_metrics(y.iloc[test], predicted), "tiempo_s": elapsed,
                            "n_train": len(train), "n_test": len(test), "n_features": feature_counts[name],
                            "grupo_excluido": str(groups[protocol].iloc[test[0]]), "inner_MAE_N": inner_mae,
                            "params_json": json.dumps(json_value(best_params), ensure_ascii=False, sort_keys=True),
                            "inicio_utc": started_utc, "fin_utc": ended_utc, "modelo_seleccionado": name})
                        predictions.extend(_predictions(frame, groups, test, predicted, protocol, name, fold))
                        audits.append(_audit(model, frame, train, test, protocol, name, fold, best_params))
                        append_jsonl(log_path, {"evento": "modelo_fin", **fold_rows[-1]})
                        print(f"  {name}: ajuste terminado ({elapsed:.1f} s)", flush=True)
                    winner = select_inner_winner(inner_scores)
                    selection = {"protocolo": protocol, "fold": fold, "modelo_seleccionado": winner,
                                 "scores_internos_neg_MAE": inner_scores,
                                 "criterio": "max(best_score_) interno; empate por orden predeclarado",
                                 "train_cases": frame.iloc[train].Case_n.astype(int).tolist(),
                                 "test_cases": frame.iloc[test].Case_n.astype(int).tolist(), "timestamp_utc": utc_now()}
                    selections.append(selection)
                    predictions.extend(_predictions(frame, groups, test, heldout_predictions[winner], protocol, SELECTOR, fold, winner))
                    fold_rows.append({"protocolo": protocol, "modelo": SELECTOR, "fold": fold,
                        **_metrics(y.iloc[test], heldout_predictions[winner]),
                        "tiempo_s": sum(fit_costs[name] for name in MODELS),
                        "n_train": len(train), "n_test": len(test), "n_features": feature_counts[winner],
                        "grupo_excluido": str(groups[protocol].iloc[test[0]]), "inner_MAE_N": -inner_scores[winner],
                        "params_json": None, "inicio_utc": None, "fin_utc": utc_now(), "modelo_seleccionado": winner})
                    append_jsonl(log_path, {"evento": "seleccion_interna", **selection})
                    # Checkpoints permiten inspeccionar progreso; manifest.ejecutado permanece False.
                    save_csv(out / "predicciones_oof.csv", pd.DataFrame(predictions))
                    save_csv(out / "metricas_folds.csv", pd.DataFrame(fold_rows))
                    save_csv(out / "busquedas.csv", pd.DataFrame(searches))
                    save_json(out / "auditoria_train.json", audits)
                    save_json(out / "seleccion_interna.json", selections)
        prediction_frame, fold_frame = pd.DataFrame(predictions), pd.DataFrame(fold_rows)
        summary = []
        for (protocol, name), rows in prediction_frame.groupby(["protocolo", "modelo"], sort=False):
            if rows.Case_n.duplicated().any() or set(rows.Case_n) != set(frame.Case_n):
                raise ValueError("La cobertura OOF no es exactamente una predicción por caso.")
            subset = fold_frame[(fold_frame.protocolo == protocol) & (fold_frame.modelo == name)]
            summary.append({"protocolo": protocol, "modelo": name, **_metrics(rows.observado_N, rows.predicho_N),
                "tiempo_s": float(subset.tiempo_s.sum()), "MAE_fold_media_N": float(subset.MAE_N.mean()),
                "MAE_fold_sd_N": float(subset.MAE_N.std(ddof=1)), "n_folds": len(subset),
                "n_features_min": int(subset.n_features.min()), "n_features_max": int(subset.n_features.max()),
                "predicciones_negativas": int((rows.predicho_N < 0).sum())})
        summary_frame = pd.DataFrame(summary)
        for protocol in splits:
            mask = summary_frame.protocolo == protocol
            reference = summary_frame.loc[mask & (summary_frame.modelo == BASELINE), "MAE_N"].iloc[0]
            summary_frame.loc[mask, "mejora_MAE_vs_baseline_pct"] = 100 * (reference - summary_frame.loc[mask, "MAE_N"]) / reference
        save_csv(out / "metricas_resumen.csv", summary_frame)
        after = {path: sha256_file(root / path) for path in before}
        source_after = {path: source_digest(root / path) for path in source_hashes}
        if before != after or source_hashes != source_after:
            raise RuntimeError("Datos, particiones o código cambiaron durante la ejecución; resultados invalidados.")
        manifest.update({"estado": "completado", "ejecutado": True, "fin_utc": utc_now(),
                         "tiempo_total_s": perf_counter() - experiment_started, "data_sha256_despues": after,
                         "fuentes_sha256_despues": source_after,
                         "n_predicciones": len(prediction_frame), "n_ajustes_externos": len(audits),
                         "n_candidatos_evaluados": len(searches)})
        append_jsonl(log_path, {"evento": "fin", "timestamp_utc": utc_now(), "estado": "completado"})
        manifest["artefactos_sha256"] = {path.relative_to(out).as_posix(): sha256_file(path)
                                         for path in out.rglob("*") if path.is_file() and path.name != "manifest.json"}
        save_json(out / "manifest.json", manifest)
        return {"output": str(out), "manifest": str(out / "manifest.json"),
                "metricas_resumen": str(out / "metricas_resumen.csv"), "estado": "completado"}
    except Exception as exc:
        manifest.update({"estado": "error", "ejecutado": False, "fin_utc": utc_now(),
                         "error": f"{type(exc).__name__}: {exc}"})
        append_jsonl(log_path, {"evento": "error", "timestamp_utc": utc_now(),
                               "error": manifest["error"], "traceback": traceback.format_exc()})
        save_json(out / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/modelos_ajustados")
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(output=args.output, n_jobs=args.n_jobs), ensure_ascii=False, indent=2))
