"""Evaluación externa anidada y persistencia local; ejecutar implica entrenamiento."""

from __future__ import annotations

from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd

from .ajuste import tune_candidates
from .ingesta import write_json
from .interpretabilidad import heldout_importance, linear_coefficients
from .modelo_baseline import build_baseline, evaluate_regression


def serializable_plan(plan: list[dict], source_records) -> list[dict]:
    """Guarda folds internos en posiciones globales y ordinales de la ingesta."""
    records = np.asarray(source_records)
    return [{
        "fold": p["fold"], "train_positions": p["train"].tolist(),
        "test_positions": p["test"].tolist(),
        "train_source_records": records[p["train"]].tolist(),
        "test_source_records": records[p["test"]].tolist(),
        "inner": [{
            "train_positions": p["train"][a].tolist(),
            "validation_positions": p["train"][b].tolist(),
        } for a, b in p["inner"]],
    } for p in plan]


def run_nested_validation(X, y, trace, plan, candidates, config, output_dir, logger) -> dict:
    """Ajusta por fold, compara con mediana y guarda OOF; no elige por error externo."""
    write_json(output_dir / "splits.json", {"folds": serializable_plan(plan, trace["__source_record"])})
    trace.to_csv(output_dir / "record_index.csv", index=False)
    metrics, predictions, selections, warnings_log = [], [], [], []
    for p in plan:
        fold, train, test = p["fold"], p["train"], p["test"]
        logger.info("Fold externo %s: entrenamiento=%s evaluación=%s", fold, len(train), len(test))
        fold_dir = output_dir / f"fold_{fold:02d}"
        fold_dir.mkdir()
        searches, selected, history, notices = tune_candidates(
            X.iloc[train], y.iloc[train], candidates, p["inner"], config["execution"]["n_jobs"]
        )
        history.to_csv(fold_dir / "inner_cv_results.csv", index=False)
        warnings_log.extend({"fold": fold, "message": message} for message in notices)
        for message in notices:
            logger.warning("Fold %s: %s", fold, message)
        baseline = build_baseline()
        # El dummy no necesita observar los predictores: solo la mediana de y_train.
        baseline.fit(np.zeros((len(train), 1)), y.iloc[train])
        estimates = {"baseline": baseline.predict(np.zeros((len(test), 1)))}
        params = {}
        for name, search in searches.items():
            estimates[name] = search.predict(X.iloc[test])
            params[name] = {"parameters": search.best_params_, "inner_mae": -search.best_score_}
            joblib.dump(search.best_estimator_, fold_dir / f"{name}.joblib")
        joblib.dump(baseline, fold_dir / "baseline.joblib")
        estimates["selected"] = estimates[selected]
        selections.append({"fold": fold, "selected_by_inner_cv": selected, "candidates": params})
        for procedure, y_pred in estimates.items():
            fold_metrics = asdict(evaluate_regression(y.iloc[test], y_pred))
            metrics.append({"fold": fold, "procedure": procedure, "rows": len(test), **fold_metrics})
            frame = trace.iloc[test].copy()
            frame["fold"] = fold
            frame["procedure"] = procedure
            frame["selected_family"] = selected if procedure == "selected" else procedure
            frame["observed"] = y.iloc[test].to_numpy()
            frame["predicted"] = y_pred
            frame["residual"] = frame["observed"] - frame["predicted"]
            predictions.append(frame)
        coefficients = linear_coefficients(searches[selected].best_estimator_, X.columns)
        coefficients.to_csv(fold_dir / "selected_coefficients.csv", index=False)
        if config["interpretation"]["permutation_enabled"]:
            heldout_importance(
                searches[selected].best_estimator_, X.iloc[test], y.iloc[test],
                config["interpretation"]["permutation_repeats"], config["protocol"]["seed"] + fold,
            ).to_csv(fold_dir / "selected_permutation_importance.csv", index=False)
    fold_metrics = pd.DataFrame(metrics)
    oof = pd.concat(predictions, ignore_index=True)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    summary = {}
    for name, rows in oof.groupby("procedure", sort=False):
        summary[name] = asdict(evaluate_regression(rows["observed"], rows["predicted"]))
    paired = fold_metrics.pivot(index="fold", columns="procedure", values="mae")
    paired["baseline_minus_selected_mae"] = paired["baseline"] - paired["selected"]
    paired.to_csv(output_dir / "paired_mae.csv")
    result = {
        "pooled_oof_metrics": summary,
        "selection_by_fold": selections,
        "warnings": warnings_log,
        "interpretation": [
            "selected evalúa selección de familia e hiperparámetros dentro de CV interna",
            "Las métricas agregadas ponderan observaciones; los folds comparten entrenamientos",
            "La dispersión entre folds no es un intervalo de confianza de muestras independientes",
            "Las métricas por familia son diagnósticas; elegir con ellas requiere nueva evaluación",
            "Importancia predictiva no implica causalidad física",
        ],
    }
    write_json(output_dir / "evaluation.json", result)
    return result
