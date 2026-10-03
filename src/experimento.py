"""Orquestación reproducible. Por defecto inspecciona preparación y NO entrena."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import ParameterGrid

from .ajuste import tune_candidates
from .ingesta import (
    ROOT, configure_logging, execution_context, load_table, new_run_directory,
    project_path, sha256_file, write_json,
)
from .modelos import candidate_specs
from .particiones import check_training_features, make_splits, nested_plan
from .preprocesamiento import validate_schema
from .validacion import run_nested_validation


def read_config(path: Path) -> tuple[dict, str]:
    """Lee y valida el contrato de configuración sin inferir tamaño ni particiones."""
    content = project_path(path).read_bytes()
    config = yaml.safe_load(content.decode("utf-8"))
    sections = ("data", "protocol", "execution", "models", "interpretation")
    if not isinstance(config, dict) or any(not isinstance(config.get(s), dict) for s in sections):
        raise ValueError(f"La configuración requiere las secciones {sections}.")
    if type(config["protocol"].get("seed")) is not int or config["protocol"]["seed"] < 0:
        raise ValueError("seed debe ser un entero no negativo.")
    if type(config["execution"].get("allow_training")) is not bool:
        raise ValueError("execution.allow_training debe ser booleano.")
    n_jobs = config["execution"].get("n_jobs")
    if type(n_jobs) is not int or n_jobs == 0:
        raise ValueError("n_jobs debe ser un entero distinto de cero.")
    interpretation = config["interpretation"]
    if type(interpretation.get("permutation_enabled")) is not bool:
        raise ValueError("permutation_enabled debe ser booleano.")
    if type(interpretation.get("permutation_repeats")) is not int or interpretation["permutation_repeats"] < 2:
        raise ValueError("permutation_repeats debe ser un entero >= 2.")
    candidate_specs(config)
    return config, hashlib.sha256(content).hexdigest()


def load_model_data(config: dict):
    """Selecciona X por lista explícita, valida procedencia y construye grupos."""
    value = config["data"].get("model_table")
    if not isinstance(value, str) or not value:
        raise ValueError("Falta data.model_table: indique una tabla real curada.")
    path = project_path(Path(value))
    if not path.is_relative_to((ROOT / "data" / "processed").resolve()):
        raise ValueError("model_table debe pertenecer a data/processed/.")
    report_path = path.parent / "data_quality_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("output_sha256") != sha256_file(path):
        raise ValueError("La tabla no coincide con el hash del reporte de curaduría.")
    roles = validate_schema(report["schema"])
    if roles != report.get("roles"):
        raise ValueError("Los roles y el esquema del reporte no coinciden.")
    table = load_table(path)
    if report["output_sha256"] != sha256_file(path):
        raise ValueError("La tabla cambió durante la lectura.")
    expected = ["__source_record"] + roles["identifiers"] + roles["groups"] + roles["predictors"] + [roles["target"]]
    if table.columns.tolist() != expected:
        raise ValueError("La tabla no coincide con el orden y los roles de la curaduría.")
    for c in roles["predictors"] + [roles["target"], "__source_record"]:
        table[c] = pd.to_numeric(table[c], errors="raise")
    X = table[roles["predictors"]].astype(float)
    y = table[roles["target"]].astype(float)
    if not np.isfinite(y).all() or np.isinf(X.to_numpy()).any():
        raise ValueError("Target incompleto o valores infinitos en la tabla.")
    source = table["__source_record"]
    if source.isna().any() or source.duplicated().any() or (source < 1).any() or (source % 1 != 0).any():
        raise ValueError("Los ordinales de procedencia deben ser enteros positivos únicos.")
    trace = table[["__source_record"] + roles["identifiers"] + roles["groups"]].copy()
    if trace.isna().any().any():
        raise ValueError("Faltan identificadores o grupos.")
    protocol = config["protocol"]
    group_columns = protocol.get("group_columns")
    if not isinstance(group_columns, list) or any(not isinstance(c, str) for c in group_columns):
        raise ValueError("group_columns debe ser una lista de nombres.")
    if len(set(group_columns)) != len(group_columns) or set(group_columns) - set(roles["identifiers"] + roles["groups"]):
        raise ValueError("Agrupe únicamente por identificadores o grupos declarados y no repetidos.")
    groups = None
    if protocol.get("strategy") == "group_kfold":
        if not group_columns:
            raise ValueError("Declare las columnas que identifican las observaciones dependientes.")
        groups = pd.factorize(pd.MultiIndex.from_frame(trace[group_columns]))[0]
        # Un mismo caso no puede dividirse por haber cambiado otra etiqueta de grupo.
        case_groups = trace[roles["identifiers"]].copy()
        case_groups["__group_code"] = groups
        if (case_groups.groupby(roles["identifiers"])["__group_code"].nunique() > 1).any():
            raise ValueError("Un identificador de caso pertenece a más de un grupo.")
        trace["__group_code"] = groups
    elif group_columns or trace.duplicated(subset=roles["identifiers"]).any():
        raise ValueError("Hay grupos o casos repetidos: revise la estrategia agrupada.")
    return X, y, trace, groups, report, path, report_path


def inspect_plan(config: dict) -> dict:
    """Revisa estructura/configuración y, si hay datos, folds. Nunca hace fit."""
    specs = candidate_specs(config)
    summary = {
        "mode": "inspection_only_no_training",
        "candidates": {name: len(ParameterGrid(grid)) for name, (_, grid) in specs.items()},
        "protocol_reviewed": config["protocol"].get("reviewed") is True,
        "training_enabled": config["execution"]["allow_training"],
    }
    if not config["data"].get("model_table"):
        return {**summary, "status": "pending_real_data_and_protocol"}
    X, _, trace, groups, _, _, _ = load_model_data(config)
    if not summary["protocol_reviewed"]:
        return {**summary, "status": "pending_protocol_review"}
    plan = nested_plan(X, config["protocol"], groups)
    return {**summary, "status": "ready_for_reviewed_execution", "rows": len(trace),
            "outer_folds": len(plan), "features": X.columns.tolist()}


def find_completed_run(config: dict, config_hash: str, *, require_final: bool = False) -> Path | None:
    """Localiza resultados completos de los datos/configuración actuales y verifica hashes.

    No reutiliza corridas parciales ni resultados de otra versión. Un artefacto
    alterado en una corrida compatible provoca error en lugar de mostrar métricas
    que ya no corresponden a su manifiesto.
    """
    _, _, _, _, _, data_path, report_path = load_model_data(config)
    data_hash, quality_hash = sha256_file(data_path), sha256_file(report_path)
    manifests = sorted((ROOT / "results").glob("*/experiment_manifest.json"), reverse=True)
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("status") != "complete"
            or manifest.get("configuration_sha256") != config_hash
            or manifest.get("dataset_sha256") != data_hash
            or manifest.get("quality_report_sha256") != quality_hash
        ):
            continue
        run_dir = manifest_path.parent.resolve()
        artifacts = manifest["artifacts_sha256"]
        if require_final and "final_model.joblib" not in artifacts:
            continue
        for relative, digest in artifacts.items():
            artifact = (run_dir / relative).resolve()
            if not artifact.is_relative_to(run_dir) or not artifact.is_file():
                raise ValueError(f"Artefacto ausente o fuera de la ejecución: {relative}")
            if sha256_file(artifact) != digest:
                raise ValueError(f"Artefacto modificado: {relative}")
        return run_dir
    return None


def run_experiment(config: dict, config_hash: str, *, fit_final: bool = False) -> Path:
    """Entrena solo mediante llamada explícita y configuración habilitada/revisada."""
    if config["execution"].get("allow_training") is not True:
        raise ValueError("Entrenamiento deshabilitado en execution.allow_training.")
    X, y, trace, groups, quality, data_path, report_path = load_model_data(config)
    data_hash, quality_hash = sha256_file(data_path), sha256_file(report_path)
    plan = nested_plan(X, config["protocol"], groups)
    candidates = candidate_specs(config)
    final_splits = None
    if fit_final:
        final_splits = make_splits(
            len(X), config["protocol"]["inner_splits"], config["protocol"]["strategy"],
            config["protocol"]["seed"], groups,
        )
        for train, _ in final_splits:
            check_training_features(X.iloc[train])
    output = new_run_directory(ROOT / "results")
    logger = configure_logging("modeling")
    write_json(output / "experiment_config.json", config)
    try:
        run_nested_validation(X, y, trace, plan, candidates, config, output, logger)
        if fit_final:
            searches, family, history, notices = tune_candidates(
                X, y, candidates, final_splits, config["execution"]["n_jobs"]
            )
            joblib.dump(searches[family].best_estimator_, output / "final_model.joblib")
            history.to_csv(output / "final_inner_cv_results.csv", index=False)
            write_json(output / "final_model_info.json", {
                "family": family, "parameters": searches[family].best_params_, "warnings": notices,
                "features": X.columns.tolist(), "target_unit": quality["schema"]["project"]["target_unit"],
                "note": "Reajuste con todos los datos; no tiene una nueva evaluación independiente",
            })
        if sha256_file(data_path) != data_hash or sha256_file(report_path) != quality_hash:
            raise ValueError("La tabla o el reporte de curaduría cambió durante el experimento.")
        artifacts = {p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob("*") if p.is_file()}
        write_json(output / "experiment_manifest.json", {
            "status": "complete", "run_id": output.name, "configuration_sha256": config_hash,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": data_hash, "quality_report_sha256": quality_hash,
            "data_file": data_path.relative_to(ROOT).as_posix(),
            "execution": execution_context(), "artifacts_sha256": artifacts,
        })
    except Exception as exc:
        write_json(output / "failure.json", {"status": "failed", "error": str(exc)})
        logger.exception("Experimento interrumpido; salidas parciales en %s", output.name)
        raise
    logger.info("Experimento completado: %s", output.name)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "model_config.yml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Inspeccionar preparación sin entrenar (defecto)")
    mode.add_argument("--run", action="store_true", help="Entrenar y evaluar con protocolo habilitado")
    parser.add_argument("--fit-final", action="store_true", help="Reajustar después de evaluar; requiere --run")
    args = parser.parse_args()
    try:
        config, config_hash = read_config(args.config)
        if args.fit_final and not args.run:
            raise ValueError("--fit-final requiere --run.")
        if args.run:
            print(run_experiment(config, config_hash, fit_final=args.fit_final))
        else:
            print(json.dumps(inspect_plan(config), indent=2, ensure_ascii=False))
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        parser.exit(1, f"No se puede completar la operación: {exc}\n")


if __name__ == "__main__":
    main()
