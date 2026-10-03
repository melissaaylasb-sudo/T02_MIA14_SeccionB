"""Curaduría determinística preliminar; no ajusta transformaciones estadísticas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

if __package__:
    from .ingesta import (
        ROOT, INTERIM_DIR, configure_logging, execution_context, load_table,
        new_run_directory, project_path, sha256_file, validate_columns, write_json,
    )
else:
    from ingesta import (
        ROOT, INTERIM_DIR, configure_logging, execution_context, load_table,
        new_run_directory, project_path, sha256_file, validate_columns, write_json,
    )

PROCESSED_DIR = ROOT / "data" / "processed"
SOURCE_RECORD = "__source_record"


def quality_report(df: pd.DataFrame) -> dict:
    """Describe calidad sin eliminar variables ni estimar parámetros del modelo."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "duplicated_rows": int(df.duplicated().sum()),
        "missing_by_column": {str(c): int(v) for c, v in df.isna().sum().items()},
        "constant_columns": [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1],
    }


def validate_schema(schema: dict) -> dict:
    """Exige roles y unidades explícitos; no infiere predictores desde el target."""
    if not isinstance(schema, dict):
        raise ValueError("El esquema YAML debe ser un objeto con project, columns y rules.")
    project = schema.get("project", {})
    roles = schema.get("columns", {})
    units = schema.get("predictor_units", {})
    rules = schema.get("rules", {})
    curation = schema.get("curation", {})
    if not all(isinstance(item, dict) for item in (project, roles, units, rules, curation)):
        raise ValueError("Las secciones del esquema deben ser objetos YAML.")
    target = roles.get("target")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Complete columns.target con el nombre real de la carga de primera falla.")
    if project.get("task") != "regression" or not isinstance(project.get("target_unit"), str):
        raise ValueError("Confirme task: regression y project.target_unit en el esquema.")
    if not project["target_unit"].strip():
        raise ValueError("La unidad del target no puede estar vacía.")
    for role in ("identifiers", "predictors", "groups"):
        values = roles.get(role)
        if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
            raise ValueError(f"columns.{role} debe ser una lista de nombres reales.")
    if not roles["identifiers"] or not roles["predictors"]:
        raise ValueError("Declare al menos un identificador y un predictor verificados.")
    names = [target] + roles["identifiers"] + roles["predictors"] + roles["groups"]
    if validate_columns(names) != names or SOURCE_RECORD in names:
        raise ValueError("Use nombres sin espacios extremos; __source_record está reservado.")
    for column in roles["predictors"]:
        if not isinstance(units.get(column), str) or not units[column].strip():
            raise ValueError(f"Confirme la unidad de {column!r} en predictor_units.")
    for rule in (
        "target_imputation", "identifier_as_predictor", "fit_statistical_preprocessing_before_split"
    ):
        if rules.get(rule) is not False:
            raise ValueError(f"La regla {rule} debe mantenerse en false.")
    if type(curation.get("drop_exact_duplicates")) is not bool:
        raise ValueError("curation.drop_exact_duplicates debe ser true o false.")
    missing = curation.get("missing_values", {})
    if not isinstance(missing, dict) or any(
        not isinstance(c, str) or not isinstance(values, list)
        or any(not isinstance(value, str) for value in values)
        for c, values in missing.items()
    ):
        raise ValueError("curation.missing_values debe asociar columnas a listas de textos.")
    return roles


def curate(df: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, dict]:
    """Construye una tabla por roles y documenta cada exclusión por registro.

    __source_record es el ordinal (desde 1) en la tabla de ingesta, no una línea
    física del CSV. Se conserva como trazabilidad y nunca debe usarse en X.
    """
    roles = validate_schema(schema)
    out = df.copy().reset_index(drop=True)
    out.columns = validate_columns(out.columns)
    if SOURCE_RECORD in out.columns:
        raise ValueError(f"La fuente ya contiene el nombre reservado {SOURCE_RECORD}.")
    selected = roles["identifiers"] + roles["groups"] + roles["predictors"] + [roles["target"]]
    absent = sorted(set(selected) - set(out.columns))
    if absent:
        raise ValueError(f"Columnas declaradas ausentes: {absent}")
    out = out.replace(r"^\s*$", pd.NA, regex=True)
    for column, markers in schema["curation"].get("missing_values", {}).items():
        if column not in out:
            raise ValueError(f"Columna de marcadores ausente: {column}")
        out[column] = out[column].replace(markers, pd.NA)
    before = quality_report(out)
    duplicated = out.duplicated()
    out.insert(0, SOURCE_RECORD, np.arange(1, len(out) + 1))
    exclusions = []
    if schema["curation"]["drop_exact_duplicates"]:
        exclusions.extend(
            {"source_record": int(i), "reason": "confirmed_exact_duplicate"}
            for i in out.loc[duplicated, SOURCE_RECORD]
        )
        out = out.loc[~duplicated].copy()
    target_missing = out[roles["target"]].isna()
    exclusions.extend(
        {"source_record": int(i), "reason": "missing_target"}
        for i in out.loc[target_missing, SOURCE_RECORD]
    )
    out = out.loc[~target_missing].copy()
    if out.empty:
        raise ValueError("No quedan registros con target disponible; no se genera model_table.csv.")
    for column in roles["identifiers"] + roles["groups"]:
        if out[column].isna().any():
            raise ValueError(f"Faltan valores de identificación o grupo en {column!r}.")
    for column in roles["predictors"] + [roles["target"]]:
        converted = pd.to_numeric(out[column], errors="coerce")
        invalid = out[column].notna() & converted.isna()
        infinite = np.isinf(converted.to_numpy(dtype=float, na_value=np.nan))
        if invalid.any() or infinite.any():
            records = out.loc[invalid | infinite, SOURCE_RECORD].tolist()
            raise ValueError(f"Valores numéricos inválidos en {column!r}, registros {records}.")
        out[column] = converted
    repeated_ids = int(out.duplicated(subset=roles["identifiers"], keep=False).sum())
    ignored = [c for c in df.columns if str(c).strip() not in selected]
    out = out[[SOURCE_RECORD] + selected].reset_index(drop=True)
    report = {
        "before": before,
        "after": quality_report(out.drop(columns=SOURCE_RECORD)),
        "excluded_records": exclusions,
        "unselected_columns": [str(c) for c in ignored],
        "rows_with_repeated_identifiers": repeated_ids,
        "roles": roles,
        "source_record_definition": "Ordinal de registro desde 1 en dataset_interim.csv; no predictor",
        "pending_review": [
            "Unidades, rangos físicos y definición operacional del primer evento de falla",
            "Discrepancias entre tablas y fichas; correspondencia con el archivo original",
            "Réplicas, grupos, dependencia y redundancia; protocolo de partición",
            "Confirmar que todos los predictores estarán disponibles antes del ensayo",
        ],
    }
    return out, report


def curate_file(input_path: Path, schema_path: Path) -> dict:
    """Valida la ingesta y guarda tabla y reporte en una versión nueva."""
    logger = configure_logging("data_quality")
    input_path = project_path(input_path)
    schema_path = project_path(schema_path)
    if not input_path.is_relative_to(INTERIM_DIR.resolve()) or input_path.suffix != ".csv":
        raise ValueError("La entrada debe ser un CSV versionado dentro de data/interim/.")
    if not input_path.is_file():
        raise FileNotFoundError("No existe la tabla intermedia. Ejecute primero src/ingesta.py.")
    manifest_path = input_path.parent / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_hash = sha256_file(input_path)
    if not isinstance(manifest, dict) or manifest.get("interim_sha256") != input_hash:
        raise ValueError("El hash de la tabla intermedia no coincide con el manifiesto.")
    if manifest.get("interim_file") != input_path.relative_to(ROOT).as_posix():
        raise ValueError("La ruta de la tabla no coincide con el manifiesto.")
    schema_bytes = schema_path.read_bytes()
    schema = yaml.safe_load(schema_bytes.decode("utf-8"))
    validate_schema(schema)
    df = load_table(input_path)
    if sha256_file(input_path) != input_hash:
        raise ValueError("La tabla intermedia cambió durante la lectura.")
    if (manifest.get("rows"), manifest.get("columns")) != df.shape:
        raise ValueError("Las dimensiones no coinciden con el manifiesto de ingesta.")
    if manifest.get("column_names") != df.columns.tolist():
        raise ValueError("Los encabezados no coinciden con el manifiesto de ingesta.")
    curated, report = curate(df, schema)
    run_dir = new_run_directory(PROCESSED_DIR)
    output_path = run_dir / "model_table.csv"
    curated.to_csv(output_path, index=False, encoding="utf-8", mode="x")
    report.update({
        "report_version": 1,
        "run_id": run_dir.name,
        "curated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": input_path.relative_to(ROOT).as_posix(),
        "input_sha256": input_hash,
        "ingestion_manifest": manifest,
        "schema": schema,
        "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        "output_file": output_path.relative_to(ROOT).as_posix(),
        "output_sha256": sha256_file(output_path),
        "execution": execution_context(),
    })
    write_json(run_dir / "data_quality_report.json", report)
    logger.info(
        "Curaduría %s: %s -> %s filas; exclusiones=%s; salida=%s",
        run_dir.name, len(df), len(curated), len(report["excluded_records"]), report["output_file"],
    )
    logger.warning("Revisión científica pendiente: %s", "; ".join(report["pending_review"]))
    if report["before"]["duplicated_rows"] or report["rows_with_repeated_identifiers"]:
        logger.warning("Revisar duplicados e identificadores repetidos; no implican réplicas válidas.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CSV de una ingesta versionada")
    parser.add_argument("--schema", type=Path, default=ROOT / "config" / "data_schema.yml")
    args = parser.parse_args()
    logger = configure_logging("data_quality")
    try:
        report = curate_file(args.input, args.schema)
    except (OSError, ValueError, csv.Error, yaml.YAMLError) as exc:
        logger.error("Curaduría detenida: %s", exc)
        parser.exit(1, f"Curaduría detenida: {exc}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
