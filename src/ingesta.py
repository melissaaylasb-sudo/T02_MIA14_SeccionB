"""Ingesta preliminar: originales inmutables y una salida trazable por ejecución."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
LOG_DIR = ROOT / "logs"


def project_path(path: Path) -> Path:
    """Resuelve rutas relativas desde el proyecto, independientemente del cwd."""
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    """Calcula el hash por bloques sin modificar ni cargar el archivo completo."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_columns(columns) -> list[str]:
    """Rechaza encabezados vacíos o ambiguos después de retirar espacios extremos."""
    names = ["" if pd.isna(c) else str(c).strip() for c in columns]
    if not names or any(not c for c in names):
        raise ValueError("La tabla tiene encabezados vacíos.")
    if len(names) != len(set(names)):
        raise ValueError("Hay encabezados repetidos o que colisionan al retirar espacios.")
    return names


def load_table(path: Path, *, sheet: str | int = 0) -> pd.DataFrame:
    """Lee CSV UTF-8 o XLSX conservando texto; solo celdas vacías son faltantes.

    No recupera ceros iniciales que Excel conserve únicamente como formato visual.
    No ejecuta fórmulas ni interpreta archivos MATLAB o PDF.
    """
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo: {path.name}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.reader(fh, strict=True)
            header = next(reader, [])
            validate_columns(header)
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise ValueError(f"Número de campos inconsistente en registro CSV {row_number}.")
        df = pd.read_csv(
            path, dtype="string", encoding="utf-8-sig", keep_default_na=False, na_values=[""]
        )
    elif suffix == ".xlsx":
        table = pd.read_excel(
            path, sheet_name=sheet, header=None, dtype="string",
            keep_default_na=False, na_values=[""], engine="openpyxl",
        )
        if table.empty:
            raise ValueError("La hoja Excel está vacía.")
        validate_columns(table.iloc[0].tolist())
        df = table.iloc[1:].reset_index(drop=True)
        df.columns = table.iloc[0].tolist()
    else:
        raise ValueError(f"Formato no soportado: {suffix}. Use CSV o XLSX.")
    validate_columns(df.columns)
    if df.empty:
        raise ValueError("La tabla no contiene registros.")
    return df


def configure_logging(name: str = "pipeline") -> logging.Logger:
    """Crea un logger por etapa, sin interferir con otros módulos o notebooks."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"pantographic.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    destination = (LOG_DIR / f"{name}.log").resolve()
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            if Path(handler.baseFilename) == destination:
                return logger
            logger.removeHandler(handler)
            handler.close()
    handler = logging.FileHandler(destination, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s UTC | %(levelname)s | %(message)s")
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def execution_context() -> dict:
    """Registra versiones y estado Git; ausencia de Git no bloquea la ingesta."""
    packages = {}
    for package in ("pandas", "numpy", "scipy", "scikit-learn", "openpyxl", "PyYAML", "joblib"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    git = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True, timeout=10,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            capture_output=True, text=True, timeout=10,
        )
        git = {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        pass
    return {"python": platform.python_version(), "dependencies": packages, "git": git}


def new_run_directory(parent: Path) -> Path:
    """Reserva una versión nueva; nunca reemplaza una ejecución previa."""
    parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = parent / run_id
    destination.mkdir(exist_ok=False)
    return destination


def write_json(path: Path, payload: dict) -> None:
    """Escribe JSON UTF-8 exclusivo, sin valores NaN no estándar."""
    with path.open("x", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, allow_nan=False)
        fh.write("\n")


def ingest(input_path: Path, *, sheet: str | int = 0) -> tuple[pd.DataFrame, dict]:
    """Versiona una fuente local en raw y devuelve la tabla y su manifiesto."""
    logger = configure_logging()
    input_path = project_path(input_path)
    if not input_path.is_relative_to(RAW_DIR.resolve()):
        raise ValueError("La fuente debe estar dentro de data/raw/.")
    if not input_path.is_file():
        raise FileNotFoundError(f"No existe el archivo: {input_path.name}")
    file_hash = sha256_file(input_path)
    df = load_table(input_path, sheet=sheet)
    if sha256_file(input_path) != file_hash:
        raise ValueError("La fuente cambió durante la lectura. Repita con una versión estable.")
    run_dir = new_run_directory(INTERIM_DIR)
    table_path = run_dir / "dataset_interim.csv"
    df.to_csv(table_path, index=False, encoding="utf-8", mode="x")
    manifest = {
        "manifest_version": 1,
        "run_id": run_dir.name,
        "source_file": input_path.relative_to(ROOT).as_posix(),
        "sha256": file_hash,
        "source_modified_at_utc": datetime.fromtimestamp(
            input_path.stat().st_mtime, timezone.utc
        ).isoformat(),
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.astype(str).tolist(),
        "reader": {
            "format": input_path.suffix.lower(),
            "sheet": sheet if input_path.suffix.lower() == ".xlsx" else None,
            "missing_values": [""],
            "dtype": "string",
        },
        "interim_file": table_path.relative_to(ROOT).as_posix(),
        "interim_sha256": sha256_file(table_path),
        "execution": execution_context(),
    }
    write_json(run_dir / "dataset_manifest.json", manifest)
    logger.info(
        "Ingesta %s: %s filas, %s columnas; fuente=%s; SHA256=%s; salida=%s",
        run_dir.name, *df.shape, manifest["source_file"], file_hash, manifest["interim_file"],
    )
    return df, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Archivo dentro de data/raw/")
    parser.add_argument("--sheet", help="Nombre de hoja XLSX; por defecto se lee la primera")
    args = parser.parse_args()
    logger = configure_logging()
    try:
        if args.sheet is not None and args.input.suffix.lower() != ".xlsx":
            raise ValueError("--sheet solo se admite para XLSX.")
        _, manifest = ingest(args.input, sheet=args.sheet if args.sheet is not None else 0)
    except (OSError, ValueError, ImportError, csv.Error, BadZipFile) as exc:
        logger.error("Ingesta detenida: %s", exc)
        parser.exit(1, f"Ingesta detenida: {exc}\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
