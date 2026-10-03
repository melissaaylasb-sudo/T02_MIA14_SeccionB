from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = ROOT / "data" / "interim"
LOG_DIR = ROOT / "logs"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Formato no soportado en esta versión preliminar: {suffix}")


def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_DIR / "pipeline.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ingest(input_path: Path) -> tuple[pd.DataFrame, dict]:
    configure_logging()
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = load_table(input_path)
    file_hash = sha256_file(input_path)

    manifest = {
        "source_file": input_path.name,
        "sha256": file_hash,
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.astype(str).tolist(),
    }

    df.to_csv(INTERIM_DIR / "dataset_interim.csv", index=False)
    (INTERIM_DIR / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logging.info("Ingesta completada: %s filas, %s columnas", *df.shape)
    logging.info("SHA256: %s", file_hash)
    return df, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta reproducible de datos experimentales.")
    parser.add_argument("--input", required=True, type=Path, help="Archivo ubicado en data/raw/")
    args = parser.parse_args()
    _, manifest = ingest(args.input)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
