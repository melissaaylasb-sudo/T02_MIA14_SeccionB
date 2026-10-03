from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim" / "dataset_interim.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
LOG_DIR = ROOT / "logs"


def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_DIR / "data_quality.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def quality_report(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicated_rows": int(df.duplicated().sum()),
        "missing_by_column": {c: int(v) for c, v in df.isna().sum().items()},
        "constant_columns": [
            c for c in df.columns if df[c].nunique(dropna=False) <= 1
        ],
    }


def curate(df: pd.DataFrame) -> pd.DataFrame:
    # Solo transformaciones determinísticas preliminares.
    # No se realizan imputación estadística, escalamiento ni selección
    # supervisada antes de definir las particiones de entrenamiento.
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.drop_duplicates()
    return out


def main() -> None:
    configure_logging()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not INTERIM.exists():
        raise FileNotFoundError(
            "No existe data/interim/dataset_interim.csv. Ejecute primero src/ingesta.py."
        )

    df = pd.read_csv(INTERIM)
    before = quality_report(df)
    curated = curate(df)
    after = quality_report(curated)

    curated.to_csv(PROCESSED_DIR / "model_table.csv", index=False)
    report = {"before": before, "after": after}
    (INTERIM.parent / "data_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logging.info("Curaduría completada: %s -> %s filas", len(df), len(curated))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
