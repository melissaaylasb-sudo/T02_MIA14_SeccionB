"""Extracción segura de la tabla ``samples`` de la campaña MATLAB.

El archivo fuente es texto con asignaciones numéricas. Este módulo no ejecuta
MATLAB ni evalúa código arbitrario: solo reconoce vectores literales, rangos
enteros y ``repmat`` escalar usados por las columnas declaradas en
``samples = table(...)``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _without_comments(text: str) -> str:
    return re.sub(r"%[^\r\n]*", "", text)


def _table_columns(text: str) -> list[str]:
    match = re.search(r"\bsamples\s*=\s*table\s*\((.*?)\)\s*;", text, re.DOTALL)
    if not match:
        raise ValueError("No se encontró la declaración samples = table(...).")
    columns = [item.strip() for item in match.group(1).split(",")]
    if not columns or any(not re.fullmatch(r"[A-Za-z]\w*", item) for item in columns):
        raise ValueError("La declaración de samples contiene nombres no reconocidos.")
    if len(columns) != len(set(columns)):
        raise ValueError("La tabla samples declara columnas repetidas.")
    return columns


def _parse_vector(text: str, name: str) -> np.ndarray:
    literal = re.search(rf"\b{re.escape(name)}\s*=\s*\[(.*?)\]\s*;", text, re.DOTALL)
    if literal:
        tokens = re.findall(rf"(?i)\bNaN\b|{_NUMBER}", literal.group(1))
        if not tokens:
            raise ValueError(f"El vector {name} está vacío.")
        return np.asarray([np.nan if token.lower() == "nan" else float(token) for token in tokens])

    integer_range = re.search(
        rf"\b{re.escape(name)}\s*=\s*\(\s*(-?\d+)\s*:\s*(-?\d+)\s*\)\s*'\s*;",
        text,
    )
    if integer_range:
        start, stop = map(int, integer_range.groups())
        step = 1 if stop >= start else -1
        return np.arange(start, stop + step, step, dtype=float)

    repeated = re.search(
        rf"\b{re.escape(name)}\s*=\s*repmat\s*\(\s*({_NUMBER})\s*,\s*(\d+)\s*,\s*1\s*\)\s*;",
        text,
    )
    if repeated:
        value, rows = repeated.groups()
        return np.repeat(float(value), int(rows))

    raise ValueError(f"No se reconoce la asignación de la columna {name}.")


def load_matlab_samples(path: Path) -> pd.DataFrame:
    """Devuelve la tabla principal y valida longitud, identificador y finitud geométrica."""
    text = _without_comments(path.read_text(encoding="utf-8-sig"))
    columns = _table_columns(text)
    vectors = {name: _parse_vector(text, name) for name in columns}
    lengths = {name: len(values) for name, values in vectors.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Las columnas MATLAB no tienen igual longitud: {lengths}")
    frame = pd.DataFrame(vectors)
    if "Case_n" not in frame or frame["Case_n"].isna().any() or frame["Case_n"].duplicated().any():
        raise ValueError("Case_n debe existir y contener identificadores únicos no nulos.")
    frame["Case_n"] = frame["Case_n"].astype(int)
    return frame

