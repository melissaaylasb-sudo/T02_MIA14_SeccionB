"""Lectura de las matrices energéticas del archivo MATLAB, sin ejecutarlo.

Solo se admiten literales numéricos, ``NaN(filas, columnas)`` y asignaciones
de vectores a una fila y un intervalo de columnas. La energía se conserva en
la escala original. El script no declara unidad; el apéndice de Venditti
la documenta en milijulios (mJ), véase docs/trazabilidad_datos.md.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_NAMES = {"Energy_Dissipated": "cycle", "Energy_Fracture": "event"}
_MAX_CELLS = 1_000_000


def _assignments(text: str) -> list[tuple[str, str, str]]:
    """Extrae asignaciones de interés y respeta los separadores del vector."""
    text = re.sub(r"%[^\r\n]*", "", text)
    pattern = re.compile(
        r"\b(Case_n|Energy_Dissipated|Energy_Fracture)\b([^=;\r\n]*)="
    )
    assignments = []
    for match in pattern.finditer(text):
        start = match.end()
        nesting = []
        for position in range(start, len(text)):
            char = text[position]
            if char in "([":
                nesting.append(char)
            elif char in ")]":
                expected = "(" if char == ")" else "["
                if not nesting or nesting.pop() != expected:
                    raise ValueError(f"Delimitadores no válidos en {match[1]}.")
            elif char == ";" and not nesting:
                assignments.append(
                    (match[1], match[2].strip(), text[start:position].strip())
                )
                break
        else:
            raise ValueError(f"Asignación incompleta en {match[1]}.")
    return assignments


def _numbers(literal: str, *, column_vector: bool = False) -> np.ndarray:
    match = re.fullmatch(r"\[([\s\S]*)\]", literal)
    if not match:
        raise ValueError("Se esperaba un vector numérico literal entre corchetes.")
    content = match[1].strip()
    if not column_vector and ";" in content:
        raise ValueError("La asignación energética debe ser un vector fila.")
    token_pattern = rf"(?:{_NUMBER}|NaN)"
    delimiter = r"(?:\s*[,;]\s*|\s+)" if column_vector else r"(?:\s*,\s*|\s+)"
    if not re.fullmatch(rf"{token_pattern}(?:{delimiter}{token_pattern})*", content, re.I):
        raise ValueError("El vector contiene elementos no numéricos o separadores no válidos.")
    if column_vector and ";" in content and any(
        not re.fullmatch(token_pattern, row.strip(), re.I) for row in content.split(";")
    ):
        raise ValueError("Case_n debe ser un vector, no una matriz rectangular.")
    separator = r"[\s,;]+" if column_vector else r"[\s,]+"
    tokens = re.split(separator, content)
    if not content or any(
        not re.fullmatch(rf"(?:{_NUMBER}|NaN)", token, flags=re.IGNORECASE)
        for token in tokens
    ):
        raise ValueError("El vector contiene elementos no numéricos o no admitidos.")
    values = np.asarray([float(token) for token in tokens], dtype=float)
    if np.isinf(values).any():
        raise ValueError("Un valor numérico excede el rango finito admitido.")
    return values


def _case_ids(assignments: list[tuple[str, str, str]]) -> np.ndarray:
    cases = [(lhs, rhs) for name, lhs, rhs in assignments if name == "Case_n"]
    if len(cases) != 1 or cases[0][0]:
        raise ValueError("Case_n debe declararse una sola vez como vector completo.")
    expression = cases[0][1]
    integer_range = re.fullmatch(r"\(\s*(\d+)\s*:\s*(\d+)\s*\)\s*'", expression)
    if integer_range:
        start, stop = map(int, integer_range.groups())
        if (
            start < 1 or stop < start or stop - start + 1 > _MAX_CELLS
            or stop >= np.iinfo(np.int64).max
        ):
            raise ValueError("El intervalo Case_n no es válido.")
        values = np.arange(start, stop + 1, dtype=np.int64)
    else:
        values = _numbers(expression, column_vector=True)
    if (
        len(values) > _MAX_CELLS
        or not np.isfinite(values).all()
        or (values < 1).any()
        or (values >= np.iinfo(np.int64).max).any()
        or (values != np.floor(values)).any()
        or len(np.unique(values)) != len(values)
    ):
        raise ValueError("Case_n debe contener enteros positivos, finitos y únicos.")
    return values.astype(np.int64)


def load_energy_tables(path: str | Path) -> dict[str, pd.DataFrame]:
    """Devuelve energía disipada por ciclo y energía de fractura por evento.

    Las claves son ``Energy_Dissipated`` y ``Energy_Fracture``. Cada tabla
    contiene un índice entero ``Case_n`` y columnas ``cycle_1...`` o
    ``event_1...``. Los valores ausentes permanecen como NaN. No se deduce
    si un NaN significa un ensayo ausente, un ciclo no alcanzado u otra causa.

    Se rechazan dimensiones incompatibles, asignaciones repetidas o fuera
    de límites, longitudes distintas y expresiones que no sean literales.
    Las matrices son resultados posteriores al ensayo y no son predictores
    válidos de la carga de primera falla conocida antes del ensayo.
    """
    assignments = _assignments(Path(path).read_text(encoding="utf-8-sig"))
    case_ids = _case_ids(assignments)
    tables = {}
    for name, prefix in _NAMES.items():
        matrix = None
        written = None
        for current, lhs, rhs in assignments:
            if current != name:
                continue
            if not lhs:
                initializer = re.fullmatch(r"NaN\(\s*(\d+)\s*,\s*(\d+)\s*\)", rhs, re.I)
                if matrix is not None or not initializer:
                    raise ValueError(f"{name} requiere una única inicialización NaN(filas, columnas).")
                rows, columns = map(int, initializer.groups())
                if rows != len(case_ids) or columns < 1 or rows * columns > _MAX_CELLS:
                    raise ValueError(f"Dimensiones no válidas para {name}.")
                matrix = np.full((rows, columns), np.nan, dtype=float)
                written = np.zeros((rows, columns), dtype=bool)
                continue
            if matrix is None:
                raise ValueError(f"{name} debe inicializarse antes de asignar valores.")
            selection = re.fullmatch(r"\(\s*(\d+)\s*,\s*(\d+)\s*(?::\s*(\d+)\s*)?\)", lhs)
            if not selection:
                raise ValueError(f"Índices no admitidos en {name}: {lhs}.")
            row, first, last = selection.groups()
            row, first, last = int(row), int(first), int(last or first)
            if not (1 <= row <= matrix.shape[0] and 1 <= first <= last <= matrix.shape[1]):
                raise ValueError(f"Asignación fuera de límites en {name}.")
            values = _numbers(rhs)
            if len(values) != last - first + 1:
                raise ValueError(f"La longitud del vector no coincide con el intervalo en {name}.")
            if written[row - 1, first - 1:last].any():
                raise ValueError(f"Asignación repetida sobre una celda de {name}.")
            matrix[row - 1, first - 1:last] = values
            written[row - 1, first - 1:last] = True
        if matrix is None:
            raise ValueError(f"Falta la matriz {name}.")
        frame = pd.DataFrame(
            matrix,
            index=pd.Index(case_ids, name="Case_n"),
            columns=[f"{prefix}_{number}" for number in range(1, matrix.shape[1] + 1)],
        )
        frame.attrs.update(source_variable=name, unit=None, measurement_stage="post_test")
        tables[name] = frame
    return tables
