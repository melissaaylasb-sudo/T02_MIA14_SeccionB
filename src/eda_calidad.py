"""Auditoría descriptiva de integridad, ausencia y soporte experimental.

No modifica datos fuente, no imputa y no decide exclusiones estadísticas.
El punto de entrada ejecuta y exporta el cuaderno académico de calidad.
"""

from __future__ import annotations

import argparse
from html import escape
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def compare_tables(left: pd.DataFrame, right: pd.DataFrame, *, key="Case_n", atol=1e-10):
    """Compara por identificador, conservando diferencias de esquema y de ausencia.

    Los tipos enteros/flotantes y el orden de filas pueden variar sin cambiar
    el contenido numérico. Identificadores ambiguos detienen la comparación.
    Devuelve diferencias; una tabla vacía certifica equivalencia al atol dado.
    """
    for frame in (left, right):
        if frame.columns.duplicated().any():
            raise ValueError("Encabezados duplicados: no es posible comparar sin ambigüedad.")
        if key not in frame or frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError("La comparación requiere identificadores únicos y presentes.")
    numeric_ids = [pd.to_numeric(frame[key], errors="coerce") for frame in (left, right)]
    if all(ids.notna().all() for ids in numeric_ids):
        if any(ids.duplicated().any() for ids in numeric_ids):
            raise ValueError("Identificadores ambiguos después de normalizar su representación numérica.")
        left = left.assign(**{key: numeric_ids[0].astype(float)})
        right = right.assign(**{key: numeric_ids[1].astype(float)})
    a, b = left.set_index(key), right.set_index(key)
    records = []
    for column in a.columns.difference(b.columns):
        records.append(("columna_solo_izquierda", "", column, "presente", "ausente"))
    for column in b.columns.difference(a.columns):
        records.append(("columna_solo_derecha", "", column, "ausente", "presente"))
    for identifier in a.index.difference(b.index):
        records.append(("caso_solo_izquierda", identifier, key, "presente", "ausente"))
    for identifier in b.index.difference(a.index):
        records.append(("caso_solo_derecha", identifier, key, "ausente", "presente"))
    shared = a.index.intersection(b.index)
    for column in a.columns.intersection(b.columns):
        x, y = a.loc[shared, column], b.loc[shared, column]
        missing_equal = x.isna() & y.isna()
        xn, yn = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        numeric = xn.notna() & yn.notna()
        equal = missing_equal | (numeric & np.isclose(xn, yn, rtol=0, atol=atol, equal_nan=False))
        equal |= (~numeric & x.notna() & y.notna() & (x.astype(str).to_numpy() == y.astype(str).to_numpy()))
        for identifier in shared[~equal.to_numpy()]:
            records.append(("valor_distinto", identifier, column, x.loc[identifier], y.loc[identifier]))
    return pd.DataFrame(records, columns=["tipo", key, "variable", "izquierda", "derecha"])


def profile_variables(frame: pd.DataFrame) -> pd.DataFrame:
    """Cuenta faltantes, ceros y signos por separado; no convierte cero a nulo."""
    rows = []
    for column in frame:
        series = frame[column]
        numeric = pd.to_numeric(series, errors="coerce")
        present = series.notna()
        values = numeric.dropna()
        finite = values[np.isfinite(values)]
        rows.append({
            "Variable": column,
            "Tipo": str(series.dtype),
            "Disponibles": int(present.sum()),
            "Ausentes": int(series.isna().sum()),
            "Ausencia_%": 100 * series.isna().mean(),
            "Ceros": int((numeric == 0).sum()),
            "Negativos": int((numeric < 0).sum()),
            "No_finitos": int(np.isinf(values).sum()),
            "No_numericos": int((present & numeric.isna()).sum()),
            "Valores_distintos": int(series.nunique(dropna=True)),
            "Minimo": float(finite.min()) if len(finite) else np.nan,
            "Mediana": float(finite.median()) if len(finite) else np.nan,
            "Maximo": float(finite.max()) if len(finite) else np.nan,
        })
    return pd.DataFrame(rows)


def missing_patterns(frame: pd.DataFrame) -> pd.DataFrame:
    """Agrupa vectores de ausencia sin inferir su mecanismo causal."""
    patterns = frame.isna().apply(lambda row: tuple(row.index[row]), axis=1)
    rows = [{"Variables_ausentes": "; ".join(pattern) or "ninguna",
             "Frecuencia": int(frequency), "Proporcion_%": 100 * frequency / len(frame)}
            for pattern, frequency in patterns.value_counts().items()]
    return pd.DataFrame(rows)


def missing_by_group(frame: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Denominadores por familia y variable; los ciclos no añaden especímenes."""
    if not frame.index.equals(groups.index):
        raise ValueError("Los grupos y la tabla deben compartir exactamente el índice.")
    if groups.isna().any():
        raise ValueError("No se admite una familia desconocida en esta auditoría.")
    rows = []
    for group in sorted(groups.unique()):
        selected = frame.loc[groups.eq(group)]
        for column in selected:
            rows.append({"Familia": group, "Variable": column,
                         "Base_familia": len(selected),
                         "Disponibles": int(selected[column].notna().sum()),
                         "Ausentes": int(selected[column].isna().sum()),
                         "Ausencia_%": 100 * selected[column].isna().mean()})
    return pd.DataFrame(rows)


def duplicate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Detecta igualdad de contenido, separada de correlación o redundancia."""
    rows = []
    for first, second in combinations(frame.columns, 2):
        x, y = frame[first], frame[second]
        if ((x == y) | (x.isna() & y.isna())).all():
            rows.append({"Variable_1": first, "Variable_2": second,
                         "Interpretacion": "Contenido idéntico; revisar significado antes de eliminar"})
    return pd.DataFrame(rows, columns=["Variable_1", "Variable_2", "Interpretacion"])


def configuration_audit(frame: pd.DataFrame, geometry: list[str], *, key="Case_n"):
    """Distingue geometría idéntica de repetición de un subconjunto de factores.

    Ninguna agrupación certifica réplicas físicas: faltan metadatos de espécimen.
    """
    rows = []
    settings = ["n_cells_Y", "Fiber_height", "Pivot_height"]
    for factors, level in [(geometry, "geometría registrada completa"),
                           (settings, "combinación parcial de factores")]:
        for _, group in frame.groupby(factors, dropna=False, sort=True):
            if len(group) <= 1:
                continue
            varying = [c for c in geometry if group[c].nunique(dropna=False) > 1]
            rows.append({"Nivel": level, "Casos": ", ".join(group[key].astype(int).astype(str)),
                         "Repeticiones": len(group),
                         "Geometria_que_difiere": ", ".join(varying) or "ninguna",
                         "Replica_confirmada": "No; falta identificación de espécimen y réplica"})
    return pd.DataFrame(rows, columns=["Nivel", "Casos", "Repeticiones", "Geometria_que_difiere", "Replica_confirmada"])


def robust_flags(frame: pd.DataFrame, columns: list[str], *, group=None, key="Case_n"):
    """Banderas IQR/MAD, sin imputación ni eliminación y con MAD=0 explícito."""
    rows = []
    groups = [("global", frame)] if group is None else frame.groupby(group, dropna=False)
    for level, subset in groups:
        for column in columns:
            values = pd.to_numeric(subset[column], errors="raise").dropna()
            if values.empty:
                continue
            q1, q3 = values.quantile([.25, .75])
            median = values.median()
            mad = (values - median).abs().median()
            iqr = q3 - q1
            for index, value in values.items():
                zi = .6744897501960817 * (value - median) / mad if mad > 0 else np.nan
                rows.append({key: subset.loc[index, key], "Variable": column, "Grupo": level,
                             "Valor": value, "IQR": iqr, "MAD": mad,
                             "Bandera_IQR": bool(value < q1 - 1.5 * iqr or value > q3 + 1.5 * iqr),
                             "z_robusto": zi, "Bandera_MAD": bool(abs(zi) > 3.5) if np.isfinite(zi) else pd.NA})
    return pd.DataFrame(rows)


def audit_matlab_literals(path: str | Path, extracted: pd.DataFrame) -> pd.DataFrame:
    """Verificación independiente del léxico y los valores de cada vector samples.

    La ingesta existente reconoce expresiones concretas. Aquí se comprueba que
    cada literal completo solo contenga números/NaN, sin descartar otros tokens.
    No se ejecuta el script original ni código de sus comentarios.
    """
    source = re.sub(r"%[^\r\n]*", "", Path(path).read_text(encoding="utf-8-sig"))
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    atom = rf"(?:{number}|NaN)"
    rows = []
    for column in extracted:
        name = re.escape(column)
        literal = re.search(rf"\b{name}\s*=\s*\[([\s\S]*?)\]\s*;", source)
        interval = re.search(rf"\b{name}\s*=\s*\(\s*(\d+)\s*:\s*(\d+)\s*\)\s*'\s*;", source)
        scalar = re.search(rf"\b{name}\s*=\s*repmat\(\s*({number})\s*,\s*(\d+)\s*,\s*1\s*\)\s*;", source)
        if literal:
            content = literal[1].strip()
            if not re.fullmatch(rf"{atom}(?:(?:\s*[,;]\s*|\s+){atom})*", content, re.I):
                raise ValueError(f"Contenido no numérico en el literal {column}.")
            values = np.array([float(t) for t in re.split(r"[\s,;]+", content)])
            form = "vector literal"
        elif interval:
            start, stop = map(int, interval.groups())
            values = np.arange(start, stop + 1, dtype=float)
            form = "intervalo entero"
        elif scalar:
            values = np.repeat(float(scalar[1]), int(scalar[2]))
            form = "repetición escalar"
        else:
            raise ValueError(f"Asignación no verificable: {column}.")
        observed = extracted[column].to_numpy(dtype=float)
        equivalent = values.shape == observed.shape and np.array_equal(values, observed, equal_nan=True)
        rows.append({"Variable": column, "Expresion": form, "Lexico_valido": True,
                     "Valores_y_NaN_coinciden": bool(equivalent)})
    return pd.DataFrame(rows)


def export_quality_notebook(*, execute=False):
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/01_Ingesta_Curaduria_y_Calidad.ipynb"
    notebook = nbformat.read(path, as_version=4)
    if execute:
        NotebookClient(notebook, timeout=600, kernel_name="python3", allow_errors=False).execute(cwd=str(root))
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
    for cell in notebook.cells:
        if cell.cell_type == "code" and (cell.execution_count is None or any(
            out.output_type == "error" for out in cell.get("outputs", [])
        )):
            raise ValueError("El notebook de calidad debe estar completamente ejecutado.")
    exporter = HTMLExporter()
    exporter.exclude_input = True
    html, _ = exporter.from_notebook_node(notebook)
    html = re.sub(r"<title>.*?</title>", "<title>Calidad y curaduría de datos pantográficos</title>", html, count=1)
    html = html.replace('href="02_EDA_Avanzado.ipynb"', 'href="02_EDA_Avanzado.html"')
    manifest = json.loads((root / "reports/eda_calidad/manifest.json").read_text(encoding="utf-8"))
    descriptions = iter(item["titulo"] + ". " + item["resultado"] for item in manifest["figures"])

    def figure_alt(match):
        tag = match.group(0)
        if "data:image/png" not in tag:
            return tag
        tag = re.sub(r'\s+alt="[^"]*"', "", tag[:-1])
        return tag + ' alt="' + escape(next(descriptions, "Figura de calidad experimental"), quote=True) + '">'

    html = re.sub(r"<img[^>]*>", figure_alt, html)
    css = """<style>
body{background:#f2f5f7!important;color:#213448}main{max-width:1270px;margin:auto;background:white;padding:32px!important}
h1{color:#173e50!important;border-bottom:3px solid #148b82;padding-bottom:18px}
h2{color:#173e50!important;border-top:1px solid #dde6eb;padding-top:22px;margin-top:36px!important}
table{font-size:12px!important;border-collapse:collapse}td,th{padding:8px!important;border-bottom:1px solid #e5e9eb;text-align:left!important}
th{background:#e9f3f1!important}.jp-RenderedHTMLCommon{line-height:1.65!important}.jp-OutputArea-output{overflow-x:auto!important}
.jp-InputPrompt,.jp-OutputPrompt{display:none!important}.quality-nav{background:#eaf4f2;padding:18px;border-radius:6px}.quality-nav ul{columns:2}
@media(max-width:720px){main{padding:12px!important}.quality-nav ul{columns:1}}
</style>"""
    html = html.replace("</head>", css + "</head>")
    headings = re.findall(r'<h2 id="([^"]+)">(.*?)<a class="anchor-link"', html, flags=re.S)
    toc = "".join(f'<li><a href="#{escape(anchor)}">{heading}</a></li>' for anchor, heading in headings)
    nav = '<nav class="quality-nav"><strong>Calidad experimental · auditoría ejecutada</strong><br><a href="../notebooks/01_Ingesta_Curaduria_y_Calidad.ipynb">Notebook</a> · <a href="../docs/informe_curaduria.md">Informe de curaduría</a> · <a href="eda_calidad/tables/14_decisiones.csv">Decisiones CSV</a><details><summary>Contenido</summary><ul>' + toc + '</ul></details></nav>'
    html = html.replace("<main>", "<main>" + nav, 1)
    destination = root / "reports/01_Ingesta_Curaduria_y_Calidad.html"
    destination.write_text(html, encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(export_quality_notebook(execute=args.execute))
