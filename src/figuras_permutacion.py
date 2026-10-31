"""Presentación reproducible de resultados de permutación y modelos fijos.

Lee únicamente los CSV de reports/permutation_importance/tables. No ajusta
modelos, no selecciona hiperparámetros ni altera las tablas de cálculo.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELOS = ("Ridge", "ElasticNet", "SVR_RBF", "RandomForest", "GaussianProcess")
REFERENCIAS = ("Mediana", "Media_por_nivel")
PROTOCOLOS = ("configuracion", "familia")
NOMBRES_MODELO = {
    "Ridge": "Ridge", "ElasticNet": "Elastic Net", "SVR_RBF": "SVR · RBF",
    "RandomForest": "Bosque aleatorio", "GaussianProcess": "Proceso gaussiano",
    "Mediana": "Mediana · referencia", "Media_por_nivel": "Media por nivel Y",
}
NOMBRES_PROTOCOLO = {"configuracion": "Configuraciones", "familia": "Familias Y"}
GEOMETRIA = (
    "n_cells_Y", "Fiber_height", "Fiber_base", "Fiber_total_length", "Fiber_total_volume",
    "Pivot_height", "Pivot_radius", "Pivot_total_number", "Pivot_total_volume", "Sample_total_volume",
)
ETIQUETAS = {
    "n_cells_Y": "Número de celdas en Y",
    "Fiber_height": "Altura de fibra",
    "Fiber_base": "Ancho de fibra",
    "Fiber_total_length": "Longitud total de fibras",
    "Fiber_total_volume": "Volumen declarado de fibras",
    "Pivot_height": "Altura de pivote",
    "Pivot_radius": "Radio de pivote",
    "Pivot_total_number": "Número total de pivotes",
    "Pivot_total_volume": "Volumen declarado de pivotes",
    "Sample_total_volume": "Volumen declarado del espécimen",
    "arquitectura": "Arquitectura: celdas, longitud y pivotes",
    "seccion": "Sección de fibra: altura y ancho",
    "volumenes": "Volúmenes declarados",
    "pivote": "Pivote: altura y radio",
    "arquitectura_volumenes": "Arquitectura y volúmenes",
}
ORDEN_GRUPOS = ("arquitectura", "seccion", "volumenes", "pivote", "arquitectura_volumenes")
COLORES_MODELO = {"Ridge": "#2B6A99", "ElasticNet": "#198477", "SVR_RBF": "#B17B22",
                  "RandomForest": "#8060A5", "GaussianProcess": "#B96064"}
BLOQUE_ABLACION = "arquitectura_volumenes"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _leer(directorio: Path, nombre: str, requeridas: set[str]) -> pd.DataFrame:
    path = directorio / nombre
    if not path.is_file():
        raise FileNotFoundError(f"Falta la tabla calculada: {path}")
    frame = pd.read_csv(path, float_precision="round_trip")
    missing = requeridas - set(frame.columns)
    if missing:
        raise ValueError(f"{nombre}: faltan columnas {sorted(missing)}")
    return frame


def _numero(value: float, decimales: int = 3, signo: bool = False) -> str:
    """Decimal español, sin transformar ni recortar los resultados negativos."""
    if not np.isfinite(value):
        raise ValueError("No se permite presentar una métrica no finita")
    if abs(value) < 0.5 * 10**(-decimales):
        value = 0.0
    text = format(float(value), f"{'+' if signo else ''},.{decimales}f")
    return text.replace(",", "\u202f").replace(".", ",").replace("-", "−")


def _eje_numero(value: float, position=None) -> str:
    return f"{value:g}".replace(".", ",").replace("-", "−")


def _validar_metricas(metricas: pd.DataFrame) -> None:
    keys = ["protocolo", "modelo", "conjunto"]
    if metricas.duplicated(keys).any():
        raise ValueError("Hay métricas repetidas para el mismo protocolo, modelo y conjunto")
    # La diferencia vs Todas puede no estar definida para las referencias.
    if not np.isfinite(metricas[["MAE_N", "MSE_N2", "RMSE_N", "R2"]].to_numpy()).all():
        raise ValueError("Métricas de desempeño no finitas")
    if (metricas[["MAE_N", "MSE_N2", "RMSE_N"]] < 0).any().any() or (metricas.R2 > 1+1e-12).any():
        raise ValueError("Métricas fuera de su dominio numérico")
    if not np.allclose(metricas.RMSE_N**2, metricas.MSE_N2, rtol=1e-10, atol=1e-10):
        raise ValueError("MSE y RMSE no concuerdan")
    for protocol in PROTOCOLOS:
        block = metricas[metricas.protocolo.eq(protocol)]
        for model in MODELOS:
            for subset in ("todas", "prioritarias"):
                if len(block[block.modelo.eq(model) & block.conjunto.eq(subset)]) != 1:
                    raise ValueError(f"Falta {protocol}/{model}/{subset}")
        for model in REFERENCIAS:
            if len(block[block.modelo.eq(model) & block.conjunto.eq("referencia")]) != 1:
                raise ValueError(f"Falta referencia {protocol}/{model}")
        median = block.loc[block.modelo.eq("Mediana") & block.conjunto.eq("referencia"), "MAE_N"].iloc[0]
        for row in block.itertuples():
            if not np.isclose(row.delta_MAE_vs_mediana_N, row.MAE_N-median, rtol=1e-10, atol=1e-10):
                raise ValueError("La diferencia frente a mediana no corresponde al mismo protocolo")
            if row.modelo in MODELOS:
                full = block.loc[block.modelo.eq(row.modelo) & block.conjunto.eq("todas"), "MAE_N"].iloc[0]
                if not np.isclose(row.delta_MAE_vs_todas_N, row.MAE_N-full, rtol=1e-10, atol=1e-10):
                    raise ValueError("La diferencia de subconjuntos no corresponde al mismo modelo/protocolo")


def _limpiar_eje(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=5)
    ax.grid(axis="x", color="#E6EAED", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(FuncFormatter(_eje_numero))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))


def _textos_figura(fig: plt.Figure) -> list[str]:
    return [artist.get_text() for artist in fig.findobj(matplotlib.text.Text) if artist.get_text()]


def _validar_render(fig: plt.Figure, nombre: str) -> list[str]:
    fig.canvas.draw()
    texts = _textos_figura(fig)
    joined = "\n".join(texts)
    for pattern in (r"\b26\s+casos\b", r"\bfuente\s*:", r"\bshap\b"):
        if re.search(pattern, joined, flags=re.IGNORECASE):
            raise ValueError(f"Texto no permitido en {nombre}: {pattern}")
    if not texts:
        raise ValueError("La figura carece de texto verificable")
    return texts


def _tabla_metricas(metricas: pd.DataFrame, prioritarias: bool) -> tuple[plt.Figure, list[dict]]:
    modelos = MODELOS if prioritarias else MODELOS + REFERENCIAS
    rows = []
    for protocol in PROTOCOLOS:
        for model in modelos:
            subset = "referencia" if model in REFERENCIAS else "prioritarias" if prioritarias else "todas"
            row = metricas[metricas.protocolo.eq(protocol) & metricas.modelo.eq(model) & metricas.conjunto.eq(subset)].iloc[0]
            delta_col = "delta_MAE_vs_todas_N" if prioritarias else "delta_MAE_vs_mediana_N"
            raw = {"MAE_N": float(row.MAE_N), "MSE_N2": float(row.MSE_N2), "RMSE_N": float(row.RMSE_N),
                   "R2": float(row.R2), "delta_MAE_N": float(row[delta_col])}
            cells = [NOMBRES_PROTOCOLO[protocol], NOMBRES_MODELO[model], _numero(raw["MAE_N"]),
                     _numero(raw["MSE_N2"]), _numero(raw["RMSE_N"]), _numero(raw["R2"]), _numero(raw["delta_MAE_N"], signo=True)]
            rows.append({"protocolo": protocol, "modelo": model, "conjunto": subset,
                         "delta_column": delta_col, "valores": raw, "celdas": cells})
    header = "#66518C" if prioritarias else "#326F53"
    tint = "#F0EBF7" if prioritarias else "#EAF4ED"
    pale = "#F8F5FC" if prioritarias else "#F5FAF6"
    n = len(rows)
    fig, ax = plt.subplots(figsize=(14.8, 2.4 + n * 0.40))
    ax.set_axis_off()
    title = "Modelos con variables prioritarias" if prioritarias else "Modelos con todas las variables"
    fig.text(0.025, 0.955, title, ha="left", va="top", fontsize=18, weight="bold", color=header)
    subtitle = ("Cuatro entradas seleccionadas dentro de cada entrenamiento; parámetros predictivos fijos."
                if prioritarias else "Geometría original y dos referencias; parámetros predictivos fijos.")
    fig.text(0.025, 0.900, subtitle, ha="left", va="top", fontsize=11.5, color="#4F5963")
    labels = ["Validación", "Modelo", "MAE [N]", "MSE [N²]", "RMSE [N]", "R²", "ΔMAE [N]"]
    widths = [0.165, 0.235, 0.112, 0.144, 0.117, 0.105, 0.122]
    table = ax.table(cellText=[r["celdas"] for r in rows], colLabels=labels, cellLoc="right", colLoc="right",
                     colWidths=widths, bbox=[0, 0.09, 1, 0.77])
    table.auto_set_font_size(False)
    table.set_fontsize(11.2)
    for (i, k), cell in table.get_celld().items():
        cell.set_linewidth(0)
        cell.set_edgecolor("none")
        cell.PAD = 0.075
        if i == 0:
            cell.set_facecolor(header)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(tint if (i-1) % 2 == 0 else pale)
            cell.get_text().set_color("#26343A")
        if k < 2:
            cell.get_text().set_ha("left")
    reference = "todas las variables del mismo modelo" if prioritarias else "la mediana del entrenamiento"
    fig.text(0.025, 0.065, f"ΔMAE respecto a {reference}: valores negativos indican menor error.", fontsize=10.5, color="#4F5963")
    fig.text(0.025, 0.032, "Configuraciones: combinación geométrica nominal. Familias Y: nivel celular reservado completo.", fontsize=10.5, color="#4F5963")
    fig.subplots_adjust(left=0.025, right=0.975, top=0.94, bottom=0.08)
    # Verify actual artist strings, not just the table inputs.
    for i, record in enumerate(rows, 1):
        displayed = [table[i, k].get_text().get_text() for k in range(7)]
        if displayed != record["celdas"]:
            raise AssertionError("La tabla visual no coincide con los valores formateados")
    return fig, rows


def _panel_importancia(ax: plt.Axes, selected: pd.DataFrame, order: list[str], model: str,
                       labels: bool, *, groups: bool = False) -> None:
    if selected.variable_grupo.duplicated().any() or set(selected.variable_grupo) != set(order):
        raise ValueError(f"Cobertura de importancia incompleta para {model}")
    selected = selected.set_index("variable_grupo").loc[order]
    mean = selected.delta_MAE_media_N.to_numpy(dtype=float)
    sd = selected.delta_MAE_DE_N.to_numpy(dtype=float)
    if not np.isfinite(np.r_[mean, sd]).all() or (sd < 0).any():
        raise ValueError("Media o dispersión de permutación inválida")
    y = np.arange(len(order))
    ax.errorbar(mean, y, xerr=sd, fmt="o", markersize=5.2, color=COLORES_MODELO[model],
                ecolor=COLORES_MODELO[model], elinewidth=1.8, capsize=3, capthick=1.3)
    ax.axvline(0, color="#485662", linewidth=1, linestyle=(0, (3, 3)), zorder=0)
    _limpiar_eje(ax)
    ax.set_title(fill(NOMBRES_MODELO[model], 18), fontsize=12, weight="bold", color=COLORES_MODELO[model], pad=12)
    ax.set_yticks(y, [fill(ETIQUETAS.get(c, str(selected.loc[c, "etiqueta"])), 27) for c in order] if labels else [""]*len(y))
    ax.set_ylim(len(order)-0.5, -0.5)
    extent = np.r_[mean-sd, mean+sd, 0.0]
    span = float(extent.max()-extent.min()) or 1.0
    ax.set_xlim(float(extent.min()-span*0.12), float(extent.max()+span*0.15))
    ax.set_xlabel("Cambio en MAE [N]", fontsize=10)
    if groups:
        ax.tick_params(axis="y", labelsize=9.7)


def _importancia_individual(data: pd.DataFrame, protocol: str) -> plt.Figure:
    frame = data[data.protocolo.eq(protocol) & data.conjunto.eq("todas") & data.tipo.eq("variable")]
    fig, axes = plt.subplots(1, 5, figsize=(18.0, 7.7))
    for ax, model in zip(axes, MODELOS):
        _panel_importancia(ax, frame[frame.modelo.eq(model)], list(GEOMETRIA), model, ax is axes[0])
    fig.text(0.025, 0.965, f"Importancia individual por permutación · {NOMBRES_PROTOCOLO[protocol]}", fontsize=18, weight="bold", color="#243A4B", va="top")
    fig.text(0.025, 0.910, "Todas las entradas originales; modelos conservados de la validación externa.", fontsize=11.5, color="#4F5963", va="top")
    fig.text(0.025, 0.058, "Puntos: cambio medio en MAE. Barras: ±1 desviación estándar entre permutaciones (variabilidad Monte Carlo).", fontsize=10.6, color="#4F5963")
    fig.text(0.025, 0.024, "Escalas horizontales propias por modelo. Un valor negativo significa menor error tras perturbar; no se recorta a cero.", fontsize=10.6, color="#4F5963")
    fig.subplots_adjust(left=0.225, right=0.98, top=0.80, bottom=0.16, wspace=0.36)
    return fig


def _importancia_grupos(data: pd.DataFrame) -> plt.Figure:
    frame = data[data.conjunto.eq("todas") & data.tipo.eq("grupo")]
    present = set(frame.variable_grupo)
    order = [name for name in ORDEN_GRUPOS if name in present]
    if not order or set(order) != present:
        raise ValueError("Los grupos deben estar declarados antes de presentar resultados")
    fig, axes = plt.subplots(2, 5, figsize=(18.0, 10.7))
    for row, protocol in enumerate(PROTOCOLOS):
        block = frame[frame.protocolo.eq(protocol)]
        for col, model in enumerate(MODELOS):
            _panel_importancia(axes[row, col], block[block.modelo.eq(model)], order, model, col == 0, groups=True)
    fig.text(0.025, 0.976, "Importancia de grupos de variables relacionadas", fontsize=18, weight="bold", color="#243A4B", va="top")
    fig.text(0.025, 0.938, "Las columnas del grupo comparten donante; la perturbación puede crear combinaciones fuera del soporte observado.", fontsize=11, color="#4F5963", va="top")
    fig.text(0.025, 0.858, "Configuraciones", fontsize=12.5, weight="bold", color="#243A4B")
    fig.text(0.025, 0.452, "Familias Y", fontsize=12.5, weight="bold", color="#243A4B")
    fig.text(0.025, 0.047, "Puntos: cambio medio en MAE. Barras: ±1 desviación estándar entre permutaciones; no son intervalos de confianza.", fontsize=10.5, color="#4F5963")
    fig.text(0.025, 0.020, "Escalas horizontales propias por modelo y protocolo. Los grupos superpuestos son análisis de sensibilidad distintos.", fontsize=10.5, color="#4F5963")
    fig.subplots_adjust(left=0.225, right=0.98, top=0.82, bottom=0.13, wspace=0.36, hspace=0.68)
    return fig


def _frecuencias(data: pd.DataFrame) -> plt.Figure:
    if data.duplicated(["protocolo", "modelo", "variable"]).any():
        raise ValueError("Frecuencias de selección repetidas")
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 8.3))
    cmap = colors.LinearSegmentedColormap.from_list("seleccion_suave", ["#F7F8FA", "#C4D9E9", "#39749A"])
    for ax, protocol in zip(axes, PROTOCOLOS):
        block = data[data.protocolo.eq(protocol)]
        matrix = np.zeros((len(GEOMETRIA), len(MODELOS)))
        annotations = {}
        for col, model in enumerate(MODELOS):
            rows = block[block.modelo.eq(model)].set_index("variable")
            if set(rows.index) != set(GEOMETRIA):
                raise ValueError("La frecuencia debe incluir también las variables nunca seleccionadas")
            if rows.n_folds.nunique() != 1 or rows.n_selecciones.sum() != 4 * rows.n_folds.iloc[0]:
                raise ValueError("La selección no corresponde a cuatro variables por entrenamiento")
            for row, name in enumerate(GEOMETRIA):
                s = rows.loc[name]
                if s.n_folds <= 0 or not 0 <= s.n_selecciones <= s.n_folds or not 0 <= s.frecuencia <= 1:
                    raise ValueError("Frecuencia o recuento de selección fuera de rango")
                if not np.isclose(s.frecuencia, s.n_selecciones/s.n_folds):
                    raise ValueError("Frecuencia incoherente con sus recuentos")
                matrix[row, col] = s.frecuencia
                annotations[row, col] = f"{100*s.frecuencia:.0f}%\n{int(s.n_selecciones)}/{int(s.n_folds)}"
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for (row, col), label in annotations.items():
            ax.text(col, row, label, ha="center", va="center", fontsize=9.1, color="white" if matrix[row, col] >= 0.7 else "#243A4B")
        ax.set_title(NOMBRES_PROTOCOLO[protocol], fontsize=13, weight="bold", pad=16)
        ax.set_xticks(np.arange(len(MODELOS)), [fill(NOMBRES_MODELO[m], 13) for m in MODELOS], fontsize=9.5)
        ax.set_yticks(np.arange(len(GEOMETRIA)), [fill(ETIQUETAS[v], 29) for v in GEOMETRIA] if ax is axes[0] else [""]*len(GEOMETRIA))
        ax.tick_params(length=0)
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.text(0.025, 0.970, "Frecuencia de selección de variables prioritarias", fontsize=18, weight="bold", color="#243A4B", va="top")
    fig.text(0.025, 0.917, "Cuatro entradas elegidas dentro de cada entrenamiento externo; se muestran todos los modelos y variables.", fontsize=11.3, color="#4F5963", va="top")
    fig.text(0.025, 0.035, "Cada celda muestra el porcentaje y el número de selecciones sobre los pliegues correspondientes. No representa probabilidad causal.", fontsize=10.3, color="#4F5963")
    fig.subplots_adjust(left=0.24, right=0.96, top=0.81, bottom=0.16, wspace=0.12)
    return fig


def _ablacion(data: pd.DataFrame) -> plt.Figure:
    block = data[data.bloque.eq(BLOQUE_ABLACION)]
    if block.duplicated(["protocolo", "modelo"]).any():
        raise ValueError("Ablación repetida")
    fig, ax = plt.subplots(figsize=(12.2, 6.6))
    y = np.arange(len(MODELOS))
    palette = {"configuracion": "#3A7B9F", "familia": "#A16C98"}
    for k, protocol in enumerate(PROTOCOLOS):
        rows = block[block.protocolo.eq(protocol)].set_index("modelo")
        if set(rows.index) != set(MODELOS):
            raise ValueError(f"Falta ablación predeclarada {BLOQUE_ABLACION}/{protocol}")
        rows = rows.loc[list(MODELOS)]
        if not np.allclose(rows.MAE_ablacion_N-rows.MAE_todas_N, rows.delta_MAE_ablacion_N, rtol=1e-10, atol=1e-10):
            raise ValueError("La diferencia de ablación no coincide con los MAE")
        ax.plot(rows.delta_MAE_ablacion_N, y+(k-.5)*.20, "o", markersize=7, color=palette[protocol], label=NOMBRES_PROTOCOLO[protocol])
    ax.axvline(0, color="#485662", linewidth=1, linestyle=(0, (3, 3)))
    _limpiar_eje(ax)
    ax.set_yticks(y, [NOMBRES_MODELO[m] for m in MODELOS])
    ax.set_ylim(len(MODELOS)-.5, -.5)
    ax.set_xlabel("ΔMAE al retirar el bloque [N]", fontsize=11)
    ax.legend(frameon=False, loc="best", fontsize=10.5)
    fig.text(0.025, 0.960, "Ablación del bloque de arquitectura y volúmenes", fontsize=17, weight="bold", color="#243A4B", va="top")
    fig.text(0.025, 0.895, "Comparación emparejada con todas las entradas del mismo modelo y protocolo.", fontsize=11.3, color="#4F5963", va="top")
    fig.text(0.025, 0.075, "Reajuste con parámetros fijos tras retirar el bloque predeclarado; un cambio negativo indica menor error.", fontsize=10.6, color="#4F5963")
    fig.text(0.025, 0.035, "Diferencias descriptivas de validación: no se representan como incertidumbre de permutación ni como intervalos de confianza.", fontsize=10.2, color="#4F5963")
    fig.subplots_adjust(left=0.22, right=0.97, top=0.78, bottom=0.22)
    return fig


def generar_figuras(root: Path | str = ROOT, output: Path | str | None = None) -> list[str]:
    """Exporta siete figuras PNG/SVG; output reemplaza solo la carpeta de figuras.

    Devuelve rutas absolutas. El manifiesto conserva hashes de CSV, textos
    realmente renderizados y celdas numéricas de las dos tablas visuales.
    """
    root = Path(root).resolve()
    tables = root / "reports/permutation_importance/tables"
    figures = Path(output).resolve() if output is not None else root / "reports/permutation_importance/figures"
    paths = {
        "metricas": tables / "metricas_resumen.csv",
        "importancia": tables / "importancia_resumen.csv",
        "frecuencias": tables / "frecuencias_seleccion.csv",
        "ablacion": tables / "ablacion_bloque.csv",
    }
    metricas = _leer(tables, paths["metricas"].name, {"protocolo", "modelo", "conjunto", "MAE_N", "MSE_N2", "RMSE_N", "R2", "delta_MAE_vs_mediana_N", "delta_MAE_vs_todas_N"})
    importance = _leer(tables, paths["importancia"].name, {"protocolo", "modelo", "conjunto", "tipo", "variable_grupo", "etiqueta", "delta_MAE_media_N", "delta_MAE_DE_N", "n_repeticiones"})
    frequency = _leer(tables, paths["frecuencias"].name, {"protocolo", "modelo", "variable", "etiqueta", "n_selecciones", "n_folds", "frecuencia"})
    ablation = _leer(tables, paths["ablacion"].name, {"protocolo", "modelo", "bloque", "MAE_todas_N", "MAE_ablacion_N", "delta_MAE_ablacion_N"})
    _validar_metricas(metricas)
    figures.mkdir(parents=True, exist_ok=True)
    artifacts, audit = [], []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.titleweight": "bold",
                         "svg.fonttype": "none", "axes.unicode_minus": True, "figure.facecolor": "white"}):
        jobs = [
            ("01_permutacion_individual_configuraciones", _importancia_individual, (importance, "configuracion")),
            ("02_permutacion_individual_familias", _importancia_individual, (importance, "familia")),
            ("03_permutacion_grupos", _importancia_grupos, (importance,)),
            ("04_frecuencia_seleccion", _frecuencias, (frequency,)),
            ("05_ablacion_bloque", _ablacion, (ablation,)),
            ("18_modelos_todas_variables", _tabla_metricas, (metricas, False)),
            ("19_modelos_variables_prioritarias", _tabla_metricas, (metricas, True)),
        ]
        for stem, builder, args in jobs:
            built = builder(*args)
            fig, cells = built if isinstance(built, tuple) else (built, None)
            texts = _validar_render(fig, stem)
            files = []
            for extension in ("png", "svg"):
                path = figures / f"{stem}.{extension}"
                fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.16, facecolor="white")
                files.append({"path": str(path), "sha256": _sha(path)})
                artifacts.append(str(path))
            audit.append({"figure": stem, "files": files, "rendered_text": texts, "numeric_table": cells})
            plt.close(fig)
    manifest = {"module": str(Path(__file__).resolve()), "module_sha256": _sha(Path(__file__)),
                "inputs": {str(path): _sha(path) for path in paths.values()}, "figures": audit,
                "checks": {"numeric_tables_match_csv": True, "delta_reference_checked": True,
                           "rendered_text_checked": True, "figures_count": len(audit)},
                "interpretation": "Error bars show Monte Carlo permutation dispersion; no population confidence intervals or retrospective winner selection"}
    (figures / "manifest_figuras.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return artifacts


if __name__ == "__main__":
    print(json.dumps(generar_figuras(), ensure_ascii=False, indent=2))
