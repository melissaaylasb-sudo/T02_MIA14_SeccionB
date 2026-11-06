"""Figuras del benchmark de primera falla, construidas desde sus resultados CSV."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELOS = ("SVR_RBF", "XGBoost", "CatBoost", "Lineal_multiple", "KNN", "Arbol_decision")
PROTOCOLOS = ("configuracion", "familia")
PROTO = {"configuracion": "Configuración nominal retenida", "familia": "Nivel celular Y retenido"}
NOMBRES = {
    "SVR_RBF": "SVR con núcleo RBF",
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost",
    "Lineal_multiple": "Lineal múltiple",
    "KNN": "K vecinos",
    "Arbol_decision": "Árbol de decisión",
    "Dummy_mediana": "Mediana de entrenamiento",
    "Media_por_nivel": "Media por nivel Y",
    "Ridge_historico": "Ridge histórico",
    "Tuning_rango_1": "Primera del cribado",
    "Tuning_rango_2": "Segunda del cribado",
    "Selector_tuning": "Selector interno",
}
COLORES = dict(zip(MODELOS, ("#2C699A", "#A74930", "#7D4E8D", "#177967", "#A56612", "#506080")))
AZUL, VERDE, GRIS = "#2C699A", "#247654", "#54616C"
ARCHIVOS = {
    "metricas": "metricas_resumen.csv",
    "oof": "predicciones_oof.csv",
    "folds": "metricas_folds.csv",
    "busqueda": "busqueda_interna.csv",
    "cribado": "cribado_interno.csv",
    "seleccion": "seleccion_tuning.csv",
}
TITULOS = (
    "Error absoluto medio de los seis modelos iniciales",
    "Raíz del error cuadrático medio de los seis modelos iniciales",
    "Error de ajuste y error de validación externa",
    "Carga de primera falla observada y predicha",
    "Distribución del error absoluto por espécimen",
    "Residuos del modelo con menor MAE inicial y de dos candidatos a tuning",
    "Diferencia de error frente a Ridge histórico",
    "Evolución de la búsqueda dentro de cada entrenamiento",
    "Efecto del ajuste de hiperparámetros sobre los mismos especímenes",
    "Variación del error entre particiones externas",
)
STEMS = (
    "01_mae",
    "02_rmse",
    "03_entrenamiento_validacion",
    "04_observado_predicho",
    "05_errores_absolutos",
    "06_residuales_candidatos",
    "07_diferencia_baseline",
    "08_busqueda_interna",
    "09_inicial_optimizada",
    "10_estabilidad_particiones",
)


def leer_resultados(root: Path = ROOT) -> dict[str, pd.DataFrame]:
    """Carga las seis tablas de auditoría. Comprueba claves OOF y cobertura común."""
    tables = Path(root) / "reports/benchmark_regresion/tables"
    data = {key: pd.read_csv(tables / name) for key, name in ARCHIVOS.items()}
    oof = data["oof"]
    if oof.duplicated(["protocolo", "procedimiento", "Case_n"]).any():
        raise ValueError("Predicciones externas repetidas por protocolo, procedimiento y caso")
    for proto in PROTOCOLOS:
        selected = oof[oof.protocolo.eq(proto)]
        pivot = selected.pivot(index="Case_n", columns="procedimiento", values="predicho_N")
        if pivot[list(MODELOS)].isna().any().any():
            raise ValueError(f"La cobertura externa de los seis modelos difiere en {proto}")
    return data


def _filas(df: pd.DataFrame, proto: str, modelos: tuple[str, ...] = MODELOS) -> pd.DataFrame:
    return df[df.protocolo.eq(proto) & df.procedimiento.isin(modelos)]


def _estilo(ax: plt.Axes, axis: str = "x") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="#E0E6E7", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=5)


def _encabezado(fig: plt.Figure, numero: int, nota: str) -> None:
    fig.suptitle(
        TITULOS[numero - 1],
        x=0.025,
        y=0.986,
        ha="left",
        va="top",
        fontsize=16,
        weight="bold",
        color="#203D40",
    )
    fig.text(0.025, 0.015, nota, ha="left", va="bottom", fontsize=8.7, color="#47575E")


def _numero(x: float, dec: int = 1) -> str:
    return f"{x:.{dec}f}".replace(".", ",")


def _barras(data: dict, metrica: str, numero: int) -> plt.Figure:
    metricas = data["metricas"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.65), sharex=True, sharey=True)
    ymax = max(_filas(metricas, p)[metrica].max() for p in PROTOCOLOS)
    y = np.arange(len(MODELOS))
    for ax, p in zip(axes, PROTOCOLOS):
        rows = _filas(metricas, p).set_index("procedimiento").loc[list(MODELOS)]
        bars = ax.barh(y, rows[metrica], height=0.56, color=[COLORES[m] for m in MODELOS])
        ax.bar_label(bars, labels=[_numero(v) for v in rows[metrica]], padding=4, fontsize=9)
        ax.set(
            title=PROTO[p],
            xlabel=f"{metrica.split('_')[0]} [N]",
            xlim=(0, ymax * 1.19),
            yticks=y,
            yticklabels=[NOMBRES[m] for m in MODELOS],
            ylim=(5.55, -0.55),
        )
        _estilo(ax)
    _encabezado(
        fig,
        numero,
        "Error fuera de muestra, calculado sobre las predicciones individuales. Un valor menor indica menor error.",
    )
    fig.subplots_adjust(left=0.17, right=0.975, top=0.80, bottom=0.16, wspace=0.12)
    return fig


def _entrenamiento(data: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    y = np.arange(len(MODELOS))
    for ax, proto in zip(axes, PROTOCOLOS):
        rows = (
            _filas(data["folds"], proto)
            .groupby("procedimiento")[["MAE_train_N", "MAE_validacion_N"]]
            .mean()
            .loc[list(MODELOS)]
        )
        ax.hlines(y, rows.MAE_train_N, rows.MAE_validacion_N, color="#AEBBC0", linewidth=1.5)
        ax.scatter(rows.MAE_train_N, y, color=AZUL, s=35, label="Entrenamiento", zorder=3)
        ax.scatter(
            rows.MAE_validacion_N,
            y,
            color=VERDE,
            marker="D",
            s=32,
            label="Validación externa",
            zorder=3,
        )
        ax.set(
            title=PROTO[proto],
            xlabel="MAE medio por pliegue [N]",
            yticks=y,
            yticklabels=[NOMBRES[m] for m in MODELOS],
            ylim=(5.6, -0.6),
        )
        _estilo(ax)
    axes[0].set_xlim(left=0)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.56, 0.08),
        ncol=2,
        frameon=False,
    )
    _encabezado(
        fig,
        3,
        "Cada pliegue pesa lo mismo en ambas series. La brecha también recoge el cambio en la geometría evaluada.",
    )
    fig.subplots_adjust(left=0.17, right=0.975, top=0.82, bottom=0.23, wspace=0.12)
    return fig


def _observado(data: dict) -> plt.Figure:
    rows = data["oof"][data["oof"].procedimiento.isin(MODELOS)]
    fig, axes = plt.subplots(2, 3, figsize=(10.3, 7.1), sharex=True, sharey=True)
    low = min(rows.observado_N.min(), rows.predicho_N.min())
    high = max(rows.observado_N.max(), rows.predicho_N.max())
    pad = 0.04 * (high - low)
    for ax, m in zip(axes.flat, MODELOS):
        ax.plot(
            [low - pad, high + pad],
            [low - pad, high + pad],
            color="#899396",
            linestyle="--",
            linewidth=1,
            zorder=1,
        )
        for proto, marker, col in zip(PROTOCOLOS, ("o", "x"), (AZUL, "#B0573A")):
            q = rows[rows.procedimiento.eq(m) & rows.protocolo.eq(proto)]
            ax.scatter(
                q.observado_N,
                q.predicho_N,
                color=col,
                marker=marker,
                s=22,
                alpha=0.82,
                label=PROTO[proto],
                zorder=2,
            )
        ax.set(
            title=NOMBRES[m],
            xlim=(low - pad, high + pad),
            ylim=(low - pad, high + pad),
            aspect="equal",
        )
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        _estilo(ax, "both")
    fig.supxlabel("Carga de primera falla observada [N]", y=0.105, fontsize=10)
    fig.supylabel("Carga de primera falla predicha [N]", x=0.032, fontsize=10)
    fig.legend(
        *axes.flat[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.53, 0.036),
        frameon=False,
    )
    _encabezado(
        fig,
        4,
        "Cada punto pertenece a un espécimen retenido. La diagonal representa predicción igual a observación.",
    )
    fig.subplots_adjust(left=0.10, right=0.97, top=0.89, bottom=0.19, hspace=0.28, wspace=0.18)
    return fig


def _distribucion(data: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharex=True, sharey=True)
    for ax, proto in zip(axes, PROTOCOLOS):
        q = _filas(data["oof"], proto)
        for pos, m in enumerate(MODELOS):
            val = q[q.procedimiento.eq(m)].sort_values("Case_n").error_absoluto_N.to_numpy()
            # Desfase determinista para separar observaciones con errores coincidentes.
            offset = np.sin(np.arange(len(val)) * 2.40) * 0.135
            ax.scatter(val, pos + offset, s=10, alpha=0.53, color=COLORES[m])
            q25, median, q75 = np.quantile(val, [0.25, 0.5, 0.75])
            ax.hlines(pos, q25, q75, color=COLORES[m], lw=4)
            ax.plot(median, pos, "|", markersize=13, markeredgewidth=2, color="#13282B")
        ax.set(
            title=PROTO[proto],
            xlabel="Error absoluto por espécimen [N]",
            yticks=np.arange(6),
            yticklabels=[NOMBRES[m] for m in MODELOS],
            ylim=(5.6, -0.6),
        )
        _estilo(ax)
    axes[0].set_xlim(left=0)
    _encabezado(
        fig,
        5,
        "Puntos por espécimen. El segmento cubre los cuartiles 1 y 3. La marca vertical indica la mediana.",
    )
    fig.subplots_adjust(left=0.17, right=0.975, top=0.82, bottom=0.17, wspace=0.12)
    return fig


def _familias_residuos(data: dict) -> list[str]:
    """Primero el menor MAE inicial y después dos familias frecuentes en el cribado."""
    q = data["seleccion"]
    counts = q[q.protocolo.eq("configuracion")].familia_real.value_counts()
    fixed = _filas(data["metricas"], "configuracion").set_index("procedimiento")
    best = fixed.loc[list(MODELOS), "MAE_N"].idxmin()
    frequent = sorted(counts.index, key=lambda m: (-counts[m], MODELOS.index(m)))
    return [best, *[m for m in frequent if m != best][:2]]


def _residuos(data: dict) -> plt.Figure:
    familias = _familias_residuos(data)
    n = len(familias)
    fig, axes = plt.subplots(
        1, n, figsize=(4 * n + 0.8, 4.6), sharex=True, sharey=True, squeeze=False
    )
    for ax, m in zip(axes.flat, familias):
        for proto, marker, col in zip(PROTOCOLOS, ("o", "x"), (AZUL, "#B0573A")):
            q = _filas(data["oof"], proto, (m,))
            ax.scatter(
                q.predicho_N,
                q.observado_N - q.predicho_N,
                s=25,
                marker=marker,
                color=col,
                alpha=0.8,
                label=PROTO[proto],
            )
        ax.axhline(0, color="#738489", ls="--", lw=1)
        ax.set(title=NOMBRES[m], xlabel="Predicción inicial [N]")
        _estilo(ax, "both")
    axes.flat[0].set_ylabel("Observación − predicción [N]")
    fig.legend(
        *axes.flat[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.54, 0.065),
        ncol=2,
        frameon=False,
    )
    _encabezado(
        fig,
        6,
        "Menor MAE inicial por configuración y dos familias frecuentes en el cribado interno. Un residuo positivo indica subestimación.",
    )
    fig.subplots_adjust(left=0.095, right=0.97, top=0.81, bottom=0.23, wspace=0.15)
    return fig


def _diferencias(data: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, proto in zip(axes, PROTOCOLOS):
        q = data["oof"][data["oof"].protocolo.eq(proto)].pivot(
            index="Case_n", columns="procedimiento", values="error_absoluto_N"
        )
        for i, m in enumerate(MODELOS):
            delta = q[m] - q.Ridge_historico
            a, b = delta.quantile([0.25, 0.75])
            ax.hlines(i, a, b, color=COLORES[m], lw=4)
            ax.scatter(
                delta.mean(),
                i,
                marker="D",
                s=31,
                color=COLORES[m],
                edgecolor="white",
                lw=0.5,
                zorder=3,
            )
        ax.axvline(0, color="#738489", lw=1, ls="--")
        ax.set(
            title=PROTO[proto],
            xlabel="Diferencia de error absoluto [N]",
            yticks=np.arange(6),
            yticklabels=[NOMBRES[m] for m in MODELOS],
            ylim=(5.6, -0.6),
        )
        _estilo(ax)
    _encabezado(
        fig,
        7,
        "Rombo = media de diferencias pareadas, igual a ΔMAE. Segmento = cuartiles 1 y 3 por espécimen. Un valor negativo favorece al modelo.",
    )
    fig.subplots_adjust(left=0.17, right=0.975, top=0.82, bottom=0.17, wspace=0.12)
    return fig


def _busqueda(data: dict) -> plt.Figure:
    trace = data["busqueda"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, proto in zip(axes, PROTOCOLOS):
        q = trace[trace.protocolo.eq(proto)].sort_values("orden").copy()
        first = q.groupby(["familia_real", "fold"]).mejor_acumulado_N.transform("first")
        q["mejora_N"] = first - q.mejor_acumulado_N
        for (m, fold), foldrows in q.groupby(["familia_real", "fold"]):
            foldrows = foldrows.sort_values("orden")
            ax.plot(foldrows.orden, foldrows.mejora_N, color=COLORES[m], alpha=0.23, lw=1)
        for m, rows in q.groupby("familia_real"):
            med = rows.groupby("orden").mejora_N.median()
            ax.plot(med.index, med.values, color=COLORES[m], label=NOMBRES[m], lw=2.6)
        ax.set(title=PROTO[proto], xlabel="Configuraciones internas evaluadas [orden]")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        _estilo(ax, "both")
    axes[0].set_ylabel("Reducción acumulada del MAE interno [N]")
    axes[0].margins(y=0.08)
    legend = [
        Line2D([0], [0], color=COLORES[m], lw=2.5, label=NOMBRES[m])
        for m in MODELOS
        if m in set(trace.loc[trace.protocolo.isin(PROTOCOLOS), "familia_real"])
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.07),
        ncol=len(legend),
        fontsize=8.5,
        frameon=False,
    )
    _encabezado(
        fig,
        8,
        "Mejora respecto a la configuración inicial de cada búsqueda. Línea tenue por entrenamiento externo y línea gruesa para la mediana de la familia.",
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.80, bottom=0.24, wspace=0.13)
    return fig


def _optimizada(data: dict) -> plt.Figure:
    slots = ("Tuning_rango_1", "Tuning_rango_2", "Selector_tuning")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.65), sharex=True, sharey=True)
    limit = 0.0
    for ax, proto in zip(axes, PROTOCOLOS):
        for i, procedure in enumerate(slots):
            q = _filas(data["oof"], proto, (procedure,))
            initial = (q.observado_N - q.predicho_inicial_misma_familia_N).abs().mean()
            adjusted = q.error_absoluto_N.mean()
            delta = adjusted - initial
            limit = max(limit, abs(delta))
            color = VERDE if delta < 0 else "#AA573F"
            ax.barh(i, delta, height=0.28, color=color)
            sign = "+" if delta >= 0 else "−"
            ax.annotate(
                f"{sign}{_numero(abs(delta), 2)}",
                (delta, i),
                xytext=((5 if delta >= 0 else -5), 0),
                textcoords="offset points",
                ha="left" if delta >= 0 else "right",
                va="center",
                fontsize=9,
                color=color,
                weight="bold",
            )
            ax.annotate(
                f"{_numero(initial, 2)} → {_numero(adjusted, 2)} N",
                (0, i),
                xytext=(0, -16),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8.7,
                color=GRIS,
            )
        ax.axvline(0, color="#738489", lw=1, ls="--")
        ax.set(
            title=PROTO[proto],
            xlabel="MAE ajustado − MAE inicial [N]",
            yticks=np.arange(len(slots)),
            yticklabels=[NOMBRES[k] for k in slots],
            ylim=(2.68, -0.55),
        )
        _estilo(ax)
    bound = max(limit * 1.4, 0.1)
    axes[0].set_xlim(-bound, bound)
    _encabezado(
        fig,
        9,
        "La etiqueta inferior muestra MAE inicial → ajustado de la misma familia y los mismos casos. Una diferencia negativa indica menor error externo.",
    )
    fig.subplots_adjust(left=0.19, right=0.975, top=0.80, bottom=0.18, wspace=0.15)
    return fig


def _estabilidad(data: dict) -> plt.Figure:
    mats = []
    for p in PROTOCOLOS:
        q = _filas(data["folds"], p)
        mats.append(
            q.pivot(index="procedimiento", columns="fold", values="MAE_validacion_N")
            .loc[list(MODELOS)]
            .sort_index(axis=1)
        )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.75),
        gridspec_kw={"width_ratios": [len(t.columns) for t in mats]},
        sharey=True,
    )
    upper = max(t.to_numpy().max() for t in mats)
    for ax, p, t in zip(axes, PROTOCOLOS, mats):
        heat = ax.imshow(
            t.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=upper, interpolation="nearest"
        )
        ax.set(
            title="Configuraciones" if p == "configuracion" else "Niveles Y",
            xlabel="Pliegue externo",
            xticks=np.arange(len(t.columns)),
            xticklabels=[str(v) for v in t.columns],
            yticks=np.arange(6),
            yticklabels=[NOMBRES[m] for m in MODELOS],
        )
        ax.tick_params(length=0, pad=4, labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.165, right=0.88, top=0.80, bottom=0.20, wspace=0.08)
    cax = fig.add_axes([0.91, 0.2, 0.016, 0.60])
    fig.colorbar(heat, cax=cax, label="MAE por pliegue [N]")
    _encabezado(
        fig,
        10,
        "Escala de color común a ambos protocolos. Cada configuración o nivel pesa una columna. Los tamaños de validación pueden diferir.",
    )
    return fig


def generar_figuras(root: Path = ROOT) -> list[Path]:
    """Genera diez figuras PNG y SVG. Devuelve los PNG en el orden del notebook."""
    root = Path(root)
    data = leer_resultados(root)
    dest = root / "reports/benchmark_regresion/figures"
    dest.mkdir(parents=True, exist_ok=True)
    jobs = (
        lambda d: _barras(d, "MAE_N", 1),
        lambda d: _barras(d, "RMSE_N", 2),
        _entrenamiento,
        _observado,
        _distribucion,
        _residuos,
        _diferencias,
        _busqueda,
        _optimizada,
        _estabilidad,
    )
    output = []
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "svg.fonttype": "none",
            "svg.hashsalt": "benchmark-regresion",
        }
    ):
        for stem, func in zip(STEMS, jobs):
            fig = func(data)
            try:
                png = dest / f"{stem}.png"
                fig.savefig(png, dpi=165, facecolor="white")
                fig.savefig(dest / f"{stem}.svg", metadata={"Date": None}, facecolor="white")
                output.append(png)
            finally:
                plt.close(fig)
    return output


if __name__ == "__main__":
    for path in generar_figuras():
        print(path.relative_to(ROOT))
