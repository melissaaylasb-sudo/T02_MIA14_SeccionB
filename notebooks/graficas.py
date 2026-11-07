"""Gráficas de evaluación de la carga de primera falla. Cada figura recibe su tabla de entrada."""
from __future__ import annotations

from io import BytesIO

from IPython.display import Image, display
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from presentacion import MODELOS, NOMBRES, PROTO, PROTOCOLOS, fmt, nombre


COLORES = dict(zip(MODELOS, ("#2C699A", "#A74930", "#7D4E8D", "#177967", "#A56612", "#506080")))
AZUL, VERDE, GRIS = "#2C699A", "#247654", "#54616C"
TITULOS = (
    'Error absoluto medio de los modelos iniciales',
    'Carga de primera falla observada y predicha',
    'Raíz del error cuadrático medio de los modelos iniciales',
    'Error de ajuste y error de validación externa',
    'Distribución del error absoluto por espécimen',
    'Residuos de las predicciones iniciales',
    'Evolución de la búsqueda de hiperparámetros',
    'Diferencia de error frente a Ridge histórico',
    'Efecto del ajuste de hiperparámetros sobre el error externo',
    'Variación del error entre particiones externas',
)


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


def mostrar(dibujar, *argumentos, destino=None):
    """Muestra un PNG nativo de Jupyter en la celda o en un panel de portada."""
    estilo = {
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11,
        "axes.titleweight": "bold", "figure.facecolor": "white",
    }
    with plt.ioff(), plt.rc_context(estilo):
        grafico = dibujar(*argumentos)
        try:
            with BytesIO() as buffer:
                grafico.savefig(buffer, format="png", dpi=165, facecolor="white")
                imagen = Image(data=buffer.getvalue(), format="png", width=1000)
        finally:
            plt.close(grafico)
    if destino is None:
        display(imagen)
    else:
        destino.update(imagen)


def error_por_modelo(metricas: pd.DataFrame, metrica: str) -> plt.Figure:
    """Contrasta la métrica externa de cada regresor con la media por nivel."""
    numero = {"MAE_N": 1, "RMSE_N": 3}[metrica]
    unidad = metrica.removesuffix("_N")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    comparables = metricas[metricas.procedimiento.isin((*MODELOS, "Media_por_nivel"))]
    limite = comparables[metrica].max() * 1.20
    y = np.arange(len(MODELOS))
    for ax, protocolo in zip(axes, PROTOCOLOS):
        datos = metricas[metricas.protocolo.eq(protocolo)].set_index("procedimiento")
        valores = datos.loc[list(MODELOS), metrica]
        baseline = datos.at["Media_por_nivel", metrica]
        ax.axvline(baseline, color=GRIS, lw=1.2, ls="--", zorder=1)
        barras = ax.barh(y, valores, height=0.52, color=[COLORES[m] for m in MODELOS], zorder=2)
        rotulos = ax.bar_label(barras, labels=[fmt(v, 2) for v in valores], padding=4, fontsize=9)
        for rotulo in rotulos:
            rotulo.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.25))
        ax.set(title=PROTO[protocolo], xlabel=f"{unidad} [N] · media por nivel: {fmt(baseline)} N",
               xlim=(0, limite), yticks=y, yticklabels=[nombre(m) for m in MODELOS],
               ylim=(len(MODELOS)-0.4, -0.6))
        _estilo(ax)
    _encabezado(fig, numero, "Línea discontinua: referencia calculada por nivel celular en entrenamiento. Menor error indica mayor precisión.")
    fig.subplots_adjust(left=0.17, right=0.98, top=0.82, bottom=0.17, wspace=0.12)
    return fig


def resumir_ajuste(pliegues):
    """Promedia por pliegue con igual peso para cada geometría reservada."""
    fijos = pliegues.loc[pliegues.procedimiento.isin(MODELOS)]
    return fijos.groupby(["protocolo", "procedimiento"])[["MAE_train_N", "MAE_validacion_N"]].mean()


def ajuste_y_validacion(errores: pd.DataFrame) -> plt.Figure:
    """Une los errores de ajuste y validación de cada familia."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharex=True, sharey=True)
    y = np.arange(len(MODELOS))
    for ax, proto in zip(axes, PROTOCOLOS):
        q = errores.loc[proto].loc[list(MODELOS)]
        ax.hlines(y, q.MAE_train_N, q.MAE_validacion_N, color="#AEBBC0", linewidth=1.6)
        for campo, color, marcador, papel, altura, alinear in [
            ("MAE_train_N", AZUL, "o", "Entrenamiento", -5, "top"),
            ("MAE_validacion_N", VERDE, "D", "Validación externa", 5, "bottom")
        ]:
            ax.scatter(q[campo], y, color=color, marker=marcador, s=30, label=papel, zorder=3)
            for pos, error in zip(y, q[campo]):
                ax.annotate(fmt(error, 1), (error, pos), xytext=(0, altura), textcoords="offset points",
                            ha="center", va=alinear, fontsize=8, color=color)
        ax.set(title=PROTO[proto], xlabel="MAE medio por pliegue [N]", xlim=(0, errores.max().max()*1.12),
               yticks=y, yticklabels=[nombre(m) for m in MODELOS], ylim=(len(MODELOS)-0.36, -0.72))
        _estilo(ax)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.56, 0.08),
               ncol=2, frameon=False)
    _encabezado(fig, 4, "Etiquetas en N. La longitud del segmento mide la diferencia entre el error de ajuste y el de validación.")
    fig.subplots_adjust(left=0.17, right=0.975, top=0.83, bottom=0.23, wspace=0.12)
    return fig


def resumir_cargas(predicciones):
    """Agrupa las cargas observadas en cuartiles comunes a todos los regresores."""
    por_probeta = predicciones.groupby("Case_n").observado_N
    if not por_probeta.nunique(dropna=False).eq(1).all():
        raise ValueError("La carga observada de una probeta debe ser la misma en todos los modelos.")
    tramos = pd.qcut(por_probeta.first(), q=4, duplicates="drop")
    por_tramo = predicciones.assign(tramo=predicciones.Case_n.map(tramos))
    return (por_tramo.groupby(["procedimiento", "protocolo", "tramo"], observed=True)
            [["observado_N", "predicho_N"]].mean().reset_index())


def carga_observada_predicha(curvas: pd.DataFrame) -> plt.Figure:
    """Contrasta la carga media observada y predicha en cada tramo."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5), sharex=True, sharey=True)
    valores = curvas[["observado_N","predicho_N"]].to_numpy()
    bajo, alto = valores.min(), valores.max()
    rango = max(1, alto-bajo)
    limites = (bajo - 0.08*rango, alto + 0.25*rango)
    for ax, modelo in zip(axes.flat, MODELOS):
        ax.plot(limites, limites, "--", color=GRIS, lw=1)
        for protocolo, simbolo, color in zip(PROTOCOLOS, ("o", "s"), (AZUL, "#B0573A")):
            q = curvas[(curvas.procedimiento == modelo) & (curvas.protocolo == protocolo)].sort_values("observado_N")
            ax.plot(q.observado_N, q.predicho_N, marker=simbolo, lw=1.4, color=color, label=PROTO[protocolo])
            ultimo = q.iloc[-1]
            ax.annotate(fmt(ultimo.predicho_N, 1), (ultimo.observado_N, ultimo.predicho_N),
                        xytext=(5, 3 if protocolo == "configuracion" else -4), textcoords="offset points",
                        va="center", ha="left", fontsize=8.5, color=color, weight="bold")
        ax.set(title=nombre(modelo), xlim=limites, ylim=limites, aspect="equal")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        _estilo(ax, "both")
    fig.supxlabel("Carga observada media [N]", y=0.10, fontsize=10)
    fig.supylabel("Carga predicha media [N]", x=0.025, fontsize=10)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center",
               ncol=2, bbox_to_anchor=(0.53, 0.036), frameon=False)
    _encabezado(fig, 2, "Se rotula la predicción media [N] en el cuartil superior. La diagonal indica igualdad entre medición y predicción.")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.89, bottom=0.20, hspace=0.32, wspace=0.15)
    return fig


def percentiles_por_probeta(predicciones):
    """Resume el error absoluto individual de cada modelo y protocolo."""
    iniciales = predicciones.loc[predicciones.procedimiento.isin(MODELOS)]
    return (iniciales.groupby(["protocolo", "procedimiento"]).error_absoluto_N
            .quantile([.10, .25, .50, .75, .90]).unstack())


def distribucion_errores(cuantiles: pd.DataFrame) -> plt.Figure:
    """Representa mediana, cuartiles y percentiles del error absoluto."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharex=True, sharey=True)
    for ax, proto in zip(axes, PROTOCOLOS):
        for pos, modelo in enumerate(MODELOS):
            p10, q25, mediana, q75, p90 = cuantiles.loc[(proto, modelo)]
            ax.hlines(pos, p10, p90, color=COLORES[modelo], lw=1.2)
            ax.hlines(pos, q25, q75, color=COLORES[modelo], lw=5)
            ax.plot(mediana, pos, "|", markersize=13, markeredgewidth=2, color="#13282B")
            ax.annotate(fmt(mediana, 2), (mediana, pos), xytext=(0, 7), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color="#13282B")
        ax.set(title=PROTO[proto], xlabel="Error absoluto por probeta [N]", xlim=(0, cuantiles[.90].max()*1.06),
               yticks=np.arange(len(MODELOS)), yticklabels=[nombre(m) for m in MODELOS],
               ylim=(len(MODELOS)-0.45, -0.76))
        _estilo(ax)
    _encabezado(fig, 5, "La cifra es la mediana [N]. La línea fina abarca P10–P90 y la gruesa, el rango intercuartílico.")
    fig.subplots_adjust(left=0.17, right=0.975, top=0.82, bottom=0.17, wspace=0.12)
    return fig


def _familias_residuos(metricas, seleccion) -> list[str]:
    """Primero el menor MAE inicial y después dos familias frecuentes en el cribado."""
    counts = seleccion.loc[seleccion.protocolo.eq("configuracion"), "familia_real"].value_counts()
    fixed = metricas.loc[metricas.protocolo.eq("configuracion")].set_index("procedimiento")
    best = fixed.loc[list(MODELOS), "MAE_N"].idxmin()
    frequent = sorted(counts.index, key=lambda m: (-counts[m], MODELOS.index(m)))
    return [best, *[m for m in frequent if m != best][:2]]


def residuos_por_tramo(predicciones, metricas, seleccion):
    """Agrupa residuos de cada familia elegida por cuartil de carga predicha."""
    grupos = []
    for modelo in _familias_residuos(metricas, seleccion):
        for protocolo in PROTOCOLOS:
            q = predicciones.loc[predicciones.protocolo.eq(protocolo) & predicciones.procedimiento.eq(modelo)]
            pred = q.predicho_N
            tramo = (pd.qcut(pred, q=4, duplicates="drop")
                     if pred.nunique() > 1 else pd.Series("constante", index=q.index))
            serie = q.assign(tramo=tramo, residuo=q.observado_N - pred)
            por_tramo = serie.groupby("tramo", observed=True).agg(
                x=("predicho_N", "median"), y=("residuo", "median"),
                inf=("residuo", lambda r: r.quantile(.25)),
                sup=("residuo", lambda r: r.quantile(.75)),
            ).sort_values("x")
            grupos.append(por_tramo.assign(modelo=modelo, protocolo=protocolo))
    return pd.concat(grupos, ignore_index=True)


def residuos(tramos):
    """Muestra el signo, la mediana y la dispersión de los residuos por tramo."""
    modelos = tramos.modelo.unique()
    fig, axes = plt.subplots(1, len(modelos), figsize=(11.6, 5), sharex=True, sharey=True)
    for ax, modelo in zip(axes.flat, modelos):
        for i, (p, marca, color) in enumerate(zip(PROTOCOLOS, ("o", "x"), (AZUL, "#B0573A"))):
            q = tramos.loc[(tramos.modelo == modelo) & (tramos.protocolo == p)]
            ax.vlines(q.x, q.inf, q.sup, color=color, alpha=.55, lw=1.8)
            ax.plot(q.x, q.y, marker=marca, color=color, lw=1.4)
            etiqueta = "Configuración" if p == "configuracion" else "Nivel Y"
            ax.text(.5, -.25 - .11*i, f"{etiqueta}: {fmt(q.y.iloc[-1], 2)} N",
                    transform=ax.transAxes, color=color, ha="center", fontsize=9)
        ax.axhline(0, color=GRIS, ls="--", lw=1)
        ax.set(title=NOMBRES[modelo], xlabel="Predicción inicial [N]")
        _estilo(ax, "both")
    axes.flat[0].set_ylabel("Observación − predicción [N]")
    _encabezado(fig, 6, "Marcas: mediana por tramo. Barras: rango intercuartílico. Etiquetas: residuo mediano en el tramo de predicción más alto de cada curva.")
    fig.subplots_adjust(left=.09, right=.98, top=.82, bottom=.32, wspace=.14)
    return fig


def progreso_busqueda(busqueda):
    """Calcula la mejora interna acumulada respecto a la primera receta evaluada."""
    orden = busqueda.sort_values("orden").copy()
    grupo = ["protocolo", "familia_real", "fold"]
    inicio = orden.groupby(grupo).mejor_acumulado_N.transform("first")
    orden["ganancia_N"] = inicio - orden.mejor_acumulado_N
    return (orden.groupby(["protocolo", "familia_real", "orden"]).ganancia_N
            .quantile([.25, .50, .75]).unstack()
            .rename(columns={.25: "p25", .50: "mediana", .75: "p75"}))


def evolucion_busqueda(progreso):
    """Compara el progreso de la búsqueda interna entre familias."""
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.8), sharex=True, sharey=True)
    for ax, p in zip(axes, PROTOCOLOS):
        panel = progreso.xs(p, level="protocolo")
        for modelo, serie in panel.groupby(level="familia_real"):
            q = serie.droplevel("familia_real")
            color = COLORES[modelo]
            etiqueta = f"{nombre(modelo)} · {fmt(q.mediana.iloc[-1], 2)} N"
            ax.fill_between(q.index, q.p25, q.p75, color=color, alpha=.17)
            ax.plot(q.index, q.mediana, color=color, label=etiqueta, lw=2)
            ax.plot(q.index[-1], q.mediana.iloc[-1], "o", color=color, ms=4)
        ax.set(title=PROTO[p], xlabel="Configuraciones de hiperparámetros evaluadas [orden]")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2, frameon=False,
                  fontsize=8.4, handlelength=1.2, columnspacing=.9, title="Reducción mediana al cierre")
        _estilo(ax, "both")
    axes[0].set_ylabel("Reducción del MAE interno [N]")
    fig.subplots_adjust(left=.085, right=.98, top=.83, bottom=.34, wspace=.13)
    _encabezado(fig, 7, "La línea muestra la mediana de la mejora acumulada sobre la receta inicial. La banda cubre el rango intercuartílico.")
    return fig


def contraste_ridge(predicciones: pd.DataFrame) -> plt.Figure:
    """Contrasta cada error absoluto con Ridge sobre el mismo espécimen."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, proto in zip(axes, PROTOCOLOS):
        q = predicciones.loc[predicciones.protocolo.eq(proto)].pivot(
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
            ax.annotate(f"{fmt(delta.mean(), 2)}", (delta.mean(), i), xytext=(0, 7),
                        textcoords="offset points", va="bottom", ha="center", fontsize=8,
                        color=COLORES[m], weight="bold")
        ax.margins(x=.12)
        ax.axvline(0, color="#738489", lw=1, ls="--")
        ax.set(
            title=PROTO[proto],
            xlabel="Diferencia de error absoluto [N]",
            yticks=np.arange(len(MODELOS)),
            yticklabels=[NOMBRES[m] for m in MODELOS],
            ylim=(len(MODELOS) - 0.4, -0.6),
        )
        _estilo(ax)
    _encabezado(
        fig,
        8,
        "Cada etiqueta indica el ΔMAE frente a Ridge [N]. El rombo es la media pareada y la línea es el rango intercuartílico de las diferencias individuales.",
    )
    fig.subplots_adjust(left=0.17, right=0.975, top=0.82, bottom=0.17, wspace=0.12)
    return fig


def efecto_tuning(pares):
    """Compara el MAE externo de la misma familia antes y después del ajuste."""
    slots = ("Tuning_rango_1", "Tuning_rango_2", "Selector_tuning")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.65), sharex=True, sharey=True)
    limit = 0.0
    for ax, proto in zip(axes, PROTOCOLOS):
        for i, procedure in enumerate(slots):
            initial, adjusted = pares.loc[(proto, procedure)]
            delta = adjusted - initial
            limit = max(limit, abs(delta))
            color = VERDE if delta < 0 else "#AA573F"
            ax.barh(i, delta, height=0.28, color=color)
            sign = "+" if delta >= 0 else "−"
            ax.annotate(
                f"{sign}{fmt(abs(delta), 3)}",
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
                f"{fmt(initial, 3)} → {fmt(adjusted, 3)} N",
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


def percentiles_por_reserva(pliegues):
    """Resume por percentiles el MAE de los pliegues externos."""
    niveles = [.10, .25, .50, .75, .90]
    iniciales = pliegues.loc[pliegues.procedimiento.isin(MODELOS)]
    return iniciales.groupby(["protocolo", "procedimiento"]).MAE_validacion_N.quantile(niveles).unstack()


def estabilidad(percentiles):
    """Muestra la variabilidad entre reservas con una escala común."""
    cuadros = [percentiles.loc[p].loc[list(MODELOS)] for p in PROTOCOLOS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.75), sharey=True)
    upper = max(t.to_numpy().max() for t in cuadros)
    for ax, p, cuadro in zip(axes, PROTOCOLOS, cuadros):
        heat = ax.imshow(cuadro, aspect="auto", cmap="viridis", vmin=0, vmax=upper)
        for (fila, col), valor_celda in np.ndenumerate(cuadro.to_numpy()):
            tinta = "#122829" if heat.norm(valor_celda) > .51 else "white"
            ax.text(col, fila, fmt(valor_celda, 1), ha="center", va="center", color=tinta, fontsize=8.8)
        ax.set(title=PROTO[p], xlabel="Percentil del MAE de los pliegues externos",
               xticks=range(len(cuadro.columns)), xticklabels=["P10", "P25", "P50", "P75", "P90"],
               yticks=range(len(MODELOS)), yticklabels=[NOMBRES[m] for m in MODELOS])
        ax.tick_params(length=0, pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.17, right=0.88, top=0.8, bottom=0.22, wspace=0.12)
    cax = fig.add_axes([0.91, 0.22, 0.016, 0.58])
    fig.colorbar(heat, cax=cax, label="MAE por pliegue [N]")
    _encabezado(fig, 10, "Los percentiles resumen la dispersión entre reservas. Se usa la misma escala en ambos paneles.")
    return fig
