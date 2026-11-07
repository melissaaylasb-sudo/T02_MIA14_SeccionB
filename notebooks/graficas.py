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


def _estilo(eje: plt.Axes, direccion: str = "x") -> None:
    eje.set_axisbelow(True)
    eje.grid(axis=direccion, color="#E0E6E7", linewidth=0.7)
    for borde in eje.spines.values():
        borde.set_visible(False)
    eje.tick_params(axis="both", length=0, pad=5)


def _encabezado(figura: plt.Figure, numero: int) -> None:
    figura.suptitle(
        TITULOS[numero - 1],
        x=0.025,
        y=0.986,
        ha="left",
        va="top",
        fontsize=16,
        weight="bold",
        color="#203D40",
    )


def mostrar(dibujar, *argumentos, destino=None):
    """Muestra un PNG nativo de Jupyter en la celda o en un panel de portada."""
    estilo = {
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 11,
        "axes.titleweight": "bold", "figure.facecolor": "white",
    }
    with plt.ioff(), plt.rc_context(estilo):
        grafico = dibujar(*argumentos)
        try:
            with BytesIO() as memoria:
                grafico.savefig(memoria, format="png", dpi=165, facecolor="white")
                imagen = Image(data=memoria.getvalue(), format="png", width=1000)
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
    figura, ejes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    comparables = metricas[metricas.procedimiento.isin((*MODELOS, "Media_por_nivel"))]
    limite = comparables[metrica].max() * 1.20
    posiciones = np.arange(len(MODELOS))
    for eje, protocolo in zip(ejes, PROTOCOLOS):
        datos = metricas[metricas.protocolo.eq(protocolo)].set_index("procedimiento")
        valores = datos.loc[list(MODELOS), metrica]
        referencia_base = datos.at["Media_por_nivel", metrica]
        eje.axvline(referencia_base, color=GRIS, lw=1.2, ls="--", zorder=1)
        barras = eje.barh(posiciones, valores, height=0.52, color=[COLORES[m] for m in MODELOS], zorder=2)
        rotulos = eje.bar_label(barras, labels=[fmt(v, 2) for v in valores], padding=4, fontsize=9)
        for rotulo in rotulos:
            rotulo.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.25))
        eje.set(title=PROTO[protocolo], xlabel=f"{unidad} [N] · media por nivel: {fmt(referencia_base)} N",
               xlim=(0, limite), yticks=posiciones, yticklabels=[nombre(m) for m in MODELOS],
               ylim=(len(MODELOS)-0.4, -0.6))
        _estilo(eje)
    _encabezado(figura, numero)
    figura.subplots_adjust(left=0.17, right=0.98, top=0.82, bottom=0.17, wspace=0.12)
    return figura


def resumir_ajuste(pliegues):
    """Promedia por pliegue con igual peso para cada geometría reservada."""
    fijos = pliegues.loc[pliegues.procedimiento.isin(MODELOS)]
    return fijos.groupby(["protocolo", "procedimiento"])[["MAE_entrenamiento_N", "MAE_validacion_N"]].mean()


def ajuste_y_validacion(errores: pd.DataFrame) -> plt.Figure:
    """Une los errores de ajuste y validación de cada familia."""
    figura, ejes = plt.subplots(1, 2, figsize=(11, 5.4), sharex=True, sharey=True)
    posiciones = np.arange(len(MODELOS))
    for eje, proto in zip(ejes, PROTOCOLOS):
        q = errores.loc[proto].loc[list(MODELOS)]
        eje.hlines(posiciones, q.MAE_entrenamiento_N, q.MAE_validacion_N, color="#AEBBC0", linewidth=1.6)
        for campo, color, marcador, papel, altura, alinear in [
            ("MAE_entrenamiento_N", AZUL, "o", "Entrenamiento", -5, "top"),
            ("MAE_validacion_N", VERDE, "D", "Validación externa", 5, "bottom")
        ]:
            eje.scatter(q[campo], posiciones, color=color, marker=marcador, s=30, label=papel, zorder=3)
            for pos, error in zip(posiciones, q[campo]):
                eje.annotate(fmt(error, 1), (error, pos), xytext=(0, altura), textcoords="offset points",
                            ha="center", va=alinear, fontsize=8, color=color)
        eje.set(title=PROTO[proto], xlabel="MAE medio por pliegue [N]", xlim=(0, errores.max().max()*1.12),
               yticks=posiciones, yticklabels=[nombre(m) for m in MODELOS], ylim=(len(MODELOS)-0.36, -0.72))
        _estilo(eje)
    figura.legend(*ejes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.56, 0.08),
               ncol=2, frameon=False)
    _encabezado(figura, 4)
    figura.subplots_adjust(left=0.17, right=0.975, top=0.83, bottom=0.23, wspace=0.12)
    return figura


def resumir_cargas(predicciones):
    """Agrupa las cargas observadas en cuartiles comunes a todos los regresores."""
    por_probeta = predicciones.groupby("id_probeta").observado_N
    if not por_probeta.nunique(dropna=False).eq(1).all():
        raise ValueError("La carga observada de una probeta debe ser la misma en todos los modelos.")
    tramos = pd.qcut(por_probeta.first(), q=4, duplicates="drop")
    por_tramo = predicciones.assign(tramo=predicciones.id_probeta.map(tramos))
    return (por_tramo.groupby(["procedimiento", "protocolo", "tramo"], observed=True)
            [["observado_N", "predicho_N"]].mean().reset_index())


def carga_observada_predicha(curvas: pd.DataFrame) -> plt.Figure:
    """Contrasta la carga media observada y predicha en cada tramo."""
    figura, ejes = plt.subplots(2, 3, figsize=(11, 7.5), sharex=True, sharey=True)
    valores = curvas[["observado_N","predicho_N"]].to_numpy()
    bajo, alto = valores.min(), valores.max()
    rango = max(1, alto-bajo)
    limites = (bajo - 0.08*rango, alto + 0.25*rango)
    for eje, modelo in zip(ejes.flat, MODELOS):
        eje.plot(limites, limites, "--", color=GRIS, lw=1)
        for protocolo, simbolo, color in zip(PROTOCOLOS, ("o", "s"), (AZUL, "#B0573A")):
            q = curvas[(curvas.procedimiento == modelo) & (curvas.protocolo == protocolo)].sort_values("observado_N")
            eje.plot(q.observado_N, q.predicho_N, marker=simbolo, lw=1.4, color=color, label=PROTO[protocolo])
            ultimo = q.iloc[-1]
            eje.annotate(fmt(ultimo.predicho_N, 1), (ultimo.observado_N, ultimo.predicho_N),
                        xytext=(5, 3 if protocolo == "configuracion" else -4), textcoords="offset points",
                        va="center", ha="left", fontsize=8.5, color=color, weight="bold")
        eje.set(title=nombre(modelo), xlim=limites, ylim=limites, aspect="equal")
        eje.xaxis.set_major_locator(MaxNLocator(4))
        eje.yaxis.set_major_locator(MaxNLocator(4))
        _estilo(eje, "both")
    figura.supxlabel("Carga observada media [N]", y=0.10, fontsize=10)
    figura.supylabel("Carga predicha media [N]", x=0.025, fontsize=10)
    figura.legend(*ejes.flat[0].get_legend_handles_labels(), loc="lower center",
               ncol=2, bbox_to_anchor=(0.53, 0.036), frameon=False)
    _encabezado(figura, 2)
    figura.subplots_adjust(left=0.08, right=0.98, top=0.89, bottom=0.20, hspace=0.32, wspace=0.15)
    return figura


def percentiles_por_probeta(predicciones):
    """Resume el error absoluto individual de cada modelo y protocolo."""
    iniciales = predicciones.loc[predicciones.procedimiento.isin(MODELOS)]
    return (iniciales.groupby(["protocolo", "procedimiento"]).error_absoluto_N
            .quantile([.10, .25, .50, .75, .90]).unstack())


def distribucion_errores(cuantiles: pd.DataFrame) -> plt.Figure:
    """Representa mediana, cuartiles y percentiles del error absoluto."""
    figura, ejes = plt.subplots(1, 2, figsize=(11, 5.0), sharex=True, sharey=True)
    for eje, proto in zip(ejes, PROTOCOLOS):
        for pos, modelo in enumerate(MODELOS):
            p10, q25, mediana, q75, p90 = cuantiles.loc[(proto, modelo)]
            eje.hlines(pos, p10, p90, color=COLORES[modelo], lw=1.2)
            eje.hlines(pos, q25, q75, color=COLORES[modelo], lw=5)
            eje.plot(mediana, pos, "|", markersize=13, markeredgewidth=2, color="#13282B")
            eje.annotate(fmt(mediana, 2), (mediana, pos), xytext=(0, 7), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color="#13282B")
        eje.set(title=PROTO[proto], xlabel="Error absoluto por probeta [N]", xlim=(0, cuantiles[.90].max()*1.06),
               yticks=np.arange(len(MODELOS)), yticklabels=[nombre(m) for m in MODELOS],
               ylim=(len(MODELOS)-0.45, -0.76))
        _estilo(eje)
    _encabezado(figura, 5)
    figura.subplots_adjust(left=0.17, right=0.975, top=0.82, bottom=0.17, wspace=0.12)
    return figura


def _familias_residuos(metricas, seleccion) -> list[str]:
    """Primero el menor MAE inicial y después dos familias frecuentes en el cribado."""
    frecuencias = seleccion.loc[seleccion.protocolo.eq("configuracion"), "familia_real"].value_counts()
    receta_fija = metricas.loc[metricas.protocolo.eq("configuracion")].set_index("procedimiento")
    mejor_valor = receta_fija.loc[list(MODELOS), "MAE_N"].idxmin()
    frecuentes = sorted(frecuencias.index, key=lambda m: (-frecuencias[m], MODELOS.index(m)))
    return [mejor_valor, *[m for m in frecuentes if m != mejor_valor][:2]]


def residuos_por_tramo(predicciones, metricas, seleccion):
    """Agrupa residuos de cada familia elegida por cuartil de carga predicha."""
    grupos = []
    for modelo in _familias_residuos(metricas, seleccion):
        for protocolo in PROTOCOLOS:
            q = predicciones.loc[predicciones.protocolo.eq(protocolo) & predicciones.procedimiento.eq(modelo)]
            carga_predicha = q.predicho_N
            tramo = (pd.qcut(carga_predicha, q=4, duplicates="drop")
                     if carga_predicha.nunique() > 1 else pd.Series("constante", index=q.index))
            serie = q.assign(tramo=tramo, residuo=q.observado_N - carga_predicha)
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
    figura, ejes = plt.subplots(1, len(modelos), figsize=(11.6, 5), sharex=True, sharey=True)
    for eje, modelo in zip(ejes.flat, modelos):
        for i, (p, marca, color) in enumerate(zip(PROTOCOLOS, ("o", "x"), (AZUL, "#B0573A"))):
            q = tramos.loc[(tramos.modelo == modelo) & (tramos.protocolo == p)]
            eje.vlines(q.x, q.inf, q.sup, color=color, alpha=.55, lw=1.8)
            eje.plot(q.x, q.y, marker=marca, color=color, lw=1.4)
            etiqueta = "Configuración" if p == "configuracion" else "Nivel Y"
            eje.text(.5, -.25 - .11*i, f"{etiqueta}: {fmt(q.y.iloc[-1], 2)} N",
                    transform=eje.transAxes, color=color, ha="center", fontsize=9)
        eje.axhline(0, color=GRIS, ls="--", lw=1)
        eje.set(title=NOMBRES[modelo], xlabel="Predicción inicial [N]")
        _estilo(eje, "both")
    ejes.flat[0].set_ylabel("Observación − predicción [N]")
    _encabezado(figura, 6)
    figura.subplots_adjust(left=.09, right=.98, top=.82, bottom=.32, wspace=.14)
    return figura


def progreso_busqueda(busqueda):
    """Calcula la mejora interna acumulada respecto a la primera receta evaluada."""
    orden = busqueda.sort_values("orden").copy()
    grupo = ["protocolo", "familia_real", "pliegue"]
    inicio = orden.groupby(grupo).mejor_acumulado_N.transform("first")
    orden["ganancia_N"] = inicio - orden.mejor_acumulado_N
    return (orden.groupby(["protocolo", "familia_real", "orden"]).ganancia_N
            .quantile([.25, .50, .75]).unstack()
            .rename(columns={.25: "p25", .50: "mediana", .75: "p75"}))


def evolucion_busqueda(progreso):
    """Compara el progreso de la búsqueda interna entre familias."""
    figura, ejes = plt.subplots(1, 2, figsize=(11.7, 5.8), sharex=True, sharey=True)
    for eje, p in zip(ejes, PROTOCOLOS):
        panel = progreso.xs(p, level="protocolo")
        for modelo, serie in panel.groupby(level="familia_real"):
            q = serie.droplevel("familia_real")
            color = COLORES[modelo]
            etiqueta = f"{nombre(modelo)} · {fmt(q.mediana.iloc[-1], 2)} N"
            eje.fill_between(q.index, q.p25, q.p75, color=color, alpha=.17)
            eje.plot(q.index, q.mediana, color=color, label=etiqueta, lw=2)
            eje.plot(q.index[-1], q.mediana.iloc[-1], "o", color=color, ms=4)
        eje.set(title=PROTO[p], xlabel="Configuraciones de hiperparámetros evaluadas [orden]")
        eje.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        eje.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2, frameon=False,
                  fontsize=8.4, handlelength=1.2, columnspacing=.9, title="Reducción mediana al cierre")
        _estilo(eje, "both")
    ejes[0].set_ylabel("Reducción del MAE interno [N]")
    figura.subplots_adjust(left=.085, right=.98, top=.83, bottom=.34, wspace=.13)
    _encabezado(figura, 7)
    return figura


def contraste_ridge(predicciones: pd.DataFrame) -> plt.Figure:
    """Contrasta cada error absoluto con Ridge sobre el mismo espécimen."""
    figura, ejes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for eje, proto in zip(ejes, PROTOCOLOS):
        q = predicciones.loc[predicciones.protocolo.eq(proto)].pivot(
            index="id_probeta", columns="procedimiento", values="error_absoluto_N"
        )
        for i, m in enumerate(MODELOS):
            delta = q[m] - q.Ridge_historico
            a, b = delta.quantile([0.25, 0.75])
            eje.hlines(i, a, b, color=COLORES[m], lw=4)
            eje.scatter(
                delta.mean(),
                i,
                marker="D",
                s=31,
                color=COLORES[m],
                edgecolor="white",
                lw=0.5,
                zorder=3,
            )
            eje.annotate(f"{fmt(delta.mean(), 2)}", (delta.mean(), i), xytext=(0, 7),
                        textcoords="offset points", va="bottom", ha="center", fontsize=8,
                        color=COLORES[m], weight="bold")
        eje.margins(x=.12)
        eje.axvline(0, color="#738489", lw=1, ls="--")
        eje.set(
            title=PROTO[proto],
            xlabel="Diferencia de error absoluto [N]",
            yticks=np.arange(len(MODELOS)),
            yticklabels=[NOMBRES[m] for m in MODELOS],
            ylim=(len(MODELOS) - 0.4, -0.6),
        )
        _estilo(eje)
    _encabezado(figura, 8)
    figura.subplots_adjust(left=0.17, right=0.975, top=0.82, bottom=0.17, wspace=0.12)
    return figura


def efecto_ajuste(pares):
    """Compara el MAE externo de la misma familia antes y después del ajuste."""
    estrategias = ("Tuning_rango_1", "Tuning_rango_2", "Selector_tuning")
    figura, ejes = plt.subplots(1, 2, figsize=(11, 4.65), sharex=True, sharey=True)
    limite = 0.0
    for eje, proto in zip(ejes, PROTOCOLOS):
        for i, procedimiento in enumerate(estrategias):
            prediccion_inicial, ajustado = pares.loc[(proto, procedimiento)]
            delta = ajustado - prediccion_inicial
            limite = max(limite, abs(delta))
            color = VERDE if delta < 0 else "#AA573F"
            eje.barh(i, delta, height=0.28, color=color)
            signo = "+" if delta >= 0 else "−"
            eje.annotate(
                f"{signo}{fmt(abs(delta), 3)}",
                (delta, i),
                xytext=((5 if delta >= 0 else -5), 0),
                textcoords="offset points",
                ha="left" if delta >= 0 else "right",
                va="center",
                fontsize=9,
                color=color,
                weight="bold",
            )
            eje.annotate(
                f"{fmt(prediccion_inicial, 3)} → {fmt(ajustado, 3)} N",
                (0, i),
                xytext=(0, -16),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8.7,
                color=GRIS,
            )
        eje.axvline(0, color="#738489", lw=1, ls="--")
        eje.set(
            title=PROTO[proto],
            xlabel="MAE ajustado − MAE inicial [N]",
            yticks=np.arange(len(estrategias)),
            yticklabels=[NOMBRES[k] for k in estrategias],
            ylim=(2.68, -0.55),
        )
        _estilo(eje)
    cota = max(limite * 1.4, 0.1)
    ejes[0].set_xlim(-cota, cota)
    _encabezado(figura, 9)
    figura.subplots_adjust(left=0.19, right=0.975, top=0.80, bottom=0.18, wspace=0.15)
    return figura


def percentiles_por_reserva(pliegues):
    """Resume por percentiles el MAE de los pliegues externos."""
    niveles = [.10, .25, .50, .75, .90]
    iniciales = pliegues.loc[pliegues.procedimiento.isin(MODELOS)]
    return iniciales.groupby(["protocolo", "procedimiento"]).MAE_validacion_N.quantile(niveles).unstack()


def estabilidad(percentiles):
    """Muestra la variabilidad entre reservas con una escala común."""
    cuadros = [percentiles.loc[p].loc[list(MODELOS)] for p in PROTOCOLOS]
    figura, ejes = plt.subplots(1, 2, figsize=(11, 4.75), sharey=True)
    superior = max(t.to_numpy().max() for t in cuadros)
    for eje, p, cuadro in zip(ejes, PROTOCOLOS, cuadros):
        mapa_color = eje.imshow(cuadro, aspect="auto", cmap="viridis", vmin=0, vmax=superior)
        for (fila, col), valor_celda in np.ndenumerate(cuadro.to_numpy()):
            tinta = "#122829" if mapa_color.norm(valor_celda) > .51 else "white"
            eje.text(col, fila, fmt(valor_celda, 1), ha="center", va="center", color=tinta, fontsize=8.8)
        eje.set(title=PROTO[p], xlabel="Percentil del MAE de los pliegues externos",
               xticks=range(len(cuadro.columns)), xticklabels=["P10", "P25", "P50", "P75", "P90"],
               yticks=range(len(MODELOS)), yticklabels=[NOMBRES[m] for m in MODELOS])
        eje.tick_params(length=0, pad=5)
        for borde in eje.spines.values():
            borde.set_visible(False)
    figura.subplots_adjust(left=0.17, right=0.88, top=0.8, bottom=0.22, wspace=0.12)
    eje_color = figura.add_axes([0.91, 0.22, 0.016, 0.58])
    figura.colorbar(mapa_color, cax=eje_color, label="MAE por pliegue [N]")
    _encabezado(figura, 10)
    return figura



def errores_ganador(metricas, ganador):
    """Compara exclusivamente el modelo elegido y el baseline principal."""
    figura, ejes = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
    procedimientos = ["Media_por_nivel", ganador]
    for eje, protocolo in zip(ejes, PROTOCOLOS):
        valores = metricas.loc[metricas.protocolo.eq(protocolo)].set_index("procedimiento").loc[procedimientos, "MAE_N"]
        barras = eje.barh([0, 1], valores, color=[GRIS, VERDE], height=.5)
        eje.bar_label(barras, labels=[fmt(v, 3) for v in valores], padding=5)
        eje.set(title=PROTO[protocolo], xlabel="MAE [N]", xlim=(0, valores.max()*1.3),
                yticks=[0, 1], yticklabels=[nombre(m) for m in procedimientos])
        _estilo(eje)
    figura.suptitle("Error del modelo seleccionado frente al baseline", fontsize=16, color="#203D40", weight="bold")
    figura.tight_layout(rect=(0, 0, 1, .9))
    return figura


def cargas_ganador(predicciones):
    """Carga media observada y predicha por cuartil, con valores etiquetados."""
    curvas = resumir_cargas(predicciones)
    figura, eje = plt.subplots(figsize=(8, 5.5))
    valores = curvas[["observado_N", "predicho_N"]].to_numpy()
    limites = (valores.min()-15, valores.max()+35)
    eje.plot(limites, limites, "--", color=GRIS, lw=1)
    for protocolo, color, marca in zip(PROTOCOLOS, (AZUL, VERDE), ("o", "s")):
        serie = curvas.loc[curvas.protocolo.eq(protocolo)].sort_values("observado_N")
        eje.plot(serie.observado_N, serie.predicho_N, marker=marca, color=color, label=PROTO[protocolo])
        for fila in serie.itertuples():
            eje.annotate(fmt(fila.predicho_N, 2), (fila.observado_N, fila.predicho_N),
                         xytext=(-8, 12) if protocolo == "configuracion" else (7, -14),
                         ha="right" if protocolo == "configuracion" else "left",
                         textcoords="offset points", color=color, fontsize=9)
    eje.set(xlabel="Carga observada media [N]", ylabel="Carga predicha media [N]",
            title="Carga observada y predicha por la regresión lineal", xlim=limites, ylim=limites)
    eje.legend(frameon=False)
    _estilo(eje, "both")
    figura.tight_layout()
    return figura


def residuos_ganador(predicciones):
    """Residuos individuales y residuo medio por protocolo."""
    figura, ejes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for eje, protocolo in zip(ejes, PROTOCOLOS):
        grupo = predicciones.loc[predicciones.protocolo.eq(protocolo)]
        residuo = grupo.observado_N - grupo.predicho_N
        eje.scatter(grupo.predicho_N, residuo, color=AZUL, alpha=.8)
        eje.axhline(0, color=GRIS, ls="--")
        eje.axhline(residuo.mean(), color=VERDE, label=f"Residuo medio {fmt(residuo.mean(), 2)} N")
        mayor = residuo.abs().idxmax()
        eje.annotate(f"{fmt(residuo.loc[mayor], 2)} N",
                     (grupo.loc[mayor, "predicho_N"], residuo.loc[mayor]),
                     xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
        eje.set(title=PROTO[protocolo], xlabel="Carga predicha [N]")
        eje.legend(frameon=False, fontsize=9, loc="lower center")
        eje.margins(y=.2)
        _estilo(eje, "both")
    ejes[0].set_ylabel("Carga observada − predicha [N]")
    figura.suptitle("Residuos de la regresión lineal", fontsize=16, color="#203D40", weight="bold")
    figura.tight_layout(rect=(0, 0, 1, .9))
    return figura


def distribucion_ganador(predicciones):
    """Mediana y percentil 90 del error del modelo ganador."""
    figura, eje = plt.subplots(figsize=(9, 3.6))
    valores = [predicciones.loc[predicciones.protocolo.eq(p), "error_absoluto_N"] for p in PROTOCOLOS]
    eje.boxplot(valores, orientation="horizontal", tick_labels=[PROTO[p] for p in PROTOCOLOS],
                widths=.4, patch_artist=True, boxprops={"facecolor": "#e9f5ec"})
    for posicion, errores in enumerate(valores, 1):
        mediana, p90 = errores.quantile([.5, .9])
        eje.scatter(p90, posicion, marker="D", color=AZUL, zorder=3)
        eje.annotate(f"Mediana {fmt(mediana, 2)} N", (mediana, posicion),
                     xytext=(0, 23), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        eje.annotate(f"P90 {fmt(p90, 2)} N", (p90, posicion),
                     xytext=(0, -20), textcoords="offset points", ha="center", fontsize=9, color=AZUL)
    eje.set(xlabel="Error absoluto [N]", title="Distribución del error de la regresión lineal", ylim=(.4, 2.6))
    _estilo(eje)
    figura.tight_layout()
    return figura
