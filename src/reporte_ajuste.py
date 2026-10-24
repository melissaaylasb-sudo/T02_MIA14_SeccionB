"""Construye y ejecuta el notebook de ajuste sin repetir el experimento completo.

Los resultados se leen de una ejecución auditada. Las figuras se calculan desde
sus predicciones y métricas, nunca desde cifras transcritas manualmente.

Uso: python -m src.reporte_ajuste --execute
"""
from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
import nbformat
import numpy as np
import pandas as pd

from .eda import write_bytes_atomic
from .feature_engineering import TARGET, feature_names, make_pipeline, nominal_groups
from .presentacion_fe import ETIQUETAS_VARIABLES, UNIDADES_VARIABLES
from .transformaciones_eda import GEOMETRY


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/05_Modelos_Ajustados.ipynb"
DEFAULT_RUN = ROOT / "reports/modelos_ajustados"
AZUL, VERDE, ROSA, TINTA = "#2563a6", "#008579", "#b94a65", "#253447"
MODELO_NOMBRES = {
    "Baseline_Ridge": "Ridge · línea base",
    "Ridge": "Ridge ajustado", "ElasticNet": "Elastic Net",
    "SVR_RBF": "SVR RBF", "RandomForest": "Bosque aleatorio",
    "ExtraTrees": "Extra Trees", "GaussianProcess": "Proceso gaussiano",
    "GradientBoosting": "Gradient Boosting",
    "Selector_interno": "Selección interna",
}
PROTOCOLOS = {"configuracion": "Configuración excluida", "familia": "Familia excluida"}


def nombre_modelo(nombre):
    return MODELO_NOMBRES.get(str(nombre), str(nombre).replace("_", " "))


def leer_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def huella(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _guardar_json(path, valor):
    write_bytes_atomic(Path(path), (json.dumps(valor, ensure_ascii=False, indent=2,
                                               allow_nan=False) + "\n").encode("utf-8"))


def datos_congelados():
    """Lee exactamente el snapshot usado en el experimento previo y verifica SHA-256."""
    historico = leer_json(ROOT / "reports/feature_engineering/manifest.json")
    ruta = ROOT / "data/processed/preliminary/model_table.csv"
    esperada = historico["inputs_sha256"][ruta.relative_to(ROOT).as_posix()]
    if huella(ruta) != esperada:
        raise ValueError("Cambió la tabla histórica: no se pueden comparar los resultados.")
    datos = pd.read_csv(ruta).sort_values("Case_n").reset_index(drop=True)
    if datos[TARGET].isna().any() or datos.Case_n.duplicated().any():
        raise ValueError("El snapshot supervisado tiene etiquetas ausentes o identificadores duplicados.")
    return datos, ruta, esperada


def tabla_variables(datos):
    filas = []
    for columna in GEOMETRY + [TARGET]:
        filas.append({
            "Variable": ETIQUETAS_VARIABLES[columna],
            "Papel": "Salida / objetivo" if columna == TARGET else "Entrada previa al ensayo",
            "Unidad": UNIDADES_VARIABLES[columna],
            "Ejemplo observado": float(datos[columna].iloc[0]),
            "Tratamiento": ("No se imputa ni transforma" if columna == TARGET else
                            "Filtro de constantes ajustado en train" if columna == "Pivot_radius" else
                            "Mediana, filtro de constantes y escala en train"),
        })
    return pd.DataFrame(filas)


def particiones_historicas(datos):
    """Convierte los identificadores publicados a posiciones sin crear otro split."""
    registros = leer_json(ROOT / "reports/feature_engineering/particiones.json")
    posiciones = {int(caso): i for i, caso in enumerate(datos.Case_n)}
    salida = {}
    for protocolo in PROTOCOLOS:
        partes = []
        for registro in registros:
            if registro["protocolo"] != protocolo:
                continue
            train = np.array([posiciones[int(c)] for c in registro["train_cases"]])
            test = np.array([posiciones[int(c)] for c in registro["test_cases"]])
            if set(train) & set(test) or set(train) | set(test) != set(range(len(datos))):
                raise ValueError("Partición histórica incompatible con la tabla congelada.")
            partes.append((train, test))
        salida[protocolo] = partes
    return salida, registros


def evidencia_preprocesamiento(modelo, train):
    """Contrasta estadísticas reales del pipeline con cálculos independientes de train."""
    imputador = modelo.named_steps["imputer"]
    medianas = train[GEOMETRY].median().to_numpy()
    np.testing.assert_allclose(imputador.statistics_, medianas, rtol=1e-12, atol=1e-12)
    representacion = modelo[:3].transform(train[GEOMETRY])
    soporte = modelo.named_steps["variance"].get_support()
    valores = np.asarray(representacion)[:, soporte]
    media = valores.mean(axis=0)
    desvio = valores.std(axis=0, ddof=0)
    escala = modelo.named_steps["scale"]
    np.testing.assert_allclose(escala.mean_, media, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(escala.scale_, desvio, rtol=1e-12, atol=1e-12)
    nombres = feature_names(modelo)
    return pd.DataFrame({
        "Variable": [ETIQUETAS_VARIABLES.get(n, n) for n in nombres],
        "Mediana de train": medianas[soporte],
        "Media calculada en train": media,
        "Media usada por scaler": escala.mean_,
        "Desviación usada por scaler": escala.scale_,
        "Diferencia absoluta de medias": np.abs(media - escala.mean_),
    })


def cargar_ejecucion(directorio=DEFAULT_RUN, verificar=True):
    directorio = Path(directorio)
    if verificar:
        from .verificar_ajuste import verify
        resultado = verify(root=ROOT, output=directorio)
        if resultado.get("passed") is not True:
            raise ValueError("La ejecución no superó la verificación.")
    tablas = {}
    for nombre in ("metricas_resumen", "predicciones_oof", "metricas_folds", "busquedas"):
        tablas[nombre] = pd.read_csv(directorio / f"{nombre}.csv")
    tablas["manifest"] = leer_json(directorio / "manifest.json")
    tablas["directorio"] = directorio
    return tablas


def _base(resumen):
    if "Baseline_Ridge" not in set(resumen.modelo):
        raise ValueError("No se encuentra la línea base histórica Baseline_Ridge.")
    return "Baseline_Ridge"


def _selector(resumen):
    return "Selector_interno" if "Selector_interno" in set(resumen.modelo) else None


def tabla_busquedas(configuracion):
    from sklearn.model_selection import ParameterGrid
    from .ajuste_modelos import model_specs
    filas = [{"Modelo": "Ridge · línea base", "Parámetros ajustados": "Ninguno; alpha=1, solver=svd",
              "Combinaciones": 1, "Selección": "Referencia histórica fija"}]
    for nombre, (_, rejilla) in model_specs(configuracion).items():
        filas.append({"Modelo": nombre_modelo(nombre),
                      "Parámetros ajustados": ", ".join(p.replace("model__", "") for p in rejilla),
                      "Combinaciones": len(ParameterGrid(rejilla)),
                      "Selección": "Menor MAE de validación interna"})
    return pd.DataFrame(filas)


def rejilla_ridge(configuracion):
    from .ajuste_modelos import model_specs
    return model_specs(configuracion)["Ridge"][1]


def tabla_ejecucion(ejecucion):
    manifest = ejecucion["manifest"]
    filas = [
        ("Estado de la ejecución", manifest["estado"]),
        ("Inicio del experimento (UTC)", manifest["inicio_utc"]),
        ("Fin del experimento (UTC)", manifest["fin_utc"]),
        ("Tiempo transcurrido (s)", round(float(manifest["tiempo_total_s"]), 2)),
        ("Datos y splits inalterados", manifest["data_sha256_antes"] == manifest["data_sha256_despues"]),
        ("Selección de parámetros", "Solo en validación interna; score = MAE"),
        ("Predicciones comparables", "Mismos casos y particiones externas para todos los modelos"),
    ]
    return pd.DataFrame(filas, columns=["Registro", "Valor"])


def frecuencias_ajuste(ejecucion):
    busquedas = ejecucion["busquedas"]
    elegidos = busquedas[busquedas.selected.astype(str).str.lower().eq("true")].copy()
    frecuencia = (elegidos.groupby(["protocolo", "modelo", "params_json"], sort=False)
                  .size().rename("frecuencia").reset_index())
    frecuencia["porcentaje"] = 100 * frecuencia.frecuencia / frecuencia.groupby(
        ["protocolo", "modelo"]).frecuencia.transform("sum")
    principal = frecuencia[frecuencia.protocolo == "configuracion"].copy()
    principal = principal.sort_values(["modelo", "frecuencia", "params_json"], ascending=[True, False, True])
    # Una fila por algoritmo: combinación modal; todos los candidatos están en busquedas.csv.
    principal = principal.groupby("modelo", sort=False).head(1)
    principal["Modelo"] = principal.modelo.map(nombre_modelo)
    principal["Combinación más frecuente"] = principal.params_json.apply(
        lambda v: "; ".join(f"{k.replace('model__', '')}={val}" for k, val in json.loads(v).items()))
    principal["Frecuencia de selección (%)"] = principal.porcentaje.round(2)
    tabla = principal[["Modelo", "Combinación más frecuente", "Frecuencia de selección (%)"]]
    selector = ejecucion["metricas_folds"]
    selector = selector[selector.modelo == "Selector_interno"]
    modelos = (selector.groupby(["protocolo", "modelo_seleccionado"])
               .size().rename("frecuencia").reset_index())
    modelos["porcentaje"] = 100 * modelos.frecuencia / modelos.groupby("protocolo").frecuencia.transform("sum")
    return tabla.reset_index(drop=True), modelos


def figura_seleccion(ejecucion, frecuencia_modelos):
    _tema()
    modelos = list(dict.fromkeys(frecuencia_modelos.modelo_seleccionado.tolist()))
    colores = ["#2563a6", "#008579", "#bf8a30", "#7955a2", "#b94a65", "#497778", "#5c638c"]
    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    acumulado = np.zeros(len(PROTOCOLOS))
    for nombre, color in zip(modelos, colores):
        valores = np.array([frecuencia_modelos.loc[(frecuencia_modelos.protocolo == p) &
                            (frecuencia_modelos.modelo_seleccionado == nombre), "porcentaje"].sum() for p in PROTOCOLOS])
        ax.barh(range(len(PROTOCOLOS)), valores, left=acumulado, label=nombre_modelo(nombre), color=color, height=.5)
        for i, valor in enumerate(valores):
            if valor >= 6:
                ax.text(acumulado[i] + valor / 2, i, f"{valor:.1f} %", ha="center", va="center", color="white", fontweight="bold")
        acumulado += valores
    ax.set_yticks(range(len(PROTOCOLOS)), list(PROTOCOLOS.values()))
    ax.set_xlim(0, 100)
    ax.set_xlabel("Frecuencia de elección del algoritmo por la validación interna (%)")
    ax.set_title("¿Qué algoritmo eligió el selector sin ver el test externo?", loc="left", fontweight="bold", pad=20)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=3, frameon=False)
    ax.grid(axis="x", alpha=.13)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _guardar(fig, ejecucion["directorio"], "08_seleccion_interna", [ejecucion["directorio"] / "metricas_folds.csv"])


def _orden(resumen, protocolo="configuracion"):
    tabla = resumen[resumen.protocolo == protocolo].set_index("modelo")
    base, selector = _base(resumen), _selector(resumen)
    modelos = [m for m in tabla.index if m not in {base, selector}]
    modelos = sorted(modelos, key=lambda m: float(tabla.loc[m, "MAE_N"]))
    return [base] + modelos + ([selector] if selector else [])


def tabla_metricas(resumen, protocolo="configuracion"):
    tabla = resumen[resumen.protocolo == protocolo].set_index("modelo").loc[_orden(resumen)].copy()
    base = float(tabla.loc[_base(resumen), "MAE_N"])
    tabla["Reducción MAE (%)"] = 100 * (base - tabla.MAE_N) / base
    tabla["Modelo"] = [nombre_modelo(m) for m in tabla.index]
    tabla = tabla[["Modelo", "MAE_N", "RMSE_N", "MedAE_N", "R2_OOF", "tiempo_s", "Reducción MAE (%)"]]
    return tabla.rename(columns={"MAE_N": "MAE (N)", "RMSE_N": "RMSE (N)",
                                 "MedAE_N": "MedAE (N)", "R2_OOF": "R² OOF",
                                 "tiempo_s": "Tiempo total CV (s)"}).reset_index(drop=True)


def estilo_metricas(tabla):
    def colorear(fila):
        color = "#eaf1fb" if "línea base" in fila["Modelo"] else (
            "#e3f3ed" if fila["Reducción MAE (%)"] > 0 else "#fae9ee")
        return [f"background-color: {color}; color: #253447"] * len(fila)
    return (tabla.style.apply(colorear, axis=1).hide(axis="index")
            .format({c: "{:.3f}" for c in tabla.columns if c != "Modelo"})
            .set_properties(**{"padding": "7px 10px", "text-align": "right"})
            .set_properties(subset=["Modelo"], **{"text-align": "left"})
            .set_table_styles([{"selector": "th", "props": [("background", "#253447"),
                                                               ("color", "white"), ("padding", "9px")]}]))


def antes_despues_ajuste(resumen):
    historico = ROOT / "reports/feature_engineering"
    ruta = historico / "tables/comparacion_modelos.csv"
    manifest = leer_json(historico / "manifest.json")
    if huella(ruta) != manifest["artifacts_sha256"]["tables/comparacion_modelos.csv"]:
        raise ValueError("Cambió la tabla de modelos fijos de la Semana 5.")
    anterior = pd.read_csv(ruta)
    anterior = anterior[anterior.protocolo == "configuracion"][["modelo", "MAE_N"]]
    actual = resumen[resumen.protocolo == "configuracion"][["modelo", "MAE_N"]]
    comparacion = anterior.merge(actual, on="modelo", suffixes=("_previo", "_ajustado"), validate="one_to_one")
    comparacion["Reducción MAE (%)"] = 100 * (comparacion.MAE_N_previo - comparacion.MAE_N_ajustado) / comparacion.MAE_N_previo
    comparacion["modelo"] = comparacion.modelo.map(nombre_modelo)
    return comparacion.rename(columns={"modelo": "Modelo", "MAE_N_previo": "MAE fijo previo (N)",
                                       "MAE_N_ajustado": "MAE ajustado (N)"})


def _tema():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelcolor": TINTA,
                         "text.color": TINTA, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})


def _guardar(fig, directorio, nombre, fuentes):
    destino = Path(directorio) / "figures"
    destino.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(destino / f"{nombre}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    manifest_path = destino / "manifest_figuras.json"
    manifest = leer_json(manifest_path) if manifest_path.exists() else {"figuras": {}}
    manifest["figuras"][nombre] = {
        "fuentes": {str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/"): huella(p)
                    for p in fuentes},
        "archivos": {f"figures/{nombre}.{ext}": huella(destino / f"{nombre}.{ext}")
                     for ext in ("png", "svg")},
    }
    manifest["generado_utc"] = datetime.now(timezone.utc).isoformat()
    _guardar_json(manifest_path, manifest)
    return destino / f"{nombre}.png"


def figura_particiones(datos, registros, internos, directorio=DEFAULT_RUN):
    """Se dibujan pertenencias reales; no se inventan tamaños o particiones."""
    _tema()
    primero = next(r for r in registros if r["protocolo"] == "configuracion")
    casos = datos.Case_n.astype(int).tolist()
    matriz = np.zeros((1 + len(internos), len(casos)), dtype=int)
    test = set(primero["test_cases"])
    matriz[:, [i for i, c in enumerate(casos) if c in test]] = 2
    train_ids = primero["train_cases"]
    for fila, (_, validacion) in enumerate(internos, 1):
        val_ids = {train_ids[int(i)] for i in validacion}
        matriz[fila, [i for i, c in enumerate(casos) if c in val_ids]] = 1
    fig, ax = plt.subplots(figsize=(13.5, 4))
    ax.imshow(matriz, aspect="auto", cmap=ListedColormap(["#c4d5ea", "#2d9887", "#b94a65"]), vmin=0, vmax=2)
    ax.set_xticks(range(len(casos)), [str(c) for c in casos], fontsize=8)
    ax.set_yticks(range(len(matriz)), ["Split externo 1"] + [f"Interno {i}" for i in range(1, len(matriz))])
    ax.set_xlabel("Identificador del caso; el grupo externo no participa en ninguna búsqueda")
    ax.set_title("Validación anidada · ejemplo real del primer split externo", loc="left", fontweight="bold")
    ax.set_xticks(np.arange(-.5, len(casos), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(matriz), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.legend(handles=[Patch(color="#c4d5ea", label="Entrenamiento"),
                       Patch(color="#2d9887", label="Validación interna"),
                       Patch(color="#b94a65", label="Evaluación externa reservada")],
              loc="upper center", bbox_to_anchor=(.5, -.25), ncol=3, frameon=False)
    fig.tight_layout()
    return _guardar(fig, directorio, "01_validacion_anidada", [ROOT / "reports/feature_engineering/particiones.json"])


def figura_metricas(ejecucion):
    _tema()
    resumen, directorio = ejecucion["metricas_resumen"], ejecucion["directorio"]
    orden = _orden(resumen)
    datos = resumen[resumen.protocolo == "configuracion"].set_index("modelo").loc[orden]
    base, selector = _base(resumen), _selector(resumen)
    fig, ejes = plt.subplots(1, 3, figsize=(16, 6.2), sharey=True)
    for ax, columna, etiqueta in zip(ejes, ["MAE_N", "RMSE_N", "R2_OOF"], ["MAE (N) ↓", "RMSE (N) ↓", "R² OOF ↑"]):
        def mejora(modelo):
            return (datos.loc[modelo, columna] > datos.loc[base, columna] if columna == "R2_OOF"
                    else datos.loc[modelo, columna] < datos.loc[base, columna])
        colores = [AZUL if m == base else VERDE if mejora(m) else ROSA for m in orden]
        barras = ax.barh(np.arange(len(orden)), datos[columna], color=colores, height=.66)
        for i, barra in enumerate(barras):
            if orden[i] == selector:
                barra.set_hatch("//")
                barra.set_edgecolor(TINTA)
        ax.axvline(datos.loc[base, columna], color=AZUL, ls="--", lw=1.1)
        ax.bar_label(barras, fmt="%.3f", padding=4, fontsize=9)
        ax.set_xlabel(etiqueta)
        ax.grid(axis="x", alpha=.16)
        ax.set_axisbelow(True)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + .22 * (hi - lo))
    ejes[0].set_yticks(np.arange(len(orden)), [nombre_modelo(m) for m in orden])
    ejes[0].invert_yaxis()
    fig.suptitle("Resultados fuera de muestra · configuración excluida", x=.01, ha="left", fontweight="bold", fontsize=15)
    fig.text(.01, .015, "Línea discontinua: baseline. Rayado: selección de algoritmo e hiperparámetros realizada solo dentro de train.", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .95))
    return _guardar(fig, directorio, "02_metricas_modelos", [directorio / "metricas_resumen.csv"])


def figura_predicciones(ejecucion, residuales=False):
    _tema()
    pred, resumen = ejecucion["predicciones_oof"], ejecucion["metricas_resumen"]
    directorio = ejecucion["directorio"]
    orden = _orden(resumen)
    valores = pred[pred.protocolo == "configuracion"]
    familias = sorted(valores.familia.unique())
    tonos = ["#2563a6", "#008579", "#b77b23"]
    nfilas = int(np.ceil(len(orden) / 3))
    fig, ejes = plt.subplots(nfilas, 3, figsize=(14, nfilas * 3.5), sharex=True, sharey=True, squeeze=False)
    minimo = float(min(valores.observado_N.min(), valores.predicho_N.min())) - 8
    maximo = float(max(valores.observado_N.max(), valores.predicho_N.max())) + 8
    for ax, modelo in zip(ejes.flat, orden):
        parte = valores[valores.modelo == modelo]
        for familia, color in zip(familias, tonos):
            puntos = parte[parte.familia == familia]
            ax.scatter(puntos.predicho_N if residuales else puntos.observado_N,
                       puntos.residuo_N if residuales else puntos.predicho_N,
                       c=color, s=38, edgecolors="white", linewidth=.5, alpha=.85, label=f"Familia Y={int(familia)}")
        if residuales:
            ax.axhline(0, color=TINTA, lw=1, ls="--")
        else:
            ax.plot([minimo, maximo], [minimo, maximo], color=TINTA, lw=1, ls="--")
            ax.set_xlim(minimo, maximo)
            ax.set_ylim(minimo, maximo)
        ax.set_title(nombre_modelo(modelo), loc="left", fontweight="bold", fontsize=11)
        ax.grid(alpha=.15)
    for ax in list(ejes.flat)[len(orden):]:
        ax.set_visible(False)
    for ax in ejes[-1]:
        ax.set_xlabel("Carga estimada (N)" if residuales else "Carga observada (N)")
    for ax in ejes[:, 0]:
        ax.set_ylabel("Observado − estimado (N)" if residuales else "Carga estimada (N)")
    handles, labels = ejes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(familias), frameon=False)
    titulo = "Residuos OOF · signo positivo = subestimación" if residuales else "Carga de primera falla · predicciones OOF"
    fig.suptitle(titulo, x=.01, ha="left", fontweight="bold", fontsize=15)
    fig.tight_layout(rect=(0, .04, 1, .96))
    nombre = "04_residuales_modelos" if residuales else "03_observado_estimado"
    return _guardar(fig, directorio, nombre, [directorio / "predicciones_oof.csv"])


def figura_familias(ejecucion):
    _tema()
    datos = ejecucion["predicciones_oof"]
    orden = _orden(ejecucion["metricas_resumen"])
    agrupado = datos.groupby(["protocolo", "modelo", "familia"]).error_absoluto_N.mean()
    familias = sorted(datos.familia.unique())
    matrices = [np.array([[agrupado.loc[(p, m, f)] for f in familias] for m in orden]) for p in PROTOCOLOS]
    vmax = max(float(m.max()) for m in matrices)
    fig, ejes = plt.subplots(1, 2, figsize=(14.5, 6.5), sharey=True)
    fig.subplots_adjust(left=.2, right=.95, top=.88, bottom=.12, wspace=.13)
    for ax, protocolo, matriz in zip(ejes, PROTOCOLOS, matrices):
        im = ax.imshow(matriz, aspect="auto", cmap="YlGnBu", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(familias)), [f"Y={int(f)}" for f in familias])
        ax.set_title(PROTOCOLOS[protocolo], loc="left", fontweight="bold")
        ax.set_xlabel("Familia evaluada")
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                ax.text(j, i, f"{matriz[i,j]:.2f}", ha="center", va="center",
                        color="white" if matriz[i,j] > vmax * .5 else TINTA, fontsize=10)
    ejes[0].set_yticks(range(len(orden)), [nombre_modelo(m) for m in orden])
    fig.colorbar(im, ax=list(ejes), label="MAE por familia (N) · menor es mejor", shrink=.8, pad=.025)
    fig.suptitle("Error por arquitectura · dos preguntas de generalización", x=.01, ha="left", fontweight="bold", fontsize=15)
    return _guardar(fig, ejecucion["directorio"], "05_error_familias", [ejecucion["directorio"] / "predicciones_oof.csv"])


def figura_mejora_folds(ejecucion):
    _tema()
    datos, resumen = ejecucion["metricas_folds"], ejecucion["metricas_resumen"]
    base = _base(resumen)
    orden = [m for m in _orden(resumen) if m != base]
    matrices, columnas = [], []
    for protocolo in PROTOCOLOS:
        pivote = datos[datos.protocolo == protocolo].pivot(index="modelo", columns="fold", values="MAE_N")
        matrices.append(np.array([pivote.loc[base] - pivote.loc[m] for m in orden]))
        columnas.append(list(pivote.columns))
    limite = max(1.0, max(float(np.abs(m).max()) for m in matrices))
    from matplotlib.colors import LinearSegmentedColormap
    mapa = LinearSegmentedColormap.from_list("mejora", [ROSA, "#ffffff", VERDE])
    fig, ejes = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [4, 1]}, sharey=True)
    fig.subplots_adjust(left=.18, right=.96, top=.86, bottom=.14, wspace=.11)
    for ax, protocolo, matriz, folds in zip(ejes, PROTOCOLOS, matrices, columnas):
        im = ax.imshow(matriz, aspect="auto", cmap=mapa, norm=TwoSlopeNorm(vmin=-limite, vcenter=0, vmax=limite))
        ax.set_xticks(range(len(folds)), [str(f) for f in folds], fontsize=8)
        ax.set_xlabel("Split externo")
        ax.set_title(PROTOCOLOS[protocolo], loc="left", fontsize=11, fontweight="bold")
    ejes[0].set_yticks(range(len(orden)), [nombre_modelo(m) for m in orden])
    fig.colorbar(im, ax=list(ejes), label="MAE baseline − MAE modelo (N) · positivo = mejora", shrink=.85, pad=.025)
    fig.suptitle("Dónde mejora y dónde empeora cada modelo", x=.01, ha="left", fontweight="bold", fontsize=15)
    return _guardar(fig, ejecucion["directorio"], "06_mejora_por_split", [ejecucion["directorio"] / "metricas_folds.csv"])


def figura_costo(ejecucion):
    _tema()
    resumen = ejecucion["metricas_resumen"]
    base, selector = _base(resumen), _selector(resumen)
    fig, ejes = plt.subplots(1, 2, figsize=(15, 5.8))
    for ax, protocolo in zip(ejes, PROTOCOLOS):
        datos = resumen[resumen.protocolo == protocolo].set_index("modelo")
        referencia = float(datos.loc[base, "MAE_N"])
        for i, modelo in enumerate(_orden(resumen)):
            fila = datos.loc[modelo]
            color = AZUL if modelo == base else VERDE if fila.MAE_N < referencia else ROSA
            ax.scatter(max(float(fila.tiempo_s), .001), fila.MAE_N, color=color,
                       s=100 if modelo == selector else 60, marker="D" if modelo == selector else "o", zorder=3)
            desplazamiento_y = 6 if i % 2 == 0 else -12
            if protocolo == "configuracion" and modelo == "GaussianProcess":
                desplazamiento_y = 8
            ax.annotate(nombre_modelo(modelo), (max(float(fila.tiempo_s), .001), fila.MAE_N),
                        xytext=(5, desplazamiento_y), textcoords="offset points", fontsize=8)
        ax.axhline(referencia, color=AZUL, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("Tiempo total CV + búsqueda (s) · escala logarítmica")
        ax.set_ylabel("MAE OOF (N) ↓")
        ax.set_title(PROTOCOLOS[protocolo], loc="left", fontweight="bold")
        ax.grid(alpha=.15)
        ax.margins(x=.35, y=.25)
    fig.suptitle("Precisión y costo de cómputo", x=.01, ha="left", fontweight="bold", fontsize=15)
    fig.text(.01, .01, "Tiempos medidos en esta ejecución; no representan latencia de inferencia ni una prueba de rendimiento entre equipos.", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .94))
    return _guardar(fig, ejecucion["directorio"], "07_costo_precision", [ejecucion["directorio"] / "metricas_resumen.csv"])


def interpretacion(ejecucion):
    """Conclusiones calculadas; separa ranking observado del selector anidado."""
    resumen = ejecucion["metricas_resumen"]
    p = resumen[resumen.protocolo == "configuracion"].set_index("modelo")
    s = resumen[resumen.protocolo == "familia"].set_index("modelo")
    base, selector = _base(resumen), _selector(resumen)
    candidatos = [m for m in p.index if m not in {base, selector}]
    mejor = p.loc[candidatos, "MAE_N"].idxmin()
    mejora = 100 * (p.loc[base, "MAE_N"] - p.loc[mejor, "MAE_N"]) / p.loc[base, "MAE_N"]
    lineas = [
        f"**Ranking descriptivo:** {nombre_modelo(mejor)} obtuvo MAE **{p.loc[mejor, 'MAE_N']:.3f} N**, "
        f"frente a **{p.loc[base, 'MAE_N']:.3f} N** del baseline ({mejora:+.2f} % de reducción). "
        "Elegirlo después de comparar estos resultados requiere una evaluación independiente posterior.",
    ]
    if selector:
        cambio = 100 * (p.loc[base, "MAE_N"] - p.loc[selector, "MAE_N"]) / p.loc[base, "MAE_N"]
        lineas.append(f"**Selección anidada:** al elegir algoritmo y parámetros solo en validación interna, "
                      f"el MAE externo fue **{p.loc[selector, 'MAE_N']:.3f} N** ({cambio:+.2f} % frente al baseline). "
                      "Este resultado evalúa el procedimiento de selección completo.")
        referencia = selector
    else:
        referencia = mejor
    cambio_familia = 100 * (s.loc[mejor, "MAE_N"] - s.loc[base, "MAE_N"]) / s.loc[base, "MAE_N"]
    lineas.append(f"**Familias nuevas:** {nombre_modelo(mejor)} obtuvo MAE **{s.loc[mejor, 'MAE_N']:.3f} N** "
                  f"({cambio_familia:+.2f} % de variación del error frente al baseline). "
                  f"{nombre_modelo(referencia)} obtuvo MAE **{s.loc[referencia, 'MAE_N']:.3f} N** "
                  f"y R² **{s.loc[referencia, 'R2_OOF']:.3f}** al reservar familias completas. "
                  + ("El R² negativo indica que la transferencia a otras arquitecturas no está resuelta."
                     if s.loc[referencia, "R2_OOF"] < 0 else
                     "La diferencia entre protocolos debe considerarse antes de generalizar a otra arquitectura."))
    config = leer_json(ejecucion["directorio"] / "config_ejecutada.json")
    busquedas = ejecucion["busquedas"]
    elegidos = busquedas[(busquedas.protocolo == "configuracion") & (busquedas.modelo == "Ridge") &
                        busquedas.selected.astype(str).str.lower().eq("true")]
    alphas = elegidos.params_json.map(lambda valor: json.loads(valor)["model__alpha"])
    limite = min(config["modelos"]["Ridge"]["grid"]["model__alpha"])
    if (alphas == limite).mean() >= .5:
        lineas.append("**Ajuste de Ridge:** predominó una regularización baja y se alcanzó el límite inferior "
                      "de la rejilla. El resultado no demuestra un óptimo global; la búsqueda declarada se mantiene.")
    lineas.append("**Alcance:** el ajuste respeta las particiones y aprende el preprocesamiento solo con train. "
                  "El EDA previo utilizó esta campaña completa; no existe un holdout externo intacto. "
                  "Estas métricas no demuestran desempeño en producción ni permiten afirmar significancia estadística.")
    return "\n\n".join(lineas)


def _celda(codigo):
    return nbformat.v4.new_code_cell(textwrap.dedent(codigo).strip())


def _md(texto):
    return nbformat.v4.new_markdown_cell(textwrap.dedent(texto).strip())


def crear_notebook():
    celdas = [
        _md("""# 05. Modelos ajustados y validación

        **Objetivo:** reducir el error al predecir la **carga de primera falla (N)** a partir de geometría previa al ensayo.
        Se mantiene la tabla y los splits de la Semana 5; el cambio experimental es el ajuste de modelos.
        """),
        _celda("""
        from pathlib import Path
        import sys, json, random
        import numpy as np
        import pandas as pd
        from IPython.display import display, Markdown, Image
        from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, GridSearchCV

        ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'src/feature_engineering.py').exists())
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src.feature_engineering import TARGET, make_pipeline, nominal_groups
        from src.transformaciones_eda import GEOMETRY
        from src import reporte_ajuste as informe

        SEMILLA = 42
        random.seed(SEMILLA)
        np.random.seed(SEMILLA)
        RUN_DIR = ROOT / 'reports/modelos_ajustados'
        pd.set_option('display.max_colwidth', 90)
        """),
        _md("""## 5.1 Datos e información de entrada

        El objetivo es una fuerza medida, no una clase de rotura. Identificadores, cargas posteriores,
        desplazamientos y energías no se usan como predictores. [Definiciones de variables](../docs/diccionario_datos.md).
        """),
        _celda("""
        datos, ruta_datos, hash_datos = informe.datos_congelados()
        X = datos[GEOMETRY].copy()
        y = datos[TARGET].copy()
        assert TARGET not in X.columns and 'Case_n' not in X.columns
        display(informe.tabla_variables(datos).style.hide(axis='index').format({'Ejemplo observado': '{:.3f}'}))
        print('Snapshot histórico verificado · SHA-256:', hash_datos)
        """),
        _md("""## 5.2 Particiones externas e internas

        **Configuración:** misma cantidad de celdas en Y, altura de fibra y altura de pivote.
        **Familia:** misma cantidad de celdas en Y; identifica una arquitectura del diseño.
        El protocolo principal reserva configuraciones y el secundario familias completas.
        La búsqueda de parámetros ocurre únicamente dentro del entrenamiento de cada split externo.
        """),
        _celda("""
        # Se recuperan los casos del split histórico; no se genera otra partición final.
        splits, registros = informe.particiones_historicas(datos)
        grupos_config = nominal_groups(datos, ['n_cells_Y', 'Fiber_height', 'Pivot_height'])
        grupos_familia = datos.n_cells_Y.astype(int).astype(str)
        train, test = splits['configuracion'][0]
        assert set(grupos_config.iloc[train]).isdisjoint(grupos_config.iloc[test])

        # Índices internos relativos exclusivamente al bloque train externo.
        cv_interna = list(GroupKFold(n_splits=5).split(X.iloc[train], y.iloc[train], grupos_config.iloc[train]))
        for ajuste, validacion in cv_interna:
            assert set(grupos_config.iloc[train].iloc[ajuste]).isdisjoint(grupos_config.iloc[train].iloc[validacion])
            assert set(train[ajuste]).isdisjoint(test) and set(train[validacion]).isdisjoint(test)

        # En el protocolo secundario se excluye una familia también dentro del train externo.
        train_familia, test_familia = splits['familia'][0]
        cv_familia = list(LeaveOneGroupOut().split(
            X.iloc[train_familia], y.iloc[train_familia], grupos_familia.iloc[train_familia]))
        assert set(grupos_familia.iloc[train_familia]).isdisjoint(grupos_familia.iloc[test_familia])
        display(Image(filename=str(informe.figura_particiones(datos, registros, cv_interna, RUN_DIR))))
        """),
        _md("""## 5.3 Preprocesamiento: ajuste solo en train

        Geometría original → imputación de mediana → filtro de constantes → estandarización → regresor.
        No se aplican PCA, target encoding, balanceo ni nuevas características en esta comparación.
        """),
        _celda("""
        # Ejecución real del pipeline sobre el train del primer split externo.
        ejemplo_pipeline = make_pipeline('A', alpha=1.0)
        ejemplo_pipeline.fit(X.iloc[train], y.iloc[train])
        display(ejemplo_pipeline)
        control_train = informe.evidencia_preprocesamiento(ejemplo_pipeline, X.iloc[train])
        display(control_train.style.hide(axis='index').format(precision=5))
        print('Medianas y estadísticas del scaler comprobadas contra train; test no intervino en fit.')
        """),
        _md("""## 5.4 Modelos y búsqueda

        El baseline conserva Ridge con α=1. Cada regresor ajustado usa el mismo preprocesamiento,
        entradas y evaluación externa. El criterio interno es **MAE**, en newtons.
        """),
        _celda("""
        # El registro de modelos y todos los valores de la búsqueda están versionados.
        from src.ajuste_modelos import load_config
        configuracion = load_config(ROOT)
        display(informe.tabla_busquedas(configuracion).style.hide(axis='index'))
        """),
        _celda("""
        # Ejemplo ejecutado del ajuste interno de Ridge; no se consulta y_test para elegir alpha.
        parametros_ridge = informe.rejilla_ridge(configuracion)
        busqueda_ejemplo = GridSearchCV(
            estimator=make_pipeline('A'), param_grid=parametros_ridge,
            scoring='neg_mean_absolute_error', cv=cv_interna,
            refit=True, n_jobs=1, error_score='raise', return_train_score=False,
        )
        busqueda_ejemplo.fit(X.iloc[train], y.iloc[train])
        evidencia_busqueda = pd.DataFrame(busqueda_ejemplo.cv_results_)[
            ['params', 'mean_test_score', 'rank_test_score']].copy()
        evidencia_busqueda['MAE interno (N)'] = -evidencia_busqueda.pop('mean_test_score')
        display(evidencia_busqueda.sort_values('rank_test_score').rename(
            columns={'params':'Parámetros', 'rank_test_score':'Orden interno'}).style.hide(axis='index').format(precision=4))
        print('Parámetros elegidos dentro de train:', busqueda_ejemplo.best_params_)
        """),
        _md("""## 5.5 Experimento completo y registros

        La ejecución publicada contiene las búsquedas y predicciones de todos los splits.
        Por defecto se comprueban sus hashes y resultados. `REEJECUTAR = True` entrena nuevamente
        en una carpeta con timestamp, conservando la ejecución publicada.
        """),
        _celda("""
        REEJECUTAR = False
        if REEJECUTAR:
            from datetime import datetime, timezone
            from src.ajuste_modelos import run
            RUN_DIR = ROOT / 'logs/modelos_ajustados' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            run(root=ROOT, output=RUN_DIR)
            # Conservar también el esquema inicial junto a la nueva ejecución.
            informe.figura_particiones(datos, registros, cv_interna, RUN_DIR)

        ejecucion = informe.cargar_ejecucion(RUN_DIR, verificar=True)
        resumen = ejecucion['metricas_resumen']
        predicciones = ejecucion['predicciones_oof']
        print('Resultados publicados verificados; no se ha repetido el entrenamiento completo.' if not REEJECUTAR
              else 'Entrenamiento completo ejecutado y verificado en una carpeta nueva.')
        display(informe.tabla_ejecucion(ejecucion).style.hide(axis='index'))
        """),
        _md("""## 5.6 Comparación de métricas

        **MAE:** error absoluto medio; **RMSE:** penaliza más los errores grandes; **MedAE:** error absoluto mediano;
        **R² OOF:** ajuste agregado de predicciones externas. Menor error es mejor; mayor R² es mejor.
        El tiempo incluye búsqueda y ajustes de CV. La selección interna es un procedimiento adicional, no un octavo algoritmo.
        """),
        _celda("""
        display(Markdown('**Configuración excluida — comparación principal**'))
        display(informe.estilo_metricas(informe.tabla_metricas(resumen, 'configuracion')))
        display(Image(filename=str(informe.figura_metricas(ejecucion))))
        display(Markdown('**Familia excluida — transferencia a otra arquitectura**'))
        display(informe.estilo_metricas(informe.tabla_metricas(resumen, 'familia')))
        display(Markdown('**Antes y después del ajuste — mismos algoritmos de la Semana 5**'))
        display(informe.antes_despues_ajuste(resumen).style.hide(axis='index').format(
            {'MAE fijo previo (N)': '{:.3f}', 'MAE ajustado (N)': '{:.3f}', 'Reducción MAE (%)': '{:+.2f}'})
            .background_gradient(subset=['Reducción MAE (%)'], cmap='RdYlGn', vmin=-20, vmax=20))
        """),
        _md("""## 5.7 Predicciones, residuos y errores por grupo

        Cada punto proviene de una predicción realizada cuando su grupo estaba reservado.
        En los residuos, valores positivos indican subestimación de la carga de primera falla.
        """),
        _celda("""
        display(Image(filename=str(informe.figura_predicciones(ejecucion))))
        display(Image(filename=str(informe.figura_predicciones(ejecucion, residuales=True))))
        """),
        _celda("""
        display(Image(filename=str(informe.figura_familias(ejecucion))))
        display(Image(filename=str(informe.figura_mejora_folds(ejecucion))))
        display(Image(filename=str(informe.figura_costo(ejecucion))))
        """),
        _md("""## 5.8 Estabilidad del ajuste

        La frecuencia indica qué parámetros o algoritmo eligió la validación interna de cada split.
        No mide importancia de variables ni probabilidad de que un modelo sea el mejor.
        """),
        _celda("""
        frecuencia_parametros, frecuencia_modelos = informe.frecuencias_ajuste(ejecucion)
        display(frecuencia_parametros.style.hide(axis='index')
            .set_properties(subset=['Combinación más frecuente'], **{'max-width':'470px', 'white-space':'normal'}))
        display(Image(filename=str(informe.figura_seleccion(ejecucion, frecuencia_modelos))))
        """),
        _md("## 5.9 Interpretación de resultados"),
        _celda("display(Markdown(informe.interpretacion(ejecucion)))"),
        _md("""## 5.10 Reproducibilidad

        Configuración, semillas, timestamps, búsqueda completa, particiones internas, estadísticas de train,
        predicciones OOF y métricas quedan en [los artefactos de ejecución](../reports/modelos_ajustados/).
        """),
        _celda("""
        auditoria = informe.leer_json(RUN_DIR / 'verificacion.json')
        display(pd.DataFrame([
            {'Control': 'Auditoría ejecutada', 'Resultado': auditoria['status']},
            {'Control': 'Fecha UTC de comprobación', 'Resultado': auditoria.get('checked_utc', '')},
            {'Control': 'Snapshot después de evaluar', 'Resultado': informe.huella(ruta_datos)},
            {'Control': 'Snapshot sin cambios', 'Resultado': informe.huella(ruta_datos) == hash_datos},
            {'Control': 'Semilla declarada', 'Resultado': SEMILLA},
        ]).style.hide(axis='index'))
        assert informe.huella(ruta_datos) == hash_datos
        """),
        _md("""Reproducir el notebook publicado y comprobar sus resultados:

        ```bash
        python -m src.reporte_ajuste --execute
        ```

        Entrenar otra ejecución por consola, usando una carpeta que aún no exista:

        ```bash
        python -m src.ajuste_modelos --output logs/modelos_ajustados/repeticion_01
        python -m src.verificar_ajuste --run-dir logs/modelos_ajustados/repeticion_01
        ```

        Para entrenar y visualizar una ejecución nueva desde este cuaderno, usar `REEJECUTAR = True` en 5.5.
        Las ejecuciones por consola no sustituyen automáticamente los resultados publicados.

        Código: [experimento](../src/ajuste_modelos.py), [verificación](../src/verificar_ajuste.py),
        [figuras y notebook](../src/reporte_ajuste.py).
        Referencias técnicas: [validación anidada](https://scikit-learn.org/1.7/auto_examples/model_selection/plot_nested_cross_validation_iris.html),
        [prevención de fuga de información](https://scikit-learn.org/1.7/common_pitfalls.html).
        """),
    ]
    notebook = nbformat.v4.new_notebook(cells=celdas, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "titulo": "Modelos ajustados y validación de carga de primera falla",
    })
    for numero, celda in enumerate(notebook.cells):
        celda["id"] = f"ajuste-{numero:02d}"
    nbformat.validate(notebook)
    return notebook


def ejecutar_notebook(notebook):
    from nbclient import NotebookClient
    log = []
    def progreso(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            mensaje = f"{datetime.now(timezone.utc).isoformat()} · celda {cell_index} completada"
            log.append(mensaje)
            print(mensaje, flush=True)
    NotebookClient(notebook, timeout=900, kernel_name="python3", allow_errors=False,
                   on_cell_executed=progreso).execute(cwd=str(ROOT))
    nbformat.validate(notebook)
    write_bytes_atomic(DEFAULT_RUN / "ejecucion_notebook.log", ("\n".join(log) + "\nKernel limpio; sin errores.\n").encode("utf-8"))
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Ejecuta el notebook en un kernel limpio.")
    args = parser.parse_args()
    notebook = crear_notebook()
    if args.execute:
        notebook = ejecutar_notebook(notebook)
    write_bytes_atomic(NOTEBOOK, nbformat.writes(notebook).encode("utf-8"))
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
