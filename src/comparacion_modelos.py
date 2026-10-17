"""Compara cinco regresores sobre la representación A y particiones compartidas.

La comparación cambia solamente el regresor. No selecciona características,
no busca hiperparámetros y no utiliza las respuestas reservadas al ajustar.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

from .eda import save_figure_atomic, write_bytes_atomic
from .experimentos_fe import metrics, save_csv, save_json
from .feature_engineering import TARGET, feature_names, make_pipeline
from .transformaciones_eda import GEOMETRY

MODELOS = ("Ridge", "ElasticNet", "SVR_RBF", "RandomForest", "GaussianProcess")
COLORES = {"Ridge": "#2563a6", "ElasticNet": "#008579", "SVR_RBF": "#c78627",
           "RandomForest": "#7955a2", "GaussianProcess": "#c65c68"}


def crear_modelo(nombre, configuracion):
    """Una instancia nueva por partición, sin estado aprendido reutilizado."""
    especificacion = configuracion["modelos"][nombre]
    parametros = especificacion["parametros"].copy()
    clases = {"Ridge": Ridge, "ElasticNet": ElasticNet, "SVR_RBF": SVR,
              "RandomForest": RandomForestRegressor}
    if nombre == "GaussianProcess":
        if parametros.pop("kernel") != "Matern":
            raise ValueError("Se requiere el núcleo Matérn declarado.")
        kernel = Matern(length_scale=parametros.pop("length_scale"),
                        length_scale_bounds="fixed", nu=parametros.pop("nu"))
        regresor = GaussianProcessRegressor(kernel=kernel, **parametros)
        if regresor.optimizer is not None:
            raise ValueError("La comparación requiere parámetros fijos, sin optimización del núcleo.")
    elif nombre in clases:
        regresor = clases[nombre](**parametros)
    else:
        raise ValueError(f"Regresor no autorizado: {nombre}")
    return make_pipeline("A").set_params(model=regresor)


def validar_particiones(datos, particiones, grupos, registros):
    """Exige exactamente los grupos y casos documentados, con cobertura OOF única."""
    if set(particiones) != {"configuracion", "familia"} or set(grupos) != set(particiones):
        raise ValueError("Se requieren los dos protocolos compartidos: configuración y familia.")
    claves = [(r["protocolo"], r["fold"]) for r in registros]
    esperadas = {(p, i) for p, partes in particiones.items() for i in range(1, len(partes) + 1)}
    if len(claves) != len(set(claves)) or set(claves) != esperadas:
        raise ValueError("Los registros no coinciden con las particiones compartidas.")
    registro_por_clave = dict(zip(claves, registros))
    for protocolo, partes in particiones.items():
        vistos = []
        if len(grupos[protocolo]) != len(datos):
            raise ValueError("Los grupos no se corresponden con las filas de datos.")
        for numero, (entrena, prueba) in enumerate(partes, 1):
            entrena, prueba = np.asarray(entrena), np.asarray(prueba)
            if not len(entrena) or not len(prueba):
                raise ValueError("Cada partición requiere entrenamiento y evaluación.")
            if not np.issubdtype(entrena.dtype, np.integer) or not np.issubdtype(prueba.dtype, np.integer):
                raise ValueError("Las particiones deben contener posiciones enteras.")
            if len(set(entrena)) != len(entrena) or len(set(prueba)) != len(prueba):
                raise ValueError("Una partición repite posiciones.")
            if set(entrena) & set(prueba) or set(entrena) | set(prueba) != set(range(len(datos))):
                raise ValueError("La partición no separa completamente entrenamiento y evaluación.")
            grupo_entrena = set(grupos[protocolo].iloc[entrena])
            grupo_prueba = set(grupos[protocolo].iloc[prueba])
            if grupo_entrena & grupo_prueba or len(grupo_prueba) != 1:
                raise ValueError("Hay fuga por grupo o no se excluye un grupo completo.")
            registro = registro_por_clave[(protocolo, numero)]
            if (datos.iloc[entrena].Case_n.astype(int).tolist() != registro["train_cases"] or
                    datos.iloc[prueba].Case_n.astype(int).tolist() != registro["test_cases"] or
                    str(grupos[protocolo].iloc[prueba[0]]) != registro["grupo_excluido"] or
                    len(entrena) != registro["n_train"] or len(prueba) != registro["n_test"]):
                raise ValueError("Los casos de las particiones difieren de los registros compartidos.")
            vistos.extend(prueba.tolist())
        if sorted(vistos) != list(range(len(datos))):
            raise ValueError("Cada caso debe evaluarse exactamente una vez por protocolo.")


def evaluar_modelos(datos, configuracion, particiones, grupos, registros):
    """Devuelve predicciones, métricas por partición y auditoría sin escribir archivos."""
    if (configuracion.get("representacion") != "A" or configuracion.get("objetivo") != TARGET or
            configuracion.get("unidad_objetivo") != "N" or
            configuracion.get("sin_busqueda_hiperparametros") is not True or
            tuple(configuracion["modelos"]) != MODELOS):
        raise ValueError("La comparación requiere cinco modelos declarados, representación A y objetivo en N.")
    validar_particiones(datos, particiones, grupos, registros)
    x, y = datos[GEOMETRY], datos[TARGET]
    if y.isna().any() or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("El objetivo observado debe ser positivo y finito; no se imputa.")
    predicciones, metricas_particion, auditorias = [], [], []
    for protocolo, partes in particiones.items():
        for nombre in MODELOS:
            for numero, (entrena, prueba) in enumerate(partes, 1):
                modelo = crear_modelo(nombre, configuracion)
                with warnings.catch_warnings(record=True) as capturadas:
                    warnings.simplefilter("always")
                    inicio = perf_counter()
                    modelo.fit(x.iloc[entrena], y.iloc[entrena])
                    estimadas = modelo.predict(x.iloc[prueba])
                    tiempo = perf_counter() - inicio
                if not np.isfinite(estimadas).all():
                    raise ValueError(f"{nombre} produjo predicciones no finitas.")
                avisos = [{"categoria": aviso.category.__name__, "mensaje": str(aviso.message)}
                          for aviso in capturadas]
                nombres = feature_names(modelo).tolist()
                metricas_particion.append({"protocolo": protocolo, "modelo": nombre, "fold": numero,
                    "grupo": str(grupos[protocolo].iloc[prueba[0]]), "n_train": len(entrena), "n_test": len(prueba),
                    "MAE_N": mean_absolute_error(y.iloc[prueba], estimadas),
                    "RMSE_N": float(np.sqrt(mean_squared_error(y.iloc[prueba], estimadas))),
                    "tiempo_s": tiempo, "n_features": len(nombres), "avisos": len(avisos)})
                auditorias.append({"protocolo": protocolo, "modelo": nombre, "experimento": "A", "fold": numero,
                    "train_cases": datos.iloc[entrena].Case_n.astype(int).tolist(),
                    "test_cases": datos.iloc[prueba].Case_n.astype(int).tolist(),
                    "imputer_columns": GEOMETRY, "imputer_median": modelo.named_steps["imputer"].statistics_.tolist(),
                    "feature_names": nombres, "scaler_mean": modelo.named_steps["scale"].mean_.tolist(),
                    "scaler_scale": modelo.named_steps["scale"].scale_.tolist(),
                    "scaler_n_samples_seen": int(modelo.named_steps["scale"].n_samples_seen_), "avisos": avisos})
                if nombre == "GaussianProcess":
                    auditorias[-1]["target_train_mean"] = float(modelo.named_steps["model"]._y_train_mean)
                    auditorias[-1]["target_train_std"] = float(modelo.named_steps["model"]._y_train_std)
                for posicion, estimada in zip(prueba, estimadas):
                    residuo = float(y.iloc[posicion] - estimada)
                    fuera = (x.iloc[posicion] < x.iloc[entrena].min()) | (x.iloc[posicion] > x.iloc[entrena].max())
                    predicciones.append({"protocolo": protocolo, "modelo": nombre, "representacion": "A", "fold": numero,
                        "Case_n": int(datos.iloc[posicion].Case_n), "familia": int(datos.iloc[posicion].n_cells_Y),
                        "configuracion": str(grupos["configuracion"].iloc[posicion]), "observado_N": float(y.iloc[posicion]),
                        "predicho_N": float(estimada), "residuo_N": residuo, "error_absoluto_N": abs(residuo),
                        "fuera_rango_train": int(fuera.sum())})
    return pd.DataFrame(predicciones), pd.DataFrame(metricas_particion), auditorias


def resumir_modelos(predicciones, particiones, configuracion):
    filas = []
    for (protocolo, nombre), valores in predicciones.groupby(["protocolo", "modelo"], sort=False):
        partes = particiones[(particiones.protocolo == protocolo) & (particiones.modelo == nombre)]
        filas.append({"protocolo": protocolo, "modelo": nombre,
            "nombre_modelo": configuracion["modelos"][nombre]["nombre"], "representacion": "A",
            **metrics(valores.observado_N, valores.predicho_N), "tiempo_s": partes.tiempo_s.sum(),
            "n_features_min": int(partes.n_features.min()), "n_features_max": int(partes.n_features.max()),
            "n_folds": len(partes), "predicciones_negativas": int((valores.predicho_N < 0).sum())})
    resumen = pd.DataFrame(filas)
    for protocolo in resumen.protocolo.unique():
        seleccion = resumen.protocolo == protocolo
        referencia = resumen.loc[seleccion & (resumen.modelo == "Ridge"), "MAE_N"].iloc[0]
        resumen.loc[seleccion, "delta_MAE_vs_Ridge_N"] = resumen.loc[seleccion, "MAE_N"] - referencia
    familias = []
    for (protocolo, nombre, familia), valores in predicciones.groupby(["protocolo", "modelo", "familia"], sort=False):
        familias.append({"protocolo": protocolo, "modelo": nombre, "familia": int(familia),
            "n_casos": len(valores), **metrics(valores.observado_N, valores.predicho_N),
            "sesgo_N": valores.residuo_N.mean(), "error_maximo_N": valores.error_absoluto_N.max()})
    return resumen, pd.DataFrame(familias)


def graficar_modelos(salida, resumen, predicciones, familias, configuracion):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "pantographic-feature-engineering", "savefig.facecolor": "white"})
    nombres = {m: configuracion["modelos"][m]["nombre"] for m in MODELOS}
    etiquetas = [nombres[m].replace("SVR con núcleo RBF", "SVR · RBF") for m in MODELOS]
    figuras = []

    def guardar(figura, nombre):
        figura.tight_layout()
        for extension in ("png", "svg"):
            save_figure_atomic(figura, salida / "figures" / f"{nombre}.{extension}", dpi=300)
        plt.close(figura)
        figuras.append(nombre)

    principal = resumen[resumen.protocolo == "configuracion"].set_index("modelo").loc[list(MODELOS)]
    fig, ejes = plt.subplots(1, 2, figsize=(12.6, 5.3), gridspec_kw={"width_ratios": [1.4, 1]})
    posiciones = np.arange(len(MODELOS))
    for desplazamiento, columna, color, etiqueta in [(-.18, "MAE_N", "#2563a6", "MAE"), (.18, "RMSE_N", "#7aadc1", "RMSE")]:
        ejes[0].barh(posiciones + desplazamiento, principal[columna], height=.32, color=color, label=etiqueta)
        for pos, valor in zip(posiciones + desplazamiento, principal[columna]):
            ejes[0].text(valor + .3, pos, f"{valor:.2f}", va="center", fontsize=9)
    ejes[0].set(yticks=posiciones, yticklabels=etiquetas, xlabel="Error fuera de muestra [N]",
                xlim=(0, principal.RMSE_N.max() * 1.26), title="Error al excluir configuraciones")
    ejes[0].invert_yaxis()
    ejes[0].legend(loc="upper right", frameon=False)
    ejes[1].barh(posiciones, principal.tiempo_s, color=[COLORES[m] for m in MODELOS], height=.5)
    for pos, valor in zip(posiciones, principal.tiempo_s):
        ejes[1].text(valor + principal.tiempo_s.max() * .02, pos, f"{valor:.3f}", va="center", fontsize=9)
    ejes[1].set(yticks=posiciones, yticklabels=[], xlabel="Suma de ajuste + predicción [s]",
                xlim=(0, principal.tiempo_s.max() * 1.3), title="Tiempo del protocolo completo")
    ejes[1].invert_yaxis()
    fig.suptitle("Cinco regresores · misma geometría original y mismas particiones", fontsize=13)
    fig.supxlabel("Tiempo local de ejecución; no equivale a latencia de despliegue ni a costo monetario.", fontsize=9)
    guardar(fig, "FE12_modelos_metricas")

    principal_pred = predicciones[predicciones.protocolo == "configuracion"]
    fig, ejes = plt.subplots(2, 3, figsize=(13.2, 8), sharex=True, sharey=True)
    colores_familia = {4: "#2563a6", 5: "#c78627", 6: "#008579"}
    for eje, nombre in zip(ejes.flat, MODELOS):
        valores = principal_pred[principal_pred.modelo == nombre]
        for familia, grupo in valores.groupby("familia"):
            eje.scatter(grupo.predicho_N, grupo.residuo_N, color=colores_familia[int(familia)],
                        label=f"Familia {familia}", s=42, edgecolors="white", linewidth=.6)
        eje.axhline(0, color="#64748b", ls="--", linewidth=1)
        extremo = valores.loc[valores.error_absoluto_N.idxmax()]
        eje.annotate(f"Caso {int(extremo.Case_n)}", (extremo.predicho_N, extremo.residuo_N),
                     xytext=(4, 4), textcoords="offset points", fontsize=8)
        eje.set(title=nombres[nombre], xlabel="Predicción fuera de muestra [N]", ylabel="Observado − predicho [N]")
    ejes.flat[-1].axis("off")
    handles, labels = ejes.flat[0].get_legend_handles_labels()
    ejes.flat[-1].legend(handles, labels, loc="center", frameon=False, title="Arquitectura: celdas en Y")
    ejes.flat[-1].text(.5, .25, "Cada punto: un caso reservado.\nResiduo positivo: subestimación.\nResiduo negativo: sobreestimación.",
                      transform=ejes.flat[-1].transAxes, ha="center", va="center", fontsize=10)
    fig.suptitle("Residuales por modelo · configuraciones excluidas", fontsize=13)
    guardar(fig, "FE13_modelos_residuos")

    fig, ejes = plt.subplots(1, 2, figsize=(12.2, 5.2), layout="constrained")
    maximo = familias.MAE_N.max()
    columnas = sorted(familias.familia.unique())
    for eje, protocolo, titulo in zip(ejes, ["configuracion", "familia"],
                                     ["Configuración completa excluida", "Familia completa excluida"]):
        tabla = familias[familias.protocolo == protocolo].pivot(index="modelo", columns="familia", values="MAE_N").loc[list(MODELOS), columnas]
        mapa = eje.imshow(tabla, vmin=0, vmax=maximo, cmap="YlGnBu", aspect="auto")
        eje.set(xticks=range(len(columnas)), xticklabels=[f"Familia {int(f)}" for f in columnas],
                yticks=posiciones, yticklabels=etiquetas, title=titulo)
        eje.tick_params(length=0, pad=8)
        for fila in range(len(MODELOS)):
            for columna in range(len(columnas)):
                valor = tabla.iloc[fila, columna]
                eje.text(columna, fila, f"{valor:.2f}", ha="center", va="center",
                          color="white" if valor > .6 * maximo else "#172b4d")
    fig.colorbar(mapa, ax=ejes, shrink=.8, label="MAE fuera de muestra [N]")
    fig.suptitle("Generalización por familia · menor error es mejor", fontsize=13)
    # Mantener constrained_layout: tight_layout no maneja bien la barra compartida.
    for extension in ("png", "svg"):
        save_figure_atomic(fig, salida / "figures" / f"FE14_modelos_generalizacion.{extension}", dpi=300)
    plt.close(fig)
    figuras.append("FE14_modelos_generalizacion")
    return figuras


def ejecutar_comparacion(root, frame, splits, groups, records):
    """Ejecuta, registra y grafica; devuelve nombres base para el manifiesto común."""
    root = Path(root)
    salida = root / "reports/feature_engineering"
    for carpeta in [salida / "tables", salida / "figures"]:
        carpeta.mkdir(parents=True, exist_ok=True)
    configuracion = json.loads((root / "config/comparacion_modelos.json").read_text(encoding="utf-8"))
    inicio = datetime.now(timezone.utc).isoformat()
    with threadpool_limits(limits=1):
        predicciones, particiones, auditorias = evaluar_modelos(frame, configuracion, splits, groups, records)
    resumen, familias = resumir_modelos(predicciones, particiones, configuracion)
    save_json(salida / "configuracion_modelos.json", configuracion)
    save_json(salida / "auditoria_modelos.json", auditorias)
    for nombre, tabla in {"comparacion_modelos": resumen, "predicciones_modelos_oof": predicciones,
                          "metricas_modelos_folds": particiones, "errores_modelos_familia": familias}.items():
        save_csv(salida / "tables" / f"{nombre}.csv", tabla)
    lineas = [f"Inicio UTC: {inicio}", "COMPARACIÓN DE CINCO REGRESORES",
              "Representación A fija: geometría original. Se cambia únicamente el regresor.",
              "Particiones idénticas a A/B/C/D; exclusión por configuración y por familia.",
              "Sin búsqueda ni selección final de hiperparámetros. Parámetros en configuracion_modelos.json.",
              "Imputación, filtro y escalamiento ajustados solo en entrenamiento; auditoría por modelo y partición.",
              "No se usa one-hot, no se imputan objetivos y no se recortan predicciones negativas.",
              "SVR ajusta directamente el objetivo en N; epsilon es una tolerancia de error de 5 N.",
              "El proceso gaussiano normaliza el objetivo con media y desviación de entrenamiento; predict devuelve N.",
              "El núcleo Matérn del proceso gaussiano permanece fijo (optimizer=None); no se reportan intervalos calibrados.",
              "MAE principal; RMSE, MedAE y R² calculados sobre predicciones fuera de muestra agregadas.",
              "Tiempo: suma de ajuste del pipeline y predicción; excluye informes y gráficos.",
              "No hay conjunto final independiente. El EDA y la comparación de candidatos hacen exploratoria esta evaluación.",
              "Resultados por protocolo:"]
    for fila in resumen.itertuples(index=False):
        lineas.append(f"{fila.protocolo} | {fila.nombre_modelo} | MAE={fila.MAE_N:.6f} N | RMSE={fila.RMSE_N:.6f} N | "
                      f"MedAE={fila.MedAE_N:.6f} N | R²={fila.R2_OOF:.6f} | tiempo={fila.tiempo_s:.6f} s")
    lineas.append(f"Avisos capturados: {sum(len(a['avisos']) for a in auditorias)}")
    for auditoria in auditorias:
        for aviso in auditoria["avisos"]:
            lineas.append(f"AVISO | {auditoria['protocolo']} | {auditoria['modelo']} | partición {auditoria['fold']} | "
                          f"{aviso['categoria']}: {aviso['mensaje']}")
    write_bytes_atomic(salida / "ejecucion_modelos.log", ("\n".join(lineas) + "\n").encode("utf-8"))
    return graficar_modelos(salida, resumen, predicciones, familias, configuracion)
