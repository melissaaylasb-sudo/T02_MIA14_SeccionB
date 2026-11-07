"""Tablas, lecturas numéricas y listas de resultados del estudio pantográfico."""

import json

import numpy as np
import pandas as pd
from IPython.display import Markdown, display


# Nombres de modelos, protocolos y leyendas compartidas.
MODELOS = ("SVR_RBF", "XGBoost", "CatBoost", "Lineal_multiple", "KNN", "Arbol_decision")
PROTOCOLOS = ("configuracion", "familia")
PROTO = {"configuracion": "Configuración reservada", "familia": "Nivel Y ausente del ajuste"}
NOMBRES = {
    "SVR_RBF": "SVR con núcleo RBF",
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost",
    "Lineal_multiple": "Lineal múltiple",
    "Lineal_optimizada": "Lineal múltiple ajustada",
    "KNN": "K vecinos",
    "Arbol_decision": "Árbol de decisión",
    "Dummy_mediana": "Mediana de entrenamiento",
    "Media_por_nivel": "Media por nivel Y",
    "Ridge_historico": "Ridge histórico",
    "Tuning_rango_1": "Primera del cribado",
    "Tuning_rango_2": "Segunda del cribado",
    "Selector_tuning": "Selector interno",
}


def fmt(x, posiciones=2):
    return f"{float(x):.{posiciones}f}".replace(".", ",")


def nombre(codigo):
    return NOMBRES.get(codigo, str(codigo))


def formato_celda(valor):
    if isinstance(valor, (float, np.floating)):
        return fmt(valor, 3) if np.isfinite(valor) else "—"
    if valor is None:
        return "—"
    return str(valor).replace("|", r"\|").replace("\n", " ")


def tabla_markdown(datos):
    """Crea una tabla Markdown, con las métricas alineadas a la derecha."""
    def fila(valores):
        return "| " + " | ".join(formato_celda(v) for v in valores) + " |"

    alineacion = ["---:" if pd.api.types.is_numeric_dtype(tipo) else ":---"
                  for tipo in datos.dtypes]
    filas = [fila(datos.columns), fila(alineacion)]
    filas.extend(fila(valores) for valores in datos.itertuples(index=False, name=None))
    return "\n".join(filas)


def tabla(datos):
    display(Markdown(tabla_markdown(datos)))


def nota(texto):
    display(Markdown(texto))


def detalle(titulo, datos):
    display(Markdown(f"**{titulo}**\n\n" + tabla_markdown(datos)))


def mostrar_geometria(ficha_variables, decisiones_fe):
    # La representación A conserva las medidas geométricas y las unidades de la fuente.
    tabla(ficha_variables[["variable", "Unidad", "Justificación física"]].rename(columns={
        "variable": "Variable", "Justificación física": "Medida geométrica"
    }))
    detalle("Variantes geométricas B–D y motivo para conservar A", decisiones_fe[[
        "experimento", "referencia", "intervencion", "delta_MAE_N", "decision"
    ]])


def mostrar_antecedentes(previo, historia):
    # Antecedentes separados del experimento actual.
    antecedentes = previo.loc[
        ((previo.modelo.isin(["Ridge", "SVR_RBF"])) & (previo.conjunto == "todas"))
        | (previo.modelo == "Media_por_nivel"),
        ["protocolo", "modelo", "MAE_N", "RMSE_N", "R2"]
    ]
    detalle("Métricas del estudio anterior", antecedentes)
    inventario = historia.groupby("etapa", sort=False)["modelo"].agg(
        lambda grupo: ", ".join(pd.unique(grupo))
    ).rename("Modelos evaluados").reset_index()
    detalle("Modelos evaluados por etapa previa", inventario)


def mostrar_referencias(metricas):
    mostrar_metricas(metricas, metricas.procedimiento.unique(), "Referencia")
    nivel = metricas.loc[metricas.procedimiento.eq("Media_por_nivel")].set_index("protocolo").MAE_N
    nota(f"Con el nivel Y presente en entrenamiento, la media por nivel tiene un error de {fmt(nivel['configuracion'])} N. "
         f"Al reservar el nivel completo, la media de entrenamiento da {fmt(nivel['familia'])} N.")


def ficha_corta(m, parametros):
    p = parametros.get(m, {})
    if m == "SVR_RBF": return f"RBF, C={p['C']:g}, ε={p['epsilon']:g} N, γ={p['gamma']}"
    if m == "XGBoost": return f"{p['n_estimators']} árboles, profundidad {p['max_depth']}, η={p['learning_rate']}"
    if m == "CatBoost": return f"{p['iterations']} iteraciones, profundidad {p['depth']}, η={p['learning_rate']}"
    if m == "KNN": return f"k={p['n_neighbors']}, p={p['p']}, pesos {p['weights']}"
    if m == "Arbol_decision": return f"Profundidad {p['max_depth']}, hoja mínima {p['min_samples_leaf']}"
    if m == "Lineal_multiple": return "Mínimos cuadrados con intercepto y base de rango completo"
    if m == "Ridge_historico": return f"α={p['alpha']:g}, solucionador {p['solver']}"
    if m == "Media_por_nivel": return "Media por Y, con media de entrenamiento para nivel ausente"
    if m == "Dummy_mediana": return "Mediana del entrenamiento"
    return "Familia y parámetros elegidos dentro del entrenamiento"


def mostrar_recetas(parametros):
    tabla(pd.DataFrame([{"Modelo": nombre(m), "Receta inicial": ficha_corta(m, parametros)} for m in MODELOS]))
    detalle("Configuración completa y semillas", pd.DataFrame([
        {"Modelo": nombre(m), "Parámetros": json.dumps(parametros[m], ensure_ascii=False)}
        for m in MODELOS
    ]))


def mostrar_dispersion_inicial(metricas):
    for p in PROTOCOLOS:
        t = metricas.loc[metricas.protocolo.eq(p) & metricas.procedimiento.isin(MODELOS)].copy()
        t["Modelo"] = t.procedimiento.map(nombre)
        detalle(f"{PROTO[p]} · sesgo, dispersión y tiempos de los seis modelos",
                t[["Modelo", "sesgo_N", "DE_MAE_pliegue_N", "RIC_MAE_pliegue_N", "tiempo_ajuste_s", "tiempo_prediccion_s"]]
                .rename(columns={"sesgo_N": "Sesgo [N]", "DE_MAE_pliegue_N": "DE MAE [N]", "RIC_MAE_pliegue_N": "IQR MAE [N]",
                                 "tiempo_ajuste_s": "Ajuste [s]", "tiempo_prediccion_s": "Predicción [s]"}))


def mostrar_errores_maximos(predicciones_iniciales):
    maximos = predicciones_iniciales.loc[predicciones_iniciales.groupby(["protocolo","procedimiento"]).error_absoluto_N.idxmax(),
                                ["protocolo","procedimiento","observado_N","predicho_N","error_absoluto_N"]]
    detalle("Mayor error por modelo y protocolo", maximos)


def mostrar_familias(seleccion, titulo):
    familias = (seleccion.groupby("protocolo").familia_real
                    .agg(lambda serie: ", ".join(nombre(m) for m in sorted(serie.unique()))))
    tabla(familias.rename(index=PROTO).rename_axis("Protocolo").reset_index(name=titulo))


def mostrar_espacios(configuracion, planes_busqueda):
    tabla(pd.DataFrame([
        {"Modelo": nombre(familia), "Espacio discreto": json.dumps(espacio, ensure_ascii=False),
         "Configuraciones por búsqueda": len(planes_busqueda[familia])}
        for familia, espacio in configuracion["espacios"].items()
    ]))
    nota(f"Se evalúan {configuracion['presupuesto_por_candidata']} configuraciones por familia candidata con semilla base {configuracion['semilla']}.")


def mostrar_reajuste(receta, configuracion):
    final = receta["selector"]
    ajustables = {k: final["parametros"][k] for k in configuracion["espacios"][final["familia_real"]]}
    nota(f"Reajuste con la campaña completa, {nombre(final['familia_real'])}, {ajustables}.")


def mostrar_metricas(metricas, procedimientos, titulo):
    seleccion = metricas.loc[metricas.procedimiento.isin(procedimientos)]
    datos = seleccion.loc[seleccion.protocolo.eq("configuracion")].set_index("procedimiento")
    tabla_resultados = datos[["MAE_N", "RMSE_N", "R2_validacion"]].copy()
    tabla_resultados.columns = ["MAE [N]", "RMSE [N]", "R²"]
    nivel = seleccion.loc[seleccion.protocolo.eq("familia")].set_index("procedimiento")
    tabla_resultados["MAE nivel Y [N]"] = nivel.MAE_N
    tabla(tabla_resultados.sort_values("MAE [N]").rename(index=NOMBRES).rename_axis(titulo).reset_index())


def mostrar_ajuste_emparejado(mae_pareado, adaptativos):
    datos = pd.concat([mae_pareado.xs(p, level="protocolo") for p in PROTOCOLOS], axis=1)
    datos.columns = ["Config. inicial [N]", "Config. ajustada [N]", "Nivel Y inicial [N]", "Nivel Y ajustada [N]"]
    tabla(datos.loc[list(adaptativos)].rename(index=NOMBRES).rename_axis("Procedimiento").reset_index())


def mostrar_integrada(metricas, procedimientos):
    # El detalle de los dos puestos conserva la familia que realmente se seleccionó en cada reserva.
    for p in PROTOCOLOS:
        t = metricas.loc[metricas.protocolo.eq(p)].set_index("procedimiento").loc[list(procedimientos)].reset_index()
        t["Modelo"] = t.procedimiento.map(nombre)
        t["ΔMAE vs Ridge [N]"] = t.delta_MAE_vs_Ridge_N
        detalle(f"{PROTO[p]} · tabla integrada, controles y dos puestos del cribado",
                t[["Modelo", "MAE_N", "RMSE_N", "MedAE_N", "R2_validacion", "ΔMAE vs Ridge [N]",
                   "DE_MAE_pliegue_N", "tiempo_ajuste_s", "tiempo_prediccion_s", "tiempo_busqueda_s"]])


def mostrar_pendientes(coef):
    co = coef[coef.protocolo.eq("configuracion") & coef.estado.eq("retenida")]
    coef_resumen = co.groupby(["variable","unidad_variable"]).coeficiente_N_por_unidad.agg(["min","median","max"]).reset_index()
    detalle("Pendientes OLS en unidades originales por pliegue", coef_resumen.rename(columns={"min":"Pendiente mínima","median":"Pendiente mediana","max":"Pendiente máxima"}))
    nota("Las pendientes se expresan en la base independiente de cada entrenamiento. "
          "La colinealidad limita su interpretación como efectos físicos individuales.")


def mostrar_diagnostico(recomendacion, diag):
    cat = recomendacion["categorias"]
    detalle("Comparación por familia y estabilidad", pd.DataFrame([{"Categoría del protocolo principal":k.replace("_"," "), "Procedimiento":nombre(v)} for k,v in cat.items()]))
    diagnostico_vecinos = diag[(diag.procedimiento=="KNN") & diag.protocolo.isin(PROTOCOLOS)].groupby("protocolo").distancia_media_vecinos.mean()
    diagnostico_arbol = diag[(diag.procedimiento=="Arbol_decision") & (diag.protocolo=="configuracion")]
    nota(f"Distancia media a los vecinos en geometría estandarizada: {fmt(diagnostico_vecinos['configuracion'])} por "
          f"configuración, {fmt(diagnostico_vecinos['familia'])} por nivel. El árbol inicial tuvo "
          f"entre {int(diagnostico_arbol.hojas.min())} y {int(diagnostico_arbol.hojas.max())} hojas en configuraciones.")


def mostrar_sensibilidad(predicciones_externas, proc):
    por_caso = predicciones_externas[(predicciones_externas.protocolo=="configuracion") & predicciones_externas.procedimiento.eq(proc)].sort_values("error_absoluto_N",ascending=False)
    alto = por_caso.iloc[0]
    mae_sin_alto = por_caso.iloc[1:].error_absoluto_N.mean()
    nota(f"Sensibilidad del MAE de {nombre(proc)}: al omitir del promedio la observación con mayor error "
          f"pasa de {fmt(por_caso.error_absoluto_N.mean())} a {fmt(mae_sin_alto)} N. Se conservaron todas las probetas.")


def mostrar_contraste_referencia(errores, diferencia_referencia):
    comparativa = pd.DataFrame({
        "MAE config. [N]": errores.configuracion,
        "Δ vs. baseline [N]": diferencia_referencia.configuracion,
        "MAE nivel Y [N]": errores.familia,
        "Δ vs. baseline en nivel Y [N]": diferencia_referencia.familia,
    })
    orden_inicial = errores.loc[list(MODELOS)].sort_values("configuracion").index
    puestos = {modelo: str(i) for i, modelo in enumerate(orden_inicial, start=1)}
    comparativa.insert(0, "Puesto inicial", [puestos.get(m, "—") for m in comparativa.index])
    tabla(comparativa.rename(index=NOMBRES).rename_axis("Modelo o referencia").reset_index())


def presentar_arquitectura(familia, estimador, unidades, ecuacion):
    """Muestra en Jupyter los pasos ajustados y la ecuación física, si es lineal."""
    etiquetas = {"geometria": "Validar geometría A", "imputacion": "Imputar con la mediana",
                 "varianza": "Quitar constantes", "escala": "Estandarizar",
                 "base": "Conservar una base independiente"}
    flujo = " → ".join(etiquetas.get(clave, type(paso).__name__) for clave, paso in estimador.steps)
    nota(f"**{nombre(familia)}**\n\n{flujo} → **Carga de primera falla [N]**")
    if ecuacion is None:
        return

    intercepto, pendientes = ecuacion
    pasos = estimador.named_steps
    imputador = pasos["imputacion"]
    variables = pd.Index(imputador.feature_names_in_)
    con_varianza = variables[pasos["varianza"].get_support()]
    constantes = variables.difference(con_varianza, sort=False)
    redundantes = con_varianza.difference(pendientes.index, sort=False)
    medianas = pd.Series(imputador.statistics_, index=variables)

    nota(f"**Ecuación en unidades originales**\n\n"
         f"**F̂ [N] = {fmt(intercepto, 8)} + Σⱼ βⱼ xⱼ**\n\n"
         "Cada xⱼ es la geometría en la unidad de la tabla. Si falta, se usa la mediana indicada. "
         "Los coeficientes βⱼ ya incorporan el escalado del ajuste por mínimos cuadrados.")
    tabla(pd.DataFrame({
        "Variable xⱼ": pendientes.index,
        "Unidad de xⱼ": [unidades[v] for v in pendientes.index],
        "Mediana de imputación": [fmt(medianas[v], 6) for v in pendientes.index],
        "βⱼ [N por unidad]": [fmt(peso, 8) for peso in pendientes],
    }))
    exclusiones = []
    if len(constantes):
        exclusiones.append("Constantes en el reajuste: " + ", ".join(f"`{v}`" for v in constantes) + ".")
    if len(redundantes):
        exclusiones.append("Redundantes en la base elegida: " + ", ".join(f"`{v}`" for v in redundantes) + ".")
    nota(" ".join([*exclusiones, f"La base conserva las columnas que aumentan el rango en el orden de entrada "
                   f"(tolerancia relativa {pasos['base'].rtol:g}). "
                   "Las pendientes describen esta base. La dependencia entre geometrías limita su interpretación física individual."]))


def mostrar_calculo_mae(ejemplo, mae_config, procedimientos):
    filas = ejemplo.set_index("procedimiento").loc[procedimientos]
    tabla(pd.DataFrame({
        "Procedimiento": [nombre(codigo) for codigo in filas.index],
        "Carga medida [N]": filas.observado_N.to_numpy(),
        "Predicción [N]": filas.predicho_N.to_numpy(),
        "Error de esta probeta [N]": filas.error_absoluto_N.to_numpy(),
        "MAE de configuraciones reservadas [N]": mae_config.loc[filas.index].to_numpy(),
    }))


def mostrar_ejemplo_vecinos(distancias, cargas, prediccion, observada):
    tabla(pd.DataFrame([{
        "Distancias estandarizadas": " · ".join(fmt(x, 3) for x in distancias),
        "Cargas de los vecinos [N]": " · ".join(fmt(x, 2) for x in cargas),
        "Media predicha [N]": prediccion,
        "Carga medida [N]": observada,
        "Error absoluto [N]": abs(prediccion - observada),
    }]))


def mostrar_semillas_modelos(configuracion):
    """Muestra las semillas que se pasan a los constructores de los regresores."""
    parametros_semilla = {
        "XGBoost": "random_state",
        "CatBoost": "random_seed",
        "Arbol_decision": "random_state",
    }
    filas = [
        {"Modelo": nombre(familia), "Parámetro": parametro,
         "Semilla": configuracion["parametros_iniciales"][familia][parametro]}
        for familia, parametro in parametros_semilla.items()
    ]
    display(Markdown(tabla_markdown(pd.DataFrame(filas))))


def mostrar_semillas_busqueda(configuracion):
    """Expone la semilla base y el desplazamiento empleado por cada familia."""
    filas = [
        {"Modelo": nombre(familia), "Posición": posicion,
         "Semilla de muestreo": configuracion["semilla"] + posicion}
        for posicion, familia in enumerate(configuracion["familias"])
        if familia in configuracion["espacios"]
    ]
    display(Markdown(tabla_markdown(pd.DataFrame(filas))))



def mostrar_resultados_ganador(analisis):
    """Métricas del modelo seleccionado y mejora frente al baseline por protocolo."""
    elegida = analisis.metricas.loc[analisis.metricas.procedimiento.eq(analisis.mejor)].set_index("protocolo")
    base = analisis.metricas.loc[analisis.metricas.procedimiento.eq("Media_por_nivel")].set_index("protocolo")
    salida = elegida[["MAE_N", "RMSE_N", "R2_validacion", "sesgo_N"]].copy()
    salida["Mejora MAE frente al baseline [N]"] = base.MAE_N - elegida.MAE_N
    salida["Mejora MAE [%]"] = 100 * (base.MAE_N - elegida.MAE_N) / base.MAE_N
    tabla(salida.rename(columns={"MAE_N": "MAE [N]", "RMSE_N": "RMSE [N]", "R2_validacion": "R²", "sesgo_N": "Sesgo [N]"})
          .rename(index=PROTO).rename_axis("Protocolo").reset_index())


def mostrar_configuracion_ganador(analisis):
    tabla(pd.DataFrame([
        {"Elemento": "Estimador", "Valor": nombre(analisis.mejor)},
        {"Elemento": "Estimación", "Valor": "Mínimos cuadrados ordinarios"},
        {"Elemento": "Intercepto", "Valor": str(analisis.configuracion["parametros_iniciales"][analisis.mejor]["fit_intercept"])},
        {"Elemento": "Tolerancia relativa de la base", "Valor": str(analisis.configuracion["ols"]["tolerancia_svd_relativa"])},
        {"Elemento": "Optimización posterior", "Valor": "Optuna sobre fit_intercept y positive, sección 5"},
        {"Elemento": "Aleatoriedad del ajuste lineal", "Valor": "Solución determinista"},
        {"Elemento": "Semilla configurada para estimadores aleatorios", "Valor": str(analisis.configuracion["parametros_iniciales"]["XGBoost"]["random_state"])},
        {"Elemento": "Semilla de la comparación previa", "Valor": "42"},
    ]))


def mostrar_resumen_errores(predicciones):
    filas = []
    for protocolo, grupo in predicciones.groupby("protocolo", sort=False):
        errores = grupo.error_absoluto_N
        filas.append({"Protocolo": PROTO[protocolo], "Mediana [N]": errores.median(),
                      "Percentil 90 [N]": errores.quantile(.9), "Máximo [N]": errores.max()})
    tabla(pd.DataFrame(filas))



def mostrar_referencias_completas(metricas):
    for protocolo in PROTOCOLOS:
        seleccion = metricas.loc[metricas.protocolo.eq(protocolo)].copy()
        seleccion["Referencia"] = seleccion.procedimiento.map(nombre)
        detalle(PROTO[protocolo], seleccion[
            ["Referencia", "MAE_N", "RMSE_N", "MedAE_N", "R2_validacion", "sesgo_N"]
        ].rename(columns={
            "MAE_N": "MAE [N]", "RMSE_N": "RMSE [N]", "MedAE_N": "Mediana del error [N]",
            "R2_validacion": "R²", "sesgo_N": "Sesgo [N]",
        }))


def mostrar_brecha_ganador(brecha):
    tabla(brecha.replace({"protocolo": PROTO}).rename(columns={
        "protocolo": "Protocolo", "entrenamiento_N": "MAE medio entrenamiento [N]",
        "validacion_N": "MAE medio prueba [N]", "brecha_N": "Brecha [N]",
        "desviacion_validacion_N": "DE entre reservas [N]",
        "mediana_validacion_N": "Mediana entre reservas [N]",
    }))


def mostrar_grupos_ganador(grupos):
    for protocolo in PROTOCOLOS:
        datos = grupos.loc[grupos.protocolo.eq(protocolo)].drop(columns=["protocolo", "pliegue"])
        detalle(PROTO[protocolo], datos.rename(columns={
            "grupo": "Grupo reservado", "MAE_modelo_N": "MAE lineal [N]",
            "MAE_baseline_N": "MAE baseline [N]", "delta_MAE_N": "Δ MAE [N]",
            "RMSE_modelo_N": "RMSE lineal [N]", "sesgo_N": "Sesgo lineal [N]",
        }))


def mostrar_casos_mayor_error(casos):
    seleccion = casos[[
        "protocolo", "id_probeta", "numero_celdas_y", "observado_N", "predicho_N",
        "residuo_N", "error_absoluto_N", "error_baseline_N",
    ]].replace({"protocolo": PROTO})
    tabla(seleccion.rename(columns={
        "protocolo": "Protocolo", "id_probeta": "Probeta", "numero_celdas_y": "Nivel Y",
        "observado_N": "Carga observada [N]", "predicho_N": "Predicción [N]",
        "residuo_N": "Residuo [N]", "error_absoluto_N": "Error lineal [N]",
        "error_baseline_N": "Error baseline [N]",
    }))


def mostrar_sensibilidad_ganador(datos):
    tabla(datos.replace({"protocolo": PROTO})[[
        "protocolo", "MAE_completo_N", "MAE_omitiendo_mayor_N", "delta_sin_mayor_N",
    ]].rename(columns={
        "protocolo": "Protocolo", "MAE_completo_N": "MAE completo [N]",
        "MAE_omitiendo_mayor_N": "MAE omitiendo el mayor error [N]",
        "delta_sin_mayor_N": "Δ frente al baseline al omitir la misma probeta [N]",
    }))
    tabla(datos.replace({"protocolo": PROTO})[[
        "protocolo", "delta_completo_N", "grupos_con_mejora_pct",
        "delta_omitiendo_grupo_min_N", "delta_omitiendo_grupo_max_N",
    ]].rename(columns={
        "protocolo": "Protocolo", "delta_completo_N": "Δ MAE completo [N]",
        "grupos_con_mejora_pct": "Grupos con mejora [%]",
        "delta_omitiendo_grupo_min_N": "Δ mínimo al omitir un grupo [N]",
        "delta_omitiendo_grupo_max_N": "Δ máximo al omitir un grupo [N]",
    }))


def mostrar_coeficientes_ganador(coeficientes):
    for protocolo in PROTOCOLOS:
        datos = coeficientes.loc[coeficientes.protocolo.eq(protocolo)].copy()
        datos["unidad"] = "N/" + datos.unidad
        detalle(PROTO[protocolo], datos[[
            "variable", "unidad", "minimo", "mediana", "maximo", "retencion_pct",
        ]].rename(columns={
            "variable": "Variable", "unidad": "Unidad del coeficiente", "minimo": "Mínimo",
            "mediana": "Mediana", "maximo": "Máximo", "retencion_pct": "Retenida en reservas [%]",
        }))



def mostrar_ensayos_optuna(ensayos):
    """Resumen descriptivo de cada combinación en los entrenamientos externos."""
    externos = ensayos.loc[ensayos.protocolo.isin(PROTOCOLOS)]
    for protocolo in PROTOCOLOS:
        resumen = externos.loc[externos.protocolo.eq(protocolo)].groupby(
            ["fit_intercept", "positive"], sort=False
        ).agg(
            MAE_interno_medio_N=("MAE_interno_N", "mean"),
            MAE_interno_minimo_N=("MAE_interno_N", "min"),
            MAE_interno_maximo_N=("MAE_interno_N", "max"),
            fraccion_elegida=("elegida", "mean"),
        ).reset_index()
        resumen["Elegida en reservas [%]"] = resumen.pop("fraccion_elegida") * 100
        detalle(PROTO[protocolo], resumen.rename(columns={
            "MAE_interno_medio_N": "MAE interno medio [N]", "MAE_interno_minimo_N": "Mínimo [N]",
            "MAE_interno_maximo_N": "Máximo [N]",
        }))


def mostrar_selecciones_optuna(ensayos, grupos):
    seleccionadas = ensayos.loc[ensayos.elegida & ensayos.protocolo.isin(PROTOCOLOS)]
    seleccionadas = seleccionadas.merge(grupos[["protocolo", "pliegue", "grupo"]], validate="one_to_one")
    for protocolo in PROTOCOLOS:
        tabla_seleccion = seleccionadas.loc[seleccionadas.protocolo.eq(protocolo),
            ["grupo", "fit_intercept", "positive", "MAE_interno_N", "semilla_busqueda"]]
        detalle(PROTO[protocolo], tabla_seleccion.rename(columns={
            "grupo": "Grupo reservado", "MAE_interno_N": "MAE interno elegido [N]",
            "semilla_busqueda": "Semilla de búsqueda",
        }))


def mostrar_comparacion_optuna(metricas):
    orden = ["Media_por_nivel", "Lineal_multiple", "Lineal_optimizada"]
    for protocolo in PROTOCOLOS:
        datos = metricas.loc[metricas.protocolo.eq(protocolo)].set_index("procedimiento").loc[orden].copy()
        datos["Δ MAE frente al inicial [N]"] = datos.MAE_N - datos.loc["Lineal_multiple", "MAE_N"]
        datos["Δ MAE frente al baseline [N]"] = datos.MAE_N - datos.loc["Media_por_nivel", "MAE_N"]
        detalle(PROTO[protocolo], datos[[
            "MAE_N", "RMSE_N", "R2_validacion", "sesgo_N",
            "Δ MAE frente al inicial [N]", "Δ MAE frente al baseline [N]",
        ]].rename(index=NOMBRES).rename_axis("Procedimiento").reset_index().rename(columns={
            "MAE_N": "MAE [N]", "RMSE_N": "RMSE [N]", "R2_validacion": "R²", "sesgo_N": "Sesgo [N]",
        }))


def mostrar_grupos_optuna(grupos):
    for protocolo in PROTOCOLOS:
        detalle(PROTO[protocolo], grupos.loc[grupos.protocolo.eq(protocolo),
            ["grupo", "MAE_inicial_N", "MAE_ajustado_N", "delta_MAE_N"]].rename(columns={
                "grupo": "Grupo reservado", "MAE_inicial_N": "MAE inicial [N]",
                "MAE_ajustado_N": "MAE ajustado [N]", "delta_MAE_N": "Δ MAE [N]",
            }))


def mostrar_receta_optuna(ensayo, semilla):
    tabla(pd.DataFrame([{
        **ensayo.params, "MAE interno de selección [N]": ensayo.value,
        "Semilla GridSampler": semilla,
    }]))
