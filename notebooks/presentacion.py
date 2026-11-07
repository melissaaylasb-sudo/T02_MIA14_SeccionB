"""Tablas, lecturas numéricas y listas de resultados del estudio pantográfico."""

import json
from base64 import b64encode
from io import BytesIO
from pathlib import Path

from matplotlib import get_data_path
from PIL import Image as PILImage, ImageDraw, ImageFont

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


def panel():
    """Reserva una salida que se actualiza cuando termina el cálculo de sus métricas."""
    return display(Markdown(""), display_id=True)


def _bloque_verde(parrafos, destino=None, numerado=False):
    """Publica un solo bloque PNG, con una alternativa de texto en el mismo resultado."""
    ancho, margen, interlinea = 1500, 36, 36
    directorio = Path(get_data_path()) / "fonts" / "ttf"
    cuerpo = ImageFont.truetype(str(directorio / "DejaVuSans.ttf"), 25)
    negrita = ImageFont.truetype(str(directorio / "DejaVuSans-Bold.ttf"), 25)
    fondo, tinta = "#E1F2E5", "#183B2A"
    renglones, alternativo = [], []
    y = margen
    for etiqueta, original in parrafos:
        texto = original.replace("**", "")
        alternativo.append(f"{etiqueta} {texto}" if numerado else f"{etiqueta}\n{texto}")
        renglones.append((margen, y, etiqueta, negrita))
        x = margen + 52 if numerado else margen
        if not numerado:
            y += interlinea
        linea = ""
        for palabra in texto.split():
            candidata = f"{linea} {palabra}" if linea else palabra
            if linea and cuerpo.getlength(candidata) > ancho - margen - x:
                renglones.append((x, y, linea, cuerpo))
                y += interlinea
                linea = palabra
            else:
                linea = candidata
        renglones.append((x, y, linea, cuerpo))
        y += interlinea + 16

    alto = y - 16 + margen
    bloque = PILImage.new("RGB", (ancho, alto), fondo)
    dibujo = ImageDraw.Draw(bloque)
    for x, y, texto, fuente in renglones:
        dibujo.text((x, y), texto, font=fuente, fill=tinta, anchor="lt")
    with BytesIO() as buffer:
        bloque.save(buffer, "PNG")
        contenido = b64encode(buffer.getvalue()).decode("ascii")
    texto = "\n\n".join(alternativo)
    datos = {"image/png": contenido, "text/plain": texto}
    metadatos = {"image/png": {"width": 1000, "alt": texto}}
    if destino is None:
        display(datos, raw=True, metadata=metadatos)
    else:
        destino.update(datos, raw=True, metadata=metadatos)


def interpretar(lectura, resultado, destino=None):
    """Separa la lectura de las marcas gráficas de lo que se obtuvo en la campaña."""
    _bloque_verde([("Lectura", lectura), ("Interpretación", resultado)], destino=destino)


def enumerar_resultados(textos, inicio=1):
    """Numera las conclusiones en el orden en que se presentan."""
    items = [(f"{n}.", texto) for n, texto in enumerate(textos, start=inicio)]
    _bloque_verde(items, numerado=True)


def mostrar_geometria(features, decisiones_fe):
    # La representación A conserva las medidas geométricas y las unidades de la fuente.
    tabla(features[["Feature", "Unidad", "Justificación física"]].rename(columns={
        "Feature": "Variable", "Justificación física": "Medida geométrica"
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


def mostrar_baselines(resumen_base):
    mae_por_escenario = resumen_base.MAE_N.unstack("protocolo").rename(index=nombre)
    tabla(mae_por_escenario.rename(columns={"configuracion": "Configuración · MAE [N]",
          "familia": "Nivel Y · MAE [N]"}).rename_axis("Referencia").reset_index())
    nivel = resumen_base.xs("Media_por_nivel", level="referencia").MAE_N
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
                t[["Modelo", "sesgo_N", "DE_MAE_fold_N", "IQR_MAE_fold_N", "tiempo_ajuste_s", "tiempo_prediccion_s"]]
                .rename(columns={"sesgo_N": "Sesgo [N]", "DE_MAE_fold_N": "DE MAE [N]", "IQR_MAE_fold_N": "IQR MAE [N]",
                                 "tiempo_ajuste_s": "Ajuste [s]", "tiempo_prediccion_s": "Predicción [s]"}))


def leer_mae(panel, principal, transferencia, referencia_nivel):
    ganador_mae = principal.MAE_N.idxmin()
    interpretar(
        'Cada barra promedia las diferencias absolutas entre la carga predicha y la medida, en newtons. La línea discontinua marca el MAE de la media por nivel. En el panel derecho, el nivel Y está ausente y la referencia usa la media global del entrenamiento. Una barra más corta que esa referencia indica menor error.',
        f"En configuraciones reservadas, **{nombre(ganador_mae)}** registra el menor MAE "
        f"({fmt(principal.at[ganador_mae, 'MAE_N'], 3)} N), frente a {fmt(referencia_nivel, 3)} N de la media por nivel. "
        f"Para el mismo regresor, el error sube a {fmt(transferencia.at[ganador_mae, 'MAE_N'], 3)} N al retirar un nivel completo.",
        destino=panel,
    )


def leer_observado(panel, curvas_carga):
    lineal = curvas_carga[curvas_carga.procedimiento.eq("Lineal_multiple")]
    alta = lineal[lineal.tramo == lineal.tramo.max()].set_index("protocolo")
    interpretar(
        'Cada marca representa las medias de carga observada y predicha en un cuartil de carga observada. Una curva por debajo de la diagonal indica subestimación en ese tramo. La diferencia vertical con signo (predicción menos medición) es el sesgo medio del tramo.',
        f"En el cuartil de mayor carga, la media observada es **{fmt(alta.at['configuracion', 'observado_N'])} N**. "
        f"La regresión lineal predice {fmt(alta.at['configuracion', 'predicho_N'])} N con niveles conocidos "
        f"y {fmt(alta.at['familia', 'predicho_N'])} N al retirar un nivel. "
        f"La subestimación media de la carga alta aumenta al retirar el nivel.",
        destino=panel,
    )


def mostrar_errores_maximos(iniciales_oof):
    maximos = iniciales_oof.loc[iniciales_oof.groupby(["protocolo","procedimiento"]).error_absoluto_N.idxmax(),
                                ["protocolo","procedimiento","observado_N","predicho_N","error_absoluto_N"]]
    detalle("Mayor error por modelo y protocolo", maximos)


def leer_rmse(principal, rmse_base):
    mejor_rmse = principal.RMSE_N.idxmin()
    rmse_modelo = principal.at[mejor_rmse, "RMSE_N"]
    interpretar(
        'Cada barra muestra la raíz del promedio de los errores al cuadrado. Un error grande pesa más en el RMSE que en el MAE. La línea discontinua indica el RMSE de la media por nivel.',
        f"En configuraciones reservadas, **{nombre(mejor_rmse)}** alcanza un RMSE de {fmt(rmse_modelo, 4)} N "
         f"y la media por nivel, {fmt(rmse_base, 4)} N. Al dar más peso a los errores grandes, "
         f"ambos procedimientos quedan separados por {fmt(abs(rmse_modelo-rmse_base), 5)} N.")


def leer_brecha(ajuste_y_reserva):
    nominal_ajuste = ajuste_y_reserva.loc["configuracion"]
    brechas = nominal_ajuste.MAE_validacion_N - nominal_ajuste.MAE_train_N
    mayor_brecha = brechas.idxmax()
    ejemplo = nominal_ajuste.loc[mayor_brecha]
    interpretar(
        'El círculo muestra el MAE sobre las geometrías usadas para ajustar el modelo y el rombo, el MAE sobre las reservadas. Cada pliegue aporta el mismo peso. La longitud del segmento mide cuánto cambia el error al pasar de ajuste a validación.',
        f"En configuraciones reservadas, **{nombre(mayor_brecha)}** pasa de {fmt(ejemplo.MAE_train_N)} N en entrenamiento "
         f"a {fmt(ejemplo.MAE_validacion_N)} N en validación. Presenta la mayor brecha ({fmt(brechas.at[mayor_brecha])} N) "
         "entre las familias iniciales. El ajuste reproduce las geometrías de entrenamiento con mayor precisión que las reservadas.")


def leer_distribucion(percentiles_error, ganador_mae):
    nominal_cuantiles = percentiles_error.loc["configuracion"]
    mejor_mediana = nominal_cuantiles[.50].idxmin()
    cuantiles = nominal_cuantiles.loc[mejor_mediana]
    interpretar(
        'Cada línea resume los errores absolutos individuales de un modelo. La marca central indica la mediana, el trazo grueso abarca los percentiles 25 a 75 y el fino los percentiles 10 a 90.',
        f"En configuraciones reservadas, **{nombre(mejor_mediana)}** tiene la menor mediana del error absoluto "
         f"({fmt(cuantiles[.50])} N), pero su percentil 90 alcanza {fmt(cuantiles[.90])} N. "
         f"Cuando se promedian todos los errores absolutos, el menor MAE corresponde a **{nombre(ganador_mae)}**.")


def leer_residuos(tramos_residuales):
    modelo_residual = tramos_residuales.modelo.iloc[0]
    extremos = (tramos_residuales.loc[tramos_residuales.modelo.eq(modelo_residual)]
                .groupby("protocolo").last())
    interpretar(
        'El residuo se calcula como carga observada menos carga predicha. Un valor positivo indica subestimación y uno negativo, sobreestimación. El signo es el inverso del sesgo global de la tabla. Las marcas son las medianas y las barras abarcan los cuartiles centrales de los residuos en cada tramo de carga predicha.',
        f"En el tramo de predicciones más altas de {nombre(modelo_residual)}, el residuo mediano "
         f"fue {fmt(extremos.at['configuracion', 'y'])} N al reservar configuraciones y "
         f"{fmt(extremos.at['familia', 'y'])} N al retirar un nivel Y. "
         "En la zona alta, el modelo pasa de sobreestimar con niveles conocidos a subestimar cuando se retira un nivel.")


def leer_busqueda(progreso_interno):
    ganancias_finales = progreso_interno.groupby(["protocolo", "familia_real"]).mediana.last()
    mayores_ganancias = []
    for p in PROTOCOLOS:
        final = ganancias_finales.loc[p]
        mayores_ganancias.append(f"{nombre(final.idxmax())} ({fmt(final.max())} N)")
    interpretar(
        'Cada punto muestra cuánto bajó el mejor MAE interno respecto de la primera configuración de hiperparámetros evaluada. La línea resume la mediana entre reservas y la banda, los cuartiles centrales.',
        f"La mayor reducción mediana del MAE interno al cierre fue la de {mayores_ganancias[0]} "
         f"al reservar configuraciones y la de {mayores_ganancias[1]} al retirar niveles Y. "
         "Estas mejoras corresponden al entrenamiento. La comparación del MAE antes y después del ajuste evalúa su efecto sobre las reservas externas.")


def leer_ridge(principal, transferencia):
    ventaja_ridge = principal.delta_MAE_vs_Ridge_N
    modelo_ridge = ventaja_ridge.idxmin()
    interpretar(
        'Para cada probeta se resta el error absoluto de Ridge al del modelo. El rombo marca la media de las diferencias y el segmento, los cuartiles centrales. La regresión lineal mejora a Ridge en ambos protocolos de esta campaña.',
        f"{nombre(modelo_ridge)} presentó la mayor reducción de MAE frente a Ridge al reservar "
         f"configuraciones ({fmt(-ventaja_ridge[modelo_ridge])} N). En el protocolo de nivel Y, "
         f"su diferencia modelo menos Ridge fue {fmt(transferencia.loc[modelo_ridge, 'delta_MAE_vs_Ridge_N'])} N. "
         "La regresión lineal mejora a Ridge en ambos protocolos de esta campaña.")


def leer_efecto_ajuste(mae_ajuste):
    cambios_selector = mae_ajuste.xs("Selector_tuning", level="procedimiento")
    def cambio_mae(fila):
        diferencia = fila.ajustado - fila.inicial
        verbo = "aumentó" if diferencia > 0 else "disminuyó" if diferencia < 0 else "cambió en"
        return f"{verbo} {fmt(abs(diferencia), 3)} N ({fmt(fila.inicial, 3)} → {fmt(fila.ajustado, 3)} N)"
    interpretar(
        'Cada barra muestra el MAE ajustado menos el MAE inicial de la misma familia, en las mismas probetas reservadas. Un valor negativo indica que la búsqueda redujo el error externo.',
        f"Al reservar configuraciones, el MAE externo del selector {cambio_mae(cambios_selector.loc['configuracion'])}. "
         f"Al retirar un nivel Y, {cambio_mae(cambios_selector.loc['familia'])}. "
         "El efecto del ajuste depende del grupo que se reserva.")


def leer_estabilidad(principal, percentiles_pliegue):
    modelo_p50 = principal.MAE_N.idxmin()
    p50_cfg = percentiles_pliegue.at[("configuracion", modelo_p50), .50]
    p90_cfg = percentiles_pliegue.at[("configuracion", modelo_p50), .90]
    p90_y = percentiles_pliegue.at[("familia", modelo_p50), .90]
    interpretar(
        'Cada celda es un percentil del MAE calculado por grupo reservado. P50 es la mediana y P90 describe el extremo de mayor error. Los dos paneles comparten la escala en newtons.',
        f"Para {nombre(modelo_p50)}, la mediana del MAE por configuración es {fmt(p50_cfg)} N "
         f"y el percentil 90 alcanza {fmt(p90_cfg)} N. Al reservar niveles Y, el percentil 90 es "
         f"{fmt(p90_y)} N. El extremo de mayor error es más alto cuando el nivel Y completo queda fuera del entrenamiento.")


def mostrar_familias(seleccion, titulo):
    familias = (seleccion.groupby("protocolo").familia_real
                    .agg(lambda serie: ", ".join(nombre(m) for m in sorted(serie.unique()))))
    tabla(familias.rename(index=PROTO).rename_axis("Protocolo").reset_index(name=titulo))


def mostrar_espacios(cfg, draws):
    tabla(pd.DataFrame([
        {"Modelo": nombre(familia), "Espacio discreto": json.dumps(espacio, ensure_ascii=False),
         "Configuraciones por búsqueda": len(draws[familia])}
        for familia, espacio in cfg["espacios"].items()
    ]))
    nota(f"Se evalúan {cfg['presupuesto_por_candidata']} configuraciones por familia candidata con semilla base {cfg['semilla']}.")


def mostrar_reajuste(receta, cfg):
    final = receta["selector"]
    ajustables = {k: final["parametros"][k] for k in cfg["espacios"][final["familia_real"]]}
    nota(f"Reajuste con la campaña completa, {nombre(final['familia_real'])}, {ajustables}.")


def tabla_inicial(metricas_nominales, mae_externo, procedimientos, titulo):
    datos = metricas_nominales.loc[list(procedimientos), ["MAE_N", "RMSE_N", "R2_OOF"]].copy()
    datos.columns = ["MAE [N]", "RMSE [N]", "R²"]
    datos["MAE nivel Y [N]"] = mae_externo["familia"]
    return datos.sort_values("MAE [N]").rename(index=NOMBRES).rename_axis(titulo).reset_index()


def actualizar_portada(panel_baselines, portada_resultados, panel_comparaciones, metricas_nominales, mae_externo, mae_pareado, referencias, adaptativos):
    modelos_portada = tabla_inicial(metricas_nominales, mae_externo, MODELOS, "Modelo")
    bases_portada = tabla_inicial(metricas_nominales, mae_externo, referencias, "Referencia")
    tabla_tuning = pd.concat([mae_pareado.xs(p, level="protocolo") for p in PROTOCOLOS], axis=1)
    tabla_tuning.columns = ["Config. inicial [N]", "Config. ajustada [N]", "Nivel Y inicial [N]", "Nivel Y ajustada [N]"]
    tabla_tuning = tabla_tuning.loc[list(adaptativos)].rename(index=NOMBRES).rename_axis("Procedimiento").reset_index()

    panel_baselines.update(Markdown(tabla_markdown(bases_portada)))
    portada_resultados.update(Markdown(tabla_markdown(modelos_portada)))
    panel_comparaciones.update(Markdown(tabla_markdown(tabla_tuning)))


def mostrar_integrada(metricas, procedimientos):
    # El detalle de los dos puestos conserva la familia que realmente se seleccionó en cada reserva.
    for p in PROTOCOLOS:
        t = metricas.loc[metricas.protocolo.eq(p)].set_index("procedimiento").loc[list(procedimientos)].reset_index()
        t["Modelo"] = t.procedimiento.map(nombre)
        t["ΔMAE vs Ridge [N]"] = t.delta_MAE_vs_Ridge_N
        detalle(f"{PROTO[p]} · tabla integrada, controles y dos puestos del cribado",
                t[["Modelo", "MAE_N", "RMSE_N", "MedAE_N", "R2_OOF", "ΔMAE vs Ridge [N]",
                   "DE_MAE_fold_N", "tiempo_ajuste_s", "tiempo_prediccion_s", "tiempo_busqueda_s"]])


def mostrar_pendientes(coef):
    co = coef[coef.protocolo.eq("configuracion") & coef.estado.eq("retenida")]
    coef_resumen = co.groupby(["variable","unidad_variable"]).coeficiente_N_por_unidad.agg(["min","median","max"]).reset_index()
    detalle("Pendientes OLS en unidades originales por pliegue", coef_resumen.rename(columns={"min":"Pendiente mínima","median":"Pendiente mediana","max":"Pendiente máxima"}))
    nota("Las pendientes se expresan en la base independiente de cada entrenamiento. "
          "La colinealidad limita su interpretación como efectos físicos individuales.")


def mostrar_diagnostico(recomendacion, diag):
    cat = recomendacion["categorias"]
    detalle("Comparación por familia y estabilidad", pd.DataFrame([{"Categoría del protocolo principal":k.replace("_"," "), "Procedimiento":nombre(v)} for k,v in cat.items()]))
    kn_diag = diag[(diag.procedimiento=="KNN") & diag.protocolo.isin(PROTOCOLOS)].groupby("protocolo").distancia_media_vecinos.mean()
    tree_diag = diag[(diag.procedimiento=="Arbol_decision") & (diag.protocolo=="configuracion")]
    nota(f"Distancia media a los vecinos en geometría estandarizada: {fmt(kn_diag['configuracion'])} por "
          f"configuración, {fmt(kn_diag['familia'])} por nivel. El árbol inicial tuvo "
          f"entre {int(tree_diag.hojas.min())} y {int(tree_diag.hojas.max())} hojas en configuraciones.")


def mostrar_sensibilidad(oof, proc):
    por_caso = oof[(oof.protocolo=="configuracion") & oof.procedimiento.eq(proc)].sort_values("error_absoluto_N",ascending=False)
    alto = por_caso.iloc[0]
    mae_sin_alto = por_caso.iloc[1:].error_absoluto_N.mean()
    nota(f"Sensibilidad del MAE de {nombre(proc)}: al omitir del promedio la observación con mayor error "
          f"pasa de {fmt(por_caso.error_absoluto_N.mean())} a {fmt(mae_sin_alto)} N. Se conservaron todas las probetas.")


def mostrar_contraste_baseline(errores, delta_baseline):
    comparativa = pd.DataFrame({
        "MAE config. [N]": errores.configuracion,
        "Δ vs. baseline [N]": delta_baseline.configuracion,
        "MAE nivel Y [N]": errores.familia,
        "Δ vs. baseline en nivel Y [N]": delta_baseline.familia,
    })
    orden_inicial = errores.loc[list(MODELOS)].sort_values("configuracion").index
    puestos = {modelo: str(i) for i, modelo in enumerate(orden_inicial, start=1)}
    comparativa.insert(0, "Puesto inicial", [puestos.get(m, "—") for m in comparativa.index])
    tabla(comparativa.rename(index=NOMBRES).rename_axis("Modelo o referencia").reset_index())


def resultados_ajuste(mae_pareado, delta_baseline):
    lecturas_ajuste = []
    selector = mae_pareado.loc["Selector_tuning"]
    for protocolo, escenario in [("configuracion", "Al reservar una configuración"),
                                  ("familia", "Al retirar un nivel Y")]:
        antes, despues = selector.loc[protocolo, ["inicial", "ajustada"]]
        diferencia = despues - antes
        frente_media = delta_baseline.at["Selector_tuning", protocolo]
        posicion = "por encima de" if frente_media > 0 else "por debajo de"
        contraste = (f"{fmt(abs(frente_media), 3)} N {posicion} la media por nivel"
                     if frente_media else "al mismo nivel que la media por nivel")
        signo = "+" if diferencia > 0 else ""
        lecturas_ajuste.append(f"**{escenario}**, el MAE del selector pasó de **{fmt(antes, 3)} a {fmt(despues, 3)} N** "
                              f"(Δ {signo}{fmt(diferencia, 3)} N frente a las mismas familias iniciales). "
                              f"El MAE ajustado quedó {contraste}.")
    enumerar_resultados(lecturas_ajuste)


def resultados_modelos(errores, delta_baseline, metricas_nominales, recomendacion, umbral, referencia):
    iniciales = errores.loc[list(MODELOS)]
    ganador = iniciales.configuracion.idxmin()
    ganador_nivel = iniciales.familia.idxmin()
    no_lineales = iniciales.drop(index="Lineal_multiple")
    mejor_no_lineal = no_lineales.configuracion.idxmin()
    mejor_no_lineal_nivel = no_lineales.familia.idxmin()
    ventaja = -delta_baseline.at[ganador, "configuracion"]
    rmse_ganador, rmse_base = metricas_nominales.loc[[ganador, referencia], "RMSE_N"]

    lecturas_decision = [
         f"**Mejor modelo por MAE externo.** En configuraciones reservadas, {nombre(ganador)} obtuvo "
         f"**{fmt(iniciales.at[ganador, 'configuracion'], 3)} N**, frente a {fmt(errores.at[referencia, 'configuracion'], 3)} N "
         f"de la media por nivel y {fmt(errores.at['Ridge_historico', 'configuracion'], 3)} N de Ridge. "
         f"Su ventaja sobre la media fue de **{fmt(ventaja, 3)} N**. Los RMSE, {fmt(rmse_ganador, 4)} y "
         f"{fmt(rmse_base, 4)} N respectivamente, fueron prácticamente iguales.",
         f"**Decisión para niveles conocidos.** La referencia operativa es **{nombre(recomendacion['referencia_operativa'])}**. "
         f"El criterio de adopción exige al menos {fmt(umbral, 1)} N de mejora en MAE, sin aumentar el RMSE, "
         "con mejora en la mayoría de los grupos y con ventaja al omitir cada grupo por turno. "
         f"Entre los modelos no lineales, {nombre(mejor_no_lineal)} obtuvo el menor MAE "
         f"({fmt(no_lineales.at[mejor_no_lineal, 'configuracion'], 3)} N).",
         f"**Transferencia a un nivel Y ausente.** {nombre(ganador_nivel)} obtuvo el menor MAE "
         f"({fmt(iniciales.at[ganador_nivel, 'familia'], 3)} N). La referencia por nivel recurrió a la media global "
         f"del entrenamiento y alcanzó {fmt(errores.at[referencia, 'familia'], 3)} N. "
         f"Entre los no lineales, el menor MAE fue el de {nombre(mejor_no_lineal_nivel)} "
         f"({fmt(no_lineales.at[mejor_no_lineal_nivel, 'familia'], 3)} N). La comparación describe esta campaña.",
    ]
    enumerar_resultados(lecturas_decision)


def presentar_arquitectura(familia, estimador, unidades, ecuacion):
    """Muestra en Jupyter los pasos ajustados y la ecuación física, si es lineal."""
    etiquetas = {"contract": "Validar geometría A", "imputer": "Imputar con la mediana",
                 "variance": "Quitar constantes", "scale": "Estandarizar",
                 "basis": "Conservar una base independiente"}
    flujo = " → ".join(etiquetas.get(clave, type(paso).__name__) for clave, paso in estimador.steps)
    nota(f"**{nombre(familia)}**\n\n{flujo} → **Carga de primera falla [N]**")
    if ecuacion is None:
        return

    intercepto, pendientes = ecuacion
    pasos = estimador.named_steps
    imputador = pasos["imputer"]
    variables = pd.Index(imputador.feature_names_in_)
    con_varianza = variables[pasos["variance"].get_support()]
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
                   f"(tolerancia relativa {pasos['basis'].rtol:g}). "
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
