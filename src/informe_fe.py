"""Documentación derivada de resultados ejecutados, sin cifras copiadas manualmente."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .eda import write_bytes_atomic
from .feature_engineering import GEOMETRY
from .presentacion_fe import NOMBRES_VARIABLES, ETIQUETAS_VARIABLES, UNIDADES_VARIABLES, EXPERIMENTOS


def md_table(frame):
    def cell(x):
        if isinstance(x, (float, np.floating)):
            return f"{x:.4f}"
        return str(x).replace("|", "/").replace("\n", " ")
    return "\n".join(["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                     ["| " + " | ".join(cell(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)])


def decision_matrix():
    return pd.DataFrame([
        ["Separación entre familias: 96.55 % de variación entre grupos", "reports/eda/tables/05_descomposicion.csv", "Arquitectura dominante; generalización incierta", "Conservar n_cells_Y y probar familia excluida", "Conteo de celdas previo al ensayo", "Conteo numérico; no target encoding", "Una familia nueva puede exigir extrapolación", "A–D; protocolo secundario"],
        ["Momentos de sección: asociación condicionada próxima a cero", "reports/eda_transformaciones/tables/T08_asociaciones_caracteristicas.csv", "Hipótesis mecánica aún no validada predictivamente", "I_fibra_bh3_mm4; I_fibra_hb3_mm4", "Segundos momentos de área de sección rectangular ideal, según dos ejes", "b*h^3/12; h*b^3/12", "No son EI, rigidez medida ni criterio de fractura; base discrepante", "B − A"],
        ["log(Vp/Vf): rho global 0.8280 y condicionado −0.4773", "reports/eda_transformaciones/tables/T08_asociaciones_caracteristicas.csv", "Asociación dependiente de familia; evaluar otra representación", "log_volumen_pivotes_sobre_fibras", "Balance de volúmenes declarados de componentes", "ln(Vp/Vf), adimensional", "Asociación explorada con todo el target; posible sesgo de selección", "C − B"],
        ["Vf + Vp = 6524.01 mm³; radio constante", "reports/eda/tables/12_identidades.csv; reports/eda_calidad/tables/14_decisiones.csv", "Redundancia algebraica y varianza nula", "Retirar Vf del bloque original; filtro de varianza en train", "Los volúmenes complementarios no son efectos independientes", "Retirada fija de Vf en D; VarianceThreshold en todos", "Ridge cambia su penalización al cambiar representación", "D − C; A–D"],
        ["Longitud y número de pivotes quedan determinados por la familia observada", "reports/eda/tables/12_mapa_topologia.csv", "Descriptores repetidos de arquitectura", "Retirar Fiber_total_length y Pivot_total_number", "Se conserva el conteo Y", "Reducción física fija, sin elegir por target", "Determinismo en este diseño no implica relación universal ni redundancia lineal exacta", "D − C"],
        ["Rango centrado 7 entre 9 entradas variables; VIF infinitos", "reports/eda_transformaciones/tables/T03_representaciones.csv; reports/eda/tables/16_VIF.csv", "Colinealidad impide atribuir efectos individuales", "Ridge y ablación de bloque", "No interpretar coeficientes como mecanismos aislados", "Regularización fija alpha=1; diagnóstico de rango en cada train", "Retirar columnas modifica regularización efectiva", "A–D"],
        ["Log de volumen CAD no reduce asimetría; log de Vp sí la atenúa", "reports/eda_transformaciones/tables/T04_logaritmos.csv", "No hay soporte para logaritmos masivos", "Una razón logarítmica dirigida", "Argumento positivo y adimensional", "No transformar target ni todos los volúmenes", "Log no añade una medición nueva", "C − B"],
        ["Geometría completa; objetivo ausente conservado fuera de tabla supervisada", "reports/eda_calidad/tables/14_decisiones.csv", "No inventar etiquetas ni señales de ausencia", "Imputación de entradas dentro del pipeline", "Mediana solo de train como contingencia; no reconstruye medición", "SimpleImputer; sin indicadores de ausencia no observados", "Futuros nulos deben revisarse; columna sin soporte en train rechazada", "A–D"],
        ["Sin alertas IQR/MAD en geometría y target; Cook señala casos 1 y 3", "reports/eda/tables/07_atipicos_resumen.csv; reports/eda/tables/20_influencia.csv", "Influencia sin justificación de eliminación", "Conservar casos y estudiar contribución a errores OOF", "Extremo geométrico puede ser un diseño real", "StandardScaler; sin winsorización; RobustScaler diferido", "Excluir scores no elimina influencia en modelos entrenados", "A–D; sensibilidad descriptiva"],
        ["Factores nominales repetidos; no constan lotes o réplicas físicas", "reports/eda_calidad/tables/09_configuraciones_repetidas.csv", "Dependencia potencial", "Grupo n_cells_Y, Fiber_height, Pivot_height", "Mismo conjunto nominal no implica espécimen idéntico", "LeaveOneGroupOut sin dividir el grupo", "No controla dependencia de lotes desconocidos", "A–D; validación principal"],
        ["Discrepancias geométricas de base y alturas; casos 23,25,26 señalados en pivotes", "reports/eda_calidad/tables/14_decisiones.csv", "Incertidumbre en atributos derivados", "Conservar snapshot y registrar sensibilidad", "No sustituir una fuente automáticamente por otra", "Comparación de scores con y sin casos señalados", "No es validación con geometría corregida", "A–D"],
        ["Energía, residuales y carga terminal se conocen durante/después del ensayo", "docs/diccionario_datos.md; reports/eda_transformaciones/tables/T06_frontera_predictiva.csv", "Fuga temporal aunque haya correlación alta", "Lista cerrada de entradas geométricas", "Solo información disponible antes del ensayo", "Contrato rechaza columnas adicionales; Case_n solo traza", "Errores de metadatos requieren verificación experimental", "A–D"],
    ], columns=["Hallazgo EDA", "Evidencia", "Problema u oportunidad", "Feature propuesta", "Justificación física", "Transformación", "Riesgo", "Experimento"])


def write_documents(root, config, summary, decisions, sensitivity, family, model_summary):
    """Guía breve; el notebook es la entrega principal y los CSV son el respaldo."""
    out = root / "reports/feature_engineering"
    def write(path, text):
        write_bytes_atomic(path, (text.rstrip() + "\n").encode("utf-8"))
    matrix = decision_matrix()
    write(out / "tables/matriz_eda_fe.csv", matrix.to_csv(index=False, lineterminator="\n"))
    primary = summary[summary.protocolo == "configuracion"].set_index("experimento")
    secondary = summary[summary.protocolo == "familia"].set_index("experimento")
    best = primary.loc[list("ABCD"), "MAE_N"].idxmin()
    improvement = 100 * (primary.loc["A", "MAE_N"] - primary.loc[best, "MAE_N"]) / primary.loc["A", "MAE_N"]
    verdict = dict(zip(decisions.experimento, decisions.decision))
    compact = primary.loc[list("ABCD"), ["MAE_N", "RMSE_N", "MedAE_N", "R2_OOF", "tiempo_s"]].reset_index()
    compact["Cambio"] = ["Geometría original", "A + dos momentos de sección", "B + ln(Vp/Vf)", "C − tres descriptores redundantes"]
    compact["experimento"] = compact["experimento"].map(EXPERIMENTOS)
    compact = compact[["experimento", "Cambio", "MAE_N", "RMSE_N", "MedAE_N", "R2_OOF", "tiempo_s"]].rename(columns={"experimento":"Experimento", "MAE_N":"MAE (N)", "RMSE_N":"RMSE (N)", "MedAE_N":"MedAE (N)", "R2_OOF":"R² OOF", "tiempo_s":"Tiempo CV (s)"})
    result = f"**{best} obtiene el menor MAE: {primary.loc[best,'MAE_N']:.4f} N**, una reducción de {improvement:.2f} % frente a A. Al excluir familias completas, su MAE es {secondary.loc[best,'MAE_N']:.4f} N; la mejora no demuestra transferencia a arquitecturas nuevas."
    brief = ("B: mantener en evaluación; C: descartar la razón añadida frente a B; D: descartar la reducción probada. "
             "Las decisiones corresponden a este modelo y protocolo.")
    model_primary = model_summary[model_summary.protocolo == "configuracion"].copy()
    model_secondary = model_summary[model_summary.protocolo == "familia"].set_index("modelo")
    model_compact = model_primary[["nombre_modelo", "MAE_N", "RMSE_N", "MedAE_N", "R2_OOF", "tiempo_s"]].rename(columns={
        "nombre_modelo": "Modelo", "MAE_N": "MAE (N)", "RMSE_N": "RMSE (N)",
        "MedAE_N": "MedAE (N)", "R2_OOF": "R² OOF", "tiempo_s": "Tiempo CV (s)",
    })
    best_model = model_primary.loc[model_primary.MAE_N.idxmin()]
    second_model = model_primary.sort_values("MAE_N").iloc[1]
    model_result = (f"**{best_model['nombre_modelo']} presenta el menor MAE por configuración: {best_model['MAE_N']:.4f} N.** "
                    f"La diferencia respecto a {second_model['nombre_modelo']} es {second_model['MAE_N'] - best_model['MAE_N']:.4f} N. "
                    f"Al excluir familias completas, su MAE es {model_secondary.loc[best_model['modelo'], 'MAE_N']:.4f} N. "
                    "La comparación es exploratoria: no constituye una selección validada en ensayos externos.")
    model_descriptions = pd.DataFrame([
        ["Ridge", "Referencia lineal regularizada frente a entradas correlacionadas."],
        ["Elastic Net", "Contrasta otra regularización lineal; puede reducir algunos coeficientes a cero."],
        ["SVR con núcleo RBF", "Estudia relaciones no lineales entre geometría y fuerza."],
        ["Bosque aleatorio (Random Forest)", "Combina árboles para representar interacciones y umbrales geométricos."],
        ["Proceso gaussiano", "Contrasta una función no lineal mediante similitud entre geometrías."],
    ], columns=["Modelo de regresión", "Motivo de inclusión"])
    metric_descriptions = pd.DataFrame([
        ["MAE [N] · principal", "Promedio del error absoluto; menor es mejor."],
        ["RMSE [N]", "Penaliza más los errores grandes; menor es mejor."],
        ["MedAE [N]", "Mediana del error absoluto; resume el error central con menor sensibilidad a extremos."],
        ["R² OOF", "Se calcula sobre todas las predicciones fuera de muestra. 1 indica ajuste perfecto; negativo indica mayor error cuadrático que la media global observada como referencia."],
        ["Tiempo CV [s]", "Suma medida de ajuste y predicción de las particiones; excluye figuras y archivos. Depende del equipo, no es latencia de despliegue."],
    ], columns=["Métrica", "Interpretación"])
    # Los campos completos del diccionario siguen disponibles en CSV.
    features = pd.DataFrame([
        ["I_fibra_bh3_mm4", "Fiber_base=b; Fiber_height=h", "b*h³/12", "mm⁴", "Segundo momento de área: eje paralelo a b", "Representar la forma de sección", "Mediana train antes de fórmula", "No usa target; hipótesis del EDA global", "B–D", verdict["B"]],
        ["I_fibra_hb3_mm4", "Fiber_base=b; Fiber_height=h", "h*b³/12", "mm⁴", "Segundo momento de área: eje paralelo a h", "Representar la otra orientación", "Mediana train antes de fórmula", "No usa target; hipótesis del EDA global", "B–D", verdict["B"]],
        ["log_volumen_pivotes_sobre_fibras", "Pivot_total_volume=Vp; Fiber_total_volume=Vf", "ln(Vp/Vf)", "1", "Razón entre volúmenes declarados", "Representar balance de componentes", "Mediana train; ingredientes positivos", "No usa target; hipótesis del EDA global", "C–D", verdict["C"]],
    ], columns=["Feature", "Variables de origen", "Fórmula", "Unidad", "Significado físico", "Hipótesis", "Tratamiento de nulos", "Riesgo de leakage", "Experimento", "Decisión"])
    write(out / "tables/diccionario_features.csv", features.to_csv(index=False, lineterminator="\n"))
    variables = [*GEOMETRY, "First_Failure_Load_N", *features["Feature"]]
    correspondence = pd.DataFrame([
        {
            "Variable en español": NOMBRES_VARIABLES.get(name, name),
            "Significado": ETIQUETAS_VARIABLES.get(name, name),
            "Columna original": name,
            "Unidad": UNIDADES_VARIABLES.get(name, "1"),
            "Rol": "Objetivo / salida predicha" if name == "First_Failure_Load_N" else "Entrada derivada" if name in set(features["Feature"]) else "Entrada original",
        }
        for name in variables
    ])
    write(out / "tables/nombres_variables.csv", correspondence.to_csv(index=False, lineterminator="\n"))
    visible_features = features[["Feature", "Variables de origen", "Fórmula", "Unidad", "Significado físico"]].copy()
    visible_features["Feature"] = visible_features["Feature"].map(lambda name: NOMBRES_VARIABLES.get(name, name))
    visible_features["Variables de origen"] = ["Ancho de fibra = b; altura de fibra = h", "Ancho de fibra = b; altura de fibra = h", "Volumen de pivotes = Vp; volumen de fibras = Vf"]
    write(root / "docs/diccionario_features.md", f"""# Diccionario de características

[Notebook principal](../notebooks/04_Feature_Engineering_Pantograficas.ipynb) · [Correspondencia español–fuente CSV](../reports/feature_engineering/tables/nombres_variables.csv) · [Diccionario técnico CSV](../reports/feature_engineering/tables/diccionario_features.csv)

## Variables de esta etapa

El notebook presenta las variables en español. La columna original permite rastrear cada medición hasta la tabla curada, sin modificar los datos de origen.

{md_table(correspondence)}

## Variables construidas

{md_table(visible_features)}

Los momentos se evalúan juntos en B; son geometría de sección, no rigidez EI medida. La razón C es adimensional y exige volúmenes positivos. Todas las fórmulas usan entradas previas al ensayo, después de la imputación ajustada en entrenamiento.

El ancho de fibra (base de la sección) se incluye en todas las variantes aunque el esquema general actual la omita. El radio constante se retira dentro de cada entrenamiento. D retira longitud total de fibras, cantidad de pivotes y volumen total de fibras del bloque original; este último sigue alimentando la razón. El objetivo es la **carga de primera falla [N]**; identificadores y otras respuestas del ensayo no son predictores. [Definiciones experimentales completas](diccionario_datos.md#consulta-rápida).
""")
    write(root / "docs/feature_engineering.md", """# Protocolo de ingeniería de características

**Entrega principal:** [notebook 04 ejecutado](../notebooks/04_Feature_Engineering_Pantograficas.ipynb). [Matriz EDA → FE completa](../reports/feature_engineering/tables/matriz_eda_fe.csv).

| Elemento | Decisión |
|---|---|
| Fuente | Tabla curada `model_table.csv`, contrastada con su manifiesto; target observado, sin imputación. |
| Objetivo | Carga de primera falla [N]: fuerza correspondiente a la primera rotura de fibra o pivote. |
| Entradas | Geometría previa al ensayo; se excluyen identificador, fuerzas, desplazamientos y energías. |
| Experimentos | Línea base A: geometría original. Variante 1 B: añadir dos momentos como bloque. Variante 2 C: añadir razón logarítmica a B. Variante 3 D: retirar tres descriptores como bloque de C. |
| Modelo de los contrastes A–D | Ridge fijo; mismos parámetros y particiones. Dummy de media como control auxiliar. |
| Comparación de algoritmos | Ridge, Elastic Net, SVR con núcleo RBF, bosque aleatorio y proceso gaussiano; todos con la geometría original A y las mismas particiones. |
| Pipeline | Contrato de entradas → imputación con mediana → representación geométrica → filtro de varianza nula → estandarización → regresor; estadísticas aprendidas solo en entrenamiento. |
| Validación principal | Validación cruzada LeaveOneGroupOut por combinación de cantidad de celdas en Y, altura de fibra y altura de pivote; configuraciones nominales repetidas permanecen juntas. |
| Comprobación adicional | Excluir cada familia Y completa; evalúa una arquitectura no vista. |
| Métricas | MAE principal; RMSE, MedAE y R² agregado OOF. Sin R² en folds individuales. |
| Tiempo | Suma de los tiempos medidos de ajuste y predicción de los folds. Excluye generación de figuras y escritura de archivos; depende del equipo. |
| Selección | Retirada fija por hipótesis geométrica; varianza nula aprendida en train. Sin selección supervisada ni tuning. |
| Decisión | Adoptar provisionalmente si MAE mejora ≥5 %, gana en ≥60 % de grupos y no empeora RMSE ni MAE agregado al excluir familias completas; si solo mejora MAE, mantener en evaluación; si no, descartar. |

Las hipótesis proceden del EDA del mismo conjunto: los resultados son exploratorios y pueden tener sesgo de selección. Faltan réplicas/lotes verificados y persisten discrepancias geométricas. Las familias no representan campañas independientes.

Las nuevas características son funciones de mediciones existentes. Retirar columnas modifica también la penalización efectiva de Ridge. No se recortan predicciones negativas ni se eliminan extremos. La sensibilidad por omisión modifica el agregado de errores OOF, no reentrena modelos.

## Técnicas utilizadas

| Técnica | Aplicación |
|---|---|
| Construcción de variables físicas | Dos momentos de área de sección rectangular y una razón logarítmica de volúmenes; fórmulas por fila, sin utilizar la respuesta. |
| Imputación por mediana | Contingencia aprendida en cada entrenamiento. La geometría actual está completa: no se rellenaron mediciones. El objetivo nunca se imputa. |
| Filtro de varianza nula | Detecta y retira columnas constantes usando únicamente el entrenamiento; aquí corresponde al radio de pivote. |
| Estandarización (StandardScaler) | Resta la media y divide por la desviación del entrenamiento. La validación reutiliza esas estadísticas. |
| Reducción de variables | En D se retiran tres descriptores por hipótesis geométrica fija, sin seleccionar columnas por el objetivo. |
| One-hot encoding | No se utilizó: las entradas admitidas son mediciones o conteos numéricos. La cantidad de celdas en Y conserva su valor numérico; usarla para agrupar no la convierte en una categoría codificada. |

Cada contraste modifica **un bloque**: B−A, C−B y D−C. Los dos momentos se prueban juntos y las tres retiradas también; el diseño no permite atribuir la mejora a cada variable individual. El modelo y las particiones se mantienen fijos en estos contrastes.

La comparación de cinco regresores es un experimento separado: conserva la representación A, el preprocesamiento y las particiones, y cambia únicamente el algoritmo con sus parámetros declarados. No se combinan cinco modelos con cuatro variantes ni se presenta esta comparación como ajuste de hiperparámetros. Las reglas de adopción de características no seleccionan automáticamente un algoritmo.

## Controles de fuga y separación

Se comprueba que entrenamiento y validación no comparten casos ni grupos; las respuestas del ensayo y los identificadores no entran al modelo. La imputación, el filtro y la estandarización se ajustan dentro de cada entrenamiento. Esto controla la fuga de procesamiento, pero **no acredita ausencia absoluta de sesgo**: las hipótesis ya se exploraron con el EDA del conjunto completo y faltan lotes experimentales verificados. Se ejecutó validación cruzada por grupos, no un holdout externo ni una partición temporal o estratificada.

[Configuración de características](../config/feature_engineering.json) · [Configuración de modelos](../config/comparacion_modelos.json) · [Particiones](../reports/feature_engineering/particiones.json) · [Estadísticas aprendidas](../reports/feature_engineering/auditoria_transformaciones.json)
""")
    pairwise = decisions[["experimento", "referencia", "delta_MAE_N", "fraccion_grupos_mejora", "decision"]].copy()
    pairwise["experimento"] = pairwise["experimento"].map(EXPERIMENTOS)
    pairwise["referencia"] = pairwise["referencia"].map(EXPERIMENTOS)
    pairwise = pairwise.rename(columns={"experimento": "Variante", "referencia": "Referencia", "delta_MAE_N": "ΔMAE (N)", "fraccion_grupos_mejora": "Fracción de grupos con mejora", "decision": "Decisión"})
    write(root / "docs/experimentos_fe.md", f"""# Resultados de ingeniería de características

[Ver el desarrollo, las figuras y la interpretación en el notebook](../notebooks/04_Feature_Engineering_Pantograficas.ipynb).

## Características: mismo modelo Ridge

{md_table(compact)}

**Tiempo CV:** suma medida de ajuste y predicción en la validación por configuración; excluye figuras y escritura de archivos. Es costo computacional de esta ejecución, no costo monetario ni latencia de una aplicación.

{result}

## Intervenciones

{md_table(pairwise)}

ΔMAE negativo indica mejora frente a la referencia de cada fila. {brief}

La variante D produce fuerzas negativas al excluir Y=4: un fallo físico de extrapolación, conservado en las métricas. La comprobación de sensibilidad omite scores del agregado OOF; no elimina su influencia en los modelos entrenados.

## Algoritmos: misma geometría original A

Los cinco regresores usan las mismas entradas, preprocesamiento y particiones. Cada fila reúne predicciones de configuraciones excluidas del entrenamiento; los parámetros se fijaron sin búsqueda de hiperparámetros.

{md_table(model_compact)}

{model_result}

El notebook muestra errores, residuales y el contraste entre excluir configuraciones o familias completas. [Resultados de ambos protocolos](../reports/feature_engineering/tables/comparacion_modelos.csv).

## Alcance

El EDA ya utilizó todo el conjunto. Estos resultados no son validación externa ni garantizan seguridad estructural. El siguiente experimento es confirmar A y {best} con geometrías conciliadas y ensayos independientes, manteniendo el protocolo congelado.

[Resultados completos y tiempos](../reports/feature_engineering/tables/comparacion_experimentos.csv) · [Predicciones OOF](../reports/feature_engineering/tables/predicciones_oof.csv) · [Errores por familia](../reports/feature_engineering/tables/errores_familia.csv) · [Sensibilidad](../reports/feature_engineering/tables/sensibilidad_scores.csv)
""")
    captions = [
        ("FE08_momentos", "Tratamiento 1. De la sección de fibra a los momentos de área", f"Los dos momentos representan orientaciones distintas de una sección rectangular; no son rigidez EI medida. Se prueban juntos: ΔMAE de variante 1 frente a la base = {primary.loc['B', 'MAE_N'] - primary.loc['A', 'MAE_N']:.4f} N."),
        ("FE09_razon_log", "Tratamiento 2. Razón de volúmenes antes y después del logaritmo", f"El logaritmo conserva el orden y modifica las distancias entre valores. Un resultado negativo indica menos volumen de pivotes que de fibras. Añadir esta característica cambia el MAE en {primary.loc['C', 'MAE_N'] - primary.loc['B', 'MAE_N']:+.4f} N frente a la variante 1."),
        ("FE10_redundancia", "Tratamiento 3. Evidencia geométrica para estudiar redundancia", f"La dependencia se observa en este diseño. El radio constante se filtra en entrenamiento; retirar además el bloque de tres descriptores cambia el MAE en {primary.loc['D', 'MAE_N'] - primary.loc['C', 'MAE_N']:+.4f} N frente a la variante 2. La redundancia no basta para recomendar su eliminación."),
        ("FE11_escalamiento", "Tratamiento 4. Escalas originales y estandarización", "Las medias y escalas proceden únicamente del entrenamiento. El caso reservado reutiliza esas estadísticas y no tiene por qué quedar centrado en cero. La estandarización cambia la escala numérica, no la geometría física."),
        ("FE06_representaciones", "Figura 1. Variables que llegan a cada experimento", "El filtro de varianza retira el radio constante dentro de cada entrenamiento. D retira además tres columnas por hipótesis geométrica; el volumen de fibras sigue interviniendo en la razón logarítmica. Añadir columnas no añade mediciones."),
        ("FE07_rango", "Figura 2. Dimensión y rango de la representación", "El rango se calcula después de transformar y estandarizar cada entrenamiento. Un rango menor que el número de columnas indica dependencia lineal en ese diseño. Las fórmulas no lineales pueden aumentar el rango sin aportar nuevas observaciones ni garantizar menor error."),
        ("FE01_error", "Figura 3. Error por representación", "Menor MAE significa menor error medio. La dispersión entre grupos se muestra en la figura 7; estas barras no son intervalos de confianza."),
        ("FE02_observado_predicho", "Figura 4. Observado frente a predicho", "La diagonal representa coincidencia perfecta. Cada predicción procede de un modelo que no entrenó con esa configuración nominal."),
        ("FE03_residuos", "Figura 5. Residuales", "Residual = observado − predicho: positivo indica subestimación. Los casos señalados tienen los mayores errores absolutos; no se eliminan."),
        ("FE04_error_familia", "Figura 6. Error por familia", "En la validación por configuración, B mejora frente a A en Y=5 y Y=6, pero empeora en Y=4. Excluir una familia completa aumenta mucho el error: generalizar a otra arquitectura sigue sin resolverse."),
        ("FE05_ablaciones", "Figura 7. Aporte de cada intervención", "Cada punto es el cambio de MAE de un grupo; negativo indica mejora. El rombo es la media de grupos, no ponderada por casos. No mide importancia causal individual."),
        ("FE12_modelos_metricas", "Figura 8. Cinco regresores con las mismas entradas", "La geometría original A y las particiones permanecen fijas; cambia el algoritmo con sus parámetros declarados. MAE es la métrica principal y RMSE penaliza errores grandes. Los tiempos corresponden al ajuste y predicción de la validación cruzada, no al despliegue."),
        ("FE13_modelos_residuos", "Figura 9. Residuales fuera de muestra por modelo", "Residual = observado − predicho: positivo indica subestimación y negativo, sobreestimación. Cada caso se predice sin entrenar con su configuración. Los errores se conservan sin recortar predicciones ni retirar casos."),
        ("FE14_modelos_generalizacion", "Figura 10. Configuraciones nuevas frente a familias excluidas", "Excluir una configuración y excluir una familia completa responden a preguntas distintas. El segundo protocolo exige predecir una arquitectura no vista; comparar ambos errores muestra el límite de transferencia de cada algoritmo. No equivale a una validación con otra campaña experimental."),
    ]
    import json
    figure_sources = {
        **{name: "Geometría y auditoría de la variante 2, partición por configuración que reserva el primer caso por identificador; sin utilizar la respuesta."
           for name in ["FE08_momentos", "FE09_razon_log", "FE10_redundancia", "FE11_escalamiento"]},
        "FE06_representaciones": "auditoria_transformaciones.json: presencia de columnas en cada entrenamiento; categorías sin unidad.",
        "FE07_rango": "tables/metricas_folds.csv: dimensión y rango de cada entrenamiento; conteos sin unidad física.",
        "FE12_modelos_metricas": "tables/comparacion_modelos.csv: métricas OOF y tiempos de la comparación de regresores.",
        "FE13_modelos_residuos": "Predicciones OOF de los cinco regresores; geometría original A y validación por configuración. Fuerzas y residuales en N.",
        "FE14_modelos_generalizacion": "tables/errores_modelos_familia.csv: MAE OOF de cada familia Y en los protocolos por configuración y por familia, en N.",
    }
    write(out / "leyendas.json", json.dumps([
        {"file": name, "title": title, "interpretation": text,
         "source": figure_sources.get(name, "Tabla curada y predicciones OOF; fuerzas y errores en N.")}
        for name, title, text in captions
    ], ensure_ascii=False, indent=2))
    readme = f"""# Ingeniería de características de estructuras pantográficas

**[Abrir notebook 04 ejecutado →](notebooks/04_Feature_Engineering_Pantograficas.ipynb)**

**Enlace para compartir esta entrega:** [notebook actualizado en la rama feature-engineering](https://github.com/melissaaylasb-sudo/T02_MIA14_SeccionB/blob/feature-engineering/notebooks/04_Feature_Engineering_Pantograficas.ipynb). Las actualizaciones de este trabajo se publican en esa misma ruta.

Melissa Dessire Aylas Barranca · UNI, Maestría en Inteligencia Artificial · Trabajo de Investigación II.

Se evalúa si transformar la geometría mejora la predicción de la **carga de primera falla**: fuerza, en newtons, de la primera rotura de una fibra o un pivote de la malla. Esta entrega vive en la rama `feature-engineering`, separada de `eda`.

| Elemento | Especificación |
|---|---|
| Entradas / INPUT | Cantidad de celdas en Y; altura, ancho, longitud y volumen de fibras; altura, radio, cantidad y volumen de pivotes; volumen CAD del espécimen. [Variables en español y correspondencia con la fuente](docs/diccionario_features.md#variables-de-esta-etapa). |
| Nuevas características | Dos momentos de sección, b·h³/12 y h·b³/12 [mm⁴], y ln(Vp/Vf) [sin unidad]. Se calculan con geometría previa al ensayo. |
| Objetivo / target | Carga de primera falla [N], observada en el ensayo; su columna de origen es `First_Failure_Load_N`. Se utiliza para entrenar y evaluar, nunca para construir entradas. |
| Salida / OUTPUT | Estimación numérica de la fuerza de primera falla [N], junto con su error fuera de muestra al validarla. No predice el instante ni el lugar de rotura. |
| Comparación de características | Línea base A y tres variantes B/C/D con Ridge fijo. Cada intervención se contrasta con la anterior. A–D son representaciones, no algoritmos. |
| Comparación de modelos | Cinco regresores, todos con geometría original A y las mismas particiones; parámetros declarados, sin búsqueda de hiperparámetros. Control auxiliar de media del entrenamiento. |

## Resultado

{md_table(compact)}

{result} **B se mantiene en evaluación**; C no mejora B y D empeora. El EDA previo usó el mismo conjunto: la evidencia es exploratoria.

## Cinco modelos de regresión

{md_table(model_descriptions)}

La comparación cambia el algoritmo y conserva las entradas A para distinguir su efecto del tratamiento de características.

{md_table(model_compact)}

{model_result}

## Qué encontrarás en el notebook

| Sección | Evidencia ejecutada |
|---|---|
| 4.1–4.2 · Datos y construcción | Entradas autorizadas, hipótesis del EDA y cálculo de las nuevas variables con un caso real. |
| 4.3–4.4 · Tratamiento visual | Sección y momentos de área, razón antes/después del logaritmo, redundancias, escalamiento, mapa de variables y rango. |
| 4.5 · Auditoría de una predicción | Separación de grupos, estadísticas aprendidas solo en entrenamiento y transformación de un caso reservado. |
| 4.6–4.8 · Evaluación | Predicciones fuera de muestra, residuales, errores por familia, ablaciones y sensibilidad. |
| 4.9 · Interpretación del tratamiento | Reglas de decisión contrastadas con los resultados de A–D. |
| 4.10 · Cinco modelos | Tabla común de métricas, tiempos, residuales y generalización por familia. |
| 4.11 · Verificación | Integridad de datos, fuentes y artefactos ejecutados. |

{md_table(metric_descriptions)}

## Técnicas y validación

El pipeline aplica **imputación por mediana → representación geométrica → filtro de varianza nula → estandarización → regresor**. La mediana es una contingencia: las entradas geométricas actuales están completas. El radio constante se retira en cada entrenamiento. D retira longitud total de fibras, cantidad de pivotes y volumen de fibras; este último sigue interviniendo en la razón logarítmica.

**No se usó one-hot encoding:** todas las entradas son mediciones o conteos numéricos. Cada comparación cambia un bloque: dos momentos, luego una razón y después la retirada de tres columnas; no se atribuyen efectos a variables individuales dentro de esos bloques.

Se ejecuta **validación cruzada LeaveOneGroupOut** por configuración nominal y una segunda comprobación por familia Y. No hay holdout externo. El notebook verifica grupos disjuntos, exclusión de respuestas e identificadores y ajuste de transformaciones solo en entrenamiento. El EDA previo utilizó el mismo conjunto, por lo que estas comprobaciones no equivalen a garantizar ausencia absoluta de sesgo.

## Estructura del proyecto

```text
notebooks/
  01_Ingesta_Curaduria_y_Calidad.ipynb
  02_EDA.ipynb
  03_Transformaciones_Exploratorias.ipynb
  04_Feature_Engineering_Pantograficas.ipynb  # Entrega actual ejecutada
src/
  feature_engineering.py   # Contrato, fórmulas y pipeline
  experimentos_fe.py       # Particiones, ajustes y métricas OOF
  comparacion_modelos.py   # Cinco regresores con representación A y particiones comunes
  informe_fe.py            # Resúmenes ligados a resultados
  presentacion_fe.py       # Nombres y etiquetas en español
  tratamiento_fe.py        # Figuras y comprobaciones del tratamiento geométrico
  reporte_fe.py            # Ejecución del notebook 04
  verificar_fe.py          # Integridad y coherencia de la entrega
  verificar_modelos.py     # Auditoría de particiones, transformaciones y métricas de modelos
config/
  feature_engineering.json  # Intervenciones sobre características
  comparacion_modelos.json  # Algoritmos y parámetros reproducibles
reports/feature_engineering/
  tables/                  # Métricas, predicciones y decisiones
  figures/                 # PNG a 300 dpi y SVG
  particiones.json         # Casos de train y validación
  auditoria_transformaciones.json  # Estadísticas aprendidas por fold
  manifest.json            # Fuentes, versiones y hashes
  verificacion.json        # Resultado de comprobaciones
  ejecucion*.log           # Registro de ejecución
data/                     # Fuente y tablas curadas heredadas del EDA
docs/                     # Diccionarios y notas metodológicas
tests/                    # Pruebas, incluidos grupos y fuga de información
```

## Consulta y reproducción

[Variables en español](docs/diccionario_features.md) · [Protocolo](docs/feature_engineering.md) · [Resultados de características](reports/feature_engineering/tables/comparacion_experimentos.csv) · [Resultados de modelos](reports/feature_engineering/tables/comparacion_modelos.csv) · [Registro de ejecución](reports/feature_engineering/ejecucion.log) · [EDA anterior](https://github.com/melissaaylasb-sudo/T02_MIA14_SeccionB/tree/eda)

```powershell
python -m src.reporte_fe --execute
python -m pytest tests -q
python -m src.verificar_fe
```

Los datos originales y los tres cuadernos del EDA se conservan. Los PDF y las referencias privadas no se publican. Siguiente paso: confirmar A y {best} con ensayos independientes y geometrías conciliadas.
"""
    write(root / "README.md", readme)
