"""Documentación derivada de resultados ejecutados, sin cifras copiadas manualmente."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .eda import write_bytes_atomic
from .feature_engineering import GEOMETRY


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


def write_documents(root, config, summary, decisions, sensitivity, family):
    out = root / "reports/feature_engineering"
    def write(path, text):
        write_bytes_atomic(path, (text.rstrip() + "\n").encode("utf-8"))
    matrix = decision_matrix()
    write(out / "tables/matriz_eda_fe.csv", matrix.to_csv(index=False, lineterminator="\n"))
    primary = summary[summary.protocolo == "configuracion"].set_index("experimento")
    secondary = summary[summary.protocolo == "familia"].set_index("experimento")
    best = primary.loc[list("ABCD"), "MAE_N"].idxmin()
    reduction = 100 * (primary.loc["A", "MAE_N"] - primary.loc[best, "MAE_N"]) / primary.loc["A", "MAE_N"]
    headline = (f"El menor MAE del protocolo principal corresponde a **{best}**, con **{primary.loc[best, 'MAE_N']:.4f} N**, "
                f"frente a **{primary.loc['A', 'MAE_N']:.4f} N** de A; reducción descriptiva de **{reduction:.2f} %**. "
                f"Al excluir familias completas, el MAE de {best} es **{secondary.loc[best, 'MAE_N']:.4f} N**. "
                "El ganador entre variantes examinadas no constituye una estimación independiente del rendimiento de un modelo seleccionado.")
    decision_map = {r.experimento: r.decision for r in decisions.itertuples()}
    shown = summary.copy()
    shown["decision"] = shown.experimento.map({"Dummy": "Referencia ingenua", "A": "Baseline", **decision_map})
    cols = ["experimento", "feature_set", "MAE_N", "RMSE_N", "MedAE_N", "R2_OOF", "delta_MAE_vs_A_N", "cambio_MAE_vs_A_pct", "tiempo_s", "n_features_min", "n_features_max", "decision"]
    comparison = md_table(shown[shown.protocolo == "configuracion"][cols])
    caution = ("El EDA previo usó todas las observaciones y el target para formular hipótesis. Aunque las transformaciones se ajustan "
               "dentro de cada entrenamiento, existe posible sesgo de selección de hipótesis. Los resultados son exploratorios: "
               "se necesita una campaña independiente o una evaluación externa con el protocolo congelado. No se afirma causalidad, "
               "significación estadística ni seguridad estructural a partir de estas métricas.")
    methodology = f'''# Ingeniería de características guiada por el EDA

**Objetivo:** estimar `First_Failure_Load_N` (N), la fuerza asociada al primer evento de rotura identificado de fibra o pivote. No es la fuerza máxima, la carga terminal ni la detección de rotura en tiempo real.

Una estructura pantográfica es una malla de barras (fibras) conectadas por uniones (pivotes). La familia indica el número de celdas en Y. Esta etapa continúa el EDA publicado en `{config['eda_commit']}` y no recalcula sus figuras.

## Fuente y frontera temporal

Se utiliza `data/processed/preliminary/model_table.csv`, contrastado mediante SHA-256 con su reporte de curaduría. Es la tabla con objetivo observado; la geometría del caso sin respuesta queda en la fuente original. No se imputa el target. `Fiber_base` se incluye expresamente en todos los experimentos: existe en el snapshot aunque el esquema general posterior la omita. El contrato específico está en `src/feature_engineering.py` y la configuración en `config/feature_engineering.json`.

Entradas: {', '.join('`'+x+'`' for x in GEOMETRY)}. `Case_n` identifica predicciones, nunca entra al ajuste. Carga terminal, desplazamientos, energías, ciclos y eventos se excluyen por disponibilidad temporal, independientemente de su correlación. El archivo contiene resultados experimentales resumidos, no señales continuas. Persiste la discrepancia entre el resumen de primera falla y ciertos primeros picos documentales; se conserva el target publicado sin redefinirlo.

## Matriz de decisiones EDA → FE

Las rutas de evidencia son relativas a la raíz del repositorio; cada archivo se mantiene intacto. La [matriz CSV](../reports/feature_engineering/tables/matriz_eda_fe.csv) facilita su consulta.

{md_table(matrix)}

## Intervenciones y control experimental

- **A:** geometría original admisible, imputación mediana, varianza nula, StandardScaler y Ridge.
- **B = A +** dos segundos momentos de área de la sección de fibra. Se evalúa la familia de descriptores conjuntamente; no se atribuye una mejora a un eje individual.
- **C = B +** logaritmo de la razón entre volumen de pivotes y de fibras.
- **D = C −** longitud total de fibras, cantidad de pivotes y volumen de fibras del bloque original. El volumen de fibras sigue siendo necesario para construir la razón; no desaparece del contrato de entrada.

Se mantienen `Ridge(alpha=1, solver='svd')`, métrica, escalador y particiones. Alpha=1 es una referencia fija, no un óptimo estimado. No se hace tuning. La media del target de entrenamiento (`DummyRegressor`) es un control adicional, no la única comparación. No existía un baseline entrenado en esta rama; el módulo del cuaderno 03 solo preparaba representaciones, por lo que se reutilizan sus fórmulas y se agrega la evaluación supervisada.

Las features derivadas no agregan mediciones independientes: cambian la base funcional que recibe Ridge. Al reducir columnas también cambia la penalización efectiva, aun manteniendo alpha; esa es parte de la intervención D y no demuestra por sí sola irrelevancia física.

## División primero; transformaciones después

`Pipeline`: contrato geométrico → mediana de entrenamiento → fórmulas por fila → filtro de varianza nula → StandardScaler ajustado en train → Ridge. El contrato rechaza columnas extra, infinitos, dimensiones no positivas y conteos no enteros. Si una columna está totalmente ausente en entrenamiento, se detiene el ajuste. La construcción posterior a imputación permite transformar observaciones nuevas con estadísticas guardadas, aunque no sustituye la revisión de mediciones faltantes.

Todos los campos elegidos son numéricos y reciben el mismo tratamiento: no hace falta un `ColumnTransformer`. `n_cells_Y` es un conteo con orden físico, no una etiqueta ordinal arbitraria. No se añade One-Hot Encoding en esta comparación: sería otra hipótesis, usaría más parámetros y no permitiría representar directamente una familia nunca observada. No se usa target encoding.

StandardScaler evita que mm, mm³ y mm⁴ penalicen de manera incompatible. RobustScaler queda como experimento posterior: no se observan alertas geométricas IQR/MAD que justifiquen cambiarlo simultáneamente. No se winsoriza ni se eliminan extremos. Los faltantes geométricos no observados no justifican crear indicadores constantes. Solo se calcula el logaritmo positivo de una razón adimensional; no se transforman respuestas.

La selección D es fija y motivada por el diseño, no aprendida con el target. La única selección ajustada es la varianza nula dentro del fold. VIF y rango del EDA motivan estudiar redundancia; el rango se registra de nuevo exclusivamente en cada entrenamiento. No se usa información mutua ni selección univariada supervisada: su estimación y estabilidad no tienen respaldo suficiente para añadirlas a esta comparación. Se entregan coeficientes por fold para auditoría, sin interpretarlos como importancia causal. La contribución se evalúa por ablación pareada; no se usa permutación marginal de variables físicamente dependientes.

## Unidad experimental y validación

La fila es el caso experimental recibido. No están verificados identificadores físicos de réplica ni lotes. Los ciclos y eventos no se expanden como filas independientes. La combinación nominal `n_cells_Y`, `Fiber_height`, `Pivot_height` define grupos conservadores, derivados únicamente de geometría. Los pares de casos 19/25 y 24/27 comparten estos factores, aunque difieren otras dimensiones; no se los declara réplicas.

**Principal:** LeaveOneGroupOut por configuración nominal. Se excluye por completo una combinación de factores; pueden permanecer casos de la misma familia en train. Por ello evalúa configuraciones excluidas dentro del dominio experimental, no transferencia a otras campañas.

**Secundaria:** LeaveOneGroupOut por familia Y. No participa en el ajuste; examina la dificultad de una arquitectura ausente. Las familias extremas exigen extrapolación y la intermedia interpolación entre familias. Las diferencias entre protocolos responden a preguntas distintas.

Ridge no impone positividad de la respuesta. Se registran predicciones negativas como resultados físicamente inadmisibles para este objetivo; no se recortan a cero porque eso cambiaría el experimento. La tabla de validación informa su presencia y el mínimo predicho. No se propone despliegue de estos ajustes.

Las listas exactas de train/test se guardan una vez en [particiones.json](../reports/feature_engineering/particiones.json) y se reutilizan en todas las variantes. Se conservan tamaños efectivos en las tablas técnicas. No se aplica holdout 80/20. La independencia respecto a lotes desconocidos no puede garantizarse.

## Métricas y criterio de decisión

MAE (N) es principal; RMSE (N), MedAE (N) y R² sobre todas las predicciones OOF son secundarios. El residual es observado menos predicho. No se calcula R² por fold individual. El score principal pondera cada caso por igual; media y desviación de MAE por fold ponderan grupos por igual y describen dispersión, no un intervalo de confianza. Los entrenamientos de CV se solapan: no se tratan folds como réplicas independientes.

El número final de características corresponde a las columnas que salen del preprocesamiento. El control Dummy ignora esas columnas y utiliza únicamente la media del target de train.

ΔMAE y ΔRMSE negativos significan mejora. La tabla principal compara con A; las ablaciones comparan B−A, C−B y D−C. Los tiempos suman ajuste y predicción, excluyen figuras y documentación y dependen del equipo.

Regla fijada antes de ejecutar: **Adoptar** provisionalmente una intervención si reduce MAE al menos 5 %, mejora en al menos 60 % de grupos, no empeora RMSE ni MAE al excluir familias. **Mantener en evaluación** si mejora MAE principal sin cumplir todos esos criterios. **Descartar** la intervención en esta comparación si no mejora MAE. Son umbrales operativos exploratorios, no pruebas de significación. Una conclusión de descarte no demuestra ausencia de efecto mecánico.

{caution}

## Reproducción

```powershell
python -m src.experimentos_fe
python -m src.reporte_fe --execute
python -m pytest tests -q
python -m src.verificar_fe
```

El primer comando ejecuta los experimentos y genera tablas, figuras y documentos. El segundo vuelve a ejecutar la comparación desde el notebook con kernel limpio y guarda los outputs. No modifica fuentes ni artefactos del EDA. Los tiempos pueden variar; las predicciones son deterministas con los datos y configuración congelados.
'''
    write(root / "docs/feature_engineering.md", methodology)
    original_units = ["conteo", "mm", "mm", "mm", "mm³", "mm", "mm", "conteo", "mm³", "mm³"]
    original_meanings = ["Celdas en dirección Y", "Altura de sección de fibra", "Ancho de sección de fibra", "Longitud sumada de fibras", "Volumen declarado de fibras", "Altura de pivote", "Radio de pivote", "Cantidad de pivotes", "Volumen declarado de pivotes", "Volumen CAD del espécimen"]
    originals = pd.DataFrame({"Variable": GEOMETRY, "Unidad": original_units, "Significado": original_meanings,
                              "Uso": ["Entrada; A–D" if v not in {"Fiber_total_length", "Pivot_total_number", "Fiber_total_volume", "Pivot_radius"}
                                      else "A–C; retirada de bloque original en D" if v != "Pivot_radius" else "Entrada conservada; filtro de varianza la retira del ajuste" for v in GEOMETRY]})
    features = pd.DataFrame([
        ["I_fibra_bh3_mm4", "Fiber_base=b; Fiber_height=h", "b*h³/12", "mm⁴; absoluta", "Segundo momento de área respecto al eje paralelo a b", "Representar dependencia cúbica de la sección", "Ingredientes imputados en train antes de fórmula", "Bajo en cálculo; hipótesis informada por EDA global", "B–D", decision_map["B"] + " (bloque de dos momentos)"],
        ["I_fibra_hb3_mm4", "Fiber_base=b; Fiber_height=h", "h*b³/12", "mm⁴; absoluta", "Segundo momento de área respecto al eje paralelo a h", "Representar la otra orientación de sección", "Ingredientes imputados en train antes de fórmula", "Bajo en cálculo; hipótesis informada por EDA global", "B–D", decision_map["B"] + " (bloque de dos momentos)"],
        ["log_volumen_pivotes_sobre_fibras", "Pivot_total_volume=Vp; Fiber_total_volume=Vf", "ln(Vp/Vf)", "1; adimensional", "Razón logarítmica entre volúmenes declarados", "Representar balance de componentes con función no lineal", "Ingredientes positivos, imputados solo en train", "No usa target; sesgo de hipótesis por EDA previo", "C–D", decision_map["C"] + " (incremento frente a B)"],
    ], columns=["Feature", "Variables de origen", "Fórmula", "Unidad", "Significado físico", "Hipótesis", "Tratamiento de nulos", "Riesgo de leakage", "Experimento", "Decisión"])
    write(out / "tables/diccionario_features.csv", features.to_csv(index=False, lineterminator="\n"))
    dictionary = f'''# Diccionario de características para la predicción

Amplía el [diccionario del EDA](diccionario_datos.md); no cambia sus definiciones. Todas las entradas se conocen antes del ensayo. `First_Failure_Load_N` es el OUTPUT objetivo, en N. `Case_n` es un identificador, no un predictor.

## Atributos originales admisibles

{md_table(originals)}

`Fiber_base` se incluye en A–D como decisión explícita frente a su omisión en el esquema general actual. En D, `Fiber_total_volume` sigue alimentando la razón logarítmica. No se sustituyen volúmenes declarados por cálculos ideales ni por la suma de partes.

## Atributos derivados evaluados

{md_table(features)}

Para los momentos, mm × mm³ = mm⁴. No se multiplican por un módulo elástico no disponible; no equivalen a rigidez EI ni permiten inferir directamente la falla. Para el logaritmo, mm³/mm³ = 1: el argumento es adimensional y estrictamente positivo. Ninguna fórmula usa fuerza, desplazamiento ni energía.

Los momentos reutilizan `geometry_features` del cuaderno 03. Se evalúan los dos ejes sin afirmar cuál domina en cada espécimen. El logaritmo se calcula como ln(Vp)−ln(Vf), numéricamente equivalente a ln(Vp/Vf) al usar las mismas unidades, para evitar desbordar el cociente.

## Propuestas que no se incorporan

Área de fibra y relación de aspecto ya se exploraron; no se agregan simultáneamente a los momentos para evitar proliferación de descriptores. Altura/diámetro de pivote replica aquí su altura porque el radio es constante. Una esbeltez de tramo requiere longitud libre y condiciones de unión verificadas, no longitud total de todas las fibras. Densidades volumétricas requieren volumen de envolvente y espesor global verificables; no se inventan. Volumen CAD/celdas Y no es volumen físico de una celda y se deja fuera de esta comparación. No se usan F/V, energías o desplazamientos como predictores.

StandardScaler crea coordenadas numéricas sin unidad para el ajuste, no nuevas mediciones físicas. Las fórmulas geométricas se calculan antes del escalamiento.
'''
    write(root / "docs/diccionario_features.md", dictionary)
    interpretations = []
    physical = {
        "B": "El cuaderno 03 evaluó b·h³/12 y h·b³/12, con asociaciones condicionadas próximas a cero. Se prueban porque la forma de la sección puede representar mecanismos de flexión, no porque su correlación prometa mejora. No se conoce aquí el módulo elástico ni se identifica el eje dominante. Añade dos columnas y hereda las discrepancias de Fiber_base.",
        "C": "El EDA encontró una asociación global positiva y condicionada negativa para ln(Vp/Vf). La hipótesis es representar de otra manera el reparto de volumen entre componentes. Añade una columna, pero Vf y Vp son complementarios en el diseño: la razón es una función no lineal de información ya disponible. La decisión depende de su diferencia de error frente a B.",
        "D": "El EDA mostró una suma constante de volúmenes y un valor de longitud y de número de pivotes por familia. Se retiran tres columnas para probar simplificación. Aunque estén determinadas por la arquitectura observada, longitud y cantidad de pivotes pueden proporcionar funciones no lineales útiles a una regresión lineal; retirarlas y redistribuir la penalización puede empeorar la aproximación. El resultado no demuestra que los descriptores sean física o estadísticamente inútiles.",
    }
    for row in decisions.itertuples():
        z = sensitivity[(sensitivity.experimento == row.experimento) & sensitivity.escenario.str.startswith("omitir_score_grupo_")]
        special = sensitivity[(sensitivity.experimento == row.experimento) & sensitivity.escenario.isin(["sin_scores_Cook_1_3", "sin_scores_revision_pivote"])]
        sign = "cambia" if z.delta_MAE_N.min() < 0 < z.delta_MAE_N.max() else "no cambia"
        fam = family[family.protocolo == "configuracion"].pivot(index="familia",columns="experimento",values="MAE_N")
        fam_deltas = "; ".join(f"Y={int(g)}: {fam.loc[g,row.experimento]-fam.loc[g,row.referencia]:+.4f} N" for g in fam.index)
        interpretations.append(f'''## {row.experimento} frente a {row.referencia}

**Intervención:** {row.intervencion}. Su evidencia y justificación se detallan en la matriz EDA → FE y en el diccionario.

{physical[row.experimento]}

El cambio de MAE es **{row.delta_MAE_N:+.4f} N** ({row.reduccion_MAE_pct:.2f} % de reducción) y el de RMSE **{row.delta_RMSE_N:+.4f} N**. Un RMSE menor indica menor penalización agregada de errores grandes, no que todos disminuyan. Mejora el MAE en **{100*row.fraccion_grupos_mejora:.1f} %** de grupos. Al excluir familias completas, el MAE cambia **{row.delta_MAE_familia_N:+.4f} N** frente a su referencia.

Dentro del protocolo principal, el cambio de MAE por familia es {fam_deltas}. Esto separa la mejora agregada de su distribución entre arquitecturas.

La mejora agrega o elimina funciones de las mismas mediciones: no aporta observaciones independientes. Los momentos y la razón heredan discrepancias de la fuente; D además cambia la distribución de la penalización Ridge. Se evalúa el bloque declarado, sin atribución individual a variables colineales.

Al omitir del agregado OOF un grupo cada vez, ΔMAE queda entre **{z.delta_MAE_N.min():+.4f} y {z.delta_MAE_N.max():+.4f} N**; su signo {sign}. Este diagnóstico cambia el score, no los modelos: los casos omitidos pudieron participar en otros entrenamientos. No reemplaza un estudio con nuevas mediciones.

{md_table(special[['escenario','delta_MAE_N']])}

**Decisión: {row.decision}.** Se aplica la regla declarada antes del experimento; el alcance es esta intervención con Ridge y estos protocolos, no una conclusión universal sobre la geometría.
''')
    captions = [
        ("FE01_error", "Figura 1. Error predictivo por representación", f"{headline} Las barras muestran MAE agrupado por caso; la dispersión entre grupos se presenta en la figura 5 y en metricas_folds.csv, no como intervalos de confianza."),
        ("FE02_observado_predicho", "Figura 2. Predicción fuera de muestra frente a observación", "Cada punto es un caso validado excluyendo su configuración nominal. La línea de identidad indica coincidencia perfecta; distancias verticales representan errores. La agrupación por fuerzas puede reflejar familias; no demuestra precisión en arquitecturas nuevas."),
        ("FE03_residuos", "Figura 3. Diagnóstico de residuales", "Residual = observado − predicho: positivo indica subestimación y negativo sobreestimación. Se identifican los dos errores absolutos mayores por variante para revisión, sin eliminarlos. La figura permite buscar sesgo y errores concentrados; no certifica normalidad ni independencia."),
        ("FE04_error_familia", "Figura 4. Error por familia y protocolo", f"La comparación distingue configuraciones ausentes y familias completas ausentes. Para {best}, MAE pasa de {primary.loc[best,'MAE_N']:.4f} N en el protocolo principal a {secondary.loc[best,'MAE_N']:.4f} N en el secundario. Son objetivos de generalización diferentes; la dispersión entre familias no es un intervalo de confianza."),
        ("FE05_ablaciones", "Figura 5. Contribución incremental de las intervenciones", "Cada punto es ΔMAE de un grupo bajo la misma partición; negativo favorece la intervención. El rombo muestra la media no ponderada de grupos, que puede diferir del ΔMAE ponderado por casos. La dispersión revela consistencia o heterogeneidad del aporte; no es importancia individual ni evidencia de información física nueva."),
    ]
    captions_text = "\n\n".join(f"### {title}\n\n![{title}](../reports/feature_engineering/figures/{name}.png)\n\n{text}\n\nFuente: tabla curada publicada; predicciones OOF en `reports/feature_engineering/tables/predicciones_oof.csv`. Fuerzas y errores en N." for name,title,text in captions)
    # El notebook reutiliza las mismas interpretaciones, sin enlaces relativos ambiguos.
    import json
    write(out / "leyendas.json", json.dumps([{"file": name, "title": title, "interpretation": text,
                                              "source": "Tabla curada publicada; predicciones fuera de muestra bajo los protocolos indicados. Fuerzas y errores en N."} for name,title,text in captions], ensure_ascii=False, indent=2))
    findings = f'''# Experimentos de ingeniería de características

{headline}

## Comparación principal: configuración nominal excluida

{comparison}

Tiempo en segundos de ajuste y predicción acumulados. ΔMAE y cambio porcentual se calculan respecto de A; negativos indican mejora. La decisión se refiere a la intervención frente a su referencia inmediata y no a una prueba de significación. [Resultados completos](../reports/feature_engineering/tables/comparacion_experimentos.csv).

## Prueba secundaria: familia completa excluida

{md_table(shown[shown.protocolo=='familia'][['experimento','MAE_N','RMSE_N','MedAE_N','R2_OOF','delta_MAE_vs_A_N','prediccion_min_N','predicciones_negativas']])}

El R² agregado puede ser negativo: indica mayor error cuadrático que la referencia de la media global de las observaciones usada en su definición. El Dummy se ajusta con la media de cada train y no coincide necesariamente con esa referencia. Los scores de familias extremas incorporan extrapolación. No se calcula R² por fold individual.

La reducción de descriptores D produce predicciones de fuerza negativas al excluir la familia Y=4 en esta ejecución. Son físicamente inadmisibles para la carga positiva estudiada y muestran un fallo de extrapolación del modelo lineal sin restricciones. Se conservan en las métricas, sin truncarlas, para no ocultar el resultado negativo.

## Ablaciones y decisiones

{md_table(decisions)}

{''.join(interpretations)}

## Errores por familia

{md_table(family)}

El sesgo se define como media de observado − predicho. Se muestran tamaños efectivos solo en esta tabla técnica; los grupos no son nuevos especímenes. Los casos de mayor error deben cotejarse con sus registros experimentales.

## Figuras e interpretación

{captions_text}

## Conclusión y siguiente experimento

{headline}

Se construyeron dos momentos de sección y una razón logarítmica; se probó también retirar un bloque redundante. Las decisiones por intervención son: {'; '.join(r.experimento+': '+r.decision for r in decisions.itertuples())}. **{best} es candidato para confirmación**, no un modelo validado para despliegue. Un resultado favorable solo en el dominio conocido no prueba transferencia a otra familia, material o campaña. Si la reducción falla, el resultado negativo delimita la utilidad de esta representación y regularización.

{caution}

Siguiente experimento propuesto, no ejecutado: congelar A y el candidato {best}, conciliar las geometrías señaladas con el proveedor y evaluar una campaña independiente con réplicas identificadas, manteniendo separados espécimen y lote. Registrar previamente MAE, límites de uso y regla de comparación. No generar datos sintéticos como sustituto de ensayos.
'''
    write(root / "docs/experimentos_fe.md", findings)
    readme = root / "README.md"
    content = readme.read_text(encoding="utf-8")
    start, end = "<!-- FEATURE_ENGINEERING_START -->", "<!-- FEATURE_ENGINEERING_END -->"
    compact = shown[(shown.protocolo == "configuracion") & (shown.experimento != "Dummy")][["experimento", "feature_set", "MAE_N", "RMSE_N", "delta_MAE_vs_A_N", "decision"]]
    section = f'''{start}
## Ingeniería de características y evaluación predictiva

**Entrega:** Ingeniería de características guiada por análisis exploratorio y evaluación experimental de su contribución a la predicción de la carga de primera falla en estructuras pantográficas.

La rama `feature-engineering` continúa el [EDA publicado](https://github.com/melissaaylasb-sudo/T02_MIA14_SeccionB/tree/eda). La pregunta es si transformar la geometría conocida antes del ensayo mejora la estimación de `First_Failure_Load_N` (N). Se conserva el EDA y se añade una comparación ejecutada, con Ridge fijo y las mismas particiones.

**Entradas:** geometría de fibras, pivotes y arquitectura. **Salida:** fuerza de primera rotura; las respuestas posteriores nunca entran al modelo. B añade dos momentos de sección, C añade ln(Vp/Vf) y D retira tres descriptores redundantes del bloque original.

{md_table(compact)}

ΔMAE negativo indica mejora frente a A. Las decisiones valoran cada intervención frente a su referencia inmediata: B−A, C−B y D−C.

**Validación:** exclusión completa de cada configuración nominal; prueba adicional excluyendo familias Y completas. Imputación, filtro de constantes y escalamiento se ajustan solo en entrenamiento. No hay tuning ni cambios de algoritmo entre variantes.

**Resultado:** {headline}

**Límites:** hipótesis informadas por el EDA del mismo conjunto, geometrías por conciliar y réplicas/lotes sin verificar. Las decisiones son exploratorias; el siguiente paso es confirmar A y {best} en una campaña independiente con protocolo congelado.

| Artefacto | Contenido |
|---|---|
| [04 · Notebook ejecutado](notebooks/04_Feature_Engineering_Pantograficas.ipynb) | Pipeline, experimentos, gráficos e interpretación |
| [Metodología](docs/feature_engineering.md) | Matriz EDA → FE, validación y control de fuga |
| [Diccionario de features](docs/diccionario_features.md) | Origen, fórmulas, unidades y decisiones |
| [Resultados y discusión](docs/experimentos_fe.md) | Ablaciones, errores y limitaciones |
| [Comparación CSV](reports/feature_engineering/tables/comparacion_experimentos.csv) | Métricas, tiempos y tamaños efectivos |
| [Predicciones OOF](reports/feature_engineering/tables/predicciones_oof.csv) | Cada predicción identificada por caso, variante y fold |
| [Configuración](config/feature_engineering.json) | Hipótesis, parámetros fijos y criterio de decisión |
| [Particiones](reports/feature_engineering/particiones.json) | Casos de entrenamiento y validación |
| [Auditoría de transformaciones](reports/feature_engineering/auditoria_transformaciones.json) | Estadísticas aprendidas en cada entrenamiento |
| [Código de características](src/feature_engineering.py) · [experimentos](src/experimentos_fe.py) | Implementación reutilizable |

```powershell
python -m src.reporte_fe --execute
python -m pytest tests -q
python -m src.verificar_fe
```

Los tres primeros notebooks conservan el EDA. El cuarto ejecuta esta etapa y genera `reports/feature_engineering/` (tablas, figuras, particiones, auditoría, manifiesto y logs) y los tres documentos FE en `docs/`.
{end}'''
    if start in content:
        a, tail = content.split(start, 1)
        _, b = tail.split(end, 1)
        content = a + section + b
    else:
        lines = content.split("\n", 1)
        content = "# Ingeniería de características de estructuras pantográficas\n\n" + section + "\n\n" + lines[1].lstrip()
        content = content.replace("Esta rama `eda` documenta", "La etapa anterior `eda` documenta")
        content = content.replace("todavía no evalúa un modelo predictivo.", "el modelo predictivo se evalúa en la nueva etapa descrita arriba.")
    # Mantener el índice y el árbol útiles en la nueva rama sin reescribir el EDA.
    content = content.replace("## Tres cuadernos para la entrega", "## Tres cuadernos de la etapa EDA")
    content = content.replace("[Tres cuadernos para la entrega](#tres-cuadernos-para-la-entrega)", "[Cuadernos del EDA](#tres-cuadernos-de-la-etapa-eda)")
    if "- [Ingeniería de características](#ingeniería-de-características-y-evaluación-predictiva)" not in content:
        content = content.replace("## Índice\n", "## Índice\n\n- [Ingeniería de características](#ingeniería-de-características-y-evaluación-predictiva)\n",1)
    if "│   └── 04_Feature_Engineering_Pantograficas.ipynb" not in content:
        content = content.replace("│   └── 03_Transformaciones_Exploratorias.ipynb", "│   ├── 03_Transformaciones_Exploratorias.ipynb\n│   └── 04_Feature_Engineering_Pantograficas.ipynb")
        content = content.replace("│   └── informe_eda.md                # Interpretación académica y RQ1–RQ12", "│   ├── informe_eda.md                # Interpretación académica y RQ1–RQ12\n│   ├── feature_engineering.md         # Matriz EDA → FE y protocolo de validación\n│   ├── diccionario_features.md        # Fórmulas, unidades y admisibilidad\n│   └── experimentos_fe.md             # Métricas, decisiones y discusión")
        content = content.replace("│   └── data_schema.yml              # Variables, unidades y reglas de curaduría", "│   ├── data_schema.yml              # Variables, unidades y reglas de curaduría\n│   └── feature_engineering.json       # Experimentos, parámetros y regla de decisión")
        content = content.replace("├── reports/\n", "├── reports/\n│   ├── feature_engineering/           # OOF, métricas, figuras, particiones, auditoría y logs\n",1)
        content = content.replace("│   └── reporte_eda.py               # Ejecución reproducible de cuadernos", "│   ├── reporte_eda.py               # Ejecución reproducible de cuadernos EDA\n│   ├── feature_engineering.py         # Contrato, fórmulas y pipelines\n│   ├── experimentos_fe.py             # Comparación A/B/C/D y exportación\n│   ├── informe_fe.py                  # Documentación ligada a los resultados\n│   ├── reporte_fe.py                  # Ejecución del notebook 04\n│   └── verificar_fe.py                # Comprobación independiente de la entrega")
        content = content.replace("│   └── test_transformaciones_eda.py", "│   ├── test_transformaciones_eda.py\n│   └── test_feature_engineering.py     # Fuga, grupos, fórmulas y dominios")
    content = content.replace("| `reports/` | Tablas, figuras y manifiestos del EDA | Evidencia exploratoria organizada por tema y vinculada a los notebooks |", "| `reports/` | Evidencia del EDA y evaluación FE | Resultados separados por etapa; FE añade predicciones OOF, particiones y auditorías |")
    write(readme, content)
