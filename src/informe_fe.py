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
    compact = primary.loc[list("ABCD"), ["MAE_N", "RMSE_N"]].reset_index()
    compact["Cambio"] = ["Geometría original", "A + dos momentos de sección", "B + ln(Vp/Vf)", "C − tres descriptores redundantes"]
    compact = compact[["experimento", "Cambio", "MAE_N", "RMSE_N"]].rename(columns={"experimento":"Experimento", "MAE_N":"MAE (N)", "RMSE_N":"RMSE (N)"})
    result = f"**{best} obtiene el menor MAE: {primary.loc[best,'MAE_N']:.4f} N**, una reducción de {improvement:.2f} % frente a A. Al excluir familias completas, su MAE es {secondary.loc[best,'MAE_N']:.4f} N; la mejora no demuestra transferencia a arquitecturas nuevas."
    brief = ("B: mantener en evaluación; C: descartar la razón añadida frente a B; D: descartar la reducción probada. "
             "Las decisiones corresponden a este modelo y protocolo.")
    # Los campos completos del diccionario siguen disponibles en CSV.
    features = pd.DataFrame([
        ["I_fibra_bh3_mm4", "Fiber_base=b; Fiber_height=h", "b*h³/12", "mm⁴", "Segundo momento de área: eje paralelo a b", "Representar la forma de sección", "Mediana train antes de fórmula", "No usa target; hipótesis del EDA global", "B–D", verdict["B"]],
        ["I_fibra_hb3_mm4", "Fiber_base=b; Fiber_height=h", "h*b³/12", "mm⁴", "Segundo momento de área: eje paralelo a h", "Representar la otra orientación", "Mediana train antes de fórmula", "No usa target; hipótesis del EDA global", "B–D", verdict["B"]],
        ["log_volumen_pivotes_sobre_fibras", "Pivot_total_volume=Vp; Fiber_total_volume=Vf", "ln(Vp/Vf)", "1", "Razón entre volúmenes declarados", "Representar balance de componentes", "Mediana train; ingredientes positivos", "No usa target; hipótesis del EDA global", "C–D", verdict["C"]],
    ], columns=["Feature", "Variables de origen", "Fórmula", "Unidad", "Significado físico", "Hipótesis", "Tratamiento de nulos", "Riesgo de leakage", "Experimento", "Decisión"])
    write(out / "tables/diccionario_features.csv", features.to_csv(index=False, lineterminator="\n"))
    write(root / "docs/diccionario_features.md", f"""# Diccionario de características

[Notebook principal](../notebooks/04_Feature_Engineering_Pantograficas.ipynb) · [Variables originales, INPUT y OUTPUT](diccionario_datos.md#consulta-rápida) · [Diccionario completo CSV](../reports/feature_engineering/tables/diccionario_features.csv)

{md_table(features[['Feature','Variables de origen','Fórmula','Unidad','Significado físico']])}

Los momentos se evalúan juntos en B; son geometría de sección, no rigidez EI medida. La razón C es adimensional y exige volúmenes positivos. Todas las fórmulas usan entradas previas al ensayo, después de la imputación ajustada en entrenamiento.

`Fiber_base` se incluye en todas las variantes aunque el esquema general actual la omita. El radio constante se retira del ajuste dentro de cada fold. D retira `Fiber_total_length`, `Pivot_total_number` y `Fiber_total_volume` del bloque original; este último sigue alimentando la razón. El target es `First_Failure_Load_N` (N); identificadores y otras respuestas no son predictores.
""")
    write(root / "docs/feature_engineering.md", """# Protocolo de ingeniería de características

**Entrega principal:** [notebook 04 ejecutado](../notebooks/04_Feature_Engineering_Pantograficas.ipynb). [Matriz EDA → FE completa](../reports/feature_engineering/tables/matriz_eda_fe.csv).

| Elemento | Decisión |
|---|---|
| Fuente | Tabla curada `model_table.csv`, contrastada con su manifiesto; target observado, sin imputación. |
| Objetivo | `First_Failure_Load_N` (N): fuerza de primera rotura de fibra o pivote. |
| Entradas | Geometría previa al ensayo; se excluyen identificador, fuerzas, desplazamientos y energías. |
| A / B / C / D | Originales / añadir momentos / añadir log de razón / retirar bloque redundante. |
| Modelo | Ridge, alpha=1, solver SVD; mismos parámetros y particiones. Dummy de media como control. |
| Pipeline | Contrato → mediana → fórmulas → filtro de varianza → StandardScaler → Ridge; ajuste solo en train. |
| Validación principal | LeaveOneGroupOut por (`n_cells_Y`, `Fiber_height`, `Pivot_height`); factores nominales repetidos permanecen juntos. |
| Comprobación adicional | Excluir cada familia Y completa; evalúa una arquitectura no vista. |
| Métricas | MAE principal; RMSE, MedAE y R² agregado OOF. Sin R² en folds individuales. |
| Selección | Retirada fija por hipótesis geométrica; varianza nula aprendida en train. Sin selección supervisada ni tuning. |
| Decisión | Adoptar provisionalmente si MAE mejora ≥5 %, gana en ≥60 % de grupos y no empeora RMSE ni MAE por familia; si solo mejora MAE, mantener en evaluación; si no, descartar. |

Las hipótesis proceden del EDA del mismo conjunto: los resultados son exploratorios y pueden tener sesgo de selección. Faltan réplicas/lotes verificados y persisten discrepancias geométricas. Las familias no representan campañas independientes.

Las nuevas features son funciones de mediciones existentes. Retirar columnas modifica también la penalización efectiva de Ridge. No se recortan predicciones negativas ni se eliminan extremos. La sensibilidad por omisión modifica el agregado de errores OOF, no reentrena modelos.

Los predictores son numéricos: no se necesita ColumnTransformer ni codificación categórica. Se difieren RobustScaler, selección por información mutua y nuevos descriptores para no mezclar intervenciones; la evidencia y las exclusiones constan en la matriz CSV.

[Configuración](../config/feature_engineering.json) · [Particiones](../reports/feature_engineering/particiones.json) · [Estadísticas aprendidas](../reports/feature_engineering/auditoria_transformaciones.json)
""")
    pairwise = decisions[["experimento", "referencia", "delta_MAE_N", "fraccion_grupos_mejora", "decision"]]
    write(root / "docs/experimentos_fe.md", f"""# Resultados de ingeniería de características

[Ver el desarrollo, las figuras y la interpretación en el notebook](../notebooks/04_Feature_Engineering_Pantograficas.ipynb).

{md_table(compact)}

{result}

## Intervenciones

{md_table(pairwise)}

ΔMAE negativo indica mejora frente a la referencia de cada fila. {brief}

La variante D produce fuerzas negativas al excluir Y=4: un fallo físico de extrapolación, conservado en las métricas. La comprobación de sensibilidad omite scores del agregado OOF; no elimina su influencia en los modelos entrenados.

## Alcance

El EDA ya utilizó todo el conjunto. Estos resultados no son validación externa ni garantizan seguridad estructural. El siguiente experimento es confirmar A y {best} con geometrías conciliadas y ensayos independientes, manteniendo el protocolo congelado.

[Resultados completos y tiempos](../reports/feature_engineering/tables/comparacion_experimentos.csv) · [Predicciones OOF](../reports/feature_engineering/tables/predicciones_oof.csv) · [Errores por familia](../reports/feature_engineering/tables/errores_familia.csv) · [Sensibilidad](../reports/feature_engineering/tables/sensibilidad_scores.csv)
""")
    captions = [
        ("FE01_error", "Figura 1. Error por representación", "Menor MAE significa menor error medio. La dispersión entre grupos se muestra en la figura 5; estas barras no son intervalos de confianza."),
        ("FE02_observado_predicho", "Figura 2. Observado frente a predicho", "La diagonal representa coincidencia perfecta. Cada predicción procede de un modelo que no entrenó con esa configuración nominal."),
        ("FE03_residuos", "Figura 3. Residuales", "Residual = observado − predicho: positivo indica subestimación. Los casos señalados tienen los mayores errores absolutos; no se eliminan."),
        ("FE04_error_familia", "Figura 4. Error por familia", "En la validación por configuración, B mejora frente a A en Y=5 y Y=6, pero empeora en Y=4. Excluir una familia completa aumenta mucho el error: generalizar a otra arquitectura sigue sin resolverse."),
        ("FE05_ablaciones", "Figura 5. Aporte de cada intervención", "Cada punto es el cambio de MAE de un grupo; negativo indica mejora. El rombo es la media de grupos, no ponderada por casos. No mide importancia causal individual."),
    ]
    import json
    write(out / "leyendas.json", json.dumps([{"file":name,"title":title,"interpretation":text,"source":"Tabla curada y predicciones OOF; fuerzas y errores en N."} for name,title,text in captions], ensure_ascii=False, indent=2))
    readme = f"""# Ingeniería de características de estructuras pantográficas

**[Abrir notebook 04 ejecutado →](notebooks/04_Feature_Engineering_Pantograficas.ipynb)**

Melissa Dessire Aylas Barranca · UNI, Maestría en Inteligencia Artificial · Trabajo de Investigación II.

Se evalúa si transformar la geometría mejora la predicción de **`First_Failure_Load_N`**: fuerza, en newtons, de la primera rotura de una barra o unión de la malla.

## Resultado

{md_table(compact)}

{result} **B se mantiene en evaluación**; C no mejora B y D empeora. El EDA previo usó el mismo conjunto: la evidencia es exploratoria.

## Qué encontrarás en el notebook

Hipótesis del EDA → fórmulas → pipeline → comparación A/B/C/D → cinco gráficos → interpretación. Ridge y particiones se mantienen fijos; las transformaciones se ajustan solo en entrenamiento. Se excluyen configuraciones completas y, en una prueba adicional, familias completas.

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
  informe_fe.py            # Resúmenes ligados a resultados
  reporte_fe.py            # Ejecución del notebook 04
  verificar_fe.py          # Integridad y coherencia de la entrega
config/feature_engineering.json  # Parámetros e intervenciones
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

[Diccionario de variables](docs/diccionario_datos.md#consulta-rápida) · [Nuevas features](docs/diccionario_features.md) · [Protocolo](docs/feature_engineering.md) · [Tabla de resultados](reports/feature_engineering/tables/comparacion_experimentos.csv) · [EDA anterior](https://github.com/melissaaylasb-sudo/T02_MIA14_SeccionB/tree/eda)

```powershell
python -m src.reporte_fe --execute
python -m pytest tests -q
python -m src.verificar_fe
```

Los datos originales y los tres cuadernos del EDA se conservan. Los PDF y las referencias privadas no se publican. Siguiente paso: confirmar A y {best} con ensayos independientes y geometrías conciliadas.
"""
    write(root / "README.md", readme)
