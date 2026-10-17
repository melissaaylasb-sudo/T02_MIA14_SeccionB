# Ingeniería de características de estructuras pantográficas

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

| Experimento | Cambio | MAE (N) | RMSE (N) | MedAE (N) | R² OOF | Tiempo CV (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Línea base (A) | Geometría original | 13.2466 | 16.3457 | 11.9409 | 0.9226 | 0.2901 |
| Variante 1 (B) | A + dos momentos de sección | 12.8346 | 16.2029 | 11.7801 | 0.9239 | 0.6825 |
| Variante 2 (C) | B + ln(Vp/Vf) | 12.9139 | 16.3271 | 11.8643 | 0.9228 | 0.5711 |
| Variante 3 (D) | C − tres descriptores redundantes | 17.0350 | 21.1361 | 14.9109 | 0.8706 | 0.5131 |

**B obtiene el menor MAE: 12.8346 N**, una reducción de 3.11 % frente a A. Al excluir familias completas, su MAE es 57.6538 N; la mejora no demuestra transferencia a arquitecturas nuevas. **B se mantiene en evaluación**; C no mejora B y D empeora. El EDA previo usó el mismo conjunto: la evidencia es exploratoria.

## Cinco modelos de regresión

| Modelo de regresión | Motivo de inclusión |
| --- | --- |
| Ridge | Referencia lineal regularizada frente a entradas correlacionadas. |
| Elastic Net | Contrasta otra regularización lineal; puede reducir algunos coeficientes a cero. |
| SVR con núcleo RBF | Estudia relaciones no lineales entre geometría y fuerza. |
| Bosque aleatorio (Random Forest) | Combina árboles para representar interacciones y umbrales geométricos. |
| Proceso gaussiano | Contrasta una función no lineal mediante similitud entre geometrías. |

La comparación cambia el algoritmo y conserva las entradas A para distinguir su efecto del tratamiento de características.

| Modelo | MAE (N) | RMSE (N) | MedAE (N) | R² OOF | Tiempo CV (s) |
| --- | --- | --- | --- | --- | --- |
| Ridge | 13.2466 | 16.3457 | 11.9409 | 0.9226 | 0.2716 |
| Elastic Net | 13.9792 | 17.1004 | 13.7200 | 0.9153 | 0.3525 |
| SVR con núcleo RBF | 15.6507 | 19.0958 | 12.6334 | 0.8943 | 0.3309 |
| Bosque aleatorio | 13.2325 | 16.3415 | 9.6994 | 0.9226 | 4.1263 |
| Proceso gaussiano | 16.0577 | 19.3414 | 14.1937 | 0.8916 | 0.2480 |

**Bosque aleatorio presenta el menor MAE por configuración: 13.2325 N.** La diferencia respecto a Ridge es 0.0141 N. Al excluir familias completas, su MAE es 60.9006 N. La comparación es exploratoria: no constituye una selección validada en ensayos externos.

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

| Métrica | Interpretación |
| --- | --- |
| MAE [N] · principal | Promedio del error absoluto; menor es mejor. |
| RMSE [N] | Penaliza más los errores grandes; menor es mejor. |
| MedAE [N] | Mediana del error absoluto; resume el error central con menor sensibilidad a extremos. |
| R² OOF | Se calcula sobre todas las predicciones fuera de muestra. 1 indica ajuste perfecto; negativo indica mayor error cuadrático que la media global observada como referencia. |
| Tiempo CV [s] | Suma medida de ajuste y predicción de las particiones; excluye figuras y archivos. Depende del equipo, no es latencia de despliegue. |

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

Los datos originales y los tres cuadernos del EDA se conservan. Los PDF y las referencias privadas no se publican. Siguiente paso: confirmar A y B con ensayos independientes y geometrías conciliadas.
