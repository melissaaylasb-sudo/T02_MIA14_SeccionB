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
| Comparación | Línea base A y tres variantes B/C/D con Ridge y parámetros fijos. Cada intervención se contrasta con la anterior; no hay búsqueda de hiperparámetros en esta etapa. |
| Modelos usados | Regresión Ridge en A–D; control auxiliar DummyRegressor que predice la media del entrenamiento. A–D son conjuntos de características, no cuatro algoritmos distintos. |

## Resultado

| Experimento | Cambio | MAE (N) | RMSE (N) | MedAE (N) | R² OOF | Tiempo CV (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Línea base (A) | Geometría original | 13.2466 | 16.3457 | 11.9409 | 0.9226 | 0.2423 |
| Variante 1 (B) | A + dos momentos de sección | 12.8346 | 16.2029 | 11.7801 | 0.9239 | 0.4573 |
| Variante 2 (C) | B + ln(Vp/Vf) | 12.9139 | 16.3271 | 11.8643 | 0.9228 | 0.4931 |
| Variante 3 (D) | C − tres descriptores redundantes | 17.0350 | 21.1361 | 14.9109 | 0.8706 | 0.5200 |

**Tiempo CV:** suma medida de ajuste y predicción de todos los folds de la validación por configuración; excluye figuras y escritura de archivos. Depende del equipo y no representa la latencia de una aplicación.

**B obtiene el menor MAE: 12.8346 N**, una reducción de 3.11 % frente a A. Al excluir familias completas, su MAE es 57.6538 N; la mejora no demuestra transferencia a arquitecturas nuevas. **B se mantiene en evaluación**; C no mejora B y D empeora. El EDA previo usó el mismo conjunto: la evidencia es exploratoria.

## Qué encontrarás en el notebook

| Sección | Evidencia ejecutada |
|---|---|
| 4.1–4.2 · Datos y construcción | Entradas autorizadas, hipótesis del EDA y cálculo de las nuevas variables con un caso real. |
| 4.3–4.4 · Tratamiento visual | Sección y momentos de área, razón antes/después del logaritmo, redundancias, escalamiento, mapa de variables y rango. |
| 4.5 · Auditoría de una predicción | Separación de grupos, estadísticas aprendidas solo en entrenamiento y transformación de un caso reservado. |
| 4.6–4.8 · Evaluación | Predicciones fuera de muestra, residuales, errores por familia, ablaciones y sensibilidad. |
| 4.9–4.10 · Conclusiones y verificación | Reglas de decisión contrastadas con resultados e integridad de los artefactos. |

**Métricas:** MAE [N], error absoluto medio principal; RMSE [N], más sensible a errores grandes; MedAE [N], error absoluto mediano; R², ajuste agregado de las predicciones fuera de muestra. Se evalúan configuraciones nominales completas y, por separado, familias Y completas. Cada predicción de validación procede de un ajuste que excluyó su grupo.

## Técnicas y validación

El pipeline aplica **imputación por mediana → construcción de variables → filtro de varianza nula → estandarización → regresión Ridge**. La mediana es una contingencia: las entradas geométricas actuales están completas. El radio constante se retira en cada entrenamiento. D retira longitud total de fibras, cantidad de pivotes y volumen de fibras; este último sigue interviniendo en la razón logarítmica.

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
  informe_fe.py            # Resúmenes ligados a resultados
  presentacion_fe.py       # Nombres y etiquetas en español
  tratamiento_fe.py        # Figuras y comprobaciones del tratamiento geométrico
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

[Variables en español](docs/diccionario_features.md) · [Protocolo](docs/feature_engineering.md) · [Tabla de resultados](reports/feature_engineering/tables/comparacion_experimentos.csv) · [Registro de ejecución](reports/feature_engineering/ejecucion.log) · [EDA anterior](https://github.com/melissaaylasb-sudo/T02_MIA14_SeccionB/tree/eda)

```powershell
python -m src.reporte_fe --execute
python -m pytest tests -q
python -m src.verificar_fe
```

Los datos originales y los tres cuadernos del EDA se conservan. Los PDF y las referencias privadas no se publican. Siguiente paso: confirmar A y B con ensayos independientes y geometrías conciliadas.
