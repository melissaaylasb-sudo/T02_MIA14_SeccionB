# Operacionalización de variables

La [matriz de consistencia](matriz_consistencia.md) conserva los objetivos e hipótesis del informe previo. Este cuadro distingue variables físicas, representaciones derivadas e indicadores analíticos. Todos los predictores proceden de información geométrica anterior al ensayo; el objetivo procede de la medición de primera falla. Los nombres, unidades y restricciones se contrastan con el [diccionario científico](diccionario_datos.md), el [diccionario FE](diccionario_features.md) y la [tabla curada](../data/processed/preliminary/model_table.csv).

| Variable | Definición conceptual | Definición operacional | Indicador | Unidad | Fuente | Método |
|---|---|---|---|---|---|---|
| `n_cells_Y` | Arquitectura de celdas en la dirección Y. | Conteo declarado en la geometría del espécimen; también define familias 4, 5 y 6. | Número de celdas. | conteo | Archivo MATLAB y tabla curada. | Lectura directa previa al ensayo; agrupar sin codificar el target. |
| `Fiber_height` | Altura de sección de fibra. | Altura registrada para el diseño. | Altura de fibra. | mm | Archivo MATLAB y tabla curada. | Lectura directa; imputación contingente en entrenamiento. |
| `Fiber_base` | Ancho de sección de fibra. | Base registrada para el diseño. | Ancho de fibra. | mm | Archivo MATLAB y tabla curada. | Lectura directa; insumo de ambos momentos. |
| `Fiber_total_length` | Longitud total declarada del conjunto de fibras. | Valor registrado para cada configuración. | Longitud total de fibras. | mm | Archivo MATLAB y tabla curada. | Lectura directa; revisar dependencia con arquitectura. |
| `Fiber_total_volume` | Volumen declarado del conjunto de fibras. | Valor geométrico registrado, que puede diferir de la suma de otros componentes. | Volumen de fibras. | mm³ | Archivo MATLAB y tabla curada. | Lectura directa; revisar dependencia algebraica. |
| `Pivot_height` | Altura de conectores entre fibras. | Altura registrada del pivote. | Altura de pivote. | mm | Archivo MATLAB y tabla curada. | Lectura directa; no inferir mecanismo causal por asociación. |
| `Pivot_radius` | Radio de la sección circular del pivote. | Radio registrado; 0.5 mm en todos los casos observados. | Radio de pivote. | mm | Archivo MATLAB y tabla curada. | Lectura directa; filtro de varianza nula ajustado en cada entrenamiento. |
| `Pivot_total_number` | Número de conectores de la estructura. | Conteo total declarado para la configuración. | Número de pivotes. | conteo | Archivo MATLAB y tabla curada. | Lectura directa; revisar redundancia con familia Y. |
| `Pivot_total_volume` | Volumen declarado del conjunto de pivotes. | Valor geométrico registrado. | Volumen de pivotes. | mm³ | Archivo MATLAB y tabla curada. | Lectura directa; insumo de la razón C. |
| `Sample_total_volume` | Volumen CAD global del espécimen. | Valor global registrado, conservado separado de los volúmenes de componentes. | Volumen CAD del espécimen. | mm³ | Archivo MATLAB y tabla curada. | Lectura directa; auditar discrepancias geométricas. |
| `I_fibra_bh3_mm4` | Segundo momento de área ideal de la sección de fibra para una orientación. | b·h³/12 con b=`Fiber_base` y h=`Fiber_height`, después de imputación aprendida en entrenamiento. | Momento de sección 1. | mm⁴ | Derivada de dos INPUT de la tabla curada. | Transformación determinista dentro del pipeline; bloque B. |
| `I_fibra_hb3_mm4` | Segundo momento de área ideal para la orientación ortogonal. | h·b³/12 con b=`Fiber_base` y h=`Fiber_height`, después de imputación aprendida en entrenamiento. | Momento de sección 2. | mm⁴ | Derivada de dos INPUT de la tabla curada. | Transformación determinista dentro del pipeline; bloque B. |
| `log_volumen_pivotes_sobre_fibras` | Balance relativo de volúmenes declarados de pivotes y fibras. | ln(`Pivot_total_volume` / `Fiber_total_volume`) para volúmenes positivos. | Logaritmo de la razón de volúmenes. | adimensional | Derivada de dos INPUT de la tabla curada. | Transformación determinista dentro del pipeline; bloque C, excluido del conjunto B provisional. |
| `First_Failure_Load_N` | Fuerza experimental asociada a la primera rotura identificada de fibra o pivote. | Valor del resumen de ensayo que enlaza con el caso; las ausencias del target no se imputan. | Carga de primera falla. | N | Archivo MATLAB y tabla curada. | Medición/resumen experimental; respuesta continua de regresión. |

**Representaciones evaluadas.** A contiene las diez entradas originales; B añade conjuntamente los dos momentos; C añade a B la razón logarítmica. El radio constante se retira de la matriz efectiva con un filtro ajustado solo en entrenamiento. B fue la mejor de A–D por MAE en la comparación Ridge previa (**12.8346 N**), pero su mejora frente A fue apenas **3.11 %** y el MAE al excluir familias completas no mejoró. Su adopción es provisional, sujeta a confirmación independiente ([resultados FE](experimentos_fe.md)). Las características derivadas no son nuevas mediciones.

## Indicadores analíticos, separados de la variable respuesta

| Indicador | Definición operacional | Unidad | Uso y límite |
|---|---|---|---|
| MAE fuera de entrenamiento | Media de `|y observado − y predicho|` sobre predicciones de grupos reservados. | N | Criterio principal; comparar de forma emparejada entre E1/E2/E3/E4. |
| RMSE | Raíz de la media de errores cuadrados. | N | Sensible a errores grandes. |
| MedAE | Mediana de errores absolutos. | N | Describe error central. |
| R² OOF | `1 − Σe²/Σ(y − media y)²` sobre predicciones agrupadas fuera de entrenamiento. | adimensional | Puede ser negativo; no es exactitud porcentual. |
| Sesgo medio | Media de `predicción − observación`. | N | Dirección promedio del error. |
| SHAP | Contribución de cada atributo a la predicción respecto de un valor base del explicador. | N si explica la salida de fuerza sin transformación | Atribución del modelo; no efecto causal físico. El background de cada fold debe provenir de su entrenamiento. |
| Permutación por MAE | MAE tras permutar una entrada en validación menos MAE original. | N | Pérdida predictiva al romper una relación; puede subestimar atributos correlacionados. |
| Ablación | Cambio de MAE al reentrenar y evaluar sin una característica o familia de atributos. | N | Evalúa utilidad del bloque bajo el protocolo; no estima efectos físicos de intervención. |

## Frontera predictiva y limitaciones de medición

`Case_n` es identificador y queda fuera de las entradas. `Ultimate_Load`, `Maximum_Displacement`, desplazamientos residuales, energías y número de fracturas observadas pertenecen al ensayo o a etapas posteriores. Usarlos para predecir antes de ensayar sería fuga temporal ([frontera predictiva](../reports/eda_transformaciones/tables/T06_frontera_predictiva.csv)).

El archivo digital presenta discrepancias con la tabla general y algunas fichas individuales, especialmente para medidas geométricas; no se corrigen sin conciliación documental ([trazabilidad](trazabilidad_datos.md)). Los momentos describen geometría de sección ideal, no rigidez `EI`, tensión ni criterio de fractura. La campaña no identifica material, lote o réplica física verificada por cada fila para modelar esas fuentes de variación.
