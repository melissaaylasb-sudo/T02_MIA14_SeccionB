# Importancia de variables: SHAP, permutación y ablación

## Pregunta

¿Qué entradas utiliza el modelo para predecir la carga de primera falla y cuánto se deteriora la predicción al perturbar o retirar información? SHAP, permutación y ablación responden preguntas distintas. La concordancia apoya una lectura predictiva; la discrepancia exige revisar correlación, escala, interacciones y estabilidad.

| Método | Cantidad evaluada | Datos que puede utilizar | Interpretación y límite |
|---|---|---|---|
| SHAP global | Media de `|SHAP|` de predicciones OOF, en N. | Modelo del fold; background tomado de su train; observación de validación. | Magnitud atribuida por el modelo bajo un supuesto de dependencia; no causalidad. |
| Permutación individual | `MAE_permutado − MAE_original`, en N. | Observaciones de validación del fold y modelo ya ajustado; repeticiones con semilla. | Pérdida al romper una columna; la redundancia puede ocultar utilidad compartida. Valores negativos son posibles. |
| Permutación de grupo | La misma diferencia al permutar conjuntamente una familia de columnas. | Validación y modelo del fold; agrupaciones predefinidas. | Evalúa dependencia del conjunto bajo una perturbación que puede generar combinaciones no observadas. |
| Ablación | `MAE_sin_bloque − MAE_completo`, en N, con nuevo fit en los mismos splits. | Solo entrenamiento para fit/tuning; validación externa para métricas. | Utilidad del bloque bajo reentrenamiento; puede incluir compensación de otras variables. |

## Cruce cuantitativo: Ridge E3, configuración reservada

La correlación es Spearman global con `First_Failure_Load_N` del [EDA](../reports/eda_relaciones/tables/04_input_output_asociaciones.csv), o de las [derivadas exploratorias](../reports/eda_transformaciones/tables/T08_asociaciones_caracteristicas.csv). El coeficiente es la **media entre 24 folds del Ridge E3 por una desviación estándar de entrada** ([coeficientes](../reports/feature_importance/tables/coeficientes_folds.csv)); no representa una pendiente física aislada. SHAP es `mean(|SHAP|)` OOF en N ([Ridge y Elastic Net](../reports/feature_importance/tables/shap_global.csv)). La permutación de fuentes originales se calculó para Ridge E3 con 12 repeticiones ([tabla](../reports/feature_importance/tables/permutacion.csv)); las columnas derivadas se recalculan al alterar sus fuentes. Las cifras se redondean.

| Feature | Correlación EDA ρ | Coef. Ridge E3 [N/z] | SHAP Ridge E3 / Elastic Net S3 [N] | Permutación Ridge E3, media ± SD [N] | Ablación | Estabilidad SHAP E3 | Interpretación mecánica plausible y límite |
|---|---:|---:|---:|---:|---|---|---|
| Número de pivotes `Pivot_total_number` | +0.931 | +87.43 | 102.89 / 66.31 | 100.74 ± 20.80 | No individual; comparte bloque de arquitectura. | Top 3 en 70.8 % de folds; SD de magnitud 92.73 N. | Conectividad de malla; coincide con familia Y y longitud, sin efecto aislado identificado. |
| Número de celdas Y `n_cells_Y` | +0.931 | −26.43 | 62.91 / 23.86 | 60.51 ± 13.39 | No individual. | Top 3 en 29.2 %; SD 72.42 N. | Arquitectura interna; coeficiente contrario a correlación global señala ajuste condicionado y colinealidad. |
| Longitud total de fibras `Fiber_total_length` | +0.931 | +37.15 | 34.04 / 30.69 | 34.11 ± 7.12 | No individual. | Top 3 en 58.3 %; SD 23.18 N. | Cantidad y distribución de fibras ligadas a arquitectura Y. |
| Volumen CAD global `Sample_total_volume` | +0.799 | −28.73 | 23.68 / 27.59 | 20.04 ± 3.39 | No individual. | Top 3 en 45.8 %; SD 14.23 N. | Escala geométrica CAD; signo condicional contrario a asociación marginal, con discrepancias documentales. |
| Altura de fibra `Fiber_height` | 0.000 | −19.26 | 16.99 / 17.41 | 3.22 ± 1.97 | Fuente de momentos. | Top 3 en 25.0 %; SD 13.94 N. | Dimensión de sección que varía junto a ancho y momentos; baja permutación individual no prueba irrelevancia. |
| Ancho de fibra `Fiber_base` | −0.552 | −20.17 | 16.46 / 17.45 | 7.09 ± 2.25 | Fuente de momentos. | Top 3 en 29.2 %; SD 19.91 N. | Ancho de sección; combinación con altura define ambos momentos. |
| Momento `I_fibra_bh3_mm4` | −0.218 | +14.32 | 13.41 / 13.77 | No individual; permutación conjunta de altura+ancho: 3.39 ± 1.75 N. | Bloque de dos momentos: retirar de B ajustado **mejora** MAE Ridge 1.2744 N. | Top 3 en 8.3 %; SD 9.84 N. | Geometría ideal de sección, no rigidez EI ni momento aplicado. |
| Momento `I_fibra_hb3_mm4` | −0.624 | +22.61 | 18.07 / 19.06 | Misma permutación conjunta de fuentes; no atribuible a un momento. | Misma ablación conjunta; no separa dos momentos. | Top 3 en 33.3 %; SD 24.49 N. | Segunda orientación ideal; gran aporte en algunos casos se cancela con fuentes correlacionadas. |
| Volumen de fibras `Fiber_total_volume` | −0.828 | +5.49 | 5.58 / 4.91 | 2.35 ± 1.08 | No individual. | Top 3 en 0 %; SD 3.61 N. | Volumen declarado, complementario al de pivotes en esta muestra. |
| Volumen de pivotes `Pivot_total_volume` | +0.828 | −5.49 | 5.58 / 4.55 | 2.47 ± 1.15 | No individual. | Top 3 en 0 %; SD 3.61 N. | Volumen de conectores; el signo condicionado no se interpreta por separado. |
| Altura de pivote `Pivot_height` | −0.148 | −0.65 | 1.20 / 1.25 | −0.05 ± 0.19 | No individual. | Top 3 en 0 %; SD 1.24 N. | Puede interactuar con arquitectura; esta muestra no da evidencia estable de uso global en E3. |

`Pivot_radius` es constante (0.5 mm) y el pipeline lo elimina mediante el filtro de varianza nula ajustado en train. Su permutación produce **0 N** porque no cambia los valores; esto no estima la sensibilidad de un radio no observado. La suma `Fiber_total_volume + Pivot_total_volume = 6524.01 mm³` y las dependencias de conteos/longitud con Y se documentaron en [EDA→FE](../reports/feature_engineering/tables/matriz_eda_fe.csv). Las correlaciones globales altas de arquitectura y los coeficientes condicionados discordantes muestran por qué las técnicas no deben tratarse como votos de una misma propiedad.

## Ablaciones y familias geométricas

La [ablación ejecutada](../reports/feature_importance/tables/ablaciones.csv) informa `MAE_sin_momentos − MAE_con_momentos`: un valor positivo favorece conservar el bloque. Al excluir configuraciones, es **+0.4119 N** para Ridge fijo y **+0.5375 N** para Elastic Net fijo; con tuning cambia a **−1.2744 N** en Ridge y **−0.6920 N** en Elastic Net. Por tanto, los momentos aportan una mejora pequeña bajo parámetros fijos, pero el conjunto A ajustado tiene menor MAE en ambas familias. Al excluir familias Y, las ablaciones cambian de signo según modelo y tuning; no hay apoyo estable para declarar B superior en arquitecturas no vistas.

La [síntesis por bloques](../reports/feature_importance/tables/shap_familias.csv) usa la media de `|suma de SHAP firmados del bloque|`: arquitectura **85.25 N** en Ridge E3 y **88.13 N** en Elastic Net S3, sección de fibra **9.38 N** y **9.62 N**. Es una agregación descriptiva de atribuciones ya calculadas, **no** un SHAP grupal formal. La permutación conjunta de fuentes de sección dio **3.39 ± 1.75 N** en Ridge E3; las cifras distintas son compatibles porque la atribución dentro del modelo y la pérdida al alterar datos miden propiedades diferentes.

## Decisiones provisionales sobre características

| Características | Decisión basada en esta evidencia | Condición para una decisión firme |
|---|---|---|
| Geometría original A | Conservar como referencia y entrada del Ridge E4, menor MAE de configuración (9.7070 N). | Confirmar en nuevos ensayos y geometrías conciliadas. |
| Dos momentos de sección B | Mantener como hipótesis de representación y comparación, **sin** adoptarlos como mejora general de A. | Variación controlada de sección y evaluación externa predefinida; ablaciones más estables. |
| Razón logarítmica C | No incorporarla al conjunto B de este sprint: el contraste previo C−B empeoró 0.0792 N por configuración. | Nueva hipótesis predefinida o datos adicionales. |
| Descriptores de arquitectura (`n_cells_Y`, pivotes, longitud) | Conservar la información para predicción dentro del dominio observado; interpretar en bloque. | Diseños que varíen cada descriptor con menor dependencia entre sí. |
| Volúmenes de componentes | Conservar en la comparación común y reconocer redundancia; no declarar irrelevante uno por importancia baja. | Ablación de bloque en datos independientes o mediciones geométricas conciliadas. |
| Radio de pivote | Mantener en el contrato de datos, con exclusión estadística por constancia en cada entrenamiento. | Experimentos con radios distintos para estimar sensibilidad. |

## Lectura final y límites

El número de pivotes y otros descriptores de arquitectura sobresalen tanto en SHAP como en permutación de Ridge E3. Sin embargo, se correlacionan estrechamente y sus atribuciones varían mucho entre folds; no es posible separar de esta campaña cuánto corresponde físicamente a cada descriptor. Los momentos tienen atribución SHAP local apreciable en ciertos casos, pero su ablación muestra una mejora predictiva solo bajo parámetros fijos y una pérdida de desempeño de B frente A tras tuning equivalente por configuración. La selección de B se basó en el mismo dataset que el EDA, de modo que cualquier decisión definitiva requiere confirmación con ensayos independientes y geometrías conciliadas.
