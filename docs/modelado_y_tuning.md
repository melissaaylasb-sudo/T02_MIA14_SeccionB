# Modelado y tuning controlado

## Pregunta y material recuperado

La respuesta es `First_Failure_Load_N`, carga de primera falla en **N**. El análisis usa la [tabla curada fija](../data/processed/preliminary/model_table.csv), con 26 casos etiquetados, y las [particiones anteriores](../reports/feature_engineering/particiones.json): validación principal LeaveOneGroupOut por configuración nominal (`n_cells_Y`, `Fiber_height`, `Pivot_height`) y comprobación adicional dejando fuera una familia Y completa. La evidencia anterior procede del [experimento FE](experimentos_fe.md), la [configuración de ajuste](../config/ajuste_modelos.json) y sus [métricas ejecutadas](../reports/modelos_ajustados/metricas_resumen.csv).

| Modelo | Feature Set | Validación | MAE [N] | RMSE [N] | R² OOF | Parámetros | Evidencia | Estado |
|---|---|---|---:|---:|---:|---|---|---|
| Dummy de media | Control anterior | Configuración | 56.3090 | 61.7164 | −0.1036 | media de train | [FE](../reports/feature_engineering/tables/comparacion_experimentos.csv) | Recuperado; E0 nuevo podrá usar mediana. |
| Ridge | A original | Configuración | 13.2466 | 16.3457 | 0.9226 | α=1, solver SVD, fijo | [FE](../reports/feature_engineering/tables/comparacion_experimentos.csv) | E1 de referencia. |
| Ridge | B = A + dos momentos | Configuración | 12.8346 | 16.2029 | 0.9239 | Igual a A | [FE](../reports/feature_engineering/tables/comparacion_experimentos.csv) | E2, candidato FE provisional. |
| Ridge | C = B + ln(Vp/Vf) | Configuración | 12.9139 | 16.3271 | 0.9228 | Igual a A | [FE](../reports/feature_engineering/tables/comparacion_experimentos.csv) | Descartado en comparación previa. |
| Ridge | D = C − tres originales | Configuración | 17.0350 | 21.1361 | 0.8706 | Igual a A | [FE](../reports/feature_engineering/tables/comparacion_experimentos.csv) | Descartado. |
| Elastic Net | A original | Configuración | 13.9792 | 17.1004 | 0.9153 | Fijos, sin búsqueda | [FE](../reports/feature_engineering/tables/comparacion_modelos.csv) | Comparación de familias previa. |
| SVR RBF | A original | Configuración | 15.6507 | 19.0958 | 0.8943 | Fijos, sin búsqueda | [FE](../reports/feature_engineering/tables/comparacion_modelos.csv) | Comparación previa. |
| Bosque aleatorio | A original | Configuración | 13.2325 | 16.3415 | 0.9226 | Fijos, sin búsqueda | [FE](../reports/feature_engineering/tables/comparacion_modelos.csv) | Ventaja de solo 0.0141 N frente Ridge fijo; no selección concluyente. |
| Proceso gaussiano | A original | Configuración | 16.0577 | 19.3414 | 0.8916 | Fijos, sin búsqueda | [FE](../reports/feature_engineering/tables/comparacion_modelos.csv) | Comparación previa. |
| Ridge | A original | Configuración | 9.7070 | 12.3308 | 0.9559 | α elegido por validación interna en cada fold | [Ajuste](../reports/modelos_ajustados/metricas_resumen.csv) | Mejor MAE observado entre modelos ajustados de A. |
| Elastic Net | A original | Configuración | 10.1804 | 12.6282 | 0.9538 | α y l1_ratio internos | [Ajuste](../reports/modelos_ajustados/metricas_resumen.csv) | Ajustado. |
| SVR RBF | A original | Configuración | 13.6745 | 17.6140 | 0.9101 | C, ε, γ internos | [Ajuste](../reports/modelos_ajustados/metricas_resumen.csv) | Ajustado. |
| Bosque aleatorio | A original | Configuración | 12.7554 | 15.6272 | 0.9292 | Complejidad interna | [Ajuste](../reports/modelos_ajustados/metricas_resumen.csv) | Ajustado. |
| Extra Trees | A original | Configuración | 12.4772 | 15.7509 | 0.9281 | Complejidad interna | [Ajuste](../reports/modelos_ajustados/metricas_resumen.csv) | Ajustado. |
| Proceso gaussiano | A original | Configuración | 13.8756 | 17.4750 | 0.9115 | Kernel/ruido internos | [Ajuste](../reports/modelos_ajustados/metricas_resumen.csv) | Ajustado. |
| Gradient Boosting | A original | Configuración | 13.8495 | 17.2385 | 0.9139 | Árboles/pérdida internos | [Ajuste](../reports/modelos_ajustados/metricas_resumen.csv) | Ajustado. |

Las filas de la tabla corresponden a experimentos previos distintos; **no** atribuyen a B el resultado del Ridge ajustado sobre A. Los parámetros seleccionados de cada fold y todos los candidatos del experimento anterior están en [métricas por fold](../reports/modelos_ajustados/metricas_folds.csv) y [búsquedas](../reports/modelos_ajustados/busquedas.csv). La variante B obtuvo el menor MAE entre A–D con Ridge fijo, pero la mejora fue **3.11 %**, inferior al criterio previo de 5 %. El MAE al excluir familias fue A = **57.6374 N** y B = **57.6538 N**. B es por tanto un **candidato para contraste**, no un conjunto confirmado como superior ante arquitecturas nuevas.

## Feature Set y secuencia nueva

`FEATURE_SET_BEST_FE` corresponde a **B**: las diez variables originales admisibles de A más `I_fibra_bh3_mm4` y `I_fibra_hb3_mm4`. El filtro de columnas constantes retira el radio de pivote después de ajustarse en train. Se excluye la razón C porque empeoró el MAE frente B y se excluye la retirada D porque deterioró el desempeño. Las fórmulas, unidades y riesgos constan en el [diccionario](diccionario_features.md) y la [operacionalización](operacionalizacion_variables.md).

| Experimento | Intervención aislada | Comparación interpretable |
|---|---|---|
| E0 | Predicción ingenua del centro del entrenamiento. | Comprueba que el modelo añade utilidad frente a una referencia trivial. |
| E1 | A original, Ridge fijo de referencia. | Punto de partida con misma geometría, modelo y particiones anteriores. |
| E2 | B, Ridge fijo idéntico a E1. | E2−E1 estima el efecto predictivo de la representación B bajo parámetros fijos. |
| E3 | B, Ridge con búsqueda interna. | E3−E2 estima el efecto adicional del tuning sobre B. |
| E4 | A, Ridge con búsqueda interna. | E3−E4 compara A y B bajo ajuste equivalente. |

La segunda familia recuperada es **Elastic Net**: S1 (A fijo), S2 (B fijo), S3 (B con tuning interno) y S4 (A con tuning interno), con los mismos casos y particiones. Esto permite contrastar si el patrón A/B de Ridge se repite bajo otra regularización lineal. La selección entre familias mediante el mismo OOF externo es descriptiva y no equivale a un selector validado; el experimento previo sí incluye un [selector interno](../reports/modelos_ajustados/seleccion_interna.json).

## Método de validación

El preprocesamiento se aprende dentro de cada entrenamiento: contrato geométrico, imputación mediana, creación de atributos, filtro de varianza nula y estandarización. En E3/E4, la búsqueda de hiperparámetros ocurre dentro del train de cada fold externo, con agrupación en la validación interna. El fold externo se reserva para predicción y métricas. Se conservan las mismas filas y splits externos en las comparaciones emparejadas. Se registran candidatos, selección por fold, semilla, versiones y tiempos; la [configuración anterior](../config/ajuste_modelos.json) ofrece la referencia del espacio Ridge (α ∈ {0.0001,…,1000}).

El MAE agregado se calcula sobre predicciones fuera de entrenamiento; RMSE, MedAE, R², sesgo y errores por grupo complementan la decisión. No se calcula R² sobre un fold individual de una sola observación. Un menor MAE observado en este mismo conjunto ya explorado no acredita generalización prospectiva.

## Resultados ejecutados y limitaciones

El EDA, las hipótesis de FE y la selección retrospectiva de B utilizaron el mismo snapshot. La validación anidada controla la selección de hiperparámetros dentro de cada fold, pero no elimina ese sesgo de investigación. Hay 26 casos y tres familias Y; las discrepancias geométricas y la falta de lotes/réplicas verificados limitan la interpretación de error y estabilidad.

La ejecución Ridge de esta rama está en [métricas](../reports/feature_importance/tables/metricas_resumen.csv), [búsquedas](../reports/feature_importance/tables/busquedas.csv) y [comparaciones pareadas](../reports/feature_importance/tables/comparacion_pareada.csv). Al excluir configuraciones, E4 (A ajustado) logró MAE **9.7070 N**, mejor que E3 (B ajustado), **10.9814 N**. El tuning mejoró B frente a B fijo en **1.8532 N**, mientras que los momentos mejoraron A fijo solo **0.4119 N**. Al excluir familias Y, E1 fijo obtuvo el menor MAE Ridge, **57.6374 N**; E3 y E4 empeoraron. Los parámetros seleccionados deben leerse por fold, pues no hay un único α final que describa todas las predicciones OOF. Esta comparación no es una confirmación experimental independiente.

Elastic Net reproduce el patrón principal al excluir configuraciones: B fijo (S2, **13.4416 N**) mejora A fijo (S1, **13.9792 N**), y el tuning mejora B (S3, **10.8724 N**); sin embargo, A ajustado S4 (**10.1804 N**) supera B ajustado. Al excluir familias, S2 tiene el menor MAE entre ambos algoritmos (**57.0958 N**), apenas **0.0172 N** mejor que S1, con R² OOF negativo. El tuning empeora S2 a **60.8060 N**. La tabla completa está en [resultados comparativos](resultados_comparativos.md).

Las [búsquedas internas](../reports/feature_importance/tables/busquedas.csv) registran parámetros por fold. En las 24 particiones de configuración, Ridge E3 eligió α=0.1 en 12 y E4 eligió α=0.0001 en 22; Elastic Net S3 eligió (α=0.01, l1_ratio=0.9) en 13 y S4 eligió (α=0.001, l1_ratio=0.9) en 23. Al excluir familias, Ridge eligió α=0.0001 en un fold y α=1000 en los otros dos para E3 y E4, señal de selección sensible a la arquitectura ausente. Son frecuencias descriptivas de la búsqueda, no estimaciones de un hiperparámetro universal.
