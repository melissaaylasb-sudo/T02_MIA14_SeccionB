# Resultados comparativos de regresión

## Pregunta

¿Cuánto del cambio en error corresponde a la representación geométrica B y cuánto al ajuste de hiperparámetros, manteniendo casos y particiones? La respuesta requiere predicciones **emparejadas** E1/E2/E3/E4, no comparar solo un baseline fijo A con un B optimizado.

## Evidencia anterior recuperada

| Contraste anterior | MAE por configuración [N] | RMSE [N] | R² OOF | MAE por familia [N] | Lectura |
|---|---:|---:|---:|---:|---|
| Dummy de media | 56.3090 | 61.7164 | −0.1036 | 80.5926 | Referencia ingenua anterior. |
| A · Ridge fijo | 13.2466 | 16.3457 | 0.9226 | 57.6374 | Geometría original. |
| B · Ridge fijo | 12.8346 | 16.2029 | 0.9239 | 57.6538 | A + dos momentos; ΔMAE = −0.4119 N frente A. |
| C · Ridge fijo | 12.9139 | 16.3271 | 0.9228 | 57.6190 | B + razón logarítmica; empeora por configuración. |
| D · Ridge fijo | 17.0350 | 21.1361 | 0.8706 | 68.8341 | Reducción de descriptores; deteriora. |
| A · Ridge ajustado | 9.7070 | 12.3308 | 0.9559 | 67.7689 | Ajuste interno previo sobre A; mejora por configuración y deterioro por familia. |

Fuente: [comparación FE](../reports/feature_engineering/tables/comparacion_experimentos.csv) y [modelos ajustados](../reports/modelos_ajustados/metricas_resumen.csv). Son experimentos ya ejecutados, pero B ajustado no estaba incluido. Por eso la comparación anterior **no** determina si B mantiene su ventaja tras el tuning ni permite atribuir la diferencia entre B fijo y A ajustado a una sola intervención.

## Resultados de esta rama: Ridge y Elastic Net

La [tabla ejecutada](../reports/feature_importance/tables/metricas_resumen.csv) permite comparar Ridge (E1–E4) y Elastic Net (S1–S4) sobre los 26 casos y las mismas particiones. E0 usa la **mediana** del train y por ello difiere del Dummy de media histórico. El tiempo es la suma de ajustes y predicciones externos de esta ejecución; depende del equipo.

| Protocolo | Experimento | Feature Set | Tuning | MAE [N] | RMSE [N] | MedAE [N] | R² OOF | MAE fold SD [N] | Tiempo [s] |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Configuración | E0 · Dummy | A | No | 49.7485 | 64.6990 | 35.8900 | −0.2128 | 41.0545 | 0.248 |
| Configuración | E1 · Ridge | A | No | 13.2466 | 16.3457 | 11.9409 | 0.9226 | 9.6765 | 0.251 |
| Configuración | E2 · Ridge | B | No | 12.8346 | 16.2029 | 11.7801 | 0.9239 | 10.1665 | 0.458 |
| Configuración | E3 · Ridge | B | Sí | 10.9814 | 14.3982 | 9.1447 | 0.9399 | 9.6449 | 19.925 |
| Configuración | E4 · Ridge | A | Sí | **9.7070** | **12.3308** | 9.2489 | **0.9559** | 7.8859 | 11.406 |
| Configuración | S1 · Elastic Net | A | No | 13.9792 | 17.1004 | 13.7200 | 0.9153 | 10.0244 | 0.267 |
| Configuración | S2 · Elastic Net | B | No | 13.4416 | 16.8718 | 12.7177 | 0.9175 | 10.5542 | 0.480 |
| Configuración | S3 · Elastic Net | B | Sí | 10.8724 | 14.1656 | **8.1519** | 0.9419 | 9.5425 | 36.580 |
| Configuración | S4 · Elastic Net | A | Sí | 10.1804 | 12.6282 | 9.2448 | 0.9538 | 7.7428 | 22.655 |
| Familia Y | E0 · Dummy | A | No | 97.5112 | 99.0092 | 96.3775 | −1.8403 | 16.2286 | 0.031 |
| Familia Y | E1 · Ridge | A | No | 57.6374 | 60.4281 | 64.5196 | −0.0580 | 17.1446 | 0.032 |
| Familia Y | E2 · Ridge | B | No | 57.6538 | 60.4740 | 61.7609 | −0.0596 | 17.2172 | 0.055 |
| Familia Y | E3 · Ridge | B | Sí | 61.3495 | 74.1288 | **45.7107** | −0.5921 | 48.3042 | 0.987 |
| Familia Y | E4 · Ridge | A | Sí | 67.7689 | 76.8187 | 54.3605 | −0.7098 | 41.6466 | 0.598 |
| Familia Y | S1 · Elastic Net | A | No | 57.1130 | 59.8879 | 64.2923 | −0.0392 | 16.9925 | 0.031 |
| Familia Y | S2 · Elastic Net | B | No | **57.0958** | **59.7843** | 62.5650 | **−0.0356** | 16.4124 | 0.055 |
| Familia Y | S3 · Elastic Net | B | Sí | 60.8060 | 64.5973 | 58.2522 | −0.2090 | 21.7847 | 1.687 |
| Familia Y | S4 · Elastic Net | A | Sí | 58.2144 | 62.3676 | 54.1102 | −0.1270 | 22.6544 | 0.879 |

El contraste de [errores pareados](../reports/feature_importance/tables/comparacion_pareada.csv) separa los efectos:

| Protocolo | Contraste | ΔMAE [N] | Casos mejorados | Interpretación |
|---|---|---:|---:|---|
| Configuración | E2−E1: momentos con parámetros fijos | −0.4119 | 57.7 % | Mejora pequeña, inferior al umbral previo de H1 (5 %). |
| Configuración | E3−E2: tuning con B | −1.8532 | 57.7 % | Tuning mejora B en el agregado, con variación amplia entre casos. |
| Configuración | E4−E1: tuning con A | −3.5395 | 61.5 % | Tuning mejora más a A que a B bajo este protocolo. |
| Configuración | E3−E4: B frente A, ambos ajustados | +1.2744 | 46.2 % | B ajustado queda por detrás de A ajustado. |
| Familia Y | E2−E1 | +0.0164 | 34.6 % | Los momentos no mejoran transferencia con parámetros fijos. |
| Familia Y | E3−E2 | +3.6957 | 53.8 % | B ajustado empeora el MAE agregado y aumenta mucho la dispersión entre folds. |
| Familia Y | E4−E1 | +10.1315 | 50.0 % | A ajustado también empeora la extrapolación de familias. |
| Familia Y | E3−E4 | −6.4194 | 53.8 % | B ajustado supera A ajustado en familia, pero ambos quedan detrás de E1 fijo. |
| Configuración | S2−S1: momentos con Elastic Net fijo | −0.5375 | 57.7 % | B mejora levemente con la segunda familia de modelo. |
| Configuración | S3−S2: tuning con B | −2.5692 | 57.7 % | Tuning mejora B en el agregado de configuración. |
| Configuración | S3−S4: B frente A, ambos ajustados | +0.6920 | 53.8 % | B ajustado queda por detrás de A ajustado pese a mejorar más de la mitad de casos. |
| Familia Y | S2−S1 | −0.0172 | 38.5 % | Mejora agregada casi nula, dentro de variación mayor entre casos. |
| Familia Y | S3−S2 | +3.7102 | 30.8 % | Tuning empeora la extrapolación con B. |
| Familia Y | S3−S4 | +2.5916 | 3.8 % | B ajustado peor que A ajustado en casi todos los casos. |

**Ridge E4 tiene el menor MAE al excluir configuraciones.** Al excluir una familia Y, el menor MAE de la tabla corresponde a **Elastic Net S2 fijo (57.0958 N)**, solo **0.0172 N** mejor que S1; todos los R² OOF por familia son negativos. No hay un ganador que domine ambos objetivos de generalización. La ventaja de B con parámetros fijos no persiste al comparar E3/E4 ni S3/S4 bajo tuning equivalente en configuraciones. La variabilidad de diferencias y la selección retrospectiva de B impiden afirmar una mejora robusta de las características derivadas.

## Criterios de lectura

- Comparar cada caso y configuración con la misma partición externa. La media de errores por fold y el MAE agrupado por observación pueden diferir cuando los tamaños de fold son desiguales.
- Examinar errores grandes, residuales y sesgo; una mejora pequeña de MAE puede coexistir con empeoramiento de RMSE o de la extrapolación a una familia Y ausente.
- Informar el número de casos y folds; no calcular R² en grupos de una sola observación.
- La selección retrospectiva de B tras examinar el mismo dataset impide considerar estas métricas una validación externa prospectiva. La conciliación geométrica y una campaña nueva siguen pendientes.
