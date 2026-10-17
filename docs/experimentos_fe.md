# Resultados de ingeniería de características

[Ver el desarrollo, las figuras y la interpretación en el notebook](../notebooks/04_Feature_Engineering_Pantograficas.ipynb).

## Características: mismo modelo Ridge

| Experimento | Cambio | MAE (N) | RMSE (N) | MedAE (N) | R² OOF | Tiempo CV (s) |
| --- | --- | --- | --- | --- | --- | --- |
| Línea base (A) | Geometría original | 13.2466 | 16.3457 | 11.9409 | 0.9226 | 0.2901 |
| Variante 1 (B) | A + dos momentos de sección | 12.8346 | 16.2029 | 11.7801 | 0.9239 | 0.6825 |
| Variante 2 (C) | B + ln(Vp/Vf) | 12.9139 | 16.3271 | 11.8643 | 0.9228 | 0.5711 |
| Variante 3 (D) | C − tres descriptores redundantes | 17.0350 | 21.1361 | 14.9109 | 0.8706 | 0.5131 |

**Tiempo CV:** suma medida de ajuste y predicción en la validación por configuración; excluye figuras y escritura de archivos. Es costo computacional de esta ejecución, no costo monetario ni latencia de una aplicación.

**B obtiene el menor MAE: 12.8346 N**, una reducción de 3.11 % frente a A. Al excluir familias completas, su MAE es 57.6538 N; la mejora no demuestra transferencia a arquitecturas nuevas.

## Intervenciones

| Variante | Referencia | ΔMAE (N) | Fracción de grupos con mejora | Decisión |
| --- | --- | --- | --- | --- |
| Variante 1 (B) | Línea base (A) | -0.4119 | 0.6250 | Mantener en evaluación |
| Variante 2 (C) | Variante 1 (B) | 0.0792 | 0.3750 | Descartar |
| Variante 3 (D) | Variante 2 (C) | 4.1211 | 0.2917 | Descartar |

ΔMAE negativo indica mejora frente a la referencia de cada fila. B: mantener en evaluación; C: descartar la razón añadida frente a B; D: descartar la reducción probada. Las decisiones corresponden a este modelo y protocolo.

La variante D produce fuerzas negativas al excluir Y=4: un fallo físico de extrapolación, conservado en las métricas. La comprobación de sensibilidad omite scores del agregado OOF; no elimina su influencia en los modelos entrenados.

## Algoritmos: misma geometría original A

Los cinco regresores usan las mismas entradas, preprocesamiento y particiones. Cada fila reúne predicciones de configuraciones excluidas del entrenamiento; los parámetros se fijaron sin búsqueda de hiperparámetros.

| Modelo | MAE (N) | RMSE (N) | MedAE (N) | R² OOF | Tiempo CV (s) |
| --- | --- | --- | --- | --- | --- |
| Ridge | 13.2466 | 16.3457 | 11.9409 | 0.9226 | 0.2716 |
| Elastic Net | 13.9792 | 17.1004 | 13.7200 | 0.9153 | 0.3525 |
| SVR con núcleo RBF | 15.6507 | 19.0958 | 12.6334 | 0.8943 | 0.3309 |
| Bosque aleatorio | 13.2325 | 16.3415 | 9.6994 | 0.9226 | 4.1263 |
| Proceso gaussiano | 16.0577 | 19.3414 | 14.1937 | 0.8916 | 0.2480 |

**Bosque aleatorio presenta el menor MAE por configuración: 13.2325 N.** La diferencia respecto a Ridge es 0.0141 N. Al excluir familias completas, su MAE es 60.9006 N. La comparación es exploratoria: no constituye una selección validada en ensayos externos.

El notebook muestra errores, residuales y el contraste entre excluir configuraciones o familias completas. [Resultados de ambos protocolos](../reports/feature_engineering/tables/comparacion_modelos.csv).

## Alcance

El EDA ya utilizó todo el conjunto. Estos resultados no son validación externa ni garantizan seguridad estructural. El siguiente experimento es confirmar A y B con geometrías conciliadas y ensayos independientes, manteniendo el protocolo congelado.

[Resultados completos y tiempos](../reports/feature_engineering/tables/comparacion_experimentos.csv) · [Predicciones OOF](../reports/feature_engineering/tables/predicciones_oof.csv) · [Errores por familia](../reports/feature_engineering/tables/errores_familia.csv) · [Sensibilidad](../reports/feature_engineering/tables/sensibilidad_scores.csv)
