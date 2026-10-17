# Resultados de ingeniería de características

[Ver el desarrollo, las figuras y la interpretación en el notebook](../notebooks/04_Feature_Engineering_Pantograficas.ipynb).

| Experimento | Cambio | MAE (N) | RMSE (N) |
| --- | --- | --- | --- |
| A | Geometría original | 13.2466 | 16.3457 |
| B | A + dos momentos de sección | 12.8346 | 16.2029 |
| C | B + ln(Vp/Vf) | 12.9139 | 16.3271 |
| D | C − tres descriptores redundantes | 17.0350 | 21.1361 |

**B obtiene el menor MAE: 12.8346 N**, una reducción de 3.11 % frente a A. Al excluir familias completas, su MAE es 57.6538 N; la mejora no demuestra transferencia a arquitecturas nuevas.

## Intervenciones

| experimento | referencia | delta_MAE_N | fraccion_grupos_mejora | decision |
| --- | --- | --- | --- | --- |
| B | A | -0.4119 | 0.6250 | Mantener en evaluación |
| C | B | 0.0792 | 0.3750 | Descartar |
| D | C | 4.1211 | 0.2917 | Descartar |

ΔMAE negativo indica mejora frente a la referencia de cada fila. B: mantener en evaluación; C: descartar la razón añadida frente a B; D: descartar la reducción probada. Las decisiones corresponden a este modelo y protocolo.

La variante D produce fuerzas negativas al excluir Y=4: un fallo físico de extrapolación, conservado en las métricas. La comprobación de sensibilidad omite scores del agregado OOF; no elimina su influencia en los modelos entrenados.

## Alcance

El EDA ya utilizó todo el conjunto. Estos resultados no son validación externa ni garantizan seguridad estructural. El siguiente experimento es confirmar A y B con geometrías conciliadas y ensayos independientes, manteniendo el protocolo congelado.

[Resultados completos y tiempos](../reports/feature_engineering/tables/comparacion_experimentos.csv) · [Predicciones OOF](../reports/feature_engineering/tables/predicciones_oof.csv) · [Errores por familia](../reports/feature_engineering/tables/errores_familia.csv) · [Sensibilidad](../reports/feature_engineering/tables/sensibilidad_scores.csv)
