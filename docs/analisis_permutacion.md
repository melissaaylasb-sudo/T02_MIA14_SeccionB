# Análisis de importancia por permutación

Esta entrega compara cinco familias con los parámetros fijos ya evaluados en `config/comparacion_modelos.json`, junto con mediana y media por nivel estimadas en entrenamiento. No se ejecuta búsqueda de hiperparámetros. El notebook reúne resultados, interpretaciones y matriz.

La selección de cuatro entradas se realiza para cada modelo dentro de su entrenamiento externo: cinco particiones internas por configuración y veinte permutaciones marginales cruzadas. La misma regla se usa al reservar configuraciones y al reservar familias. Se conservan cuatro entradas no constantes incluso si una importancia interna no es positiva; no existe un subconjunto universal declarado relevante.

La evaluación de importancia utiliza cien permutaciones marginales cruzadas sobre la matriz de predicciones externas. Se mantiene el modelo de cada receptor, se permutan solo sus entradas y se conservan los valores negativos. El donante puede pertenecer al entrenamiento del modelo receptor; no se presenta como un test común independiente. La desviación estándar y los percentiles describen variación Monte Carlo, no intervalos poblacionales.

Los grupos se declaran antes del cálculo. Para arquitectura y volúmenes se añade una ablación con reajuste en las mismas particiones. Los descriptores correlacionados y las combinaciones geométricas sintéticas limitan cualquier interpretación causal.

## Resultados externos

| Protocolo | Modelo | Entradas | MAE [N] | MSE [N²] | RMSE [N] | R² |
|---|---|---|---:|---:|---:|---:|
| configuracion | Ridge | todas | 13.247 | 267.183 | 16.346 | 0.923 |
| configuracion | Ridge | prioritarias | 14.067 | 275.915 | 16.611 | 0.920 |
| configuracion | Elastic Net | todas | 13.979 | 292.424 | 17.100 | 0.915 |
| configuracion | Elastic Net | prioritarias | 14.319 | 287.040 | 16.942 | 0.917 |
| configuracion | SVR RBF | todas | 15.651 | 364.649 | 19.096 | 0.894 |
| configuracion | SVR RBF | prioritarias | 10.285 | 172.411 | 13.131 | 0.950 |
| configuracion | Bosque aleatorio | todas | 13.233 | 267.044 | 16.341 | 0.923 |
| configuracion | Bosque aleatorio | prioritarias | 12.005 | 224.667 | 14.989 | 0.935 |
| configuracion | Proceso gaussiano | todas | 16.058 | 374.088 | 19.341 | 0.892 |
| configuracion | Proceso gaussiano | prioritarias | 9.972 | 164.323 | 12.819 | 0.952 |
| configuracion | Mediana | referencia | 49.748 | 4185.965 | 64.699 | -0.213 |
| configuracion | Media_por_nivel | referencia | 10.126 | 151.496 | 12.308 | 0.956 |
| familia | Ridge | todas | 57.637 | 3651.560 | 60.428 | -0.058 |
| familia | Ridge | prioritarias | 53.471 | 3153.825 | 56.159 | 0.086 |
| familia | Elastic Net | todas | 57.113 | 3586.558 | 59.888 | -0.039 |
| familia | Elastic Net | prioritarias | 53.400 | 3153.593 | 56.157 | 0.086 |
| familia | SVR RBF | todas | 82.358 | 7750.904 | 88.039 | -1.246 |
| familia | SVR RBF | prioritarias | 82.505 | 7733.359 | 87.940 | -1.241 |
| familia | Bosque aleatorio | todas | 60.901 | 5122.308 | 71.570 | -0.484 |
| familia | Bosque aleatorio | prioritarias | 64.813 | 5384.947 | 73.382 | -0.560 |
| familia | Proceso gaussiano | todas | 77.680 | 6993.737 | 83.629 | -1.026 |
| familia | Proceso gaussiano | prioritarias | 69.727 | 5916.508 | 76.919 | -0.714 |
| familia | Mediana | referencia | 97.511 | 9802.825 | 99.009 | -1.840 |
| familia | Media_por_nivel | referencia | 80.593 | 7660.002 | 87.521 | -1.219 |

## Contrastes de la matriz

| Hipótesis | Modelos que cumplen | Lectura |
|---|---|---|
| General | Ninguno | No respaldada bajo el protocolo |
| H1 | Ninguno | No respaldada bajo el protocolo |
| H2 | Proceso gaussiano | Respaldada descriptivamente bajo el protocolo |
| H3 | Ridge, Elastic Net, SVR RBF, Bosque aleatorio, Proceso gaussiano | Respaldada descriptivamente bajo el protocolo |

Los cortes de cero y la mayoría de tres de cinco expresan dirección y concordancia descriptivas. No son pruebas de significancia, equivalencia estadística ni umbrales de utilidad mecánica. Los criterios se documentaron después del EDA: no se presentan como prerregistro.

## Reproducción

```powershell
python -m pip install -r requirements-permutation-importance.txt
python -m src.importancia_permutacion
python -m src.revision_entrega_permutacion
python -m src.entrega_permutacion
python -m pytest tests/test_importancia_permutacion.py tests/test_entrega_permutacion.py tests/test_sprint_importancia.py -q
```

El módulo de análisis reutiliza una ejecución terminada solo si sus hashes y verificaciones concuerdan. Una modificación del protocolo requiere otra salida versionada. El notebook se ejecuta con kernel limpio y genera figuras desde los CSV comprobados. Los antecedentes de otras representaciones, tuning y explicaciones permanecen en sus rutas históricas; no se usan como resultados nuevos de esta actividad.

La procedencia declarada por la usuaria se conserva en `procedencia_declarada.json`. Las condiciones de uso y publicación y las aclaraciones de la fuente sobre carga terminal, geometría y ciclos continúan pendientes.

Referencias: [importancia por permutación](https://scikit-learn.org/1.7/modules/permutation_importance.html), [variables correlacionadas](https://scikit-learn.org/1.7/auto_examples/inspection/plot_permutation_importance_multicollinear.html).
