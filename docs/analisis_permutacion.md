# Análisis de importancia por permutación

Esta entrega compara seis familias no lineales con la mediana y la media por nivel estimadas en entrenamiento. `GridSearchCV` examina cuatro configuraciones por familia sobre la geometría completa dentro del entrenamiento de cada partición externa, con validación interna agrupada. Los parámetros elegidos se reutilizan para el subconjunto de cuatro variables y la ablación. El cuaderno reúne las predicciones externas, las métricas, la importancia de variables y la matriz de consistencia.

La selección de cuatro entradas se realiza para cada modelo dentro de su entrenamiento externo mediante permutación interna. La misma regla se usa al reservar configuraciones y al reservar familias. Se conservan cuatro entradas no constantes incluso si una importancia interna no es positiva; no existe un subconjunto universal declarado relevante.

La evaluación de importancia utiliza cien permutaciones marginales cruzadas sobre la matriz de predicciones externas. Se mantiene el modelo de cada receptor, se permutan solo sus entradas y se conservan los valores negativos. El donante puede pertenecer al entrenamiento del modelo receptor; no se presenta como un test común independiente. La desviación estándar y los percentiles describen variación Monte Carlo, no intervalos poblacionales.

Los grupos se declaran antes del cálculo. Para arquitectura y volúmenes se añade una ablación con reajuste en las mismas particiones. Los descriptores correlacionados y las combinaciones geométricas sintéticas limitan cualquier interpretación causal.

## Resultados externos

| Protocolo | Modelo | Entradas | MAE [N] | MSE [N²] | RMSE [N] | R² |
|---|---|---|---:|---:|---:|---:|
| configuracion | SVR RBF | todas | 16.365 | 439.801 | 20.971 | 0.873 |
| configuracion | SVR RBF | prioritarias | 11.860 | 216.326 | 14.708 | 0.937 |
| configuracion | Bosque aleatorio | todas | 13.808 | 292.658 | 17.107 | 0.915 |
| configuracion | Bosque aleatorio | prioritarias | 12.681 | 268.174 | 16.376 | 0.922 |
| configuracion | Árboles extremadamente aleatorios | todas | 12.951 | 259.415 | 16.106 | 0.925 |
| configuracion | Árboles extremadamente aleatorios | prioritarias | 11.963 | 204.448 | 14.299 | 0.941 |
| configuracion | Proceso gaussiano | todas | 16.058 | 374.088 | 19.341 | 0.892 |
| configuracion | Proceso gaussiano | prioritarias | 9.972 | 164.323 | 12.819 | 0.952 |
| configuracion | Gradient Boosting | todas | 13.788 | 315.112 | 17.751 | 0.909 |
| configuracion | Gradient Boosting | prioritarias | 11.282 | 229.807 | 15.159 | 0.933 |
| configuracion | XGBoost | todas | 11.758 | 231.564 | 15.217 | 0.933 |
| configuracion | XGBoost | prioritarias | 12.225 | 241.166 | 15.530 | 0.930 |
| configuracion | Mediana | referencia | 49.748 | 4185.965 | 64.699 | -0.213 |
| configuracion | Media_por_nivel | referencia | 10.126 | 151.496 | 12.308 | 0.956 |
| familia | SVR RBF | todas | 83.884 | 7962.441 | 89.233 | -1.307 |
| familia | SVR RBF | prioritarias | 79.545 | 7327.401 | 85.600 | -1.123 |
| familia | Bosque aleatorio | todas | 61.472 | 5172.466 | 71.920 | -0.499 |
| familia | Bosque aleatorio | prioritarias | 66.471 | 5553.662 | 74.523 | -0.609 |
| familia | Árboles extremadamente aleatorios | todas | 65.005 | 5570.009 | 74.632 | -0.614 |
| familia | Árboles extremadamente aleatorios | prioritarias | 62.101 | 5018.287 | 70.840 | -0.454 |
| familia | Proceso gaussiano | todas | 80.593 | 7660.061 | 87.522 | -1.219 |
| familia | Proceso gaussiano | prioritarias | 80.595 | 7660.263 | 87.523 | -1.219 |
| familia | Gradient Boosting | todas | 63.719 | 5308.472 | 72.859 | -0.538 |
| familia | Gradient Boosting | prioritarias | 63.573 | 5013.361 | 70.805 | -0.453 |
| familia | XGBoost | todas | 56.189 | 4580.788 | 67.682 | -0.327 |
| familia | XGBoost | prioritarias | 57.426 | 4707.978 | 68.615 | -0.364 |
| familia | Mediana | referencia | 97.511 | 9802.825 | 99.009 | -1.840 |
| familia | Media_por_nivel | referencia | 80.593 | 7660.002 | 87.521 | -1.219 |

## Contrastes de la matriz

| Hipótesis | Modelos que cumplen | Lectura |
|---|---|---|
| General | Ninguno | No respaldada bajo el protocolo |
| H1 | Ninguno | No respaldada bajo el protocolo |
| H2 | SVR RBF, Árboles extremadamente aleatorios, Gradient Boosting | Respaldada descriptivamente bajo el protocolo |
| H3 | SVR RBF, Bosque aleatorio, Árboles extremadamente aleatorios, Proceso gaussiano, Gradient Boosting, XGBoost | Respaldada descriptivamente bajo el protocolo |

Los cortes de cero y la mayoría de cuatro de seis expresan dirección y concordancia descriptivas. No son pruebas de significancia, equivalencia estadística ni umbrales de utilidad mecánica. Los criterios se documentaron después del EDA: no se presentan como prerregistro.

## Lectura conjunta de desempeño e interpretabilidad

Con todas las entradas, XGBoost obtuvo el menor MAE al reservar configuraciones (11,758 N) y familias completas (56,189 N). Al seleccionar cuatro entradas dentro del entrenamiento, el proceso gaussiano alcanzó el menor MAE por configuraciones (9,972 N), pero su ventaja frente a la media por nivel de celdas (10,126 N) es pequeña y no se mantiene al reservar familias. En este segundo escenario, el R² de XGBoost es −0,327: el menor MAE entre los modelos evaluados no implica una transferencia satisfactoria a arquitecturas no vistas.

SHAP se aplicó a los estimadores XGBoost con predicciones externas por configuración y referencias tomadas del entrenamiento correspondiente. El número de celdas en Y tiene la mayor atribución absoluta media (51,271 N), seguido del volumen declarado del espécimen (2,815 N). Su importancia por permutación para XGBoost es 49,699 N, pero los tres descriptores de arquitectura están muy correlacionados. La concentración de atribución en el número de celdas no permite aislar un mecanismo físico de falla. En el caso de mayor error, la predicción fue 95,735 N frente a 47,550 N observados. Las [tablas SHAP](../reports/modelos_no_lineales/tables/shap_global.csv) y las figuras del cuaderno muestran tanto esta contribución como su estabilidad entre particiones.

## Reproducción

```powershell
python -m pip install -r requirements-modelos-no-lineales.txt
python -m src.importancia_permutacion --output reports/modelos_no_lineales
python -m src.revision_entrega_permutacion
python -m src.entrega_permutacion
python -m pytest tests/test_importancia_permutacion.py tests/test_entrega_permutacion.py tests/test_sprint_importancia.py -q
```

El módulo de análisis reutiliza una ejecución terminada solo si sus hashes y verificaciones concuerdan. Una modificación del protocolo requiere otra salida versionada. El notebook se ejecuta con kernel limpio y genera figuras desde los CSV comprobados. Los antecedentes de otras representaciones permanecen en sus rutas históricas y no se mezclan con las métricas de esta actividad.

La procedencia declarada por la usuaria se conserva en `procedencia_declarada.json`. Las condiciones de uso y publicación y las aclaraciones de la fuente sobre carga terminal, geometría y ciclos continúan pendientes.

Referencias: [importancia por permutación](https://scikit-learn.org/1.7/modules/permutation_importance.html), [variables correlacionadas](https://scikit-learn.org/1.7/auto_examples/inspection/plot_permutation_importance_multicollinear.html).
