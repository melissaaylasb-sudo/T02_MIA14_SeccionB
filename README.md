# Evaluación comparativa de seis regresores para primera falla pantográfica

**Entrega: [un único notebook ejecutado y autocontenido](notebooks/Comparativa_6_Modelos_Regresion_Pantograficas.ipynb)** · [Descargar solo el .ipynb](https://raw.githubusercontent.com/melissaaylasb-sudo/T02_MIA14_SeccionB/benchmark-regresion-6-modelos/notebooks/Comparativa_6_Modelos_Regresion_Pantograficas.ipynb)

La portada muestra el MAE al reservar una configuración y al retirar un nivel celular completo. El análisis calcula primero las tres referencias (§4), entrena los seis modelos con parámetros iniciales (§9), muestra sus errores (§10–11) y selecciona candidatas dentro de cada entrenamiento (§12). Después define el presupuesto (§13), ajusta hiperparámetros (§14) y compara las predicciones sobre los mismos especímenes (§15).

Para reproducirlo, descarga el archivo, ábrelo en Jupyter y selecciona **Reiniciar el kernel y ejecutar todo**. El archivo conserva una copia verificada de las entradas en sus metadatos y los diez gráficos en sus salidas. Con las bibliotecas indicadas en §19 instaladas, puede ejecutarse por sí solo en una carpeta sin conexión. La [comprobación de reproducción](reports/benchmark_regresion/auditoria/cuaderno_unico.json) contrasta las predicciones y métricas con la ejecución anterior.

Universidad Nacional de Ingeniería · Maestría en Inteligencia Artificial · Trabajo de Investigación II. **Investigadora:** Melissa Dessire Aylas Barranca. **Docente:** Glen Dario Rodríguez Rafael.

Se estima la carga de primera rotura de fibra o pivote en newtons con geometría conocida antes del ensayo. En una misma campaña se comparan SVR-RBF, XGBoost, CatBoost, regresión lineal múltiple, KNN y árbol de decisión con tres controles: mediana, media por nivel celular y Ridge histórico. Se distinguen las configuraciones de niveles ya observados y la transferencia a un nivel Y ausente. El cribado y la búsqueda acotada se repiten dentro de cada entrenamiento. La tabla final separa el mejor MAE descriptivo de la recomendación para uso posterior.

| Qué revisar | Evidencia |
|---|---|
| Elección de algoritmos y literatura de materiales | [Matriz de justificación](docs/justificacion_seleccion_modelos.md) |
| Elección de entradas, exclusiones y particiones | [Metodología](docs/metodologia_comparativa_modelos.md), [diez variables](docs/benchmark/features.csv), [80 resultados históricos](docs/benchmark/antecedentes_metricas.csv) |
| Configuraciones iniciales, selección y tuning | [Diseño de validación](docs/hiperparametros_y_validacion.md), [configuración](config/benchmark_regresion.json) |
| Métricas, tiempos y predicciones por caso | [Tablas numéricas](reports/benchmark_regresion/tables/), [figuras](reports/benchmark_regresion/figures/), [recomendación legible por máquina](reports/benchmark_regresion/interpretacion/recomendacion.json) |

**Resultado de esta ejecución:** la regresión lineal múltiple obtuvo el menor MAE de las seis recetas en configuraciones reservadas (9,664 N). Su ventaja sobre la media por nivel fue 0,462 N y ganó en 12 de 24 grupos; el RMSE fue prácticamente idéntico. La regla de decisión conserva por ello la media por nivel como referencia operativa para niveles conocidos. Al retirar un nivel completo, el MAE lineal fue 40,915 N. El tuning del selector aumentó 0,633 N el MAE nominal frente a sus propias familias iniciales. [Tabla integrada con configuraciones y tiempos](reports/benchmark_regresion/interpretacion/tabla_final_integrada.csv).

**Auditoría opcional desde el repositorio.** Para quienes quieran contrastar el cuaderno con los archivos de la implementación modular, instalar `python -m pip install -r requirements-benchmark-regresion.txt` (Python de referencia: 3.13.5) y ejecutar desde la raíz:

```powershell
python -m src.benchmark_regresion --verify
python -m src.auditoria_benchmark
python -m src.lectura_benchmark_regresion
python -m pytest -q tests/test_benchmark_regresion.py tests/test_lectura_benchmark.py tests/test_entrega_benchmark.py
```

La implementación modular también permite repetir los entrenamientos en un directorio nuevo: `python -m src.benchmark_regresion --output reports/repeticion_benchmark`. La [auditoría independiente](reports/benchmark_regresion/auditoria/revision_independiente.json) registra el contraste con la fuente y los controles históricos. Los cuadernos de etapas anteriores se conservan como antecedentes; ninguno se utiliza para ejecutar esta entrega.

---

## Antecedente: importancia de variables y matriz de consistencia

## Predicción de la carga de primera falla en estructuras pantográficas

**Rama de la entrega anterior:** [feature-importance-consistencia](https://github.com/melissaaylasb-sudo/T02_MIA14_SeccionB/tree/feature-importance-consistencia).

### Entrega de la etapa de permutación y selección interna de variables

El [notebook histórico de la etapa de permutación](notebooks/06_Modelado_Tuning_SHAP.ipynb) compara cinco familias con parámetros fijos: Ridge, Elastic Net, SVR RBF, bosque aleatorio y proceso gaussiano. Incluye referencias de mediana y media por nivel, las tablas verde y morada con métricas, selección de cuatro entradas dentro del entrenamiento, permutación repetida y ablación del bloque de arquitectura y volúmenes.

Las dos validaciones reservan configuraciones nominales o familias de celdas de la misma campaña. Las cuatro entradas seleccionadas pueden cambiar entre modelos y particiones. La actividad no ejecuta una nueva búsqueda de hiperparámetros ni presenta a Ridge o Elastic Net como óptimos. Las importancias usan 100 permutaciones marginales cruzadas, mantienen los signos negativos y distinguen variación de Monte Carlo de incertidumbre poblacional.

### Matriz y reproducción

La [matriz de consistencia](docs/matriz_consistencia.md), su [CSV canónico](docs/matriz_consistencia.csv) y el notebook comparten las mismas formulaciones. El [análisis de permutación](docs/analisis_permutacion.md) y la [trazabilidad de resultados](docs/trazabilidad_objetivos_resultados.md) corresponden a esta ejecución. Los criterios son posteriores al EDA y no se presentan como prerregistrados.

Para reproducir el experimento desde la raíz del repositorio:

```powershell
python -m pip install -r requirements-permutation-importance.txt
python -m src.importancia_permutacion
python -m src.revision_entrega_permutacion
python -m src.entrega_permutacion
python -m pytest tests/test_importancia_permutacion.py tests/test_entrega_permutacion.py tests/test_sprint_importancia.py -q
```

La ejecución conserva [tablas](reports/permutation_importance/tables/), [figuras](reports/permutation_importance/figures/) y [manifiesto](reports/permutation_importance/manifest.json). El notebook se ejecuta desde un kernel limpio y comprueba los artefactos antes de mostrarlos. Las condiciones de procedencia y las aclaraciones aún pendientes se registran en [procedencia declarada](docs/procedencia_declarada.json).

## Antecedente histórico: etapa de modelos ajustados

Los resultados siguientes pertenecen a la etapa anterior de tuning. Sus protocolos y cifras no se mezclan con las tablas de parámetros fijos de la entrega actual.

**[Abrir notebook 05 — modelos, ajuste y evaluación](notebooks/05_Modelos_Ajustados.ipynb)**

[Enlace para compartir](https://github.com/melissaaylasb-sudo/T02_MIA14_SeccionB/blob/codex/modelos-ajustados/notebooks/05_Modelos_Ajustados.ipynb) · [Etapa anterior: ingeniería de características](notebooks/04_Feature_Engineering_Pantograficas.ipynb)

Melissa Dessire Aylas Barranca · UNI · Maestría en Inteligencia Artificial.

Estimar, a partir de la geometría previa al ensayo, la **fuerza en newtons de la primera rotura de una fibra o un pivote**. Se compara el Ridge de referencia con regresores ajustados mediante validación cruzada anidada. El desarrollo, las imágenes y la interpretación están en el notebook.

| Elemento | Especificación |
|---|---|
| Entrada | Número de celdas en Y; altura, ancho, longitud y volumen de fibras; altura, radio, cantidad y volumen de pivotes; volumen CAD del espécimen. |
| Target | `First_Failure_Load_N`: carga medida en la primera falla, en N. |
| Salida | Estimación numérica de esa fuerza; se evalúa el error frente a la medición experimental. |
| Variables | Geometría original A, igual que el baseline. El radio constante se filtra dentro de cada entrenamiento. [Diccionario y unidades](docs/diccionario_features.md#variables-de-esta-etapa). |
| Referencia | Ridge con parámetros fijos del experimento anterior; sus predicciones se vuelven a calcular y se contrastan con las publicadas. |
| Modelos ajustados | Ridge, Elastic Net, SVR con núcleo RBF, Random Forest, Extra Trees, proceso gaussiano y Gradient Boosting. |
| Selección del algoritmo | Se incluye un selector que elige con la validación **interna**, sin usar los errores del grupo externo. |

## Resultado ejecutado

| Procedimiento | MAE configuración [N] | RMSE configuración [N] | R² configuración | MAE familia [N] |
|---|---:|---:|---:|---:|
| Ridge · baseline | 13.247 | 16.346 | 0.9226 | 57.637 |
| Ridge ajustado | 9.707 | 12.331 | 0.9559 | 67.769 |
| Elastic Net ajustado | 10.180 | 12.628 | 0.9538 | 58.214 |
| Selección interna de algoritmo | 11.818 | 16.135 | 0.9246 | 72.089 |

**Ridge ajustado redujo el MAE un 26.72 % frente al baseline** al excluir configuraciones. Es el menor MAE observado entre los algoritmos comparados. La selección interna evalúa por separado la elección automática del algoritmo. Al excluir familias completas, ninguno de los modelos ajustados mejora el baseline; la transferencia a arquitecturas nuevas sigue sin resolverse.

[Tabla completa de métricas y tiempos](reports/modelos_ajustados/metricas_resumen.csv) · [Verificación de la ejecución](reports/modelos_ajustados/verificacion.json)

![Métricas fuera de muestra de todos los modelos](reports/modelos_ajustados/figures/02_metricas_modelos.png)

## Protocolo de comparación

| Requisito | Implementación verificable |
|---|---|
| Split correcto | Se reutilizan exactamente los casos de train/evaluación de la Semana 5: LeaveOneGroupOut por configuración nominal; comprobación adicional dejando fuera una familia Y completa. |
| Ajuste de hiperparámetros | GridSearchCV dentro de cada train externo. GroupKFold de cinco particiones por configuración; LeaveOneGroupOut por familias para la comprobación adicional. |
| Fit solo en train | Contrato de entradas → imputación por mediana → geometría A → filtro de constantes → estandarización → regresor. Se ajusta el pipeline completo también en cada partición interna. |
| PCA / target encoding | No forman parte de este experimento. Las entradas son numéricas y no se incorporan transformaciones aprendidas fuera del pipeline. |
| Semillas | 42 en algoritmos aleatorios. Las particiones de grupos son deterministas. Versiones, parámetros y semillas quedan registrados. |
| Datos constantes | Misma tabla curada, mismo target y mismos casos en todos los modelos. SHA-256 comprobado antes y después; sin nuevos datos, eliminación de casos ni imputación del target. |
| Logs | Configuración congelada, resultados de todos los candidatos, parámetros por partición, predicciones, tiempos, avisos y fechas UTC. |

El ajuste interno es la incorporación nueva; **la evaluación externa conserva las particiones anteriores**. La selección interna minimiza el MAE promedio entre particiones; las métricas externas se calculan reuniendo las predicciones fuera de muestra. La clasificación por menor error externo es descriptiva: no reemplaza la evaluación del selector interno.

## Métricas e imágenes

| Métrica | Qué indica |
|---|---|
| **MAE [N]** | Error absoluto promedio; criterio principal. Menor es mejor. |
| RMSE [N] | Error con mayor penalización de desviaciones grandes. Menor es mejor. |
| MedAE [N] | Mediana del error absoluto; describe el error central. |
| R² fuera de muestra | Variación explicada respecto a la media del conjunto evaluado; puede ser negativo. No es un porcentaje de exactitud. |
| Mejora frente al baseline [%] | 100 × (MAE baseline − MAE modelo) / MAE baseline. Un valor negativo significa deterioro. |
| Tiempo [s] | Búsqueda, reajuste y predicción medidos; depende del equipo. No equivale a latencia de una aplicación. |

El notebook incluye comparación de métricas, observado frente a predicho, residuales, errores por familia, diferencias por partición y resultados de la búsqueda. Los modelos que no mejoren también permanecen en la comparación.

## Estructura del proyecto

```text
notebooks/
  01_Ingesta_Curaduria_y_Calidad.ipynb
  02_EDA.ipynb
  03_Transformaciones_Exploratorias.ipynb
  04_Feature_Engineering_Pantograficas.ipynb  # Experimento previo, conservado
  05_Modelos_Ajustados.ipynb                 # Entrega de esta rama
src/
  feature_engineering.py   # Contrato geométrico y pipeline compartido
  ajuste_modelos.py        # Búsqueda anidada y predicciones de evaluación
  verificar_ajuste.py      # Grupos, estadísticas train, hashes y métricas
  reporte_ajuste.py        # Gráficos y ejecución del notebook 05
config/
  ajuste_modelos.json      # Modelos, rangos, semillas y protocolo congelado
reports/
  feature_engineering/    # Referencia histórica: splits y resultados
  modelos_ajustados/
    metricas_resumen.csv  # Comparación común de errores y tiempos
    predicciones_oof.csv  # Fuerza observada y predicha fuera de muestra
    metricas_folds.csv    # Resultados y parámetros por grupo reservado
    busquedas.csv         # Todos los candidatos de la validación interna
    figures/              # Gráficos PNG y SVG
    auditoria_train.json  # Estadísticas aprendidas en entrenamiento
    particiones_internas.json
    manifest.json         # Fechas, versiones e integridad
    ejecucion.jsonl       # Registro de ejecución
    verificacion.json    # Comprobaciones independientes
    config_ejecutada.json  # Copia de la configuración ejecutada
    particiones_externas.json  # Copia exacta de los splits anteriores
    seleccion_interna.json    # Algoritmo elegido solo con train
    logs_ajustes/         # Cada fit, validación, aviso y timestamp
  semana_6/               # Informe anterior; esta entrega es el notebook
config/feature_engineering.json  # Especificación del baseline anterior
data/processed/preliminary/model_table.csv  # Snapshot fijo compartido
requirements-modelos-ajustados.txt          # Versiones de esta ejecución
tests/test_ajuste_modelos.py                # Pruebas de aislamiento
```

## Reproducir

Desde la raíz del repositorio, en un entorno virtual con Python 3.13:

```powershell
python -m pip install -r requirements-modelos-ajustados.txt
python -m src.ajuste_modelos --output reports/modelos_ajustados_repeticion
python -m src.verificar_ajuste --run-dir reports/modelos_ajustados_repeticion
python -m pytest tests/test_ajuste_modelos.py -q
```

Para volver a ejecutar las celdas y gráficos sobre los resultados publicados: `python -m src.reporte_ajuste --execute`.

La ejecución del entrenamiento exige una carpeta de salida nueva para conservar la evidencia anterior. El notebook permite revisar la ejecución publicada y activar una repetición independiente; ambos modos se identifican explícitamente.

**Alcance:** el EDA previo ya examinó este conjunto. La separación de entrenamiento y evaluación evita reutilizar el grupo reservado durante el ajuste, pero no convierte los resultados en una validación externa. La prueba por familias muestra el comportamiento ante arquitecturas ausentes del entrenamiento.
