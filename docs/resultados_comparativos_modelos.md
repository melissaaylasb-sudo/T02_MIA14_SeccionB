# Resultados comparativos de los seis regresores

**Para configuraciones de los tres niveles conocidos**, la recomendación entre regresores es **Regresión lineal múltiple**: MAE 9,66 N, RMSE 12,31 N y DE del MAE entre grupos 7,91 N. La referencia de media por nivel obtiene 10,13 N. La regla previa deja **Media por nivel Y** como opción operativa para ese dominio.

Entre las seis recetas fijas, Regresión lineal múltiple pertenece a la banda de 1,00 N y tiene prioridad de parsimonia. El selector ajustado no satisface conjuntamente los cuatro criterios frente a esa receta. El procedimiento principal cambia el MAE frente a la media por nivel en -0,462 N y gana en 12/24 configuraciones. Al suprimir un grupo por turno del agregado, el intervalo descriptivo de esa diferencia es [-0,731, -0,129] N.

## Tabla integrada según la unidad reservada

MAE, RMSE y MedAE en N; R² agrupado fuera de muestra. ΔMAE = procedimiento − Ridge fijo en los mismos casos; DE = desviación estándar de los MAE de los grupos externos. Los rangos son estrategias dependientes del entrenamiento. Se informan tiempos de ajuste y predicción; el coste de búsqueda se consulta por separado en el registro del experimento.

### Configuraciones nominales

| Procedimiento | Grupo | Configuración | Entradas | Optim. | MAE | RMSE | MedAE | R² | ΔMAE | DE | Ajuste (s) | Pred. (s) | Búsqueda (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ridge histórico | Control | Referencia | A | No | 13,25 | 16,35 | 11,94 | 0,923 | 0,00 | 9,68 | 0,1660 | 0,0686 | 0,000 |
| Mediana de entrenamiento | Control | Referencia | Ninguna | No | 49,75 | 64,70 | 35,89 | -0,213 | 36,50 | 41,05 | 0,0693 | 0,0293 | 0,000 |
| Media por nivel Y | Control | Referencia | Solo Y | No | 10,13 | 12,31 | 8,69 | 0,956 | -3,12 | 7,04 | 0,0709 | 0,0338 | 0,000 |
| SVR-RBF | A · SVR | Inicial | A | No | 15,65 | 19,10 | 12,63 | 0,894 | 2,40 | 11,12 | 0,1639 | 0,0686 | 0,000 |
| XGBoost | A · boosting | Inicial | A | No | 11,75 | 15,09 | 9,59 | 0,934 | -1,49 | 9,95 | 0,4427 | 0,0781 | 0,000 |
| CatBoost | A · boosting | Inicial | A | No | 12,15 | 15,60 | 10,51 | 0,930 | -1,10 | 10,24 | 0,8558 | 0,0956 | 0,000 |
| Regresión lineal múltiple | B · lineal | Inicial | A | No | 9,66 | 12,31 | 9,14 | 0,956 | -3,58 | 7,91 | 0,1876 | 0,0725 | 0,000 |
| K vecinos | B · distancias | Inicial | A | No | 12,43 | 14,92 | 11,52 | 0,935 | -0,82 | 8,10 | 0,1632 | 0,0985 | 0,000 |
| Árbol de decisión | B · árbol | Inicial | A | No | 10,96 | 14,89 | 8,58 | 0,936 | -2,29 | 10,64 | 0,1635 | 0,0666 | 0,000 |
| Primer rango ajustado | Selección interna | Anidada | A | Sí | 11,26 | 14,64 | 8,76 | 0,938 | -1,98 | 9,82 | 0,2054 | 0,0759 | 32,598 |
| Segundo rango ajustado | Selección interna | Anidada | A | Sí | 13,75 | 18,85 | 9,70 | 0,897 | 0,50 | 12,56 | 0,2738 | 0,0842 | 32,598 |
| Selector interno ajustado | Selección interna | Anidada | A | Sí | 13,00 | 17,18 | 9,34 | 0,914 | -0,25 | 10,52 | 0,2015 | 0,0759 | 32,598 |

### Nivel Y completo

| Procedimiento | Grupo | Configuración | Entradas | Optim. | MAE | RMSE | MedAE | R² | ΔMAE | DE | Ajuste (s) | Pred. (s) | Búsqueda (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ridge histórico | Control | Referencia | A | No | 57,64 | 60,43 | 64,52 | -0,058 | 0,00 | 17,14 | 0,0210 | 0,0092 | 0,000 |
| Mediana de entrenamiento | Control | Referencia | Ninguna | No | 97,51 | 99,01 | 96,38 | -1,840 | 39,87 | 16,23 | 0,0091 | 0,0040 | 0,000 |
| Media por nivel Y | Control | Referencia | Solo Y | No | 80,59 | 87,52 | 84,39 | -1,219 | 22,96 | 38,87 | 0,0090 | 0,0046 | 0,000 |
| SVR-RBF | A · SVR | Inicial | A | No | 82,36 | 88,04 | 87,60 | -1,246 | 24,72 | 33,87 | 0,0224 | 0,0088 | 0,000 |
| XGBoost | A · boosting | Inicial | A | No | 60,68 | 71,89 | 44,81 | -0,497 | 3,05 | 44,64 | 0,0373 | 0,0096 | 0,000 |
| CatBoost | A · boosting | Inicial | A | No | 74,73 | 82,25 | 90,31 | -0,960 | 17,09 | 34,09 | 0,0841 | 0,0123 | 0,000 |
| Regresión lineal múltiple | B · lineal | Inicial | A | No | 40,91 | 46,34 | 35,74 | 0,378 | -16,72 | 13,56 | 0,0322 | 0,0112 | 0,000 |
| K vecinos | B · distancias | Inicial | A | No | 64,85 | 76,46 | 58,66 | -0,694 | 7,21 | 41,24 | 0,0195 | 0,0097 | 0,000 |
| Árbol de decisión | B · árbol | Inicial | A | No | 61,29 | 71,37 | 46,64 | -0,476 | 3,65 | 42,25 | 0,0204 | 0,0086 | 0,000 |
| Primer rango ajustado | Selección interna | Anidada | A | Sí | 81,11 | 87,84 | 88,77 | -1,235 | 23,47 | 37,85 | 0,0206 | 0,0089 | 1,865 |
| Segundo rango ajustado | Selección interna | Anidada | A | Sí | 73,89 | 80,82 | 71,69 | -0,892 | 16,25 | 31,64 | 0,0573 | 0,0108 | 1,865 |
| Selector interno ajustado | Selección interna | Anidada | A | Sí | 81,11 | 87,84 | 88,77 | -1,235 | 23,47 | 37,85 | 0,0206 | 0,0089 | 1,865 |

En configuraciones, 6 de los seis superan la mediana sin geometría y 5 superan Ridge fijo: Regresión lineal múltiple, K vecinos, Árbol de decisión, XGBoost, CatBoost. El Ridge ajustado histórico sobre A obtuvo 9,707 N: el cambio del recomendado frente a ese antecedente es -0,043 N. Las particiones son comparables, aunque la búsqueda histórica y su presupuesto son diferentes. No se interpreta una diferencia tan pequeña como un avance confirmado.

## Qué cambia según la familia de regresión

El modelo lineal explica parte de la variación nominal (R² 0,956, MAE 9,66 N). El mejor MAE inicial es el de Regresión lineal múltiple (9,66 N). La diferencia de cada modelo contra las referencias se conserva en los [contrastes emparejados](../reports/benchmark_regresion/interpretacion/comparaciones_pareadas.csv). El rango deficiente impide interpretar cada pendiente de OLS como efecto mecánico independiente.

OLS conserva 7 de 9 columnas no constantes en los entrenamientos nominales; la condición de la base estandarizada va de 216 a 315. La pendiente de altura de fibra varía de -9,54 a 12,10 N/mm entre reservas. El cambio de signo ilustra por qué predecir bien aquí no identifica el efecto aislado de aumentar la altura. Véanse todos los [coeficientes y estados](../reports/benchmark_regresion/tables/coeficientes_ols.csv).

KNN obtiene 12,43 N frente a 49,75 N de mediana sin geometría y 10,13 N de media por nivel. El 98,7 % de las asignaciones de los tres vecinos corresponde al mismo nivel Y del caso reservado. La proximidad es informativa frente al control ingenuo, pero parte de esa información ya está en el nivel. Al pasar a un nivel no visto su MAE es 64,85 N. Un promedio de vecinos no proporciona una ley de extrapolación.

SVR-RBF alcanza MAE 15,65 N, con DE entre grupos 11,12 N frente a 7,91 N de Regresión lineal múltiple, el inicial con menor DE. Esa dispersión describe las configuraciones observadas; no es una banda de confianza del predictor.

Con ε=5 N, la fracción de casos de entrenamiento que son vectores de soporte varía entre 72 % y 88 %. La receta depende de muchas observaciones y su desempeño es heterogéneo; no se considera preferente frente a OLS para esta decisión.

Entre árboles, Árbol de decisión presenta el menor MAE inicial (10,96 N). XGBoost y CatBoost arrojan 11,75 y 12,15 N; el árbol sencillo, 10,96 N. El antecedente bibliográfico justifica ensayarlos, pero no establece que ensambles más costosos aporten valor en esta campaña.

Las siguientes brechas se calculan con igual peso por grupo exterior (MAE en validación − MAE en entrenamiento). Una brecha positiva es compatible con sobreajuste o con cambio de soporte, especialmente en la prueba por nivel.

| Escenario | Árbol (N) | XGBoost (N) | CatBoost (N) |
|---|---|---|---|
| Configuraciones nominales | 3,25 | 4,72 | 3,72 |
| Nivel Y completo | 52,98 | 53,13 | 66,26 |

## Estabilidad y observaciones influyentes

Mayor error de cada familia en la prueba nominal. Un residual grande describe un caso difícil para el predictor; no justifica excluirlo ni corregir su etiqueta. El [archivo por caso](../reports/benchmark_regresion/interpretacion/casos_mayor_error.csv) conserva los tres mayores de cada familia y escenario.

| Familia | Caso | Observado (N) | Predicho (N) | Error absoluto (N) |
|---|---|---|---|---|
| Árbol de decisión | 1 | 47,55 | 98,09 | 50,54 |
| CatBoost | 1 | 47,55 | 94,79 | 47,24 |
| K vecinos | 19 | 208,74 | 179,81 | 28,93 |
| Regresión lineal múltiple | 1 | 47,55 | 77,83 | 30,28 |
| SVR-RBF | 1 | 47,55 | 99,21 | 51,66 |
| XGBoost | 1 | 47,55 | 94,69 | 47,14 |

## Qué aportó el tuning

Cambio externo frente a la configuración inicial de la misma familia que recibió cada observación. Valores negativos indican reducción del MAE. El primer o segundo rango puede representar algoritmos diferentes en pliegues distintos.

| Escenario / ocasiones que recibió tuning | SVR-RBF | XGBoost | CatBoost | K vecinos | Árbol de decisión |
|---|---|---|---|---|---|
| Configuraciones nominales | 0 | 17 | 2 | 10 | 19 |
| Nivel Y completo | 3 | 0 | 2 | 1 | 0 |

Sobre toda la campaña nominal, el cribado posterior selecciona Árbol de decisión, XGBoost; el selector elige Árbol de decisión con la receta Arbol_decision_00. Es el reajuste entre las familias elegibles para tuning; OLS no participa en esa búsqueda. Al evaluar el procedimiento en configuraciones reservadas, el tuning del selector cambia su MAE en 0,633 N frente a sus mismas familias iniciales.

| Escenario | Primer rango (N) | Segundo rango (N) | Selector (N) |
|---|---|---|---|
| Configuraciones nominales | 0,167 | 0,393 | 0,633 |
| Nivel Y completo | -1,252 | -2,306 | -1,252 |

La tabla siguiente conserva los **nombres de las familias ajustadas** y enfrenta su receta inicial y ajustada en exactamente los casos donde el cribado interno las seleccionó. Las coberturas difieren: estas filas no sirven para clasificar algoritmos entre sí. La comparación completa, con todas las reservas, es la tabla de procedimientos anterior.

| Escenario | Familia ajustada | Grupos | Casos | MAE inicial (N) | MAE ajustado (N) | ΔMAE (N) |
|---|---|---|---|---|---|---|
| Configuraciones nominales | Árbol de decisión | 19 | 20 | 12,68 | 13,79 | 1,102 |
| Configuraciones nominales | CatBoost | 2 | 2 | 8,44 | 7,78 | -0,660 |
| Configuraciones nominales | K vecinos | 10 | 11 | 12,28 | 11,08 | -1,201 |
| Configuraciones nominales | XGBoost | 17 | 19 | 12,11 | 12,48 | 0,372 |
| Nivel Y completo | CatBoost | 2 | 17 | 57,99 | 55,14 | -2,855 |
| Nivel Y completo | K vecinos | 1 | 9 | 110,57 | 109,30 | -1,268 |
| Nivel Y completo | SVR-RBF | 3 | 26 | 82,36 | 81,11 | -1,252 |

## Decisión y límites

| Categoría (configuraciones) | Procedimiento | MAE (N) |
|---|---|---|
| Lineal (único candidato) | Regresión lineal múltiple | 9,664 |
| Vecinos (único candidato) | K vecinos | 12,429 |
| Menor MAE entre los tres modelos de árboles | Árbol de decisión | 10,956 |
| Menor MAE del grupo A | XGBoost | 11,754 |
| Menor MAE de los seis fijos | Regresión lineal múltiple | 9,664 |
| Menor DE entre los seis fijos | Regresión lineal múltiple | 9,664 |
| Menor MAE incluyendo procedimientos ajustados | Regresión lineal múltiple | 9,664 |
| Menor MAE incluyendo controles | Regresión lineal múltiple | 9,664 |
| Compromiso recomendado entre regresores | Regresión lineal múltiple | 9,664 |

Receta recomendada entre regresores: **Regresión lineal múltiple**, parámetros: `{"fit_intercept": true}`. Entradas A, preprocesamiento y semillas del [protocolo congelado](../config/benchmark_regresion.json). Respaldos descriptivos: Árbol de decisión (MAE 10,96 N; DE 10,64 N), XGBoost (MAE 11,75 N; DE 9,95 N).

El procedimiento principal alcanza MAE 40,91 N y R² 0,378 al retirar un nivel completo. Entre los seis fijos, Regresión lineal múltiple registra el menor MAE de transferencia (40,91 N). Solo hay tres reservas de este tipo. En una arquitectura ausente la referencia por nivel recurre a la media del entrenamiento; el criterio operativo para niveles observados no constituye una recomendación de diseño para nuevos niveles.

El EDA ya mostraba concentración de la variación de primera falla entre tres niveles y descriptores geométricos relacionados. Las predicciones son compatibles con agrupación por arquitectura, sin identificar el efecto causal del número de celdas. Los rangos externos se describen en una sola campaña ya explorada; no suministran una evaluación independiente de la familia que se acaba de recomendar. El paso siguiente es fijar el uso y un error aceptable con los investigadores, verificar identidades, geometrías y lotes, y contrastar la receta sin nuevos ajustes en probetas o campañas independientes.

[Notebook ejecutado](../notebooks/Comparativa_6_Modelos_Regresion_Pantograficas.ipynb) · [Tabla integrada con parámetros y costes](../reports/benchmark_regresion/interpretacion/tabla_final_integrada.csv) · [Auditoría independiente de datos y selección](../reports/benchmark_regresion/auditoria/revision_independiente.json) · [Fundamento bibliográfico](justificacion_seleccion_modelos.md) · [Antecedentes con su protocolo](benchmark/antecedentes_metricas.csv)
