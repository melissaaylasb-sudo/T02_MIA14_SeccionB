# Matriz de consistencia metodológica

## Procedencia y alcance

La formulación que sigue procede literalmente del [planteamiento del informe de Semana 6](../reports/semana_6/secciones/planteamiento.tex), documento versionado de este proyecto. No consta en el repositorio una constancia de aprobación formal de esa formulación ni un preregistro. El presente sprint añade tuning e interpretabilidad como **extensión analítica**, sin sustituir los objetivos e hipótesis anteriores para acomodarlos a resultados observados.

La unidad de análisis es el caso experimental de una estructura pantográfica. La variable respuesta física es `First_Failure_Load_N`, fuerza de la primera rotura identificada de fibra o pivote, en **N**. Las entradas físicas son geometría y arquitectura disponibles antes del ensayo. MAE, RMSE, R², SHAP y permutación son **indicadores analíticos**, no variables físicas dependientes.

## Formulación previa conservada

| Problema | Objetivos | Hipótesis | Variables independientes y dependientes | Indicadores |
|---|---|---|---|---|
| **Problema general:** «Las dimensiones originales no representan explícitamente todas las combinaciones geométricas de fibras y pivotes. Se desconoce si incorporar esas combinaciones mejora la estimación de la carga de primera falla y si la mejora se mantiene al excluir una familia del entrenamiento». **Pregunta:** «¿En qué medida los momentos de sección y la razón logarítmica de volúmenes reducen el error de predicción frente a las entradas originales, manteniendo el mismo modelo y validación?» | **General:** «evaluar el aporte de características geométricas derivadas a la predicción de la carga de primera falla mediante una línea base y dos variantes de datos». | No se formuló una hipótesis general explícita en la fuente. H1 y H2 son hipótesis específicas de trabajo. | **VI físicas:** geometría de fibras, pivotes y malla; momentos de sección y razón de volúmenes derivados de ella. **VD física:** carga de primera falla [N]. **Factor comparativo:** representación A, B o C. | MAE fuera de entrenamiento [N] como indicador principal; RMSE y MedAE [N], R² OOF, diferencia de MAE, variación por configuración y familia. |
| **Problema específico 1, derivado del objetivo previo:** calidad, distribución, asociaciones y redundancia de las entradas y la respuesta. | «Caracterizar calidad, distribución y asociaciones de las variables para justificar su tratamiento». | No se formuló hipótesis específica para la caracterización. | VI: dimensiones, volúmenes y conteos previos. VD descriptiva: carga de primera falla [N]. Los identificadores y otras respuestas no son predictores. | Cobertura, ausencias, rango, asociaciones y dependencias algebraicas; evidencia EDA. |
| **Problema específico 2, derivado del objetivo previo:** representación de combinaciones geométricas sin usar respuestas posteriores. | «Construir momentos de sección y razón de volúmenes sin utilizar respuestas posteriores al ensayo». | **H1:** «añadir los momentos (B) reduce el MAE por configuración al menos 5\% frente a la geometría original (A)». | VI: A; B = A + `I_fibra_bh3_mm4` y `I_fibra_hb3_mm4`; VD: carga de primera falla [N]. La transformación de entradas es una intervención de representación, no una intervención física. | Fórmulas b·h³/12 y h·b³/12 [mm⁴], ausencia de variables posteriores, MAE pareado B−A [N] y reducción porcentual. |
| **Problema específico 3, derivado del objetivo previo:** desempeño incremental y transferencia entre configuraciones/familias. | «Comparar A, B y C mediante errores fuera de muestra por configuración y familia». | **H2:** «añadir la razón logarítmica (C) reduce el MAE respecto de B». | VI: B y C = B + ln(Vp/Vf), adimensional; VD: carga de primera falla [N]. Familia de celdas Y es factor de agrupación de validación. | MAE C−B [N], MAE y RMSE por configuración y familia, estabilidad de errores; condición de Vp,Vf > 0. |

La grafía `5\%` de H1 es la del archivo LaTeX. En lectura ordinaria significa **5 %**. Las hipótesis son contrastables por desempeño predictivo bajo el protocolo fijado; no afirman que un descriptor cause físicamente la fractura.

## Estado de las hipótesis con la evidencia anterior

La comparación previa utilizó Ridge fijo y las mismas particiones por configuración. A registró MAE **13.2466 N**, B **12.8346 N** y C **12.9139 N** ([resultados FE](experimentos_fe.md); [tabla numérica](../reports/feature_engineering/tables/comparacion_experimentos.csv)). B mejoró **3.11 %**, por debajo del umbral de H1; C empeoró **0.0792 N** respecto de B, contrario a H2. Al excluir familias completas, A y B obtuvieron aproximadamente **57.64 N** de MAE. Estos resultados no confirman H1 ni H2 bajo el protocolo previo. El EDA y las decisiones de FE se basaron en el mismo conjunto de 26 casos, por lo que la selección retrospectiva de B puede ser optimista. Tampoco se dispone de una campaña experimental externa.

La [ejecución de este sprint](../reports/feature_importance/tables/metricas_resumen.csv) reproduce A/B con Ridge fijo y añade Elastic Net fijo: en esta segunda familia, B reduce MAE por configuración de **13.9792 a 13.4416 N** (aproximadamente **3.85 %**, también por debajo de H1). Tras tuning interno, A supera B en ambas familias al excluir configuraciones (Ridge **9.7070 vs 10.9814 N**; Elastic Net **10.1804 vs 10.8724 N**). Al excluir familias Y, todos los R² OOF son negativos. La extensión no modifica retrospectivamente el umbral ni confirma H1/H2. C no se reevalúa aquí porque el conjunto B provisional excluye la razón que la define; H2 conserva el contraste anterior y requiere prueba independiente para generalizar.

## Extensión analítica de este sprint

El nuevo análisis compara E0 (referencia ingenua), E1 (A con parámetros de referencia), E2 (B con la misma familia, parámetros y particiones), E3 (B con tuning interno) y, cuando sea viable, E4 (A con el mismo tuning). SHAP, permutación y ablación evalúan propiedades distintas del modelo y de su capacidad predictiva; ninguna identifica por sí sola un mecanismo causal. La extensión responde estas preguntas de trabajo:

1. ¿Cuánto cambia el MAE al pasar de A a B con modelo y parámetros fijos?
2. ¿Cuánto cambia adicionalmente al ajustar hiperparámetros exclusivamente en entrenamiento?
3. ¿Qué contribuciones predictivas aparecen en modelos validados y cuán estables son entre folds y familias?
4. ¿Coinciden las atribuciones SHAP con permutación y ablaciones bajo colinealidad geométrica?

Estas preguntas son **propuestas de ampliación**, no reformulaciones retrospectivas de H1 y H2. La interpretación mecánica se expresa como hipótesis plausible sujeta a nuevas mediciones, conciliación geométrica y ensayos independientes.

## Controles de coherencia

- El diccionario fuente identifica diez entradas originales, dos momentos derivados y una razón logarítmica; `Pivot_radius` es constante en la muestra y se filtra dentro de entrenamiento. Véase el [diccionario de características](diccionario_features.md).
- `Case_n`, `Ultimate_Load`, desplazamientos residuales, energías de ensayo y eventos observados no integran la VI predictiva. Véase el [diccionario científico](diccionario_datos.md).
- El protocolo principal agrupa por configuración nominal; el análisis por familia Y evalúa extrapolación más exigente. Una familia no equivale a una campaña experimental independiente. Véase el [protocolo FE](feature_engineering.md).
- Los momentos [mm⁴] describen la distribución ideal del área de una sección rectangular; no miden rigidez EI ni determinan la primera fractura. La razón de volúmenes es adimensional.
- La separación train/test controla la fuga de procesamiento; no elimina el sesgo potencial de hipótesis desarrolladas tras examinar el conjunto completo.
