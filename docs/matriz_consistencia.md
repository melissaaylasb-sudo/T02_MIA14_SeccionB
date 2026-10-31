# Matriz de consistencia metodológica

## Procedencia y alcance

Los objetivos e hipótesis se recuperan del [planteamiento del informe de Semana 6](../reports/semana_6/secciones/planteamiento.tex), documento versionado de este proyecto. Las preguntas específicas de la tabla son formulaciones operativas derivadas de esos objetivos. No consta en el repositorio una aprobación formal de esa formulación ni un preregistro. El presente sprint añade tuning e interpretabilidad como **extensión analítica**, sin sustituir los objetivos e hipótesis anteriores para acomodarlos a resultados observados.

La unidad de análisis es el caso experimental de una estructura pantográfica. La variable respuesta física es `First_Failure_Load_N`, fuerza de la primera rotura identificada de fibra o pivote, en **N**. Las entradas físicas son geometría y arquitectura disponibles antes del ensayo. MAE, RMSE, R², SHAP y permutación son **indicadores analíticos**, no variables físicas dependientes.

## Matriz principal de consistencia

Los **objetivos e hipótesis** de las cinco filas siguientes conservan la formulación de [Semana 6](../reports/semana_6/secciones/planteamiento.tex). Las preguntas específicas son su **operacionalización para esta tabla**; no se presentan como citas literales ni como nuevos objetivos aprobados. El objetivo específico 3 se desdobla en dos contrastes para alinear **H1 con A→B** y **H2 con B→C**. Construir características (objetivo 2) no prueba por sí mismo una mejora predictiva.

| Problema o pregunta | Objetivo | Hipótesis | Variables independientes y dependiente | Indicadores de contraste |
|---|---|---|---|---|
| **General.** ¿En qué medida los momentos de sección y la razón logarítmica de volúmenes reducen el error de predicción frente a las entradas originales, manteniendo el mismo modelo y validación? | Evaluar el aporte de características geométricas derivadas a la predicción de la carga de primera falla mediante una línea base y dos variantes de datos. | No se formuló una hipótesis general explícita. H1 y H2 son específicas. | **VI:** geometría de fibras, pivotes y malla (A); dos momentos (B); razón logarítmica (C). **VD:** `First_Failure_Load_N` [N]. **Factor metodológico:** representación A/B/C. | MAE OOF [N] principal; RMSE y MedAE [N]; R² OOF; ΔMAE [N] por configuración y familia. |
| **Específico 1.** ¿Qué calidad, distribución y asociaciones presentan las entradas geométricas y la respuesta? | Caracterizar calidad, distribución y asociaciones de las variables para justificar su tratamiento. | No se formuló hipótesis para este objetivo descriptivo. | **VI:** diez medidas geométricas previas al ensayo. **VD:** carga de primera falla [N] para asociaciones; otras salidas e identificador excluidos de X. | Cobertura, ausencias, rangos, correlaciones y redundancia; [EDA](informe_eda.md). |
| **Específico 2.** ¿Pueden construirse los descriptores derivados sin utilizar datos obtenidos después del ensayo? | Construir momentos de sección y razón de volúmenes sin utilizar respuestas posteriores al ensayo. | No se formuló hipótesis para la construcción determinista. | **VI:** base y altura de fibra [mm], volúmenes de pivotes y fibras [mm³]. **VD:** carga de primera falla [N], excluida de las fórmulas. | `b·h³/12`, `h·b³/12` [mm⁴], `ln(Vp/Vf)` [1]; auditoría de insumos y fuga temporal. |
| **Específico 3a.** ¿Mejora B al menos 5 % el MAE de A por configuración? | Comparar A, B y C mediante errores fuera de muestra por configuración y familia. | **H1:** añadir los momentos (B) reduce el MAE por configuración al menos **5 %** frente a la geometría original (A). | **VI:** representación A frente a B = A + dos momentos; mismos modelo, parámetros y particiones. **VD:** carga de primera falla [N]. | `100·(MAE_A−MAE_B)/MAE_A` [%], MAE pareado B−A [N] y estabilidad por familia. |
| **Específico 3b.** ¿Mejora C el MAE de B? | Mismo objetivo específico 3; la subdivisión no modifica su texto. | **H2:** añadir la razón logarítmica (C) reduce el MAE respecto de B. | **VI:** B frente a C = B + `ln(Vp/Vf)`; mismos modelo, parámetros y particiones. **VD:** carga de primera falla [N]. | MAE pareado C−B [N]; MAE y RMSE OOF [N] por configuración y familia. |

La fuente LaTeX escribe `5\%`; equivale a **5 %**. H1 y H2 tratan de **error predictivo bajo un protocolo fijado**, no de un efecto causal físico de alterar un descriptor. Los momentos [mm⁴] expresan geometría ideal de sección; no son una medición de rigidez `EI`.

## Extensión metodológica de este sprint

Esta ampliación conserva la pregunta y los objetivos previos. Sus preguntas analíticas no se etiquetan como hipótesis formales retrospectivas.

| Pregunta de extensión | Método | VI/factor comparativo y VD | Indicadores | Estado |
|---|---|---|---|---|
| ¿Cuánto aportan separadamente B y el tuning? | E0–E4 Ridge y S1–S4 Elastic Net; mismas particiones externas, búsqueda interna por grupos. | **Factor metodológico:** A/B y parámetros fijos/ajustados. **VD física:** carga de primera falla [N]. | ΔMAE E2−E1, E3−E2 y E3−E4 [N]; RMSE y variación por fold. | [Ejecutado](resultados_comparativos.md); A ajustado superó B ajustado por configuración. |
| ¿Qué atributos usa el modelo y cuán estables son sus contribuciones? | SHAP OOF con background de train; permutación y ablación de momentos. | **VI:** geometría original/derivada. **VD física:** carga de primera falla [N]. | `mean(|SHAP|)` [N], reconstrucción base+SHAP [N], ΔMAE [N] por permutación/ablación y estabilidad entre folds. | [Ejecutado](importancia_variables.md); las atribuciones no demuestran causalidad. |

## Estado de las hipótesis con la evidencia anterior

La comparación previa utilizó Ridge fijo y las mismas particiones por configuración. A registró MAE **13.2466 N**, B **12.8346 N** y C **12.9139 N** ([resultados FE](experimentos_fe.md); [tabla numérica](../reports/feature_engineering/tables/comparacion_experimentos.csv)). B mejoró **3.11 %**, por debajo del umbral de H1; C empeoró **0.0792 N** respecto de B, contrario a H2. Al excluir familias completas, A y B obtuvieron aproximadamente **57.64 N** de MAE. Estos resultados no confirman H1 ni H2 bajo el protocolo previo. El EDA y las decisiones de FE se basaron en el mismo conjunto de 26 casos, por lo que la selección retrospectiva de B puede ser optimista. Tampoco se dispone de una campaña experimental externa.

La [ejecución de este sprint](../reports/feature_importance/tables/metricas_resumen.csv) reproduce A/B con Ridge fijo y añade Elastic Net fijo: en esta segunda familia, B reduce MAE por configuración de **13.9792 a 13.4416 N** (aproximadamente **3.85 %**, también por debajo de H1). Tras tuning interno, A supera B en ambas familias al excluir configuraciones (Ridge **9.7070 vs 10.9814 N**; Elastic Net **10.1804 vs 10.8724 N**). Al excluir familias Y, todos los R² OOF son negativos. La extensión no modifica retrospectivamente el umbral ni confirma H1/H2. C no se reevalúa aquí porque el conjunto B provisional excluye la razón que la define; H2 conserva el contraste anterior y requiere prueba independiente para generalizar.

## Controles de coherencia

- El diccionario fuente identifica diez entradas originales, dos momentos derivados y una razón logarítmica; `Pivot_radius` es constante en la muestra y se filtra dentro de entrenamiento. Véase el [diccionario de características](diccionario_features.md).
- `Case_n`, `Ultimate_Load`, desplazamientos residuales, energías de ensayo y eventos observados no integran la VI predictiva. Véase el [diccionario científico](diccionario_datos.md).
- El protocolo principal agrupa por configuración nominal; el análisis por familia Y evalúa extrapolación más exigente. Una familia no equivale a una campaña experimental independiente. Véase el [protocolo FE](feature_engineering.md).
- Los momentos [mm⁴] describen la distribución ideal del área de una sección rectangular; no miden rigidez EI ni determinan la primera fractura. La razón de volúmenes es adimensional.
- La separación train/test controla la fuga de procesamiento; no elimina el sesgo potencial de hipótesis desarrolladas tras examinar el conjunto completo.
- La interpretación mecánica se expresa como hipótesis plausible, sujeta a nuevas mediciones, conciliación geométrica y ensayos independientes.
