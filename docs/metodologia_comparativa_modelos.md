# Diseño del benchmark de regresión para primera falla

**Pregunta.** Antes de ensayar una estructura, ¿con qué error puede estimarse la fuerza de la primera rotura identificada de una fibra o un pivote? La respuesta es `First_Failure_Load_N`, en newtons. Es una respuesta continua bajo tracción. Fuerza terminal, máximo global, desplazamiento, energías, tiempos, marcas de ciclo y número de eventos **no son predictores**. Los identificadores enlazan las predicciones con el caso y con la partición; nunca se ajusta un modelo con ellos.

## Qué se conserva y qué cambia

La base de esta rama es `ce322ecfcbd291982efb577b644d47356c87f4cd`. Se conserva la [tabla curada](../data/processed/preliminary/model_table.csv), el [diccionario experimental](diccionario_datos.md) y la [asignación de casos a las particiones](../reports/feature_engineering/particiones.json). El cálculo no corrige dimensiones disputadas ni imputa la respuesta ausente del caso 2. La tesis y MATLAB se recibieron de los investigadores; siguen pendientes las condiciones de publicación y aclaraciones sobre geometría, fuerza terminal y ciclos. Este sprint no añade documentos fuente al repositorio.

La [tabla de antecedentes](benchmark/antecedentes_metricas.csv) conserva por fila la fuente, el conjunto de entradas, el protocolo, la familia y si hubo ajuste. Se obtiene con `python -m src.antecedentes_benchmark`.

| Etapa | Antecedente recuperado | Uso aquí |
|---|---|---|
| EDA e ingeniería de características (04) | Ridge con A–D, predictor de media global y cinco familias fijas sobre A: Ridge, Elastic Net, SVR-RBF, Random Forest y proceso gaussiano. | Designar A como conjunto común; cotejar Ridge y SVR por predicción y escenario. |
| Ajuste (05) | Se añadieron Extra Trees, Gradient Boosting y un selector interno. Ridge con ajuste alcanzó MAE 9,707 N al reservar configuraciones. | Antecedente ajustado con otra búsqueda. El control reproducido de este sprint es **Ridge fijo** (13,247 N); no asignar 9,707 N al fijo. |
| Importancia (06) | Se contrastaron Ridge y Elastic Net, y luego cinco familias completas frente a selección de cuatro entradas **dentro** del entrenamiento; GP reducido 9,972 N frente a 10,126 N de la media por nivel en configuraciones, con peor RMSE. | El subconjunto prioritario cambiaba por partición y familia; no existe una lista global de cuatro variables ganadoras que se pueda reutilizar como dato independiente. |
| Nuevo sprint | SVR, XGBoost, CatBoost, mínimos cuadrados ordinarios, KNN y árbol de decisión. | Seis algoritmos obligatorios, más controles separados. SVR ya existía: la coincidencia es intencional y permite comprobar reproducibilidad. |

El encargo del docente, según la investigadora, es justificar la elección de algoritmos; la revisión anterior cuestionó presentar Ridge y Elastic Net como óptimos. No consta una lista de algoritmos prohibidos ni un rechazo experimental individual. Las categorías históricas y las seis actuales permanecen separadas.

## Características admisibles

La [tabla de variables](benchmark/features.csv) da columna, origen, fórmula, unidad, justificación y riesgo para las diez entradas del conjunto A. Todas son originales: número de celdas en Y; altura, base, longitud y volumen de fibras; altura, radio, número y volumen de pivotes; volumen CAD del espécimen. No se usan etiquetas posensayo ni SHAP para elegir entradas.

**Por qué A.** El [registro de decisiones](../reports/feature_engineering/tables/decisiones.csv) exigía una reducción de MAE de al menos 5 %, mejora en al menos 60 % de los grupos y ausencia de deterioro en RMSE y en transferencia. Al añadir los dos momentos de sección (B), el MAE nominal bajó solo 3,11 % y el familiar empeoró 0,016 N. B quedó en evaluación; C y D se descartaron en sus contrastes sucesivos. A es la representación común sustentada por esas reglas, no una afirmación de que su MAE sea siempre el mínimo. Las hipótesis de geometría se desarrollaron en esta misma campaña, por lo que la comparación sigue siendo exploratoria.

Se aprende la mediana de una entrada faltante y se detectan constantes solo con el entrenamiento de cada ajuste. El radio vale 0,5 mm en esta campaña. SVR y KNN requieren distancias comparables; la geometría se estandariza en el entrenamiento, sin normalizar la respuesta en N. Los árboles trabajan sobre las magnitudes numéricas. En OLS se diagnostican rango y redundancias **sin usar la respuesta externa**. Longitud y cantidad de pivotes se vinculan al nivel Y; los volúmenes de fibras y pivotes se complementan. Un coeficiente condicional inestable o una partición del árbol no son parámetros mecánicos identificados. El grado de colinealidad no autoriza sustituir OLS por Ridge.

## Unidad experimental y dominios de generalización

Hay 27 registros de geometría y 26 casos con carga observada. La unidad analítica es el caso asociado al ensayo de una estructura, no cada ciclo ni cada pico. Los pares 19/25 y 24/27 comparten la clave nominal, pero difieren en otras dimensiones; la [auditoría de repeticiones](../reports/eda_calidad/tables/09_configuraciones_repetidas.csv) no los acredita como réplicas de fabricación independientes.

| Escenario | Reserva externa | Pregunta |
|---|---|---|
| Principal: configuración | 24 grupos definidos por (número de celdas Y, altura de fibra, altura de pivote); 22 grupos de un caso y dos de dos. | Estimar otra combinación nominal de la campaña cuando los niveles celulares ya están representados. |
| Secundario: familia Y | Se retiran, por turno, todos los casos Y=4, 5 o 6 (8, 9 y 9). | Transferir a un nivel no visto. Omitir el nivel intermedio y omitir un extremo no presentan el mismo problema geométrico. |

Los seis algoritmos comparten las reservas externas exactas. Cada caso aparece una vez como evaluación por protocolo y procedimiento. No se mezclan los resultados de ambos escenarios. Se usarán MAE agrupando las predicciones externas por caso, RMSE, MedAE, R² sobre las 26 predicciones reunidas y sesgo firmado. No se calcula R² en un grupo de un solo caso. La DE y el IQR del MAE entre grupos describen heterogeneidad; los entrenamientos se solapan y tres niveles no dan una estimación precisa de incertidumbre poblacional. Los tiempos están en segundos en el equipo empleado y se interpretan junto al presupuesto de búsqueda.

## Controles y decisión

- **DummyRegressor de mediana:** referencia sin geometría, apropiada para pérdida absoluta. Se distingue del control de **media** del primer EDA.
- **Ridge histórico:** reproduce A con `alpha=1, solver=svd`. Es la referencia para el cambio de MAE; un cambio negativo favorece el procedimiento actual.
- **Media por nivel celular:** se estima con las respuestas de entrenamiento; si falta el nivel, se usa la media global de ese entrenamiento. Es un control físico-estadístico particularmente exigente entre configuraciones conocidas.

La fase inicial mide las seis recetas sin búsqueda. En la optimización se decide **dentro de cada entrenamiento** qué dos familias ajustables reciben presupuesto; OLS continúa como referencia sin hiperparámetros inventados. Los procedimientos de primer rango, segundo rango y selector pueden usar distintas familias de un pliegue a otro. La diferencia de tuning pertinente empareja cada caso con la receta inicial **de la misma familia que se seleccionó para ese caso**. El reajuste con toda la campaña proporciona una receta para una predicción futura, no una evaluación nueva. Véase [hiperparámetros y validación](hiperparametros_y_validacion.md).

La elección se presenta como recomendación para el escenario nominal y se contrasta, por separado, con la transferencia. Se considera la variabilidad por grupo, el error de casos particulares, la interpretación, el coste y la media por nivel. Una mejora pequeña sin umbral mecánico de utilidad no establece superioridad general. La verificación final requiere identificar probetas/lotes y evaluar en otra campaña con el objetivo y las reglas de decisión fijados de antemano.
