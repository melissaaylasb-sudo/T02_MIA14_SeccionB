# Protocolo de ingeniería de características

**Entrega principal:** [notebook 04 ejecutado](../notebooks/04_Feature_Engineering_Pantograficas.ipynb). [Matriz EDA → FE completa](../reports/feature_engineering/tables/matriz_eda_fe.csv).

| Elemento | Decisión |
|---|---|
| Fuente | Tabla curada `model_table.csv`, contrastada con su manifiesto; target observado, sin imputación. |
| Objetivo | Carga de primera falla [N]: fuerza correspondiente a la primera rotura de fibra o pivote. |
| Entradas | Geometría previa al ensayo; se excluyen identificador, fuerzas, desplazamientos y energías. |
| Experimentos | Línea base A: geometría original. Variante 1 B: añadir dos momentos como bloque. Variante 2 C: añadir razón logarítmica a B. Variante 3 D: retirar tres descriptores como bloque de C. |
| Modelo de los contrastes A–D | Ridge fijo; mismos parámetros y particiones. Dummy de media como control auxiliar. |
| Comparación de algoritmos | Ridge, Elastic Net, SVR con núcleo RBF, bosque aleatorio y proceso gaussiano; todos con la geometría original A y las mismas particiones. |
| Pipeline | Contrato de entradas → imputación con mediana → representación geométrica → filtro de varianza nula → estandarización → regresor; estadísticas aprendidas solo en entrenamiento. |
| Validación principal | Validación cruzada LeaveOneGroupOut por combinación de cantidad de celdas en Y, altura de fibra y altura de pivote; configuraciones nominales repetidas permanecen juntas. |
| Comprobación adicional | Excluir cada familia Y completa; evalúa una arquitectura no vista. |
| Métricas | MAE principal; RMSE, MedAE y R² agregado OOF. Sin R² en folds individuales. |
| Tiempo | Suma de los tiempos medidos de ajuste y predicción de los folds. Excluye generación de figuras y escritura de archivos; depende del equipo. |
| Selección | Retirada fija por hipótesis geométrica; varianza nula aprendida en train. Sin selección supervisada ni tuning. |
| Decisión | Adoptar provisionalmente si MAE mejora ≥5 %, gana en ≥60 % de grupos y no empeora RMSE ni MAE agregado al excluir familias completas; si solo mejora MAE, mantener en evaluación; si no, descartar. |

Las hipótesis proceden del EDA del mismo conjunto: los resultados son exploratorios y pueden tener sesgo de selección. Faltan réplicas/lotes verificados y persisten discrepancias geométricas. Las familias no representan campañas independientes.

Las nuevas características son funciones de mediciones existentes. Retirar columnas modifica también la penalización efectiva de Ridge. No se recortan predicciones negativas ni se eliminan extremos. La sensibilidad por omisión modifica el agregado de errores OOF, no reentrena modelos.

## Técnicas utilizadas

| Técnica | Aplicación |
|---|---|
| Construcción de variables físicas | Dos momentos de área de sección rectangular y una razón logarítmica de volúmenes; fórmulas por fila, sin utilizar la respuesta. |
| Imputación por mediana | Contingencia aprendida en cada entrenamiento. La geometría actual está completa: no se rellenaron mediciones. El objetivo nunca se imputa. |
| Filtro de varianza nula | Detecta y retira columnas constantes usando únicamente el entrenamiento; aquí corresponde al radio de pivote. |
| Estandarización (StandardScaler) | Resta la media y divide por la desviación del entrenamiento. La validación reutiliza esas estadísticas. |
| Reducción de variables | En D se retiran tres descriptores por hipótesis geométrica fija, sin seleccionar columnas por el objetivo. |
| One-hot encoding | No se utilizó: las entradas admitidas son mediciones o conteos numéricos. La cantidad de celdas en Y conserva su valor numérico; usarla para agrupar no la convierte en una categoría codificada. |

Cada contraste modifica **un bloque**: B−A, C−B y D−C. Los dos momentos se prueban juntos y las tres retiradas también; el diseño no permite atribuir la mejora a cada variable individual. El modelo y las particiones se mantienen fijos en estos contrastes.

La comparación de cinco regresores es un experimento separado: conserva la representación A, el preprocesamiento y las particiones, y cambia únicamente el algoritmo con sus parámetros declarados. No se combinan cinco modelos con cuatro variantes ni se presenta esta comparación como ajuste de hiperparámetros. Las reglas de adopción de características no seleccionan automáticamente un algoritmo.

## Controles de fuga y separación

Se comprueba que entrenamiento y validación no comparten casos ni grupos; las respuestas del ensayo y los identificadores no entran al modelo. La imputación, el filtro y la estandarización se ajustan dentro de cada entrenamiento. Esto controla la fuga de procesamiento, pero **no acredita ausencia absoluta de sesgo**: las hipótesis ya se exploraron con el EDA del conjunto completo y faltan lotes experimentales verificados. Se ejecutó validación cruzada por grupos, no un holdout externo ni una partición temporal o estratificada.

[Configuración de características](../config/feature_engineering.json) · [Configuración de modelos](../config/comparacion_modelos.json) · [Particiones](../reports/feature_engineering/particiones.json) · [Estadísticas aprendidas](../reports/feature_engineering/auditoria_transformaciones.json)
