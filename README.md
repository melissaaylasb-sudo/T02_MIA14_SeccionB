# Análisis predictivo e interpretable de la respuesta mecánica de estructuras pantográficas

Una estructura pantográfica es una red de barras delgadas o **fibras**, conectadas mediante pequeñas uniones llamadas **pivotes**. Sus espacios repetidos forman **celdas**. Se estudia como metamaterial mecánico porque la forma y organización de esa red contribuyen a su respuesta, además del material con el que se fabrica. Los especímenes de interés están hechos de **poliamida o nylon**, un polímero, mediante impresión 3D por **sinterización láser selectiva (SLS)**: un láser une selectivamente polvo del material, capa por capa.

En el ensayo se tira de la estructura y se registra la fuerza que resiste mientras se alarga. La tesis busca predecir **cuánta fuerza actúa cuando se rompe por primera vez una fibra o un pivote**: esa fuerza se llama **carga de primera falla**. Se utilizarán medidas geométricas y estructurales conocidas antes del ensayo. Es un problema de **regresión supervisada**: aprender de ejemplos con una fuerza medida para estimar un valor numérico en otros casos.


## Autora

**Melissa Dessire Aylas Barranca**
Maestría en Ciencias con mención en Inteligencia Artificial

Universidad Nacional de Ingeniería (UNI)

## Resumen del proyecto

| Elemento | Explicación para el lector |
|---|---|
| Objeto de estudio | Una red de fibras delgadas conectadas por pivotes; se analiza cómo su geometría influye en el inicio de la rotura |
| Material | Poliamida, también llamada nylon; constituye las fibras y uniones y se mantiene como contexto del experimento |
| Fabricación | Impresión 3D SLS: construcción de la pieza por capas a partir de polvo de polímero unido mediante láser |
| Tipo de ensayo | Tracción: una máquina sujeta la pieza y separa sus extremos para estirarla, registrando fuerza y desplazamiento |
| Tipo de datos | Una tabla: cada fila representa un caso ensayado y las columnas contienen sus medidas y la fuerza de primera rotura |
| Entradas del modelo | Geometría disponible antes del ensayo: celdas, dimensiones de fibras y pivotes y otros descriptores verificados |
| Problema de ML | Regresión supervisada: aprender una relación entre esas medidas y una fuerza observada, y estimar la fuerza en casos no usados para aprender |
| Variable objetivo | **Carga de primera falla: fuerza registrada cuando ocurre el primer daño localizado de rotura de una fibra o un pivote** |
| Unidad de salida | Newton (N), unidad de fuerza; no es una masa en kilogramos ni un esfuerzo por unidad de área |
| Finalidad predictiva | Estimar el inicio del daño de configuraciones comparables a las ensayadas y estudiar qué características ayudan a esa estimación |
| Cómo se comprobará | Comparar predicciones con fuerzas reales de casos reservados para evaluación, frente a una referencia simple |
| Métrica principal | MAE fuera de muestra: error absoluto medio expresado en la misma unidad que la carga |

## Estructura del repositorio

> **Esta estructura es parte central de la metodología de tesis.** Separa evidencia experimental, decisiones de curaduría, código de modelamiento y resultados para que cada predicción pueda rastrearse hasta su archivo de origen.

### Mapa rápido de responsabilidades

| Componente | Rol principal | Responde a la pregunta |
|---|---|---|
| `data/` | Custodia y versiona la evidencia experimental | ¿De qué datos proviene el análisis? |
| `config/` | Declara las decisiones metodológicas sin ocultarlas en el código | ¿Qué columnas, unidades, modelos y particiones se autorizaron? |
| `src/` | Contiene la implementación reutilizable y verificable | ¿Cómo se procesa, entrena y evalúa? |
| `notebooks/` | Explica y revisa cada etapa de la investigación | ¿Cómo puede el investigador inspeccionar el proceso? |
| `tests/` | Comprueba reglas críticas del flujo | ¿Cómo se verifica que no se mezclen grupos ni se produzca leakage? |
| `results/` | Conserva las salidas de cada experimento | ¿Qué predicciones, métricas y modelos produjo una ejecución? |
| `logs/` | Registra eventos, advertencias y fallos | ¿Qué ocurrió durante la ejecución? |
| `README.md` | Integra el problema físico, el método de ML y el uso del proyecto | ¿Qué hace el repositorio y cómo debe utilizarse? |

En términos simples: **`data/` contiene la evidencia, `config/` las decisiones, `src/` el procedimiento, `notebooks/` la explicación y `results/` los resultados reproducibles.**

La organización completa es la siguiente:

```text
T02_MIA14_SeccionB/
├── config/
│   ├── data_schema.yml
│   └── model_config.yml
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── interim/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── notebooks/
│   ├── 01_Ingesta_y_curaduria.ipynb
│   ├── EDA_basico.ipynb
│   ├── 03_Particiones_y_preprocesamiento.ipynb
│   ├── Baseline_basico.ipynb
│   ├── 05_Ajuste_y_validacion.ipynb
│   └── 06_Evaluacion_e_interpretabilidad.ipynb
├── src/
│   ├── __init__.py
│   ├── ingesta.py
│   ├── preprocesamiento.py
│   ├── modelo_baseline.py
│   ├── modelos.py
│   ├── particiones.py
│   ├── ajuste.py
│   ├── validacion.py
│   ├── interpretabilidad.py
│   └── experimento.py
├── tests/
│   └── test_workflow.py
├── results/
│   └── .gitkeep
├── logs/
│   └── .gitkeep
├── slides/
│   └── .gitkeep
├── .gitattributes
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

### Directorios de datos y resultados

| Ruta | Contenido | Regla metodológica |
|---|---|---|
| `data/raw/` | Archivos experimentales recibidos de la fuente autorizada | Son originales inmutables: el código no los modifica ni sobrescribe |
| `data/interim/` | Una carpeta por ejecución de ingesta, con tabla tabular, hash y manifiesto | Conserva la trazabilidad respecto del original; todavía no es la tabla de modelamiento |
| `data/processed/` | Tabla analítica curada y reporte de calidad, ambos versionados | Contiene únicamente columnas con roles explícitos; puede conservar faltantes de predictores |
| `results/` | Futuros folds, búsquedas internas, pipelines, predicciones OOF, métricas e interpretaciones | Se genera solo al ejecutar un experimento autorizado; no contiene resultados en este avance |
| `logs/` | Reservado para registros de ingesta, calidad y modelamiento | Actualmente está vacío; cuando se ejecuta el flujo documenta advertencias y fallos sin sustituir los manifiestos |
| `slides/` | Reservado para presentaciones del proyecto | Actualmente está vacío y se mantiene separado del código, los datos y los resultados |

El recorrido de los datos es:

```text
data/raw/  →  data/interim/VERSION/  →  data/processed/VERSION/  →  results/VERSION/
 original        ingesta trazable             tabla curada              evaluación ML
```

La imputación estadística, el escalamiento y la eliminación de constantes no se aplican globalmente en `data/processed/`. Se ajustan dentro del entrenamiento de cada fold mediante los pipelines definidos en `src/modelos.py`, lo cual evita que la información del conjunto de evaluación influya en el modelo.

### Notebooks de la investigación

| Orden | Notebook | Propósito |
|---:|---|---|
| 1 | `01_Ingesta_y_curaduria.ipynb` | Revisar procedencia, target, unidades, exclusiones y construcción de la tabla válida |
| 2 | `EDA_basico.ipynb` | Examinar distribuciones, faltantes, constantes, correlaciones y posibles anomalías |
| 3 | `03_Particiones_y_preprocesamiento.ipynb` | Justificar grupos, folds y transformaciones ajustadas exclusivamente con entrenamiento |
| 4 | `Baseline_basico.ipynb` | Definir la referencia de mediana sobre las mismas particiones externas |
| 5 | `05_Ajuste_y_validacion.ipynb` | Ejecutar búsqueda interna de hiperparámetros y evaluación externa cuando existan datos reales |
| 6 | `06_Evaluacion_e_interpretabilidad.ipynb` | Analizar métricas, residuos, estabilidad e interpretabilidad de una ejecución completa |

Los notebooks explican y documentan las decisiones de tesis. La lógica reutilizable se mantiene en `src/` para que el resultado no dependa de ejecutar celdas manualmente en un orden desconocido.

### Módulos de código científico

| Archivo | Responsabilidad |
|---|---|
| `src/ingesta.py` | Lee CSV/XLSX, valida encabezados, calcula SHA-256 y crea una versión intermedia sin alterar el original |
| `src/preprocesamiento.py` | Aplica curaduría determinística, valida roles y tipos y registra exclusiones; no imputa ni escala |
| `src/modelo_baseline.py` | Construye el `DummyRegressor` de mediana y calcula métricas de regresión |
| `src/modelos.py` | Define pipelines y familias candidatas con el preprocesamiento dentro de cada ajuste |
| `src/particiones.py` | Construye validación cruzada externa e interna y mantiene juntos los grupos dependientes |
| `src/ajuste.py` | Realiza el tuneo de hiperparámetros únicamente con los folds internos de entrenamiento |
| `src/validacion.py` | Compara candidatos y baseline en folds externos y conserva predicciones fuera de muestra |
| `src/interpretabilidad.py` | Extrae coeficientes e importancia por permutación sin atribuir causalidad física |
| `src/experimento.py` | Orquesta inspección, ejecución, manifiestos y artefactos; no entrena por defecto |

### Configuración, pruebas y archivos raíz

| Ruta | Función |
|---|---|
| `config/data_schema.yml` | Contrato de columnas, identificadores, predictores, target, unidades y reglas de curaduría |
| `config/model_config.yml` | Protocolo de validación, grupos, seed, familias candidatas, rejillas y autorización de ejecución |
| `tests/test_workflow.py` | Comprueba aislamiento de folds, protección contra leakage y construcción sin entrenamiento |
| `requirements.txt` | Dependencias necesarias para reproducir el flujo |
| `pyproject.toml` | Metadatos del proyecto y versión mínima de Python |
| `.gitignore` | Evita versionar datos, PDF privados, logs, modelos y resultados experimentales |
| `.gitattributes` | Normaliza archivos de texto en Git |
| `LICENSE` | Condiciones académicas del código y separación respecto de los derechos sobre los datos |
| `README.md` | Documento central del problema científico, metodología y uso del repositorio |

## Qué fuerzas y aspectos físicos intervienen

La **solicitación externa** de interés es la tracción: tirar de la pieza para alargarla. Dentro de la red, esa acción produce varios **mecanismos de deformación**. No deben confundirse las fuerzas aplicadas por la máquina con lo que sucede localmente en cada fibra o unión.

| Concepto | Qué significa físicamente | Papel en esta tesis |
|---|---|---|
| Fuerza de tracción | Acción que estira el espécimen; la máquina registra la fuerza transmitida | La fuerza en el primer evento de rotura es la respuesta a predecir |
| Desplazamiento | Cambio de posición impuesto a un extremo de la pieza, normalmente expresado en mm | Permite interpretar el ensayo; no es una entrada disponible antes de ensayar |
| Extensión de fibras | Alargamiento de sus elementos delgados | Mecanismo que ayuda a relacionar geometría y respuesta |
| Flexión de fibras | Curvatura de las fibras al deformarse la red | Explica por qué importan las dimensiones de su sección |
| Rotación y torsión de pivotes | Cambio relativo de orientación entre fibras; puede torcer las uniones | Sustenta estudiar la geometría de los pivotes |
| Cambio de ángulos de las celdas | Reorganización de la red; se describe también mediante deformación de corte o cizallamiento | Aspecto de la cinemática interna; no implica que se haya aplicado un ensayo externo independiente de corte |
| Carga y descarga | Aumento y reducción programados del desplazamiento durante el ensayo | Ayudan a observar deformación residual y disipación de energía; una descarga no debe etiquetarse automáticamente como rotura |
| Daño localizado y fallas sucesivas | Rotura de un elemento, seguida eventualmente de otras roturas | El target corresponde al **primer** evento; los posteriores no son nuevas muestras independientes |

Los mecanismos de extensión, flexión y deformación de pivotes están estudiados en los trabajos de [Turco y colaboradores](https://doi.org/10.1007/s00161-018-0678-y). La documentación experimental de Venditti describe tracción con ciclos de carga y descarga. El alcance actual no incluye predecir compresión, impacto, fatiga ni respuesta sísmica a partir de estos ensayos.

Para construir el target se deberá vincular el primer evento de rotura con su registro de fuerza. No basta con escoger el máximo de toda la curva fuerza–desplazamiento ni cualquier descenso de fuerza: puede haber máximos posteriores y descargas programadas. Los valores, unidades y criterio de identificación deberán quedar trazados a la fuente.

## Dataset

**Procedencia e investigador.** Enrico Venditti es el autor de la investigación doctoral sobre daño y resiliencia de materiales y estructuras complejas desarrollada en la **Università degli Studi di Sassari (Universidad de Sassari), Italia**, en el Departamento de Arquitectura, Diseño y Urbanismo y el programa de Arquitectura y Ambiente. Su trayectoria y línea de investigación se describen en su [perfil institucional](https://www.architettura.uniss.it/en/department/people/phd-student/enrico-venditti). El [repositorio de Sassari](https://iris.uniss.it/handle/11388/379270) registra la tesis doctoral y su discusión el 18 de febrero de 2026; la copia consultada indica curso académico 2024/2025.

El plan de tesis de Melissa identifica como fuente prevista la campaña experimental de Venditti y un archivo MATLAB proporcionado por Emilio Turco para fines académicos. Ese archivo aún no está incorporado al repositorio. La relación de entrega procede del plan de tesis; no se presume una colaboración institucional formal ni se atribuyen a Melissa los ensayos o resultados originales.

Una **campaña experimental** reúne especímenes y ensayos bajo un protocolo. El **dataset** será la tabla que se construya a partir de sus registros mediante extracción, revisión y decisiones documentadas. La campaña puede ampliarse durante la investigación.

| Aspecto | Descripción |
|---|---|
| Fuente | Registros de ensayos físicos de estructuras pantográficas y documentación de procedencia |
| Material, fabricación y ensayo | Poliamida/nylon, impresión SLS y tracción con seguimiento de fuerza y desplazamiento |
| Unidad de análisis | Un caso asociado con un espécimen ensayado; verificar correspondencia, réplicas y configuraciones relacionadas |
| Predictores | Medidas de la geometría y estructura que se conocerían antes de ensayar una pieza nueva |
| Target | Fuerza correspondiente al primer evento localizado de rotura, con unidad confirmada |
| Tipo de target | Número continuo: se estima una fuerza, no una categoría |
| Tamaño del conjunto | Se determinará a partir de la versión consolidada de los datos |
| Versionado | La ingesta registra observaciones, columnas, fecha UTC, SHA-256 y opciones de lectura por ejecución |
| Metadatos actuales | Se completarán automáticamente al ejecutar la ingesta sobre la versión consolidada |

El adaptador MATLAB queda pendiente de disponer del archivo y verificar su estructura. Actualmente se leen CSV y XLSX; cualquier exportación deberá conservar su relación con el original. Los PDF sirven para consulta: no se convierten automáticamente en un dataset ni se distribuyen con el proyecto.

### Pertinencia del material y de la investigación para Perú

La poliamida permite estudiar una arquitectura compleja producida por fabricación aditiva. Para el contexto peruano, el interés se apoya en capacidades y antecedentes de investigación locales: la PUCP registra un [trabajo sobre impresión de materiales de ingeniería, incluido nylon](https://tesis.pucp.edu.pe/items/211025b0-40d0-4eb5-b5e5-4fd5674eaf27), y su laboratorio [LIBRA](https://departamento-ingenieria.pucp.edu.pe/laboratorio/laboratorio-de-ingenieria-biomecanica-y-robotica-aplicada-libra/) estudia propiedades mecánicas de piezas impresas en 3D y desarrolla investigación en biomecánica y robótica.

| Nivel de pertinencia | Qué puede sostenerse |
|---|---|
| Material | El nylon es objeto de trabajo académico sobre impresión 3D en Perú; su estudio no está limitado al país de origen de los ensayos |
| Arquitectura | Como posibilidad de investigación, redes deformables podrían estudiarse para componentes ligeros y flexibles; esta tesis no demuestra una aplicación específica |
| Método | La ingesta, curaduría, regresión y validación pueden reutilizarse para futuras campañas locales de estructuras comparables |
| Transferencia experimental | Fabricar y ensayar especímenes en Perú permitiría comprobar si la respuesta y las predicciones se mantienen bajo las condiciones locales |

**La pertinencia local es una justificación de investigación, no una validación industrial.** El antecedente peruano citado sobre nylon se refiere a impresión por filamento, que no equivale a SLS. Para transferir resultados se deberán verificar grado de poliamida, proceso y orientación de fabricación, geometría, acondicionamiento del material y protocolo de ensayo. La tesis no presupone disponibilidad local de una máquina SLS específica ni certifica un producto médico o estructural.

## Variables del estudio

Estas variables están documentadas conceptualmente en la fuente de Venditti. Su presencia, nombre y unidad en la tabla consolidada deben verificarse antes de activarlas en `config/data_schema.yml`.

| Grupo | Variable conceptual | Tipo | Uso |
|---|---|---|---|
| Identificación | Identificador de caso | ID | Trazabilidad; nunca predictor |
| Geometría | Número de celdas en la dirección Y | Numérica discreta | Predictor candidato |
| Geometría | Altura y base de las fibras | Numérica continua | Predictores candidatos |
| Geometría | Altura y radio de los pivotes | Numérica continua | Candidatos; revisar constantes |
| Estructural | Longitud total de fibras y número de pivotes | Continua / discreta | Candidatos; revisar dependencia geométrica |
| Estructural | Volúmenes de fibras, pivotes y espécimen | Numérica continua | Candidatos; verificar fórmulas y redundancia |
| Respuesta | Carga de primera falla | Numérica continua | Target |

Las tablas de referencia rotulan longitudes en mm, volúmenes en mm³ y primera falla en N. Esto no verifica por sí solo las unidades de un archivo aún no recibido. Hay diferencias entre la tabla geométrica general y algunas fichas individuales que deben conciliarse contra la fuente original antes de consolidar datos.

La carga última, los desplazamientos y las energías medidos durante el ensayo no son entradas del modelo principal. Los eventos sucesivos de un mismo ensayo tampoco constituyen observaciones independientes. Los nombres conceptuales del YAML no se presentan como columnas reales detectadas.

## Flujo reproducible

```mermaid
flowchart TD
    A[Fuente experimental original] --> B[Ingesta: hash y manifiesto]
    B --> C[Curaduría: roles, unidades y trazabilidad]
    C --> D[EDA y definición del protocolo]
    D --> E[Partición externa por casos o grupos]
    E --> F[Entrenamiento externo]
    E --> G[Evaluación externa reservada]
    F --> H[CV interna: preprocesamiento, familia e hiperparámetros]
    H --> I[Reajuste en entrenamiento externo]
    I --> J[Predicciones sobre evaluación externa]
    G --> J
    F --> K[Baseline de mediana del entrenamiento]
    K --> L[Comparación fuera de muestra]
    J --> L
    L --> M[Errores, interpretabilidad y artefactos]
```

| Etapa | Entrada | Proceso | Salida y estado |
|---|---|---|---|
| Ingesta | Original en `raw` | Lectura, hashes, dimensiones y esquema observado | Versión trazable en `interim` |
| Curaduría | Ingesta + esquema revisado | Tipos, faltantes, duplicados, roles y revisión física | Tabla y reporte en `processed` |
| EDA | Tabla curada | Distribuciones, correlaciones, anomalías y redundancia | Diagnóstico descriptivo |
| Partición | Tabla, grupos y protocolo | CV externa e interna sin solapamiento | Índices y ordinales de origen |
| Preprocesamiento | Entrenamiento de cada fold | Medianas, constantes y escalamiento según modelo | Pipeline ajustado solo con entrenamiento |
| Baseline | Target del entrenamiento externo | Mediana | Referencia para la comparación |
| Ajuste o tuneo | Entrenamiento externo | `GridSearchCV` con MAE interno | Familia e hiperparámetros seleccionados |
| Validación | Fold externo | Predicciones y errores | Métricas y predicciones OOF |
| Interpretación | Pipeline ajustado + fold externo | Coeficientes y permutación opcional | Diagnósticos por fold |
| Reajuste final | Toda la tabla, después de evaluar | Selección interna y ajuste para reutilización | Modelo opcional sin una evaluación nueva |

**Fold** significa partición de validación cruzada; **OOF** identifica predicciones obtenidas cuando cada observación estuvo fuera del entrenamiento. El código está preparado, pero el modelamiento requiere incorporar y revisar primero los datos reales.

## Curaduría y control de calidad de datos experimentales

**Curaduría** significa construir y documentar el dataset válido. **Preprocesamiento del modelo** significa ajustar transformaciones sobre entrenamiento. Son etapas distintas.

| Control | Alcance actual |
|---|---|
| Trazabilidad | Hash de fuente y tabla, versión de ingesta, configuración, índice de registro y exclusiones |
| Esquema | Encabezados no vacíos ni repetidos; columnas y roles explícitos antes de producir la tabla analítica |
| Faltantes y tipos | Reporte por columna; target numérico finito, nunca imputado; exclusión registrada del target ausente |
| Duplicados | Se reportan y conservan por defecto; eliminación exacta solo si se confirma que son duplicaciones de registro |
| Constantes | Se identifican para revisión; no se seleccionan variables automáticamente con todo el dataset |
| Unidades y anomalías físicas | Verificación manual de unidades, convenciones de signo y rangos; no se inventan umbrales ni conversiones |
| Réplicas y dependencia | Revisar identificadores repetidos y grupos; no confundir réplicas con duplicados |
| Redundancia | Revisar relaciones determinísticas y colinealidad; selección estadística posterior dentro de entrenamiento |
| Definición del target | Confirmar evento, criterio de detección y correspondencia con el registro de fuerza |

Solo se incluyen las columnas declaradas como target, predictores, identificadores o grupos. El script impide roles superpuestos, pero la revisión científica debe confirmar que ningún predictor procede de una respuesta del ensayo. Los marcadores de ausencia distintos de celdas vacías deben declararse en el YAML.

## Prevención de fuga de información

| Operación | Puede hacerse antes del split | Debe ajustarse solo con train |
|---|---:|---:|
| Renombrar columnas | Sí | No |
| Convertir unidades con regla fija y documentada | Sí | No |
| Eliminar duplicados exactos confirmados | Sí | No |
| Derivar variables con fórmulas físicas predefinidas | Sí | No |
| Imputación estadística | No | Sí |
| Escalamiento | No | Sí |
| Selección supervisada o basada en la distribución | No | Sí |
| Ajuste de hiperparámetros | No | Sí, con validación interna |

Las réplicas y registros relacionados deben permanecer en el mismo grupo de partición. El EDA previo puede describir calidad; las decisiones de selección, transformación y ajuste guiadas por datos deben quedar dentro del entrenamiento. El test o los pliegues externos no se usan para elegir modelos ni hiperparámetros.

## Baseline y modelos candidatos

El **baseline** predice la mediana de la carga observada en el entrenamiento. Permite comprobar si los modelos candidatos aportan una reducción real del error.

| Familia | Qué representa | Estado de la implementación |
|---|---|---|
| Ridge | Relación lineal con penalización para controlar coeficientes | Pipeline y rejilla preliminar |
| Elastic Net | Relación lineal con regularización combinada; puede reducir contribuciones redundantes | Pipeline y rejilla preliminar |
| Árbol de regresión | Reglas sucesivas que dividen el espacio geométrico | Pipeline con control de profundidad y tamaño de hojas |
| Bosque aleatorio | Promedio de varios árboles de regresión | Pipeline y rejilla preliminar |
| Proceso gaussiano | Relación no lineal definida mediante un kernel de similitud | Pipeline y búsqueda preliminar; intervalos calibrados pendientes |
| Red neuronal de regresión | Combinación no lineal de entradas mediante capas | Constructor disponible; deshabilitado por defecto hasta justificar su complejidad |

Los rangos de `config/model_config.yml` son puntos de partida. Cada pipeline contiene su preprocesamiento; los árboles no requieren escalamiento. Una columna completamente ausente en un entrenamiento detiene el protocolo. No se presupone un modelo ganador ni que una red neuronal será superior.

### Cómo se separan ajuste y validación

1. Reservar un fold externo y mantener juntos los registros dependientes.
2. Usar folds internos para seleccionar familia e hiperparámetros mediante MAE.
3. Reajustar la selección con todo el entrenamiento externo.
4. Predecir una sola vez el fold externo y compararlo con el baseline.

La salida `selected` evalúa el procedimiento completo. Escoger posteriormente una familia por sus errores externos requeriría una nueva evaluación. Véase la [validación cruzada anidada de scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

## Evaluación

Las métricas se calculan con predicciones OOF y se comparan con el baseline usando exactamente los mismos folds. No se reportará solo un ajuste sobre los datos empleados para entrenar.

| Métrica | Prioridad | Qué responde | Lectura en el proyecto |
|---|---|---|---|
| **MAE** | Principal | ¿Cuántos newtons se desvía, en promedio, la predicción? | Promedio de `|carga observada − carga predicha|`. Menor es mejor y mantiene una interpretación directa en N |
| **RMSE** | Complementaria | ¿El modelo está cometiendo algunos errores especialmente grandes? | Eleva los errores al cuadrado antes de promediarlos, por lo que penaliza más las desviaciones grandes. Menor es mejor |
| **MedAE** | Complementaria | ¿Cuál es el error absoluto típico sin quedar dominado por casos extremos? | Mediana de los errores absolutos, expresada en N. Menor es mejor |
| **R² OOF** | Complementaria | ¿Cuánto mejora el procedimiento respecto de predecir la media global de las observaciones evaluadas? | Un valor cercano a 1 indica mejor ajuste; 0 equivale a esa referencia y un valor negativo indica peor desempeño. No se interpreta si el target es constante o hay menos de dos observaciones |

El **criterio principal de ajuste interno será el MAE**. En la evaluación externa se informarán las cuatro métricas por fold y sobre el conjunto de predicciones OOF. También se reportará `MAE_baseline − MAE_modelo`: un valor positivo indica que el procedimiento reduce el error frente a la referencia de mediana.

No se fijará anticipadamente un valor de MAE “aceptable” sin conocer la variabilidad experimental y la utilidad mecánica requerida. La dispersión entre folds se mostrará como estabilidad del resultado, no como un intervalo de confianza automático porque los entrenamientos se superponen.

Accuracy, precision, recall, F1 y matriz de confusión no corresponden al objetivo principal, ya que la respuesta es una fuerza continua. Si posteriormente se evalúan intervalos probabilísticos, se añadirán cobertura y amplitud media como métricas específicas de incertidumbre; no sustituirán al MAE.

## Interpretabilidad

| Enfoque | Qué permite examinar | Alcance actual |
|---|---|---|
| Intrínseco: coeficientes lineales | Dirección y magnitud de asociaciones, considerando escalas y colinealidad | Extracción por fold para el modelo seleccionado cuando es lineal |
| Post hoc: permutación | Cuánto aumenta el MAE al perturbar una entrada externa | Disponible de forma opcional; deshabilitada por defecto |
| SHAP | Contribuciones a predicciones bajo un método de referencia | Extensión pendiente, condicionada al modelo y a la dependencia entre entradas |
| PDP/ALE | Forma de la relación estimada entre una entrada y la respuesta | Extensión pendiente; requiere comprobar combinaciones físicamente plausibles |

No todos los modelos son interpretables por construcción. La importancia por permutación puede alterarse por predictores correlacionados o crear combinaciones incompatibles con relaciones geométricas. Se revisarán estabilidad entre folds y coherencia con extensión/flexión de fibras y deformación de pivotes; no se utilizarán estas explicaciones externas para retunear el mismo experimento.

**Importancia predictiva no implica causalidad física.** Las explicaciones deben contrastarse con conocimiento mecánico y sus limitaciones. Los coeficientes guardados corresponden a entradas estandarizadas en su entrenamiento; no se presentan como leyes constitutivas ni parámetros materiales identificados.

## Reproducibilidad

| Artefacto | Registro previsto por el código |
|---|---|
| Dataset | Fecha UTC, dimensiones y hash de fuente, tabla intermedia y tabla curada |
| Código y entorno | Commit Git, cambios locales, Python y dependencias instaladas |
| Curaduría | Contenido/hash del esquema, columnas, ordinales y exclusiones |
| Protocolo | Configuración completa, justificación, seed, estrategia y rejillas |
| Particiones | Índices externos/internos y vínculo con ordinales de origen |
| Preprocesamiento y modelo | Pipeline ajustado de cada familia/fold, con parámetros aprendidos, en formato joblib |
| Búsqueda interna | Combinaciones evaluadas, puntuaciones internas y configuración seleccionada |
| Predicciones y errores | OOF por caso, procedimiento, fold y familia; residuos observada − predicha |
| Métricas | MAE, RMSE, R² y MedAE por fold y agregadas; comparación pareada con baseline |
| Interpretación | Coeficientes y, si se activa, importancia por permutación por fold |
| Integridad | Manifiesto de experimento y hashes de sus artefactos |
| Logs | `pipeline.log`, `data_quality.log`, `modeling.log`; advertencias y fallos |

Las versiones anteriores no se sobrescriben. Una ejecución interrumpida conserva salidas parciales con `failure.json` y no se considera completa sin `experiment_manifest.json`. El modelo final opcional se etiqueta como reajuste sobre todos los datos. Guardar versiones no sustituye congelar el entorno y conservar el commit definitivo antes de reportar resultados.

## Cómo ejecutar el pipeline

Desde la raíz, con Python 3.11 o superior:

```bash
python -m venv .venv
```

Activar con `.venv\Scripts\Activate.ps1` en PowerShell, `.venv\Scripts\activate.bat` en CMD o `source .venv/bin/activate` en Linux/macOS. Después:

```bash
python -m pip install -r requirements.txt
```

### Datos y curaduría

1. Incorporar un original autorizado en `data/raw/`; CSV UTF-8 separado por comas o XLSX. Para una hoja específica usar `--sheet NOMBRE`. Los números deben usar punto decimal. Se preserva texto como `NA`; Excel no permite recuperar ceros iniciales guardados únicamente como formato visual.
2. Sustituir el nombre del archivo y ejecutar:

   ```bash
   python src/ingesta.py --input data/raw/NOMBRE_ARCHIVO.csv
   ```

3. Completar `config/data_schema.yml` con columnas reales, roles, unidades y marcadores de ausencia. Sustituir `VERSION` por la ingesta informada:

   ```bash
   python src/preprocesamiento.py --input data/interim/VERSION/dataset_interim.csv
   ```

4. Revisar el reporte y seguir los seis notebooks en el orden documentado en **Estructura del repositorio**.

### Inspección sin entrenamiento

El comando siguiente existe y puede ejecutarse ahora. Con la configuración distribuida informa que faltan datos y protocolo; no produce métricas:

```bash
python -m src.experimento --check
```

Las pruebas del avance validan contratos, construcción sin ajuste, separación de grupos e índices de CV; no entrenan estimadores:

```bash
python -m unittest discover -s tests -v
```

### Ejecución futura con datos y protocolo revisados

En `config/model_config.yml`, indicar `data.model_table`, justificar la estrategia, definir folds/grupos, revisar rejillas y marcar `protocol.reviewed: true`. El entrenamiento permanece desactivado mientras `execution.allow_training` sea `false`.

Solo después de completar esa revisión y habilitar explícitamente el entrenamiento:

```bash
python -m src.experimento --run
```

Para añadir un reajuste final después de la evaluación, sin atribuirle una nueva métrica independiente:

```bash
python -m src.experimento --run --fit-final
```

El comando histórico `python src/modelo_baseline.py` solo muestra un aviso; la evaluación real integra el baseline en las mismas particiones de los candidatos.

## Resultados esperados del avance

El flujo, los notebooks y los controles de software están preparados. Faltan la fuente original, la conciliación científica, el diccionario definitivo, la elección de grupos/folds y la ejecución experimental. Por ello aún no existen métricas, predicciones ni un modelo ganador.

## Roadmap

| Fase | Actividad |
|---|---|
| 1 | Recibir fuente original, verificar exportación, ingesta y versionado |
| 2 | Conciliar discrepancias, completar diccionario, curaduría y EDA |
| 3 | Definir particiones y ejecutar baseline |
| 4 | Comparar modelos candidatos y ajuste interno |
| 5 | Validar fuera de muestra e interpretar estabilidad y coherencia física |
| 6 | Consolidar código, entorno, datos autorizados y artefactos reproducibles |

## Referencias

Referencias verificadas en los documentos consultados; los documentos fuente no se distribuyen con el proyecto.

- Aylas Barranca, M. D. (2026). *Análisis predictivo e interpretable de la respuesta mecánica de estructuras pantográficas mediante aprendizaje automático supervisado aplicado a datos experimentales*. Plan de tesis, Universidad Nacional de Ingeniería.
- Venditti, E. (2026; copia consultada: curso académico 2024/2025). *Analisi strutturale per lo studio del danneggiamento in materiali e strutture complesse*. Tesis doctoral, Università degli Studi di Sassari. [Registro institucional](https://iris.uniss.it/handle/11388/379270).
- Turco, E., Golaszewski, M., Giorgio, I. y Placidi, L. (2017). *Can a Hencky-Type Model Predict the Mechanical Behaviour of Pantographic Lattices?* En *Mathematical Modelling in Solid Mechanics*, pp. 285–311. [DOI: 10.1007/978-981-10-3764-1_18](https://doi.org/10.1007/978-981-10-3764-1_18).
- Turco, E., Misra, A., Sarikaya, R. y Lekszycki, T. (2019; publicación en línea en 2018). *Quantitative analysis of deformation mechanisms in pantographic substructures: experiments and modeling*. *Continuum Mechanics and Thermodynamics*, 31, 209–223. [DOI: 10.1007/s00161-018-0678-y](https://doi.org/10.1007/s00161-018-0678-y).

- Toyama Higa, P. M. (2022). *Diseño mecatrónico de un sistema de calefacción cerrado y deshumedecedor de materiales de ingeniería para impresoras 3D de escritorio de código libre*. PUCP. [Repositorio institucional](https://tesis.pucp.edu.pe/items/211025b0-40d0-4eb5-b5e5-4fd5674eaf27).
- PUCP. *Laboratorio de Ingeniería Biomecánica y Robótica Aplicada (LIBRA)*. [Descripción institucional](https://departamento-ingenieria.pucp.edu.pe/laboratorio/laboratorio-de-ingenieria-biomecanica-y-robotica-aplicada-libra/).
- Scikit-learn. *Nested versus non-nested cross-validation*. [Documentación oficial](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

## Licencia

El **código** se distribuye para uso académico y de investigación en el marco de la UNI, según `LICENSE`. Los **datos experimentales** conservan las condiciones de sus titulares; su publicación o redistribución requiere autorización expresa. La disponibilidad del código no otorga derechos sobre los datos o documentos consultados. `.gitignore` excluye los PDF, las referencias privadas y los datos de ejecución.
