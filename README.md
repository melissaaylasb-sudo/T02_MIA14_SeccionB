# Análisis predictivo e interpretable de la respuesta mecánica de estructuras pantográficas

Proyecto de investigación de la **Maestría en Ciencias con mención en Inteligencia Artificial de la Universidad Nacional de Ingeniería (UNI)** orientado al desarrollo de un flujo reproducible de ciencia de datos para estudiar y predecir la respuesta mecánica de estructuras pantográficas a partir de información experimental.

Las estructuras pantográficas son metamateriales mecánicos reticulares formados por familias de elementos esbeltos interconectados mediante pivotes. Su comportamiento global no depende únicamente del material constitutivo, sino también de la geometría y organización de su arquitectura interna. En este proyecto se trabaja con especímenes fabricados en poliamida (nylon) mediante sinterización láser selectiva y sometidos a ensayos mecánicos de tracción.

El problema se formula como **regresión supervisada**. El modelo no busca identificar el material del espécimen. A partir de características geométricas y estructurales de una configuración pantográfica, se busca estimar su **carga de primera falla**, entendida como la fuerza asociada con el primer evento localizado de rotura o pérdida de integridad registrado durante el ensayo. Este evento no equivale necesariamente al colapso total, ya que la estructura puede continuar soportando carga después del primer daño.

Este repositorio forma parte del curso **Proyecto de Investigación II**.

---

## Autora

- Melissa Dessire Aylas Barranca  
  Maestría en Ciencias con mención en Inteligencia Artificial – Universidad Nacional de Ingeniería

---

## Dataset

- **Fuente**: datos experimentales obtenidos mediante ensayos mecánicos de tracción sobre estructuras pantográficas fabricadas en poliamida/nylon mediante sinterización láser selectiva.
- **Naturaleza de los datos**: datos tabulares cuantitativos provenientes de ensayos físicos; no corresponden a datos sintéticos ni a simulaciones generadas específicamente para este avance.
- **Unidad de análisis**: caso experimental asociado con un espécimen o configuración pantográfica sometida a ensayo. La correspondencia entre casos, posibles réplicas y configuraciones repetidas se verificará durante la curaduría.
- **Variable objetivo principal**: carga de primera falla, expresada en newtons cuando la fuente experimental tenga la unidad confirmada.
- **Tipo de problema**: regresión supervisada.
- **Variables principales de entrada**: características geométricas y estructurales verificadas, entre ellas número de celdas, dimensiones de fibras, dimensiones de pivotes y descriptores derivados físicamente justificables.
- **Otras respuestas disponibles**: carga última, desplazamientos, variables energéticas y eventos sucesivos de falla. Estas variables se conservarán para análisis descriptivos o secundarios y no se utilizarán como predictores cuando correspondan a información generada después del inicio del ensayo.
- **Registros**: el repositorio no fija un tamaño final de muestra. Cada versión de datos registrará automáticamente el número de observaciones disponibles después de la ingesta y curaduría.
- **Versión usada**: se identifica mediante fecha de ejecución, nombre de archivo y manifiesto de versión.
- **Hash SHA-256**: se calcula automáticamente durante la ingesta y se almacena en `data/interim/dataset_manifest.json`.

Los archivos originales se consideran inmutables. Cualquier ampliación de la base experimental debe incorporarse como una nueva versión, sin sobrescribir silenciosamente los datos utilizados en ejecuciones anteriores.

---

## Estructura del repositorio

```text
pantographic-first-failure/
│
├── data/
│   ├── raw/                     # datos experimentales originales, sin modificar
│   ├── interim/                 # datos después de ingesta y curaduría inicial
│   └── processed/               # tabla analítica lista para modelamiento
│
├── notebooks/
│   ├── EDA_basico.ipynb         # análisis exploratorio inicial
│   └── Baseline_basico.ipynb    # evaluación inicial del baseline de regresión
│
├── src/
│   ├── __init__.py
│   ├── ingesta.py               # lectura, trazabilidad, hash y manifiesto
│   ├── preprocesamiento.py      # curaduría y construcción de tabla analítica
│   └── modelo_baseline.py       # baseline preliminar de regresión
│
├── logs/
│   └── .gitkeep                 # logs automáticos de ejecución y métricas
│
├── slides/
│   └── .gitkeep                 # presentaciones del avance
│
├── config/
│   └── data_schema.yml          # nombres canónicos y roles de variables
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── LICENSE
```

### `data/raw`

Contiene exclusivamente los archivos originales recibidos de la fuente experimental. No se editan ni sobrescriben desde el código. Su función es preservar una referencia reproducible de la información de origen.

### `data/interim`

Contiene datos intermedios después de operaciones determinísticas de ingesta y curaduría inicial, por ejemplo normalización de nombres de columnas, revisión de tipos, identificación de duplicados y generación de variables auxiliares de trazabilidad.

Esta carpeta **no debe contener transformaciones estadísticas aprendidas con todo el dataset**, como escalamiento global, imputación basada en la distribución completa o selección de variables basada en el target.

### `data/processed`

Contiene la tabla analítica curada utilizada como entrada del modelamiento. La tabla debe conservar únicamente variables con definición, unidad y procedencia verificables, además del identificador necesario para trazabilidad.

La imputación estadística, escalamiento y selección dependiente de los datos se ajustarán posteriormente dentro de cada conjunto de entrenamiento para evitar fuga de información.

---

## Requisitos

Se recomienda Python 3.11 o superior.

Instalación mediante `pip`:

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

En Linux/macOS:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Dependencias iniciales:

- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- openpyxl
- PyYAML
- jupyter

Las versiones efectivamente utilizadas deberán quedar fijadas antes de reportar resultados comparables.

---

## Cómo ejecutar el pipeline

### 1. Ingesta de datos

Colocar el archivo original en:

```text
data/raw/
```

Ejecutar:

```bash
python src/ingesta.py --input data/raw/NOMBRE_ARCHIVO.csv
```

La ingesta:

- carga el archivo sin modificarlo;
- registra dimensiones y nombres de columnas;
- calcula el hash SHA-256;
- genera un manifiesto de versión;
- guarda una copia tabular intermedia para las etapas siguientes;
- registra la ejecución en `logs/pipeline.log`.

Salidas esperadas:

```text
data/interim/dataset_interim.csv
data/interim/dataset_manifest.json
logs/pipeline.log
```

Si la fuente se encuentra en Excel, puede utilizarse `.xlsx`. La incorporación de otros formatos requerirá un adaptador explícito en `src/ingesta.py`.

### 2. Curaduría y preprocesamiento determinístico

Ejecutar:

```bash
python src/preprocesamiento.py
```

Esta etapa revisa:

- existencia de la variable objetivo;
- tipos de datos;
- valores faltantes;
- duplicados;
- variables constantes;
- identificadores no predictivos;
- consistencia de nombres y unidades;
- posibles variables redundantes;
- disponibilidad de predictores geométricos y estructurales.

La carga de primera falla no será imputada para entrenar el modelo. Los registros sin target verificable se documentarán antes de cualquier exclusión del conjunto de modelamiento.

Salidas esperadas:

```text
data/processed/model_table.csv
logs/data_quality.log
```

### 3. Exploración inicial

Abrir y ejecutar:

```text
notebooks/EDA_basico.ipynb
```

El EDA se utilizará para estudiar la distribución de las variables, rangos, valores faltantes, relaciones entre predictores, posibles dependencias y comportamiento de la variable objetivo. Los gráficos exploratorios no se interpretarán como evidencia de capacidad predictiva.

### 4. Baseline preliminar

El baseline sirve como referencia mínima para determinar si un modelo más complejo aporta una mejora real.

La primera referencia propuesta para este problema de regresión es un `DummyRegressor` que predice la **mediana de la variable objetivo calculada únicamente con los datos de entrenamiento**.

El código se encuentra preparado en:

```text
src/modelo_baseline.py
```

y podrá ejecutarse cuando la tabla analítica y el protocolo de partición hayan sido revisados:

```bash
python src/modelo_baseline.py
```

El script no forma parte de la ingesta ni se ejecuta automáticamente.

Las métricas previstas son:

- **MAE** como métrica principal;
- **RMSE** como medida complementaria sensible a errores grandes;
- **R²** calculado sobre predicciones fuera de muestra;
- **MedAE** como medida robusta complementaria cuando resulte útil.

Al tratarse de un problema de regresión, no se utilizarán matriz de confusión, accuracy, precision, recall o F1 como métricas principales.

---

## Curaduría y control de calidad

La curaduría tiene como finalidad construir un conjunto analítico trazable a partir de datos experimentales previamente obtenidos. No se limita a detectar errores; documenta las decisiones mediante las cuales un registro o variable puede incorporarse al modelamiento.

Se verificará la procedencia de cada caso, la correspondencia entre identificador y ensayo, la definición de las variables, las unidades físicas, los valores faltantes, duplicados, constantes, rangos físicamente plausibles y relaciones determinísticas entre predictores. También se revisará la existencia de posibles réplicas o configuraciones relacionadas, debido a que esta información condicionará el protocolo de validación.

Los identificadores administrativos o secuenciales se conservarán para trazabilidad, pero no se utilizarán como predictores.

Las variables de respuesta generadas durante o después del ensayo no se incorporarán como entradas del modelo principal si no estarían disponibles al momento en que se pretende realizar la predicción.

---

## Prevención de fuga de información

La separación entre curaduría del dataset y transformaciones aprendidas por el modelo es deliberada.

Pueden ejecutarse antes de la partición operaciones determinísticas que no utilizan información estadística de otros registros, por ejemplo:

- renombrar columnas;
- verificar unidades;
- convertir tipos;
- eliminar duplicados exactamente identificados;
- marcar variables constantes;
- documentar registros sin target;
- construir variables derivadas mediante fórmulas físicas previamente definidas.

En cambio, las siguientes operaciones deberán ajustarse únicamente con los datos de entrenamiento de cada partición:

- imputación estadística de predictores;
- estandarización o normalización;
- selección estadística de variables;
- transformaciones basadas en la distribución;
- ajuste de hiperparámetros;
- entrenamiento del modelo.

Esta separación evita que información del conjunto de validación influya indirectamente en el modelo.

---

## Modelamiento previsto

El repositorio no fija todavía un modelo final. La comparación se realizará de forma progresiva, comenzando por un baseline y aumentando la complejidad únicamente si la evidencia fuera de muestra lo justifica.

Entre las familias candidatas se consideran:

- regresiones regularizadas;
- árboles y ensambles de regresión;
- procesos gaussianos;
- modelos neuronales de regresión, condicionados al tamaño y representación final de los datos.

La selección de hiperparámetros y la estimación del desempeño deberán permanecer separadas. Si el tamaño y estructura final del conjunto lo permiten, se utilizará validación cruzada anidada o una estrategia equivalente que evite utilizar la evaluación externa para seleccionar el modelo.

Si se identifican réplicas o configuraciones dependientes, la partición deberá conservarlas dentro del mismo grupo.

---

## Interpretación de los modelos

La interpretabilidad se tratará de acuerdo con la familia de modelamiento seleccionada y no como una propiedad automática de cualquier algoritmo.

En modelos lineales podrán analizarse coeficientes estandarizados. Para modelos no lineales se considerarán herramientas post hoc como importancia por permutación, SHAP o gráficos de dependencia/efectos cuando su uso sea metodológicamente compatible con el modelo y la dependencia entre variables.

Las explicaciones se contrastarán con conocimiento de mecánica estructural. Una característica importante para la predicción no se interpretará por sí sola como causa física de la primera falla.

---

## Logging y reproducibilidad

Las ejecuciones deben dejar evidencia suficiente para reconstruir el estado del experimento.

Los logs registrarán, según la etapa:

- fecha y hora;
- archivo de entrada;
- hash SHA-256;
- número de filas y columnas;
- columnas detectadas;
- advertencias de calidad;
- registros descartados y motivo;
- archivos de salida generados;
- configuración utilizada;
- métricas cuando corresponda.

El manifiesto de datos permite distinguir versiones de la campaña experimental. La incorporación de nuevas observaciones generará una nueva versión y no reemplazará silenciosamente una versión ya utilizada.

Cuando se inicie el modelamiento también se registrarán particiones, hiperparámetros, seeds cuando correspondan, versión de dependencias, predicciones fuera de muestra y artefactos entrenados.

---

## Resultados esperados del avance

En esta etapa se espera disponer de:

- estructura reproducible del repositorio;
- datos originales separados de las versiones intermedias y procesadas;
- script de ingesta;
- hash y manifiesto de versión de datos;
- procedimiento inicial de curaduría y control de calidad;
- logs automáticos;
- notebook de EDA preparado;
- baseline de regresión implementado pero no ejecutado automáticamente;
- definición explícita de la variable objetivo y de las métricas previstas.

No se reportan en este repositorio valores de MAE, RMSE, R², rankings de modelos ni conclusiones predictivas mientras esos experimentos no hayan sido ejecutados bajo el protocolo de validación definido.

---

## Roadmap

- **Etapa 1 — Ingesta y trazabilidad**  
  Incorporación de los archivos originales, cálculo de hashes, manifiesto y logging.

- **Etapa 2 — Curaduría y preprocesamiento**  
  Verificación de calidad, definición de la tabla analítica y documentación de criterios de inclusión/exclusión.

- **Etapa 3 — Análisis exploratorio y baseline**  
  EDA reproducible y establecimiento de una referencia mínima de desempeño.

- **Etapa 4 — Modelos candidatos y ajuste**  
  Evaluación progresiva de familias de regresión, ajuste de hiperparámetros y control de complejidad.

- **Etapa 5 — Evaluación e interpretación**  
  Predicciones fuera de muestra, métricas, estabilidad e interpretación compatible con el modelo seleccionado.

- **Etapa 6 — Consolidación reproducible**  
  Versionado de datos, modelos, transformadores, resultados y documentación final del experimento.

---

## Licencia

Uso académico en el marco de la **Universidad Nacional de Ingeniería (UNI)**.

Los datos experimentales originales no deben redistribuirse públicamente hasta verificar expresamente sus condiciones de uso y autorización. La publicación del código no implica autorización para publicar las fuentes experimentales.
