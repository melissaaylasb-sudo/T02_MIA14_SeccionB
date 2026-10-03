# Análisis predictivo e interpretable de la respuesta mecánica de estructuras pantográficas

Las estructuras pantográficas son metamateriales mecánicos reticulares: redes de fibras esbeltas interconectadas mediante pivotes. Su geometría permite mecanismos de deformación asociados con la flexión y extensión de las fibras y la deformación de los pivotes. Los especímenes de interés se fabrican en **poliamida (nylon) mediante sinterización láser selectiva (SLS)** y se estudian mediante ensayos de tracción.

Esta investigación plantea una **regresión supervisada**: utilizar características geométricas y estructurales disponibles antes del ensayo para predecir la **carga de primera falla**. Es la fuerza asociada con el primer evento localizado de rotura o pérdida de integridad registrado; no equivale necesariamente a la carga máxima, la falla total ni el colapso global. La estructura puede continuar soportando carga después de ese evento. El material es contexto experimental, no la variable a predecir.

**Estado: PRELIMINAR**, correspondiente al curso Proyecto de Investigación II. El repositorio prepara la ingesta, la curaduría y la referencia de regresión; todavía no contiene un dataset experimental consolidado ni modelos entrenados o resultados predictivos.

## Autora

**Melissa Dessire Aylas Barranca**

Maestría en Ciencias con mención en Inteligencia Artificial

Universidad Nacional de Ingeniería (UNI)

## Resumen del proyecto

| Elemento | Definición |
|---|---|
| Objeto de estudio | Estructuras pantográficas de fibras y pivotes |
| Material | Poliamida / nylon; contexto experimental |
| Tipo de ensayo | Tracción |
| Tipo de datos | Tabulares, cuantitativos, procedentes de ensayos físicos |
| Problema de ML | Regresión supervisada |
| Variable objetivo | Carga de primera falla |
| Unidad de salida | Newton (N), previa comprobación de la fuente incorporada |
| Finalidad predictiva | Estimar la primera falla de configuraciones dentro del dominio experimental representado y complementar el análisis mecánico |

## Dataset

El plan de tesis identifica como fuente prevista una campaña experimental de Enrico Venditti, proporcionada por Emilio Turco para uso académico, y un archivo MATLAB aún no incorporado al repositorio. La documentación de Venditti describe fabricación SLS, ensayos de tracción, registros fuerza–desplazamiento y eventos de rotura. Estos documentos sustentan el contexto; sus resultados no son resultados de esta tesis.

Una **campaña experimental** es el conjunto organizado de especímenes y ensayos realizados bajo un protocolo definido. El **dataset** será la tabla derivada de esa campaña mediante extracción y curaduría documentadas. La campaña puede ampliarse durante la investigación.

| Aspecto | Descripción |
|---|---|
| Fuente | Ensayos mecánicos experimentales y documentación de procedencia |
| Material / fabricación / ensayo | Poliamida o nylon / SLS / tracción |
| Unidad de análisis | Un caso asociado con un espécimen ensayado; verificar correspondencia, réplicas y configuraciones relacionadas |
| Predictores | Geometría y descriptores estructurales verificados, disponibles antes del ensayo |
| Target / tipo | Carga de primera falla / numérico continuo |
| Tamaño del conjunto | Se determinará a partir de la versión consolidada de los datos |
| Versionado | La ingesta registra observaciones, columnas, fecha UTC, SHA-256 y opciones de lectura por ejecución |
| Metadatos actuales | Se completarán automáticamente al ejecutar la ingesta sobre la versión consolidada |

No se extraen automáticamente tablas de los PDF ni se ejecutan archivos MATLAB. El adaptador de la fuente original queda pendiente de disponer del archivo y verificar su estructura. El lector preliminar admite CSV y XLSX; cualquier exportación deberá conservar la relación con el original y documentar sus decisiones.

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

## Estructura del repositorio

```text
T02_MIA14_SeccionB/
├── data/
│   ├── raw/                     # originales inmutables
│   ├── interim/                 # ingestas versionadas y manifiestos
│   └── processed/               # tablas analíticas y reportes versionados
├── notebooks/
│   ├── EDA_basico.ipynb
│   └── Baseline_basico.ipynb
├── src/
│   ├── __init__.py
│   ├── ingesta.py
│   ├── preprocesamiento.py
│   └── modelo_baseline.py
├── logs/                        # registros locales de ejecución
├── slides/                      # reservado para presentaciones
├── config/
│   └── data_schema.yml
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── LICENSE
```

| Directorio | Contenido y regla |
|---|---|
| `data/raw/` | Datos originales: nunca editar ni sobrescribir; incorporar ampliaciones como archivos nuevos |
| `data/interim/` | Copias tabulares por ingesta; lectura y trazabilidad sin transformaciones aprendidas |
| `data/processed/` | Tabla analítica con roles explícitos y curaduría determinística; entrada al futuro proceso de modelamiento |

La imputación estadística, el escalamiento y la selección que aprende parámetros **no se aplican globalmente en `processed`**. Se ajustarán dentro de cada conjunto de entrenamiento. Los datos y las salidas de ejecución están excluidos de Git por defecto.

## Flujo reproducible

| Etapa | Entrada | Proceso | Salida / estado |
|---|---|---|---|
| Ingesta | Original en `raw` | Lectura, hash, dimensiones y esquema observado | Versión en `interim`; implementada |
| Curaduría | `interim` + esquema revisado | Roles, tipos, faltantes y duplicados; revisión física complementaria | Versión en `processed`; controles preliminares implementados |
| EDA | Tabla curada | Descripción, faltantes y distribuciones | Notebook preparado, sin resultados |
| Partición | Tabla y grupos revisados | Train/validation/test o validación cruzada | Pendiente de protocolo |
| Preprocesamiento | Entrenamiento de cada partición | Ajustar imputación, escalamiento y selección | Pendiente; se integrará en el pipeline de ML |
| Baseline | Entrenamiento | Mediana del target | Constructor implementado; sin ajuste |
| Modelamiento | Entrenamiento | Familias candidatas y ajuste interno | Pendiente |
| Evaluación | Observaciones fuera del ajuste | MAE, RMSE, R² y MedAE | Función de métricas disponible; sin evaluación ejecutada |

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

La referencia inicial es `DummyRegressor(strategy="median")`, cuya mediana se calculará **solo con el target de entrenamiento**. Debe compararse con todos los modelos usando las mismas particiones fuera de muestra.

| Familia | Condición de estudio |
|---|---|
| Regresiones regularizadas | Evaluar representaciones parsimoniosas y colinealidad |
| Árboles y ensambles de regresión | Controlar complejidad y sobreajuste |
| Procesos gaussianos | Evaluar relaciones no lineales e incertidumbre |
| Redes neuronales de regresión | Familia candidata condicionada a la cantidad y representación final de los datos |

No existe modelo ganador ni se presupone que deep learning será superior. La validación cruzada anidada o un esquema equivalente deberá separar selección y evaluación, con agrupación cuando corresponda. El número de particiones queda pendiente.

## Evaluación

| Métrica | Rol | Interpretación |
|---|---|---|
| MAE | Principal | Error absoluto medio en unidades de carga |
| RMSE | Complementaria | Error en unidades de carga; penaliza más los errores grandes |
| R² | Complementaria | Proporción de variación explicada fuera de muestra; puede ser negativa; indefinida con target constante o menos de dos observaciones |
| MedAE | Complementaria | Mediana del error absoluto; medida robusta en unidades de carga |

La comparación contra el baseline será siempre **fuera de muestra**, considerando variabilidad entre particiones. No se usan matriz de confusión, accuracy, precision, recall ni F1 para el problema principal: el target es continuo y la tarea es regresión.

## Interpretabilidad

La interpretabilidad intrínseca de modelos lineales permite estudiar coeficientes y su estabilidad, teniendo en cuenta escalas y colinealidad. Para otros modelos se considerarán explicaciones post hoc: importancia por permutación, SHAP y PDP/ALE cuando sean compatibles con la dependencia entre predictores y el dominio experimental.

**Importancia predictiva no implica causalidad física.** Las explicaciones se contrastarán con conocimiento mecánico y su estabilidad entre particiones. No todos los modelos son interpretables por construcción.

## Reproducibilidad

| Artefacto | Qué se registra |
|---|---|
| Dataset | Fecha UTC, dimensiones, hash de fuente y salidas, versión de ingesta |
| Código | Commit Git y estado de cambios locales en manifiestos; conservar el commit definitivo al reportar resultados |
| Entorno | Python y versiones instaladas de dependencias del pipeline; rangos de instalación en `requirements.txt` |
| Esquema | Contenido y hash del YAML en cada reporte de curaduría |
| Split | Índices y grupos: pendiente de implementación |
| Preprocesamiento | Parámetros ajustados por entrenamiento: pendiente |
| Modelo | Clase e hiperparámetros: pendiente de ejecución |
| Aleatoriedad | Seeds y configuración de particiones: pendiente |
| Predicciones | Fuera de muestra, vinculadas a identificadores: pendiente |
| Métricas | MAE, RMSE, R² y MedAE: sin resultados disponibles |
| Logs | `pipeline.log` y `data_quality.log`, con advertencias y errores |

Cada ingesta y curaduría crea un directorio nuevo; las ejecuciones anteriores se conservan. Registrar versiones no sustituye congelar el entorno y el código antes de comparar experimentos. No se garantiza reproducibilidad exacta con cambios locales sin conservar.

## Cómo ejecutar el pipeline

Desde la raíz del repositorio, con Python 3.11 o superior:

```bash
python -m venv .venv
```

Activar con `.venv\Scripts\Activate.ps1` en PowerShell, `.venv\Scripts\activate.bat` en CMD o `source .venv/bin/activate` en Linux/macOS. Después:

```bash
python -m pip install -r requirements.txt
```

1. Colocar una versión original en `data/raw/`. Se admiten CSV UTF-8 separado por comas y XLSX (primera hoja por defecto, seleccionable con `--sheet NOMBRE`). Los datos numéricos deben usar punto decimal; no se interpretan automáticamente otras convenciones.
2. Ejecutar la ingesta sustituyendo el nombre del archivo:

   ```bash
   python src/ingesta.py --input data/raw/NOMBRE_ARCHIVO.csv
   ```

   La salida indica `data/interim/VERSION/dataset_interim.csv` y `dataset_manifest.json`. Registra filas, columnas, hash y fecha sin modificar el original. La lectura conserva texto e identificadores; no interpreta `NA` como faltante automáticamente. En XLSX no recupera ceros iniciales guardados solo como formato visual.

3. Completar `config/data_schema.yml` con los nombres reales de target, identificadores, predictores, grupos si existen y unidades verificadas. Luego sustituir `VERSION` por el directorio informado:

   ```bash
   python src/preprocesamiento.py --input data/interim/VERSION/dataset_interim.csv
   ```

   Genera una versión en `data/processed/` con `model_table.csv` y `data_quality_report.json`. El esquema sin completar, un manifiesto inconsistente o valores numéricos inválidos detienen la etapa con un mensaje. No imputa ni escala.

4. Abrir `notebooks/EDA_basico.ipynb`, indicar la ruta de la tabla generada y revisar su calidad. El notebook permanece sin ejecutar mientras no haya datos reales.
5. Definir y revisar el protocolo de partición antes de ajustar cualquier baseline. `notebooks/Baseline_basico.ipynb` solo construye el estimador. El comando existente:

   ```bash
   python src/modelo_baseline.py
   ```

   muestra un aviso de preparación; **no entrena ni evalúa**. El flujo de entrenamiento y partición aún no está implementado.

## Resultados esperados del avance

Están disponibles la ingesta con hashes y manifiesto, la curaduría preliminar condicionada al esquema, logs separados, notebooks sin resultados y funciones de baseline y métricas. Las tablas y reportes experimentales se generarán al incorporar datos reales. No hay métricas obtenidas, predicciones ni conclusiones sobre capacidad predictiva.

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
- Venditti, E. (curso académico 2024/2025). *Analisi strutturale per lo studio del danneggiamento in materiali e strutture complesse*. Tesis doctoral, Università degli Studi di Sassari. Contexto experimental, tabla geométrica y fichas de ensayo del apéndice A.
- Turco, E., Golaszewski, M., Giorgio, I. y Placidi, L. (2017). *Can a Hencky-Type Model Predict the Mechanical Behaviour of Pantographic Lattices?* En *Mathematical Modelling in Solid Mechanics*, pp. 285–311. [DOI: 10.1007/978-981-10-3764-1_18](https://doi.org/10.1007/978-981-10-3764-1_18).
- Turco, E., Misra, A., Sarikaya, R. y Lekszycki, T. (2019; publicación en línea en 2018). *Quantitative analysis of deformation mechanisms in pantographic substructures: experiments and modeling*. *Continuum Mechanics and Thermodynamics*, 31, 209–223. [DOI: 10.1007/s00161-018-0678-y](https://doi.org/10.1007/s00161-018-0678-y).

## Licencia

El **código** se distribuye para uso académico y de investigación en el marco de la UNI, según `LICENSE`. Los **datos experimentales** conservan las condiciones de sus titulares; su publicación o redistribución requiere autorización expresa. La disponibilidad del código no otorga derechos sobre los datos o documentos consultados. `.gitignore` excluye los PDF, las referencias privadas y los datos de ejecución.
