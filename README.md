# Predicción de la primera falla en estructuras pantográficas

<div align="center">
  <strong>PIPELINE REPRODUCIBLE</strong> · <strong>DATA PRELIMINAR VERSIONADA</strong> · <strong>EDA EJECUTADO</strong> · <strong>DEMOSTRADOR HTML</strong>
</div>

<p align="center">
  <a href="demo/index.html"><strong>Abrir demostrador</strong></a> ·
  <a href="notebooks/01_Ingesta_Curaduria_y_Calidad.ipynb">Ver ingesta y calidad</a> ·
  <a href="notebooks/02_EDA_Avanzado.ipynb">Ver EDA avanzado</a> ·
  <a href="reports/02_EDA_Avanzado.html">Abrir reporte EDA</a> ·
  <a href="data/README.md">Ver datos</a>
</p>

Este proyecto estima la **fuerza de primera falla** de una estructura pantográfica a partir de su geometría. La salida es una fuerza en newtons y el problema se aborda mediante regresión supervisada con validación fuera de muestra.

> **Producto de tesis:** modelo predictivo versionado, pipeline científico reproducible y demostrador interactivo para explorar configuraciones geométricas.

**Autora:** Melissa Dessire Aylas Barranca<br>
Maestría en Ciencias con mención en Inteligencia Artificial — Universidad Nacional de Ingeniería (UNI)

## Resumen ejecutivo

| Entrada | Proceso | Salida científica | Artefacto aplicado |
|---|---|---|---|
| Geometría del espécimen | Curaduría, EDA, validación anidada y comparación de modelos | Estimación de carga de primera falla, métricas e interpretación | Demostrador de predicción estructural |

**Pregunta de investigación:** ¿en qué medida las características geométricas y estructurales permiten estimar la carga de primera falla con precisión fuera de muestra e interpretación mecánicamente coherente?

**Objetivo general:** desarrollar y evaluar un modelo supervisado que estime la carga de primera falla y pueda integrarse en una herramienta interactiva de consulta.

## Ficha técnica

| Elemento | Especificación |
|---|---|
| Objeto de estudio | Red de fibras conectadas mediante pivotes, fabricada en poliamida por sinterización láser selectiva |
| Ensayo | Tracción con registro de fuerza y desplazamiento |
| Input | Geometría y estructura conocidas antes del ensayo |
| Target | `First_Failure_Load_N`: carga de primera falla en N |
| Output | Carga estimada de primera falla para un espécimen comparable |
| Problema de ML | Regresión supervisada |
| Modelos principales | Ridge, Elastic Net, SVR-RBF, árbol, bosque aleatorio y proceso gaussiano |
| Referencia | Baseline que predice un valor central del entrenamiento |
| Métrica principal | MAE fuera de muestra, expresado en N |
| Validación | Validación cruzada anidada; tuneo interno y evaluación externa |
| Estado | Ingesta, curaduría y EDA ejecutados; entrenamiento final pendiente de aprobar el protocolo de partición |

### Alcance

- **Incluye:** estructuras pantográficas comparables, poliamida fabricada por SLS, ensayo de tracción, geometría previa al ensayo y predicción de primera falla.
- **No incluye:** simulación por elementos finitos, predicción de toda la curva fuerza-desplazamiento, fatiga, impacto, compresión ni extrapolación automática a otros materiales.
- **Generalización prevista:** nuevas configuraciones dentro del dominio geométrico y experimental que finalmente sea validado.

## Input, target y output

### Input

Cada fila representa un caso experimental. Solo se consideran variables disponibles antes del ensayo.

| Grupo | Variables | Unidad |
|---|---|---|
| Configuración | `n_cells_Y` | conteo |
| Fibra | `Fiber_height`, `Fiber_base`, `Fiber_total_length` | mm |
| Volumen de fibra | `Fiber_total_volume` | mm³ |
| Pivote | `Pivot_height`, `Pivot_radius`, `Pivot_total_number` | mm / conteo |
| Volumen de pivotes | `Pivot_total_volume` | mm³ |
| Espécimen | `Sample_total_volume` | mm³ |

`Case_n` se conserva únicamente para trazabilidad. `Pivot_radius` es constante en la versión analizada; el pipeline elimina constantes dentro de cada fold.

No se usan como predictores `Ultimate_Load`, desplazamientos residuales, desplazamiento máximo ni energías. Son respuestas obtenidas durante o después del ensayo y producirían fuga de información.

### Target

| Aspecto | Definición |
|---|---|
| Nombre | **Carga de primera falla** |
| Columna | `First_Failure_Load_N` |
| Significado | Fuerza registrada en el primer evento localizado de daño o rotura durante el ensayo de tracción |
| Tipo | Variable numérica continua |
| Unidad | Newton (N) |
| Faltantes | Un target ausente se excluye y se registra; nunca se imputa |
| Diferencia esencial | No es carga última, desplazamiento, energía, esfuerzo ni una clase “falla/no falla” |

La definición experimental exacta del primer evento debe verificarse contra la documentación de la campaña antes de cerrar la tesis.

### Output

El modelo devuelve una estimación de `First_Failure_Load_N` en newtons. Cada ejecución completa también debe producir:

| Salida | Contenido |
|---|---|
| Predicciones OOF | Observado, predicho, residuo, modelo y fold por caso |
| Métricas | MAE, RMSE, MedAE y R² por fold y agregadas |
| Comparación | Diferencia de MAE respecto del baseline de mediana |
| Modelo | Pipeline ajustado, parámetros seleccionados y versión de datos |
| Interpretación | Coeficientes o importancia por permutación según el estimador |
| Trazabilidad | Configuración, índices de partición, hashes, logs y manifiesto |
| Demostrador | Interfaz para ingresar una geometría y consultar predicción, alcance y trazabilidad |

### Variables de estudio

| Rol metodológico | Variables | Uso |
|---|---|---|
| Dependiente | Carga de primera falla | Respuesta que se desea predecir |
| Independientes | Geometría de fibras, pivotes, celdas y volúmenes | Entradas conocidas antes del ensayo |
| Identificación | Número de caso | Trazabilidad; nunca entra al modelo |
| Contexto experimental | Material, fabricación SLS y ensayo de tracción | Define el dominio de aplicación |
| Control de dependencia | Configuración, réplica o familia, si la fuente las confirma | Evita separar casos relacionados entre train y evaluación |
| Respuestas secundarias | Carga última, desplazamientos y energías | Análisis complementario; excluidas del modelo principal |

## Modelos

Los seis candidatos se compararán en las mismas particiones. El preprocesamiento y el tuneo se ajustan exclusivamente con entrenamiento.

| Rol | Modelo | Propósito |
|---:|---|---|
| Referencia | Baseline de valor central | Verificar que el aprendizaje automático supere una regla elemental |
| 1 | Ridge | Modelo lineal regularizado, estable ante colinealidad |
| 2 | Elastic Net | Regularización y reducción del efecto de variables redundantes |
| 3 | SVR con kernel RBF | Relación no lineal suave con control de complejidad |
| 4 | Árbol de regresión | Reglas e interacciones no lineales interpretables |
| 5 | Bosque aleatorio | Promedio de árboles para reducir inestabilidad |
| 6 | Proceso gaussiano RBF | Relación no lineal apropiada para datos experimentales limitados |

El **baseline** no compite como solución final. Predice para cada caso el valor típico aprendido solo en el entrenamiento. Se usa la mediana porque cambia menos ante cargas extremas: si un modelo no reduce su error, no aporta evidencia de aprendizaje útil.

Como línea de ampliación se consideran modelos de **deep learning tabular**: MLP, TabNet y FT-Transformer. La MLP ya dispone de un constructor experimental; TabNet y FT-Transformer son propuestas futuras. Solo se activarán si la ampliación del dataset y la validación justifican su complejidad. No se presupone un modelo ganador.

## Métricas

Las métricas se calculan con predicciones fuera de muestra. Sea \(y_i\) la fuerza real y \(\hat{y}_i\) la predicción.

| Métrica | Cálculo | Unidad | Interpretación |
|---|---|---|---|
| **MAE** | \(\frac{1}{n}\sum |y_i-\hat{y}_i|\) | N | Error medio directamente interpretable; **criterio principal de selección** |
| **RMSE** | \(\sqrt{\frac{1}{n}\sum(y_i-\hat{y}_i)^2}\) | N | Penaliza con mayor intensidad los errores grandes |
| **MedAE** | \(\operatorname{mediana}(|y_i-\hat{y}_i|)\) | N | Error típico, menos sensible a observaciones extremas |
| **R² OOF** | \(1-\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar y)^2}\) | Sin unidad | 1 es perfecto; 0 equivale a la referencia de la media; puede ser negativo |
| **Mejora sobre baseline** | \(MAE_{baseline}-MAE_{modelo}\) | N | Un valor positivo indica reducción del error respecto de la mediana |

El MAE selecciona hiperparámetros en la validación interna. La evaluación externa informa todas las métricas, su variación entre folds y los residuos. Accuracy, precision, recall y F1 no corresponden porque el target es continuo.

## Flujo reproducible

```mermaid
flowchart TB
    subgraph A["A. Evidencia experimental"]
        direction LR
        A1["Fuente MATLAB<br/>geometría + respuesta"] --> A2["Ingesta<br/>hash + manifiesto"] --> A3["Curaduría<br/>roles + unidades + calidad"]
    end

    subgraph B["B. Exploración y protocolo"]
        direction LR
        B1["EDA avanzado<br/>univariado + bivariado<br/>multivariado + atípicos"] --> B2["Dependencia<br/>grupos + réplicas"] --> B3["Partición externa<br/>casos reservados"]
    end

    subgraph C["C. Modelado sin fuga"]
        direction LR
        C1["Train externo"] --> C2["CV interna<br/>preprocesamiento + tuneo"] --> C3["Reajuste"] --> C4["Predicción OOF"]
    end

    subgraph D["D. Evaluación y entrega"]
        direction LR
        D1["Comparación<br/>baseline + candidatos"] --> D2["Métricas<br/>MAE + RMSE + MedAE + R²"] --> D3["Interpretación"] --> D4["Modelo + reporte<br/>+ trazabilidad"]
    end

    A3 --> B1
    B3 --> C1
    C4 --> D1

    classDef evidence fill:#e5f5f2,stroke:#087d7f,color:#17324d
    classDef analysis fill:#eaf0f7,stroke:#345d7e,color:#17324d
    classDef model fill:#fff3d6,stroke:#b7791f,color:#5f4314
    classDef output fill:#f2eafa,stroke:#76519b,color:#39264f
    class A1,A2,A3 evidence
    class B1,B2,B3 analysis
    class C1,C2,C3,C4 model
    class D1,D2,D3,D4 output
```

El fold externo simula casos no vistos. Dentro de su entrenamiento, la validación interna ajusta preprocesamiento, modelo e hiperparámetros. El fold externo se usa una sola vez para medir generalización.

## Protocolo de evaluación

La evaluación separa selección y medición para evitar resultados optimistas:

1. Se reserva un fold externo que representa casos no vistos.
2. Solo con el entrenamiento externo se ajustan imputación, eliminación de constantes y escalamiento.
3. La validación interna compara modelos y realiza el tuneo.
4. El procedimiento seleccionado se reajusta con todo el entrenamiento externo.
5. Se predice una sola vez el fold externo.
6. Las predicciones OOF se reúnen para calcular métricas y residuos.

| Riesgo | Control aplicado |
|---|---|
| Fuga por variables posensayo | Carga última, desplazamientos y energías quedan fuera del input |
| Fuga por preprocesamiento | Cada transformación se ajusta dentro del fold de entrenamiento |
| Optimismo por tuneo | Selección interna y evaluación externa permanecen separadas |
| Dependencia entre casos | Réplicas o familias confirmadas se mantienen en el mismo grupo |
| Sobreajuste | Regularización, baseline, modelos de complejidad distinta y resultados por fold |
| Interpretación incorrecta | Importancia predictiva se contrasta con mecanismos físicos y no se declara causalidad |

## Etapas ejecutadas

| Etapa | Evidencia visible | Estado |
|---:|---|---|
| 1. Ingesta, curaduría y calidad | [Notebook](notebooks/01_Ingesta_Curaduria_y_Calidad.ipynb) · [Reporte HTML](reports/01_Ingesta_Curaduria_y_Calidad.html) | Ejecutado; verifica hash, esquema, tipos, faltantes, duplicados, constantes, exclusiones y salidas |
| 2. EDA avanzado | [Notebook](notebooks/02_EDA_Avanzado.ipynb) · [Reporte HTML](reports/02_EDA_Avanzado.html) | Ejecutado; contiene análisis univariado, bivariado, multivariado y detección de atípicos |

La fuente cruda está en [data/raw](data/raw/), la extracción revisable en [data/interim/preliminary](data/interim/preliminary/) y la tabla analítica en [data/processed/preliminary](data/processed/preliminary/). La explicación de cada nivel está en [data/README.md](data/README.md).

## EDA avanzado

El [notebook ejecutado](notebooks/02_EDA_Avanzado.ipynb) y su [reporte HTML](reports/02_EDA_Avanzado.html) separan cuatro niveles de análisis:

| Nivel | Análisis realizados | Insight esperado |
|---|---|---|
| Univariado | rango, cuartiles, dispersión, CV, asimetría, histogramas, ECDF y cajas | escala del target, variables constantes, niveles discretos y extremos |
| Atípicos | cercas IQR, puntuación robusta MAD y revisión por identificador | casos que requieren contraste con la fuente; no se eliminan automáticamente |
| Bivariado | Pearson, Spearman, intervalos bootstrap, dispersión predictor–target y comparación por celdas | forma y estabilidad de asociaciones geométricas con la carga |
| Multivariado | matriz de correlación, pares con alta redundancia, VIF, PCA y distancia de Mahalanobis | colinealidad, dimensión efectiva del diseño y combinaciones geométricas inusuales |

La **calidad de datos** se resuelve primero en el cuaderno 01: procedencia, esquema, faltantes, duplicados, constantes y exclusiones. El EDA usa la tabla ya curada y añade controles físicos y temporales para comprobar que ninguna respuesta posensayo entre como predictor.

El análisis detecta un radio de pivote constante y relaciones fuertes entre medidas derivadas de la misma geometría. Estas asociaciones orientan la selección y regularización dentro de cada fold, pero no demuestran causalidad ni desempeño fuera de muestra.

## Propuesta de artefacto

El entregable aplicado se denomina **Demostrador de predicción estructural**. La [versión HTML preliminar](demo/index.html) ya permite ingresar una geometría, revisar el dominio experimental y consultar los datos y notebooks. Cuando el modelo quede validado, la interfaz entregará:

- carga de primera falla estimada en N;
- modelo y versión de datos utilizados;
- comparación con el valor central de referencia;
- explicación de las variables más influyentes;
- advertencia cuando la geometría esté fuera del dominio estudiado.

El demostrador presentará resultados del modelo ya validado; no realizará entrenamiento ni sustituirá un ensayo físico.

```mermaid
flowchart LR
    U["Usuario<br/>geometría"] --> V["Validación<br/>unidades + dominio"]
    V --> API["Servicio de inferencia<br/>Python"]
    API --> M["Pipeline versionado<br/>preproceso + modelo"]
    M --> R["Respuesta<br/>carga N + alcance"]
    R --> UI["Demostrador HTML<br/>resultado + explicación"]
```

| Componente | Estado | Responsabilidad |
|---|---|---|
| Interfaz HTML | Preliminar disponible | Captura geometría, valida dominio y presenta el flujo |
| Pipeline Python | Implementado | Preprocesamiento, modelos, evaluación y serialización |
| Modelo validado | Pendiente | Inferencia de primera falla con procedimiento seleccionado |
| Servicio de inferencia | Propuesto | Conectar el pipeline versionado con la interfaz |
| Explicación | Implementación parcial | Coeficientes o permutación según el modelo ganador |

## Estructura del proyecto

La estructura separa evidencia, decisiones metodológicas, implementación y resultados. Esta separación permite rastrear cada predicción hasta su fuente.

```text
T02_MIA14_SeccionB/
├── config/
│   ├── data_schema.yml
│   └── model_config.yml
├── data/
│   ├── raw/dati_campagna_venditti.m
│   ├── interim/preliminary/
│   ├── processed/preliminary/
│   └── README.md
├── demo/
│   └── index.html
├── notebooks/
│   ├── 01_Ingesta_Curaduria_y_Calidad.ipynb
│   ├── 02_EDA_Avanzado.ipynb
│   ├── 03_Diseno_de_Particiones_y_Preprocesamiento.ipynb
│   ├── 04_Baseline_de_Referencia.ipynb
│   ├── 05_Entrenamiento_Ajuste_y_Validacion.ipynb
│   └── 06_Evaluacion_Final_e_Interpretabilidad.ipynb
├── reports/
│   ├── 01_Ingesta_Curaduria_y_Calidad.html
│   └── 02_EDA_Avanzado.html
├── src/
│   ├── importar_matlab.py
│   ├── ingesta.py
│   ├── preprocesamiento.py
│   ├── particiones.py
│   ├── modelos.py
│   ├── modelo_baseline.py
│   ├── ajuste.py
│   ├── validacion.py
│   ├── interpretabilidad.py
│   └── experimento.py
├── tests/
│   └── test_workflow.py
├── results/
├── logs/
├── slides/
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Datos

| Ruta | Responsabilidad | Regla |
|---|---|---|
| `data/raw/` | Fuente experimental original | Se conserva sin modificaciones |
| `data/interim/preliminary/` | Tabla extraída, hash y manifiesto | Etapa intermedia visible antes de seleccionar variables |
| `data/processed/preliminary/` | Tabla analítica y reporte de calidad | Versión preliminar para revisión, todavía no definitiva |
| `results/` | Predicciones, métricas, modelos e interpretación | Una carpeta nueva por experimento |
| `logs/` | Eventos, advertencias y fallos | Complementan los manifiestos |

### Demostrador

| Ruta | Responsabilidad |
|---|---|
| `demo/index.html` | Prototipo interactivo y autónomo; funciona sin instalar dependencias web |
| `data/processed/preliminary/` | Proporciona rangos verificables para controlar el dominio de entrada |
| Futuro servicio de inferencia | Cargará el pipeline validado y devolverá predicción y explicación |

**Interim** significa etapa intermedia: traduce la fuente MATLAB a una tabla verificable, pero todavía no es la entrada definitiva del modelo. Las ejecuciones fechadas permanecen fuera de Git; las carpetas `preliminary/` son copias estables para que el profesor pueda revisar la evidencia.

La ampliación hacia un conjunto del orden de cientos o aproximadamente mil observaciones se presenta como **meta de adquisición**, condicionada a nuevos ensayos o simulaciones físicamente validadas. No se declara como volumen actual ni se crearán filas duplicadas o sintéticas solo para alcanzar una cifra.

### Configuración

| Archivo | Responsabilidad |
|---|---|
| `config/data_schema.yml` | Target, identificadores, predictores, unidades y reglas de curaduría |
| `config/model_config.yml` | Particiones, seed, modelos, rejillas y autorización de entrenamiento |

### Código científico

| Módulo | Responsabilidad |
|---|---|
| `importar_matlab.py` | Extrae la tabla `samples` sin ejecutar código MATLAB |
| `ingesta.py` | Valida la fuente, calcula SHA-256 y crea la versión intermedia |
| `preprocesamiento.py` | Construye la tabla analítica y registra exclusiones y calidad |
| `particiones.py` | Genera folds externos e internos y protege grupos relacionados |
| `modelos.py` | Define los seis candidatos y su preprocesamiento |
| `modelo_baseline.py` | Implementa la mediana y las métricas de regresión |
| `ajuste.py` | Ejecuta el tuneo exclusivamente en CV interna |
| `validacion.py` | Produce predicciones OOF y compara candidatos con el baseline |
| `interpretabilidad.py` | Extrae coeficientes e importancia por permutación |
| `experimento.py` | Orquesta el protocolo y guarda artefactos reproducibles |

### Notebooks

| Orden | Notebook | Función |
|---:|---|---|
| 1 | `01_Ingesta_Curaduria_y_Calidad.ipynb` | Procedencia, esquema, calidad, exclusiones y tabla analítica |
| 2 | `02_EDA_Avanzado.ipynb` | Análisis univariado, bivariado, multivariado y atípicos |
| 3 | `03_Diseno_de_Particiones_y_Preprocesamiento.ipynb` | Grupos, folds y transformaciones dentro de train |
| 4 | `04_Baseline_de_Referencia.ipynb` | Referencia de valor central en los folds externos |
| 5 | `05_Entrenamiento_Ajuste_y_Validacion.ipynb` | Tuneo interno y predicción externa |
| 6 | `06_Evaluacion_Final_e_Interpretabilidad.ipynb` | Métricas, residuos, estabilidad y explicación |

Los notebooks documentan la investigación. La lógica reutilizable permanece en `src/` y las pruebas de contratos están en `tests/`.

## Controles metodológicos

- El target y los identificadores nunca se usan como predictores.
- El target ausente no se imputa.
- Imputación de predictores, eliminación de constantes y escalamiento se ajustan dentro de cada fold.
- Los grupos o réplicas deben permanecer juntos cuando la fuente confirme su dependencia.
- Los hiperparámetros se eligen en CV interna; el fold externo no participa en esa decisión.
- Todos los modelos y el baseline usan las mismas particiones.
- La interpretación predictiva no se presenta como causalidad física.

## Contexto físico y fuente

La solicitación externa es tracción. La arquitectura distribuye esa acción mediante extensión y flexión de fibras, rotación o torsión de pivotes y cambio angular de las celdas. El target corresponde al inicio del daño, no necesariamente al máximo de toda la curva fuerza–desplazamiento.

La fuente es la campaña experimental documentada por Enrico Venditti y el archivo MATLAB proporcionado para uso académico por Emilio Turco. El material es poliamida fabricada por SLS. Su interés para Perú reside en el estudio de manufactura aditiva y estructuras ligeras; cualquier transferencia requiere verificar material, proceso, orientación de fabricación y protocolo local.

## Entregables de tesis

| Entregable | Contenido verificable |
|---|---|
| Dataset preliminar | Fuente raw, extracción interim, tabla processed y reportes de calidad |
| Código científico | Módulos de ingesta, partición, modelado, tuneo, validación e interpretación |
| Evidencia exploratoria | Notebooks ejecutados de ingesta, curaduría y EDA |
| Evaluación predictiva | Predicciones OOF, baseline, métricas, residuos y estabilidad entre folds |
| Modelo versionado | Pipeline completo, configuración, hashes y manifiesto del experimento |
| Artefacto aplicado | Demostrador HTML y posterior conexión con el servicio de inferencia |
| Documentación | README técnico, esquema de datos, protocolo y limitaciones de uso |

## Estado y siguientes hitos

| Fase | Estado | Siguiente decisión |
|---|---|---|
| Ingesta y curaduría | Ejecutada | Conciliar definiciones experimentales pendientes |
| EDA | Ejecutado | Formalizar tratamiento de redundancia geométrica |
| Datos preliminares | Publicados en el repositorio | Confirmar condiciones para publicación abierta |
| Protocolo de partición | Pendiente de aprobación | Confirmar grupos, réplicas y número de folds |
| Comparación de modelos | Preparada | Ejecutar validación anidada después de aprobar el protocolo |
| Demostrador HTML | Preliminar disponible | Conectar el modelo validado mediante servicio de inferencia |

## Ejecución

Desde la raíz del proyecto:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Ingesta y curaduría:

```bash
python src/ingesta.py --input data/raw/dati_campagna_venditti.m
python src/preprocesamiento.py --input data/interim/VERSION/dataset_interim.csv
```

Inspección sin entrenamiento y pruebas:

```bash
python -m src.experimento --check
python -m pytest -q
```

El entrenamiento solo se habilita después de revisar grupos, folds y rejillas en `config/model_config.yml`:

```bash
python -m src.experimento --run
```

## Referencias

- Aylas Barranca, M. D. (2026). *Análisis predictivo e interpretable de la respuesta mecánica de estructuras pantográficas mediante aprendizaje automático supervisado aplicado a datos experimentales*. Plan de tesis, UNI.
- Venditti, E. (2026). *Analisi strutturale per lo studio del danneggiamento in materiali e strutture complesse*. Università degli Studi di Sassari.
- Turco, E., Golaszewski, M., Giorgio, I. y Placidi, L. (2017). *Can a Hencky-Type Model Predict the Mechanical Behaviour of Pantographic Lattices?*
- Turco, E., Misra, A., Sarikaya, R. y Lekszycki, T. (2019). *Quantitative analysis of deformation mechanisms in pantographic substructures: experiments and modeling*. *Continuum Mechanics and Thermodynamics*, 31, 209–223.

## Licencia y privacidad

El código se distribuye para uso académico según `LICENSE`. El repositorio incluye la versión preliminar de datos para revisión académica. Antes de una publicación abierta deben confirmarse las condiciones de distribución de la fuente. Los PDF, referencias privadas, ejecuciones fechadas y modelos ajustados permanecen excluidos mediante `.gitignore`.
