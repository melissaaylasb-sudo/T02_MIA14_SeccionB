# Diccionario científico de los datos

## Consulta rápida

**Target: `First_Failure_Load_N`.** Fuerza, en newtons, asociada a la primera rotura identificada de una fibra o un pivote. Es la respuesta que se buscará estimar a partir de la geometría previa al ensayo; no representa la rotura completa ni necesariamente la fuerza máxima.

**INPUT** identifica una entrada candidata disponible antes del ensayo, no una selección definitiva. **OUTPUT** identifica una respuesta experimental; **OUTPUT-target**, la respuesta objetivo. Un **metadato** identifica u organiza los registros y no describe su comportamiento mecánico.

### Geometría e identificador

Estos nombres corresponden a columnas reales de [`dataset_interim.csv`](../data/interim/preliminary/dataset_interim.csv).

| Nombre exacto | Significado sencillo | Unidad | Tipo de dato | Rol |
|---|---|---|---|---|
| `Case_n` | Identificador que vincula la geometría con las respuestas del mismo caso. | Sin unidad | Entero nominal (`int64`) | metadato |
| `n_cells_Y` | Número de celdas en la dirección Y del retículo. | Conteo | Entero discreto¹ | INPUT |
| `Fiber_height` | Altura de la sección de una fibra. | mm | Real (`float64`) | INPUT |
| `Fiber_base` | Base o ancho de la sección de una fibra. | mm | Real (`float64`) | INPUT |
| `Fiber_total_length` | Longitud total declarada del conjunto de fibras. | mm | Real (`float64`) | INPUT |
| `Fiber_total_volume` | Volumen declarado del conjunto de fibras. | mm³ | Real (`float64`) | INPUT |
| `Pivot_height` | Altura del pivote que conecta las fibras. | mm | Real (`float64`) | INPUT |
| `Pivot_radius` | Radio de la sección circular del pivote. | mm | Real (`float64`), constante | INPUT |
| `Pivot_total_number` | Número total de pivotes de la estructura. | Conteo | Entero discreto¹ | INPUT |
| `Pivot_total_volume` | Volumen declarado del conjunto de pivotes. | mm³ | Real (`float64`) | INPUT |
| `Sample_total_volume` | Volumen global del espécimen registrado a partir de su geometría CAD. | mm³ | Real (`float64`) | INPUT |

¹ Los conteos tienen significado entero; la lectura actual del CSV los almacena como `float64`. `n_cells_Y` describe la arquitectura interna, no el tamaño exterior. `Sample_total_volume` se conserva separado de la suma de los volúmenes de fibras y pivotes.

`Fiber_base` **existe en la fuente, la tabla intermedia y el EDA**, pero no está seleccionada en `columns.predictors` del [`config/data_schema.yml`](../config/data_schema.yml) actual. `Pivot_radius` vale **0.5 mm** en el dominio observado: se conserva como dato geométrico, aunque su constancia impide estudiar una asociación estadística con la respuesta.

### Respuestas de la tabla principal

Todas las columnas siguientes están en `dataset_interim.csv`. Los valores faltantes se conservan como `NaN`; no equivalen a una fuerza o un desplazamiento de cero.

| Nombre exacto | Significado sencillo | Unidad | Tipo de dato | Rol |
|---|---|---|---|---|
| `First_Failure_Load_N` | Fuerza asociada a la primera rotura identificada de fibra o pivote. | N | Real (`float64`) | **OUTPUT-target** |
| `Ultimate_Load` | Fuerza reportada para el evento terminal o rotura final. No es necesariamente la fuerza máxima. | N | Real (`float64`) | OUTPUT |
| `Maximum_Displacement` | Máximo desplazamiento consignado en el resumen del ensayo. | mm | Real (`float64`) | OUTPUT |
| `residual_Displacement_1cycle_10mm` | Desplazamiento que permanece tras descargar el ciclo 1, con amplitud nominal de 10 mm. | mm | Real (`float64`) | OUTPUT |
| `residual_Displacement_2cycle_20mm` | Desplazamiento que permanece tras descargar el ciclo 2, con amplitud nominal de 20 mm. | mm | Real (`float64`) | OUTPUT |
| `residual_Displacement_3cycle_30mm` | Desplazamiento que permanece tras descargar el ciclo 3, con amplitud nominal de 30 mm. | mm | Real (`float64`) | OUTPUT |
| `residual_Displacement_4cycle_40mm` | Desplazamiento que permanece tras descargar el ciclo 4, con amplitud nominal de 40 mm. | mm | Real (`float64`) | OUTPUT |
| `residual_Displacement_5cycle_50mm` | Desplazamiento que permanece tras descargar el ciclo 5, con amplitud nominal de 50 mm. | mm | Real (`float64`) | OUTPUT |
| `residual_Displacement_6cycle_60mm` | Desplazamiento que permanece tras descargar el ciclo 6, con amplitud nominal de 60 mm. | mm | Real (`float64`) | OUTPUT |

Los residuales describen ciclos del mismo caso, no especímenes independientes. Las amplitudes del nombre son referencias nominales del ciclo; el valor de la columna es el desplazamiento residual medido.

### Energías del archivo MATLAB

Estas matrices proceden de [`dati_campagna_venditti.m`](../data/raw/dati_campagna_venditti.m) y se extraen por separado en el EDA: **no son columnas del CSV principal**. La unidad mJ está corroborada con la tesis; el script MATLAB no la declara explícitamente.

| Nombre en la fuente o componente | Significado sencillo | Unidad | Tipo de dato | Rol |
|---|---|---|---|---|
| `Energy_Dissipated` | Energías disipadas tabuladas por caso y ciclo de carga–descarga. | mJ | Matriz de reales (`float64`) | OUTPUT |
| `Energy_Dissipated(:,k)` | Energía disipada del ciclo `k`; `k = 1, …, 7`. Python conserva los componentes como `cycle_1` a `cycle_7`. | mJ | Real por caso y ciclo (`float64`) | OUTPUT |
| `Energy_Fracture` | Energías tabuladas asociadas a los eventos de fractura de cada caso. | mJ | Matriz de reales (`float64`) | OUTPUT |
| `Energy_Fracture(:,j)` | Energía del evento ordinal `j`; `j = 1, …, 13`. Python conserva los componentes como `event_1` a `event_13`. | mJ | Real por caso y evento (`float64`) | OUTPUT |

En la notación MATLAB, `:` selecciona todos los casos y `k` o `j` identifica una columna. Los ciclos y eventos pertenecen al mismo caso. Una posición energética vacía no demuestra que el evento no haya ocurrido; la energía asociada a fractura tampoco se interpreta automáticamente como tenacidad o energía total del ensayo. Ninguna respuesta medida durante el ensayo es una entrada previa admisible para predecir la primera falla.

## Definiciones y trazabilidad detalladas

Cada entrada explica el campo original, su significado físico y su admisibilidad predictiva. El [CSV](diccionario_datos.csv) conserva los diecisiete campos de definición para consulta automática. Se incluyen las columnas escalares, las matrices energéticas y sus componentes, y las derivadas geométricas o de respuesta utilizadas/propuestas.

**INPUT** es una característica disponible antes del ensayo; **OUTPUT** es una respuesta del ensayo. Una **derivada** se construye mediante una fórmula explícita y hereda el momento de disponibilidad de sus ingredientes. Un **metadato** describe procedencia u organización. Las tablas MATLAB samples y energy son contenedores, no variables físicas. Los alias cycle_k y event_j no constituyen nuevos especímenes.

Un dato **medido** procede de instrumentación o detección experimental; aquí se recibe como resumen secundario, sin señal cruda ni incertidumbre instrumental específica. Un dato **calculado** procede de geometría CAD o una operación documentada. Un **proxy** aproxima un aspecto mecánico: b h³/12 caracteriza la sección ideal, pero no es rigidez experimental. El área b h no es automáticamente área resistente global.

Por ejemplo, relacionar altura del pivote (INPUT) con primera falla (OUTPUT) plantea una asociación geométrico-mecánica. Relacionar energía inicial con primera falla relaciona dos respuestas y no proporciona una entrada disponible para predecir antes del ensayo.

## Fuentes y alcance

- **S1:** archivo real data/raw/dati_campagna_venditti.m.
- **S2:** Enrico Venditti, *Analisi strutturale per lo studio del danneggiamento in materiali e strutture complesse*, Universidad de Sassari, año académico 2024/2025. Capítulo 9 y apéndice A; páginas por posición PDF desde la portada.
- **S3:** Turco, Golaszewski, Giorgio y Placidi (2017), modelo Hencky; PDF 3–4 / pp. impresas 287–288.
- **S4:** Turco, Misra, Sarikaya y Lekszycki (2019, publicación en línea 2018), mecanismos y Piola–Hencky; PDF 2–4 / pp. 210–212.

La energía en **mJ** está expresamente verificada en S2. Las geometrías presentan discordancias entre S1, la tabla general y fichas: este diccionario no autoriza sustituir valores. Véase [trazabilidad](trazabilidad_datos.md). Las dimensiones externas nominales son comunes; el número de celdas describe arquitectura interna. Los alias históricos hacen legible la evidencia previa y no deben duplicarse como nuevas mediciones.

## Índice

- [Entradas geométricas y estructurales](#entradas-geométricas-y-estructurales)
- [Respuestas mecánicas, ciclos y eventos](#respuestas-mecánicas-ciclos-y-eventos)
- [Variables derivadas](#variables-derivadas)
- [Identificadores y metadatos](#identificadores-y-metadatos)

## Entradas geométricas y estructurales

<details>
<summary><strong>n_cells_Y</strong> — Número de celdas en la dirección Y</summary>

| Campo | Definición |
|---|---|
| Nombre original | n_cells_Y |
| Nombre científico comprensible | Número de celdas en la dirección Y |
| Símbolo | n_y |
| Definición física | Conteo de celdas del retículo en la dirección designada Y. |
| Unidad | conteo |
| Tipo de variable | entero discreto |
| Clasificación | INPUT |
| Fuente de procedencia | S1, línea 8; S2 PDF 94–95 y fichas del apéndice |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | diseño; no se recalcula |
| Dominio y restricciones | entero positivo |
| Interpretación experimental | Arquitectura interna dentro de dimensiones externas nominales comunes. |
| Relaciones con otras variables | misma familia que longitud total y número de pivotes |
| Momento de disponibilidad | antes del ensayo: geometría de diseño |
| Utilidad como predictor | candidata para arquitectura; controlar redundancia |
| Riesgo de fuga de información | bajo si se obtiene de diseño previo; no asegura validez física |
| Advertencias y limitaciones | Más celdas no significa mayor longitud externa. No es número total de celdas del espécimen. |

</details>

<details>
<summary><strong>Fiber_height</strong> — Altura de la sección de fibra</summary>

| Campo | Definición |
|---|---|
| Nombre original | Fiber_height |
| Nombre científico comprensible | Altura de la sección de fibra |
| Símbolo | h_f |
| Definición física | Dimensión nominal de la sección transversal denominada altura. |
| Unidad | mm |
| Tipo de variable | real continuo con niveles de diseño |
| Clasificación | INPUT |
| Fuente de procedencia | S1, línea 10; S2 PDF 94–95 y fichas del apéndice |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | diseño |
| Dominio y restricciones | positiva |
| Interpretación experimental | Distingue configuraciones de sección. |
| Relaciones con otras variables | junto con base define área e inercias idealizadas |
| Momento de disponibilidad | antes del ensayo: geometría de diseño |
| Utilidad como predictor | candidata geométrica |
| Riesgo de fuga de información | bajo si se obtiene de diseño previo; no asegura validez física |
| Advertencias y limitaciones | No asignar eje global de flexión sin planos/CAD; no es altura total del espécimen. |

</details>

<details>
<summary><strong>Fiber_base</strong> — Base de la sección de fibra</summary>

| Campo | Definición |
|---|---|
| Nombre original | Fiber_base |
| Nombre científico comprensible | Base de la sección de fibra |
| Símbolo | b_f |
| Definición física | Anchura nominal de la sección denominada base. |
| Unidad | mm |
| Tipo de variable | real continuo |
| Clasificación | INPUT |
| Fuente de procedencia | S1, línea 12; S2 PDF 94–95 y fichas del apéndice |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | diseño |
| Dominio y restricciones | positiva |
| Interpretación experimental | Describe forma y área de sección. |
| Relaciones con otras variables | área b_f h_f; dependencia con arquitectura |
| Momento de disponibilidad | antes del ensayo: geometría de diseño |
| Utilidad como predictor | candidata sujeta a conciliación documental |
| Riesgo de fuga de información | bajo si se obtiene de diseño previo; no asegura validez física |
| Advertencias y limitaciones | Discrepa con tabla general y fichas. Variable real presente en la fuente y el snapshot; el esquema actual no la selecciona como predictor. |

</details>

<details>
<summary><strong>Fiber_total_length</strong> — Longitud total declarada de fibras</summary>

| Campo | Definición |
|---|---|
| Nombre original | Fiber_total_length |
| Nombre científico comprensible | Longitud total declarada de fibras |
| Símbolo | L_f |
| Definición física | Suma de longitudes de elementos de fibra registrada. |
| Unidad | mm |
| Tipo de variable | real continuo |
| Clasificación | INPUT |
| Fuente de procedencia | S1, línea 14; S2 PDF 94–95 y fichas del apéndice |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | geometría de diseño; algoritmo no suministrado |
| Dominio y restricciones | positiva |
| Interpretación experimental | Resume material lineal y arquitectura. |
| Relaciones con otras variables | redundancia de familia con n_y |
| Momento de disponibilidad | antes del ensayo: geometría de diseño |
| Utilidad como predictor | representación alternativa de arquitectura |
| Riesgo de fuga de información | bajo si se obtiene de diseño previo; no asegura validez física |
| Advertencias y limitaciones | No es longitud útil entre mordazas; diferencias de versión documental identificadas. |

</details>

<details>
<summary><strong>Fiber_total_volume</strong> — Volumen total declarado de fibras</summary>

| Campo | Definición |
|---|---|
| Nombre original | Fiber_total_volume |
| Nombre científico comprensible | Volumen total declarado de fibras |
| Símbolo | V_f |
| Definición física | Volumen asignado al conjunto de fibras. |
| Unidad | mm³ |
| Tipo de variable | real continuo |
| Clasificación | INPUT |
| Fuente de procedencia | S1, línea 16; S2 PDF 94–95 y fichas del apéndice |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | diseño; contraste ideal b_f h_f L_f |
| Dominio y restricciones | positivo |
| Interpretación experimental | Asignación de volumen a fibras. |
| Relaciones con otras variables | V_f + V_p constante en archivo |
| Momento de disponibilidad | antes del ensayo: geometría de diseño |
| Utilidad como predictor | alternativa a V_p, controlar identidad algebraica |
| Riesgo de fuga de información | bajo si se obtiene de diseño previo; no asegura validez física |
| Advertencias y limitaciones | No asumir volumen CAD neto ni volumen medido. |

</details>

<details>
<summary><strong>Pivot_height</strong> — Altura de pivote</summary>

| Campo | Definición |
|---|---|
| Nombre original | Pivot_height |
| Nombre científico comprensible | Altura de pivote |
| Símbolo | h_p |
| Definición física | Altura nominal del conector cilíndrico entre familias de fibras. |
| Unidad | mm |
| Tipo de variable | real continuo con niveles de diseño |
| Clasificación | INPUT |
| Fuente de procedencia | S1, línea 18; S2 PDF 94–95 y fichas del apéndice |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | diseño |
| Dominio y restricciones | positiva |
| Interpretación experimental | Geometría local de conexión. |
| Relaciones con otras variables | volumen ideal π r_p² h_p N_p |
| Momento de disponibilidad | antes del ensayo: geometría de diseño |
| Utilidad como predictor | candidata de conexión |
| Riesgo de fuga de información | bajo si se obtiene de diseño previo; no asegura validez física |
| Advertencias y limitaciones | Discrepancias de asignación en ciertos casos; no es módulo torsional. |

</details>

<details>
<summary><strong>Pivot_radius</strong> — Radio del pivote</summary>

| Campo | Definición |
|---|---|
| Nombre original | Pivot_radius |
| Nombre científico comprensible | Radio del pivote |
| Símbolo | r_p |
| Definición física | Radio nominal de la sección circular del conector. |
| Unidad | mm |
| Tipo de variable | real continuo constante en la fuente |
| Clasificación | INPUT |
| Fuente de procedencia | S1, línea 20; S2 PDF 94–95 y fichas del apéndice |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | diseño |
| Dominio y restricciones | positivo |
| Interpretación experimental | Define sección geométrica del pivote. |
| Relaciones con otras variables | A_p=π r_p² |
| Momento de disponibilidad | antes del ensayo: geometría de diseño |
| Utilidad como predictor | sin información discriminante mientras sea constante |
| Riesgo de fuga de información | bajo si se obtiene de diseño previo; no asegura validez física |
| Advertencias y limitaciones | No estimar correlación ni fabricar variación de una constante. |

</details>

<details>
<summary><strong>Pivot_total_number</strong> — Número total de pivotes</summary>

| Campo | Definición |
|---|---|
| Nombre original | Pivot_total_number |
| Nombre científico comprensible | Número total de pivotes |
| Símbolo | N_p |
| Definición física | Cantidad declarada de conectores en el espécimen. |
| Unidad | conteo |
| Tipo de variable | entero discreto |
| Clasificación | INPUT |
| Fuente de procedencia | S1, línea 22; S2 PDF 94–95 y fichas del apéndice |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | conteo de diseño |
| Dominio y restricciones | entero positivo |
| Interpretación experimental | Resume conectividad del retículo. |
| Relaciones con otras variables | misma familia que n_y y L_f |
| Momento de disponibilidad | antes del ensayo: geometría de diseño |
| Utilidad como predictor | alternativa de arquitectura |
| Riesgo de fuga de información | bajo si se obtiene de diseño previo; no asegura validez física |
| Advertencias y limitaciones | No es número de pivotes rotos ni efecto causal separado del diseño. |

</details>

<details>
<summary><strong>Pivot_total_volume</strong> — Volumen total declarado de pivotes</summary>

| Campo | Definición |
|---|---|
| Nombre original | Pivot_total_volume |
| Nombre científico comprensible | Volumen total declarado de pivotes |
| Símbolo | V_p |
| Definición física | Volumen asignado al conjunto de conexiones. |
| Unidad | mm³ |
| Tipo de variable | real continuo |
| Clasificación | INPUT |
| Fuente de procedencia | S1, línea 24; S2 PDF 94–95 y fichas del apéndice |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | diseño; contraste ideal π r_p² h_p N_p |
| Dominio y restricciones | positivo |
| Interpretación experimental | Resume volumen de conexiones. |
| Relaciones con otras variables | complemento de V_f bajo suma constante |
| Momento de disponibilidad | antes del ensayo: geometría de diseño |
| Utilidad como predictor | candidata alternativa evitando redundancia |
| Riesgo de fuga de información | bajo si se obtiene de diseño previo; no asegura validez física |
| Advertencias y limitaciones | Contraste cilíndrico es diagnóstico; no autoriza reemplazar datos. |

</details>

<details>
<summary><strong>Sample_total_volume</strong> — Volumen declarado del espécimen</summary>

| Campo | Definición |
|---|---|
| Nombre original | Sample_total_volume |
| Nombre científico comprensible | Volumen declarado del espécimen |
| Símbolo | V_s |
| Definición física | Volumen global registrado; en fichas concordantes corresponde a Solidworks Sample total volume. |
| Unidad | mm³ |
| Tipo de variable | real continuo |
| Clasificación | INPUT |
| Fuente de procedencia | S1, línea 26; S2 PDF 94–95 y fichas del apéndice |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | geometría CAD según fichas concordantes; no se recalcula |
| Dominio y restricciones | positivo |
| Interpretación experimental | Caracteriza geometría global en escala de fuente. |
| Relaciones con otras variables | no equivale necesariamente a V_f+V_p |
| Momento de disponibilidad | antes del ensayo: geometría de diseño |
| Utilidad como predictor | candidata con conciliación de versiones |
| Riesgo de fuga de información | bajo si se obtiene de diseño previo; no asegura validez física |
| Advertencias y limitaciones | No llamarlo masa, volumen envolvente ni volumen neto medido. Ver discrepancias en trazabilidad. |

</details>

## Respuestas mecánicas, ciclos y eventos

<details>
<summary><strong>First_Failure_Load_N</strong> — Carga de primera falla</summary>

| Campo | Definición |
|---|---|
| Nombre original | First_Failure_Load_N |
| Nombre científico comprensible | Carga de primera falla |
| Símbolo | F_FF |
| Definición física | Fuerza experimental asociada al primer evento de rotura identificado de una fibra o un pivote. |
| Unidad | N |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, línea 28; S2 PDF 96 y apéndice A |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | valor resumen tabulado por la fuente |
| Dominio y restricciones | fuerza según convención de tracción; conservar NaN |
| Interpretación experimental | Target: inicio de daño detectado, no colapso completo ni máximo global. |
| Relaciones con otras variables | distinta de carga terminal; asociada a primer evento energético |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | variable objetivo; no forma parte de X |
| Riesgo de fuga de información | directo si se incorpora a X o una derivada de X |
| Advertencias y limitaciones | Se usa resumen First Failure Load; difiere del primer Upper Force en casos 3, 9 y 12. No imputar etiquetas. |

</details>

<details>
<summary><strong>Ultimate_Load</strong> — Carga del evento terminal reportado</summary>

| Campo | Definición |
|---|---|
| Nombre original | Ultimate_Load |
| Nombre científico comprensible | Carga del evento terminal reportado |
| Símbolo | F_term |
| Definición física | Fuerza registrada como Ultimate Load; el protocolo la asocia a rotura total/final del espécimen. |
| Unidad | N |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, línea 30; S2 PDF 96 y apéndice A |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | resumen de fuente, no max(F) |
| Dominio y restricciones | no exigir F_term ≥ F_FF |
| Interpretación experimental | Respuesta en el evento terminal tras la secuencia de daño. |
| Relaciones con otras variables | comparación con F_FF y ratio terminal/primera |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | No es máximo global. Hay diferencias entre resumen y último pico; en caso 27 el resumen 70.31 coincide con desplazamiento máximo, pero el único pico PDF239 es 191.39 N. Requiere aclaración, no corrección automática. |

</details>

<details>
<summary><strong>Maximum_Displacement</strong> — Desplazamiento máximo reportado</summary>

| Campo | Definición |
|---|---|
| Nombre original | Maximum_Displacement |
| Nombre científico comprensible | Desplazamiento máximo reportado |
| Símbolo | u_max |
| Definición física | Máximo desplazamiento consignado en el resumen del ensayo. |
| Unidad | mm |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, línea 32; S2 PDF 96 y apéndice A |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | resumen experimental |
| Dominio y restricciones | no negativo en convención tabulada |
| Interpretación experimental | Amplitud global reportada. |
| Relaciones con otras variables | comparación con cargas y ciclos |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | No es deformación unitaria; faltan longitud útil individual y serie numérica para recalcular máximos. |

</details>

<details>
<summary><strong>residual_Displacement_1cycle_10mm</strong> — Desplazamiento residual del ciclo 1, amplitud 10 mm</summary>

| Campo | Definición |
|---|---|
| Nombre original | residual_Displacement_1cycle_10mm |
| Nombre científico comprensible | Desplazamiento residual del ciclo 1, amplitud 10 mm |
| Símbolo | u_res,1 |
| Definición física | Desplazamiento restante al terminar la descarga del ciclo indicado. |
| Unidad | mm |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, línea 34; S2 PDF 96 y apéndice A |
| Nivel de observación | espécimen × ciclo |
| Fórmula de construcción | resumen experimental por ciclo |
| Dominio y restricciones | NaN no es cero; contraste con amplitud nominal |
| Interpretación experimental | Recuperación incompleta tras carga y descarga. |
| Relaciones con otras variables | fracción residual u_res,1/10 |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Mediciones relacionadas dentro del espécimen; NaN no distingue ciclo no alcanzado, pérdida o no aplicable. |

</details>

<details>
<summary><strong>residual_Displacement_2cycle_20mm</strong> — Desplazamiento residual del ciclo 2, amplitud 20 mm</summary>

| Campo | Definición |
|---|---|
| Nombre original | residual_Displacement_2cycle_20mm |
| Nombre científico comprensible | Desplazamiento residual del ciclo 2, amplitud 20 mm |
| Símbolo | u_res,2 |
| Definición física | Desplazamiento restante al terminar la descarga del ciclo indicado. |
| Unidad | mm |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, línea 36; S2 PDF 96 y apéndice A |
| Nivel de observación | espécimen × ciclo |
| Fórmula de construcción | resumen experimental por ciclo |
| Dominio y restricciones | NaN no es cero; contraste con amplitud nominal |
| Interpretación experimental | Recuperación incompleta tras carga y descarga. |
| Relaciones con otras variables | fracción residual u_res,2/20 |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Mediciones relacionadas dentro del espécimen; NaN no distingue ciclo no alcanzado, pérdida o no aplicable. |

</details>

<details>
<summary><strong>residual_Displacement_3cycle_30mm</strong> — Desplazamiento residual del ciclo 3, amplitud 30 mm</summary>

| Campo | Definición |
|---|---|
| Nombre original | residual_Displacement_3cycle_30mm |
| Nombre científico comprensible | Desplazamiento residual del ciclo 3, amplitud 30 mm |
| Símbolo | u_res,3 |
| Definición física | Desplazamiento restante al terminar la descarga del ciclo indicado. |
| Unidad | mm |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, línea 38; S2 PDF 96 y apéndice A |
| Nivel de observación | espécimen × ciclo |
| Fórmula de construcción | resumen experimental por ciclo |
| Dominio y restricciones | NaN no es cero; contraste con amplitud nominal |
| Interpretación experimental | Recuperación incompleta tras carga y descarga. |
| Relaciones con otras variables | fracción residual u_res,3/30 |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Mediciones relacionadas dentro del espécimen; NaN no distingue ciclo no alcanzado, pérdida o no aplicable. |

</details>

<details>
<summary><strong>residual_Displacement_4cycle_40mm</strong> — Desplazamiento residual del ciclo 4, amplitud 40 mm</summary>

| Campo | Definición |
|---|---|
| Nombre original | residual_Displacement_4cycle_40mm |
| Nombre científico comprensible | Desplazamiento residual del ciclo 4, amplitud 40 mm |
| Símbolo | u_res,4 |
| Definición física | Desplazamiento restante al terminar la descarga del ciclo indicado. |
| Unidad | mm |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, línea 40; S2 PDF 96 y apéndice A |
| Nivel de observación | espécimen × ciclo |
| Fórmula de construcción | resumen experimental por ciclo |
| Dominio y restricciones | NaN no es cero; contraste con amplitud nominal |
| Interpretación experimental | Recuperación incompleta tras carga y descarga. |
| Relaciones con otras variables | fracción residual u_res,4/40 |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Mediciones relacionadas dentro del espécimen; NaN no distingue ciclo no alcanzado, pérdida o no aplicable. |

</details>

<details>
<summary><strong>residual_Displacement_5cycle_50mm</strong> — Desplazamiento residual del ciclo 5, amplitud 50 mm</summary>

| Campo | Definición |
|---|---|
| Nombre original | residual_Displacement_5cycle_50mm |
| Nombre científico comprensible | Desplazamiento residual del ciclo 5, amplitud 50 mm |
| Símbolo | u_res,5 |
| Definición física | Desplazamiento restante al terminar la descarga del ciclo indicado. |
| Unidad | mm |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, línea 42; S2 PDF 96 y apéndice A |
| Nivel de observación | espécimen × ciclo |
| Fórmula de construcción | resumen experimental por ciclo |
| Dominio y restricciones | NaN no es cero; contraste con amplitud nominal |
| Interpretación experimental | Recuperación incompleta tras carga y descarga. |
| Relaciones con otras variables | fracción residual u_res,5/50 |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Mediciones relacionadas dentro del espécimen; NaN no distingue ciclo no alcanzado, pérdida o no aplicable. |

</details>

<details>
<summary><strong>residual_Displacement_6cycle_60mm</strong> — Desplazamiento residual del ciclo 6, amplitud 60 mm</summary>

| Campo | Definición |
|---|---|
| Nombre original | residual_Displacement_6cycle_60mm |
| Nombre científico comprensible | Desplazamiento residual del ciclo 6, amplitud 60 mm |
| Símbolo | u_res,6 |
| Definición física | Desplazamiento restante al terminar la descarga del ciclo indicado. |
| Unidad | mm |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, línea 44; S2 PDF 96 y apéndice A |
| Nivel de observación | espécimen × ciclo |
| Fórmula de construcción | resumen experimental por ciclo |
| Dominio y restricciones | NaN no es cero; contraste con amplitud nominal |
| Interpretación experimental | Recuperación incompleta tras carga y descarga. |
| Relaciones con otras variables | fracción residual u_res,6/60 |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Mediciones relacionadas dentro del espécimen; NaN no distingue ciclo no alcanzado, pérdida o no aplicable. |

</details>

<details>
<summary><strong>Energy_Dissipated</strong> — Energía disipada por ciclo</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Dissipated |
| Nombre científico comprensible | Energía disipada por ciclo |
| Símbolo | E_d |
| Definición física | Matriz de respuestas energéticas por espécimen y ciclo. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, línea 47; S2 tablas energéticas PDF 117–241 |
| Nivel de observación | espécimen × ciclo |
| Fórmula de construcción | agregado de fuente; límites de integración no suministrados |
| Dominio y restricciones | NaN preservado; conservar valores negativos iniciales de disipación |
| Interpretación experimental | Magnitud tabulada con unidad millijoule. |
| Relaciones con otras variables | 1 mJ = 1 N·mm; no requiere cambio de escala |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Unidad mJ verificada en S2. Integración original no suministrada: no recalcular histéresis ni tenacidad. |

</details>

<details>
<summary><strong>Energy_Dissipated(:,1)</strong> — Energía disipada por ciclo: ciclo 1</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Dissipated(:,1) |
| Nombre científico comprensible | Energía disipada por ciclo: ciclo 1 |
| Símbolo | E_d,1 |
| Definición física | Columna 1 de Energy_Dissipated; alias Python cycle_1. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, asignaciones de Energy_Dissipated; S2 apéndice A |
| Nivel de observación | espécimen × ciclo |
| Fórmula de construcción | Energy_Dissipated[Case_n,1] |
| Dominio y restricciones | finito o NaN; signo original |
| Interpretación experimental | Valor del ciclo ordinal 1. |
| Relaciones con otras variables | dependencia por Case_n; no nueva muestra |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | No deducir causa de ausencia ni tratar ciclos como especímenes independientes. |

</details>

<details>
<summary><strong>Energy_Dissipated(:,2)</strong> — Energía disipada por ciclo: ciclo 2</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Dissipated(:,2) |
| Nombre científico comprensible | Energía disipada por ciclo: ciclo 2 |
| Símbolo | E_d,2 |
| Definición física | Columna 2 de Energy_Dissipated; alias Python cycle_2. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, asignaciones de Energy_Dissipated; S2 apéndice A |
| Nivel de observación | espécimen × ciclo |
| Fórmula de construcción | Energy_Dissipated[Case_n,2] |
| Dominio y restricciones | finito o NaN; signo original |
| Interpretación experimental | Valor del ciclo ordinal 2. |
| Relaciones con otras variables | dependencia por Case_n; no nueva muestra |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | No deducir causa de ausencia ni tratar ciclos como especímenes independientes. |

</details>

<details>
<summary><strong>Energy_Dissipated(:,3)</strong> — Energía disipada por ciclo: ciclo 3</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Dissipated(:,3) |
| Nombre científico comprensible | Energía disipada por ciclo: ciclo 3 |
| Símbolo | E_d,3 |
| Definición física | Columna 3 de Energy_Dissipated; alias Python cycle_3. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, asignaciones de Energy_Dissipated; S2 apéndice A |
| Nivel de observación | espécimen × ciclo |
| Fórmula de construcción | Energy_Dissipated[Case_n,3] |
| Dominio y restricciones | finito o NaN; signo original |
| Interpretación experimental | Valor del ciclo ordinal 3. |
| Relaciones con otras variables | dependencia por Case_n; no nueva muestra |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | No deducir causa de ausencia ni tratar ciclos como especímenes independientes. |

</details>

<details>
<summary><strong>Energy_Dissipated(:,4)</strong> — Energía disipada por ciclo: ciclo 4</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Dissipated(:,4) |
| Nombre científico comprensible | Energía disipada por ciclo: ciclo 4 |
| Símbolo | E_d,4 |
| Definición física | Columna 4 de Energy_Dissipated; alias Python cycle_4. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, asignaciones de Energy_Dissipated; S2 apéndice A |
| Nivel de observación | espécimen × ciclo |
| Fórmula de construcción | Energy_Dissipated[Case_n,4] |
| Dominio y restricciones | finito o NaN; signo original |
| Interpretación experimental | Valor del ciclo ordinal 4. |
| Relaciones con otras variables | dependencia por Case_n; no nueva muestra |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | No deducir causa de ausencia ni tratar ciclos como especímenes independientes. |

</details>

<details>
<summary><strong>Energy_Dissipated(:,5)</strong> — Energía disipada por ciclo: ciclo 5</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Dissipated(:,5) |
| Nombre científico comprensible | Energía disipada por ciclo: ciclo 5 |
| Símbolo | E_d,5 |
| Definición física | Columna 5 de Energy_Dissipated; alias Python cycle_5. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, asignaciones de Energy_Dissipated; S2 apéndice A |
| Nivel de observación | espécimen × ciclo |
| Fórmula de construcción | Energy_Dissipated[Case_n,5] |
| Dominio y restricciones | finito o NaN; signo original |
| Interpretación experimental | Valor del ciclo ordinal 5. |
| Relaciones con otras variables | dependencia por Case_n; no nueva muestra |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | No deducir causa de ausencia ni tratar ciclos como especímenes independientes. |

</details>

<details>
<summary><strong>Energy_Dissipated(:,6)</strong> — Energía disipada por ciclo: ciclo 6</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Dissipated(:,6) |
| Nombre científico comprensible | Energía disipada por ciclo: ciclo 6 |
| Símbolo | E_d,6 |
| Definición física | Columna 6 de Energy_Dissipated; alias Python cycle_6. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, asignaciones de Energy_Dissipated; S2 apéndice A |
| Nivel de observación | espécimen × ciclo |
| Fórmula de construcción | Energy_Dissipated[Case_n,6] |
| Dominio y restricciones | finito o NaN; signo original |
| Interpretación experimental | Valor del ciclo ordinal 6. |
| Relaciones con otras variables | dependencia por Case_n; no nueva muestra |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | No deducir causa de ausencia ni tratar ciclos como especímenes independientes. |

</details>

<details>
<summary><strong>Energy_Dissipated(:,7)</strong> — Energía disipada por ciclo: ciclo 7</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Dissipated(:,7) |
| Nombre científico comprensible | Energía disipada por ciclo: ciclo 7 |
| Símbolo | E_d,7 |
| Definición física | Columna 7 de Energy_Dissipated; alias Python cycle_7. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, asignaciones de Energy_Dissipated; S2 apéndice A |
| Nivel de observación | espécimen × ciclo |
| Fórmula de construcción | Energy_Dissipated[Case_n,7] |
| Dominio y restricciones | finito o NaN; signo original |
| Interpretación experimental | Valor del ciclo ordinal 7. |
| Relaciones con otras variables | dependencia por Case_n; no nueva muestra |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | No deducir causa de ausencia ni tratar ciclos como especímenes independientes. |

</details>

<details>
<summary><strong>Energy_Fracture</strong> — Energía tabulada asociada a fractura</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Fracture |
| Nombre científico comprensible | Energía tabulada asociada a fractura |
| Símbolo | E_fr |
| Definición física | Matriz de respuestas energéticas por espécimen y evento. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, línea 48; S2 tablas energéticas PDF 117–241 |
| Nivel de observación | espécimen × evento |
| Fórmula de construcción | agregado de fuente; límites de integración no suministrados |
| Dominio y restricciones | NaN preservado; conservar valores negativos iniciales de disipación |
| Interpretación experimental | Magnitud tabulada con unidad millijoule. |
| Relaciones con otras variables | 1 mJ = 1 N·mm; no requiere cambio de escala |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Unidad mJ verificada en S2. Integración original no suministrada: no recalcular histéresis ni tenacidad. La etiqueta energía en curvas de fractura no distingue inequívocamente trabajo previo de energía liberada en la caída. |

</details>

<details>
<summary><strong>Energy_Fracture(:,1)</strong> — Energía tabulada asociada a fractura: evento 1</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Fracture(:,1) |
| Nombre científico comprensible | Energía tabulada asociada a fractura: evento 1 |
| Símbolo | E_fr,1 |
| Definición física | Columna 1 de Energy_Fracture; alias Python event_1. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, asignaciones de Energy_Fracture; S2 apéndice A |
| Nivel de observación | espécimen × evento |
| Fórmula de construcción | Energy_Fracture[Case_n,1] |
| Dominio y restricciones | finito o NaN; signo original |
| Interpretación experimental | Valor del evento ordinal 1. |
| Relaciones con otras variables | dependencia por Case_n; no nueva muestra |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Una posición ausente no equivale a inexistencia del evento. Caso 5 tiene cuarto evento documentado sin energía. |

</details>

<details>
<summary><strong>Energy_Fracture(:,2)</strong> — Energía tabulada asociada a fractura: evento 2</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Fracture(:,2) |
| Nombre científico comprensible | Energía tabulada asociada a fractura: evento 2 |
| Símbolo | E_fr,2 |
| Definición física | Columna 2 de Energy_Fracture; alias Python event_2. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, asignaciones de Energy_Fracture; S2 apéndice A |
| Nivel de observación | espécimen × evento |
| Fórmula de construcción | Energy_Fracture[Case_n,2] |
| Dominio y restricciones | finito o NaN; signo original |
| Interpretación experimental | Valor del evento ordinal 2. |
| Relaciones con otras variables | dependencia por Case_n; no nueva muestra |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Una posición ausente no equivale a inexistencia del evento. Caso 5 tiene cuarto evento documentado sin energía. |

</details>

<details>
<summary><strong>Energy_Fracture(:,3)</strong> — Energía tabulada asociada a fractura: evento 3</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Fracture(:,3) |
| Nombre científico comprensible | Energía tabulada asociada a fractura: evento 3 |
| Símbolo | E_fr,3 |
| Definición física | Columna 3 de Energy_Fracture; alias Python event_3. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, asignaciones de Energy_Fracture; S2 apéndice A |
| Nivel de observación | espécimen × evento |
| Fórmula de construcción | Energy_Fracture[Case_n,3] |
| Dominio y restricciones | finito o NaN; signo original |
| Interpretación experimental | Valor del evento ordinal 3. |
| Relaciones con otras variables | dependencia por Case_n; no nueva muestra |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Una posición ausente no equivale a inexistencia del evento. Caso 5 tiene cuarto evento documentado sin energía. |

</details>

<details>
<summary><strong>Energy_Fracture(:,4)</strong> — Energía tabulada asociada a fractura: evento 4</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Fracture(:,4) |
| Nombre científico comprensible | Energía tabulada asociada a fractura: evento 4 |
| Símbolo | E_fr,4 |
| Definición física | Columna 4 de Energy_Fracture; alias Python event_4. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, asignaciones de Energy_Fracture; S2 apéndice A |
| Nivel de observación | espécimen × evento |
| Fórmula de construcción | Energy_Fracture[Case_n,4] |
| Dominio y restricciones | finito o NaN; signo original |
| Interpretación experimental | Valor del evento ordinal 4. |
| Relaciones con otras variables | dependencia por Case_n; no nueva muestra |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Una posición ausente no equivale a inexistencia del evento. Caso 5 tiene cuarto evento documentado sin energía. |

</details>

<details>
<summary><strong>Energy_Fracture(:,5)</strong> — Energía tabulada asociada a fractura: evento 5</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Fracture(:,5) |
| Nombre científico comprensible | Energía tabulada asociada a fractura: evento 5 |
| Símbolo | E_fr,5 |
| Definición física | Columna 5 de Energy_Fracture; alias Python event_5. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, asignaciones de Energy_Fracture; S2 apéndice A |
| Nivel de observación | espécimen × evento |
| Fórmula de construcción | Energy_Fracture[Case_n,5] |
| Dominio y restricciones | finito o NaN; signo original |
| Interpretación experimental | Valor del evento ordinal 5. |
| Relaciones con otras variables | dependencia por Case_n; no nueva muestra |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Una posición ausente no equivale a inexistencia del evento. Caso 5 tiene cuarto evento documentado sin energía. |

</details>

<details>
<summary><strong>Energy_Fracture(:,6)</strong> — Energía tabulada asociada a fractura: evento 6</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Fracture(:,6) |
| Nombre científico comprensible | Energía tabulada asociada a fractura: evento 6 |
| Símbolo | E_fr,6 |
| Definición física | Columna 6 de Energy_Fracture; alias Python event_6. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, asignaciones de Energy_Fracture; S2 apéndice A |
| Nivel de observación | espécimen × evento |
| Fórmula de construcción | Energy_Fracture[Case_n,6] |
| Dominio y restricciones | finito o NaN; signo original |
| Interpretación experimental | Valor del evento ordinal 6. |
| Relaciones con otras variables | dependencia por Case_n; no nueva muestra |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Una posición ausente no equivale a inexistencia del evento. Caso 5 tiene cuarto evento documentado sin energía. |

</details>

<details>
<summary><strong>Energy_Fracture(:,7)</strong> — Energía tabulada asociada a fractura: evento 7</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Fracture(:,7) |
| Nombre científico comprensible | Energía tabulada asociada a fractura: evento 7 |
| Símbolo | E_fr,7 |
| Definición física | Columna 7 de Energy_Fracture; alias Python event_7. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, asignaciones de Energy_Fracture; S2 apéndice A |
| Nivel de observación | espécimen × evento |
| Fórmula de construcción | Energy_Fracture[Case_n,7] |
| Dominio y restricciones | finito o NaN; signo original |
| Interpretación experimental | Valor del evento ordinal 7. |
| Relaciones con otras variables | dependencia por Case_n; no nueva muestra |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Una posición ausente no equivale a inexistencia del evento. Caso 5 tiene cuarto evento documentado sin energía. |

</details>

<details>
<summary><strong>Energy_Fracture(:,8)</strong> — Energía tabulada asociada a fractura: evento 8</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Fracture(:,8) |
| Nombre científico comprensible | Energía tabulada asociada a fractura: evento 8 |
| Símbolo | E_fr,8 |
| Definición física | Columna 8 de Energy_Fracture; alias Python event_8. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, asignaciones de Energy_Fracture; S2 apéndice A |
| Nivel de observación | espécimen × evento |
| Fórmula de construcción | Energy_Fracture[Case_n,8] |
| Dominio y restricciones | finito o NaN; signo original |
| Interpretación experimental | Valor del evento ordinal 8. |
| Relaciones con otras variables | dependencia por Case_n; no nueva muestra |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Una posición ausente no equivale a inexistencia del evento. Caso 5 tiene cuarto evento documentado sin energía. |

</details>

<details>
<summary><strong>Energy_Fracture(:,9)</strong> — Energía tabulada asociada a fractura: evento 9</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Fracture(:,9) |
| Nombre científico comprensible | Energía tabulada asociada a fractura: evento 9 |
| Símbolo | E_fr,9 |
| Definición física | Columna 9 de Energy_Fracture; alias Python event_9. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, asignaciones de Energy_Fracture; S2 apéndice A |
| Nivel de observación | espécimen × evento |
| Fórmula de construcción | Energy_Fracture[Case_n,9] |
| Dominio y restricciones | finito o NaN; signo original |
| Interpretación experimental | Valor del evento ordinal 9. |
| Relaciones con otras variables | dependencia por Case_n; no nueva muestra |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Una posición ausente no equivale a inexistencia del evento. Caso 5 tiene cuarto evento documentado sin energía. |

</details>

<details>
<summary><strong>Energy_Fracture(:,10)</strong> — Energía tabulada asociada a fractura: evento 10</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Fracture(:,10) |
| Nombre científico comprensible | Energía tabulada asociada a fractura: evento 10 |
| Símbolo | E_fr,10 |
| Definición física | Columna 10 de Energy_Fracture; alias Python event_10. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, asignaciones de Energy_Fracture; S2 apéndice A |
| Nivel de observación | espécimen × evento |
| Fórmula de construcción | Energy_Fracture[Case_n,10] |
| Dominio y restricciones | finito o NaN; signo original |
| Interpretación experimental | Valor del evento ordinal 10. |
| Relaciones con otras variables | dependencia por Case_n; no nueva muestra |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Una posición ausente no equivale a inexistencia del evento. Caso 5 tiene cuarto evento documentado sin energía. |

</details>

<details>
<summary><strong>Energy_Fracture(:,11)</strong> — Energía tabulada asociada a fractura: evento 11</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Fracture(:,11) |
| Nombre científico comprensible | Energía tabulada asociada a fractura: evento 11 |
| Símbolo | E_fr,11 |
| Definición física | Columna 11 de Energy_Fracture; alias Python event_11. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, asignaciones de Energy_Fracture; S2 apéndice A |
| Nivel de observación | espécimen × evento |
| Fórmula de construcción | Energy_Fracture[Case_n,11] |
| Dominio y restricciones | finito o NaN; signo original |
| Interpretación experimental | Valor del evento ordinal 11. |
| Relaciones con otras variables | dependencia por Case_n; no nueva muestra |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Una posición ausente no equivale a inexistencia del evento. Caso 5 tiene cuarto evento documentado sin energía. |

</details>

<details>
<summary><strong>Energy_Fracture(:,12)</strong> — Energía tabulada asociada a fractura: evento 12</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Fracture(:,12) |
| Nombre científico comprensible | Energía tabulada asociada a fractura: evento 12 |
| Símbolo | E_fr,12 |
| Definición física | Columna 12 de Energy_Fracture; alias Python event_12. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, asignaciones de Energy_Fracture; S2 apéndice A |
| Nivel de observación | espécimen × evento |
| Fórmula de construcción | Energy_Fracture[Case_n,12] |
| Dominio y restricciones | finito o NaN; signo original |
| Interpretación experimental | Valor del evento ordinal 12. |
| Relaciones con otras variables | dependencia por Case_n; no nueva muestra |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Una posición ausente no equivale a inexistencia del evento. Caso 5 tiene cuarto evento documentado sin energía. |

</details>

<details>
<summary><strong>Energy_Fracture(:,13)</strong> — Energía tabulada asociada a fractura: evento 13</summary>

| Campo | Definición |
|---|---|
| Nombre original | Energy_Fracture(:,13) |
| Nombre científico comprensible | Energía tabulada asociada a fractura: evento 13 |
| Símbolo | E_fr,13 |
| Definición física | Columna 13 de Energy_Fracture; alias Python event_13. |
| Unidad | mJ |
| Tipo de variable | real continuo |
| Clasificación | OUTPUT |
| Fuente de procedencia | S1, asignaciones de Energy_Fracture; S2 apéndice A |
| Nivel de observación | espécimen × evento |
| Fórmula de construcción | Energy_Fracture[Case_n,13] |
| Dominio y restricciones | finito o NaN; signo original |
| Interpretación experimental | Valor del evento ordinal 13. |
| Relaciones con otras variables | dependencia por Case_n; no nueva muestra |
| Momento de disponibilidad | durante/después del ensayo |
| Utilidad como predictor | no como predictor previo a primera falla |
| Riesgo de fuga de información | alto: respuesta experimental posterior |
| Advertencias y limitaciones | Una posición ausente no equivale a inexistencia del evento. Caso 5 tiene cuarto evento documentado sin energía. |

</details>

## Variables derivadas

<details>
<summary><strong>area_fibra_mm2</strong> — Área ideal de sección de fibra</summary>

| Campo | Definición |
|---|---|
| Nombre original | area_fibra_mm2 |
| Nombre científico comprensible | Área ideal de sección de fibra |
| Símbolo | A_f |
| Definición física | Área de sección rectangular ideal. |
| Unidad | mm² |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | Fiber_base × Fiber_height |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Área de sección rectangular ideal. |
| Relaciones con otras variables | b_f,h_f |
| Momento de disponibilidad | antes del ensayo |
| Utilidad como predictor | candidata a validar posteriormente |
| Riesgo de fuga de información | bajo si usa solo geometría previa |
| Advertencias y limitaciones | No es área resistente efectiva global ni autoriza calcular esfuerzo global. |

</details>

<details>
<summary><strong>aspecto_fibra</strong> — Relación altura/base de fibra</summary>

| Campo | Definición |
|---|---|
| Nombre original | aspecto_fibra |
| Nombre científico comprensible | Relación altura/base de fibra |
| Símbolo | a_f |
| Definición física | Descriptor de forma de sección. |
| Unidad | adimensional |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | Fiber_height / Fiber_base |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Descriptor de forma de sección. |
| Relaciones con otras variables | recíproco de base_sobre_altura |
| Momento de disponibilidad | antes del ensayo |
| Utilidad como predictor | candidata a validar posteriormente |
| Riesgo de fuga de información | bajo si usa solo geometría previa |
| Advertencias y limitaciones | No es nueva medición; hereda discrepancias de dimensiones. |

</details>

<details>
<summary><strong>I_fibra_bh3_mm4</strong> — Segundo momento geométrico b h³/12</summary>

| Campo | Definición |
|---|---|
| Nombre original | I_fibra_bh3_mm4 |
| Nombre científico comprensible | Segundo momento geométrico b h³/12 |
| Símbolo | I_1 |
| Definición física | Segundo momento de área rectangular respecto a eje centroidal. |
| Unidad | mm⁴ |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | Fiber_base × Fiber_height³ / 12 |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Segundo momento de área rectangular respecto a eje centroidal. |
| Relaciones con otras variables | alias histórico I_bh3_mm4 |
| Momento de disponibilidad | antes del ensayo |
| Utilidad como predictor | candidata a validar posteriormente |
| Riesgo de fuga de información | bajo si usa solo geometría previa |
| Advertencias y limitaciones | No es rigidez; faltan módulo y orientación física verificada. |

</details>

<details>
<summary><strong>I_fibra_hb3_mm4</strong> — Segundo momento geométrico h b³/12</summary>

| Campo | Definición |
|---|---|
| Nombre original | I_fibra_hb3_mm4 |
| Nombre científico comprensible | Segundo momento geométrico h b³/12 |
| Símbolo | I_2 |
| Definición física | Segundo momento del rectángulo respecto al eje centroidal ortogonal. |
| Unidad | mm⁴ |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | Fiber_height × Fiber_base³ / 12 |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Segundo momento del rectángulo respecto al eje centroidal ortogonal. |
| Relaciones con otras variables | alias histórico I_hb3_mm4 |
| Momento de disponibilidad | antes del ensayo |
| Utilidad como predictor | candidata a validar posteriormente |
| Riesgo de fuga de información | bajo si usa solo geometría previa |
| Advertencias y limitaciones | No es rigidez ni tenacidad; no elegir eje dominante sin planos. |

</details>

<details>
<summary><strong>area_pivote_mm2</strong> — Área ideal del pivote</summary>

| Campo | Definición |
|---|---|
| Nombre original | area_pivote_mm2 |
| Nombre científico comprensible | Área ideal del pivote |
| Símbolo | A_p |
| Definición física | Área de sección circular ideal. |
| Unidad | mm² |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | π × Pivot_radius² |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Área de sección circular ideal. |
| Relaciones con otras variables | r_p |
| Momento de disponibilidad | antes del ensayo |
| Utilidad como predictor | candidata a validar posteriormente |
| Riesgo de fuga de información | bajo si usa solo geometría previa |
| Advertencias y limitaciones | Constante cuando radio es constante; no discrimina casos. |

</details>

<details>
<summary><strong>esbeltez_pivote</strong> — Relación altura/diámetro del pivote</summary>

| Campo | Definición |
|---|---|
| Nombre original | esbeltez_pivote |
| Nombre científico comprensible | Relación altura/diámetro del pivote |
| Símbolo | lambda_p |
| Definición física | Descriptor altura/diámetro de conexión. |
| Unidad | adimensional |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | Pivot_height / (2 × Pivot_radius) |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Descriptor altura/diámetro de conexión. |
| Relaciones con otras variables | h_p,r_p |
| Momento de disponibilidad | antes del ensayo |
| Utilidad como predictor | candidata a validar posteriormente |
| Riesgo de fuga de información | bajo si usa solo geometría previa |
| Advertencias y limitaciones | No es esbeltez de pandeo ni confirma un mecanismo de fallo. |

</details>

<details>
<summary><strong>fraccion_volumen_pivotes</strong> — Fracción de volumen declarado en pivotes</summary>

| Campo | Definición |
|---|---|
| Nombre original | fraccion_volumen_pivotes |
| Nombre científico comprensible | Fracción de volumen declarado en pivotes |
| Símbolo | phi_p |
| Definición física | Participación de pivotes en suma declarada de componentes. |
| Unidad | adimensional |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | Pivot_total_volume / (Fiber_total_volume + Pivot_total_volume) |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Participación de pivotes en suma declarada de componentes. |
| Relaciones con otras variables | V_p,V_f |
| Momento de disponibilidad | antes del ensayo |
| Utilidad como predictor | candidata a validar posteriormente |
| Riesgo de fuga de información | bajo si usa solo geometría previa |
| Advertencias y limitaciones | No es porosidad ni fracción de volumen CAD. |

</details>

<details>
<summary><strong>volumen_por_celda_mm3</strong> — Volumen declarado por celda en Y</summary>

| Campo | Definición |
|---|---|
| Nombre original | volumen_por_celda_mm3 |
| Nombre científico comprensible | Volumen declarado por celda en Y |
| Símbolo | V_s/n_y |
| Definición física | Normalización por conteo direccional disponible. |
| Unidad | mm³ por celda en Y |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | Sample_total_volume / n_cells_Y |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Normalización por conteo direccional disponible. |
| Relaciones con otras variables | V_s,n_y |
| Momento de disponibilidad | antes del ensayo |
| Utilidad como predictor | candidata a validar posteriormente |
| Riesgo de fuga de información | bajo si usa solo geometría previa |
| Advertencias y limitaciones | n_y no es conteo total de celdas: no representa volumen de una celda física. |

</details>

<details>
<summary><strong>razon_volumenes</strong> — Razón de volúmenes fibras/pivotes</summary>

| Campo | Definición |
|---|---|
| Nombre original | razon_volumenes |
| Nombre científico comprensible | Razón de volúmenes fibras/pivotes |
| Símbolo | R_V |
| Definición física | Balance geométrico entre componentes. |
| Unidad | adimensional |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | Fiber_total_volume / Pivot_total_volume |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Balance geométrico entre componentes. |
| Relaciones con otras variables | V_f,V_p |
| Momento de disponibilidad | antes del ensayo |
| Utilidad como predictor | candidata a validar posteriormente |
| Riesgo de fuga de información | bajo si usa solo geometría previa |
| Advertencias y limitaciones | No es balance de energía; bajo suma constante depende de una sola cantidad. |

</details>

<details>
<summary><strong>base_sobre_altura</strong> — Relación base/altura de fibra</summary>

| Campo | Definición |
|---|---|
| Nombre original | base_sobre_altura |
| Nombre científico comprensible | Relación base/altura de fibra |
| Símbolo | b_f/h_f |
| Definición física | Descriptor histórico de forma de sección. |
| Unidad | adimensional |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | Fiber_base / Fiber_height |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Descriptor histórico de forma de sección. |
| Relaciones con otras variables | recíproco de aspecto_fibra |
| Momento de disponibilidad | antes del ensayo |
| Utilidad como predictor | candidata a validar posteriormente |
| Riesgo de fuga de información | bajo si usa solo geometría previa |
| Advertencias y limitaciones | No incorporar simultáneamente representaciones redundantes sin justificación. |

</details>

<details>
<summary><strong>I_bh3_mm4</strong> — Segundo momento ideal, alias histórico</summary>

| Campo | Definición |
|---|---|
| Nombre original | I_bh3_mm4 |
| Nombre científico comprensible | Segundo momento ideal, alias histórico |
| Símbolo | I_1 |
| Definición física | Alias histórico de segundo momento b h³/12. |
| Unidad | mm⁴ |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | Fiber_base × Fiber_height³ / 12 |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Alias histórico de segundo momento b h³/12. |
| Relaciones con otras variables | idéntico a I_fibra_bh3_mm4 |
| Momento de disponibilidad | antes del ensayo |
| Utilidad como predictor | candidata a validar posteriormente |
| Riesgo de fuga de información | bajo si usa solo geometría previa |
| Advertencias y limitaciones | No constituye otra variable física ni rigidez. |

</details>

<details>
<summary><strong>I_hb3_mm4</strong> — Segundo momento ideal ortogonal, alias histórico</summary>

| Campo | Definición |
|---|---|
| Nombre original | I_hb3_mm4 |
| Nombre científico comprensible | Segundo momento ideal ortogonal, alias histórico |
| Símbolo | I_2 |
| Definición física | Alias histórico de segundo momento h b³/12. |
| Unidad | mm⁴ |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | Fiber_height × Fiber_base³ / 12 |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Alias histórico de segundo momento h b³/12. |
| Relaciones con otras variables | idéntico a I_fibra_hb3_mm4 |
| Momento de disponibilidad | antes del ensayo |
| Utilidad como predictor | candidata a validar posteriormente |
| Riesgo de fuga de información | bajo si usa solo geometría previa |
| Advertencias y limitaciones | No duplicar ambas denominaciones como predictores. |

</details>

<details>
<summary><strong>log_vol_pivote_sobre_fibra</strong> — Logaritmo de razón pivotes/fibras</summary>

| Campo | Definición |
|---|---|
| Nombre original | log_vol_pivote_sobre_fibra |
| Nombre científico comprensible | Logaritmo de razón pivotes/fibras |
| Símbolo | ln(V_p/V_f) |
| Definición física | Transformación de razón geométrica. |
| Unidad | adimensional |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | ln(Pivot_total_volume / Fiber_total_volume) |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Transformación de razón geométrica. |
| Relaciones con otras variables | opuesto del logaritmo de razon_volumenes |
| Momento de disponibilidad | antes del ensayo |
| Utilidad como predictor | candidata a validar posteriormente |
| Riesgo de fuga de información | bajo si usa solo geometría previa |
| Advertencias y limitaciones | Transformación monótona algebraicamente dependiente. |

</details>

<details>
<summary><strong>carga_por_volumen_N_mm3</strong> — Carga de primera falla por volumen</summary>

| Campo | Definición |
|---|---|
| Nombre original | carga_por_volumen_N_mm3 |
| Nombre científico comprensible | Carga de primera falla por volumen |
| Símbolo | F_FF/V_s |
| Definición física | Normalización descriptiva de fuerza por volumen. |
| Unidad | N/mm³ |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | First_Failure_Load_N / Sample_total_volume |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Normalización descriptiva de fuerza por volumen. |
| Relaciones con otras variables | F_FF,V_s |
| Momento de disponibilidad | después del ensayo |
| Utilidad como predictor | no como entrada previa |
| Riesgo de fuga de información | alto: incluye respuesta experimental |
| Advertencias y limitaciones | No es esfuerzo (N/mm²), rigidez ni resistencia específica por masa. |

</details>

<details>
<summary><strong>razon_carga_terminal_primera</strong> — Razón carga terminal/primera falla</summary>

| Campo | Definición |
|---|---|
| Nombre original | razon_carga_terminal_primera |
| Nombre científico comprensible | Razón carga terminal/primera falla |
| Símbolo | F_term/F_FF |
| Definición física | Comparación de dos momentos del ensayo. |
| Unidad | adimensional |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | Ultimate_Load / First_Failure_Load_N |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Comparación de dos momentos del ensayo. |
| Relaciones con otras variables | F_term,F_FF |
| Momento de disponibilidad | después del ensayo |
| Utilidad como predictor | no como entrada previa |
| Riesgo de fuga de información | alto: incluye respuesta experimental |
| Advertencias y limitaciones | No es razón máxima/inicial ni índice universal de resiliencia. |

</details>

<details>
<summary><strong>fraccion_residual_ciclo_1</strong> — Fracción residual nominal, ciclo 1</summary>

| Campo | Definición |
|---|---|
| Nombre original | fraccion_residual_ciclo_1 |
| Nombre científico comprensible | Fracción residual nominal, ciclo 1 |
| Símbolo | r_1 |
| Definición física | Residual normalizado por amplitud máxima nominal. |
| Unidad | adimensional |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | residual_Displacement_1cycle_10mm / 10 |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Residual normalizado por amplitud máxima nominal. |
| Relaciones con otras variables | complemento de recuperación nominal |
| Momento de disponibilidad | después del ensayo |
| Utilidad como predictor | no como entrada previa |
| Riesgo de fuga de información | alto: incluye respuesta experimental |
| Advertencias y limitaciones | No es deformación unitaria ni ley de fatiga. |

</details>

<details>
<summary><strong>recuperacion_nominal_ciclo_1</strong> — Fracción de recuperación nominal, ciclo 1</summary>

| Campo | Definición |
|---|---|
| Nombre original | recuperacion_nominal_ciclo_1 |
| Nombre científico comprensible | Fracción de recuperación nominal, ciclo 1 |
| Símbolo | R_1 |
| Definición física | Complemento descriptivo de fracción residual. |
| Unidad | adimensional |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | 1 − residual_Displacement_1cycle_10mm / 10 |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Complemento descriptivo de fracción residual. |
| Relaciones con otras variables | misma información que fracción residual |
| Momento de disponibilidad | después del ensayo |
| Utilidad como predictor | no como entrada previa |
| Riesgo de fuga de información | alto: incluye respuesta experimental |
| Advertencias y limitaciones | No mide recuperación material intrínseca; referencia nominal del ciclo. |

</details>

<details>
<summary><strong>fraccion_residual_ciclo_2</strong> — Fracción residual nominal, ciclo 2</summary>

| Campo | Definición |
|---|---|
| Nombre original | fraccion_residual_ciclo_2 |
| Nombre científico comprensible | Fracción residual nominal, ciclo 2 |
| Símbolo | r_2 |
| Definición física | Residual normalizado por amplitud máxima nominal. |
| Unidad | adimensional |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | residual_Displacement_2cycle_20mm / 20 |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Residual normalizado por amplitud máxima nominal. |
| Relaciones con otras variables | complemento de recuperación nominal |
| Momento de disponibilidad | después del ensayo |
| Utilidad como predictor | no como entrada previa |
| Riesgo de fuga de información | alto: incluye respuesta experimental |
| Advertencias y limitaciones | No es deformación unitaria ni ley de fatiga. |

</details>

<details>
<summary><strong>recuperacion_nominal_ciclo_2</strong> — Fracción de recuperación nominal, ciclo 2</summary>

| Campo | Definición |
|---|---|
| Nombre original | recuperacion_nominal_ciclo_2 |
| Nombre científico comprensible | Fracción de recuperación nominal, ciclo 2 |
| Símbolo | R_2 |
| Definición física | Complemento descriptivo de fracción residual. |
| Unidad | adimensional |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | 1 − residual_Displacement_2cycle_20mm / 20 |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Complemento descriptivo de fracción residual. |
| Relaciones con otras variables | misma información que fracción residual |
| Momento de disponibilidad | después del ensayo |
| Utilidad como predictor | no como entrada previa |
| Riesgo de fuga de información | alto: incluye respuesta experimental |
| Advertencias y limitaciones | No mide recuperación material intrínseca; referencia nominal del ciclo. |

</details>

<details>
<summary><strong>fraccion_residual_ciclo_3</strong> — Fracción residual nominal, ciclo 3</summary>

| Campo | Definición |
|---|---|
| Nombre original | fraccion_residual_ciclo_3 |
| Nombre científico comprensible | Fracción residual nominal, ciclo 3 |
| Símbolo | r_3 |
| Definición física | Residual normalizado por amplitud máxima nominal. |
| Unidad | adimensional |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | residual_Displacement_3cycle_30mm / 30 |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Residual normalizado por amplitud máxima nominal. |
| Relaciones con otras variables | complemento de recuperación nominal |
| Momento de disponibilidad | después del ensayo |
| Utilidad como predictor | no como entrada previa |
| Riesgo de fuga de información | alto: incluye respuesta experimental |
| Advertencias y limitaciones | No es deformación unitaria ni ley de fatiga. |

</details>

<details>
<summary><strong>recuperacion_nominal_ciclo_3</strong> — Fracción de recuperación nominal, ciclo 3</summary>

| Campo | Definición |
|---|---|
| Nombre original | recuperacion_nominal_ciclo_3 |
| Nombre científico comprensible | Fracción de recuperación nominal, ciclo 3 |
| Símbolo | R_3 |
| Definición física | Complemento descriptivo de fracción residual. |
| Unidad | adimensional |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | 1 − residual_Displacement_3cycle_30mm / 30 |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Complemento descriptivo de fracción residual. |
| Relaciones con otras variables | misma información que fracción residual |
| Momento de disponibilidad | después del ensayo |
| Utilidad como predictor | no como entrada previa |
| Riesgo de fuga de información | alto: incluye respuesta experimental |
| Advertencias y limitaciones | No mide recuperación material intrínseca; referencia nominal del ciclo. |

</details>

<details>
<summary><strong>fraccion_residual_ciclo_4</strong> — Fracción residual nominal, ciclo 4</summary>

| Campo | Definición |
|---|---|
| Nombre original | fraccion_residual_ciclo_4 |
| Nombre científico comprensible | Fracción residual nominal, ciclo 4 |
| Símbolo | r_4 |
| Definición física | Residual normalizado por amplitud máxima nominal. |
| Unidad | adimensional |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | residual_Displacement_4cycle_40mm / 40 |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Residual normalizado por amplitud máxima nominal. |
| Relaciones con otras variables | complemento de recuperación nominal |
| Momento de disponibilidad | después del ensayo |
| Utilidad como predictor | no como entrada previa |
| Riesgo de fuga de información | alto: incluye respuesta experimental |
| Advertencias y limitaciones | No es deformación unitaria ni ley de fatiga. |

</details>

<details>
<summary><strong>recuperacion_nominal_ciclo_4</strong> — Fracción de recuperación nominal, ciclo 4</summary>

| Campo | Definición |
|---|---|
| Nombre original | recuperacion_nominal_ciclo_4 |
| Nombre científico comprensible | Fracción de recuperación nominal, ciclo 4 |
| Símbolo | R_4 |
| Definición física | Complemento descriptivo de fracción residual. |
| Unidad | adimensional |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | 1 − residual_Displacement_4cycle_40mm / 40 |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Complemento descriptivo de fracción residual. |
| Relaciones con otras variables | misma información que fracción residual |
| Momento de disponibilidad | después del ensayo |
| Utilidad como predictor | no como entrada previa |
| Riesgo de fuga de información | alto: incluye respuesta experimental |
| Advertencias y limitaciones | No mide recuperación material intrínseca; referencia nominal del ciclo. |

</details>

<details>
<summary><strong>fraccion_residual_ciclo_5</strong> — Fracción residual nominal, ciclo 5</summary>

| Campo | Definición |
|---|---|
| Nombre original | fraccion_residual_ciclo_5 |
| Nombre científico comprensible | Fracción residual nominal, ciclo 5 |
| Símbolo | r_5 |
| Definición física | Residual normalizado por amplitud máxima nominal. |
| Unidad | adimensional |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | residual_Displacement_5cycle_50mm / 50 |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Residual normalizado por amplitud máxima nominal. |
| Relaciones con otras variables | complemento de recuperación nominal |
| Momento de disponibilidad | después del ensayo |
| Utilidad como predictor | no como entrada previa |
| Riesgo de fuga de información | alto: incluye respuesta experimental |
| Advertencias y limitaciones | No es deformación unitaria ni ley de fatiga. |

</details>

<details>
<summary><strong>recuperacion_nominal_ciclo_5</strong> — Fracción de recuperación nominal, ciclo 5</summary>

| Campo | Definición |
|---|---|
| Nombre original | recuperacion_nominal_ciclo_5 |
| Nombre científico comprensible | Fracción de recuperación nominal, ciclo 5 |
| Símbolo | R_5 |
| Definición física | Complemento descriptivo de fracción residual. |
| Unidad | adimensional |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | 1 − residual_Displacement_5cycle_50mm / 50 |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Complemento descriptivo de fracción residual. |
| Relaciones con otras variables | misma información que fracción residual |
| Momento de disponibilidad | después del ensayo |
| Utilidad como predictor | no como entrada previa |
| Riesgo de fuga de información | alto: incluye respuesta experimental |
| Advertencias y limitaciones | No mide recuperación material intrínseca; referencia nominal del ciclo. |

</details>

<details>
<summary><strong>fraccion_residual_ciclo_6</strong> — Fracción residual nominal, ciclo 6</summary>

| Campo | Definición |
|---|---|
| Nombre original | fraccion_residual_ciclo_6 |
| Nombre científico comprensible | Fracción residual nominal, ciclo 6 |
| Símbolo | r_6 |
| Definición física | Residual normalizado por amplitud máxima nominal. |
| Unidad | adimensional |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | residual_Displacement_6cycle_60mm / 60 |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Residual normalizado por amplitud máxima nominal. |
| Relaciones con otras variables | complemento de recuperación nominal |
| Momento de disponibilidad | después del ensayo |
| Utilidad como predictor | no como entrada previa |
| Riesgo de fuga de información | alto: incluye respuesta experimental |
| Advertencias y limitaciones | No es deformación unitaria ni ley de fatiga. |

</details>

<details>
<summary><strong>recuperacion_nominal_ciclo_6</strong> — Fracción de recuperación nominal, ciclo 6</summary>

| Campo | Definición |
|---|---|
| Nombre original | recuperacion_nominal_ciclo_6 |
| Nombre científico comprensible | Fracción de recuperación nominal, ciclo 6 |
| Símbolo | R_6 |
| Definición física | Complemento descriptivo de fracción residual. |
| Unidad | adimensional |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | 1 − residual_Displacement_6cycle_60mm / 60 |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Complemento descriptivo de fracción residual. |
| Relaciones con otras variables | misma información que fracción residual |
| Momento de disponibilidad | después del ensayo |
| Utilidad como predictor | no como entrada previa |
| Riesgo de fuga de información | alto: incluye respuesta experimental |
| Advertencias y limitaciones | No mide recuperación material intrínseca; referencia nominal del ciclo. |

</details>

<details>
<summary><strong>fraccion_energia_primer_evento</strong> — Fracción energética del primer evento observado</summary>

| Campo | Definición |
|---|---|
| Nombre original | fraccion_energia_primer_evento |
| Nombre científico comprensible | Fracción energética del primer evento observado |
| Símbolo | E_fr,1/ΣE_fr,j |
| Definición física | Contribución inicial a suma energética disponible. |
| Unidad | adimensional |
| Tipo de variable | real continuo |
| Clasificación | derivada |
| Fuente de procedencia | cálculo del proyecto desde S1; interpretación contrastada con S2–S4 |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | Energy_Fracture(:,1) / suma de energías finitas por caso |
| Dominio y restricciones | entradas finitas; denominadores positivos |
| Interpretación experimental | Contribución inicial a suma energética disponible. |
| Relaciones con otras variables | primer evento y posteriores |
| Momento de disponibilidad | después del ensayo |
| Utilidad como predictor | no como entrada previa |
| Riesgo de fuga de información | alto: incluye respuesta experimental |
| Advertencias y limitaciones | Suma disponible no demuestra energía total del ensayo ni todos sus eventos. |

</details>

<details>
<summary><strong>eventos_energia_observados</strong> — Eventos con energía disponible</summary>

| Campo | Definición |
|---|---|
| Nombre original | eventos_energia_observados |
| Nombre científico comprensible | Eventos con energía disponible |
| Símbolo | n_E |
| Definición física | Conteo de energías no ausentes por caso. |
| Unidad | conteo |
| Tipo de variable | entero discreto |
| Clasificación | derivada |
| Fuente de procedencia | S1; cálculo del proyecto |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | count_finite(Energy_Fracture[i,:]) |
| Dominio y restricciones | entero no negativo |
| Interpretación experimental | Mide disponibilidad de registros energéticos. |
| Relaciones con otras variables | depende de presencia de datos |
| Momento de disponibilidad | después del ensayo |
| Utilidad como predictor | no |
| Riesgo de fuga de información | alto si se usa para predecir primera falla |
| Advertencias y limitaciones | No es número verdadero de fracturas: caso 5 documenta cuarto evento sin energía. |

</details>

## Identificadores y metadatos

<details>
<summary><strong>Case_n</strong> — Identificador del caso experimental</summary>

| Campo | Definición |
|---|---|
| Nombre original | Case_n |
| Nombre científico comprensible | Identificador del caso experimental |
| Símbolo | i |
| Definición física | Clave asignada al espécimen/ensayo en la campaña. |
| Unidad | sin unidad |
| Tipo de variable | entero nominal |
| Clasificación | metadato |
| Fuente de procedencia | S1, línea 6; S2 PDF 113 y fichas |
| Nivel de observación | espécimen / caso |
| Fórmula de construcción | enumeración de la fuente |
| Dominio y restricciones | entero positivo; único en tabla principal |
| Interpretación experimental | Vincula geometría, respuestas, ciclos y eventos. |
| Relaciones con otras variables | clave de unión de tablas |
| Momento de disponibilidad | registro de campaña |
| Utilidad como predictor | no; no describe la mecánica |
| Riesgo de fuga de información | asociación espuria con orden de la campaña |
| Advertencias y limitaciones | No identifica lotes, réplicas ni fecha; no inferir independencia por ser único. |

</details>


## Alias y campos del módulo de relaciones

Las tablas de relaciones utilizan los nombres siguientes. El CSV incluye una entrada completa de diecisiete campos para cada alias o derivada; no son nuevas mediciones independientes. Cuando cambia el sentido de una razón, prevalece la fórmula indicada.

| Nombre en tablas | Referencia conceptual | Fórmula exacta |
|---|---|---|
| Dissipated_cycle_1 | Energy_Dissipated(:,1) | Energy_Dissipated leído sin ejecución de MATLAB |
| Dissipated_cycle_2 | Energy_Dissipated(:,2) | Energy_Dissipated leído sin ejecución de MATLAB |
| Dissipated_cycle_3 | Energy_Dissipated(:,3) | Energy_Dissipated leído sin ejecución de MATLAB |
| Dissipated_cycle_4 | Energy_Dissipated(:,4) | Energy_Dissipated leído sin ejecución de MATLAB |
| Dissipated_cycle_5 | Energy_Dissipated(:,5) | Energy_Dissipated leído sin ejecución de MATLAB |
| Dissipated_cycle_6 | Energy_Dissipated(:,6) | Energy_Dissipated leído sin ejecución de MATLAB |
| Dissipated_cycle_7 | Energy_Dissipated(:,7) | Energy_Dissipated leído sin ejecución de MATLAB |
| Fracture_event_1 | Energy_Fracture(:,1) | Energy_Fracture leído sin ejecución de MATLAB |
| Fracture_event_2 | Energy_Fracture(:,2) | Energy_Fracture leído sin ejecución de MATLAB |
| Fracture_event_3 | Energy_Fracture(:,3) | Energy_Fracture leído sin ejecución de MATLAB |
| Fracture_event_4 | Energy_Fracture(:,4) | Energy_Fracture leído sin ejecución de MATLAB |
| Fracture_event_5 | Energy_Fracture(:,5) | Energy_Fracture leído sin ejecución de MATLAB |
| Fracture_event_6 | Energy_Fracture(:,6) | Energy_Fracture leído sin ejecución de MATLAB |
| Fracture_event_7 | Energy_Fracture(:,7) | Energy_Fracture leído sin ejecución de MATLAB |
| Fracture_event_8 | Energy_Fracture(:,8) | Energy_Fracture leído sin ejecución de MATLAB |
| Fracture_event_9 | Energy_Fracture(:,9) | Energy_Fracture leído sin ejecución de MATLAB |
| Fracture_event_10 | Energy_Fracture(:,10) | Energy_Fracture leído sin ejecución de MATLAB |
| Fracture_event_11 | Energy_Fracture(:,11) | Energy_Fracture leído sin ejecución de MATLAB |
| Fracture_event_12 | Energy_Fracture(:,12) | Energy_Fracture leído sin ejecución de MATLAB |
| Fracture_event_13 | Energy_Fracture(:,13) | Energy_Fracture leído sin ejecución de MATLAB |
| Fiber_area_proxy | area_fibra_mm2 | Fiber_height * Fiber_base |
| Fiber_aspect_ratio | aspecto_fibra | Fiber_height / Fiber_base |
| Pivot_aspect_ratio | esbeltez_pivote | Pivot_height / (2 * Pivot_radius) |
| Fiber_I_proxy | I_fibra_bh3_mm4 | Fiber_base * Fiber_height^3 / 12 |
| Load_ratio_first_over_ultimate | carga_por_volumen_N_mm3 | First_Failure_Load_N / Ultimate_Load |
| Load_per_specimen_volume | carga_por_volumen_N_mm3 | First_Failure_Load_N / Sample_total_volume |
| Load_per_pivot | carga_por_volumen_N_mm3 | First_Failure_Load_N / Pivot_total_number |
| Load_per_cell_Y | carga_por_volumen_N_mm3 | First_Failure_Load_N / n_cells_Y |
| Fracture_observed_sum | carga_por_volumen_N_mm3 | sum(Energy_Fracture disponible), min_count=1 |
| Fracture_registered_events | eventos_energia_observados | número de valores Energy_Fracture presentes |
| Fracture_initial_share | fraccion_energia_primer_evento | Energy_Fracture(evento1) / suma registrada |
| Residual_fraction_cycle1 | fraccion_residual_ciclo_1 | residual_Displacement_1cycle_10mm / 10 mm |
| Residual_fraction_cycle2 | fraccion_residual_ciclo_2 | residual_Displacement_2cycle_20mm / 20 mm |
| Residual_fraction_cycle3 | fraccion_residual_ciclo_3 | residual_Displacement_3cycle_30mm / 30 mm |
| Residual_fraction_cycle4 | fraccion_residual_ciclo_4 | residual_Displacement_4cycle_40mm / 40 mm |
| Residual_fraction_cycle5 | fraccion_residual_ciclo_5 | residual_Displacement_5cycle_50mm / 50 mm |
| Residual_fraction_cycle6 | fraccion_residual_ciclo_6 | residual_Displacement_6cycle_60mm / 60 mm |
| Recovery_complement_cycle6 | recuperacion_nominal_ciclo_6 | 1 - residual_Displacement_6cycle_60mm / 60 mm |
