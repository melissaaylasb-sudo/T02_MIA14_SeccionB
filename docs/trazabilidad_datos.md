# Trazabilidad documental y de los datos

**Criterio:** el archivo digital es la fuente operativa de los cálculos; la tesis y los artículos permiten verificar significado, unidades y protocolo. La concordancia informática con el archivo no garantiza concordancia entre documentos. No se cambió ningún valor original a partir de una discrepancia.

## Fuentes examinadas

| Código | Fuente realmente disponible | Uso y alcance |
|---|---|---|
| S1 | data/raw/dati_campagna_venditti.m | Vectores de geometría y respuestas; matrices Energy_Dissipated y Energy_Fracture. Lectura literal, sin ejecutar MATLAB. |
| S2 | 22639_ENRICO_VENDITTI_Analisi_strutturale_per_lo_studio_del_danneggiamento_in_materiali_e__s_1609769822.pdf | Tesis doctoral de Enrico Venditti, Universidad de Sassari, año académico 2024/2025. Capítulo 9 y apéndice A. |
| S3 | 2017_turcoetal_perpignan.pdf | Contexto del modelo discreto Hencky y de protocolos diferentes. No aporta filas a la tabla analítica. |
| S4 | s00161-018-0678-y.pdf | Contexto de mecanismos y modelo Piola–Hencky; publicación en línea 2018, volumen de revista 2019. No se importan sus condiciones experimentales a S1. |
| S5 | Proyecto_de_Investigación_Cientifico_Melissa_Parcial.pdf | Plan de tesis de Melissa Aylas Barranca, UNI, 2026. Objetivo predictivo y procedencia académica declarada. |

Los PDF se consultaron en las rutas privadas suministradas; no se copian al repositorio público. Las extracciones intermedias de texto e imágenes permanecen fuera del repositorio. El archivo MATLAB real no incluye el sufijo “(2)” sugerido en el encargo.

**Convención de páginas:** en S2 las referencias “PDF n” cuentan desde la portada; no hay rótulos de página PDF ni numeración impresa visible. Su índice ubica el capítulo 9 en PDF 91 y el apéndice A en PDF 113. En S3, página impresa = página PDF + 284; en S4, página impresa = página PDF + 208. En S5 se indican ambas cuando hay foliación impresa.

## Matriz de trazabilidad

| Fuente | Sección / página | Variable o concepto | Definición verificada | Aplicación en el EDA | Limitación |
|---|---|---|---|---|---|
| S2 | Portada, PDF 1 | Autor e institución | Enrico Venditti, doctorando; DADU, Università degli Studi di Sassari; doctorado Arquitectura y Ambiente; supervisor Emilio Barchiesi; 2024/2025 | Identificación bibliográfica | No establece afiliación actual ni fecha de defensa |
| S5 | PDF 1; PDF 24 / p. 18; PDF 26 / p. 20 | Procedencia académica | Plan fechado 2026; campaña de Venditti proporcionada por Emilio Turco | Distinguir procedencia declarada de elaboración propia | No equivale a licencia pública de redistribución |
| S2 | Cap. 9, PDF 91 | Pantógrafo, fibras y pivotes | Dos familias ortogonales de fibras en planos paralelos conectadas por pivotes cilíndricos elásticos | Introducción y lenguaje mecánico | No identifica rotura de un elemento concreto para cada Case_n |
| S2 | Cap. 9, PDF 94 | Material y fabricación | Poliamida/nylon, sinterización láser selectiva, CAD SolidWorks y exportación STL | Delimitar población material y diseño | No especifica aquí grado comercial, lote o tolerancias medidas |
| S2 | Cap. 9, PDF 94 | Diseño experimental | Dimensiones externas nominales 210 × 70 mm; criterio de masa idéntica/similar; variación de alturas y celdas Y | Separar arquitectura interna y escala externa | No hay columna de masa medida ni longitud útil individual |
| S2 | Tabla general, PDF 95 | Entradas geométricas | Nombres de dimensiones y unidades mm/mm³; conteos de celdas y pivotes | Diccionario y contraste documental | Existen discordancias con S1 y fichas, detalladas abajo |
| S2 | Cap. 9, PDF 96 | Ensayo | Tracción con carga–descarga, velocidad 15 mm/min y niveles cada 10 mm | Interpretación de ciclos y fuerzas | No atribuir sensores o incertidumbres de otros artículos |
| S2 | Cap. 9, PDF 96–97 | Equipo y registro de roturas | Probetas sujetas al equipo, registro fuerza–desplazamiento y grabación del sonido de las roturas | Explicación del procedimiento e identificación de eventos | La descripción consultada no identifica marca, modelo ni sensores concretos; no acredita instrumentación de emisión acústica |
| S2 | Cap. 9, PDF 96; fichas PDF 115, 119, etc. | First_Failure_Load_N | Fuerza del primer evento identificado de rotura de fibra o pivote; N explícitos en tabla | Target: carga de primera falla | Resumen y primer Upper Force difieren en algunas fichas |
| S2 | Cap. 9, PDF 96; PDF 115–116 | Ultimate_Load | Fuerza asociada a rotura total/final en protocolo; columna de resumen Ultimate Load | Carga terminal reportada | No es máximo global; diferencias locales con último pico |
| S2 | PDF 115–116 y fichas equivalentes | Maximum_Displacement | Resumen de desplazamiento máximo; eje de curva en mm | Respuesta secundaria | No es deformación unitaria; no se recalcula sin serie |
| S2 | Cap. 9, PDF 96; PDF 115 | Residuales | Desplazamiento al final de descarga de cada ciclo; amplitudes nominales en mm | Trayectorias y residual/amplitud nominal | No dividir todos los ciclos por un intervalo constante de 10 mm |
| S2 | PDF 117, 121, 131 y fichas energéticas | Energy_Dissipated | Energía disipada por ciclo, rotulada en millijoule | Ejes en mJ y análisis repetido por ciclo | Se conserva signo; algoritmo de integración no disponible |
| S2 | Cap. 9, PDF 96–98; PDF 117, 121 | Energy_Fracture | Energía tabulada en curvas de fractura, rotulada en millijoule | Relaciones por evento y con primera falla | No se conocen límites de integración; no llamarla tenacidad o tasa de liberación |
| S2 | Apéndice A, PDF 113 | Ausencia de respuestas de caso 2 | Problema de datos de salida, con intento de recuperación indicado en la tesis | Marcar ausencia documental conocida | No imputar outputs ni tratar el caso como ensayo sin fallo |
| S1 y S2 | S1 asignaciones; S2 PDF 131 | NaN energético caso 5 / evento 4 | Cuarto evento existe en tabla de picos PDF 129, energía ausente en PDF 131 | No equiparar conteo de energías al conteo total de fracturas | No atribuir causa específica a la energía faltante |
| S2 | PDF 204, 209, 219, 224, 229, 234 y 239 | Residual de séptimo ciclo | Hay valores a 70 mm en esas fichas | Necesidad de futura extracción validada | La columna no está incorporada en S1 ni se añade silenciosamente |
| S2 | Fichas de picos y figuras fuerza–desplazamiento | Upper/Lower Force y Displacement por evento | Resúmenes adicionales presentes en PDF | Aclaran semántica de primera falla y terminal | No están en S1; figuras no son series instrumentales completas |
| S3 | §2, PDF 3–4 / pp. 287–288 | Modelo Hencky | Representa extensión, flexión y conexiones mediante contribuciones discretas | Hipótesis geométrico-mecánicas | No prueba efectos causales de S1 |
| S3 | §3.1, PDF 8–10 / pp. 292–294 | Ensayo push-out | Condiciones particulares de carga en fibras; otra velocidad y otro diseño | Advertencia contra mezclar protocolos | No trasladar PA2200, módulo o 5 mm/min a campaña Venditti |
| S4 | §1–2.1, PDF 2–4 / pp. 210–212 | Piola–Hencky y mecanismos | Extensión/flexión de fibras y microtorsión de pivotes en subestructura específica | Interpretar posibles descriptores geométricos | No trasladar dimensiones, sensores ni parámetros de esa subestructura |

## Concordancia y discrepancias

Se extrajeron las tablas del PDF con PyMuPDF y se contrastaron numéricamente con los vectores MATLAB. La tabla general PDF 95 y páginas representativas se inspeccionaron visualmente para verificar posición de columnas. El contraste distingue diferencias de datos y tablas con encabezados incongruentes; no se toma una extracción automática como una corrección experimental.

### Tabla general geométrica frente al archivo digital

Cada fila enumera discrepancias del caso indicado: **valor del archivo → valor de la tabla PDF 95**. Las unidades son las del diccionario. Se presentan todas las diferencias detectadas, incluidas diferencias de redondeo.

| Caso | Campo y discrepancia con tabla general PDF 95 |
|---|---|
| 2 | Sample_total_volume: 6148.59 → 6184.58 |
| 9 | Sample_total_volume: 6124.9 → 6120.81 |
| 14 | Fiber_base: 1.56 → 1.66; Sample_total_volume: 6371.6 → 6425.3 |
| 17 | Sample_total_volume: 6452.1 → 6452.16 |
| 19 | Fiber_base: 2 → 1.7; Fiber_total_length: 3725.65 → 3725.66; Sample_total_volume: 6575.88 → 6609.36 |
| 20 | Fiber_total_length: 3725.65 → 3725.66; Sample_total_volume: 6580.83 → 6575.88 |
| 21 | Fiber_base: 1.69 → 1.21; Fiber_total_length: 3725.65 → 3725.66; Fiber_total_volume: 6296.99 → 6334.82; Pivot_height: 1.2 → 1; Pivot_total_volume: 227.02 → 189.19; Sample_total_volume: 6607.73 → 6502.18 |
| 22 | Fiber_base: 1.41 → 1.69; Fiber_total_length: 3725.65 → 3725.66; Sample_total_volume: 6486.15 → 6607.73 |
| 23 | Fiber_base: 1.69 → 1.41; Fiber_total_length: 3725.65 → 3725.66; Pivot_height: 1.4 → 1.2; Sample_total_volume: 6486.15 → 6567.04 |
| 24 | Fiber_base: 1.68 → 1.2; Fiber_total_length: 3725.65 → 3725.66; Fiber_total_volume: 6259.11 → 6296.99; Pivot_height: 1.4 → 1.2; Pivot_total_volume: 264.9 → 227.02; Sample_total_volume: 6558.92 → 6486.15 |
| 25 | Fiber_total_length: 3725.65 → 3725.66; Pivot_height: 1 → 1.4; Sample_total_volume: 6558.92 → 6606.12 |
| 26 | Fiber_base: 1.42 → 1.4; Fiber_total_length: 3725.65 → 3725.66; Pivot_height: 1.2 → 1.4; Sample_total_volume: 6524.01 → 6558.22 |
| 27 | Fiber_base: 1.42 → 1.2; Fiber_total_length: 3725.65 → 3725.66 |

La desviación de 0.01 mm en longitud total de la familia con seis celdas es pequeña; cambios de altura del pivote, base o asignación de volumen tienen otra relevancia. No deben agruparse todos como errores de redondeo. Tampoco puede concluirse que el PDF sea la versión corregida: el propio apéndice contiene discrepancias internas.

### Fichas del apéndice frente al archivo digital

En las filas siguientes se compara la ficha general individual; para Sample_total_volume se utiliza la columna identificada como **SolidWorks Sample total volume**, que coincide con el archivo en las fichas concordantes. Las fichas con un desplazamiento aparente de valores bajo encabezados se tratan aparte y no se interpretan como mediciones físicas.

| Caso | Página PDF | Campo y discrepancia: archivo → ficha |
|---|---|---|
| 1 | 115 | Fiber_base: 2.4 → 4.15; Fiber_total_length: 2685.82 → 1485.8; Fiber_total_volume: 6435.3 → 6163.67; Sample_total_volume: 6249.74 → 11235.1 |
| 7 | 139 | Fiber_base: 2.38 → 4.12; Fiber_total_length: 2685.82 → 1485.8; Fiber_total_volume: 6399.82 → 6128.19; Sample_total_volume: 6230.67 → 11181.7 |
| 9 | 149 | Sample_total_volume: 6124.9 → 6120.81 |
| 14 | 174 | Fiber_base: 1.56 → 1.71; Fiber_total_length: 3178.56 → 2970; Fiber_total_volume: 6362.93 → 6091.3; Sample_total_volume: 6371.6 → 6622.68 |
| 15 | 179 | Fiber_base: 1.43 → 1.46; Fiber_total_length: 3178.56 → 2970; Fiber_total_volume: 6362.93 → 6091.3; Sample_total_volume: 6417.67 → 6554.13 |
| 16 | 184 | Fiber_base: 1.99 → 2.04; Fiber_total_length: 3178.56 → 2970; Fiber_total_volume: 6336.08 → 6064.45; Sample_total_volume: 6501.59 → 6668.88 |
| 17 | 189 | Sample_total_volume: 6452.1 → 6452.16 |
| 19 | 199 | Fiber_base: 2 → 1.7; Fiber_total_length: 3725.65 → 3725.66; Sample_total_volume: 6575.88 → 6609.36 |
| 20 | 204 | Fiber_total_length: 3725.65 → 3564; Fiber_total_volume: 6334.82 → 6063.19; Sample_total_volume: 6580.83 → 6575.88 |
| 21 | 209 | Fiber_base: 1.69 → 1.21; Fiber_total_length: 3725.65 → 3725.66; Fiber_total_volume: 6296.99 → 6334.82; Pivot_height: 1.2 → 1; Pivot_total_volume: 227.02 → 189.19; Sample_total_volume: 6607.73 → 6502.18 |
| 22 | 214 | Fiber_base: 1.41 → 1.69; Fiber_total_length: 3725.65 → 3725.66; Sample_total_volume: 6486.15 → 6607.73 |
| 23 | 219 | Fiber_base: 1.69 → 1.41; Fiber_total_length: 3725.65 → 3725.66; Pivot_height: 1.4 → 1.2; Sample_total_volume: 6486.15 → 6567.04 |
| 24 | 224 | Fiber_base: 1.68 → 1.2; Fiber_total_length: 3725.65 → 3725.66; Fiber_total_volume: 6259.11 → 6296.99; Pivot_height: 1.4 → 1.2; Pivot_total_volume: 264.9 → 227.02; Sample_total_volume: 6558.92 → 6486.15 |
| 25 | 229 | Fiber_total_length: 3725.65 → 3725.66; Pivot_height: 1 → 1.4; Sample_total_volume: 6558.92 → 6606.12 |
| 26 | 234 | Fiber_base: 1.42 → 1.4; Fiber_total_length: 3725.65 → 3725.66; Pivot_height: 1.2 → 1.4; Sample_total_volume: 6524.01 → 6558.22 |
| 27 | 239 | Fiber_base: 1.42 → 1.2; Fiber_total_length: 3725.65 → 3725.66 |

| Incidencia adicional | Evidencia documental | Decisión |
|---|---|---|
| Encabezados geométricos incongruentes | PDF 129, 164 y 169: valores aparecen desplazados respecto de dimensiones/conteos; PDF 129 inspeccionado visualmente muestra, por ejemplo, altura de pivote 6417.56 bajo ese encabezado | No corregir el archivo a partir de esas celdas; pedir conciliación con CAD/ficha original |
| Dos nociones de volumen | PDF 119: Sample total volume = suma declarada de componentes, mientras SolidWorks Sample total volume corresponde al valor global de S1 | Mantener nombre original; aclarar procedencia CAD en fichas concordantes |
| Geometría de caso 1 | PDF 95 y S1 concuerdan en campos de fibras; PDF 115 presenta otra base y longitud | La discrepancia no prueba un error de importación; registrar versiones de documento |
| Geometría de caso 14 | S1, tabla PDF 95 y ficha PDF 174 presentan valores distintos para algunos campos | No elegir una fuente por conveniencia estadística |
| Conteo de energías vs fracturas | Caso 5 tiene evento 4 en PDF 129 y energía ausente en PDF 131 | Llamar al conteo “eventos con energía disponible”; no número total de fracturas |

### Respuestas, unidades y eventos

Los campos de resumen **First Failure Load, Ultimate Load, Maximum Displacement y residuales incorporados en S1** coinciden numéricamente con las fichas POST-TENSILE TEST DATA examinadas. Las matrices energéticas coinciden con las tablas energéticas del apéndice, incluidos sus signos y ausencias. La fuente energética del caso 7 requiere leer celdas contiguas en una misma línea de texto, lo que se controló durante la extracción.

La coincidencia del resumen no implica que todas las definiciones operacionales puedan reconstruirse. En particular:

| Caso | Página PDF | Resumen First Failure Load (N) | Primer Upper Force (N) | Tratamiento |
|---|---|---|---|---|
| 3 | 119 | 88.50 | 88.78 | Conservar resumen target; diferencia de selección/redondeo pendiente de aclaración |
| 9 | 149 | 61.43 | 61.35 | Conservar resumen target |
| 12 | 164 | 111.98 | 112.22 | Conservar resumen target |

La columna Ultimate Load tampoco es un máximo global: la ficha del caso 1, PDF 115, registra 51.19 N en el resumen terminal y un pico intermedio de 74.71 N. La curva PDF 116 muestra carga posterior al primer evento. El EDA compara respuestas sin imponer el orden “terminal ≥ primera falla”.

### Contraste adicional del resumen terminal

El protocolo explica el concepto de fuerza terminal, pero su resumen numérico no coincide exactamente con el último pico en todas las fichas. Se comparó Ultimate_Load con la última posición Upper Force de cada tabla de eventos. Las diferencias siguientes son internas a la documentación y también están reproducidas en S1.

| Caso | Página PDF | Ultimate_Load del resumen (N) | Último Upper Force (N) |
|---|---|---:|---:|
| 1 | 115 | 51.19 | 50.99 |
| 3 | 119 | 73.79 | 74.49 |
| 5 | 129 | 76.33 | 76.43 |
| 7 | 139 | 53.12 | 53.32 |
| 9 | 149 | 62.95 | 62.94 |
| 10 | 154 | 178.49 | 178.55 |
| 12 | 164 | 75.02 | 75.59 |
| 13 | 169 | 66.50 | 66.35 |
| 17 | 189 | 91.86 | 91.87 |
| 27 | 239 | 70.31 | 191.39 |

**Caso 27 requiere aclaración prioritaria:** el resumen Ultimate_Load = 70.31 coincide numéricamente con Maximum_Displacement = 70.31, mientras que la ficha registra un único pico Upper Force = 191.39 N. Esto sugiere una inconsistencia de origen o una definición no documentada; no demuestra por sí solo cuál valor debe reemplazarse. La columna se mantiene como **carga terminal reportada**, con calidad semántica condicionada a esta verificación. No se modifica la fuerza de primera falla ni se inventa un evento terminal.

## Decisiones de trazabilidad

1. Conservar el MATLAB y las respuestas originales; registrar cada discrepancia como incidencia documental.
2. No “corregir” geometrías con relaciones idealizadas ni seleccionar la versión que produzca mejores asociaciones.
3. Utilizar N, mm, mm³ y mJ según evidencia documental. La equivalencia dimensional 1 N·mm = 1 mJ no implica que se disponga del procedimiento original de integración.
4. Distinguir geometría de diseño, geometría medida, cálculos CAD y proxies: el archivo no permite verificar tolerancias dimensionales reales.
5. Mantener la dependencia por espécimen para ciclos/eventos. La ausencia de información de réplicas no equivale a demostrar independencia.
6. Incorporar mediciones adicionales del PDF solo mediante una extracción futura validada, con procedencia por página y control de versión. La presente tabla digital conserva su alcance.
7. Tratar las asociaciones de geometría discordante como evidencia condicionada a la versión digital y evaluar sensibilidad; no convertirlas en conclusiones causales.

## Archivos y huellas de verificación

Las huellas permiten identificar exactamente las versiones examinadas; no publican los PDF privados.

| Fuente | SHA-256 |
|---|---|
| S1 | 3d82f876bb8ec652184707235aa4df1dfb4b1a6615551f78fe0e0662a6723f0c |
| S2 | ff78de7f05508453cae6d66854d477552f2316fa62a5592e157dcd3f3d384e30 |
| S3 | 578b77ba1612b3898c10ab23cc59a82503b97daf0892d413d9e120941c574a79 |
| S4 | 935b7fd0cf7be785e7233fe4364ce64e124562912436e96001e45bd3984cb78b |
| S5 | c11b0047b530ab6734153521950b4ab329ebda640578badbddcc68da3b9b7f16 |

## Referencias

Las referencias bibliográficas completas están en [introducción del experimento](introduccion_experimento.md#referencias-verificadas). El [diccionario](diccionario_datos.md) detalla disponibilidad temporal, unidades y restricciones de cada variable; el [informe de curaduría](informe_curaduria.md) registra el tratamiento computacional.

**Lectura metodológica:** conocer la unidad de una energía no resuelve sus límites de integración; saber que una variable es geométrica no confirma cada valor; reproducir el MATLAB no demuestra que no existan errores en la fuente. Estas distinciones delimitan qué conclusiones permite sostener el EDA.
