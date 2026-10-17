# Diccionario de características

[Notebook principal](../notebooks/04_Feature_Engineering_Pantograficas.ipynb) · [Correspondencia español–fuente CSV](../reports/feature_engineering/tables/nombres_variables.csv) · [Diccionario técnico CSV](../reports/feature_engineering/tables/diccionario_features.csv)

## Variables de esta etapa

El notebook presenta las variables en español. La columna original permite rastrear cada medición hasta la tabla curada, sin modificar los datos de origen.

| Variable en español | Significado | Columna original | Unidad | Rol |
| --- | --- | --- | --- | --- |
| numero_celdas_y | Número de celdas en Y | n_cells_Y | conteo | Entrada original |
| altura_fibra_mm | Altura de fibra | Fiber_height | mm | Entrada original |
| ancho_fibra_mm | Ancho de fibra | Fiber_base | mm | Entrada original |
| longitud_total_fibras_mm | Longitud total de fibras | Fiber_total_length | mm | Entrada original |
| volumen_total_fibras_mm3 | Volumen total de fibras | Fiber_total_volume | mm³ | Entrada original |
| altura_pivote_mm | Altura de pivote | Pivot_height | mm | Entrada original |
| radio_pivote_mm | Radio de pivote | Pivot_radius | mm | Entrada original |
| numero_pivotes | Número de pivotes | Pivot_total_number | conteo | Entrada original |
| volumen_total_pivotes_mm3 | Volumen total de pivotes | Pivot_total_volume | mm³ | Entrada original |
| volumen_total_especimen_mm3 | Volumen total del espécimen | Sample_total_volume | mm³ | Entrada original |
| carga_primera_falla_n | Carga de primera falla | First_Failure_Load_N | N | Objetivo / salida predicha |
| momento_fibra_bh3_mm4 | Segundo momento de área b·h³/12 | I_fibra_bh3_mm4 | mm⁴ | Entrada derivada |
| momento_fibra_hb3_mm4 | Segundo momento de área h·b³/12 | I_fibra_hb3_mm4 | mm⁴ | Entrada derivada |
| log_razon_volumenes_pivotes_fibras | Logaritmo de la razón de volúmenes pivotes/fibras | log_volumen_pivotes_sobre_fibras | adimensional | Entrada derivada |

## Variables construidas

| Feature | Variables de origen | Fórmula | Unidad | Significado físico |
| --- | --- | --- | --- | --- |
| momento_fibra_bh3_mm4 | Ancho de fibra = b; altura de fibra = h | b*h³/12 | mm⁴ | Segundo momento de área: eje paralelo a b |
| momento_fibra_hb3_mm4 | Ancho de fibra = b; altura de fibra = h | h*b³/12 | mm⁴ | Segundo momento de área: eje paralelo a h |
| log_razon_volumenes_pivotes_fibras | Volumen de pivotes = Vp; volumen de fibras = Vf | ln(Vp/Vf) | 1 | Razón entre volúmenes declarados |

Los momentos se evalúan juntos en B; son geometría de sección, no rigidez EI medida. La razón C es adimensional y exige volúmenes positivos. Todas las fórmulas usan entradas previas al ensayo, después de la imputación ajustada en entrenamiento.

El ancho de fibra (base de la sección) se incluye en todas las variantes aunque el esquema general actual la omita. El radio constante se retira dentro de cada entrenamiento. D retira longitud total de fibras, cantidad de pivotes y volumen total de fibras del bloque original; este último sigue alimentando la razón. El objetivo es la **carga de primera falla [N]**; identificadores y otras respuestas del ensayo no son predictores. [Definiciones experimentales completas](diccionario_datos.md#consulta-rápida).
