# Diccionario de características para la predicción

Amplía el [diccionario del EDA](diccionario_datos.md); no cambia sus definiciones. Todas las entradas se conocen antes del ensayo. `First_Failure_Load_N` es el OUTPUT objetivo, en N. `Case_n` es un identificador, no un predictor.

## Atributos originales admisibles

| Variable | Unidad | Significado | Uso |
| --- | --- | --- | --- |
| n_cells_Y | conteo | Celdas en dirección Y | Entrada; A–D |
| Fiber_height | mm | Altura de sección de fibra | Entrada; A–D |
| Fiber_base | mm | Ancho de sección de fibra | Entrada; A–D |
| Fiber_total_length | mm | Longitud sumada de fibras | A–C; retirada de bloque original en D |
| Fiber_total_volume | mm³ | Volumen declarado de fibras | A–C; retirada de bloque original en D |
| Pivot_height | mm | Altura de pivote | Entrada; A–D |
| Pivot_radius | mm | Radio de pivote | Entrada conservada; filtro de varianza la retira del ajuste |
| Pivot_total_number | conteo | Cantidad de pivotes | A–C; retirada de bloque original en D |
| Pivot_total_volume | mm³ | Volumen declarado de pivotes | Entrada; A–D |
| Sample_total_volume | mm³ | Volumen CAD del espécimen | Entrada; A–D |

`Fiber_base` se incluye en A–D como decisión explícita frente a su omisión en el esquema general actual. En D, `Fiber_total_volume` sigue alimentando la razón logarítmica. No se sustituyen volúmenes declarados por cálculos ideales ni por la suma de partes.

## Atributos derivados evaluados

| Feature | Variables de origen | Fórmula | Unidad | Significado físico | Hipótesis | Tratamiento de nulos | Riesgo de leakage | Experimento | Decisión |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I_fibra_bh3_mm4 | Fiber_base=b; Fiber_height=h | b*h³/12 | mm⁴; absoluta | Segundo momento de área respecto al eje paralelo a b | Representar dependencia cúbica de la sección | Ingredientes imputados en train antes de fórmula | Bajo en cálculo; hipótesis informada por EDA global | B–D | Mantener en evaluación (bloque de dos momentos) |
| I_fibra_hb3_mm4 | Fiber_base=b; Fiber_height=h | h*b³/12 | mm⁴; absoluta | Segundo momento de área respecto al eje paralelo a h | Representar la otra orientación de sección | Ingredientes imputados en train antes de fórmula | Bajo en cálculo; hipótesis informada por EDA global | B–D | Mantener en evaluación (bloque de dos momentos) |
| log_volumen_pivotes_sobre_fibras | Pivot_total_volume=Vp; Fiber_total_volume=Vf | ln(Vp/Vf) | 1; adimensional | Razón logarítmica entre volúmenes declarados | Representar balance de componentes con función no lineal | Ingredientes positivos, imputados solo en train | No usa target; sesgo de hipótesis por EDA previo | C–D | Descartar (incremento frente a B) |

Para los momentos, mm × mm³ = mm⁴. No se multiplican por un módulo elástico no disponible; no equivalen a rigidez EI ni permiten inferir directamente la falla. Para el logaritmo, mm³/mm³ = 1: el argumento es adimensional y estrictamente positivo. Ninguna fórmula usa fuerza, desplazamiento ni energía.

Los momentos reutilizan `geometry_features` del cuaderno 03. Se evalúan los dos ejes sin afirmar cuál domina en cada espécimen. El logaritmo se calcula como ln(Vp)−ln(Vf), numéricamente equivalente a ln(Vp/Vf) al usar las mismas unidades, para evitar desbordar el cociente.

## Propuestas que no se incorporan

Área de fibra y relación de aspecto ya se exploraron; no se agregan simultáneamente a los momentos para evitar proliferación de descriptores. Altura/diámetro de pivote replica aquí su altura porque el radio es constante. Una esbeltez de tramo requiere longitud libre y condiciones de unión verificadas, no longitud total de todas las fibras. Densidades volumétricas requieren volumen de envolvente y espesor global verificables; no se inventan. Volumen CAD/celdas Y no es volumen físico de una celda y se deja fuera de esta comparación. No se usan F/V, energías o desplazamientos como predictores.

StandardScaler crea coordenadas numéricas sin unidad para el ajuste, no nuevas mediciones físicas. Las fórmulas geométricas se calculan antes del escalamiento.
