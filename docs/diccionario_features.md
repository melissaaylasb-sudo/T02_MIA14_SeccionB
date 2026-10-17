# Diccionario de características

[Notebook principal](../notebooks/04_Feature_Engineering_Pantograficas.ipynb) · [Variables originales, INPUT y OUTPUT](diccionario_datos.md#consulta-rápida) · [Diccionario completo CSV](../reports/feature_engineering/tables/diccionario_features.csv)

| Feature | Variables de origen | Fórmula | Unidad | Significado físico |
| --- | --- | --- | --- | --- |
| I_fibra_bh3_mm4 | Fiber_base=b; Fiber_height=h | b*h³/12 | mm⁴ | Segundo momento de área: eje paralelo a b |
| I_fibra_hb3_mm4 | Fiber_base=b; Fiber_height=h | h*b³/12 | mm⁴ | Segundo momento de área: eje paralelo a h |
| log_volumen_pivotes_sobre_fibras | Pivot_total_volume=Vp; Fiber_total_volume=Vf | ln(Vp/Vf) | 1 | Razón entre volúmenes declarados |

Los momentos se evalúan juntos en B; son geometría de sección, no rigidez EI medida. La razón C es adimensional y exige volúmenes positivos. Todas las fórmulas usan entradas previas al ensayo, después de la imputación ajustada en entrenamiento.

`Fiber_base` se incluye en todas las variantes aunque el esquema general actual la omita. El radio constante se retira del ajuste dentro de cada fold. D retira `Fiber_total_length`, `Pivot_total_number` y `Fiber_total_volume` del bloque original; este último sigue alimentando la razón. El target es `First_Failure_Load_N` (N); identificadores y otras respuestas no son predictores.
