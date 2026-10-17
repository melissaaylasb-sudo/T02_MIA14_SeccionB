# Ingeniería de características guiada por el EDA

**Objetivo:** estimar `First_Failure_Load_N` (N), la fuerza asociada al primer evento de rotura identificado de fibra o pivote. No es la fuerza máxima, la carga terminal ni la detección de rotura en tiempo real.

Una estructura pantográfica es una malla de barras (fibras) conectadas por uniones (pivotes). La familia indica el número de celdas en Y. Esta etapa continúa el EDA publicado en `e9b2cc3f9824922601ff40a8e8960b02f44de876` y no recalcula sus figuras.

## Fuente y frontera temporal

Se utiliza `data/processed/preliminary/model_table.csv`, contrastado mediante SHA-256 con su reporte de curaduría. Es la tabla con objetivo observado; la geometría del caso sin respuesta queda en la fuente original. No se imputa el target. `Fiber_base` se incluye expresamente en todos los experimentos: existe en el snapshot aunque el esquema general posterior la omita. El contrato específico está en `src/feature_engineering.py` y la configuración en `config/feature_engineering.json`.

Entradas: `n_cells_Y`, `Fiber_height`, `Fiber_base`, `Fiber_total_length`, `Fiber_total_volume`, `Pivot_height`, `Pivot_radius`, `Pivot_total_number`, `Pivot_total_volume`, `Sample_total_volume`. `Case_n` identifica predicciones, nunca entra al ajuste. Carga terminal, desplazamientos, energías, ciclos y eventos se excluyen por disponibilidad temporal, independientemente de su correlación. El archivo contiene resultados experimentales resumidos, no señales continuas. Persiste la discrepancia entre el resumen de primera falla y ciertos primeros picos documentales; se conserva el target publicado sin redefinirlo.

## Matriz de decisiones EDA → FE

Las rutas de evidencia son relativas a la raíz del repositorio; cada archivo se mantiene intacto. La [matriz CSV](../reports/feature_engineering/tables/matriz_eda_fe.csv) facilita su consulta.

| Hallazgo EDA | Evidencia | Problema u oportunidad | Feature propuesta | Justificación física | Transformación | Riesgo | Experimento |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Separación entre familias: 96.55 % de variación entre grupos | reports/eda/tables/05_descomposicion.csv | Arquitectura dominante; generalización incierta | Conservar n_cells_Y y probar familia excluida | Conteo de celdas previo al ensayo | Conteo numérico; no target encoding | Una familia nueva puede exigir extrapolación | A–D; protocolo secundario |
| Momentos de sección: asociación condicionada próxima a cero | reports/eda_transformaciones/tables/T08_asociaciones_caracteristicas.csv | Hipótesis mecánica aún no validada predictivamente | I_fibra_bh3_mm4; I_fibra_hb3_mm4 | Segundos momentos de área de sección rectangular ideal, según dos ejes | b*h^3/12; h*b^3/12 | No son EI, rigidez medida ni criterio de fractura; base discrepante | B − A |
| log(Vp/Vf): rho global 0.8280 y condicionado −0.4773 | reports/eda_transformaciones/tables/T08_asociaciones_caracteristicas.csv | Asociación dependiente de familia; evaluar otra representación | log_volumen_pivotes_sobre_fibras | Balance de volúmenes declarados de componentes | ln(Vp/Vf), adimensional | Asociación explorada con todo el target; posible sesgo de selección | C − B |
| Vf + Vp = 6524.01 mm³; radio constante | reports/eda/tables/12_identidades.csv; reports/eda_calidad/tables/14_decisiones.csv | Redundancia algebraica y varianza nula | Retirar Vf del bloque original; filtro de varianza en train | Los volúmenes complementarios no son efectos independientes | Retirada fija de Vf en D; VarianceThreshold en todos | Ridge cambia su penalización al cambiar representación | D − C; A–D |
| Longitud y número de pivotes quedan determinados por la familia observada | reports/eda/tables/12_mapa_topologia.csv | Descriptores repetidos de arquitectura | Retirar Fiber_total_length y Pivot_total_number | Se conserva el conteo Y | Reducción física fija, sin elegir por target | Determinismo en este diseño no implica relación universal ni redundancia lineal exacta | D − C |
| Rango centrado 7 entre 9 entradas variables; VIF infinitos | reports/eda_transformaciones/tables/T03_representaciones.csv; reports/eda/tables/16_VIF.csv | Colinealidad impide atribuir efectos individuales | Ridge y ablación de bloque | No interpretar coeficientes como mecanismos aislados | Regularización fija alpha=1; diagnóstico de rango en cada train | Retirar columnas modifica regularización efectiva | A–D |
| Log de volumen CAD no reduce asimetría; log de Vp sí la atenúa | reports/eda_transformaciones/tables/T04_logaritmos.csv | No hay soporte para logaritmos masivos | Una razón logarítmica dirigida | Argumento positivo y adimensional | No transformar target ni todos los volúmenes | Log no añade una medición nueva | C − B |
| Geometría completa; objetivo ausente conservado fuera de tabla supervisada | reports/eda_calidad/tables/14_decisiones.csv | No inventar etiquetas ni señales de ausencia | Imputación de entradas dentro del pipeline | Mediana solo de train como contingencia; no reconstruye medición | SimpleImputer; sin indicadores de ausencia no observados | Futuros nulos deben revisarse; columna sin soporte en train rechazada | A–D |
| Sin alertas IQR/MAD en geometría y target; Cook señala casos 1 y 3 | reports/eda/tables/07_atipicos_resumen.csv; reports/eda/tables/20_influencia.csv | Influencia sin justificación de eliminación | Conservar casos y estudiar contribución a errores OOF | Extremo geométrico puede ser un diseño real | StandardScaler; sin winsorización; RobustScaler diferido | Excluir scores no elimina influencia en modelos entrenados | A–D; sensibilidad descriptiva |
| Factores nominales repetidos; no constan lotes o réplicas físicas | reports/eda_calidad/tables/09_configuraciones_repetidas.csv | Dependencia potencial | Grupo n_cells_Y, Fiber_height, Pivot_height | Mismo conjunto nominal no implica espécimen idéntico | LeaveOneGroupOut sin dividir el grupo | No controla dependencia de lotes desconocidos | A–D; validación principal |
| Discrepancias geométricas de base y alturas; casos 23,25,26 señalados en pivotes | reports/eda_calidad/tables/14_decisiones.csv | Incertidumbre en atributos derivados | Conservar snapshot y registrar sensibilidad | No sustituir una fuente automáticamente por otra | Comparación de scores con y sin casos señalados | No es validación con geometría corregida | A–D |
| Energía, residuales y carga terminal se conocen durante/después del ensayo | docs/diccionario_datos.md; reports/eda_transformaciones/tables/T06_frontera_predictiva.csv | Fuga temporal aunque haya correlación alta | Lista cerrada de entradas geométricas | Solo información disponible antes del ensayo | Contrato rechaza columnas adicionales; Case_n solo traza | Errores de metadatos requieren verificación experimental | A–D |

## Intervenciones y control experimental

- **A:** geometría original admisible, imputación mediana, varianza nula, StandardScaler y Ridge.
- **B = A +** dos segundos momentos de área de la sección de fibra. Se evalúa la familia de descriptores conjuntamente; no se atribuye una mejora a un eje individual.
- **C = B +** logaritmo de la razón entre volumen de pivotes y de fibras.
- **D = C −** longitud total de fibras, cantidad de pivotes y volumen de fibras del bloque original. El volumen de fibras sigue siendo necesario para construir la razón; no desaparece del contrato de entrada.

Se mantienen `Ridge(alpha=1, solver='svd')`, métrica, escalador y particiones. Alpha=1 es una referencia fija, no un óptimo estimado. No se hace tuning. La media del target de entrenamiento (`DummyRegressor`) es un control adicional, no la única comparación. No existía un baseline entrenado en esta rama; el módulo del cuaderno 03 solo preparaba representaciones, por lo que se reutilizan sus fórmulas y se agrega la evaluación supervisada.

Las features derivadas no agregan mediciones independientes: cambian la base funcional que recibe Ridge. Al reducir columnas también cambia la penalización efectiva, aun manteniendo alpha; esa es parte de la intervención D y no demuestra por sí sola irrelevancia física.

## División primero; transformaciones después

`Pipeline`: contrato geométrico → mediana de entrenamiento → fórmulas por fila → filtro de varianza nula → StandardScaler ajustado en train → Ridge. El contrato rechaza columnas extra, infinitos, dimensiones no positivas y conteos no enteros. Si una columna está totalmente ausente en entrenamiento, se detiene el ajuste. La construcción posterior a imputación permite transformar observaciones nuevas con estadísticas guardadas, aunque no sustituye la revisión de mediciones faltantes.

Todos los campos elegidos son numéricos y reciben el mismo tratamiento: no hace falta un `ColumnTransformer`. `n_cells_Y` es un conteo con orden físico, no una etiqueta ordinal arbitraria. No se añade One-Hot Encoding en esta comparación: sería otra hipótesis, usaría más parámetros y no permitiría representar directamente una familia nunca observada. No se usa target encoding.

StandardScaler evita que mm, mm³ y mm⁴ penalicen de manera incompatible. RobustScaler queda como experimento posterior: no se observan alertas geométricas IQR/MAD que justifiquen cambiarlo simultáneamente. No se winsoriza ni se eliminan extremos. Los faltantes geométricos no observados no justifican crear indicadores constantes. Solo se calcula el logaritmo positivo de una razón adimensional; no se transforman respuestas.

La selección D es fija y motivada por el diseño, no aprendida con el target. La única selección ajustada es la varianza nula dentro del fold. VIF y rango del EDA motivan estudiar redundancia; el rango se registra de nuevo exclusivamente en cada entrenamiento. No se usa información mutua ni selección univariada supervisada: su estimación y estabilidad no tienen respaldo suficiente para añadirlas a esta comparación. Se entregan coeficientes por fold para auditoría, sin interpretarlos como importancia causal. La contribución se evalúa por ablación pareada; no se usa permutación marginal de variables físicamente dependientes.

## Unidad experimental y validación

La fila es el caso experimental recibido. No están verificados identificadores físicos de réplica ni lotes. Los ciclos y eventos no se expanden como filas independientes. La combinación nominal `n_cells_Y`, `Fiber_height`, `Pivot_height` define grupos conservadores, derivados únicamente de geometría. Los pares de casos 19/25 y 24/27 comparten estos factores, aunque difieren otras dimensiones; no se los declara réplicas.

**Principal:** LeaveOneGroupOut por configuración nominal. Se excluye por completo una combinación de factores; pueden permanecer casos de la misma familia en train. Por ello evalúa configuraciones excluidas dentro del dominio experimental, no transferencia a otras campañas.

**Secundaria:** LeaveOneGroupOut por familia Y. No participa en el ajuste; examina la dificultad de una arquitectura ausente. Las familias extremas exigen extrapolación y la intermedia interpolación entre familias. Las diferencias entre protocolos responden a preguntas distintas.

Ridge no impone positividad de la respuesta. Se registran predicciones negativas como resultados físicamente inadmisibles para este objetivo; no se recortan a cero porque eso cambiaría el experimento. La tabla de validación informa su presencia y el mínimo predicho. No se propone despliegue de estos ajustes.

Las listas exactas de train/test se guardan una vez en [particiones.json](../reports/feature_engineering/particiones.json) y se reutilizan en todas las variantes. Se conservan tamaños efectivos en las tablas técnicas. No se aplica holdout 80/20. La independencia respecto a lotes desconocidos no puede garantizarse.

## Métricas y criterio de decisión

MAE (N) es principal; RMSE (N), MedAE (N) y R² sobre todas las predicciones OOF son secundarios. El residual es observado menos predicho. No se calcula R² por fold individual. El score principal pondera cada caso por igual; media y desviación de MAE por fold ponderan grupos por igual y describen dispersión, no un intervalo de confianza. Los entrenamientos de CV se solapan: no se tratan folds como réplicas independientes.

El número final de características corresponde a las columnas que salen del preprocesamiento. El control Dummy ignora esas columnas y utiliza únicamente la media del target de train.

ΔMAE y ΔRMSE negativos significan mejora. La tabla principal compara con A; las ablaciones comparan B−A, C−B y D−C. Los tiempos suman ajuste y predicción, excluyen figuras y documentación y dependen del equipo.

Regla fijada antes de ejecutar: **Adoptar** provisionalmente una intervención si reduce MAE al menos 5 %, mejora en al menos 60 % de grupos, no empeora RMSE ni MAE al excluir familias. **Mantener en evaluación** si mejora MAE principal sin cumplir todos esos criterios. **Descartar** la intervención en esta comparación si no mejora MAE. Son umbrales operativos exploratorios, no pruebas de significación. Una conclusión de descarte no demuestra ausencia de efecto mecánico.

El EDA previo usó todas las observaciones y el target para formular hipótesis. Aunque las transformaciones se ajustan dentro de cada entrenamiento, existe posible sesgo de selección de hipótesis. Los resultados son exploratorios: se necesita una campaña independiente o una evaluación externa con el protocolo congelado. No se afirma causalidad, significación estadística ni seguridad estructural a partir de estas métricas.

## Reproducción

```powershell
python -m src.experimentos_fe
python -m src.reporte_fe --execute
python -m pytest tests -q
python -m src.verificar_fe
```

El primer comando ejecuta los experimentos y genera tablas, figuras y documentos. El segundo vuelve a ejecutar la comparación desde el notebook con kernel limpio y guarda los outputs. No modifica fuentes ni artefactos del EDA. Los tiempos pueden variar; las predicciones son deterministas con los datos y configuración congelados.
