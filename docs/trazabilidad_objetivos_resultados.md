# Trazabilidad de la actividad de permutación

Las formulaciones vigentes son las de [la matriz](matriz_consistencia.md); se distinguen de los contrastes históricos de ingeniería y tuning. Todas las filas siguientes se calculan desde la nueva ejecución.

| Hipótesis | Contraste | Resultado | Lectura |
|---|---|---|---|
| General | Modelo completo con menor MAE que ambas referencias en configuraciones | Ninguno | No respaldada bajo el protocolo |
| H1 | Mismo modelo completo supera ambas referencias en ambos protocolos | Ninguno | No respaldada bajo el protocolo |
| H2 | Mismo modelo reducido no incrementa MAE en ambos protocolos | Proceso gaussiano | Respaldada descriptivamente bajo el protocolo |
| H3 | PFI y ablación positivas simultáneamente en al menos tres modelos completos | Ridge, Elastic Net, SVR RBF, Bosque aleatorio, Proceso gaussiano | Respaldada descriptivamente bajo el protocolo |

Evidencia: `reports/permutation_importance/tables/contrastes_hipotesis.csv`, métricas OOF, selección interna, permutación repetida y ablación. El estudio sigue siendo exploratorio y la revisión no resuelve las aclaraciones pendientes de la fuente.
