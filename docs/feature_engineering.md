# Protocolo de ingeniería de características

**Entrega principal:** [notebook 04 ejecutado](../notebooks/04_Feature_Engineering_Pantograficas.ipynb). [Matriz EDA → FE completa](../reports/feature_engineering/tables/matriz_eda_fe.csv).

| Elemento | Decisión |
|---|---|
| Fuente | Tabla curada `model_table.csv`, contrastada con su manifiesto; target observado, sin imputación. |
| Objetivo | `First_Failure_Load_N` (N): fuerza de primera rotura de fibra o pivote. |
| Entradas | Geometría previa al ensayo; se excluyen identificador, fuerzas, desplazamientos y energías. |
| A / B / C / D | Originales / añadir momentos / añadir log de razón / retirar bloque redundante. |
| Modelo | Ridge, alpha=1, solver SVD; mismos parámetros y particiones. Dummy de media como control. |
| Pipeline | Contrato → mediana → fórmulas → filtro de varianza → StandardScaler → Ridge; ajuste solo en train. |
| Validación principal | LeaveOneGroupOut por (`n_cells_Y`, `Fiber_height`, `Pivot_height`); factores nominales repetidos permanecen juntos. |
| Comprobación adicional | Excluir cada familia Y completa; evalúa una arquitectura no vista. |
| Métricas | MAE principal; RMSE, MedAE y R² agregado OOF. Sin R² en folds individuales. |
| Selección | Retirada fija por hipótesis geométrica; varianza nula aprendida en train. Sin selección supervisada ni tuning. |
| Decisión | Adoptar provisionalmente si MAE mejora ≥5 %, gana en ≥60 % de grupos y no empeora RMSE ni MAE por familia; si solo mejora MAE, mantener en evaluación; si no, descartar. |

Las hipótesis proceden del EDA del mismo conjunto: los resultados son exploratorios y pueden tener sesgo de selección. Faltan réplicas/lotes verificados y persisten discrepancias geométricas. Las familias no representan campañas independientes.

Las nuevas features son funciones de mediciones existentes. Retirar columnas modifica también la penalización efectiva de Ridge. No se recortan predicciones negativas ni se eliminan extremos. La sensibilidad por omisión modifica el agregado de errores OOF, no reentrena modelos.

Los predictores son numéricos: no se necesita ColumnTransformer ni codificación categórica. Se difieren RobustScaler, selección por información mutua y nuevos descriptores para no mezclar intervenciones; la evidencia y las exclusiones constan en la matriz CSV.

[Configuración](../config/feature_engineering.json) · [Particiones](../reports/feature_engineering/particiones.json) · [Estadísticas aprendidas](../reports/feature_engineering/auditoria_transformaciones.json)
