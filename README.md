# Ingeniería de características de estructuras pantográficas

**[Abrir notebook 04 ejecutado →](notebooks/04_Feature_Engineering_Pantograficas.ipynb)**

Melissa Dessire Aylas Barranca · UNI, Maestría en Inteligencia Artificial · Trabajo de Investigación II.

Se evalúa si transformar la geometría mejora la predicción de **`First_Failure_Load_N`**: fuerza, en newtons, de la primera rotura de una barra o unión de la malla.

## Resultado

| Experimento | Cambio | MAE (N) | RMSE (N) |
| --- | --- | --- | --- |
| A | Geometría original | 13.2466 | 16.3457 |
| B | A + dos momentos de sección | 12.8346 | 16.2029 |
| C | B + ln(Vp/Vf) | 12.9139 | 16.3271 |
| D | C − tres descriptores redundantes | 17.0350 | 21.1361 |

**B obtiene el menor MAE: 12.8346 N**, una reducción de 3.11 % frente a A. Al excluir familias completas, su MAE es 57.6538 N; la mejora no demuestra transferencia a arquitecturas nuevas. **B se mantiene en evaluación**; C no mejora B y D empeora. El EDA previo usó el mismo conjunto: la evidencia es exploratoria.

## Qué encontrarás en el notebook

Hipótesis del EDA → fórmulas → pipeline → comparación A/B/C/D → cinco gráficos → interpretación. Ridge y particiones se mantienen fijos; las transformaciones se ajustan solo en entrenamiento. Se excluyen configuraciones completas y, en una prueba adicional, familias completas.

## Estructura del proyecto

```text
notebooks/
  01_Ingesta_Curaduria_y_Calidad.ipynb
  02_EDA.ipynb
  03_Transformaciones_Exploratorias.ipynb
  04_Feature_Engineering_Pantograficas.ipynb  # Entrega actual ejecutada
src/
  feature_engineering.py   # Contrato, fórmulas y pipeline
  experimentos_fe.py       # Particiones, ajustes y métricas OOF
  informe_fe.py            # Resúmenes ligados a resultados
  reporte_fe.py            # Ejecución del notebook 04
  verificar_fe.py          # Integridad y coherencia de la entrega
config/feature_engineering.json  # Parámetros e intervenciones
reports/feature_engineering/
  tables/                  # Métricas, predicciones y decisiones
  figures/                 # PNG a 300 dpi y SVG
  particiones.json         # Casos de train y validación
  auditoria_transformaciones.json  # Estadísticas aprendidas por fold
  manifest.json            # Fuentes, versiones y hashes
  verificacion.json        # Resultado de comprobaciones
  ejecucion*.log           # Registro de ejecución
data/                     # Fuente y tablas curadas heredadas del EDA
docs/                     # Diccionarios y notas metodológicas
tests/                    # Pruebas, incluidos grupos y fuga de información
```

## Consulta y reproducción

[Diccionario de variables](docs/diccionario_datos.md#consulta-rápida) · [Nuevas features](docs/diccionario_features.md) · [Protocolo](docs/feature_engineering.md) · [Tabla de resultados](reports/feature_engineering/tables/comparacion_experimentos.csv) · [EDA anterior](https://github.com/melissaaylasb-sudo/T02_MIA14_SeccionB/tree/eda)

```powershell
python -m src.reporte_fe --execute
python -m pytest tests -q
python -m src.verificar_fe
```

Los datos originales y los tres cuadernos del EDA se conservan. Los PDF y las referencias privadas no se publican. Siguiente paso: confirmar A y B con ensayos independientes y geometrías conciliadas.
