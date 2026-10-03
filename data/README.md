# Datos preliminares

Esta carpeta contiene una versión **preliminar y trazable** de la campaña experimental. Permite revisar la ingesta, la curaduría y el EDA; no representa todavía el dataset definitivo de entrenamiento.

| Nivel | Ruta visible | Significado |
|---|---|---|
| Raw | `raw/dati_campagna_venditti.m` | Fuente original recibida. Se conserva sin modificaciones |
| Interim | `interim/preliminary/` | Extracción tabular automática de la fuente, acompañada por su manifiesto y hashes |
| Processed | `processed/preliminary/` | Tabla analítica preliminar y reporte de calidad después de aplicar el esquema |

**Interim** significa etapa intermedia. Su función es convertir la fuente MATLAB en una tabla legible y verificable sin decidir todavía qué variables entrarán al modelo. La versión **processed preliminar** selecciona identificador, predictores y target, registra exclusiones y mantiene las decisiones de calidad.

Las carpetas con identificadores de fecha corresponden a ejecuciones reproducibles y permanecen fuera de Git. `preliminary/` es una copia estable para revisión académica.

La ejecución completa puede revisarse en [01_Ingesta_Curaduria_y_Calidad.ipynb](../notebooks/01_Ingesta_Curaduria_y_Calidad.ipynb) o abrirse directamente como [reporte HTML](../reports/01_Ingesta_Curaduria_y_Calidad.html). El análisis posterior está en [02_EDA_Avanzado.ipynb](../notebooks/02_EDA_Avanzado.ipynb) y en su [reporte HTML](../reports/02_EDA_Avanzado.html).

El conjunto podrá ampliarse con nuevos ensayos o simulaciones físicamente validadas, con una fuente y un protocolo documentados.
