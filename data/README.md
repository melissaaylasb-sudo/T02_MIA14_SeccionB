# Datos preliminares

Esta carpeta contiene una versión **preliminar y trazable** de la campaña experimental. Permite revisar la ingesta, la curaduría y el EDA; no representa todavía el dataset definitivo de entrenamiento.

| Nivel | Ruta visible | Significado |
|---|---|---|
| Raw | `raw/dati_campagna_venditti.m` | Fuente original recibida. Se conserva sin modificaciones |
| Interim | `interim/preliminary/` | Extracción tabular automática de la fuente, acompañada por su manifiesto y hashes |
| Processed | `processed/preliminary/` | Tabla analítica preliminar y reporte de calidad después de aplicar el esquema |

**Interim** significa etapa intermedia. Su función es convertir la fuente MATLAB en una tabla legible y verificable sin decidir todavía qué variables entrarán al modelo. La versión **processed preliminar** selecciona identificador, predictores y target, registra exclusiones y mantiene las decisiones de calidad.

Las carpetas con identificadores de fecha corresponden a ejecuciones reproducibles y permanecen fuera de Git. `preliminary/` es una copia estable para revisión académica.

El conjunto podrá ampliarse con nuevos ensayos o simulaciones físicamente validadas. Una meta cercana al orden de mil observaciones debe entenderse como una **proyección de adquisición**, condicionada a una fuente y un protocolo aprobados; no describe el volumen disponible actualmente.
