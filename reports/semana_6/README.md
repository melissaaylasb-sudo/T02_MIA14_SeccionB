# Informe de sprint — Semana 6

**[Leer el informe](Informe_Sprint_Semana_6.pdf)** · **[Descargar fuentes para Overleaf](Informe_Sprint_Semana_6_Overleaf.zip)**

UNI · Maestría en Inteligencia Artificial · Trabajo de Investigación II · Melissa Dessire Aylas Barranca.

**EDA + Feature Engineering + línea base y dos variantes de datos.** Incluye diccionario con entradas y salidas, curaduría, análisis univariado y bivariado, atípicos e influencia, interpretación y conexión con las transformaciones. Los modelos utilizados se identifican en una tabla; las métricas y la validación cierran el experimento.

La apertura resume contexto y objetivo. Las interpretaciones del EDA se sintetizan en tablas; la matriz de características distingue entradas originales, derivadas y filtradas mediante colores. El balance experimental usa azul para la referencia, ámbar para mantener en evaluación y rojo para una adición no favorable.

| Representación | Intervención | MAE OOF por configuración (N) | Resultado |
|---|---|---:|---|
| A: línea base | Geometría original | 13,247 | Conservar como referencia |
| B: variante 1 | A + dos momentos de sección | 12,835 | Mantener en evaluación |
| C: variante 2 | B + razón logarítmica de volúmenes | 12,914 | No incorporar la adición respecto de B |

Se mantiene Ridge fijo para medir el aporte de las características. B reduce MAE 3,11 %, pero no cumple todos los criterios de adopción. La validación por familias muestra un error cercano a 58 N; las conclusiones son exploratorias y requieren contraste independiente. No se reentrenaron modelos para redactar el informe.

## Contenido y archivos

| Archivo o carpeta | Función |
|---|---|
| `Informe_Sprint_Semana_6.tex`, `main.tex` | Documento principal y entrada para Overleaf |
| `secciones/datos_eda.tex` | Diccionario, preparación y análisis exploratorio |
| `secciones/features.tex` | Construcción, tratamiento y conjuntos A/B/C |
| `secciones/evaluacion_y_discusion.tex` | Modelos utilizados, validación, métricas y resultados |
| `figures/`, `tablas/` | Figuras científicas y tablas exportadas de resultados existentes |
| `referencias.bib` | Bibliografía |
| `preparar_eda_sprint.py` | Recuperación y representación de evidencia EDA |
| `generar_tablas.py`, `generar_figuras.py` | Exportación de resultados experimentales |
| `compilar.py`, `verificar_documento.py` | Compilación y comprobación del entregable |
| `verificacion_documento.json` | Resultado de comprobaciones y revisión visual |

Desarrollo original: [curaduría](../../notebooks/01_Ingesta_Curaduria_y_Calidad.ipynb), [EDA](../../notebooks/02_EDA.ipynb), [transformaciones](../../notebooks/03_Transformaciones_Exploratorias.ipynb) y [Feature Engineering](../../notebooks/04_Feature_Engineering_Pantograficas.ipynb). Las comparaciones adicionales de la ejecución histórica permanecen en [los resultados originales](../feature_engineering/); el informe delimita su experimento principal a A, B y C.

## Compilar

En Overleaf: **Upload Project → subir el ZIP → seleccionar `main.tex` → pdfLaTeX**. El paquete contiene todo lo necesario para compilar; no requiere datos ni documentos privados. La carátula usa el logo proporcionado por la autora, sin modificarlo.

Para compilar localmente desde la raíz del repositorio, con Python y TeX Live/MiKTeX:

```text
python reports/semana_6/compilar.py
```

Si TeX no está en PATH, añadir `--tex-bin "DIRECTORIO_DE_LATEX"`. Para regenerar las figuras y tablas desde las fuentes existentes, sin entrenamiento:

```text
python reports/semana_6/preparar_eda_sprint.py
python reports/semana_6/generar_tablas.py
python reports/semana_6/generar_figuras.py
python reports/semana_6/compilar.py
python reports/semana_6/verificar_documento.py
python reports/semana_6/empaquetar_overleaf.py
```

La regeneración requiere pandas, NumPy, Matplotlib, PyMuPDF y Pillow. Los manifiestos conservan hashes de las fuentes y exportaciones. La revisión visual registrada corresponde al PDF entregado; debe repetirse tras cambiar su contenido o formato.
