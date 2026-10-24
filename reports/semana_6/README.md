# Informe científico — Semana 6

**[Leer el PDF](Informe_Sprint_Semana_6.pdf)** · **[Descargar proyecto Overleaf](Informe_Sprint_Semana_6_Overleaf.zip)** · [Fuente principal](Informe_Sprint_Semana_6.tex)

UNI · Maestría en Inteligencia Artificial · Trabajo de Investigación II · Melissa Dessire Aylas Barranca.

Sprint centrado en **EDA, ingeniería de características, arquitectura de cinco regresores y evaluación de sus métricas**. El contraste A–D mantiene Ridge fijo; la comparación de algoritmos mantiene la geometría A. No se entrenaron modelos nuevos para esta revisión.

| Representación | MAE OOF (N) | Decisión |
|---|---:|---|
| A: geometría original | 13,247 | Conservar como referencia |
| B: A + dos momentos de sección | 12,835 | Mantener en evaluación |
| C: B + razón logarítmica de volúmenes | 12,914 | Descartar la adición respecto de B |
| D: C − tres columnas directas | 17,035 | Descartar la reducción respecto de C |

B reduce el MAE **3,11 %**, pero no alcanza todos los criterios de adopción. Al reservar familias completas, su MAE sube a **57,654 N**. La comparación es exploratoria: el ajuste de transformaciones está aislado dentro de entrenamiento, pero el EDA completo informó los candidatos y no existe un holdout externo independiente.

La arquitectura de **Ridge, Elastic Net, SVR con núcleo RBF, bosque aleatorio y proceso gaussiano** se explica en la sección 4. La sección 7 compara MAE, RMSE, MedAE, R² y tiempo de los cinco modelos, con una figura y una interpretación por protocolo. Bosque y Ridge presentan MAE casi idénticos por configuración (13,233 y 13,247 N); todos obtienen R² negativo al reservar familias.

## Documento y evidencia

El PDF contiene **15 páginas sustantivas y 19 páginas totales**: portada, índice, cuerpo (folios 1–15), referencias (16) y anexo (17). Incluye cinco figuras, diez tablas y trece referencias. La introducción está resumida y se retiraron las secciones de procedencia y trazabilidad del PDF. Compilación y revisión visual registradas en [verificacion_documento.json](verificacion_documento.json).

| Archivo | Función |
|---|---|
| `Informe_Sprint_Semana_6.tex` | Documento principal editable |
| `main.tex` | Entrada de Overleaf, incluye el documento principal |
| `secciones/` | Objetivo y EDA; features; arquitectura de modelos; validación, métricas e interpretación; anexo |
| `referencias.bib` | Referencias científicas verificadas, estilo IEEE |
| `figures/` | Sello oficial UNI y cinco figuras científicas; fuentes y hashes |
| `tablas/` | Tablas LaTeX exportadas desde los CSV existentes |
| `auditoria_resultados.json` | Verificación de datos, predicciones, particiones y estadísticas train |
| `trazabilidad_tablas.json` | Fuentes y hashes de tablas |
| `generar_tablas.py`, `generar_figuras.py` | Regeneración editorial; no ajustan modelos |
| `compilar.py` | Compilación local con pdfLaTeX y BibTeX |
| `verificar_documento.py` | Comprobación de PDF, fuentes, hashes y métricas |

**Base científica:** commit [`8fdb7961d7eaaaebd9651d0b9a044168893d448a`](https://github.com/melissaaylasb-sudo/T02_MIA14_SeccionB/commit/8fdb7961d7eaaaebd9651d0b9a044168893d448a), rama `feature-engineering`; ejecución finalizada el 17/10/2026, 00:45:30 UTC. El informe se publica exclusivamente en `sprint-semana-6`.

Artefactos originales: [métricas](../feature_engineering/tables/comparacion_experimentos.csv), [predicciones OOF](../feature_engineering/tables/predicciones_oof.csv), [particiones](../feature_engineering/particiones.json), [configuración](../feature_engineering/configuracion.json), [auditoría de transformaciones](../feature_engineering/auditoria_transformaciones.json), [log](../feature_engineering/ejecucion.log) y [manifiesto](../feature_engineering/manifest.json). El [notebook ejecutado](../../notebooks/04_Feature_Engineering_Pantograficas.ipynb) conserva el desarrollo del experimento.

Los conteos metodológicos, las discrepancias documentales, las dependencias nominales, los resultados negativos y el sesgo exploratorio se explican en el cuerpo. No se redistribuyen PDF fuente privados. El [sello UNI](figures/LOGO_FUENTE.md) procede del manual institucional oficial.

## Compilar en Overleaf

1. Crear un proyecto mediante **Upload Project** y subir `Informe_Sprint_Semana_6_Overleaf.zip`.
2. Seleccionar `main.tex` como documento principal y **pdfLaTeX** como compilador.
3. Compilar. Overleaf ejecuta BibTeX para las referencias.

El ZIP contiene los fuentes, tablas, figuras, bibliografía e instrucciones. Compila de forma autónoma: no requiere datos experimentales, Python ni acceso a las fuentes privadas. Los enlaces a artefactos del repositorio se consultan en GitHub.

## Compilar localmente

Con TeX Live o MiKTeX y Python disponibles, ejecutar desde la raíz del repositorio:

```text
python reports/semana_6/compilar.py
```

Si los ejecutables TeX no están en PATH, añadir `--tex-bin "DIRECTORIO_DE_LATEX"`. El script realiza pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX y comprueba diagnósticos. Alternativamente, dentro de esta carpeta:

```text
pdflatex -interaction=nonstopmode -halt-on-error Informe_Sprint_Semana_6.tex
bibtex Informe_Sprint_Semana_6
pdflatex -interaction=nonstopmode -halt-on-error Informe_Sprint_Semana_6.tex
pdflatex -interaction=nonstopmode -halt-on-error Informe_Sprint_Semana_6.tex
```

Para regenerar figuras y tablas, se necesita el repositorio completo y pandas, NumPy y Matplotlib; las versiones de la ejecución constan en el manifiesto original. Desde la raíz:

```text
python reports/semana_6/generar_tablas.py
python reports/semana_6/generar_figuras.py
python reports/semana_6/compilar.py
python reports/semana_6/verificar_documento.py
```

El verificador del documento requiere PyMuPDF y Pillow, además de pandas/NumPy para contrastar métricas. `python -m src.verificar_fe` comprueba la entrega experimental existente. **`python -m src.reporte_fe --execute` vuelve a entrenar y sobrescribe artefactos de esa ejecución**; utilizarlo solo en una copia de reproducción, no para compilar este informe.

La comprobación automática no sustituye revisar visualmente un PDF que se modifique posteriormente. La revisión registrada corresponde al hash del PDF entregado.
