"""Ejecuta opcionalmente el EDA y exporta su informe HTML de lectura."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import re

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from .ingesta import ROOT
from .eda import write_bytes_atomic

NOTEBOOKS = ["01_Ingesta_Curaduria_y_Calidad.ipynb", "02_EDA_Avanzado.ipynb",
             "03_Transformaciones_Exploratorias.ipynb"]


def export_report(*, execute: bool = False, notebook_name: str = NOTEBOOKS[1]) -> Path:
    if notebook_name not in NOTEBOOKS:
        raise ValueError("Solo se exportan los tres cuadernos de la entrega exploratoria.")
    path = ROOT / "notebooks" / notebook_name
    notebook = nbformat.read(path, as_version=4)
    if execute:
        def progress(cell, cell_index, **kwargs):
            if cell.cell_type == "code":
                print(f"Bloque de cómputo completado: {cell_index}", flush=True)
        NotebookClient(
            notebook, timeout=1800, kernel_name="python3", allow_errors=False,
            on_cell_executed=progress,
        ).execute(cwd=str(ROOT))
        nbformat.validate(notebook)
        write_bytes_atomic(path, nbformat.writes(notebook).encode("utf-8"))
    for cell in notebook.cells:
        if cell.cell_type == "code" and (
            cell.execution_count is None
            or any(output.output_type == "error" for output in cell.get("outputs", []))
        ):
            raise ValueError("El EDA debe estar completamente ejecutado; use --execute.")
    exporter = HTMLExporter()
    exporter.exclude_input = True
    html, _ = exporter.from_notebook_node(notebook)
    title = notebook.cells[0].source.splitlines()[0].lstrip("# ")
    html = re.sub(r"<title>.*?</title>", "<title>" + escape(title) + "</title>", html, count=1, flags=re.DOTALL)
    for other in (ROOT / "notebooks").glob("*.ipynb"):
        html = html.replace(f'href="{other.name}"', f'href="{other.stem}.html"')
    descriptions_list = []
    # Las figuras de varios análisis pueden estar intercaladas en el notebook.
    # Usar su leyenda inmediata conserva la correspondencia al reordenar celdas.
    for cell in notebook.cells:
        outputs = cell.get("outputs", [])
        for index, output in enumerate(outputs):
            if "image/png" not in output.get("data", {}):
                continue
            caption = "Figura exploratoria"
            for following in outputs[index + 1:]:
                data = following.get("data", {})
                if "image/png" in data:
                    break
                candidate = data.get("text/markdown", "")
                if isinstance(candidate, list):
                    candidate = "".join(candidate)
                if candidate.lstrip(" *").startswith("Figura"):
                    caption = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", candidate)
                    caption = caption.replace("**", "").replace("`", "").strip()
                    break
            descriptions_list.append(caption)
    descriptions = iter(descriptions_list)

    def alt_text(match):
        tag = match.group(0)
        if "data:image/png" not in tag:
            return tag
        tag = re.sub(r'\s+alt="[^"]*"', "", tag[:-1])
        return tag + ' alt="' + escape(next(descriptions, "Figura exploratoria"), quote=True) + '">'

    html = re.sub(r"<img[^>]*>", alt_text, html)
    style = """<style>
body{background:#f4f6f8!important;color:#203345}
main{max-width:1250px;margin:auto;background:white;padding:30px 45px!important}
h1{color:#14354c!important;border-bottom:4px solid #008579;padding-bottom:18px}
h2{color:#14354c!important;margin-top:45px!important;border-top:1px solid #d9e3e8;padding-top:24px}
table{font-size:12px!important;width:100%;border-collapse:collapse}
th{background:#e9f3f1!important;color:#153d40!important}
td,th{padding:8px!important;text-align:left!important;border-bottom:1px solid #e2e8ed}
.jp-OutputArea-output{overflow-x:auto!important}.jp-RenderedHTMLCommon{line-height:1.65!important}
.jp-InputPrompt,.jp-OutputPrompt{display:none!important}
.eda-nav{background:#edf5f4;border:1px solid #cbe1de;padding:18px;border-radius:8px;margin-bottom:25px}
.eda-nav a{margin-right:16px;display:inline-block}.eda-toc ul{columns:2}.eda-toc li{margin:5px 0}
@media(max-width:750px){main{padding:15px!important}table{font-size:11px!important}.eda-toc ul{columns:1}}
</style>"""
    html = html.replace("</head>", style + "</head>")
    headings = re.findall(r'<h2 id="([^"]+)">(.*?)<a class="anchor-link"', html, flags=re.DOTALL)
    toc = "".join(f'<li><a href="#{escape(anchor)}">{text}</a></li>' for anchor, text in headings)
    navigation = (
        '<div class="eda-nav"><strong>EDA integral · resultados ejecutados</strong><br>'
        f'<a href="../notebooks/{escape(notebook_name)}">Notebook reproducible</a>'
        '<a href="01_Ingesta_Curaduria_y_Calidad.html">01 · Calidad</a>'
        '<a href="02_EDA_Avanzado.html">02 · Análisis</a>'
        '<a href="03_Transformaciones_Exploratorias.html">03 · Transformaciones</a>'
        '<a href="../docs/informe_eda.md">Informe académico</a>'
        '<a href="../docs/diccionario_datos.md">Diccionario</a>'
        '<a href="eda/hallazgos.md">Hallazgos</a>'
        '<a href="eda/tables/28_hallazgos.csv">Síntesis CSV</a>'
        '<a href="eda/manifest.json">Trazabilidad</a>'
        '<details class="eda-toc"><summary>Contenido del estudio</summary><ul>'
        + toc + "</ul></details></div>"
    )
    html = html.replace("<main>", "<main>" + navigation, 1)
    destination = ROOT / "reports" / (Path(notebook_name).stem + ".html")
    write_bytes_atomic(destination, html.encode("utf-8"))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Recalcular antes de exportar")
    parser.add_argument("--all", action="store_true", help="Ejecutar/exportar los tres cuadernos en orden")
    parser.add_argument("--notebook", choices=NOTEBOOKS, default=NOTEBOOKS[1])
    args = parser.parse_args()
    for name in NOTEBOOKS if args.all else [args.notebook]:
        print(export_report(execute=args.execute, notebook_name=name))


if __name__ == "__main__":
    main()
