"""Ejecuta el notebook FE en kernel limpio y conserva sus resultados embebidos."""
import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from .eda import write_bytes_atomic

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/04_Feature_Engineering_Pantograficas.ipynb"


def execute_notebook():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    log = []
    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            message = f"Celda de código completada: {cell_index}"
            log.append(message)
            print(message, flush=True)
    NotebookClient(notebook, timeout=600, kernel_name="python3", allow_errors=False,
                   on_cell_executed=progress).execute(cwd=str(ROOT))
    nbformat.validate(notebook)
    write_bytes_atomic(NOTEBOOK, nbformat.writes(notebook).encode("utf-8"))
    write_bytes_atomic(ROOT / "reports/feature_engineering/ejecucion_notebook.log", ("\n".join(log)+"\nKernel limpio; ejecución completada sin errores.\n").encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    execute_notebook()
