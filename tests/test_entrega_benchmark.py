"""Verificación de la entrega visible, independiente del constructor del cuaderno."""
import base64
import hashlib
from pathlib import Path
import re
import nbformat

ROOT=Path(__file__).resolve().parents[1]


def test_imagenes_embebidas_son_las_diez_figuras_publicadas():
    nb=nbformat.read(ROOT/"notebooks/Comparativa_6_Modelos_Regresion_Pantograficas.ipynb",as_version=4)
    nbformat.validate(nb)
    code=[c for c in nb.cells if c.cell_type=="code"]
    assert [c.execution_count for c in code]==list(range(1,len(code)+1))
    outputs=[o for c in code for o in c.outputs]
    assert all(o.output_type!="error" for o in outputs)
    pngs=[hashlib.sha256(base64.b64decode(o.data["image/png"])).hexdigest() for o in outputs if "image/png" in o.get("data",{})]
    stored={hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"reports/benchmark_regresion/figures").glob("*.png")}
    assert len(pngs)==len(stored)==10 and set(pngs)==stored
    sections=[int(m[1]) for c in nb.cells if c.cell_type=="markdown" and (m:=re.match(r"^#{1,3} (\d+)\. ",c.source))]
    assert sections==list(range(1,20))
    boxes=[o.data["text/html"] for o in outputs if "Resultado verificado" in o.get("data",{}).get("text/html","")]
    assert len(boxes)>=10
    elements=["Resultado verificado","Interpretación estadística","Interpretación mecánica plausible","Limitación","Decisión",
              "#E8F5E9","#2E7D32","#1B4332"]
    assert all(all(item in box for item in elements) for box in boxes)
