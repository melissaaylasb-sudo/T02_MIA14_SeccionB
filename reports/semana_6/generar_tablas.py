"""Exporta el experimento A/B/C y los modelos de referencia; sin entrenar."""
from pathlib import Path
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "tablas"
SOURCE = ROOT / "reports/feature_engineering/tables"
OUT.mkdir(exist_ok=True)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(name):
    with (SOURCE / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def num(value, digits=3):
    return f"{float(value):.{digits}f}".replace(".", "{,}")

outputs = []
def write(name, rows):
    path = OUT / name
    path.write_text("\n".join(" & ".join(row) + r" \\" for row in rows) + "\n", encoding="utf-8")
    outputs.append(path)

summary = read("comparacion_experimentos.csv")
labels = {"A": "Línea base (A)", "B": "Variante 1 (B)", "C": "Variante 2 (C)"}
for protocol, filename in [("configuracion", "principal.tex"), ("familia", "familias.tex")]:
    selected = {r["experimento"]: r for r in summary if r["protocolo"] == protocol}
    metrics = ["MAE_N", "RMSE_N", "MedAE_N", "R2_OOF"]
    if protocol == "configuracion":
        metrics.append("tiempo_s")
    write(filename, [[labels[k]] + [num(selected[k][m]) for m in metrics] for k in "ABC"])

models = {r["modelo"]: r for r in read("comparacion_modelos.csv") if r["protocolo"] == "configuracion"}
order = ["Ridge", "ElasticNet", "SVR_RBF", "RandomForest", "GaussianProcess"]
assert set(models) == set(order)
names = ["Ridge", "Elastic Net", "SVR (RBF)", "Bosque aleatorio", "Proceso gaussiano"]
metrics = ["MAE_N", "RMSE_N", "MedAE_N", "R2_OOF", "tiempo_s"]
write("modelos.tex", [[name] + [num(models[k][m]) for m in metrics] for name, k in zip(names, order)])

families = {(r["familia"], r["experimento"]): r for r in read("errores_familia.csv") if r["protocolo"] == "configuracion"}
write("error_familia.tex", [
    [f"Y = {family}"] + [num(families[(str(family), k)]["MAE_N"]) for k in "ABC"]
    + [num(families[(str(family), "B")]["sesgo_N"])] for family in [4, 5, 6]
])

sources = ["comparacion_experimentos.csv", "comparacion_modelos.csv", "errores_familia.csv"]
manifest = {
    "purpose": "Tablas del sprint centradas en A, B y C; resultados históricos sin reentrenamiento.",
    "reference_commit": "8fdb7961d7eaaaebd9651d0b9a044168893d448a",
    "generator": "reports/semana_6/generar_tablas.py",
    "generator_sha256": digest(Path(__file__)),
    "representations": list("ABC"),
    "source_sha256": {(SOURCE / name).relative_to(ROOT).as_posix(): digest(SOURCE / name) for name in sources},
    "output_sha256": {path.relative_to(HERE).as_posix(): digest(path) for path in outputs}
}
(HERE / "trazabilidad_tablas.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Exportadas cuatro tablas: variantes, familias, errores locales y modelos.")
