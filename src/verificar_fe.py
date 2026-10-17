"""Verificación independiente de predicciones, particiones, métricas y entrega FE."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from PIL import Image

from .experimentos_fe import load_dataset, metrics, save_json, summarize, source_digest
from .feature_engineering import GEOMETRY, nominal_groups
from .ingesta import sha256_file
from .verificar_eda import markdown_destinations

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    root = Path(root)
    out = root / "reports/feature_engineering"
    config = json.loads((root / "config/feature_engineering.json").read_text(encoding="utf-8"))
    data = load_dataset(root, config)
    pred = pd.read_csv(out / "tables/predicciones_oof.csv")
    folds = pd.read_csv(out / "tables/metricas_folds.csv")
    summary = pd.read_csv(out / "tables/comparacion_experimentos.csv")
    partitions = json.loads((out / "particiones.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "auditoria_transformaciones.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    checks = []
    def check(condition, label):
        if not bool(condition):
            raise AssertionError(label)
        checks.append(label)
    for field in ["inputs_sha256", "eda_evidence_sha256"]:
        for name, digest in manifest[field].items():
            check(sha256_file(root / name) == digest, "SHA256 " + name)
    for name, digest in manifest["artifacts_sha256"].items():
        check(sha256_file(out / name) == digest, "Artefacto " + name)
    for name, digest in manifest["source_sha256_normalized_lf"].items():
        check(source_digest(root / name) == digest, "Código normalizado LF " + name)
    by_case = data.set_index("Case_n")
    partition_index = {(r["protocolo"],r["fold"]): r for r in partitions}
    check(set(pred.experimento) == {"Dummy", "A", "B", "C", "D"}, "Experimentos previstos")
    for (protocol, experiment), group in pred.groupby(["protocolo", "experimento"]):
        check(not group.Case_n.duplicated().any() and set(group.Case_n) == set(data.Case_n), f"Una OOF por caso {protocol} {experiment}")
        np.testing.assert_allclose(group.observado_N, by_case.loc[group.Case_n, config["target"]])
        np.testing.assert_allclose(group.residuo_N, group.observado_N-group.predicho_N, atol=1e-12)
        np.testing.assert_allclose(group.error_absoluto_N, group.residuo_N.abs(), atol=1e-12)
        row = summary[(summary.protocolo == protocol) & (summary.experimento == experiment)].iloc[0]
        for key, value in metrics(group.observado_N, group.predicho_N).items():
            check(np.isclose(row[key], value, rtol=1e-10, atol=1e-10), f"Métrica {protocol} {experiment} {key}")
        for fold, gg in group.groupby("fold"):
            record = partition_index[(protocol, fold)]
            check(set(gg.Case_n) == set(record["test_cases"]), "Misma partición entre variantes")
            check(set(record["test_cases"]).isdisjoint(record["train_cases"]), "Train/test disjuntos")
            grouping = nominal_groups(by_case, config["group_columns"]) if protocol == "configuracion" else by_case.n_cells_Y
            check(set(grouping.loc[record["train_cases"]]).isdisjoint(grouping.loc[record["test_cases"]]), "Grupo no dividido")
    for record in audit:
        split = partition_index[(record["protocolo"],record["fold"])]
        check(record["train_cases"] == split["train_cases"] and record["test_cases"] == split["test_cases"], "Auditoría y partición coinciden")
        np.testing.assert_allclose(record["imputer_median"], by_case.loc[record["train_cases"],GEOMETRY].median().to_numpy())
        check(record["scaler_n_samples_seen"] == len(record["train_cases"]), "Escalador solo ve train")
    _, _, decisions = summarize(pred, folds, config)
    actual = pd.read_csv(out / "tables/decisiones.csv")
    pd.testing.assert_frame_equal(actual, decisions, check_exact=False, rtol=1e-10, atol=1e-10)
    nb_path = root / "notebooks/04_Feature_Engineering_Pantograficas.ipynb"
    nb = nbformat.read(nb_path,4); nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    check([c.execution_count for c in code] == list(range(1,len(code)+1)), "Notebook ejecutado en orden desde kernel limpio")
    check(not any(o.output_type == "error" for c in code for o in c.get("outputs",[])), "Notebook sin errores")
    check(sum("image/png" in o.get("data",{}) for c in code for o in c.get("outputs",[])) == 5, "Cinco figuras embebidas")
    for path in (out / "figures").glob("*.png"):
        with Image.open(path) as img:
            check(min(img.info.get("dpi",(0,0))) >= 299, "Figura de alta resolución " + path.name)
    docs = [root / "README.md", *[root / "docs" / n for n in ["feature_engineering.md","experimentos_fe.md","diccionario_features.md"]]]
    for doc in docs:
        content = doc.read_text(encoding="utf-8")
        for link in markdown_destinations(content):
            if not link.startswith(("http:","https:","#","mailto:")):
                local = link.split("#",1)[0]
                check((doc.parent / local).exists(), "Enlace local " + local)
    for doc in [root / "README.md",root / "docs/experimentos_fe.md"]:
        content = doc.read_text(encoding="utf-8")
        for row in summary[(summary.protocolo == "configuracion") & (summary.experimento != "Dummy")].itertuples():
            check(f"{row.MAE_N:.4f}" in content and f"{row.RMSE_N:.4f}" in content, "Cifras coinciden " + doc.name + " " + row.experimento)
    # Se conservan módulos, datos y resultados del EDA; no se ejecutan ni reescriben.
    protected = ["data", "config/data_schema.yml", "reports/eda", "reports/eda_calidad", "reports/eda_relaciones", "reports/eda_transformaciones",
                 "src/eda.py", "src/eda_calidad.py", "src/eda_relaciones.py", "src/eda_energia.py", "src/transformaciones_eda.py",
                 "notebooks/01_Ingesta_Curaduria_y_Calidad.ipynb", "notebooks/02_EDA.ipynb", "notebooks/03_Transformaciones_Exploratorias.ipynb"]
    changed = subprocess.check_output(["git", "diff", "--name-only", config["eda_commit"], "--", *protected],cwd=root,text=True)
    check(not changed.strip(), "EDA y datos originales conservados")
    save_json(out / "verificacion.json", {"status":"passed", "checks":checks, "code_cells_executed":len(code),
                                         "notebook_sha256":sha256_file(nb_path)})
    print("Verificacion FE: passed; metricas, particiones, notebook, enlaces y preservacion EDA conformes.")


if __name__ == "__main__":
    verify()
