"""Recuperar los resultados históricos y el contrato de variables, sin entrenar."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FE = "reports/feature_engineering/tables/"
FEATURES = {
    "n_cells_Y": ("conteo", "Número de celdas en la dirección Y; indicador de la arquitectura."),
    "Fiber_height": ("mm", "Altura de la sección de las fibras."),
    "Fiber_base": ("mm", "Ancho de sección; algunas fuentes geométricas discrepan."),
    "Fiber_total_length": ("mm", "Longitud declarada del conjunto de fibras, vinculada al nivel Y."),
    "Fiber_total_volume": ("mm³", "Volumen declarado de fibras; relacionado con el de pivotes."),
    "Pivot_height": ("mm", "Altura nominal de los elementos de unión."),
    "Pivot_radius": ("mm", "Radio nominal del pivote; constante en la campaña."),
    "Pivot_total_number": ("conteo", "Cantidad de conexiones; depende de la arquitectura observada."),
    "Pivot_total_volume": ("mm³", "Volumen declarado del conjunto de uniones."),
    "Sample_total_volume": ("mm³", "Volumen declarado del espécimen según CAD; no se sustituye por una suma."),
}

def build(root: Path = ROOT) -> Path:
    """Compilar una tabla trazable de antecedentes; conservar protocolo y representación."""
    out = root / "docs/benchmark"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    sources = {
        "04 · variantes": FE + "comparacion_experimentos.csv",
        "04 · familias fijas": FE + "comparacion_modelos.csv",
        "05 · ajuste anidado": "reports/modelos_ajustados/metricas_resumen.csv",
        "06 · importancia anterior": "reports/feature_importance/tables/metricas_resumen.csv",
        "06 · permutación": "reports/permutation_importance/tables/metricas_resumen.csv",
    }
    for stage, source in sources.items():
        for row in pd.read_csv(root/source, float_precision="round_trip").to_dict("records"):
            model = row["modelo"]
            representation = row.get("representacion", row.get("feature_set", "A"))
            setting = row.get("experimento", row.get("conjunto", model))
            if "conjunto" in row:
                representation = {"todas":"A", "prioritarias":"4 elegidas en cada entrenamiento",
                                  "referencia":"Según referencia"}[row["conjunto"]]
            adjusted = (stage.startswith("05") and model != "Baseline_Ridge") or bool(row.get("tuning", False))
            rows.append({"etapa":stage, "procedimiento":setting, "modelo":model,
                         "protocolo":row["protocolo"], "entradas":representation,
                         "ajuste_hiperparametros":adjusted,
                         "MAE_N":row["MAE_N"], "RMSE_N":row["RMSE_N"],
                         "R2_OOF":row.get("R2_OOF",row.get("R2")), "fuente":source})
    pd.DataFrame(rows).to_csv(out/"antecedentes_metricas.csv", index=False)
    variables = [{"Feature":k, "Original/Derivada":"Original", "Fórmula":"Valor documentado, sin derivación",
                  "Unidad":u, "Justificación física":why,
                  "Riesgo de leakage":"Disponible antes del ensayo; transformaciones aprendidas solo en entrenamiento."}
                 for k,(u,why) in FEATURES.items()]
    raw = pd.read_csv(root/"data/processed/preliminary/model_table.csv")
    if missing := set(FEATURES).difference(raw.columns):
        raise ValueError(f"Faltan variables geométricas: {sorted(missing)}")
    pd.DataFrame(variables).to_csv(out/"features.csv", index=False)
    return out

if __name__ == "__main__":
    print(build().relative_to(ROOT))
