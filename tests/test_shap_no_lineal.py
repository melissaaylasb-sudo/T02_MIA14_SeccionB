"""Controles de las atribuciones del ensamble explicado fuera de muestra."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.shap_no_lineal import TREE_FAMILIES

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports/modelos_no_lineales/tables"


def test_shap_reconstruye_predicciones_y_background_excluye_evaluacion():
    cases = pd.read_csv(TABLES / "shap_casos.csv")
    details = pd.read_csv(TABLES / "shap_oof.csv")
    metrics = pd.read_csv(TABLES / "metricas_resumen.csv")
    candidates = metrics[(metrics.protocolo == "configuracion") &
                         (metrics.conjunto == "todas") &
                         (metrics.modelo.isin(TREE_FAMILIES))].sort_values(["MAE_N", "modelo"])
    assert len(cases) == cases.Case_n.nunique()
    assert set(cases.modelo) == {candidates.iloc[0].modelo}
    assert set(details.Case_n) == set(cases.Case_n)
    summed = details.groupby("Case_n").SHAP_N.sum()
    aligned = cases.set_index("Case_n").loc[summed.index]
    assert np.allclose(aligned.base_SHAP_N + summed, aligned.predicho_N, atol=1e-4, rtol=1e-6)
    assert np.allclose(aligned.predicho_N - aligned.observado_N, aligned.error_N)
    assert all(str(case) not in str(train).split(";") for case, train in
               zip(cases.Case_n, cases.background_train_cases))
