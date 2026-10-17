"""La auditoría detecta artefactos corruptos sin repetir el entrenamiento."""
import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from src.experimentos_fe import load_dataset
from src.verificar_modelos import auditar_comparacion

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def executed_bundle():
    out = ROOT / "reports/feature_engineering"
    config = json.loads((ROOT / "config/feature_engineering.json").read_text(encoding="utf-8"))
    read = lambda name: pd.read_csv(out / "tables" / f"{name}.csv")
    return {
        "data": load_dataset(ROOT, config), "config": config,
        "partitions": json.loads((out / "particiones.json").read_text(encoding="utf-8")),
        "predictions": read("predicciones_modelos_oof"),
        "folds": read("metricas_modelos_folds"),
        "summary": read("comparacion_modelos"), "family": read("errores_modelos_familia"),
        "audits": json.loads((out / "auditoria_modelos.json").read_text(encoding="utf-8")),
        "reference": read("predicciones_oof"),
        "model_config": json.loads((ROOT / "config/comparacion_modelos.json").read_text(encoding="utf-8")),
    }


def test_executed_models_match_oof_metrics_splits_and_training_statistics(executed_bundle):
    checks = auditar_comparacion(**executed_bundle)
    assert any("Ridge reproduce exactamente" in check for check in checks)
    assert sum("exclusivos de train:" in check for check in checks) == 5


def test_wrong_summary_metric_is_rejected(executed_bundle):
    corrupted = copy.deepcopy(executed_bundle)
    corrupted["summary"].loc[corrupted["summary"].index[-1], "RMSE_N"] += 1.0
    with pytest.raises(AssertionError, match="métrica OOF RMSE_N"):
        auditar_comparacion(**corrupted)


def test_case_assigned_to_another_reserved_fold_is_rejected(executed_bundle):
    corrupted = copy.deepcopy(executed_bundle)
    predictions = corrupted["predictions"]
    index = predictions.index[-1]
    other_fold = 1 if predictions.at[index, "fold"] != 1 else 2
    predictions.at[index, "fold"] = other_fold
    with pytest.raises(AssertionError, match="casos OOF del grupo reservado|todas las particiones compartidas"):
        auditar_comparacion(**corrupted)


def test_last_model_fold_scaler_leak_or_corruption_is_rejected(executed_bundle):
    corrupted = copy.deepcopy(executed_bundle)
    corrupted["audits"][-1]["scaler_mean"][0] += 100.0
    with pytest.raises(AssertionError, match="Media de entrenamiento incorrecta"):
        auditar_comparacion(**corrupted)


def test_ridge_reference_must_reproduce_feature_baseline(executed_bundle):
    corrupted = copy.deepcopy(executed_bundle)
    reference = corrupted["reference"]
    index = reference.index[reference.experimento.eq("A")][-1]
    reference.loc[index, "predicho_N"] += 1.0
    with pytest.raises(AssertionError, match="Ridge reproduce exactamente"):
        auditar_comparacion(**corrupted)


def test_gaussian_process_target_normalization_is_train_only(executed_bundle):
    corrupted = copy.deepcopy(executed_bundle)
    record = next(r for r in corrupted["audits"] if r["modelo"] == "GaussianProcess")
    record["target_train_std"] += 1.0
    with pytest.raises(AssertionError, match="escala del objetivo GPR aprendida en train"):
        auditar_comparacion(**corrupted)


def test_family_error_cannot_disagree_with_reserved_predictions(executed_bundle):
    corrupted = copy.deepcopy(executed_bundle)
    family = corrupted["family"]
    family.loc[family.index[-1], "sesgo_N"] += 1.0
    with pytest.raises(AssertionError, match="sesgo por familia"):
        auditar_comparacion(**corrupted)
