"""SHAP fuera de muestra para un ensamble de árboles de la comparación no lineal.

El modelo se elige descriptivamente por MAE entre los ensambles evaluados. Cada
observación se explica con el estimador que la excluyó y con un background del
entrenamiento de ese estimador. La selección posterior del modelo no constituye
una estimación independiente del desempeño del ganador.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ajuste_modelos import ROOT, load_frozen_data
from .importancia_permutacion import LABELS, TARGET
from .transformaciones_eda import GEOMETRY

TREE_FAMILIES = ("RandomForest", "ExtraTrees", "GradientBoosting", "XGBoost")
OUT = Path("reports/modelos_no_lineales")


def _load(out: Path, table: str) -> pd.DataFrame:
    path = out / "tables" / f"{table}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def run(root: Path | str = ROOT, output: Path | str | None = None, *, force: bool = False) -> dict:
    """Calcula SHAP OOF de un ensamble elegido por menor MAE de configuración."""
    import shap

    root = Path(root).resolve()
    out = Path(output).resolve() if output is not None else root / OUT
    cache = out / "logs" / "shap_no_lineal.json"
    expected = [out / "tables" / f"{name}.csv" for name in
                ("shap_oof", "shap_casos", "shap_global", "shap_estabilidad")]
    expected += [out / "figures" / f"{name}.png" for name in
                 ("20_shap_global_arbol", "21_shap_local_error_alto", "22_shap_estabilidad_arbol")]
    if not force and cache.is_file() and all(path.is_file() for path in expected):
        result = json.loads(cache.read_text(encoding="utf-8"))
        cases = pd.read_csv(out / "tables" / "shap_casos.csv")
        detail = pd.read_csv(out / "tables" / "shap_oof.csv")
        summed = detail.groupby("Case_n").SHAP_N.sum()
        aligned = cases.set_index("Case_n").loc[summed.index]
        if (len(cases) != cases.Case_n.nunique() or
                not np.allclose(aligned.base_SHAP_N + summed, aligned.predicho_N,
                                atol=1e-4, rtol=1e-6)):
            raise ValueError("La caché SHAP no reconstruye las predicciones.")
        return result
    metrics = _load(out, "metricas_resumen")
    predictions = _load(out, "predicciones_oof")
    folds = _load(out, "particiones_externas")
    frame = load_frozen_data(root).set_index("Case_n")
    X = frame[GEOMETRY]
    candidates = metrics[(metrics.protocolo == "configuracion") &
                         (metrics.conjunto == "todas") &
                         (metrics.modelo.isin(TREE_FAMILIES))].sort_values(["MAE_N", "modelo"])
    if candidates.empty:
        raise ValueError("No hay ensambles de árboles con evaluación externa completa.")
    family = str(candidates.iloc[0].modelo)
    predictions = predictions[(predictions.protocolo == "configuracion") &
                              (predictions.modelo == family) &
                              (predictions.conjunto == "todas")]
    if predictions.Case_n.duplicated().any() or set(predictions.Case_n) != set(X.index):
        raise ValueError("Las predicciones externas SHAP no cubren cada caso una vez.")

    rows: list[dict] = []
    cases: list[dict] = []
    max_error = 0.0
    for fold in sorted(predictions.fold.unique()):
        current = predictions[predictions.fold == fold].sort_values("Case_n")
        outer = folds[(folds.protocolo == "configuracion") & (folds.fold == fold)]
        if len(outer) != 1:
            raise ValueError("Falta la partición externa del explicador.")
        train_ids = [int(x) for x in outer.iloc[0].casos_train.split(";")]
        test_ids = current.Case_n.astype(int).tolist()
        if set(train_ids) & set(test_ids):
            raise ValueError("Solapamiento entre background y observaciones explicadas.")
        paths = current.estimador.unique()
        if len(paths) != 1:
            raise ValueError("Más de un estimador para un fold externo.")
        model = joblib.load(out / paths[0])
        names = np.asarray(model.named_steps["seleccion"].columns)[model.named_steps["varianza"].get_support()].tolist()
        background = np.asarray(model[:-1].transform(X.loc[train_ids]), dtype=float)
        explained = np.asarray(model[:-1].transform(X.loc[test_ids]), dtype=float)
        if background.shape[1] != len(names) or explained.shape[1] != len(names):
            raise ValueError("Orden o cantidad de nombres SHAP incorrectos.")
        explainer = shap.TreeExplainer(model.named_steps["modelo"], data=background,
                                      feature_perturbation="interventional", model_output="raw")
        exp = explainer(explained, check_additivity=False)
        values = np.asarray(exp.values, dtype=float)
        base = np.broadcast_to(np.asarray(exp.base_values, dtype=float), (len(test_ids),))
        predicted = current.predicho_N.to_numpy(dtype=float)
        reconstruction = base + values.sum(axis=1)
        local_error = float(np.max(np.abs(reconstruction - predicted)))
        max_error = max(max_error, local_error)
        if not np.allclose(reconstruction, predicted, atol=1e-4, rtol=1e-6):
            raise ValueError(f"SHAP no reconstruye las predicciones del fold {fold}.")
        for i, case in enumerate(test_ids):
            cases.append({"Case_n": case, "fold": int(fold), "modelo": family,
                          "observado_N": float(current.iloc[i].observado_N),
                          "predicho_N": float(predicted[i]), "base_SHAP_N": float(base[i]),
                          "error_N": float(predicted[i] - current.iloc[i].observado_N),
                          "background_train_cases": ";".join(map(str, train_ids)),
                          "error_reconstruccion_N": float(abs(reconstruction[i] - predicted[i]))})
            for j, name in enumerate(names):
                rows.append({"Case_n": case, "fold": int(fold), "modelo": family,
                             "variable": name, "nombre": LABELS[name],
                             "valor_fisico": float(X.loc[case, name]),
                             "valor_escalado": float(explained[i, j]),
                             "SHAP_N": float(values[i, j])})
    details = pd.DataFrame(rows)
    local = pd.DataFrame(cases)
    global_table = details.assign(abs_SHAP_N=details.SHAP_N.abs()).groupby(
        ["modelo", "variable", "nombre"], as_index=False).agg(
            SHAP_abs_medio_N=("abs_SHAP_N", "mean"),
            SHAP_firmado_medio_N=("SHAP_N", "mean"),
            n_casos=("Case_n", "nunique"))
    global_table = global_table.sort_values("SHAP_abs_medio_N", ascending=False)
    by_fold = details.assign(abs_SHAP_N=details.SHAP_N.abs()).groupby(
        ["fold", "variable", "nombre"], as_index=False).agg(
            SHAP_abs_medio_N=("abs_SHAP_N", "mean"))
    by_fold["rango"] = by_fold.groupby("fold").SHAP_abs_medio_N.rank(
        method="min", ascending=False).astype(int)
    stability = by_fold.groupby(["variable", "nombre"], as_index=False).agg(
        presencia_top3=("rango", lambda s: float((s <= 3).mean())),
        rango_mediano=("rango", "median"),
        SHAP_abs_DE_entre_folds_N=("SHAP_abs_medio_N", "std"))
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for name, data in [("shap_oof", details), ("shap_casos", local),
                       ("shap_global", global_table), ("shap_estabilidad", stability)]:
        data.to_csv(tables / f"{name}.csv", index=False, encoding="utf-8-sig")

    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    top = global_table.head(9).iloc[::-1]
    bars = ax.barh(top.nombre, top.SHAP_abs_medio_N, color="#7c4bb1")
    ax.bar_label(bars, fmt="%.2f", padding=4)
    ax.set_xlabel("Contribución absoluta media SHAP [N]")
    ax.set_title(f"Atribución global fuera de muestra · {family}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "20_shap_global_arbol.png", dpi=220)
    fig.savefig(figures / "20_shap_global_arbol.svg")
    plt.close(fig)

    case = int(local.assign(abs_error=local.error_N.abs()).sort_values(
        ["abs_error", "Case_n"], ascending=[False, True]).iloc[0].Case_n)
    specific = details[(details.Case_n == case) & (details.SHAP_N.abs() > .01)].sort_values("SHAP_N")
    fig, ax = plt.subplots(figsize=(10, 5.3))
    colors = np.where(specific.SHAP_N.to_numpy() >= 0, "#247c63", "#b45b71")
    bars = ax.barh(specific.nombre, specific.SHAP_N, color=colors)
    ax.bar_label(bars, fmt="%+.2f", padding=3)
    ax.axvline(0, color="#333333", lw=.8)
    left = min(-1.0, float(specific.SHAP_N.min()) * 1.24)
    right = max(1.0, float(specific.SHAP_N.max()) * 1.24)
    ax.set_xlim(left, right)
    ax.set_xlabel("Contribución SHAP a la predicción [N]")
    ax.set_title(f"Atribuciones locales · caso {case} · {family}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "21_shap_local_error_alto.png", dpi=220)
    fig.savefig(figures / "21_shap_local_error_alto.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    stable = stability.sort_values("presencia_top3").tail(9)
    bars = ax.barh(stable.nombre, stable.presencia_top3 * 100, color="#599a85")
    ax.bar_label(bars, fmt="%.0f %%", padding=3)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Presencia entre las tres primeras posiciones por partición [%]")
    ax.set_title(f"Estabilidad del rango SHAP · {family}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figures / "22_shap_estabilidad_arbol.png", dpi=220)
    fig.savefig(figures / "22_shap_estabilidad_arbol.svg")
    plt.close(fig)

    result = {"modelo_explicado": family, "criterio_descriptivo": "menor MAE entre ensambles de árboles completos al reservar configuraciones",
              "explainer": "TreeExplainer", "feature_perturbation": "interventional",
              "background": "solo entrenamiento externo correspondiente",
              "n_explicaciones": len(local), "max_error_reconstruccion_N": max_error,
              "caso_error_alto": case,
              "advertencia": "Selección descriptiva sobre la misma evaluación; SHAP no identifica causalidad."}
    (out / "logs").mkdir(exist_ok=True)
    (out / "logs" / "shap_no_lineal.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
