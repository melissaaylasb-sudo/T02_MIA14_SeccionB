"""Ejecuta A/B/C/D con particiones compartidas y guarda evidencia fuera de muestra.

Uso: python -m src.experimentos_fe
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import hashlib
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from threadpoolctl import threadpool_limits

from .eda import save_figure_atomic, write_bytes_atomic
from .feature_engineering import LOG_RATIO, MOMENTS, REDUNDANT, TARGET, feature_names, make_pipeline, nominal_groups
from .ingesta import sha256_file
from .presentacion_fe import ETIQUETAS_VARIABLES, EXPERIMENTOS, UNIDADES_VARIABLES
from .transformaciones_eda import GEOMETRY

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"A": "Línea base (A) · Geometría original", "B": "Variante 1 (B) · Momentos de sección",
          "C": "Variante 2 (C) · Logaritmo de razón de volúmenes",
          "D": "Variante 3 (D) · Reducción de descriptores", "Dummy": "Control de media"}
COLORS = {"A": "#2563a6", "B": "#008579", "C": "#d78930", "D": "#7755a1", "Dummy": "#777777"}


def save_json(path, value):
    write_bytes_atomic(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def save_csv(path, frame):
    write_bytes_atomic(path, frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def source_digest(path):
    """Huellas de código portables entre LF/CRLF sin tocar fuentes heredadas."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def metrics(observed, predicted):
    return {"MAE_N": mean_absolute_error(observed, predicted),
            "RMSE_N": float(np.sqrt(mean_squared_error(observed, predicted))),
            "MedAE_N": median_absolute_error(observed, predicted),
            "R2_OOF": r2_score(observed, predicted)}


def load_dataset(root, config):
    path = root / config["dataset"]
    quality_path = path.with_name("data_quality_report.json")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if sha256_file(path) != quality["output_sha256"]:
        raise ValueError("El dataset no coincide con la curaduría publicada.")
    if config["target"] != TARGET or quality["schema"]["columns"]["target"] != TARGET or config["target_unit"] != "N":
        raise ValueError("Objetivo o unidad incompatible con el EDA.")
    frame = pd.read_csv(path).sort_values("Case_n").reset_index(drop=True)
    if frame.Case_n.duplicated().any() or frame.Case_n.isna().any():
        raise ValueError("Se requiere una fila identificada por caso, sin duplicados de identificador.")
    if frame[TARGET].isna().any() or not np.isfinite(frame[TARGET]).all() or (frame[TARGET] <= 0).any():
        raise ValueError("La tabla supervisada debe contener objetivos positivos observados; no se imputan.")
    # Fiber_base existe en el snapshot histórico. Se incluye explícitamente en todas las variantes.
    make_pipeline().named_steps["contract"].fit(frame[GEOMETRY])
    return frame


def build_splits(frame, config):
    groups = {"configuracion": nominal_groups(frame, config["group_columns"]),
              "familia": frame.n_cells_Y.astype(int).astype(str)}
    splits, records = {}, []
    for protocol, grouping in groups.items():
        splits[protocol] = list(LeaveOneGroupOut().split(frame, groups=grouping))
        for fold, (train, test) in enumerate(splits[protocol], 1):
            assert not set(grouping.iloc[train]) & set(grouping.iloc[test])
            records.append({"protocolo": protocol, "fold": fold,
                            "grupo_excluido": str(grouping.iloc[test[0]]),
                            "train_cases": frame.iloc[train].Case_n.astype(int).tolist(),
                            "test_cases": frame.iloc[test].Case_n.astype(int).tolist(),
                            "n_train": len(train), "n_test": len(test)})
    return splits, groups, records


def evaluate(frame, config, splits, groups):
    x, y = frame[GEOMETRY], frame[TARGET]
    predictions, folds, audits, coefficients = [], [], [], []
    for protocol, partitions in splits.items():
        for experiment in ["Dummy", *config["experiments"]]:
            for fold, (train, test) in enumerate(partitions, 1):
                model = make_pipeline("A" if experiment == "Dummy" else experiment,
                                      alpha=config["model"]["alpha"], dummy=experiment == "Dummy")
                start = perf_counter()
                model.fit(x.iloc[train], y.iloc[train])
                estimate = model.predict(x.iloc[test])
                elapsed = perf_counter() - start
                names = feature_names(model).tolist()
                xt = model[:-1].transform(x.iloc[train])
                singular = np.linalg.svd(xt, compute_uv=False)
                rank = int(np.count_nonzero(singular > singular[0] * 1e-10))
                folds.append({"protocolo": protocol, "experimento": experiment, "fold": fold,
                              "grupo": str(groups[protocol].iloc[test[0]]), "n_train": len(train), "n_test": len(test),
                              "MAE_N": mean_absolute_error(y.iloc[test], estimate),
                              "RMSE_N": float(np.sqrt(mean_squared_error(y.iloc[test], estimate))),
                              "tiempo_s": elapsed, "n_features": len(names), "rango_train": rank})
                audits.append({"protocolo": protocol, "experimento": experiment, "fold": fold,
                               "train_cases": frame.iloc[train].Case_n.astype(int).tolist(),
                               "test_cases": frame.iloc[test].Case_n.astype(int).tolist(),
                               "imputer_columns": GEOMETRY, "imputer_median": model.named_steps["imputer"].statistics_.tolist(),
                               "feature_names": names, "scaler_mean": model.named_steps["scale"].mean_.tolist(),
                               "scaler_scale": model.named_steps["scale"].scale_.tolist(),
                               "scaler_n_samples_seen": int(model.named_steps["scale"].n_samples_seen_)})
                if experiment != "Dummy":
                    for name, coef in zip(names, model.named_steps["model"].coef_):
                        coefficients.append({"protocolo": protocol, "experimento": experiment, "fold": fold,
                                             "feature": name, "coeficiente_estandarizado_N": float(coef)})
                for pos, pred in zip(test, estimate):
                    outside = ((x.iloc[pos] < x.iloc[train].min()) | (x.iloc[pos] > x.iloc[train].max()))
                    residual = float(y.iloc[pos] - pred)
                    predictions.append({"protocolo": protocol, "experimento": experiment, "fold": fold,
                                        "Case_n": int(frame.iloc[pos].Case_n), "familia": int(frame.iloc[pos].n_cells_Y),
                                        "configuracion": str(groups["configuracion"].iloc[pos]),
                                        "observado_N": float(y.iloc[pos]), "predicho_N": float(pred),
                                        "residuo_N": residual, "error_absoluto_N": abs(residual),
                                        "fuera_rango_train": int(outside.sum())})
    return pd.DataFrame(predictions), pd.DataFrame(folds), audits, pd.DataFrame(coefficients)


def summarize(predictions, folds, config):
    rows = []
    for (protocol, experiment), p in predictions.groupby(["protocolo", "experimento"], sort=False):
        f = folds[(folds.protocolo == protocol) & (folds.experimento == experiment)]
        rows.append({"protocolo": protocol, "experimento": experiment, "feature_set": LABELS[experiment],
                     "modelo": "Media de entrenamiento" if experiment == "Dummy" else "Ridge(alpha=1, solver=svd)",
                     **metrics(p.observado_N, p.predicho_N), "MAE_fold_media_N": f.MAE_N.mean(),
                     "MAE_fold_sd_N": f.MAE_N.std(ddof=1), "tiempo_s": f.tiempo_s.sum(),
                     "prediccion_min_N": p.predicho_N.min(), "predicciones_negativas": int((p.predicho_N < 0).sum()),
                     "n_features_min": int(f.n_features.min()), "n_features_max": int(f.n_features.max()),
                     "n_casos_evaluados": len(p), "n_folds": len(f)})
    summary = pd.DataFrame(rows)
    for protocol in summary.protocolo.unique():
        baseline = summary[(summary.protocolo == protocol) & (summary.experimento == "A")].iloc[0]
        ids = summary.protocolo == protocol
        summary.loc[ids, "delta_MAE_vs_A_N"] = summary.loc[ids, "MAE_N"] - baseline.MAE_N
        summary.loc[ids, "cambio_MAE_vs_A_pct"] = 100 * summary.loc[ids, "delta_MAE_vs_A_N"] / baseline.MAE_N
        summary.loc[ids, "delta_RMSE_vs_A_N"] = summary.loc[ids, "RMSE_N"] - baseline.RMSE_N
    contrasts = []
    for protocol in summary.protocolo.unique():
        for exp, spec in config["experiments"].items():
            ref = spec["reference"]
            if ref is None:
                continue
            ff = folds[folds.protocolo == protocol].pivot(index="fold", columns="experimento", values="MAE_N")
            for fold, row in ff.iterrows():
                contrasts.append({"protocolo": protocol, "experimento": exp, "referencia": ref,
                                  "fold": fold, "delta_MAE_N": row[exp] - row[ref]})
    contrasts = pd.DataFrame(contrasts)
    decisions = []
    for exp, spec in config["experiments"].items():
        ref = spec["reference"]
        if ref is None:
            continue
        a = summary[(summary.protocolo == "configuracion") & (summary.experimento == ref)].iloc[0]
        b = summary[(summary.protocolo == "configuracion") & (summary.experimento == exp)].iloc[0]
        ss = summary[summary.protocolo == "familia"].set_index("experimento")
        delta = contrasts[(contrasts.protocolo == "configuracion") & (contrasts.experimento == exp)].delta_MAE_N
        improvement = 100 * (a.MAE_N - b.MAE_N) / a.MAE_N
        wins = (delta < -1e-12).mean()
        threshold = config["decision_rule"]
        adopt = (improvement >= threshold["adopt_min_mae_reduction_percent"] and
                 wins >= threshold["adopt_min_group_win_fraction"] and b.RMSE_N <= a.RMSE_N and
                 ss.loc[exp, "MAE_N"] <= ss.loc[ref, "MAE_N"])
        decision = "Adoptar" if adopt else "Mantener en evaluación" if improvement > 0 else "Descartar"
        decisions.append({"experimento": exp, "referencia": ref, "intervencion": spec["intervention"],
                          "delta_MAE_N": b.MAE_N - a.MAE_N, "reduccion_MAE_pct": improvement,
                          "delta_RMSE_N": b.RMSE_N - a.RMSE_N, "fraccion_grupos_mejora": wins,
                          "delta_MAE_familia_N": ss.loc[exp, "MAE_N"] - ss.loc[ref, "MAE_N"],
                          "decision": decision})
    return summary, contrasts, pd.DataFrame(decisions)


def sensitivity(predictions, config):
    """Diagnóstico de contribución al score OOF; no reentrena ni elimina datos."""
    rows = []
    p = predictions[predictions.protocolo == "configuracion"]
    pivot = p.pivot(index="Case_n", columns="experimento", values="error_absoluto_N")
    groups = p.drop_duplicates("Case_n").set_index("Case_n").configuracion
    for exp, spec in config["experiments"].items():
        ref = spec["reference"]
        if ref is None:
            continue
        delta = pivot[exp] - pivot[ref]
        scenarios = {"todos": set(), "sin_scores_Cook_1_3": {1, 3}, "sin_scores_revision_pivote": {23, 25, 26}}
        scenarios.update({"omitir_score_grupo_" + str(g): set(groups.index[groups == g]) for g in groups.unique()})
        for name, excluded in scenarios.items():
            kept = ~delta.index.isin(excluded)
            rows.append({"experimento": exp, "referencia": ref, "escenario": name,
                         "n_scores": int(kept.sum()), "delta_MAE_N": delta[kept].mean(),
                         "alcance": "Solo cambia el agregado de errores OOF; modelos sin reentrenar"})
    return pd.DataFrame(rows)


def figures(out, summary, predictions, contrasts, folds, audits):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "pantographic-feature-engineering", "savefig.facecolor": "white"})
    paths = []
    def save(fig, name):
        fig.tight_layout()
        for ext in ["png", "svg"]:
            save_figure_atomic(fig, out / "figures" / f"{name}.{ext}", dpi=300)
        plt.close(fig)
        paths.append(name)

    # La presencia se obtiene de las columnas realmente retenidas en cada train.
    # La operación fija de D se distingue del filtro de varianza aprendido.
    feature_order = [*GEOMETRY, *MOMENTS, LOG_RATIO]
    descriptions = {name: f"{ETIQUETAS_VARIABLES[name]} [{UNIDADES_VARIABLES[name]}]"
                    for name in feature_order}
    descriptions[LOG_RATIO] = "Logaritmo de la razón de volúmenes\npivotes/fibras [adimensional]"
    state_labels = ["No incorporada", "Original", "Añadida", "Retirada en variante 3",
                    "Constante filtrada", "Filtrada en algunas particiones"]
    cell_labels = ["—", "Original", "+ Derivada", "Retirada", "Constante", "Según partición"]
    state_colors = ["#f4f5f7", "#c9def0", "#b6dfd1", "#f1ced0", "#d6d7dd", "#f4dfaf"]
    state = np.zeros((len(feature_order), 4), dtype=int)
    for j, exp in enumerate("ABCD"):
        current = [a for a in audits if a["protocolo"] == "configuracion" and a["experimento"] == exp]
        if not current:
            raise ValueError(f"Falta la auditoría de entrenamiento de {exp} para el mapa de características.")
        expected = set(GEOMETRY)
        if exp != "A":
            expected.update(MOMENTS)
        if exp in "CD":
            expected.add(LOG_RATIO)
        if exp == "D":
            expected.difference_update(REDUNDANT)
        for i, name in enumerate(feature_order):
            if exp == "D" and name in REDUNDANT:
                state[i, j] = 3
            elif name in expected:
                retained = [name in a["feature_names"] for a in current]
                state[i, j] = (1 if name in GEOMETRY else 2) if all(retained) else 5 if any(retained) else 4
    fig, ax = plt.subplots(figsize=(12.5, 8.4))
    ax.imshow(state, cmap=ListedColormap(state_colors), vmin=-.5, vmax=5.5, aspect="auto")
    ax.set(xticks=range(4), xticklabels=[EXPERIMENTOS[exp] for exp in "ABCD"],
           yticks=range(len(feature_order)),
           yticklabels=[descriptions[name] for name in feature_order],
           title="Figura 1. Construcción de las representaciones geométricas")
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(feature_order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i, row in enumerate(state):
        for j, value in enumerate(row):
            ax.text(j, i, cell_labels[value], ha="center", va="center", fontsize=9, color="#243144")
    used_states = np.unique(state)
    ax.legend(handles=[Patch(facecolor=state_colors[v], label=state_labels[v]) for v in used_states],
              loc="upper center", bbox_to_anchor=(.5, -.04), ncol=3, frameon=False, fontsize=9)
    ax.text(0, -.18, "Fuente: columnas retenidas al excluir cada configuración. Las constantes se detectan solo en entrenamiento.",
            transform=ax.transAxes, fontsize=9, va="top", color="#435166")
    save(fig, "FE06_representaciones")

    primary_folds = folds[(folds.protocolo == "configuracion") & folds.experimento.isin(list("ABCD"))]
    dimensions = primary_folds.groupby("experimento")[["n_features", "rango_train"]].agg(["min", "max"]).loc[list("ABCD")]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    positions = np.arange(4)
    for offset, column, label, color in [(-.18, "n_features", "Columnas tras filtrar constantes", "#7899b8"),
                                         (.18, "rango_train", "Rango de la matriz de entrenamiento", "#238575")]:
        low = dimensions[(column, "min")].to_numpy()
        high = dimensions[(column, "max")].to_numpy()
        ax.bar(positions + offset, low, width=.32, color=color, label=label)
        ax.errorbar(positions + offset, low, yerr=np.vstack([np.zeros(4), high - low]),
                    fmt="none", ecolor="#243144", capsize=5, linewidth=1.5)
        for pos, minimum, maximum in zip(positions + offset, low, high):
            ax.text(pos, maximum + .18, f"{int(minimum)}–{int(maximum)}", ha="center", fontsize=10)
    upper = int(dimensions.max().max())
    ax.set(xticks=positions, xticklabels=[EXPERIMENTOS[exp] for exp in "ABCD"],
           ylabel="Dimensión / rango", ylim=(0, upper + 3), yticks=np.arange(0, upper + 2, 2),
           title="Figura 2. Dimensión y rango efectivo en entrenamiento")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.text(0, -.14, "Etiquetas: mínimo–máximo entre particiones por configuración; no son intervalos de confianza.\n"
            "Rango tras filtro y estandarización. Un rango completo no garantiza una mejor predicción.",
            transform=ax.transAxes, fontsize=9, va="top", color="#435166")
    save(fig, "FE07_rango")

    s = summary[(summary.protocolo == "configuracion") & (summary.experimento != "Dummy")].set_index("experimento").loc[list("ABCD")]
    p = predictions[(predictions.protocolo == "configuracion") & (predictions.experimento != "Dummy")]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(s.feature_set, s.MAE_N, color=[COLORS[e] for e in s.index])
    for i, v in enumerate(s.MAE_N):
        ax.text(v + .15, i, f"{v:.2f}", va="center")
    ax.set(xlabel="MAE de predicciones fuera de muestra [N]", title="Figura 3. Error por representación geométrica", xlim=(0, s.MAE_N.max() * 1.16))
    ax.invert_yaxis(); save(fig, "FE01_error")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    lim = [min(p.observado_N.min(), p.predicho_N.min()) - 10, max(p.observado_N.max(), p.predicho_N.max()) + 10]
    for ax, exp in zip(axes.flat, "ABCD"):
        z = p[p.experimento == exp]
        ax.scatter(z.observado_N, z.predicho_N, color=COLORS[exp], edgecolors="white")
        ax.plot(lim, lim, "--", color="gray", label="Predicción = observado")
        ax.set(xlabel="Carga de primera falla observada [N]", ylabel="Carga de primera falla predicha [N]",
               title=LABELS[exp].replace(" · ", "\n"))
        ax.legend(fontsize=8)
    fig.suptitle("Figura 4. Observaciones y predicciones fuera de muestra"); save(fig, "FE02_observado_predicho")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    for ax, exp in zip(axes.flat, "ABCD"):
        z = p[p.experimento == exp]
        ax.scatter(z.predicho_N, z.residuo_N, color=COLORS[exp])
        ax.axhline(0, color="gray", ls="--")
        for row in z.loc[z.residuo_N.abs().nlargest(2).index].itertuples():
            ax.annotate(f"Caso {row.Case_n}", (row.predicho_N, row.residuo_N), fontsize=8)
        ax.set(xlabel="Predicción fuera de muestra [N]", ylabel="Observado − predicho [N]",
               title=LABELS[exp].replace(" · ", "\n"))
    fig.suptitle("Figura 5. Residuales de validación"); save(fig, "FE03_residuos")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, protocol, title in zip(axes, ["configuracion", "familia"], ["Configuraciones excluidas", "Familia completa excluida"]):
        z = predictions[(predictions.protocolo == protocol) & (predictions.experimento != "Dummy")]
        for exp in "ABCD":
            v = z[z.experimento == exp].groupby("familia").error_absoluto_N.mean()
            ax.plot(v.index, v, "o-", label=EXPERIMENTOS[exp], color=COLORS[exp])
        ax.set(xlabel="Familia: número de celdas en Y", ylabel="MAE fuera de muestra [N]", title=title, xticks=[4,5,6])
        ax.legend(title="Experimento")
    fig.suptitle("Figura 6. Error por arquitectura y alcance de generalización"); save(fig, "FE04_error_familia")
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for i, exp in enumerate("BCD"):
        z = contrasts[(contrasts.protocolo == "configuracion") & (contrasts.experimento == exp)].delta_MAE_N.to_numpy()
        offset = np.linspace(-.15, .15, len(z))
        ax.scatter(z, i + offset, alpha=.65, color=COLORS[exp], label=f"{EXPERIMENTOS[exp]}: un punto por grupo")
        ax.scatter(z.mean(), i, marker="D", color="black", s=45)
    ax.axvline(0, color="gray", ls="--")
    ax.scatter([], [], marker="D", color="black", s=45, label="Media no ponderada de grupos")
    ax.set(yticks=[0,1,2], yticklabels=["Variante 1 − línea base: momentos", "Variante 2 − variante 1: razón de volúmenes",
                                    "Variante 3 − variante 2: reducción"],
           xlabel="Cambio de MAE por grupo [N]; negativo = mejora", title="Figura 7. Cambio por bloque de variables y dispersión entre grupos")
    ax.legend(fontsize=8); save(fig, "FE05_ablaciones")
    return paths


def execution_log(started, config, summary, folds):
    """Describe las intervenciones y los tiempos medidos, sin añadir evaluaciones."""
    model = config["model"]
    lines = [
        f"Inicio UTC: {started}",
        "Fuente curada y objetivo de carga de primera falla verificados.",
        "DISEÑO COMPARABLE",
        f"Modelo fijo: {model['name']}; alpha={model['alpha']}; solver={model['solver']}; sin búsqueda de hiperparámetros.",
        "Una línea base y tres variantes acumulativas; mismas particiones y mismo modelo para A/B/C/D.",
        "Comparaciones por bloque: B frente a A, C frente a B y D frente a C.",
        "B añade conjuntamente los dos momentos de área; D retira conjuntamente tres descriptores.",
        "Los efectos corresponden a cada bloque; no se atribuyen a una sola variable de esos bloques.",
        "El control de media es una referencia adicional y no una variante de ingeniería de características.",
        "VALIDACIÓN",
        "Principal: validación cruzada dejando fuera una configuración nominal completa en cada partición.",
        "Configuración = número de celdas en Y + altura de fibra + altura de pivote.",
        "Secundaria: validación cruzada dejando fuera una familia completa, definida por el número de celdas en Y.",
        "Se verifica que entrenamiento y evaluación no compartan grupos en ninguna partición.",
        "Cada predicción fuera de muestra procede de un modelo que no entrenó con su grupo excluido.",
        "No se utiliza un conjunto final independiente reservado.",
        "TRANSFORMACIONES",
        "Entradas numéricas: no se aplicó codificación one-hot; los conteos permanecen numéricos.",
        "Imputación de entradas mediante mediana: estadísticas ajustadas solo en entrenamiento.",
        "Construcción algebraica por fila: segundos momentos de área y logaritmo de la razón de volúmenes pivotes/fibras.",
        "Filtro de varianza cero y estandarización: ajustados solo en entrenamiento y reutilizados en evaluación.",
        "El identificador, el objetivo y las respuestas posteriores al ensayo se excluyen de las entradas.",
        "No se imputan objetivos ni se recortan predicciones; las respuestas negativas se conservan para diagnóstico.",
        "Auditoría: particiones.json y auditoria_transformaciones.json guardan casos, columnas y estadísticas por ajuste.",
        "Alcance: el protocolo evita fuga operativa entre entrenamiento y evaluación; el EDA previo con la muestra completa mantiene la evaluación como exploratoria.",
        "INTERVENCIONES EJECUTADAS",
    ]
    for experiment, spec in config["experiments"].items():
        reference = "sin referencia; línea base" if spec["reference"] is None else EXPERIMENTOS[spec["reference"]]
        lines.append(f"{EXPERIMENTOS[experiment]} | referencia: {reference} | intervención: {spec['intervention']}")
    lines.extend([
        "RESULTADOS Y TIEMPO",
        "Métrica principal: MAE [N]. Secundarias: RMSE [N], MedAE [N] y R² sobre predicciones fuera de muestra agregadas.",
        "Tiempo [s] = suma de ajuste del pipeline completo + predicción en cada partición; excluye informes, gráficos y auditoría posterior.",
        "No se estiman costos monetarios. Los tiempos dependen del equipo y de la ejecución.",
    ])
    protocols = {"configuracion": "Configuración excluida", "familia": "Familia excluida"}
    for row in summary.itertuples(index=False):
        lines.append(
            f"{protocols[row.protocolo]} | {EXPERIMENTOS[row.experimento]} | "
            f"MAE={row.MAE_N:.6f} N | RMSE={row.RMSE_N:.6f} N | MedAE={row.MedAE_N:.6f} N | "
            f"R²={row.R2_OOF:.6f} | tiempo={row.tiempo_s:.6f} s | particiones={row.n_folds}"
        )
    lines.append(f"Ajustes terminados: {len(folds)}")
    return "\n".join(lines) + "\n"


def run(root=ROOT):
    root = Path(root)
    config_path = root / "config/feature_engineering.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    out = root / "reports/feature_engineering"
    for folder in [out, out / "tables", out / "figures"]:
        folder.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    frame = load_dataset(root, config)
    splits, groups, records = build_splits(frame, config)
    save_json(out / "configuracion.json", config)
    save_json(out / "particiones.json", records)
    with threadpool_limits(limits=1):
        pred, folds, audits, coef = evaluate(frame, config, splits, groups)
    summary, contrasts, decisions = summarize(pred, folds, config)
    score_sensitivity = sensitivity(pred, config)
    per_family = pred.groupby(["protocolo", "experimento", "familia"]).agg(
        n_casos=("Case_n", "size"), MAE_N=("error_absoluto_N", "mean"),
        sesgo_N=("residuo_N", "mean"), error_maximo_N=("error_absoluto_N", "max")).reset_index()
    for name, data in {"comparacion_experimentos": summary, "predicciones_oof": pred, "metricas_folds": folds,
                       "contrastes_pareados": contrasts, "decisiones": decisions, "sensibilidad_scores": score_sensitivity,
                       "errores_familia": per_family, "coeficientes_folds": coef}.items():
        save_csv(out / "tables" / f"{name}.csv", data)
    save_json(out / "auditoria_transformaciones.json", audits)
    chart_paths = figures(out, summary, pred, contrasts, folds, audits)
    from .informe_fe import write_documents
    write_documents(root, config, summary, decisions, score_sensitivity, per_family)
    inputs = [config_path, root / config["dataset"], root / "data/processed/preliminary/data_quality_report.json"]
    sources = [root / "src" / name for name in ["feature_engineering.py", "experimentos_fe.py", "informe_fe.py", "transformaciones_eda.py", "reporte_fe.py", "verificar_fe.py", "presentacion_fe.py"]]
    evidence = list((root / "reports/eda/tables").glob("*.csv")) + list((root / "reports/eda_transformaciones/tables").glob("*.csv"))
    manifest = {"status": "executed", "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                "eda_base": config["eda_commit"], "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip(),
                "python": platform.python_version(), "packages": {p: version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib"]},
                "inputs_sha256": {p.relative_to(root).as_posix(): sha256_file(p) for p in inputs},
                "source_sha256_normalized_lf": {p.relative_to(root).as_posix(): source_digest(p) for p in sources},
                "eda_evidence_sha256": {p.relative_to(root).as_posix(): sha256_file(p) for p in evidence},
                "n_cases": len(frame), "split_counts": {k: len(v) for k,v in splits.items()}, "figures": chart_paths,
                "artifacts_sha256": {p.relative_to(out).as_posix(): sha256_file(p) for p in out.rglob("*") if p.is_file() and p.name not in {"manifest.json", "ejecucion.log", "verificacion.json", "ejecucion_notebook.log"}}}
    save_json(out / "manifest.json", manifest)
    write_bytes_atomic(out / "ejecucion.log", execution_log(started, config, summary, folds).encode("utf-8"))
    return summary


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(run()[["protocolo", "experimento", "MAE_N", "RMSE_N", "R2_OOF"]].to_string(index=False))
