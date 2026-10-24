"""Figuras del sprint: reutiliza resultados ejecutados, sin ajustar modelos.

Desde cualquier directorio: python /ruta/reports/semana_6/generar_figuras.py
La salida predeterminada es figures/ junto a este script. --output cambia la ruta.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FAMILY_COLORS = {4: "#28547A", 5: "#A86527", 6: "#367663"}
BLUE, GREEN, RED = "#385C78", "#367663", "#A25B50"
SOURCE_PATHS = {
    "datos": "data/processed/preliminary/model_table.csv",
    "metricas": "reports/feature_engineering/tables/comparacion_experimentos.csv",
    "predicciones": "reports/feature_engineering/tables/predicciones_oof.csv",
    "pareadas": "reports/feature_engineering/tables/contrastes_pareados.csv",
    "folds": "reports/feature_engineering/tables/metricas_folds.csv",
    "correlaciones_globales": "reports/eda/tables/08_asociaciones_globales.csv",
    "correlaciones_condicionadas": "reports/eda/tables/09_asociaciones_condicionadas.csv",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decimal_text(value: float, digits: int) -> str:
    """Formato español para anotaciones, sin modificar los valores de origen."""
    return f"{value:.{digits}f}".replace(".", ",")


def decimal_tick(value: float, position: int | None = None) -> str:
    return f"{value:g}".replace(".", ",").replace("-", "−")


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.titlesize": 11, "axes.labelsize": 10.5,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.7,
        "axes.edgecolor": "#555555", "text.color": "#222222",
        "axes.labelcolor": "#222222", "xtick.color": "#333333",
        "ytick.color": "#333333", "savefig.facecolor": "white",
    })


def save(fig, output: Path, name: str, sources: list[str], description: str,
         interpretation: str) -> dict:
    path = output / f"{name}.png"
    # Apply only to automatically formatted numerical axes; preserve category labels.
    for axes in fig.axes:
        for axis in [axes.xaxis, axes.yaxis]:
            if isinstance(axis.get_major_formatter(), ScalarFormatter):
                axis.set_major_formatter(FuncFormatter(decimal_tick))
    # Fixed canvas: dimensions remain reproducible and suitable for a 16 cm text width.
    fig.savefig(path, dpi=300, facecolor="white")
    width, height = fig.get_size_inches()
    plt.close(fig)
    return {
        "file": f"figures/{path.name}", "sha256": digest(path),
        "width_inches": float(width), "height_inches": float(height), "dpi": 300,
        "sources": [SOURCE_PATHS[s] for s in sources],
        "description": description, "interpretation": interpretation,
        "new_training": False,
    }


def plot_mae(data: dict, output: Path) -> dict:
    frame = data["metricas"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35), layout="constrained")
    for ax, protocol, title, ymax in zip(
        axes, ["configuracion", "familia"],
        ["a) Configuración excluida", "b) Familia excluida"], [21, 83]
    ):
        selected = frame[frame.protocolo.eq(protocol)].set_index("experimento").loc[list("ABCD")]
        vals = selected.MAE_N.to_numpy()
        ax.bar(list("ABCD"), vals, color=[BLUE, GREEN, BLUE, BLUE], width=0.6)
        for pos, value in enumerate(vals):
            ax.text(pos, value + ymax * .025, decimal_text(value, 2), ha="center", va="bottom")
        ax.set(title=title, xlabel="Conjunto de características", ylabel="MAE [N]", ylim=(0, ymax))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E4E4E4", linewidth=.6)
    return save(fig, output, "01_mae", ["metricas"],
                "MAE OOF agregado por caso de las representaciones A–D, con Ridge fijo. "
                "Los paneles muestran protocolos distintos y escalas verticales distintas.",
                "B reduce el MAE de configuración de 13.246554 a 12.834625 N; "
                "el MAE de familia permanece cercano a 58 N. Sin barras inferenciales.")


def plot_diagnostic(data: dict, output: Path) -> dict:
    pred = data["predicciones"]
    subset = pred[pred.protocolo.eq("configuracion") & pred.experimento.eq("B")].copy()
    assert subset.Case_n.is_unique
    np.testing.assert_allclose(subset.residuo_N, subset.observado_N - subset.predicho_N)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.9), layout="constrained")
    ax, rx = axes
    for family, color in FAMILY_COLORS.items():
        p = subset[subset.familia.eq(family)]
        ax.scatter(p.observado_N, p.predicho_N, s=32, color=color, edgecolors="white", linewidth=.4,
                   label=f"Familia {family}", zorder=3)
        rx.scatter(p.predicho_N, p.residuo_N, s=32, color=color, edgecolors="white", linewidth=.4,
                   zorder=3)
    ax.plot([35, 235], [35, 235], color="#777777", linestyle="--", linewidth=1, zorder=1)
    ax.set(title="a) Observado y predicho", xlabel="Carga observada [N]",
           ylabel="Carga predicha [N]", xlim=(35, 235), ylim=(35, 235))
    ax.set_xticks([50, 100, 150, 200]); ax.set_yticks([50, 100, 150, 200])
    rx.axhline(0, color="#777777", linestyle="--", linewidth=1, zorder=1)
    rx.set(title="b) Residuales", xlabel="Carga predicha [N]",
           ylabel="Observada − predicha [N]", xlim=(35, 235), ylim=(-43, 43))
    rx.set_xticks([50, 100, 150, 200]); rx.set_yticks([-40, -20, 0, 20, 40])
    for _, row in subset[subset.Case_n.isin([1, 17, 22])].iterrows():
        label = f"{int(row.Case_n)}"
        offset = (5, -11) if int(row.Case_n) == 17 else (5, 5)
        ax.annotate(label, (row.observado_N, row.predicho_N), xytext=offset,
                    textcoords="offset points", fontsize=9)
        rx.annotate(label, (row.predicho_N, row.residuo_N), xytext=offset,
                    textcoords="offset points", fontsize=9)
    for panel in axes:
        panel.set_axisbelow(True); panel.grid(color="#E7E7E7", linewidth=.6)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    return save(fig, output, "02_diagnostico", ["predicciones"],
                "Predicciones OOF de B al excluir configuraciones. Identidad en panel a; "
                "residuo observado menos predicho y referencia cero en b. Etiquetas: casos 1, 17 y 22.",
                "Las familias ocupan bandas distintas; no se trata de predicciones de entrenamiento. "
                "Residuo positivo equivale a subestimación.")


def plot_paired(data: dict, output: Path) -> dict:
    d = data["pareadas"]
    d = d[d.protocolo.eq("configuracion") & d.experimento.eq("B") & d.referencia.eq("A")].copy()
    keys = data["folds"]
    keys = keys[keys.protocolo.eq("configuracion") & keys.experimento.eq("A")][["fold", "grupo"]]
    d = d.merge(keys, on="fold", validate="one_to_one").sort_values("fold")
    assert len(d) == 24
    fig, ax = plt.subplots(figsize=(7.1, 4.55), layout="constrained")
    positions = np.arange(len(d))
    ax.barh(positions, d.delta_MAE_N, color=np.where(d.delta_MAE_N < 0, GREEN, RED), height=.72)
    ax.axvline(0, color="#333333", linewidth=.85)
    group_labels = d.grupo.str.replace("|", " / ", regex=False).str.replace(".", ",", regex=False)
    ax.set_yticks(positions, group_labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set(xlabel="ΔMAE del grupo: B − A [N]", ylabel="Configuración: Y / altura fibra / altura pivote",
           xlim=(-6.7, 7.3), title="Diferencias pareadas por configuración")
    ax.set_xticks([-6, -4, -2, 0, 2, 4, 6])
    ax.set_axisbelow(True); ax.grid(axis="x", color="#E7E7E7", linewidth=.6)
    ax.text(.015, .985, "Menor error con B", color=GREEN, transform=ax.transAxes, va="top", fontsize=9.5)
    ax.text(.985, .985, "Mayor error con B", color=RED, transform=ax.transAxes, va="top", ha="right", fontsize=9.5)
    return save(fig, output, "03_pareadas", ["pareadas", "folds"],
                "Diferencia de MAE por grupo, variante B menos A. Alturas en mm, Y como conteo. "
                "Cada barra corresponde al mismo grupo reservado para ambas representaciones.",
                "B mejora en 15 de 24 grupos. Los grupos no son experimentos independientes: "
                "sus conjuntos de entrenamiento se superponen.")


def plot_eda(data: dict, output: Path) -> dict:
    frame = data["datos"]
    target = "First_Failure_Load_N"
    global_corr = data["correlaciones_globales"].set_index("variable")
    conditional = data["correlaciones_condicionadas"].set_index("variable")
    rho = [global_corr.loc["Pivot_total_volume", "Spearman"],
           conditional.loc["Pivot_total_volume", "rho_condicionado"]]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.65), layout="constrained",
                             gridspec_kw={"width_ratios": [1.1, 1]})
    ax, cx = axes
    for position, family in enumerate([4, 5, 6], start=1):
        group = frame[frame.n_cells_Y.eq(family)].sort_values("Case_n")
        values = group[target].to_numpy()
        color = FAMILY_COLORS[family]
        bp = ax.boxplot([values], positions=[position], widths=.48, showfliers=False,
                        patch_artist=True, medianprops={"color": "#222222", "linewidth": 1.3},
                        boxprops={"linewidth": .8}, whiskerprops={"linewidth": .8},
                        capprops={"linewidth": .8})
        bp["boxes"][0].set(facecolor=color, alpha=.2)
        # Fixed offsets avoid a new random state and preserve the actual measured y-values.
        ax.scatter(position + np.linspace(-.14, .14, len(values)), values, s=24,
                   color=color, edgecolors="white", linewidth=.3, zorder=3)
    ax.set(title="a) Primera falla por familia", xlabel="Celdas en Y (familia)",
           ylabel="Carga de primera falla [N]", ylim=(35, 230))
    ax.set_xticks([1, 2, 3], ["4", "5", "6"])
    ax.set_axisbelow(True); ax.grid(axis="y", color="#E7E7E7", linewidth=.6)
    cx.bar([0, 1], rho, color=[BLUE, RED], width=.55)
    cx.axhline(0, color="#777777", linewidth=.8)
    for i, value in enumerate(rho):
        cx.text(i, value + (.04 if value > 0 else -.04), decimal_text(value, 3),
                ha="center", va="bottom" if value > 0 else "top")
    cx.set_xticks([0, 1], ["Global", "Condicionada\npor familia"])
    cx.set(title="b) Volumen de pivotes y carga", ylabel="Asociación de rangos", ylim=(-1, 1))
    cx.set_yticks([-1, -.5, 0, .5, 1]); cx.set_axisbelow(True)
    cx.grid(axis="y", color="#E7E7E7", linewidth=.6)
    return save(fig, output, "04_eda", ["datos", "correlaciones_globales", "correlaciones_condicionadas"],
                "a) Valores individuales, mediana e intervalo intercuartílico de la primera falla por familia. "
                "b) Spearman global y correlación entre rangos globales residualizados por familia "
                "del volumen de pivotes con el objetivo.",
                "El cambio de signo de +0.828017 a −0.477342 muestra dependencia de la arquitectura; "
                "no es una relación causal ni una prueba de mejora predictiva.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    style()
    data = {key: pd.read_csv(ROOT / path) for key, path in SOURCE_PATHS.items()}
    figures = [function(data, output) for function in [plot_mae, plot_diagnostic, plot_paired, plot_eda]]
    manifest = {
        "purpose": "Figuras académicas de síntesis del sprint Semana 6; sin nuevos modelos ni predicciones.",
        "generator": "reports/semana_6/generar_figuras.py", "generator_sha256": digest(Path(__file__)),
        "source_sha256": {path: digest(ROOT / path) for path in SOURCE_PATHS.values()},
        "matplotlib_version": matplotlib.__version__, "figures": figures,
        "review_note": "Fuentes ejecutadas conservadas. Los gráficos no muestran intervalos inferenciales.",
    }
    (output / "manifest_figuras.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for record in figures:
        print(f"{Path(record['file']).name}: {record['width_inches']} x {record['height_inches']} in, 300 dpi")


if __name__ == "__main__":
    main()
