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
    "metricas": "reports/feature_engineering/tables/comparacion_experimentos.csv",
    "predicciones": "reports/feature_engineering/tables/predicciones_oof.csv",
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
        selected = frame[frame.protocolo.eq(protocol)].set_index("experimento").loc[list("ABC")]
        vals = selected.MAE_N.to_numpy()
        ax.bar(["Base A", "Var. 1 B", "Var. 2 C"], vals, color=[BLUE, GREEN, BLUE], width=0.6)
        for pos, value in enumerate(vals):
            ax.text(pos, value + ymax * .025, decimal_text(value, 2), ha="center", va="bottom")
        ax.set(title=title, xlabel="Conjunto de características", ylabel="MAE [N]", ylim=(0, ymax))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E4E4E4", linewidth=.6)
    return save(fig, output, "01_mae", ["metricas"],
                "MAE OOF agregado por caso de las representaciones A, B y C, con Ridge fijo. "
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
    for panel in axes:
        panel.set_axisbelow(True); panel.grid(color="#E7E7E7", linewidth=.6)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    return save(fig, output, "02_diagnostico", ["predicciones"],
                "Predicciones OOF de B al excluir configuraciones. Identidad en panel a; "
                "residuo observado menos predicho y referencia cero en b.",
                "Las familias ocupan bandas distintas; no se trata de predicciones de entrenamiento. "
                "Residuo positivo equivale a subestimación.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    style()
    data = {key: pd.read_csv(ROOT / path) for key, path in SOURCE_PATHS.items()}
    figures = [function(data, output) for function in [plot_mae, plot_diagnostic]]
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
