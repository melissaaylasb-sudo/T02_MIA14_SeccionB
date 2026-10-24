"""Figuras y tablas del informe a partir de evidencia EDA existente; no ajusta modelos."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
TAB = OUT / "tablas"
COLORS = {4: "#365c7a", 5: "#ac6f24", 6: "#2b7b67"}
INPUTS: dict[str, str] = {}
FIGURES: dict[str, str] = {}
TABLES: dict[str, str] = {}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative: str) -> pd.DataFrame:
    path = ROOT / relative
    INPUTS[relative] = sha(path)
    return pd.read_csv(path)


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def axis_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.18, zorder=0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))


def legend(fig, y=.01):
    handles = [Line2D([], [], marker="o", linestyle="", color=COLORS[f], label=f"Familia {f}") for f in COLORS]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, y))


def save(fig, name: str):
    for ext in ("png", "svg"):
        path = FIG / f"EDA_{name}.{ext}"
        fig.savefig(path, dpi=320, bbox_inches="tight", facecolor="white")
        FIGURES[path.relative_to(ROOT).as_posix()] = sha(path)
    plt.close(fig)


def table(name: str, source: str):
    path = TAB / f"EDA_{name}.tex"
    path.write_text(source.strip() + "\n", encoding="utf-8")
    TABLES[path.relative_to(ROOT).as_posix()] = sha(path)


def run():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "font.family": "DejaVu Sans"})
    data = read("data/processed/preliminary/model_table.csv")
    profiles = read("reports/eda/tables/03_geometrias.csv")
    target = read("reports/eda/tables/02_objetivo.csv").iloc[0]
    families = read("reports/eda/tables/05_familias.csv").set_index("n_cells_Y")
    global_assoc = read("reports/eda/tables/08_asociaciones_globales.csv").set_index("variable")
    conditional = read("reports/eda/tables/09_asociaciones_condicionadas.csv").set_index("variable")
    influence = read("reports/eda/tables/20_influencia.csv")
    consistency = read("reports/eda_calidad/tables/10_consistencia_geometrica.csv")
    consistency = consistency.merge(data[["Case_n", "Pivot_height"]], on="Case_n", how="inner", validate="one_to_one")
    # También se registran las evidencias usadas en interpretación, sin recalcular pruebas.
    for relative in ["reports/eda/tables/07_atipicos_resumen.csv", "reports/eda/tables/12_identidades.csv",
                     "reports/eda/tables/05_descomposicion.csv", "reports/eda/tables/17_PCA_varianza.csv",
                     "reports/eda_calidad/tables/09_duplicados_resumen.csv",
                     "reports/eda_calidad/tables/09_configuraciones_repetidas.csv",
                     "reports/eda_calidad/tables/03_equivalencia.csv",
                     "reports/eda_calidad/tables/14_decisiones.csv",
                     "reports/eda_calidad/tables/08_dominios.csv",
                     "reports/eda_transformaciones/tables/T08_asociaciones_caracteristicas.csv",
                     "reports/eda_relaciones/tables/08_objetivo_priorizado.csv"]:
        read(relative)
    for name in ["01_Ingesta_Curaduria_y_Calidad.ipynb", "02_EDA.ipynb", "03_Transformaciones_Exploratorias.ipynb"]:
        relative = f"notebooks/{name}"
        INPUTS[relative] = sha(ROOT / relative)

    # Geometría: valores originales, sin densidades estimadas ni anotaciones de tamaño muestral.
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 7.3))
    for ax in axes.flat:
        axis_style(ax)
    for f, color in COLORS.items():
        sub = data[data.n_cells_Y.eq(f)]
        offset = (f - 5) * .16
        for col, y in [("Fiber_height", 0), ("Fiber_base", 1)]:
            axes[0, 0].scatter(sub[col], np.full(len(sub), y + offset), color=color, s=34, alpha=.8, zorder=3)
        for col, y in [("Pivot_height", 0), ("Pivot_radius", 1)]:
            axes[0, 1].scatter(sub[col], np.full(len(sub), y + offset), color=color, s=34, alpha=.8, zorder=3)
        axes[0, 2].scatter(sub.n_cells_Y, sub.Fiber_total_length, color=color, s=45, alpha=.8, zorder=3)
        axes[1, 0].scatter(sub.n_cells_Y, sub.Pivot_total_number, color=color, s=45, alpha=.8, zorder=3)
        axes[1, 1].scatter(sub.Fiber_total_volume, sub.Pivot_total_volume, color=color, s=34, alpha=.8, zorder=3)
        axes[1, 2].scatter(sub.n_cells_Y + np.linspace(-.1, .1, len(sub)), sub.Sample_total_volume,
                           color=color, s=34, alpha=.8, zorder=3)
    for ax, labels in [(axes[0, 0], ["Altura", "Ancho"]), (axes[0, 1], ["Altura", "Radio"])]:
        ax.set_yticks([0, 1], labels)
        ax.set_ylim(-.45, 1.45)
        ax.set_xlabel("Dimensión [mm]")
    axes[0, 0].set_title("a) Sección de las fibras")
    axes[0, 1].set_title("b) Dimensiones de pivotes")
    axes[0, 2].set_title("c) Longitud total de fibras")
    axes[0, 2].set_ylabel("Longitud [mm]")
    axes[1, 0].set_title("d) Cantidad de pivotes")
    axes[1, 0].set_ylabel("Pivotes de la estructura")
    axes[1, 1].set_title("e) Volúmenes de componentes")
    axes[1, 1].set_xlabel("Volumen de fibras [mm³]")
    axes[1, 1].set_ylabel("Volumen de pivotes [mm³]")
    axes[1, 2].set_title("f) Volumen CAD del espécimen")
    axes[1, 2].set_ylabel("Volumen [mm³]")
    for ax in [axes[0, 2], axes[1, 0], axes[1, 2]]:
        ax.set_xticks([4, 5, 6])
        ax.set_xlabel("Celdas en Y")
    fig.subplots_adjust(wspace=.4, hspace=.43, bottom=.14)
    legend(fig)
    save(fig, "geometria")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax in axes: axis_style(ax)
    values = np.sort(data.First_Failure_Load_N.to_numpy())
    axes[0].step(np.r_[values[0], values], np.r_[0, np.arange(1, len(values) + 1) / len(values) * 100],
                 where="post", color="#365c7a", lw=2)
    axes[0].axvspan(target.Q1_N, target.Q3_N, color="#cbdce7", alpha=.5, label="Rango intercuartílico")
    axes[0].axvline(target.mediana_N, color="#272727", ls="--", lw=1.2, label="Mediana")
    axes[0].set(xlabel="Carga de primera falla [N]", ylabel="Porcentaje acumulado [%]",
                title="a) Distribución acumulada", ylim=(0, 102))
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    for f, color in COLORS.items():
        sub = data.loc[data.n_cells_Y.eq(f), "First_Failure_Load_N"].to_numpy()
        bp = axes[1].boxplot([sub], positions=[f], widths=.5, patch_artist=True,
                             medianprops={"color": "#222222", "linewidth": 1.5}, showfliers=False)
        bp["boxes"][0].set(facecolor=color, alpha=.2)
        axes[1].scatter(f + np.linspace(-.13, .13, len(sub)), sub, color=color, s=28, zorder=3)
        med = families.loc[f, "mediana_N"]
        axes[1].annotate(f"Mediana: {fmt(med)} N", (f, sub.max()), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9)
    axes[1].set(xticks=[4, 5, 6], xticklabels=["Familia 4", "Familia 5", "Familia 6"],
                ylabel="Carga de primera falla [N]", title="b) Distribución por arquitectura", ylim=(35, 240))
    fig.tight_layout(w_pad=2.2)
    save(fig, "primera_falla")

    names = [("n_cells_Y", "Celdas en Y"), ("Fiber_height", "Altura de fibra"),
             ("Fiber_base", "Ancho de fibra"), ("Fiber_total_length", "Longitud de fibras"),
             ("Fiber_total_volume", "Volumen de fibras"), ("Pivot_height", "Altura de pivote"),
             ("Pivot_radius", "Radio de pivote"), ("Pivot_total_number", "Cantidad de pivotes"),
             ("Pivot_total_volume", "Volumen de pivotes"), ("Sample_total_volume", "Volumen CAD")]
    matrix = np.full((len(names), 3), np.nan)
    for i, (source, _) in enumerate(names):
        if source in global_assoc.index:
            matrix[i, :2] = global_assoc.loc[source, ["Pearson", "Spearman"]]
        if source in conditional.index:
            matrix[i, 2] = conditional.loc[source, "rho_condicionado"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.1), gridspec_kw={"width_ratios": [1.13, 1]})
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#efefef")
    axes[0].imshow(matrix, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
    axes[0].set_yticks(range(len(names)), [label for _, label in names])
    axes[0].set_xticks(range(3), ["Pearson\nglobal", "Spearman\nglobal", "Rangos\ncondicionados"])
    axes[0].set_title("a) Asociación con la primera falla")
    axes[0].tick_params(length=0)
    for (i, j), val in np.ndenumerate(matrix):
        axes[0].text(j, i, "—" if np.isnan(val) else fmt(val), ha="center", va="center",
                     color="white" if np.isfinite(val) and abs(val) > .62 else "#222222", fontsize=10)
    axis_style(axes[1])
    for f, color in COLORS.items():
        sub = data[data.n_cells_Y.eq(f)]
        axes[1].scatter(sub.Pivot_total_volume, sub.First_Failure_Load_N, color=color, s=42, label=f"Familia {f}", zorder=3)
    axes[1].set(xlabel="Volumen de pivotes [mm³]", ylabel="Carga de primera falla [N]",
                title="b) Comparación global y por familia")
    axes[1].legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=2.6)
    save(fig, "asociaciones")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax in axes: axis_style(ax)
    diagnostic = influence.merge(data[["Case_n", "First_Failure_Load_N", "n_cells_Y"]], left_on="caso", right_on="Case_n", validate="one_to_one")
    for f, color in COLORS.items():
        sub = diagnostic[diagnostic.n_cells_Y.eq(f)]
        axes[0].scatter(sub.First_Failure_Load_N, sub.Cook, color=color, s=42, zorder=3)
        sub2 = consistency[consistency.n_cells_Y.eq(f)]
        axes[1].scatter(sub2.Pivot_height, sub2["Desviacion_pivote_%"], color=color, s=36, alpha=.75, zorder=3)
    for _, row in diagnostic[diagnostic.bandera_Cook.eq(True)].iterrows():
        axes[0].annotate(f"Caso {int(row.caso)}", (row.First_Failure_Load_N, row.Cook), xytext=(7, 5), textcoords="offset points", fontsize=9)
    for _, row in consistency[consistency.Revision_pivote_5pct.eq(True)].iterrows():
        axes[1].annotate(f"Caso {int(row.Case_n)}", (row.Pivot_height, row["Desviacion_pivote_%"]), xytext=(7, 5), textcoords="offset points", fontsize=9)
    axes[0].set(xlabel="Carga de primera falla observada [N]", ylabel="Distancia de Cook",
                title="a) Influencia en la tendencia descriptiva", ylim=(0, .38))
    axes[1].axhline(5, color="#a74940", ls="--", lw=1, label="Umbral de revisión ±5 %")
    axes[1].axhline(-5, color="#a74940", ls="--", lw=1)
    axes[1].axhline(0, color="gray", lw=.6)
    axes[1].set(xlabel="Altura de pivote [mm]", ylabel="Desviación del volumen ideal [%]",
                title="b) Coherencia de pivotes", xlim=(.94, 1.54), xticks=[1, 1.2, 1.4])
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.tight_layout(w_pad=2)
    save(fig, "influencia")

    table("diccionario", DICTIONARY)
    table("calidad", QUALITY)
    table("puente", BRIDGE)
    manifest = {"description": "Reutilización descriptiva de EDA; sin nuevos modelos ni selección de observaciones",
                "inputs": INPUTS, "figures": FIGURES, "tables": TABLES,
                "source_code": {Path(__file__).relative_to(ROOT).as_posix(): sha(Path(__file__))}}
    (OUT / "manifest_eda.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Artefactos EDA generados y fuentes registradas en manifest_eda.json")


DICTIONARY = r"""
\begin{table}[htbp]
\centering\small\setstretch{1.0}
\caption{Diccionario de variables del análisis. Las salidas posteriores se excluyen de los predictores.}
\label{tab:diccionario}
\begin{tabularx}{\linewidth}{@{}p{3.3cm}lYl@{}}
\toprule
Variable & Unidad & Significado & Rol \\
\midrule
Celdas en Y & conteo & Celdas de la malla en esa dirección; define familia. & Entrada \\
Altura de fibra & mm & Altura de la sección rectangular, $h$. & Entrada \\
Ancho de fibra & mm & Base de la sección rectangular, $b$. & Entrada \\
Longitud de fibras & mm & Longitud total declarada de fibras. & Entrada \\
Volumen de fibras & mm$^3$ & Volumen declarado del conjunto de fibras, $V_f$. & Entrada \\
Altura de pivote & mm & Altura del conector entre fibras. & Entrada \\
Radio de pivote & mm & Radio del conector; constante en los datos. & Entrada \\
Cantidad de pivotes & conteo & Conectores de la estructura. & Entrada \\
Volumen de pivotes & mm$^3$ & Volumen declarado de los conectores, $V_p$. & Entrada \\
Volumen CAD & mm$^3$ & Volumen del espécimen en CAD, distinto de $V_f+V_p$. & Entrada \\
\midrule
Carga de primera falla & N & Fuerza asociada a la primera rotura identificada. & Salida objetivo \\
Carga terminal & N & Fuerza terminal reportada; no necesariamente máxima. & Salida \\
Desplazamiento máximo & mm & Máximo desplazamiento registrado en el resumen. & Salida \\
Desplazamientos residuales & mm & Desplazamiento que permanece tras descargar cada ciclo. & Salida \\
Energía disipada & mJ & Energía tabulada por ciclo de carga--descarga. & Salida \\
Energía de fractura & mJ & Energía tabulada asociada a un evento de rotura. & Salida \\
\midrule
Identificador del caso & --- & Vincula geometría y respuestas; no entra al modelo. & Control \\
\bottomrule
\end{tabularx}
\par\smallskip\small Fuente: diccionario del repositorio y variables de los cuadernos 01--03. Los valores son dimensiones declaradas, no tolerancias de fabricación medidas.
\end{table}
"""

QUALITY = r"""
\begin{table}[htbp]
\centering\small\setstretch{1.0}
\caption{Curaduría aplicada y efecto sobre el análisis.}
\label{tab:curaduria}
\begin{tabularx}{\linewidth}{@{}YYY@{}}
\toprule
Control & Resultado observado & Tratamiento aplicado \\
\midrule
Extracción y tipos & Las asignaciones numéricas y sus nulos coinciden con las tablas extraídas. & Lectura literal sin ejecutar MATLAB; conversión numérica y conservación de identificadores. \\
\addlinespace
Valores faltantes & Geometría completa; ausencia conjunta de respuestas en una entrada de la fuente. & No inventar el objetivo; conservar la geometría y excluir de supervisión la respuesta ausente. \\
\addlinespace
Dominios & Dimensiones positivas, conteos enteros y carga objetivo positiva. & Validar unidades y dominios; ninguna corrección de valores por estos controles. \\
\addlinespace
Repetición & Sin filas exactamente duplicadas; coincidencias de factores con otras medidas distintas. & Conservar sin promediar; reunir configuraciones relacionadas durante validación. \\
\addlinespace
Consistencia y extremos & Radio constante; discrepancias de volúmenes frente a formas ideales; energías iniciales negativas. & Filtrar el radio en entrenamiento; conservar signos y casos señalados; no corregir mediante fórmulas ideales. \\
\addlinespace
Respuestas por ciclo & La disponibilidad cambia entre ciclos y eventos. & Comparar cohortes comunes cuando procede; no convertir ausencia en cero ni cada ciclo en muestra independiente. \\
\bottomrule
\end{tabularx}
\par\smallskip\small Fuente: notebook 01 y tablas de validación, duplicación, dominios y decisiones de curaduría. Imputación estadística y escalamiento se ajustan posteriormente dentro de entrenamiento.
\end{table}
"""

BRIDGE = r"""
\begin{table}[htbp]
\centering\small\setstretch{1.0}
\caption{Del hallazgo exploratorio a las representaciones A, B y C.}
\label{tab:puente-fe}
\begin{tabularx}{\linewidth}{@{}YYY@{}}
\toprule
Hallazgo y evidencia & Interpretación & Decisión para el experimento \\
\midrule
Escalas heterogéneas y radio de 0,5~mm constante & La magnitud numérica influye en la regularización; una constante no distingue geometrías. & A: entradas originales, filtro de constantes y escalamiento dentro de entrenamiento. \\
\addlinespace
Ancho y altura de fibra combinables; momentos con asociación condicionada próxima a cero & La hipótesis es una representación geométrica no lineal, no una alta correlación individual. & B: añadir conjuntamente $bh^3/12$ y $hb^3/12$; comparar con A. \\
\addlinespace
$V_f+V_p$ constante; log-razón con $\rho$ global 0,828 y condicionado $-0,477$ & Relación dependiente de la arquitectura; representar proporción no garantiza mejora. & C: añadir $\ln(V_p/V_f)$ a B; comparar el incremento con B. \\
\addlinespace
Separación de cargas entre familias y coincidencia de factores & El agregado global puede ocultar diferencias dentro de familia. & Mantener el mismo agrupamiento en A--C y revisar error por arquitectura. \\
\addlinespace
Sin banderas IQR/MAD, pero con influencia OLS e incidencias geométricas & Ausencia de extremos univariados no asegura estabilidad del ajuste. & Conservar casos; evaluar errores individuales sin depurar observaciones para mejorar métricas. \\
\bottomrule
\end{tabularx}
\par\smallskip\small Fuente: análisis exploratorio y transformaciones de los cuadernos 02--03. Las propuestas son hipótesis que deben contrastarse con predicciones reservadas.
\end{table}
"""


if __name__ == "__main__":
    run()
