"""Evidencia visual del tratamiento geométrico de un fold ya ejecutado.

Reconstruye las transformaciones con las estadísticas guardadas de entrenamiento.
No ajusta regresores, no usa la respuesta y no introduce experimentos adicionales.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from .eda import save_figure_atomic, write_bytes_atomic
from .feature_engineering import LOG_RATIO, MOMENTS, nominal_groups
from .presentacion_fe import ETIQUETAS_VARIABLES, NOMBRES_VARIABLES, UNIDADES_VARIABLES
from .transformaciones_eda import GEOMETRY


TRAIN_COLOR = "#285d89"
TEST_COLOR = "#bc493c"
FAMILY_COLORS = {4: "#2563a6", 5: "#008579", 6: "#d78930"}
FAMILY_MARKERS = {4: "o", 5: "s", 6: "^"}
FIGURES = ["FE08_momentos", "FE09_razon_log", "FE10_redundancia", "FE11_escalamiento"]


def _save_table(path, frame):
    write_bytes_atomic(path, frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def _save_figure(root, name, figure):
    for extension in ("png", "svg"):
        save_figure_atomic(figure, root / "figures" / f"{name}.{extension}", dpi=300)
    plt.close(figure)


def _style_axes(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=.17, zorder=0)
    axis.tick_params(labelsize=10)


def _representation(frame):
    """Las tres fórmulas de C, aplicadas por fila sin parámetros aprendidos."""
    output = frame[GEOMETRY].copy()
    output[MOMENTS[0]] = output.Fiber_base * output.Fiber_height.pow(3) / 12
    output[MOMENTS[1]] = output.Fiber_height * output.Fiber_base.pow(3) / 12
    output[LOG_RATIO] = np.log(output.Pivot_total_volume) - np.log(output.Fiber_total_volume)
    return output


def _reconstruct(frame, config, partitions, audits):
    """Selecciona el fold principal que reserva el menor identificador de caso."""
    first_case = int(frame.Case_n.min())
    partition = next(item for item in partitions
                     if item["protocolo"] == "configuracion" and first_case in item["test_cases"])
    audit = next(item for item in audits
                 if item["protocolo"] == "configuracion" and item["experimento"] == "C"
                 and item["fold"] == partition["fold"])
    for key in ("train_cases", "test_cases"):
        assert audit[key] == partition[key], f"Auditoría incompatible: {key}"
    assert not set(partition["train_cases"]) & set(partition["test_cases"])
    indexed = frame.set_index("Case_n", drop=False)
    train = indexed.loc[partition["train_cases"]].copy()
    reserved = indexed.loc[partition["test_cases"]].copy()
    train_groups = set(nominal_groups(train, config["group_columns"]))
    reserved_groups = set(nominal_groups(reserved, config["group_columns"]))
    assert not train_groups & reserved_groups, "La configuración nominal cruza el split."
    medians = pd.Series(audit["imputer_median"], index=audit["imputer_columns"])
    np.testing.assert_allclose(train[GEOMETRY].median(), medians[GEOMETRY], rtol=1e-12, atol=1e-12)
    train_geometry = train[GEOMETRY].fillna(medians)
    reserved_geometry = reserved[GEOMETRY].fillna(medians)
    train_features = _representation(train_geometry)
    reserved_features = _representation(reserved_geometry)
    names = audit["feature_names"]
    retained = train_features.var(ddof=0) > 0
    assert train_features.columns[retained].tolist() == names, "Filtro de constantes incompatible."
    means = pd.Series(audit["scaler_mean"], index=names)
    scales = pd.Series(audit["scaler_scale"], index=names)
    np.testing.assert_allclose(train_features[names].mean(), means, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(train_features[names].std(ddof=0), scales, rtol=1e-12, atol=1e-12)
    assert audit["scaler_n_samples_seen"] == len(train)
    train_scaled = (train_features[names] - means) / scales
    reserved_scaled = (reserved_features[names] - means) / scales
    np.testing.assert_allclose(train_scaled.mean(), 0, atol=1e-11)
    np.testing.assert_allclose(train_scaled.std(ddof=0), 1, atol=1e-11)
    return partition, train, reserved, train_features, reserved_features, train_scaled, reserved_scaled, means, scales


def _family_scatter(axis, data, x_name, y_name, *, reserved=None):
    for family, group in data.groupby("n_cells_Y", sort=True):
        family = int(family)
        axis.scatter(group[x_name], group[y_name], s=62, alpha=.78,
                     color=FAMILY_COLORS.get(family, TRAIN_COLOR),
                     marker=FAMILY_MARKERS.get(family, "o"), edgecolors="white", linewidths=.5,
                     label=f"Familia Y = {family}", zorder=3)
    if reserved is not None:
        axis.scatter(reserved[x_name], reserved[y_name], s=170, marker="*", color=TEST_COLOR,
                     edgecolors="#51221d", linewidths=.8, label="Caso reservado", zorder=5)
    _style_axes(axis)


def _figure_moments(out, train, reserved):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 1.15, 1.15]})
    fig.suptitle("Construcción de segundos momentos de área de la fibra", x=.05, ha="left", fontsize=17, weight="bold")
    diagram = axes[0]
    diagram.add_patch(Rectangle((.8, .8), 2.4, 1.5, facecolor="#d9e8f3", edgecolor=TRAIN_COLOR, linewidth=2))
    diagram.plot([.4, 3.6], [1.55, 1.55], "--", color="#77838a", linewidth=1)
    diagram.plot([2, 2], [.45, 2.7], "--", color="#77838a", linewidth=1)
    diagram.annotate("", xy=(.8, .45), xytext=(3.2, .45), arrowprops={"arrowstyle": "<->", "color": TRAIN_COLOR})
    diagram.annotate("", xy=(3.55, .8), xytext=(3.55, 2.3), arrowprops={"arrowstyle": "<->", "color": TRAIN_COLOR})
    diagram.text(2, .18, "b: ancho de fibra [mm]", ha="center", fontsize=11)
    diagram.text(3.77, 1.55, "h: altura [mm]", rotation=90, va="center", fontsize=11)
    diagram.text(2, 3.1, "Sección rectangular idealizada", ha="center", fontsize=11, weight="bold")
    diagram.text(2, -.45, r"$I_{bh^3}=\frac{b h^3}{12}$     $I_{hb^3}=\frac{h b^3}{12}$", ha="center", fontsize=17)
    diagram.text(2, -1.05, "Dos ejes de flexión posibles\nDescriptores geométricos [mm⁴]", ha="center", fontsize=10, color="#44525e")
    diagram.set(xlim=(.1, 4.25), ylim=(-1.35, 3.5))
    diagram.set_axis_off()
    for axis, x, y, title in [(axes[1], "Fiber_height", MOMENTS[0], "Altura y momento b·h³/12"),
                               (axes[2], "Fiber_base", MOMENTS[1], "Ancho y momento h·b³/12")]:
        _family_scatter(axis, train, x, y, reserved=reserved)
        axis.set_title(title, fontsize=12, pad=12)
        axis.set_xlabel(f"{ETIQUETAS_VARIABLES[x]} [mm]", fontsize=11)
        axis.set_ylabel(f"{ETIQUETAS_VARIABLES[y]} [mm⁴]", fontsize=10)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.56, .045), ncols=4, frameon=False, fontsize=10)
    fig.text(.05, .01, "Puntos observados, sin curva ajustada. El ancho y la altura varían conjuntamente; estos momentos no son rigidez medida.", fontsize=10, color="#44525e")
    fig.subplots_adjust(left=.05, right=.98, top=.84, bottom=.23, wspace=.42)
    _save_figure(out, FIGURES[0], fig)


def _figure_ratio(out, train, reserved):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    fig.suptitle("Razón de volúmenes y transformación logarítmica", x=.07, ha="left", fontsize=17, weight="bold")
    for axis, values, held, xlabel, title in [
        (axes[0], train.Pivot_total_volume / train.Fiber_total_volume,
         reserved.Pivot_total_volume / reserved.Fiber_total_volume,
         "Volumen de pivotes / volumen de fibras [adimensional]", "Antes: razón positiva"),
        (axes[1], train[LOG_RATIO], reserved[LOG_RATIO],
         "ln(volumen de pivotes / volumen de fibras) [adimensional]", "Después: logaritmo natural")]:
        sorted_values = np.sort(values.to_numpy())
        cumulative = np.arange(1, len(values) + 1) / len(values)
        axis.step(sorted_values, cumulative, where="post", color=TRAIN_COLOR, linewidth=2.1,
                  label="Distribución empírica de entrenamiento")
        axis.scatter(sorted_values, cumulative, s=13, color=TRAIN_COLOR, alpha=.6)
        for position, value in enumerate(held):
            axis.axvline(value, color=TEST_COLOR, linestyle="--", linewidth=1.5,
                         label="Caso reservado" if position == 0 else None)
            percentile = np.mean(sorted_values <= value)
            axis.scatter([value], [percentile], marker="*", s=150, color=TEST_COLOR, zorder=5)
        axis.set(xlabel=xlabel, ylabel="Proporción acumulada", ylim=(-.03, 1.04))
        axis.set_title(title, fontsize=12, pad=10)
        _style_axes(axis)
    axes[0].legend(loc="lower right", frameon=False, fontsize=9)
    fig.text(.07, .06, "ln(Vp/Vf) = ln(Vp) − ln(Vf). La transformación cambia la escala y conserva el orden; no garantiza normalidad.", fontsize=10, color="#44525e")
    fig.subplots_adjust(left=.07, right=.98, bottom=.22, top=.83, wspace=.25)
    _save_figure(out, FIGURES[1], fig)


def _figure_redundancy(out, train):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    fig.suptitle("Dependencias geométricas observadas en entrenamiento", x=.06, ha="left", fontsize=17, weight="bold")
    _family_scatter(axes[0], train, "Fiber_total_volume", "Pivot_total_volume")
    total = train.Fiber_total_volume + train.Pivot_total_volume
    reference_total = float(total.median())
    values = np.linspace(train.Fiber_total_volume.min(), train.Fiber_total_volume.max(), 100)
    axes[0].plot(values, reference_total - values, "--", color="#66717a", linewidth=1.4, zorder=1)
    axes[0].set(xlabel="Volumen de fibras [mm³]", ylabel="Volumen de pivotes [mm³]")
    axes[0].set_title("Volúmenes complementarios", fontsize=12, pad=11)
    sum_label = (f"Vf + Vp = {reference_total:.2f} mm³" if np.ptp(total) < .011
                 else f"Vf + Vp: {total.min():.2f}–{total.max():.2f} mm³")
    axes[0].text(.04, .04, sum_label, transform=axes[0].transAxes, fontsize=10,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    for axis, y, title in [(axes[1], "Fiber_total_length", "Familia y longitud de fibras"),
                           (axes[2], "Pivot_total_number", "Familia y número de pivotes")]:
        _family_scatter(axis, train, "n_cells_Y", y)
        axis.set_xlabel("Número de celdas en Y [conteo]", fontsize=11)
        axis.set_ylabel(f"{ETIQUETAS_VARIABLES[y]} [{UNIDADES_VARIABLES[y]}]", fontsize=11)
        axis.set_title(title, fontsize=12, pad=11)
        axis.set_xticks(sorted(train.n_cells_Y.unique()))
    radius = train.Pivot_radius
    constant_label = (f"Radio de pivote constante: {float(radius.iloc[0]):g} mm; se excluye por varianza cero."
                      if radius.nunique() == 1 else "El radio se filtra solo si es constante en el fold.")
    fig.text(.06, .09, constant_label, fontsize=10, color="#44525e")
    fig.text(.06, .035, "Son dependencias del diseño observado. Retirar descriptores es una hipótesis de la Variante 3; no implica conservar la capacidad predictiva.", fontsize=10, color="#44525e")
    fig.subplots_adjust(left=.06, right=.98, top=.82, bottom=.26, wspace=.39)
    _save_figure(out, FIGURES[2], fig)


def _figure_scaling(out, train, reserved, train_scaled, reserved_scaled):
    selected = ["Fiber_height", "Fiber_total_length", "Pivot_total_volume", *MOMENTS, LOG_RATIO]
    short_labels = ["Altura\nde fibra", "Longitud total\nde fibras", "Volumen total\nde pivotes",
                    "Momento\nb·h³/12", "Momento\nh·b³/12", "Logaritmo de\nrazón Vp/Vf"]
    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 6, height_ratios=[1, 1.35], hspace=.72, wspace=.65)
    fig.suptitle("Estandarización con estadísticas exclusivas de entrenamiento", x=.06, ha="left", fontsize=17, weight="bold", y=.98)
    fig.text(.06, .92, "Antes: cada panel conserva su unidad y escala originales", fontsize=12, weight="bold")
    box_options = {"patch_artist": True, "showfliers": True,
                   "boxprops": {"facecolor": "#d9e8f3", "edgecolor": TRAIN_COLOR},
                   "medianprops": {"color": TRAIN_COLOR, "linewidth": 1.7},
                   "whiskerprops": {"color": TRAIN_COLOR}, "capprops": {"color": TRAIN_COLOR},
                   "flierprops": {"marker": ".", "markersize": 4, "markeredgecolor": TRAIN_COLOR}}
    for index, name in enumerate(selected):
        axis = fig.add_subplot(grid[0, index])
        axis.boxplot(train[name].to_numpy(), positions=[0], widths=.5, **box_options)
        axis.scatter(np.zeros(len(reserved)), reserved[name], marker="*", s=140,
                     color=TEST_COLOR, edgecolors="white", linewidth=.5, zorder=5)
        axis.set_xticks([])
        axis.set_title(short_labels[index], fontsize=11, pad=9)
        axis.set_ylabel(f"[{UNIDADES_VARIABLES[name]}]", fontsize=10)
        axis.set_xlim(-.7, .7)
        _style_axes(axis)
    scaled_axis = fig.add_subplot(grid[1, :])
    scaled_axis.boxplot([train_scaled[name].to_numpy() for name in selected], positions=range(6),
                        widths=.45, **box_options)
    for index, name in enumerate(selected):
        scaled_axis.scatter(np.full(len(reserved), index), reserved_scaled[name], marker="*", s=150,
                            color=TEST_COLOR, edgecolors="white", linewidth=.5, zorder=5)
    scaled_axis.axhline(0, color="#66717a", linewidth=1, linestyle="--")
    scaled_axis.set_xticks(range(6), short_labels, fontsize=11)
    scaled_axis.set_ylabel("Valor estandarizado [adimensional]", fontsize=11)
    scaled_axis.set_title(r"Después: $z=(x-\mu_{entrenamiento})/\sigma_{entrenamiento}$", loc="left", fontsize=13, pad=13, weight="bold")
    _style_axes(scaled_axis)
    legend = [Line2D([0], [0], color=TRAIN_COLOR, linewidth=6, label="Distribución de entrenamiento"),
              Line2D([0], [0], color=TEST_COLOR, marker="*", linestyle="", markersize=12,
                     label="Caso reservado: se transforma, no se reajusta")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.52, .055), ncols=2, frameon=False, fontsize=10)
    fig.text(.06, .025, "Media 0 y desviación 1 en entrenamiento (ddof = 0). Las cajas muestran mediana y cuartiles; el escalamiento no elimina atípicos.", fontsize=10, color="#44525e")
    fig.subplots_adjust(left=.065, right=.98, top=.84, bottom=.18)
    _save_figure(out, FIGURES[3], fig)


def ejecutar_tratamiento(root, frame, config, partitions, audits):
    """Exporta cuatro figuras y evidencia tabular del tratamiento del mismo fold.

    El argumento ``frame`` contiene la tabla supervisada para conservar sus casos;
    solo se leen identificadores y geometría, nunca la variable objetivo.
    """
    out = Path(root) / "reports" / "feature_engineering"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (partition, train, reserved, train_features, reserved_features,
     train_scaled, reserved_scaled, means, scales) = _reconstruct(frame, config, partitions, audits)

    pieces = []
    for role, original, features in [("entrenamiento", train, train_features), ("reservado", reserved, reserved_features)]:
        piece = features[["Fiber_base", "Fiber_height", "Fiber_total_volume", "Pivot_total_volume", *MOMENTS, LOG_RATIO]].copy()
        piece.insert(0, "identificador_caso", original.Case_n.astype(int))
        piece.insert(1, "particion", role)
        piece.insert(2, "fold", partition["fold"])
        piece["razon_volumenes_pivotes_fibras"] = piece.Pivot_total_volume / piece.Fiber_total_volume
        pieces.append(piece.rename(columns=NOMBRES_VARIABLES))
    _save_table(out / "tables" / "tratamiento_formulas_por_caso.csv", pd.concat(pieces, ignore_index=True))

    distributions = []
    for name, values, held in [("razon_volumenes_pivotes_fibras", train_features.Pivot_total_volume / train_features.Fiber_total_volume,
                              reserved_features.Pivot_total_volume / reserved_features.Fiber_total_volume),
                             (NOMBRES_VARIABLES[LOG_RATIO], train_features[LOG_RATIO], reserved_features[LOG_RATIO])]:
        for case, value in held.items():
            distributions.append({"variable": name, "fold": partition["fold"], "unidad": "adimensional",
                                  "minimo_train": values.min(), "q25_train": values.quantile(.25),
                                  "mediana_train": values.median(), "q75_train": values.quantile(.75),
                                  "maximo_train": values.max(), "caso_reservado": int(case),
                                  "valor_reservado": value, "proporcion_train_menor_igual": float((values <= value).mean())})
    _save_table(out / "tables" / "tratamiento_razon_log.csv", pd.DataFrame(distributions))

    scaling = []
    for name in train_scaled:
        for case in reserved_features.index:
            scaling.append({"variable": NOMBRES_VARIABLES[name], "significado": ETIQUETAS_VARIABLES[name],
                            "unidad_original": UNIDADES_VARIABLES[name], "fold": partition["fold"],
                            "media_train_original": means[name], "desviacion_train_original_ddof0": scales[name],
                            "media_train_estandarizada": train_scaled[name].mean(),
                            "desviacion_train_estandarizada_ddof0": train_scaled[name].std(ddof=0),
                            "caso_reservado": int(case), "valor_reservado_original": reserved_features.loc[case, name],
                            "valor_reservado_estandarizado": reserved_scaled.loc[case, name]})
    _save_table(out / "tables" / "tratamiento_escalamiento.csv", pd.DataFrame(scaling))

    decisions = [
        ("Contrato de entrada", "Medidas positivas; conteos enteros; geometría previa al ensayo", "Reglas fijas", "Excluye respuesta e identificador del predictor"),
        ("Codificación categórica", "No se usa one-hot: todas las entradas son medidas o conteos", "No requiere ajuste", "Número de celdas se conserva como conteo numérico"),
        ("Imputación por mediana", "Contingencia para geometría faltante; no se imputa la respuesta", "Mediana solo de entrenamiento", f"Valores imputados en train: {int(train[GEOMETRY].isna().sum().sum())}; en reservado: {int(reserved[GEOMETRY].isna().sum().sum())}"),
        ("Momentos de sección", "b·h³/12 y h·b³/12 resumen la sección para dos ejes", "Fórmulas por fila; sin ajuste", "Variante 1 añade ambos; no representan rigidez medida"),
        ("Razón logarítmica", "ln(Vp/Vf) expresa balance relativo de los volúmenes", "Fórmula por fila; exige volúmenes positivos", "Variante 2 añade una columna; no garantiza normalidad"),
        ("Filtro de constantes", "Radio de pivote sin variación en este fold", "Varianza calculada solo en entrenamiento", "Se retira antes de estandarizar"),
        ("Reducción de descriptores", "Dependencias de volúmenes, longitud y número de pivotes en el diseño", "Bloque predefinido para Variante 3", "La comparación predictiva decide si conviene retirarlos"),
        ("Estandarización", "Evita que las unidades dominen la penalización Ridge", "Media y desviación ddof=0 solo de entrenamiento", "El reservado usa las mismas estadísticas; sin reajuste"),
        ("Tratamiento de atípicos", "Se conservan las geometrías observadas", "Sin recorte, winsorización ni eliminación", "El análisis de errores no modifica las entradas"),
    ]
    _save_table(out / "tables" / "tratamiento_decisiones.csv", pd.DataFrame(decisions, columns=["tecnica", "motivo", "ajuste", "aplicacion"]))

    checks = [
        ("Separación de casos", str(partition["train_cases"]), str(partition["test_cases"]), "Sin intersección"),
        ("Separación por configuración nominal", " | ".join(config["group_columns"]), partition["grupo_excluido"], "Sin grupos compartidos"),
        ("Medianas", "Calculadas en entrenamiento", "Reutilizadas sin reajuste", "Coinciden con la auditoría del fold"),
        ("Media y desviación", "Calculadas en entrenamiento", "Reutilizadas sin reajuste", "Coinciden con la auditoría de Variante 2"),
        ("Respuesta experimental", "No utilizada para construir ni escalar", "No utilizada", "Tratamiento exclusivamente geométrico"),
    ]
    _save_table(out / "tables" / "tratamiento_validacion.csv", pd.DataFrame(checks, columns=["control", "entrenamiento", "reservado", "verificacion"]))

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titleweight": "semibold", "svg.fonttype": "none"}):
        _figure_moments(out, train_features, reserved_features)
        _figure_ratio(out, train_features, reserved_features)
        _figure_redundancy(out, train_features)
        _figure_scaling(out, train_features, reserved_features, train_scaled, reserved_scaled)
    return FIGURES.copy()
