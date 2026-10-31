"""Figuras reproducibles del sprint de importancia, a partir de resultados OOF.

Cada figura indica la fuente y una limitación breve. Los CSV, no las figuras,
son la fuente de los valores numéricos del informe.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "n_cells_Y": "Número de celdas en Y [conteo]",
    "Fiber_height": "Altura de sección de fibra [mm]",
    "Fiber_base": "Base de sección de fibra [mm]",
    "Fiber_total_length": "Longitud total de fibras [mm]",
    "Fiber_total_volume": "Volumen declarado de fibras [mm³]",
    "Pivot_height": "Altura de pivote [mm]",
    "Pivot_radius": "Radio de pivote [mm]",
    "Pivot_total_number": "Número total de pivotes [conteo]",
    "Pivot_total_volume": "Volumen declarado de pivotes [mm³]",
    "Sample_total_volume": "Volumen declarado del espécimen [mm³]",
    "I_fibra_bh3_mm4": "Momento de sección b·h³/12 [mm⁴]",
    "I_fibra_hb3_mm4": "Momento de sección h·b³/12 [mm⁴]",
    "seccion_fibra_fuentes": "Sección de fibra: altura y base [grupo]",
    "volumenes_componentes": "Volúmenes de fibras y pivotes [grupo]",
}
COLORS = {"E0": "#7a8793", "E1": "#3076a8", "E2": "#2c9b84", "E3": "#dd875c", "E4": "#7654a8"}


def _read(tables: Path, name: str) -> pd.DataFrame:
    path = tables / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _save(fig: plt.Figure, figures: Path, stem: str, note: str) -> None:
    fig.text(0.02, 0.015, f"Fuente: dataset preliminar (26 casos) y artefactos OOF; {note}",
             fontsize=8, color="#40515d")
    fig.savefig(figures / f"{stem}.png", dpi=250, bbox_inches="tight", facecolor="white")
    fig.savefig(figures / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _decorate(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", weight="bold", fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, axis="y")
    ax.spines[["top", "right"]].set_visible(False)


def _shap_matrix(shap_rows: pd.DataFrame, global_rows: pd.DataFrame):
    """Alinea por Case_n y feature; cero solo representa atributo filtrado en train."""
    chosen = global_rows.sort_values("rank_global").feature.tolist()
    values = shap_rows.pivot(index="Case_n", columns="feature", values="shap_N").reindex(columns=chosen).fillna(0)
    raw = shap_rows.pivot(index="Case_n", columns="feature", values="feature_value").reindex(columns=chosen)
    base = shap_rows.groupby("Case_n").base_N.first().reindex(values.index).to_numpy()
    return values, raw, base


def run(root: Path = ROOT) -> list[str]:
    """Exporta PNG y SVG de las comparaciones observadas; devuelve nombres generados."""
    root = Path(root)
    tables = root / "reports" / "feature_importance" / "tables"
    figures = root / "reports" / "feature_importance" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = _read(tables, "metricas_resumen.csv")
    pred = _read(tables, "predicciones_oof.csv")
    srows = _read(tables, "shap_oof.csv")
    sg = _read(tables, "shap_global.csv")
    stability = _read(tables, "shap_estabilidad.csv")
    perm = _read(tables, "permutacion.csv")
    ablation = _read(tables, "ablaciones.csv")
    produced: list[str] = []

    def save(fig: plt.Figure, stem: str, note: str) -> None:
        _save(fig, figures, stem, note)
        produced.append(stem)

    # 01. Desempeño de las dos familias ya estudiadas en dos protocolos.
    order = [x for x in ["E0", "E1", "E2", "E3", "E4", "S1", "S2", "S3", "S4"]
             if x in set(summary.experimento)]
    fig, ax = plt.subplots(figsize=(12.2, 5.6))
    selected = summary[summary.experimento.isin(order)]
    for j, protocol in enumerate(["configuracion", "familia"]):
        subset = selected[selected.protocolo.eq(protocol)].set_index("experimento").reindex(order)
        x = np.arange(len(order)) + (j - 0.5) * 0.36
        ax.bar(x, subset.MAE_N, width=0.35, label=protocol, alpha=0.84)
    labels = {"E0": "E0 · mediana", "E1": "E1 · Ridge A", "E2": "E2 · Ridge B",
              "E3": "E3 · Ridge B ajustado", "E4": "E4 · Ridge A ajustado",
              "S1": "S1 · Elastic Net A", "S2": "S2 · Elastic Net B",
              "S3": "S3 · Elastic Net B ajustado", "S4": "S4 · Elastic Net A ajustado"}
    ax.set_xticks(np.arange(len(order)), [labels.get(x, x) for x in order],
                  fontsize=8, rotation=35, ha="right")
    _decorate(ax, "MAE fuera de muestra por experimento y protocolo", "Experimento", "MAE [N]")
    ax.legend(title="Exclusión externa")
    fig.subplots_adjust(bottom=0.35)
    save(fig, "01_mae_modelos", "n pequeño; los protocolos estiman generalizaciones distintas.")

    # 02. Descomposición de mejoras en pares controlados.
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    cfg = selected[selected.protocolo.eq("configuracion")].drop_duplicates("experimento")
    mapping = dict(zip(cfg.experimento, cfg.MAE_N))
    pairs = [("Ridge: A → B fijo", "E1", "E2"),
             ("Ridge: tuning sobre B", "E2", "E3"),
             ("Ridge: A ajustado → B ajustado", "E4", "E3"),
             ("Elastic Net: A → B fijo", "S1", "S2"),
             ("Elastic Net: tuning sobre B", "S2", "S3"),
             ("Elastic Net: A ajustado → B ajustado", "S4", "S3")]
    pairs = [p for p in pairs if p[1] in mapping and p[2] in mapping]
    changes = [mapping[b] - mapping[a] for _, a, b in pairs]
    ax.barh([p[0] for p in pairs], changes, color=["#2c9b84" if d < 0 else "#c96c63" for d in changes])
    ax.axvline(0, color="#24323d", lw=1)
    _decorate(ax, "Cambio de MAE en comparaciones emparejadas", "ΔMAE [N]; negativo = mejora")
    save(fig, "02_efecto_fe_tuning", "B fue elegida retrospectivamente; diferencias exploratorias.")

    # 03–04. Predicciones y residuales primarios E3.
    primary = pred[pred.protocolo.eq("configuracion") & pred.experimento.eq("E3")].copy()
    if not primary.empty:
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        dots = ax.scatter(primary.observado_N, primary.predicho_N,
                          c=primary.error_absoluto_N, cmap="viridis", s=48)
        limits = [min(primary.observado_N.min(), primary.predicho_N.min()),
                  max(primary.observado_N.max(), primary.predicho_N.max())]
        ax.plot(limits, limits, "--", color="#475569", lw=1)
        _decorate(ax, "Carga observada y estimada fuera de muestra · E3", "Observada [N]", "Estimada [N]")
        fig.colorbar(dots, ax=ax, label="Error absoluto [N]")
        save(fig, "03_observado_predicho", "cada punto se predijo sin incluir su configuración en train.")
        fig, ax = plt.subplots(figsize=(6.2, 4.7))
        dots = ax.scatter(primary.predicho_N, primary.residuo_N,
                          c=primary.observado_N, cmap="viridis", s=48)
        ax.axhline(0, color="#475569", lw=1)
        _decorate(ax, "Residuales fuera de muestra · E3", "Predicción [N]", "Observado − estimado [N]")
        fig.colorbar(dots, ax=ax, label="Carga observada [N]")
        save(fig, "04_residuales", "la dispersión no estima incertidumbre calibrada.")

    sp = srows[srows.protocolo.eq("configuracion") & srows.experimento.eq("E3")]
    gp = sg[sg.protocolo.eq("configuracion") & sg.experimento.eq("E3")]
    if not sp.empty and not gp.empty:
        vals, raw, bases = _shap_matrix(sp, gp)
        names = [LABELS.get(c, c) for c in vals.columns]
        explanation = shap.Explanation(values=vals.to_numpy(), base_values=bases,
                                       data=raw.to_numpy(), feature_names=names)
        shap.plots.beeswarm(explanation, max_display=min(12, len(names)), show=False)
        fig = plt.gcf()
        fig.suptitle("Distribución de contribuciones SHAP OOF · E3", x=0.02, ha="left", weight="bold")
        fig.axes[0].set_xlabel("Contribución SHAP a la predicción [N]")
        if len(fig.axes) > 1:
            fig.axes[-1].set_ylabel("Valor de la característica")
        save(fig, "05_shap_beeswarm", "atribución condicional al modelo, no causal.")

        fig, ax = plt.subplots(figsize=(8.4, 5.8))
        rank = gp.sort_values("mean_abs_shap_N")
        ax.barh([LABELS.get(x, x) for x in rank.feature], rank.mean_abs_shap_N, color="#2c9b84")
        _decorate(ax, "Magnitud media de SHAP fuera de muestra · E3", "mean(|SHAP|) [N]")
        save(fig, "06_shap_global", "atributos correlacionados pueden compartir crédito.")

        if not primary.empty:
            representative = int(primary.iloc[(primary.observado_N - primary.observado_N.median()).abs().argmin()].Case_n)
            high_error = int(primary.loc[primary.error_absoluto_N.idxmax(), "Case_n"])
            for stem, case, title in [
                ("07_shap_representativo", representative, "Contribuciones de un caso representativo"),
                ("08_shap_error_alto", high_error, "Contribuciones del caso con mayor error"),
            ]:
                if case in vals.index:
                    idx = vals.index.get_loc(case)
                    shap.plots.waterfall(explanation[idx], max_display=min(12, len(names)), show=False)
                    fig = plt.gcf()
                    observation = primary.set_index("Case_n").loc[case]
                    fig.suptitle(
                        f"{title} · caso {case} | observado {observation.observado_N:.2f} N, "
                        f"estimado {observation.predicho_N:.2f} N, "
                        f"error {observation.error_absoluto_N:.2f} N",
                        x=0.02, ha="left", weight="bold", fontsize=11)
                    save(fig, stem, "base + aportes reconstruyen la predicción, no el valor observado.")

        top = gp.sort_values("rank_global").feature.iloc[0]
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        dots = ax.scatter(raw[top], vals[top],
                          c=primary.set_index("Case_n").reindex(vals.index).observado_N,
                          cmap="viridis", s=50)
        ax.axhline(0, color="#475569", lw=1)
        _decorate(ax, f"Dependencia SHAP · {LABELS.get(top, top)}", LABELS.get(top, top), "SHAP [N]")
        fig.colorbar(dots, ax=ax, label="Carga observada [N]")
        save(fig, "09_shap_dependencia", "asociación descriptiva; los casos comparten predictores.")

    # 10. Comparación de modelos con la misma representación B y tuning interno.
    two = sg[sg.protocolo.eq("configuracion") & sg.experimento.isin(["E3", "S3"])]
    if two.experimento.nunique() == 2:
        wide = two.pivot(index="feature", columns="experimento", values="mean_abs_shap_N").dropna()
        wide = wide.assign(max_value=wide[["E3", "S3"]].max(axis=1)).sort_values("max_value")
        yaxis = np.arange(len(wide))
        fig, ax = plt.subplots(figsize=(9.2, 6.3))
        ax.barh(yaxis - 0.18, wide.E3, height=0.34, label="Ridge E3", color="#2c9b84")
        ax.barh(yaxis + 0.18, wide.S3, height=0.34, label="Elastic Net S3", color="#dd875c")
        ax.set_yticks(yaxis, [LABELS.get(x, x) for x in wide.index], fontsize=8)
        _decorate(ax, "SHAP global · Ridge y Elastic Net ajustados sobre B",
                  "mean(|SHAP|) [N]")
        ax.legend()
        save(fig, "10_shap_entre_modelos", "atribuciones dependen de coeficientes y referencia de train.")

    stable = stability[stability.protocolo.eq("configuracion") & stability.experimento.eq("E3")]
    if not stable.empty:
        top_features = gp.sort_values("rank_global").feature.head(8).tolist()
        wide = (stable[stable.feature.isin(top_features)]
                .pivot(index="feature", columns="fold", values="rank_fold")
                .reindex(top_features))
        fig, ax = plt.subplots(figsize=(14.8, 6.2))
        im = ax.imshow(wide.to_numpy(), aspect="auto", cmap="viridis_r", vmin=1, vmax=max(8, len(gp)))
        ax.set_yticks(range(len(wide)), [LABELS.get(f, f) for f in wide.index], fontsize=8)
        ax.set_xticks(range(len(wide.columns)), wide.columns, fontsize=8)
        ax.set_xlabel("Partición externa por configuración")
        ax.set_title("Posición de importancia SHAP entre particiones · Ridge E3", loc="left", weight="bold")
        fig.colorbar(im, ax=ax, label="Posición (1 = mayor importancia)")
        for row in range(len(wide.index)):
            for col in range(len(wide.columns)):
                value = wide.iat[row, col]
                if pd.isna(value):
                    continue
                red, green, blue, _ = im.cmap(im.norm(value))
                luminance = 0.299 * red + 0.587 * green + 0.114 * blue
                ax.text(col, row, f"{int(value)}", ha="center", va="center",
                        fontsize=7.3, color="white" if luminance < 0.52 else "#18212b")
        save(fig, "11_estabilidad_shap", "cada celda muestra el rango; particiones de un caso generan rangos locales variables.")

    pp = perm[perm.protocolo.eq("configuracion") & perm.experimento.eq("E3")]
    if not pp.empty:
        pp = pp.sort_values("importancia_media_N")
        fig, ax = plt.subplots(figsize=(9, 5.8))
        ax.barh([LABELS.get(f, f) for f in pp.feature_or_group], pp.importancia_media_N,
                xerr=pp.importancia_sd_N, color="#dd875c", ecolor="#3d5061")
        ax.axvline(0, color="#475569", lw=1)
        _decorate(ax, "Cambio de MAE al permutar atributos OOF · E3", "MAE permutado − MAE original [N]")
        save(fig, "12_permutacion", "permuta entre casos; puede romper dependencias geométricas.")

        both = gp.merge(pp, left_on="feature", right_on="feature_or_group", how="inner")
        if not both.empty:
            fig, ax = plt.subplots(figsize=(6.8, 5.2))
            ax.scatter(both.mean_abs_shap_N, both.importancia_media_N, s=50, color="#2c9b84")
            for _, row in both.iterrows():
                ax.annotate(LABELS.get(row.feature, row.feature).split(" [")[0],
                            (row.mean_abs_shap_N, row.importancia_media_N), fontsize=7)
            ax.axhline(0, color="#475569", lw=1)
            _decorate(ax, "SHAP y permutación miden propiedades distintas", "mean(|SHAP|) [N]", "ΔMAE por permutación [N]")
            save(fig, "13_shap_vs_permutacion", "correlación puede reducir la importancia individual.")

    # Matriz de correlación de entradas originales y momentos, antes de escalar.
    data = pd.read_csv(root / "data" / "processed" / "preliminary" / "model_table.csv")
    from .feature_engineering import MOMENTS
    from .transformaciones_eda import GEOMETRY, geometry_features

    features = data[GEOMETRY].copy()
    features[MOMENTS] = geometry_features(features)[MOMENTS]
    features = features.loc[:, features.nunique(dropna=True) > 1]
    corr = features.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11.8, 10.2))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [LABELS.get(x, x).split(" [")[0] for x in corr.index]
    ax.set_xticks(range(len(labels)), labels, rotation=50, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title("Correlación de Spearman entre descriptores geométricos", loc="left", weight="bold")
    fig.colorbar(im, ax=ax, label="ρ de Spearman")
    for row in range(len(corr.index)):
        for col in range(len(corr.columns)):
            value = corr.iat[row, col]
            if pd.isna(value):
                continue
            red, green, blue, _ = im.cmap(im.norm(value))
            luminance = 0.299 * red + 0.587 * green + 0.114 * blue
            ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=7.5, color="white" if luminance < 0.52 else "#18212b")
    fig.subplots_adjust(bottom=0.24)
    save(fig, "14_correlaciones", "cada celda muestra ρ; dependencias y n = 26 limitan atribuciones individuales.")

    if not ablation.empty:
        subset = ablation[ablation.protocolo.eq("configuracion")]
        if not subset.empty:
            labels = []
            for _, row in subset.iterrows():
                model = "Ridge" if str(row.referencia).startswith("E") else "Elastic Net"
                adjustment = "ajustado" if row.referencia in ["E3", "S3"] else "fijo"
                labels.append(f"{model}: retirar momentos, {adjustment}")
            fig, ax = plt.subplots(figsize=(7.8, max(4.1, 0.62 * len(subset) + 2.0)))
            ax.barh(labels, subset.delta_MAE_N,
                    color=["#c96c63" if v > 0 else "#2c9b84" for v in subset.delta_MAE_N])
            ax.axvline(0, color="#475569", lw=1)
            _decorate(ax, "Efecto predictivo de ablaciones predefinidas", "ΔMAE respecto de referencia [N]")
            fig.subplots_adjust(bottom=0.22)
            save(fig, "15_ablaciones", "sin prueba independiente tras selección retrospectiva de B.")

    # Suma firmada de SHAP dentro de familias disjuntas por caso, y luego magnitud media.
    family_rows = _read(tables, "shap_familias.csv")
    family_rows = family_rows[family_rows.protocolo.eq("configuracion") &
                              family_rows.experimento.eq("E3")]
    if not family_rows.empty:
        grouped = (family_rows.groupby("familia_geometrica", as_index=False)
                   .mean_abs_sum_shap_N.first().sort_values("mean_abs_sum_shap_N"))
        family_labels = {"arquitectura": "Arquitectura y conectividad",
                         "seccion_fibra": "Sección de fibra",
                         "pivotes": "Geometría de pivotes",
                         "volumen_global_y_fibras": "Volumen global y de fibras"}
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.barh([family_labels.get(x, x) for x in grouped.familia_geometrica],
                grouped.mean_abs_sum_shap_N, color="#5d86a9")
        _decorate(ax, "Contribución neta SHAP por familia geométrica",
                  "mean(|Σ SHAP de la familia|) [N]")
        fig.subplots_adjust(bottom=0.21)
        save(fig, "16_familias_geometricas", "agregación de SHAP, no juego de coaliciones grupal.")

    (figures / "manifest_figuras.json").write_text(json.dumps({"figuras": produced,
        "formatos": ["png", "svg"], "origen": "reports/feature_importance/tables",
        "nota": "E3 principal: Ridge con variante B y tuning interno; ver tablas para datos."},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return produced


if __name__ == "__main__":
    print("\n".join(run()))
