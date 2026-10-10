"""Representaciones geométricas exploratorias; no selecciona ni entrena regresores."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from .importar_matlab import load_matlab_samples
from .ingesta import sha256_file

GEOMETRY = ["n_cells_Y", "Fiber_height", "Fiber_base", "Fiber_total_length",
            "Fiber_total_volume", "Pivot_height", "Pivot_radius",
            "Pivot_total_number", "Pivot_total_volume", "Sample_total_volume"]
CORE = ["n_cells_Y", "Fiber_height", "Fiber_base", "Pivot_height", "Sample_total_volume"]
FORMULAS = [
    ("area_fibra_mm2", "Área rectangular de fibra", "b·h", "mm²", "Geometría ideal; no área resistente global"),
    ("aspecto_fibra", "Relación de aspecto de fibra", "h/b", "1", "Forma de sección; no deformación"),
    ("I_fibra_bh3_mm4", "Segundo momento geométrico bh³/12", "b·h³/12", "mm⁴", "Eje paralelo a b; sección rectangular ideal; no EI"),
    ("I_fibra_hb3_mm4", "Segundo momento geométrico hb³/12", "h·b³/12", "mm⁴", "Eje paralelo a h; sección rectangular ideal; no EI"),
    ("area_pivote_mm2", "Área circular ideal de pivote", "π·r²", "mm²", "Sección geométrica; constante en esta fuente"),
    ("esbeltez_pivote", "Altura/diámetro de pivote", "h_p/(2·r)", "1", "Proxy geométrico; no umbral de pandeo"),
    ("fraccion_volumen_pivotes", "Fracción declarada de volumen de pivotes", "V_p/(V_f+V_p)", "1", "Denominador suma declarada; distinto del volumen CAD"),
    ("volumen_por_celda_mm3", "Volumen CAD por celda en Y", "V_CAD/n_Y", "mm³", "Normalización por conteo en Y; no volumen físico de una celda"),
    ("razon_volumenes", "Razón de volúmenes declarados", "V_f/V_p", "1", "Variables complementarias por diseño; no efectos independientes"),
]


def geometry_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Fórmulas por fila: no aprenden parámetros ni leen el objetivo o respuestas."""
    missing = set(GEOMETRY) - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan variables geométricas: {sorted(missing)}")
    x = frame[GEOMETRY].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(x.to_numpy(dtype=float)).all() or (x <= 0).any().any():
        raise ValueError("La construcción geométrica requiere dimensiones y conteos positivos y finitos.")
    b, h, r, hp = x.Fiber_base, x.Fiber_height, x.Pivot_radius, x.Pivot_height
    return pd.DataFrame({
        "area_fibra_mm2": b * h,
        "aspecto_fibra": h / b,
        "I_fibra_bh3_mm4": b * h ** 3 / 12,
        "I_fibra_hb3_mm4": h * b ** 3 / 12,
        "area_pivote_mm2": np.pi * r ** 2,
        "esbeltez_pivote": hp / (2 * r),
        "fraccion_volumen_pivotes": x.Pivot_total_volume / (x.Fiber_total_volume + x.Pivot_total_volume),
        "volumen_por_celda_mm3": x.Sample_total_volume / x.n_cells_Y,
        "razon_volumenes": x.Fiber_total_volume / x.Pivot_total_volume,
    }, index=frame.index)


def make_preprocessor(scale="standard") -> Pipeline:
    """Pipeline sin estimador: todo parámetro se aprende solo en fit(X_train)."""
    if scale not in {"standard", "robust"}:
        raise ValueError("Escalamiento desconocido.")
    scaler = StandardScaler() if scale == "standard" else RobustScaler()
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("variance", VarianceThreshold()), ("scale", scaler)])


def representation_diagnostics(frame: pd.DataFrame) -> dict:
    variable = frame.loc[:, frame.nunique(dropna=True) > 1]
    if variable.empty or variable.isna().any().any():
        raise ValueError("Diagnóstico requiere variables observadas con variación.")
    z = StandardScaler().fit_transform(variable)
    singular = np.linalg.svd(z, compute_uv=False)
    # Relaciones exactas de diseño pueden dejar ruido de centrado ~1e-13.
    # Umbral relativo explícito para no interpretarlo como una dimensión física.
    relative_tolerance = 1e-10
    rank = int(np.count_nonzero(singular > singular[0] * relative_tolerance))
    return {"variables": frame.shape[1], "constantes": frame.shape[1] - variable.shape[1],
            "rango_centrado": rank, "tolerancia_relativa": relative_tolerance,
            "columnas_variables": variable.shape[1],
            "condicion": float(singular[0] / singular[-1]) if rank == variable.shape[1] else float("inf")}


def run_transformations(root: Path) -> dict:
    root = Path(root)
    raw = root / "data/raw/dati_campagna_venditti.m"
    source_hash = sha256_file(raw)
    full = load_matlab_samples(raw)
    x = full[GEOMETRY].copy()
    derived = geometry_features(x)
    out = root / "reports/eda_transformaciones"
    figdir, tabdir = out / "figures", out / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)
    tables, figures = {}, {}

    def table(name, value):
        tables[name] = value
        value.to_csv(tabdir / f"{name}.csv", index=False, encoding="utf-8")

    def figure(name, fig, title):
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(figdir / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures[name] = {"path": str(figdir / f"{name}.png"), "title": title}

    table("T01_formulas", pd.DataFrame(FORMULAS, columns=["variable", "nombre", "formula", "unidad", "limite"]))
    table("T02_geometria_derivada", pd.concat([full[["Case_n"]], derived], axis=1))
    representatives = {"Original geométrica": x, "Núcleo dimensional": x[CORE],
                       "Forma y arquitectura": pd.concat([x[["n_cells_Y", "Sample_total_volume"]],
                                                         derived[["area_fibra_mm2", "aspecto_fibra", "esbeltez_pivote"]]], axis=1)}
    diagnostics = pd.DataFrame([{"representacion": label, **representation_diagnostics(v)}
                                for label, v in representatives.items()])
    table("T03_representaciones", diagnostics)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pos = np.arange(len(diagnostics))
    axes[0].bar(pos - .17, diagnostics.columnas_variables, .34, label="Variables no constantes", color="#2563a6")
    axes[0].bar(pos + .17, diagnostics.rango_centrado, .34, label="Rango numérico centrado", color="#008579")
    axes[0].set(xticks=pos, xticklabels=["Original", "Núcleo", "Forma"], ylabel="Dimensión algebraica", title="Redundancia de representaciones")
    axes[0].legend(fontsize=8)
    for label, v in representatives.items():
        z = StandardScaler().fit_transform(v.loc[:, v.nunique() > 1])
        singular = np.linalg.svd(z, compute_uv=False)
        axes[1].semilogy(range(1, len(singular) + 1), np.maximum(singular, 1e-16), "o-", label=label)
    axes[1].set(xlabel="Índice singular", ylabel="Valor singular [sin unidad]", title="Dependencias lineales y casi dependencias")
    axes[1].legend(fontsize=8)
    figure("F01_representaciones", fig, "Dimensión algebraica de representaciones geométricas")

    log_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for col in ("Fiber_total_volume", "Pivot_total_volume", "Sample_total_volume"):
        logged = np.log(x[col] / 1.0)  # Referencia explícita: 1 mm³; argumento adimensional.
        log_rows.append([col, stats.skew(x[col], bias=False), stats.skew(logged, bias=False),
                         stats.spearmanr(x[col], logged).statistic, "ln(V / 1 mm³)"])
    table("T04_logaritmos", pd.DataFrame(log_rows, columns=["variable", "asimetria_original", "asimetria_log", "rho_original_log", "formula"]))
    for g, color in {4: "#2563a6", 5: "#008579", 6: "#d78930"}.items():
        values = x.loc[x.n_cells_Y == g, "Sample_total_volume"].sort_values()
        ecdf = np.arange(1, len(values) + 1) / len(values)
        axes[0].step(values, ecdf, where="post", color=color, label=f"Celdas Y = {g}")
        axes[1].step(np.log(values / 1.0), ecdf, where="post", color=color)
    axes[0].set(xlabel="Volumen CAD [mm³]", ylabel="Proporción acumulada", title="Escala original")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="ln(volumen CAD / 1 mm³)", ylabel="Proporción acumulada", title="Cambio de escala, mismo orden")
    figure("F02_logaritmos", fig, "Efecto del logaritmo sobre la distribución del volumen CAD")

    fold_rows = []
    first_outputs = {}
    for fold, (train, test) in enumerate(KFold(5, shuffle=True, random_state=42).split(x), 1):
        for strategy in ("standard", "robust"):
            pipe = make_preprocessor(strategy)
            a = pipe.fit_transform(x.iloc[train])
            b = pipe.transform(x.iloc[test])
            mask = pipe.named_steps["variance"].get_support()
            np.testing.assert_allclose(pipe.named_steps["imputer"].statistics_, x.iloc[train].median())
            train_after = pipe.named_steps["imputer"].transform(x.iloc[train])[:, mask]
            center = train_after.mean(axis=0) if strategy == "standard" else np.median(train_after, axis=0)
            observed = pipe.named_steps["scale"].mean_ if strategy == "standard" else pipe.named_steps["scale"].center_
            np.testing.assert_allclose(observed, center)
            assert not set(train) & set(test)
            assert np.isfinite(a).all() and np.isfinite(b).all()
            fold_rows.append([fold, strategy, len(train), len(test), "; ".join(x.columns[~mask]),
                              float(np.max(np.abs(observed - center))), "Verificado solo con entrenamiento"])
            if fold == 1:
                first_outputs[strategy] = (x.columns[mask], a, b)
    table("T05_aislamiento_transformaciones", pd.DataFrame(fold_rows, columns=[
        "fold", "escalamiento", "n_train", "n_reservado", "constantes_en_train", "error_centro_train", "verificacion"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, strategy in zip(axes, ("standard", "robust")):
        names, a, b = first_outputs[strategy]
        indexes = [list(names).index(col) for col in ("Fiber_height", "Pivot_height", "Sample_total_volume")]
        ax.boxplot([a[:, i] for i in indexes], tick_labels=["Altura fibra", "Altura pivote", "Volumen CAD"], showfliers=False)
        for k, i in enumerate(indexes, 1):
            ax.scatter(np.full(len(b), k), b[:, i], marker="x", color="#d78930", s=30, label="Reservado" if k == 1 else None)
        ax.set(ylabel="Valor transformado [sin unidad]", title="Escala estándar" if strategy == "standard" else "Escala robusta")
        ax.legend(fontsize=8)
    figure("F03_escalamiento", fig, "Transformaciones ajustadas en entrenamiento y aplicadas a casos reservados")

    target = "First_Failure_Load_N"
    leakage = []
    for c in full.columns:
        role = "INPUT geométrico" if c in GEOMETRY else "target" if c == target else "identificador" if c == "Case_n" else "OUTPUT posensayo"
        leakage.append([c, role, c in GEOMETRY, "Disponible antes del ensayo" if c in GEOMETRY else "Excluir de X"])
    leakage.extend([[c, "OUTPUT posensayo", False, "Excluir de X: energía obtenida durante el ensayo"]
                    for c in ["Energy_Dissipated", "Energy_Fracture"]])
    table("T06_frontera_predictiva", pd.DataFrame(leakage, columns=["variable", "rol", "admisible_temporalmente", "decision"]))
    decisions = pd.DataFrame([
        ["Fórmulas geométricas", "Candidatas, no selección definitiva", "Construcción determinística sin target; contrastar geometría con fichas"],
        ["Volúmenes fibra/pivote", "Controlar redundancia", "Su suma declarada es constante; no interpretar efectos independientes"],
        ["Radio del pivote", "Eliminar constante dentro de train", "Fuera de este dominio podría variar; no se elimina de la fuente"],
        ["Escala estándar/robusta", "Comparar dentro de validación interna", "Este cuaderno verifica implementación, no cuál predice mejor"],
        ["Logaritmos", "Alternativa descriptiva", "Preservan orden; no crean información ni garantizan normalidad"],
        ["Imputación", "Solo predictores dentro de train", "No imputar target ni completar eventos no observados"],
        ["Particiones", "Escenario demostrativo de aislamiento", "Confirmar réplicas/lotes; una familia de diseño no es por sí sola una réplica"],
        ["Transferencia", "No evaluada", "Nuevas arquitecturas, materiales y protocolos requieren validación externa"],
    ], columns=["elemento", "decision", "justificacion"])
    table("T07_decisiones", decisions)
    assert sha256_file(raw) == source_hash
    manifest = {"status": "executed", "scope": "transformaciones exploratorias sin entrenamiento predictivo",
                "source": {raw.relative_to(root).as_posix(): source_hash}, "seed": 42,
                "tables": list(tables), "figures": {k: v["title"] for k, v in figures.items()},
                "artifacts_sha256": {p.relative_to(out).as_posix(): sha256_file(p)
                                      for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"}}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"tables": tables, "figures": figures, "manifest": manifest}
