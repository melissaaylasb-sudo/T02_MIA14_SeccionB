"""Relaciones exploratorias completas, con soporte efectivo y sensibilidad.

Cada fila analítica representa un caso experimental. Los ciclos y eventos son
columnas repetidas del mismo caso, nunca observaciones independientes. No se
imputan resultados, no se seleccionan predictores y no se entrenan modelos.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .eda import safe_correlation, rank_association, save_figure_atomic, write_bytes_atomic
from .eda_energia import load_energy_tables
from .ingesta import execution_context, sha256_file


TARGET = "First_Failure_Load_N"
INPUTS = ["n_cells_Y", "Fiber_height", "Fiber_base", "Fiber_total_length",
          "Fiber_total_volume", "Pivot_height", "Pivot_radius",
          "Pivot_total_number", "Pivot_total_volume", "Sample_total_volume"]
CYCLE_COLUMNS = [f"residual_Displacement_{i}cycle_{10*i}mm" for i in range(1, 7)]
COLORS = {4: "#2466a2", 5: "#078377", 6: "#c9872a"}
LABELS = {
    "n_cells_Y": "Celdas en Y", "Fiber_height": "Altura de fibra",
    "Fiber_base": "Base de fibra", "Fiber_total_length": "Longitud de fibras",
    "Fiber_total_volume": "Volumen de fibras", "Pivot_height": "Altura de pivote",
    "Pivot_radius": "Radio de pivote", "Pivot_total_number": "Número de pivotes",
    "Pivot_total_volume": "Volumen de pivotes", "Sample_total_volume": "Volumen del espécimen",
    TARGET: "Primera falla", "Ultimate_Load": "Carga de rotura final",
    "Maximum_Displacement": "Desplazamiento máximo",
    "Fracture_observed_sum": "Suma de energías de fractura registradas",
    "Fracture_registered_events": "Eventos con energía registrada",
    "Fracture_initial_share": "Fracción energética del evento inicial",
    "Load_ratio_first_over_ultimate": "Primera falla / Ultimate_Load",
    "Recovery_complement_cycle6": "Complemento residual nominal, ciclo 6",
    "Fiber_area_proxy": "Área rectangular de fibra (proxy)",
    "Fiber_aspect_ratio": "Altura / base de fibra",
    "Pivot_aspect_ratio": "Altura / diámetro de pivote",
    "Fiber_I_proxy": "Momento de área rectangular (proxy)",
    "Load_per_specimen_volume": "Carga / volumen del espécimen",
    "Load_per_pivot": "Carga / número de pivotes",
    "Load_per_cell_Y": "Carga / celdas en Y",
}
for _i in range(1, 8):
    LABELS[f"Dissipated_cycle_{_i}"] = f"Energía disipada, ciclo {_i}"
for _i in range(1, 14):
    LABELS[f"Fracture_event_{_i}"] = f"Energía de fractura, evento {_i}"
for _i, _c in enumerate(CYCLE_COLUMNS, 1):
    LABELS[_c] = f"Residual, ciclo {_i}"
    LABELS[f"Residual_fraction_cycle{_i}"] = f"Residual / amplitud, ciclo {_i}"


def _finite_pair(x, y):
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Las series deben ser vectores con la misma longitud.")
    valid = np.isfinite(a) & np.isfinite(b)
    return a[valid], b[valid], valid


def _coefficients(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan, np.nan
    return (safe_correlation(x, y), safe_correlation(stats.rankdata(x), stats.rankdata(y)),
            float(stats.kendalltau(x, y, variant="b").statistic))


def association_diagnostics(x, y, *, case_ids=None):
    """Asociación por parejas completas y omisión de un caso, sin inferencia.

    Los intervalos LOO son rangos de sensibilidad, NO intervalos de confianza.
    No se informa conservación del signo si el coeficiente original es cero.
    """
    a, b, valid = _finite_pair(x, y)
    ids = np.arange(len(valid)) if case_ids is None else np.asarray(case_ids)
    if len(ids) != len(valid):
        raise ValueError("Los identificadores no coinciden con las series.")
    ids = ids[valid]
    pearson, spearman, kendall = _coefficients(a, b)
    result = {"n_efectivo": len(a), "Pearson": pearson, "Spearman": spearman,
              "Kendall_tau_b": kendall, "empates_X_pct": np.nan,
              "empates_Y_pct": np.nan, "diferencia_Pearson_Spearman": pearson - spearman}
    if len(a):
        result["empates_X_pct"] = 100 * (1 - len(np.unique(a)) / len(a))
        result["empates_Y_pct"] = 100 * (1 - len(np.unique(b)) / len(b))
    for label, original in [("Pearson", pearson), ("Spearman", spearman)]:
        values, kept_ids = [], []
        if len(a) >= 4 and np.isfinite(original):
            for i in range(len(a)):
                keep = np.arange(len(a)) != i
                xa, ya = a[keep], b[keep]
                if label == "Spearman":
                    xa, ya = stats.rankdata(xa), stats.rankdata(ya)
                value = safe_correlation(xa, ya)
                if np.isfinite(value):
                    values.append(value)
                    kept_ids.append(ids[i])
        values = np.asarray(values)
        result[f"{label}_LOO_min"] = float(values.min()) if len(values) else np.nan
        result[f"{label}_LOO_max"] = float(values.max()) if len(values) else np.nan
        result[f"{label}_LOO_cambio_max"] = float(np.abs(values-original).max()) if len(values) else np.nan
        result[f"{label}_LOO_signo_pct"] = (float(100*np.mean(np.sign(values) == np.sign(original)))
            if len(values) and abs(original) > 1e-12 else np.nan)
        result[f"{label}_LOO_caso_influyente"] = (kept_ids[int(np.argmax(np.abs(values-original)))]
            if len(values) else np.nan)
        result[f"{label}_LOO_evaluaciones_validas"] = len(values)
    if not np.isfinite(spearman):
        result["estabilidad"] = "No identificable: soporte insuficiente o variable constante"
    else:
        result["estabilidad"] = (f"rango LOO rho [{result['Spearman_LOO_min']:.3f}, "
            f"{result['Spearman_LOO_max']:.3f}]; cambio máximo {result['Spearman_LOO_cambio_max']:.3f}")
    return result


def univariate_summary(frame, columns, metadata):
    """Momentos, cuantiles, dispersión robusta y concentración del soporte."""
    result = []
    for col in columns:
        s = frame[col].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
        base = {"variable": col, "nombre": LABELS.get(col, col), **metadata[col],
                "n_efectivo": len(s), "valores_distintos": s.nunique()}
        if s.empty:
            result.append(base)
            continue
        q = s.quantile([.05, .1, .25, .5, .75, .9, .95])
        base.update(media=s.mean(), mediana=s.median(), desviacion=s.std(ddof=1),
            minimo=s.min(), maximo=s.max(), rango=s.max()-s.min(), Q05=q.loc[.05],
            Q10=q.loc[.1], Q25=q.loc[.25], Q75=q.loc[.75], Q90=q.loc[.9], Q95=q.loc[.95],
            IQR=q.loc[.75]-q.loc[.25], MAD=float(np.median(np.abs(s-s.median()))),
            CV_pct=(100*s.std(ddof=1)/abs(s.mean()) if abs(s.mean()) > 1e-12 else np.nan),
            asimetria=(stats.skew(s, bias=False) if len(s) >= 3 and s.nunique() > 1 else np.nan),
            curtosis_excedente=(stats.kurtosis(s, bias=False) if len(s) >= 4 and s.nunique() > 1 else np.nan),
            masa_valor_mas_frecuente_pct=100*s.value_counts(normalize=True).max())
        base["advertencia_CV"] = ("CV energético requiere cautela: referencia de cero y valores con signo pendientes de conciliación"
            if col.startswith(("Dissipated_", "Fracture_event", "Fracture_observed")) else
            "CV descriptivo; requiere escala de razón y media distinta de cero")
        result.append(base)
    return pd.DataFrame(result)


def build_analysis_frame(full, energies):
    """Conserva casos sin objetivo para relaciones secundarias y une por Case_n."""
    if full.Case_n.isna().any() or full.Case_n.duplicated().any():
        raise ValueError("Case_n debe ser completo y único para unir las energías.")
    frame = full.set_index("Case_n").copy()
    metadata = {}
    for col in INPUTS:
        unit = "conteo" if col in ["n_cells_Y", "Pivot_total_number"] else ("mm³" if "volume" in col else "mm")
        metadata[col] = {"rol": "INPUT", "unidad": unit, "formula": "valor fuente",
                         "limitacion": "Geometría previa; dependencia del diseño experimental"}
    for col in [TARGET, "Ultimate_Load", "Maximum_Displacement"] + CYCLE_COLUMNS:
        metadata[col] = {"rol": "OUTPUT", "unidad": "N" if "Load" in col else "mm",
            "formula": "valor fuente", "limitacion": "Posensayo; excluido de INPUT"}
    metadata["Ultimate_Load"]["limitacion"] += "; fuerza del evento terminal/rotura final, no máximo global (docs/trazabilidad_datos.md)"
    for source, prefix in [("Energy_Dissipated", "Dissipated"), ("Energy_Fracture", "Fracture")]:
        energy = energies[source].add_prefix(prefix + "_")
        if not energy.index.equals(frame.index):
            if set(energy.index) != set(frame.index):
                raise ValueError("Los identificadores de energía no coinciden con la tabla de ensayos.")
            energy = energy.reindex(frame.index)
        frame = frame.join(energy, validate="one_to_one")
        for col in energy:
            metadata[col] = {"rol": "OUTPUT", "unidad": "mJ",
                "formula": f"{source} leído sin ejecución de MATLAB",
                "limitacion": "Posensayo; unidad corroborada con tesis PDF pp.117/121 (docs/trazabilidad_datos.md); ausencia de energía no demuestra ausencia de evento/ciclo"}

    derived = {
        "Fiber_area_proxy": (frame.Fiber_height*frame.Fiber_base, "INPUT derivada", "mm²", "Fiber_height * Fiber_base", "Sección rectangular idealizada; no área resistente global"),
        "Fiber_aspect_ratio": (frame.Fiber_height/frame.Fiber_base, "INPUT derivada", "adimensional", "Fiber_height / Fiber_base", "Aspecto local; no aísla un efecto causal de forma"),
        "Pivot_aspect_ratio": (frame.Pivot_height/(2*frame.Pivot_radius), "INPUT derivada", "adimensional", "Pivot_height / (2 * Pivot_radius)", "Esbeltez geométrica nominal; radio constante en el diseño"),
        "Fiber_I_proxy": (frame.Fiber_base*frame.Fiber_height**3/12, "INPUT derivada", "mm⁴", "Fiber_base * Fiber_height^3 / 12", "Momento de área respecto al eje asociado a altura; no rigidez medida ni mecanismo demostrado"),
        "Load_ratio_first_over_ultimate": (frame[TARGET]/frame.Ultimate_Load.where(frame.Ultimate_Load != 0), "OUTPUT derivada", "adimensional", "First_Failure_Load_N / Ultimate_Load", "Cociente primera/rotura final; no factor de seguridad ni reserva de carga"),
        "Load_per_specimen_volume": (frame[TARGET]/frame.Sample_total_volume, "OUTPUT derivada", "N/mm³", "First_Failure_Load_N / Sample_total_volume", "Índice descriptivo de carga por volumen; no esfuerzo ni resistencia material"),
        "Load_per_pivot": (frame[TARGET]/frame.Pivot_total_number, "OUTPUT derivada", "N/pivote", "First_Failure_Load_N / Pivot_total_number", "No presupone reparto uniforme de cargas entre pivotes"),
        "Load_per_cell_Y": (frame[TARGET]/frame.n_cells_Y, "OUTPUT derivada", "N/celda en Y", "First_Failure_Load_N / n_cells_Y", "Normalización por conteo en Y; no fuerza medida en cada celda"),
    }
    fracture = energies["Energy_Fracture"].reindex(frame.index)
    total = fracture.sum(axis=1, min_count=1)
    event_count = fracture.notna().sum(axis=1).astype(float).where(fracture.notna().any(axis=1))
    derived.update({
        "Fracture_observed_sum": (total, "OUTPUT derivada", "mJ", "sum(Energy_Fracture disponible), min_count=1", "Suma observada de longitud variable; no energía total del ensayo"),
        "Fracture_registered_events": (event_count, "OUTPUT derivada", "conteo", "número de valores Energy_Fracture presentes", "Disponibilidad de registro, no conteo verificado de todas las fracturas"),
        "Fracture_initial_share": (fracture.event_1/total.where(total != 0), "OUTPUT derivada", "adimensional", "Energy_Fracture(evento1) / suma registrada", "Comparte numerador y denominador con otras respuestas; sensible a cobertura de eventos"),
    })
    for i, col in enumerate(CYCLE_COLUMNS, 1):
        derived[f"Residual_fraction_cycle{i}"] = (frame[col]/(10*i), "OUTPUT derivada", "adimensional",
            f"{col} / {10*i} mm", "Amplitud nominal del nombre de columna; no deformación unitaria")
    derived["Recovery_complement_cycle6"] = (1-frame[CYCLE_COLUMNS[-1]]/60, "OUTPUT derivada", "adimensional",
        "1 - residual_Displacement_6cycle_60mm / 60 mm", "Proxy complemento nominal; requiere protocolo para interpretar recuperación elástica")
    for col, (values, role, unit, formula, limitation) in derived.items():
        frame[col] = values.replace([np.inf, -np.inf], np.nan)
        metadata[col] = {"rol": role, "unidad": unit, "formula": formula, "limitacion": limitation}
    return frame, metadata


def _algebraic_warning(x, y, metadata):
    """Marca dependencias conocidas; no equipara identidad con mecanismo físico."""
    if x in metadata[y]["formula"] or y in metadata[x]["formula"]:
        return "Dependencia algebraica directa: una variable participa en la definición de la otra"
    if {x,y} == {"Fiber_total_volume","Pivot_total_volume"}:
        return "Volúmenes complementarios por construcción; suma declarada constante"
    if {x,y} == {"Recovery_complement_cycle6","Residual_fraction_cycle6"}:
        return "Complementos algebraicos exactos: suman uno"
    shared_load = [TARGET,"Load_per_specimen_volume","Load_per_pivot","Load_per_cell_Y","Load_ratio_first_over_ultimate"]
    if x in shared_load and y in shared_load:
        return "Las variables comparten la carga de primera falla en su construcción"
    if ((x.startswith("Fracture_event_") and y in ["Fracture_observed_sum","Fracture_initial_share"])
            or (y.startswith("Fracture_event_") and x in ["Fracture_observed_sum","Fracture_initial_share"])
            or {x,y} == {"Fracture_observed_sum","Fracture_initial_share"}):
        return "Dependencia de construcción: evento, suma o fracción comparten energías registradas"
    return "Sin identidad directa predeclarada; evaluar dependencia por arquitectura y protocolo"


def _pair_table(frame, left, right, metadata, *, symmetric=False):
    rows = []
    pairs = itertools.combinations(left, 2) if symmetric else itertools.product(left, right)
    for x, y in pairs:
        row = {"INPUT" if not symmetric else "variable_X": x,
               "OUTPUT" if not symmetric else "variable_Y": y}
        diagnostic = association_diagnostics(frame[x], frame[y], case_ids=frame.index)
        row.update(diagnostic)
        row["dependencia_construccion"] = _algebraic_warning(x,y,metadata)
        rho = diagnostic["Spearman"]
        if np.isfinite(rho):
            row["interpretacion"] = (f"Asociación {'positiva' if rho > 0 else 'negativa' if rho < 0 else 'nula'} "
                f"en rangos; diferencia r−rho={diagnostic['diferencia_Pearson_Spearman']:.3f}; "
                "valorar junto con soporte, empates y sensibilidad")
        else:
            row["interpretacion"] = "Asociación no identificable con la variación y disponibilidad observadas"
        row["limitacion"] = (metadata[x]["limitacion"] + "; " + metadata[y]["limitacion"]
            + "; LOO no es IC ni validación predictiva; no implica causalidad")
        rows.append(row)
    return pd.DataFrame(rows)


def _matrix_pairwise(frame, rows, columns, method):
    values = np.full((len(rows), len(columns)), np.nan)
    ns = np.zeros_like(values, dtype=int)
    for i, a in enumerate(rows):
        for j, b in enumerate(columns):
            x, y, _ = _finite_pair(frame[a], frame[b])
            ns[i, j] = len(x)
            if len(x) >= 3 and np.ptp(x) > 0 and np.ptp(y) > 0:
                if method == "spearman":
                    x, y = stats.rankdata(x), stats.rankdata(y)
                values[i, j] = safe_correlation(x, y)
    return (pd.DataFrame(values, index=rows, columns=columns),
            pd.DataFrame(ns, index=rows, columns=columns))


def _heatmap(matrix, title, *, figsize=None):
    fig, ax = plt.subplots(figsize=figsize or (max(9, .72*len(matrix.columns)), max(5, .48*len(matrix))))
    im = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), [LABELS.get(c, c) for c in matrix.columns], rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix)), [LABELS.get(c, c) for c in matrix.index], fontsize=8)
    ax.set_title(title + "\nNaN: constante o sin soporte; tamaños efectivos en la tabla técnica", fontsize=12)
    if len(matrix)*len(matrix.columns) <= 240:
        for i in range(len(matrix)):
            for j in range(len(matrix.columns)):
                v = matrix.iloc[i,j]
                ax.text(j, i, "—" if pd.isna(v) else f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(v) > .55 else "#152536")
    fig.colorbar(im, ax=ax, fraction=.028, pad=.03, label="Coeficiente descriptivo")
    return fig


def run_analysis(root):
    """Ejecuta y exporta; devuelve tablas, figuras, hallazgos y manifiesto.

    ``tables`` contiene DataFrames; ``figures`` metadatos con rutas relativas a
    la raíz; ``findings`` una lista serializable. Las fuentes son inmutables.
    """
    root = Path(root).resolve()
    source = root/"data/raw/dati_campagna_venditti.m"
    interim = root/"data/interim/preliminary/dataset_interim.csv"
    inputs = {p.relative_to(root).as_posix(): sha256_file(p) for p in [source, interim]}
    frame, metadata = build_analysis_frame(pd.read_csv(interim), load_energy_tables(source))
    outputs = [c for c, item in metadata.items() if item["rol"].startswith("OUTPUT")]
    out = root/"reports/eda_relaciones"
    tab, figdir = out/"tables", out/"figures"
    tab.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    tables, figures = {}, {}

    def save_table(name, table, *, index=False):
        tables[name] = table.copy()
        table.to_csv(tab/f"{name}.csv", index=index, encoding="utf-8")

    def save_fig(name, fig, caption):
        fig.tight_layout()
        for extension in ["png", "svg"]:
            fig.set_facecolor("white")
            save_figure_atomic(fig, figdir/f"{name}.{extension}", dpi=300)
        plt.close(fig)
        figures[name] = {"png": (figdir/f"{name}.png").relative_to(root).as_posix(),
                        "svg": (figdir/f"{name}.svg").relative_to(root).as_posix(), "caption": caption}

    save_table("01_definiciones", pd.DataFrame([{"variable": k, "nombre": LABELS.get(k,k), **v} for k,v in metadata.items()]))
    save_table("02_univariado_completo", univariate_summary(frame, list(metadata), metadata))
    family_rows = []
    for col in ["n_cells_Y", "Fiber_height", "Pivot_height"]:
        frequencies = frame[col].value_counts(normalize=True).sort_index()
        for val, proportion in frequencies.items():
            family_rows.append({"variable": col, "nivel": val, "porcentaje": 100*proportion,
                "entropia_nats_variable": float(stats.entropy(frequencies)),
                "diversidad_efectiva_variable": float(np.exp(stats.entropy(frequencies)))})
    save_table("03_cobertura_niveles", pd.DataFrame(family_rows))
    save_table("04_input_output_asociaciones", _pair_table(frame, INPUTS, outputs, metadata))
    save_table("05_input_input_asociaciones", _pair_table(frame, INPUTS, INPUTS, metadata, symmetric=True))
    save_table("06_output_output_asociaciones", _pair_table(frame, outputs, outputs, metadata, symmetric=True))

    focus = [TARGET, "Ultimate_Load", "Maximum_Displacement", CYCLE_COLUMNS[0], CYCLE_COLUMNS[-1],
             "Dissipated_cycle_1", "Dissipated_cycle_5", "Dissipated_cycle_6", "Fracture_event_1",
             "Fracture_observed_sum", "Fracture_registered_events", "Fracture_initial_share"]
    for left, right, tag in [(INPUTS, INPUTS, "input_input"), (INPUTS, outputs, "input_output"),
                             (outputs, outputs, "output_output")]:
        for method in ["pearson", "spearman"]:
            corr, ns = _matrix_pairwise(frame, left, right, method)
            corr.index.name = "variable"
            ns.index.name = "variable"
            save_table(f"07_{tag}_{method}", corr, index=True)
            if method == "pearson":
                save_table(f"07_{tag}_n_efectivo", ns, index=True)
            plot = corr if tag == "input_input" else (corr[focus] if tag == "input_output" else corr.loc[focus, focus])
            save_fig(f"R01_{tag}_{method}", _heatmap(plot, f"{tag.replace('_', '–').upper()} · {method.title()}"),
                "Matriz por parejas completas. Las tablas CSV incluyen todas las variables; esta figura usa respuestas representativas predefinidas." if tag != "input_input" else
                "Matriz completa de geometrías. Un coeficiente perfecto puede reflejar una identidad o el diseño; no identifica un efecto mecánico independiente.")

    io = tables["04_input_output_asociaciones"]
    target_assoc = io.loc[io.OUTPUT.eq(TARGET)].copy()
    save_table("08_objetivo_priorizado", target_assoc)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for method, offset, color, marker in [("Pearson",-.16,"#2466a2","o"),("Spearman",0,"#078377","s"),("Kendall_tau_b",.16,"#c9872a","^")]:
        axes[0].scatter(target_assoc[method], np.arange(len(target_assoc))+offset, color=color,marker=marker,label=method.replace("_tau_b"," tau-b"),s=24)
    for i, row in enumerate(target_assoc.itertuples(index=False)):
        axes[1].plot([row.Spearman_LOO_min,row.Spearman_LOO_max], [i,i], lw=3, color="#2466a2")
        axes[1].scatter(row.Spearman, i, color="#c9872a", zorder=3)
    for ax in axes:
        ax.axvline(0, color="gray", ls="--", lw=.7)
        ax.set_yticks(range(len(target_assoc)), [LABELS[c] for c in target_assoc.INPUT], fontsize=8)
        ax.set_xlim(-1.05,1.05)
    axes[0].set(title="Pearson, Spearman y Kendall", xlabel="Coeficiente; valores exactos en tabla")
    axes[0].legend(fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.16),ncol=3)
    axes[1].set(title="Sensibilidad Spearman al omitir un caso", xlabel="Rango LOO; punto = rho original")
    save_fig("R02_coeficientes_estabilidad", fig, "Los rangos LOO miden sensibilidad a omisión, no incertidumbre inferencial ni validación predictiva. Orden de variables por familia física, sin ranking automático.")

    # Volumen declarado y forma local: exterior fijo en la fuente documental.
    size_shape = []
    for col in ["Sample_total_volume", "Fiber_area_proxy", "Fiber_aspect_ratio", "Pivot_aspect_ratio", "Fiber_I_proxy"]:
        d = frame[[col,TARGET,"n_cells_Y"]].dropna()
        for family, s in [("global",d)] + [(str(int(g)), s) for g,s in d.groupby("n_cells_Y")]:
            size_shape.append({"variable":col,"familia":family,**association_diagnostics(s[col],s[TARGET],case_ids=s.index),
                "limitacion":"Familias de discretización del retículo, no tamaños exteriores; contraste no separa cambios conjuntos del diseño ni identifica causalidad"})
    save_table("09_tamano_forma", pd.DataFrame(size_shape))
    fig, axes = plt.subplots(1,3,figsize=(14,4.5))
    for ax,col in zip(axes,["Sample_total_volume","Fiber_area_proxy","Fiber_aspect_ratio"]):
        for g,s in frame.groupby("n_cells_Y"):
            ax.scatter(s[col],s[TARGET],label=f"Y={int(g)}",color=COLORS[int(g)],alpha=.85)
        ax.set(xlabel=f"{LABELS[col]} [{metadata[col]['unidad']}]", ylabel="Primera falla [N]",title=LABELS[col])
    axes[0].legend(fontsize=8)
    save_fig("R03_tamano_forma",fig,"Contraste de volumen declarado, sección idealizada y proporción local. Las dimensiones exteriores son fijas en la tesis; n_cells_Y expresa discretización de arquitectura, no crecimiento exterior. Se analizan valores del archivo MATLAB recibido, pendientes de conciliación geométrica con la tesis.")

    redundancy = []
    component_sum = frame.Fiber_total_volume + frame.Pivot_total_volume
    redundancy.append({"relacion":"Fiber_total_volume + Pivot_total_volume", "evidencia":f"rango de suma [{component_sum.min():.8g}, {component_sum.max():.8g}] mm³",
        "residuo_max":float(np.abs(component_sum-component_sum.median()).max()),"implicacion":"Identidad del volumen declarado de componentes; no equivale a Sample_total_volume"})
    for col in ["Fiber_total_length","Pivot_total_number"]:
        redundancy.append({"relacion":f"n_cells_Y → {col}","evidencia":"Cada familia tiene un valor único" if frame.groupby("n_cells_Y")[col].nunique().eq(1).all() else "Hay variación interna",
            "residuo_max":float(frame.groupby("n_cells_Y")[col].apply(lambda s: np.ptp(s)).max()),"implicacion":"Codificación del mismo eje de configuración; coeficientes separados no son efectos independientes"})
    save_table("10_redundancias",pd.DataFrame(redundancy))

    # Ultimate_Load es fuerza del evento final; no se impone máximo global.
    secondary=frame[[TARGET,"Ultimate_Load","n_cells_Y","Load_ratio_first_over_ultimate"]].dropna()
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for g,s in secondary.groupby("n_cells_Y"):
        axes[0].scatter(s.Ultimate_Load,s[TARGET],color=COLORS[int(g)],label=f"Y={int(g)}")
        axes[1].scatter(s.Ultimate_Load,s.Load_ratio_first_over_ultimate,color=COLORS[int(g)])
    limits=[secondary[[TARGET,"Ultimate_Load"]].min().min(),secondary[[TARGET,"Ultimate_Load"]].max().max()]
    axes[0].plot(limits,limits,"--",color="gray",label="Igualdad numérica")
    axes[0].set(xlabel="Carga de rotura final (Ultimate_Load) [N]",ylabel="Primera falla [N]",title="Dos eventos distintos del ensayo"); axes[0].legend(fontsize=8)
    axes[1].axhline(1,color="gray",ls="--"); axes[1].set(xlabel="Carga de rotura final [N]",ylabel="Primera falla / carga de rotura final",title="Cociente descriptivo; no factor de seguridad")
    save_fig("R04_comparacion_cargas",fig,"La fuerza de rotura final no coincide necesariamente con la máxima del ensayo. Definición cotejada con fichas de la tesis (docs/trazabilidad_datos.md); el cociente no mide reserva estructural.")
    load_rows=[]
    for family,s in [("global",secondary)]+[(str(int(g)),s) for g,s in secondary.groupby("n_cells_Y")]:
        load_rows.append({"familia":family,"mediana_cociente":s.Load_ratio_first_over_ultimate.median(),
            "primera_mayor_Ultimate_pct":100*s[TARGET].gt(s.Ultimate_Load).mean(),
            **association_diagnostics(s[TARGET],s.Ultimate_Load,case_ids=s.index)})
    save_table("11_cargas_por_familia",pd.DataFrame(load_rows))
    grouped_responses=[]
    for output in focus[1:]:
        for family,s in frame.groupby("n_cells_Y"):
            grouped_responses.append({"variable_X":TARGET,"variable_Y":output,"familia":int(family),
                **association_diagnostics(s[TARGET],s[output],case_ids=s.index),
                "limitacion":"Asociación dentro de arquitectura; soporte efectivo variable, sin contraste inferencial. " + metadata[output]["limitacion"]})
    save_table("15_respuestas_por_familia",pd.DataFrame(grouped_responses))

    # Comparación pareada por amplitud, con la MISMA cohorte en todos los ciclos.
    common=frame.dropna(subset=CYCLE_COLUMNS)
    cyclic=[]
    for family,s in [("global",common)]+[(str(int(g)),s) for g,s in common.groupby("n_cells_Y")]:
        for i,col in enumerate(CYCLE_COLUMNS,1):
            fraction=s[col]/(10*i)
            cyclic.append({"familia":family,"ciclo":i,"amplitud_nominal_mm":10*i,"n_efectivo":len(s),
                "mediana_residual_mm":s[col].median(),"mediana_fraccion_residual":fraction.median(),
                "Q1_fraccion_residual":fraction.quantile(.25),"Q3_fraccion_residual":fraction.quantile(.75),
                "mediana_complemento_nominal":(1-fraction).median()})
    save_table("12_ciclos_cohorte_comun",pd.DataFrame(cyclic))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for g in sorted(common.n_cells_Y.unique()):
        s=tables["12_ciclos_cohorte_comun"].loc[lambda t:t.familia.eq(str(int(g)))]
        axes[0].plot(s.ciclo,s.mediana_fraccion_residual,"o-",color=COLORS[int(g)],label=f"Y={int(g)}")
        axes[1].plot(s.ciclo,s.mediana_complemento_nominal,"o-",color=COLORS[int(g)])
    axes[0].set(xlabel="Ciclo registrado",ylabel="Mediana residual / amplitud",title="Deformación residual nominal relativa"); axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Ciclo registrado",ylabel="Mediana 1 − residual / amplitud",title="Complemento nominal (proxy de recuperación)")
    save_fig("R05_residual_recuperacion",fig,"Mismos casos a lo largo de los ciclos. Las dos magnitudes son complementos algebraicos, no evidencias independientes; no se reconstruyen curvas ni histéresis.")

    normalized=[]
    for col in [TARGET,"Load_per_specimen_volume","Load_per_pivot","Load_per_cell_Y"]:
        for g,s in frame.groupby("n_cells_Y"):
            values=s[col].dropna()
            normalized.append({"variable":col,"familia":int(g),"n_efectivo":len(values),"unidad":metadata[col]["unidad"],
                "mediana":values.median(),"Q1":values.quantile(.25),"Q3":values.quantile(.75),
                "limitacion":metadata[col]["limitacion"]})
    save_table("13_normalizaciones_carga",pd.DataFrame(normalized))
    fig,axes=plt.subplots(1,4,figsize=(15,4))
    for ax,col in zip(axes,[TARGET,"Load_per_specimen_volume","Load_per_pivot","Load_per_cell_Y"]):
        s=tables["13_normalizaciones_carga"].loc[lambda t:t.variable.eq(col)]
        ax.plot(s.familia,s.mediana,"o-",color="#078377")
        ax.fill_between(s.familia,s.Q1,s.Q3,alpha=.18,color="#078377")
        ax.set(title=LABELS[col],xlabel="Celdas en Y",ylabel=metadata[col]["unidad"],xticks=sorted(s.familia))
    save_fig("R06_normalizaciones_carga",fig,"Medianas e IQR por familia: normalizar responde a otra pregunta. N/mm³ no es esfuerzo y N/pivote no es reparto interno medido.")

    target_idx=target_assoc.set_index("INPUT")
    volume_by=tables["09_tamano_forma"].query("variable == 'Sample_total_volume'").set_index("familia")
    energy_diag=association_diagnostics(frame.Fracture_event_1,frame[TARGET],case_ids=frame.index)
    cycles_table=tables["12_ciclos_cohorte_comun"].query("familia == 'global'").set_index("ciclo")
    per_pivot=tables["13_normalizaciones_carga"].query("variable == 'Load_per_pivot'").set_index("familia")
    findings=[
        {"id":"R-H01","pregunta":"¿La asociación del volumen es uniforme entre familias?",
         "metodo":"Pearson/Spearman/Kendall globales, por familia y omisión de caso",
         "resultado":f"Volumen–primera falla: r={target_idx.loc['Sample_total_volume','Pearson']:.4f}, rho={target_idx.loc['Sample_total_volume','Spearman']:.4f}; rho por familia " + ", ".join(f"Y={g}: {volume_by.loc[g,'Spearman']:.4f}" for g in volume_by.index if g!='global'),
         "interpretacion":"La arquitectura confunde la asociación global; el signo no puede atribuirse a volumen aislado.",
         "limitacion":"Correlaciones descriptivas internas; no identificación causal ni evidencia de generalización.",
         "tabla":"09_tamano_forma","figura":"R03_tamano_forma"},
        {"id":"R-H02","pregunta":"¿Todas las cargas registradas representan el mismo evento?",
         "metodo":"Diagrama de igualdad y cociente observado por familia",
         "resultado":f"Mediana de primera falla / Ultimate_Load={secondary.Load_ratio_first_over_ultimate.median():.4f}; porcentaje por encima de igualdad={100*secondary[TARGET].gt(secondary.Ultimate_Load).mean():.2f}%.",
         "interpretacion":"Ultimate_Load registra fuerza de rotura final, evento distinto de primera rotura y del máximo global de fuerza.",
         "limitacion":"No interpretar el cociente como capacidad posterior o seguridad; no hay series digitales fuerza–desplazamiento en el MATLAB para estimar la evolución completa.",
         "tabla":"11_cargas_por_familia","figura":"R04_comparacion_cargas"},
        {"id":"R-H03","pregunta":"¿Cómo cambia el residual respecto de la amplitud?",
         "metodo":"Medianas de residual/amplitud nominal en cohorte completa y por familia",
         "resultado":f"Mediana global de residual/amplitud: {cycles_table.loc[1,'mediana_fraccion_residual']:.4f} en ciclo 1 y {cycles_table.loc[6,'mediana_fraccion_residual']:.4f} en ciclo 6.",
         "interpretacion":"El complemento nominal disminuye porque se define como uno menos la fracción residual.",
         "limitacion":"No es observación independiente de recuperación ni ley de fatiga; hay figuras de curvas en la tesis, pero no series digitales fuerza–desplazamiento en el MATLAB.",
         "tabla":"12_ciclos_cohorte_comun","figura":"R05_residual_recuperacion"},
        {"id":"R-H04","pregunta":"¿El evento energético inicial acompaña a la primera falla?",
         "metodo":"Correlaciones entre respuestas, tamaño efectivo y sensibilidad LOO",
         "resultado":f"Evento energético inicial–primera falla: r={energy_diag['Pearson']:.4f}, rho={energy_diag['Spearman']:.4f}, tau-b={energy_diag['Kendall_tau_b']:.4f}; rho LOO [{energy_diag['Spearman_LOO_min']:.4f}, {energy_diag['Spearman_LOO_max']:.4f}].",
         "interpretacion":"Ambas respuestas del ensayo covarían de forma intensa en el dominio observado.",
         "limitacion":"La energía se conoce después del ensayo; unidad mJ corroborada documentalmente. No es INPUT previo ni prueba de ley mecánica; geometría del MATLAB pendiente de conciliación con tesis.",
         "tabla":"06_output_output_asociaciones","figura":"R01_output_output_spearman"},
        {"id":"R-H05","pregunta":"¿Más columnas geométricas equivalen a más información independiente?",
         "metodo":"Verificación de identidades declaradas y mapeos dentro de familia",
         "resultado":f"La suma declarada de volúmenes de fibra y pivote tiene rango [{component_sum.min():.2f}, {component_sum.max():.2f}] mm³.",
         "interpretacion":"Su anticorrelación está introducida por construcción; longitud total y conteo de pivotes también codifican familia.",
         "limitacion":"No equivale a volumen neto o masa constantes; las representaciones deben compararse dentro del protocolo predictivo futuro.",
         "tabla":"10_redundancias","figura":"R01_input_input_spearman"},
        {"id":"R-H06","pregunta":"¿Los contrastes de arquitectura dependen de la normalización elegida?",
         "metodo":"Medianas e IQR de fuerza absoluta, por volumen y por conteos",
         "resultado":"Mediana de primera falla / pivotes [N/pivote]: " + ", ".join(f"Y={g}: {per_pivot.loc[g,'mediana']:.4f}" for g in per_pivot.index) + ".",
         "interpretacion":"La diferencia entre arquitecturas Y=4 y Y=5 se atenúa al normalizar por pivotes; Y=6 mantiene una mediana mayor bajo esta representación.",
         "limitacion":"El denominador cambia entre arquitecturas; no prueba reparto uniforme, causalidad ni efecto independiente de densidad. Dimensiones exteriores fijas en la tesis.",
         "tabla":"13_normalizaciones_carga","figura":"R06_normalizaciones_carga"},
    ]
    save_table("14_hallazgos",pd.DataFrame(findings))
    write_bytes_atomic(out/"hallazgos.json", (json.dumps(findings,ensure_ascii=False,indent=2)+"\n").encode("utf-8"))
    assert inputs == {p.relative_to(root).as_posix():sha256_file(p) for p in [source,interim]}, "Se alteró una fuente durante el análisis"
    manifest={"status":"executed","purpose":"Relaciones exploratorias completas y sensibilidad",
        "inputs_sha256":inputs,"analysis_source_sha256":sha256_file(Path(__file__)),"execution":execution_context(),"tables":list(tables),"figures":figures,
        "method":{"pairwise_complete":True,"minimum_for_correlation":3,"minimum_for_LOO":4,
                  "LOO_is_confidence_interval":False,"pvalues_computed":False,
                  "resampling":"ninguno; sensibilidad determinista por caso","energy_unit":"mJ; ver docs/trazabilidad_datos.md"},
        "artifact_sha256":{p.relative_to(out).as_posix():sha256_file(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}}
    write_bytes_atomic(out/"manifest.json", (json.dumps(manifest,ensure_ascii=False,indent=2)+"\n").encode("utf-8"))
    return {"tables":tables,"figures":figures,"findings":findings,"manifest":manifest}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    run_analysis(args.root)
    print("Relaciones ejecutadas: reports/eda_relaciones/ (CSV, PNG 300 dpi, SVG y manifiesto).")
