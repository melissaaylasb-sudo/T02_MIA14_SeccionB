"""Auditoría independiente de datos, particiones y resultados del ajuste anidado.

Las métricas se reconstruyen a partir de predicciones reservadas; la auditoría
no usa esas métricas para reajustar modelos ni para modificar la búsqueda.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .experimentos_fe import metrics, save_json
from .feature_engineering import GEOMETRY, TARGET, nominal_groups
from .verificar_fe import auditar_preprocesamiento

ROOT = Path(__file__).resolve().parents[1]
DATA_SHA = "2d0a17f54bdc6a6361aaa6e25a57cf67775c7d47a5e2adefb408f114defa21f8"
SPLIT_SHA = "763f20193cd8c58beeb471bc2024015588068a48a1cc0540c0b10ee25ff61e59"
MODEL_NAMES = ("Ridge", "ElasticNet", "SVR_RBF", "RandomForest", "ExtraTrees",
               "GaussianProcess", "GradientBoosting")
BASELINE = "Baseline_Ridge"
SELECTOR = "Selector_interno"


class Checks:
    """Acumula comprobaciones legibles y falla inmediatamente ante incoherencias."""

    def __init__(self):
        self.labels = []
        self.counts = {}

    def require(self, condition, label):
        if not bool(condition):
            raise AssertionError("Ajuste anidado: " + label)
        if label not in self.counts:
            self.labels.append(label)
        self.counts[label] = self.counts.get(label, 0) + 1

    def close(self, actual, expected, label):
        self.require(np.allclose(actual, expected, rtol=1e-9, atol=1e-9,
                                 equal_nan=False), label)


def audit_predictions(data, outer, predictions, folds, summary, reference, checks=None):
    """Recalcula todos los errores y contrasta la referencia sin ajustar regresores."""
    ck = Checks() if checks is None else checks
    models = {BASELINE, SELECTOR, *MODEL_NAMES}
    protocols = {"configuracion", "familia"}
    by_case = data.set_index("Case_n")
    cases = set(by_case.index)
    split_map = {(r["protocolo"], r["fold"]): r for r in outer}
    pairs = {(p, m) for p in protocols for m in models}
    triples = {(p, m, f) for p, f in split_map for m in models}
    ck.require(set(predictions.modelo) == models, "baseline, siete regresores y selector presentes")
    ck.require(set(predictions.protocolo) == protocols, "ambos protocolos externos presentes")
    ck.require(not predictions.duplicated(["protocolo", "modelo", "Case_n"]).any(),
               "una predicción reservada por caso, modelo y protocolo")
    ck.require(set(folds[["protocolo", "modelo", "fold"]].itertuples(index=False, name=None)) == triples
               and not folds.duplicated(["protocolo", "modelo", "fold"]).any(),
               "todas las métricas por partición externa, sin duplicados")
    ck.require(set(summary[["protocolo", "modelo"]].itertuples(index=False, name=None)) == pairs
               and not summary.duplicated(["protocolo", "modelo"]).any(),
               "resumen completo y único")
    ck.require(np.isfinite(predictions[["observado_N", "predicho_N", "residuo_N", "error_absoluto_N"]]).all().all(),
               "predicciones y errores finitos")
    sf = folds.set_index(["protocolo", "modelo", "fold"])
    ss = summary.set_index(["protocolo", "modelo"])
    for (protocol, model), pred in predictions.groupby(["protocolo", "modelo"]):
        label = f"{protocol}/{model}"
        ck.require(set(pred.Case_n) == cases, "cobertura de todos los casos: " + label)
        ck.close(pred.observado_N, by_case.loc[pred.Case_n, TARGET], "objetivo original conservado: " + label)
        ck.close(pred.residuo_N, pred.observado_N - pred.predicho_N, "residuos recalculados: " + label)
        ck.close(pred.error_absoluto_N, pred.residuo_N.abs(), "errores absolutos recalculados: " + label)
        for key, value in metrics(pred.observado_N, pred.predicho_N).items():
            ck.close(ss.loc[(protocol, model), key], value, f"métrica OOF {key}: " + label)
        for fold, sub in pred.groupby("fold"):
            record = split_map[(protocol, int(fold))]
            ck.require(set(sub.Case_n) == set(record["test_cases"]),
                       f"casos reservados exactos: {label}/{fold}")
            frow = sf.loc[(protocol, model, fold)]
            ck.close(frow.MAE_N, sub.error_absoluto_N.mean(), f"MAE del fold: {label}/{fold}")
            ck.close(frow.RMSE_N, np.sqrt(np.mean(sub.residuo_N ** 2)), f"RMSE del fold: {label}/{fold}")
    old = reference[reference.experimento == "A"].set_index(["protocolo", "Case_n"]).sort_index()
    new = predictions[predictions.modelo == BASELINE].set_index(["protocolo", "Case_n"]).sort_index()
    ck.require(new.index.equals(old.index), "baseline evalúa los mismos casos históricos")
    ck.close(new.predicho_N, old.predicho_N, "baseline reproduce predicciones de Semana 5")
    ck.close(new.fold, old.fold, "baseline reproduce folds de Semana 5")
    return ck


def audit_outer(data, outer, checks=None):
    ck = Checks() if checks is None else checks
    by_case = data.set_index("Case_n")
    groups = {
        "configuracion": nominal_groups(by_case, ["n_cells_Y", "Fiber_height", "Pivot_height"]),
        "familia": by_case.n_cells_Y.astype(int).astype(str),
    }
    seen = {key: [] for key in groups}
    keys = [(r["protocolo"], r["fold"]) for r in outer]
    ck.require(len(keys) == len(set(keys)), "particiones externas únicas")
    for row in outer:
        protocol, fold = row["protocolo"], row["fold"]
        train, test = row["train_cases"], row["test_cases"]
        label = f"{protocol}/{fold}"
        ck.require(bool(train) and bool(test) and len(train) == len(set(train)) and len(test) == len(set(test)),
                   "casos externos únicos y no vacíos: " + label)
        ck.require(set(train).isdisjoint(test) and set(train) | set(test) == set(by_case.index),
                   "partición externa completa y disjunta: " + label)
        ck.require(set(groups[protocol].loc[train]).isdisjoint(groups[protocol].loc[test]),
                   "grupos externos disjuntos: " + label)
        ck.require(len(set(groups[protocol].loc[test])) == 1,
                   "un grupo externo completo reservado: " + label)
        seen[protocol].extend(test)
    for protocol, ids in seen.items():
        ck.require(sorted(ids) == sorted(by_case.index), "cobertura OOF externa única: " + protocol)
    return ck


def audit_train(data, audits, outer, checks=None):
    ck = Checks() if checks is None else checks
    outer_map = {(r["protocolo"], r["fold"]): r for r in outer}
    # El selector reutiliza el ajuste elegido: no se entrena una novena instancia.
    models = {BASELINE, *MODEL_NAMES}
    keys = [(r["protocolo"], r["modelo"], r["fold"]) for r in audits]
    expected = {(p, m, f) for p, f in outer_map for m in models}
    ck.require(len(keys) == len(set(keys)) and set(keys) == expected,
               "auditoría de cada ajuste final externo, sin duplicados")
    for model in models:
        rows = [r for r in audits if r["modelo"] == model]
        for row in rows:
            split = outer_map[(row["protocolo"], row["fold"])]
            ck.require(row["train_cases"] == split["train_cases"] and row["test_cases"] == split["test_cases"],
                       f"casos auditados exclusivos del train externo: {model}/{row['protocolo']}/{row['fold']}")
            ck.require(row["experimento"] == "A" and row["imputer_columns"] == list(GEOMETRY),
                       f"geometría original sin respuestas ni identificadores: {model}/{row['protocolo']}/{row['fold']}")
            if model == "GaussianProcess":
                target = data.set_index("Case_n").loc[row["train_cases"], TARGET]
                ck.close(row["target_train_mean"], target.mean(),
                         f"normalización GPR: media del target solo en train {row['protocolo']}/{row['fold']}")
                std = float(target.std(ddof=0))
                ck.close(row["target_train_std"], std if std else 1.0,
                         f"normalización GPR: escala del target solo en train {row['protocolo']}/{row['fold']}")
        checked = auditar_preprocesamiento(data, rows)
        ck.require(len(checked) == len(outer), "medianas, filtro y escala reconstruidos solo con train: " + model)
    return ck


def audit_inner(data, outer, inner, checks=None):
    """Comprueba IDs globales, cobertura y la separación grupal en ambos niveles."""
    ck = Checks() if checks is None else checks
    by_case = data.set_index("Case_n")
    groups = {
        "configuracion": nominal_groups(by_case, ["n_cells_Y", "Fiber_height", "Pivot_height"]),
        "familia": by_case.n_cells_Y.astype(int).astype(str),
    }
    external = {(r["protocolo"], r["fold"]): r for r in outer}
    keys = [(r["protocolo"], r["fold"], r["inner_fold"]) for r in inner]
    ck.require(len(keys) == len(set(keys)), "particiones internas únicas")
    ck.require({key[:2] for key in keys} == set(external), "todos los folds externos tienen validación interna")
    for key, record in external.items():
        rows = sorted((r for r in inner if (r["protocolo"], r["fold"]) == key), key=lambda r: r["inner_fold"])
        expected_count = 5 if key[0] == "configuracion" else len(set(groups["familia"].loc[record["train_cases"]]))
        ck.require([r["inner_fold"] for r in rows] == list(range(1, expected_count + 1)),
                   f"número y orden interno predeclarado: {key}")
        seen = []
        for row in rows:
            train, valid = row["train_cases"], row["validation_cases"]
            label = f"{key}/{row['inner_fold']}"
            ck.require(bool(train) and bool(valid) and len(train) == len(set(train)) and len(valid) == len(set(valid)),
                       "casos internos únicos y no vacíos: " + label)
            ck.require(set(train).isdisjoint(valid) and set(train) | set(valid) == set(record["train_cases"]),
                       "partición interna completa dentro del train externo: " + label)
            ck.require(row["outer_test_cases"] == record["test_cases"] and
                       (set(train) | set(valid)).isdisjoint(row["outer_test_cases"]),
                       "ningún caso externo participa del ajuste interno: " + label)
            ck.require(set(groups[key[0]].loc[train]).isdisjoint(groups[key[0]].loc[valid]),
                       "grupos internos disjuntos: " + label)
            seen.extend(valid)
        ck.require(sorted(seen) == sorted(record["train_cases"]),
                   f"cada caso train externo se valida internamente una vez: {key}")
    return ck


def audit_search(searches, selections, inner, predictions, folds, checks=None):
    """Recalcula medias internas y selección; ningún error externo decide el ganador."""
    ck = Checks() if checks is None else checks
    inside = {}
    for row in inner:
        inside.setdefault((row["protocolo"], row["fold"]), []).append(row)
    expected = {(p, m, f) for p, f in inside for m in MODEL_NAMES}
    ck.require(set(searches[["protocolo", "modelo", "fold"]].itertuples(index=False, name=None)) == expected,
               "rejillas completas para siete modelos en todas las reservas externas")
    ck.require(not searches.duplicated(["protocolo", "modelo", "fold", "candidate_id"]).any(),
               "candidatos de búsqueda sin duplicados")
    ck.require(np.isfinite(searches[["mean_inner_MAE_N", "std_inner_MAE_N"]]).all().all(),
               "métricas internas finitas")
    ck.require(searches.mean_inner_MAE_N.ge(0).all(), "errores internos no negativos")
    winner_scores, selected_params = {}, {}
    for (protocol, model, fold), rows in searches.groupby(["protocolo", "modelo", "fold"], sort=False):
        label = f"{protocol}/{model}/{fold}"
        rows = rows.sort_values("candidate_id")
        ck.require(rows.candidate_id.tolist() == list(range(len(rows))), "rejilla enumerada sin huecos: " + label)
        split_columns = [f"split{i}_MAE_N" for i in range(len(inside[(protocol, fold)]))]
        ck.require(np.isfinite(rows[split_columns]).all().all(), "scores de todos los folds internos: " + label)
        ck.close(rows.mean_inner_MAE_N, rows[split_columns].mean(axis=1), "media interna recalculada: " + label)
        ck.close(rows.std_inner_MAE_N, rows[split_columns].std(axis=1, ddof=0), "dispersión interna recalculada: " + label)
        expected_rank = rows.mean_inner_MAE_N.rank(method="min").astype(int)
        ck.close(rows["rank"], expected_rank, "ranking interno recalculado: " + label)
        selected = rows[rows.selected.eq(True)]
        ck.require(len(selected) == 1, "un candidato elegido por modelo y fold: " + label)
        chosen = selected.iloc[0]
        best_pos = int(np.argmin(rows.mean_inner_MAE_N.to_numpy()))
        ck.require(chosen.candidate_id == rows.iloc[best_pos].candidate_id,
                   "candidato elegido por mínimo MAE interno: " + label)
        winner_scores.setdefault((protocol, fold), {})[model] = -float(chosen.mean_inner_MAE_N)
        selected_params[(protocol, model, fold)] = json.loads(chosen.params_json)
        frow = folds[(folds.protocolo == protocol) & (folds.modelo == model) & (folds.fold == fold)].iloc[0]
        ck.close(frow.inner_MAE_N, chosen.mean_inner_MAE_N, "score interno del modelo publicado: " + label)
        ck.require(json.loads(frow.params_json) == json.loads(chosen.params_json),
                   "parámetros publicados coinciden con búsqueda interna: " + label)
    keys = [(r["protocolo"], r["fold"]) for r in selections]
    ck.require(len(keys) == len(set(keys)) and set(keys) == set(inside),
               "decisión del selector registrada en todos los folds")
    for row in selections:
        key = (row["protocolo"], row["fold"])
        scores = winner_scores[key]
        ck.require(set(row["scores_internos_neg_MAE"]) == set(MODEL_NAMES),
                   f"selector compara los siete scores internos: {key}")
        ck.close([row["scores_internos_neg_MAE"][m] for m in MODEL_NAMES],
                 [scores[m] for m in MODEL_NAMES], f"scores del selector reconstruidos desde rejillas: {key}")
        winner = max(MODEL_NAMES, key=lambda name: scores[name])
        ck.require(row["modelo_seleccionado"] == winner, f"selector decidido solo por validación interna: {key}")
        chosen = predictions[(predictions.protocolo == key[0]) & (predictions.fold == key[1]) &
                             (predictions.modelo == winner)].sort_values("Case_n")
        result = predictions[(predictions.protocolo == key[0]) & (predictions.fold == key[1]) &
                             (predictions.modelo == SELECTOR)].sort_values("Case_n")
        ck.require(result.modelo_seleccionado.eq(winner).all(), f"modelo seleccionado identificado en predicción: {key}")
        ck.close(result.predicho_N, chosen.predicho_N, f"selector reutiliza predicción del modelo elegido: {key}")
        ck.require(sorted(result.Case_n) == sorted(row["test_cases"]), f"selector mantiene la reserva externa: {key}")
    return ck


def audit_execution_logs(output, outer, inner, searches, checks=None):
    """Cruza cada ajuste real y cada score con los casos y candidatos autorizados."""
    ck = Checks() if checks is None else checks
    fits, scores = {}, []
    for path in sorted(Path(output).glob("logs_ajustes/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["evento"] == "fit":
                ck.require(event["fit_id"] not in fits, "identificador único de ajuste registrado")
                fits[event["fit_id"]] = event
            elif event["evento"] == "score_interno":
                scores.append(event)
            else:
                raise AssertionError("Evento de ajuste desconocido: " + event["evento"])
    outer_map = {(r["protocolo"], r["fold"]): r for r in outer}
    inner_map = {}
    inner_pair_map = {}
    for row in inner:
        inner_map.setdefault((row["protocolo"], row["fold"]), []).append(row)
        inner_pair_map[(row["protocolo"], row["fold"], tuple(row["train_cases"]), tuple(row["validation_cases"]))] = row
    search_map, grid_keys, selected_rows = {}, {}, {}
    for candidate in searches.to_dict(orient="records"):
        key = (candidate["protocolo"], candidate["fold"], candidate["modelo"])
        params = json.loads(candidate["params_json"])
        grid_keys[key] = sorted(params)
        signature = json.dumps(params, sort_keys=True)
        ck.require((*key, signature) not in search_map, "parámetros de candidatos únicos por búsqueda")
        search_map[(*key, signature)] = candidate
        if candidate["selected"]:
            selected_rows[key] = candidate
    seen_scores = set()
    observations = {}
    for score in scores:
        fit_id = score["fit_id"]
        ck.require(fit_id in fits and fit_id not in seen_scores, "cada score interno corresponde a un fit único")
        seen_scores.add(fit_id)
        fit = fits[fit_id]
        key = (fit["protocolo"], fit["fold"])
        ck.require(all(score[k] == fit[k] for k in ["protocolo", "fold", "modelo"]),
                   "score y ajuste comparten contexto")
        match = inner_pair_map.get((*key, tuple(fit["train_cases"]), tuple(score["validation_cases"])))
        ck.require(match is not None, "fit y score usan exactamente train/validación internos")
        model_key = (*key, fit["modelo"])
        actual_params = {name: fit["parametros_modelo"].get(name.removeprefix("model__")) for name in grid_keys[model_key]}
        candidate = search_map.get((*model_key, json.dumps(actual_params, sort_keys=True)))
        ck.require(candidate is not None, "parámetros del ajuste pertenecen a un candidato predeclarado")
        cid, inner_fold = candidate["candidate_id"], match["inner_fold"]
        ck.close(score["MAE_N"], candidate[f"split{inner_fold - 1}_MAE_N"],
                 "score del log coincide con el candidato y fold interno")
        observation_key = (*key, fit["modelo"], cid, inner_fold)
        ck.require(observation_key not in observations, "un ajuste por candidato y fold interno")
        observations[observation_key] = True
    expected_observations = {(row.protocolo, row.fold, row.modelo, row.candidate_id, record["inner_fold"])
                             for row in searches.itertuples(index=False)
                             for record in inner_map[(row.protocolo, row.fold)]}
    ck.require(set(observations) == expected_observations, "logs completos de todos los candidatos y folds internos")
    refits = {}
    for fit_id, fit in fits.items():
        key = (fit["protocolo"], fit["fold"])
        ck.require(fit["error"] is None, "ningún ajuste fallido oculto en logs")
        ck.require(np.isfinite(fit["tiempo_s"]) and fit["tiempo_s"] >= 0,
                   "duración de cada ajuste finita y no negativa")
        ck.require(datetime.fromisoformat(fit["inicio_utc"]) <= datetime.fromisoformat(fit["fin_utc"]),
                   "timestamps ordenados de cada ajuste")
        if fit_id not in seen_scores:
            ck.require(fit["train_cases"] == outer_map[key]["train_cases"],
                       "refit final usa exclusivamente train externo")
            refit_key = (*key, fit["modelo"])
            ck.require(refit_key not in refits, "un único refit final por modelo y fold externo")
            refits[refit_key] = fit
            desired = ({"model__alpha": 1.0, "model__solver": "svd"} if fit["modelo"] == BASELINE
                       else json.loads(selected_rows[refit_key]["params_json"]))
            ck.require(all(fit["parametros_modelo"].get(name.removeprefix("model__")) == value for name, value in desired.items()),
                       "parámetros del refit coinciden con candidato elegido o baseline fijo")
    expected_refits = {(p, f, m) for p, f in outer_map for m in [BASELINE, *MODEL_NAMES]}
    ck.require(set(refits) == expected_refits, "logs completos de todos los refits finales y baseline")
    return ck


def verify(root=ROOT, output=None):
    """Audita sin entrenar; escribe verificacion.json solo si todas las pruebas pasan."""
    from .ajuste_modelos import load_config, load_frozen_data, model_specs, source_digest
    from sklearn.model_selection import ParameterGrid

    root = Path(root).resolve()
    out = root / "reports/modelos_ajustados" if output is None else Path(output)
    if not out.is_absolute():
        out = root / out
    out = out.resolve()
    read_json = lambda name: json.loads((out / name).read_text(encoding="utf-8"))
    manifest = read_json("manifest.json")
    config = read_json("config_ejecutada.json")
    ck = Checks()
    ck.require(manifest["estado"] == "completado" and manifest["ejecutado"] is True,
               "ejecución completa, sin resultados parciales")
    ck.require(config == load_config(root), "configuración ejecutada coincide con configuración publicada")
    ck.require(manifest["seed"] == config["seed"] == 42, "semilla 42 fija")
    ck.require(manifest["target"] == TARGET and manifest["target_unit"] == "N",
               "target y unidad constantes")
    ck.require(manifest["entradas"] == list(GEOMETRY) and manifest["representacion"] == "A",
               "mismas entradas geométricas del baseline")
    ck.require(manifest["data_sha256_antes"] == manifest["data_sha256_despues"],
               "datos y particiones no cambian durante la ejecución")
    for path, digest in manifest["data_sha256_antes"].items():
        ck.require(hashlib.sha256((root / path).read_bytes()).hexdigest() == digest,
                   "huella de entrada vigente: " + path)
    ck.require(manifest["data_sha256_antes"][config["dataset"]] == DATA_SHA,
               "snapshot de datos idéntico a Semana 5")
    ck.require(manifest["data_sha256_antes"][config["particiones"]] == SPLIT_SHA,
               "particiones idénticas a Semana 5")
    ck.require(manifest["fuentes_sha256"] == manifest["fuentes_sha256_despues"],
               "código y rejillas no cambian durante la ejecución")
    for path, digest in manifest["fuentes_sha256"].items():
        ck.require(source_digest(root / path) == digest, "fuente ejecutada coincide con fuente publicada: " + path)
    for path, digest in manifest["artefactos_sha256"].items():
        target = (out / path).resolve()
        ck.require(target.is_relative_to(out), "artefacto dentro de la carpeta del experimento: " + path)
        ck.require(hashlib.sha256(target.read_bytes()).hexdigest() == digest,
                   "artefacto sin modificaciones posteriores: " + path)
    ck.require(datetime.fromisoformat(manifest["inicio_utc"]) <= datetime.fromisoformat(manifest["particiones_predeclaradas_utc"])
               <= datetime.fromisoformat(manifest["fin_utc"]), "protocolo registrado antes de terminar la evaluación")
    data = load_frozen_data(root)
    outer, inner = read_json("particiones_externas.json"), read_json("particiones_internas.json")
    historical = json.loads((root / config["particiones"]).read_text(encoding="utf-8"))
    ck.require(outer == historical, "registros de reserva externa exactamente iguales al histórico")
    predictions = pd.read_csv(out / "predicciones_oof.csv")
    folds = pd.read_csv(out / "metricas_folds.csv")
    summary = pd.read_csv(out / "metricas_resumen.csv")
    searches = pd.read_csv(out / "busquedas.csv")
    audits, selections = read_json("auditoria_train.json"), read_json("seleccion_interna.json")
    reference = pd.read_csv(root / "reports/feature_engineering/tables/predicciones_oof.csv")
    audit_outer(data, outer, ck)
    audit_inner(data, outer, inner, ck)
    audit_predictions(data, outer, predictions, folds, summary, reference, ck)
    audit_train(data, audits, outer, ck)
    audit_search(searches, selections, inner, predictions, folds, ck)
    specs = model_specs(config)
    for model, (_, grid) in specs.items():
        expected_count = len(ParameterGrid(grid))
        ck.require(manifest["candidatos_por_modelo"][model] == expected_count,
                   "presupuesto de búsqueda respetado: " + model)
        counts = searches[searches.modelo == model].groupby(["protocolo", "fold"]).size()
        ck.require(counts.eq(expected_count).all(), "misma rejilla en todos los folds: " + model)
    for protocol, group in summary.groupby("protocolo"):
        baseline = group[group.modelo == BASELINE].iloc[0]
        ck.close(group.mejora_MAE_vs_baseline_pct, 100 * (baseline.MAE_N - group.MAE_N) / baseline.MAE_N,
                 "mejoras relativas recalculadas: " + protocol)
    for row in summary.itertuples(index=False):
        ff = folds[(folds.protocolo == row.protocolo) & (folds.modelo == row.modelo)]
        ck.close(row.tiempo_s, ff.tiempo_s.sum(), f"costo agregado recalculado: {row.protocolo}/{row.modelo}")
        ck.require(row.n_folds == len(ff), f"folds agregados correctamente: {row.protocolo}/{row.modelo}")
    audit_execution_logs(out, outer, inner, searches, ck)
    report = {"status": "passed", "passed": True, "checked_utc": datetime.now(timezone.utc).isoformat(),
              "n_checks": sum(ck.counts.values()), "checks": ck.labels,
              "repeated_check_counts": {label: count for label, count in ck.counts.items() if count > 1},
              "dataset_sha256": DATA_SHA, "particiones_sha256": SPLIT_SHA,
              "manifest_sha256": hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest(),
              "scope": "Integridad de ejecución, aislamiento train y métricas; no es validación externa prospectiva."}
    save_json(out / "verificacion.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="reports/modelos_ajustados")
    args = parser.parse_args()
    result = verify(output=args.run_dir)
    print(json.dumps({"status": result["status"], "n_checks": result["n_checks"]}, ensure_ascii=False))
