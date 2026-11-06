"""Auditoría independiente del benchmark exportado; no importa ni ajusta modelos.

Desde la raíz: python -m src.auditoria_benchmark
Lee los artefactos congelados y el MATLAB; solo escribe su propio informe JSON.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = "First_Failure_Load_N"
GEOM = ["n_cells_Y", "Fiber_height", "Fiber_base", "Fiber_total_length", "Fiber_total_volume",
        "Pivot_height", "Pivot_radius", "Pivot_total_number", "Pivot_total_volume", "Sample_total_volume"]
KEY = ["n_cells_Y", "Fiber_height", "Pivot_height"]
RANKS = ["Tuning_rango_1", "Tuning_rango_2"]
BASES = {"Dummy_mediana": ("Mediana", "referencia"), "Media_por_nivel": ("Media_por_nivel", "referencia"),
         "Ridge_historico": ("Ridge", "todas"), "SVR_RBF": ("SVR_RBF", "todas")}
N_FOLDS = {"configuracion": 24, "familia": 3}
ATOL = 1e-9


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8-sig"))


def _ids(s: str | int) -> list[int]:
    return [int(x) for x in str(s).split(";")]


def _truth(raw: Path) -> pd.DataFrame:
    """Extraer solo los vectores MATLAB explícitos requeridos; no ejecutar el archivo."""
    s = raw.read_bytes().decode("latin-1")
    if not re.search(r"^Case_n\s*=\s*\(1:27\)'\s*;", s, re.M):
        raise ValueError("Declaración MATLAB de Case_n no reconocida.")
    columns = {}
    for name in [*GEOM, TARGET]:
        vec = re.search(rf"^{name}\s*=\s*\[([^\]]+)\]\s*;", s, re.M)
        if vec:
            columns[name] = [float(t.strip()) for t in vec[1].split(";")]
        elif name == "Pivot_radius" and (rep := re.search(
                r"^Pivot_radius\s*=\s*repmat\(([\d.]+),\s*27,\s*1\)\s*;", s, re.M)):
            columns[name] = [float(rep[1])] * 27
        else:
            raise ValueError(f"No se reconoce el vector MATLAB {name}.")
    return pd.DataFrame(columns, index=pd.Index(range(1, 28), name="Case_n"))


class _Audit:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def check(self, kind: str, assertion: bool, where: Any) -> None:
        if not assertion:
            raise ValueError(f"{kind}: {where}")
        self.counts[kind] += 1

    def equal(self, kind: str, actual, expected, where, *, atol: float = ATOL,
              rtol: float = 0) -> None:
        x, y = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.check(kind, x.shape == y.shape and bool(np.isfinite(x).all()) and
                   bool(np.isfinite(y).all()) and np.allclose(x, y, atol=atol, rtol=rtol), where)


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    err = prediction - y
    abs_err = np.abs(err)
    return {"MAE_N": float(abs_err.mean()), "RMSE_N": float(np.sqrt(np.mean(err**2))),
            "MedAE_N": float(np.median(abs_err)), "sesgo_N": float(err.mean()),
            "R2_OOF": float(1 - (err @ err) / np.sum((y - y.mean())**2))}


def audit(root: Path = ROOT) -> dict:
    """Recalcular los contrastes desde archivos; abortar ante la primera discrepancia."""
    root = root.resolve()
    out = root / "reports/benchmark_regresion"
    manifest = out / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("Falta manifest.json del motor; no se leen tablas ni se genera informe.")
    start_hash = _sha(manifest)
    man = _json(manifest)
    a = _Audit()
    a.check("manifiesto", man["status"] == "complete", man["status"])
    for category in ("inputs", "sources", "outputs"):
        base = out if category == "outputs" else root
        for name, expected in man[category].items():
            a.check("huellas_" + category, _sha(base / name) == expected, name)
    loaded = {}
    def table(name):
        if name not in loaded:
            df = pd.read_csv(out / "tables" / (name + ".csv"), float_precision="round_trip")
            a.check("filas_manifestadas", len(df) == man["conteos"][name], name)
            loaded[name] = df
        return loaded[name]
    cfg = _json(root / "config/benchmark_regresion.json")
    frozen = _json(out / "protocolo_congelado.json")
    a.check("congelacion", cfg == frozen["config"] and frozen["entradas_sha256"] == man["inputs"]
            and frozen["fuentes_sha256"] == man["sources"], "configuración y huellas")
    draws = frozen["candidatos"]
    fixed = cfg["parametros_iniciales"]
    procedures = set(cfg["familias"] + cfg["referencias"] + RANKS + ["Selector_tuning"])
    tunable = set(cfg["espacios"])
    a.check("contrato", len(procedures) == 12 and len(tunable) == 5 and "Lineal_multiple" not in tunable
            and cfg["presupuesto_por_candidata"] == 8 and cfg["numero_familias_tuning"] == 2,
            "seis iniciales, tres referencias, dos rangos y selector")
    a.check("candidatos", set(draws) == tunable, "familias congeladas")
    for family, configs in draws.items():
        a.check("candidatos", len(configs) == len({json.dumps(v,sort_keys=True) for v in configs}) == 8
                and configs[0] == fixed[family], family)
        for parameters in configs:
            space = cfg["espacios"][family]
            a.check("candidatos", set(parameters) == set(fixed[family]) and all(
                (value in space[k] if k in space else value == fixed[family][k])
                for k, value in parameters.items()), (family, parameters))

    raw = root / "data/raw/dati_campagna_venditti.m"
    data = _truth(raw)
    labelled = data[TARGET].dropna()
    ids = set(labelled.index)
    a.check("verdad_MATLAB", set(data.index) - ids == {2} and len(ids) == 26
            and data[GEOM].notna().all().all(), "único target ausente: caso 2")
    model_file = root / "data/processed/preliminary/model_table.csv"
    frame = pd.read_csv(model_file, float_precision="round_trip").set_index("Case_n").sort_index()
    a.check("verdad_MATLAB", frame.index.is_unique and ids.issubset(frame.index)
            and set(frame.index).issubset(data.index), "identificadores tabla procesada")
    a.equal("verdad_MATLAB", frame.loc[labelled.index, GEOM + [TARGET]],
            data.loc[labelled.index, GEOM + [TARGET]], "tabla procesada", atol=0)
    if 2 in frame.index:
        a.check("verdad_MATLAB", pd.isna(frame.loc[2,TARGET]), "caso 2 no imputado")
    nominal = {i: tuple(row) for i, row in data.loc[:, KEY].iterrows()}
    level = {i: int(v) for i,v in data.n_cells_Y.items()}
    def groups(indices, protocol):
        keys = level if protocol == "familia" else nominal
        return {keys[i] for i in indices}

    old = root / "reports/permutation_importance/tables"
    old_splits = pd.read_csv(old/"particiones_externas.csv").set_index(["protocolo","fold"])
    splits, inner = {}, {}
    outer = table("particiones_externas")
    a.check("particiones_externas", outer.groupby("protocolo").size().to_dict() == N_FOLDS
            and not outer.duplicated(["protocolo","fold"]).any(), "24 configuraciones + 3 niveles")
    for protocol, g in outer.groupby("protocolo"):
        hold_ids = []
        for row in g.itertuples():
            ctx = (protocol, row.fold)
            tr, te = _ids(row.casos_train), _ids(row.casos_test)
            otr, ote = set(tr), set(te)
            a.check("particiones_externas", len(tr)==len(otr) and len(te)==len(ote) and not (otr & ote)
                    and otr | ote == ids and not groups(tr,protocol) & groups(te,protocol)
                    and len(groups(te,protocol)) == 1, ctx)
            hist = old_splits.loc[ctx]
            a.check("particiones_historicas", otr == set(_ids(hist.casos_train)) and
                    ote == set(_ids(hist.casos_test)), ctx)
            splits[ctx] = (otr,ote)
            hold_ids.extend(te)
        a.check("cobertura", sorted(hold_ids) == sorted(ids), protocol)
    campaign = ("campana_completa",0)
    splits[campaign] = (ids,set())
    parts = table("particiones_internas")
    a.check("particiones_internas", set(parts.groupby(["protocolo","fold"]).groups)==set(splits),
            "todos los folds, más reajuste separado")
    for ctx,g in parts.groupby(["protocolo","fold"]):
        active, outer_hold = splits[ctx]; covered = []
        ns = 2 if ctx[0]=="familia" else 5
        a.check("particiones_internas", len(g)==ns and set(g.split)==set(range(1,ns+1)), ctx)
        for rr in g.itertuples():
            key = (*ctx,rr.split); tr = _ids(rr.casos_train); va = _ids(rr.casos_validacion)
            a.check("particiones_internas", len(tr)==len(set(tr)) and len(va)==len(set(va)) and
                    set(tr)|set(va)==active and not set(tr)&set(va) and not set(va)&outer_hold
                    and not groups(tr,ctx[0])&groups(va,ctx[0]), key)
            inner[key] = (set(tr),set(va)); covered.extend(va)
        a.check("cobertura_interna", sorted(covered)==sorted(active), ctx)

    oof = table("predicciones_oof")
    a.check("cobertura_OOF", set(oof.protocolo)==set(N_FOLDS) and len(oof)==len(ids)*len(procedures)*2
            and set(oof.procedimiento)==procedures and not oof.duplicated(["protocolo","procedimiento","Case_n"]).any(),
            "624 OOF, sin campaña completa, duplicados ni caso 2")
    metrics = table("metricas_resumen").set_index(["protocolo","procedimiento"])
    a.check("cobertura_metricas", len(metrics)==2*len(procedures) and metrics.index.is_unique, "24 series")
    trainpred = table("predicciones_train")
    by_train = trainpred.groupby(["protocolo","fold","procedimiento"])
    by_case = oof.set_index(["protocolo","procedimiento","Case_n"]).sort_index()
    fold_metrics = table("metricas_folds").set_index(["protocolo","fold","procedimiento"])
    for (proto,proc),g in oof.groupby(["protocolo","procedimiento"]):
        a.check("cobertura_OOF", set(g.Case_n)==ids, (proto,proc))
        y = labelled.loc[g.Case_n].to_numpy()
        pred = g.predicho_N.to_numpy()
        a.equal("verdad_observada", g.observado_N, y, (proto,proc), atol=0)
        a.equal("errores_signed", g.residuo_N,y-pred, (proto,proc))
        a.equal("errores_signed", g.error_absoluto_N,np.abs(y-pred), (proto,proc))
        summary = metrics.loc[(proto,proc)]
        for name,v in _metrics(y,pred).items():
            a.equal("metricas_pooled", summary[name], v, (proto,proc,name))
        loss_group = []
        for fold,part in g.groupby("fold"):
            ctx=(proto,fold); tri,tei=splits[ctx]
            a.check("procedimiento_por_reserva", set(part.Case_n)==tei,(ctx,proc))
            loss_group.append(part.error_absoluto_N.mean())
            t = by_train.get_group((*ctx,proc))
            a.check("train_aparte", len(t)==len(tri) and set(t.Case_n)==tri, (ctx,proc))
            a.equal("train_aparte",t.observado_N,labelled.loc[t.Case_n],(ctx,proc),atol=0)
            fm = fold_metrics.loc[(*ctx,proc)]
            a.equal("error_train_verificado",fm.MAE_train_N,np.mean(abs(t.predicho_N-t.observado_N)),(ctx,proc))
            a.equal("error_test_verificado",fm.MAE_validacion_N,loss_group[-1],(ctx,proc))
        a.equal("dispersion_entre_grupos",summary.DE_MAE_fold_N,np.std(loss_group,ddof=1),(proto,proc))
        a.equal("dispersion_entre_grupos",summary.IQR_MAE_fold_N,np.ptp(np.quantile(loss_group,[.25,.75])),(proto,proc))

    history=pd.read_csv(old/"predicciones_oof.csv",float_precision="round_trip")
    historics, fallback = {}, Counter()
    for proc,(oldname,origset) in BASES.items():
        actual=oof[oof.procedimiento.eq(proc)]
        oldrows=history[history.modelo.eq(oldname)&history.conjunto.eq(origset)]
        cmp=actual.merge(oldrows,on=["protocolo","fold","Case_n"],suffixes=("","_previa"),validate="one_to_one")
        a.check("invariancia_historica",len(cmp)==52,proc)
        a.equal("invariancia_historica",cmp.predicho_N,cmp.predicho_N_previa,proc)
        historics[proc]={"casos_protocolo":len(cmp),"diferencia_maxima_N":float(abs(cmp.predicho_N-cmp.predicho_N_previa).max())}
        if proc in ("Dummy_mediana","Media_por_nivel"):
            for rr in cmp.itertuples():
                ctx=(rr.protocolo,rr.fold); tr=sorted(splits[ctx][0]); same=[i for i in tr if level[i]==level[rr.Case_n]]
                if proc=="Media_por_nivel":
                    use_fallback=not same; fallback[rr.protocolo]+=int(use_fallback)
                    expected=labelled.loc[same or tr].mean()
                    a.check("fallback_historico",str(rr.fallback_media_global).lower()==str(use_fallback).lower(),(ctx,rr.Case_n))
                else: expected=labelled.loc[tr].median()
                a.equal("referencias_desde_MATLAB",rr.predicho_N,expected,(ctx,rr.Case_n,proc))
    a.check("fallback",fallback=={"configuracion":0,"familia":26},dict(fallback))

    def keyed(name):
        return table(name).groupby(["protocolo","fold"])
    screens, searches, selected, internal = map(keyed,["cribado_interno","busqueda_interna","seleccion_tuning","predicciones_internas"])
    for group in (screens,searches,selected,internal):
        a.check("contextos_busqueda",set(group.groups)==set(splits),"27 evaluaciones y 1 reajuste")
    winners_all={}
    for ctx,(train,hold) in splits.items():
        pred_int=internal.get_group(ctx); screen=screens.get_group(ctx); searched=searches.get_group(ctx); chosen=selected.get_group(ctx)
        a.check("presupuesto",len(screen)==5 and set(screen.familia_real)==tunable and len(searched)==16
                and len(chosen)==2 and set(chosen.rango_cribado)=={1,2},ctx)
        scores={}
        for (family,cid),cg in pred_int.groupby(["familia_real","config_id"]):
            a.check("OOF_internas",len(cg)==len(train) and set(cg.Case_n)==train,(ctx,cid))
            for sid,rows in cg.groupby("split"):
                a.check("OOF_internas",set(rows.Case_n)==inner[(*ctx,sid)][1],(ctx,cid,sid))
            a.equal("verdad_interna",cg.observado_N,labelled.loc[cg.Case_n],(ctx,cid),atol=0)
            er=np.abs(cg.predicho_N.to_numpy()-labelled.loc[cg.Case_n].to_numpy())
            grouping=level if ctx[0]=="familia" else nominal
            groups_from_raw={}
            for case,e in zip(cg.Case_n,er):
                groups_from_raw.setdefault(grouping[case],[]).append(e)
            av=np.array([np.mean(x) for x in groups_from_raw.values()])
            scores[cid]=(float(er.mean()),float(np.median(np.abs(av-np.median(av)))))
        a.check("reutilizacion_cribado",set(scores)==set(searched.config_id)|{f+"_00" for f in tunable}
                and len(scores)==19,ctx)
        remaining=set(tunable); ranking=[]
        while remaining:
            cut=min(scores[f+"_00"][0] for f in remaining)+cfg["banda_cribado_N"]
            pool=[f for f in remaining if scores[f+"_00"][0]<=cut+1e-12]
            f=min(pool,key=lambda x:(scores[x+"_00"][1],scores[x+"_00"][0],cfg["orden_complejidad"].index(x),x))
            ranking.append(f);remaining.remove(f)
        for rr in screen.itertuples():
            family=rr.familia_real
            a.equal("metricas_cribado",[rr.MAE_interno_N,rr.MAD_grupos_N],scores[family+"_00"],(ctx,family))
            a.check("rango_cribado",rr.rango_cribado==ranking.index(family)+1 and
                    bool(rr.seleccionada_tuning)==(family in ranking[:2]),(ctx,family))
        winners=[]
        a.check("solo_2_familias",set(searched.familia_real)==set(chosen.familia_real)==set(ranking[:2]),ctx)
        for rank,family in enumerate(ranking[:2],1):
            current=searched[searched.familia_real.eq(family)].sort_values("orden")
            cids=[f"{family}_{i:02d}" for i in range(8)]
            a.check("presupuesto",current.config_id.tolist()==cids and current.orden.tolist()==list(range(1,9)),(ctx,family))
            for i,row in enumerate(current.itertuples()):
                a.check("parametrizacion",json.loads(row.parametros_json)==draws[family][i] and
                        bool(row.reuso_cribado)==(i==0),(ctx,row.config_id))
                a.equal("busqueda_desde_OOF",[row.MAE_interno_N,row.MAD_grupos_N],scores[row.config_id],(ctx,row.config_id))
                a.equal("busqueda_desde_OOF",row.mejor_acumulado_N,min(scores[c][0] for c in cids[:i+1]),(ctx,row.config_id))
            best=min(scores[c][0] for c in cids)
            win=min((c for c in cids if scores[c][0]<=best+1e-10),key=lambda c:(scores[c][1],cids.index(c)))
            a.check("ganadora_interna",current.loc[current.seleccionada,"config_id"].tolist()==[win],(ctx,family))
            rr=chosen.loc[chosen.rango_cribado.eq(rank)].iloc[0]
            a.check("ganadora_interna",rr.config_id==win and rr.familia_real==family and
                    json.loads(rr.parametros_json)==draws[family][cids.index(win)],(ctx,rank))
            a.equal("ganadora_interna",[rr.MAE_interno_N,rr.MAD_grupos_N],scores[win],(ctx,rank))
            winners.append(win)
        low=min(scores[c][0] for c in winners)
        win=min((c for c in winners if scores[c][0]<=low+1e-10),key=lambda c:(scores[c][1],winners.index(c)))
        a.check("selector_interno",chosen.loc[chosen.elegida_selector,"config_id"].tolist()==[win],ctx)
        winners_all[ctx]=(winners,win)
        for case in sorted(hold):
            for j,procedure in enumerate([*RANKS,"Selector_tuning"]):
                rank=j if j<2 else winners.index(win)
                row=by_case.loc[(ctx[0],procedure,case)]
                a.check("dos_rangos_y_selector",row.familia_real==ranking[rank],(ctx,case,procedure))
                a.equal("comparador_misma_familia",row.predicho_inicial_misma_familia_N,
                        by_case.loc[(ctx[0],ranking[rank],case)].predicho_N,(ctx,case,procedure))
                if j==2:
                    a.equal("prediccion_selector",row.predicho_N,by_case.loc[(ctx[0],RANKS[rank],case)].predicho_N,
                            (ctx,case),atol=0)
    refit=_json(out/"campana_completa.json")
    a.check("reajuste_separado",set(refit["casos_train"])==ids and
            [x["config_id"] for x in refit["candidatas"]]==winners_all[campaign][0] and
            refit["selector"]["config_id"]==winners_all[campaign][1], "campaña completa sin OOF propia")

    fits=table("ajustes"); prep={}
    for row in fits.itertuples():
        ctx=(row.protocolo,row.fold); subset=tuple(sorted(_ids(row.casos_train)))
        if row.etapa_ajuste in ("cribado","busqueda"):
            expected=inner[(*ctx,int(row.split))][0]
        else:
            a.check("etapa_ajuste",row.etapa_ajuste in {"externa_inicial","externa_optimizada","reajuste_campana"},ctx)
            expected=splits[ctx][0]
        a.check("identificadores_de_ajuste",set(subset)==expected and len(subset)==row.n_train==len(expected),
                (ctx,row.etapa_ajuste,row.config_id))
        if row.etapa_ajuste in ("externa_optimizada","reajuste_campana"):
            a.check("ajustes_ganadores",row.config_id in winners_all[ctx][0],(ctx,row.config_id))
        family=row.familia_real; c=int(row.config_id.rsplit("_",1)[-1])
        a.check("parametros_ajustados",json.loads(row.parametros_json)==(draws[family][c] if family in tunable else fixed[family]),
                (ctx,row.config_id))
        if family in {"Dummy_mediana","Media_por_nivel"}:continue
        if subset not in prep:
            x=data.loc[list(subset),GEOM].to_numpy(); reduced=x[:,np.ptp(x,axis=0)>0]
            prep[subset]=np.median(x,axis=0), reduced.mean(axis=0),reduced.std(axis=0)
        med,mu,sd=prep[subset]
        a.equal("medianas_solo_entrenamiento",json.loads(row.imputerstatistics_),med,(ctx,row.config_id))
        if family in {"Lineal_multiple","Ridge_historico","KNN","SVR_RBF"}:
            a.equal("escala_solo_entrenamiento",json.loads(row.scalemean_),mu,(ctx,row.config_id))
            a.equal("escala_solo_entrenamiento",json.loads(row.scalescale_),sd,(ctx,row.config_id))
    # Comprobación algebraica de OLS en todos los test externos, sin volver a entrenar.
    coef=table("coeficientes_ols")
    diags=table("diagnosticos_modelos")
    for ctx,g in coef.groupby(["protocolo","fold"]):
        train,hold=splits[ctx]; tr=sorted(train);te=sorted(hold)
        a.check("base_OLS",len(g)==len(GEOM) and set(g.variable)==set(GEOM),ctx)
        retained=g[g.estado.eq("retenida")].set_index("variable")
        x=data.loc[tr,GEOM].to_numpy(); x=x[:,np.ptp(x,axis=0)>0]; x=(x-x.mean(axis=0))/x.std(axis=0)
        s=np.linalg.svd(x-x.mean(axis=0),compute_uv=False); correct_rank=int(np.sum(s>cfg["ols"]["tolerancia_svd_relativa"]*s[0]))
        d=diags[(diags.protocolo==ctx[0])&(diags.fold==ctx[1])&(diags.procedimiento=="Lineal_multiple")].iloc[0]
        a.check("base_OLS",len(retained)==correct_rank==d.rango_centrado==d.rango_base and
                g.intercepto_fisico_N.nunique()==1,ctx)
        prediction=data.loc[te,retained.index].to_numpy()@retained.coeficiente_N_por_unidad.to_numpy()+g.intercepto_fisico_N.iloc[0]
        a.equal("prediccion_OLS_fisica",by_case.loc[[(ctx[0],"Lineal_multiple",i) for i in te],"predicho_N"],prediction,ctx,atol=1e-7)
    # El promedio de los k vecinos uniformes publicados debe reproducir el KNN inicial.
    neighbors=table("vecinos_knn")
    for (proto,fold,i),g in neighbors[neighbors.procedimiento.eq("KNN")].groupby(["protocolo","fold","Case_n"]):
        tr,te=splits[(proto,fold)];k=fixed["KNN"]["n_neighbors"]
        a.check("vecinos_entrenamiento",i in te and set(g.caso_vecino_train).issubset(tr)
                and g.caso_vecino_train.nunique()==len(g)==k and set(g.weights)=={"uniform"},(proto,fold,i))
        a.equal("KNN_desde_vecinos_MATLAB",by_case.loc[(proto,"KNN",i)].predicho_N,
                labelled.loc[g.caso_vecino_train].mean(),(proto,fold,i))
    a.check("manifiesto_sin_cambios",start_hash==_sha(manifest), "sin cambios durante la auditoría")
    return {"status":"PASS","utc":datetime.now(timezone.utc).isoformat(),"auditor":"src/auditoria_benchmark.py",
            "sha256_auditor":_sha(Path(__file__)),"sha256_manifiesto":start_hash,"sha256_MATLAB":_sha(raw),
            "sha256_historico_permutacion":_sha(old/"predicciones_oof.csv"),"tolerancia_general_abs":ATOL,
            "tolerancia_reconstruccion_OLS_N":1e-7,"numero_comprobaciones":sum(a.counts.values()),
            "comprobaciones_por_tipo":dict(a.counts),"invariancia_historica":historics,"fallback_media_nivel":dict(fallback),
            "alcance":"Comprobación independiente desde MATLAB, geometría, bitácoras internas y predicciones guardadas; sin importar ni ejecutar el motor.",
            "limites":["La auditoría verifica coherencia y separación de identificadores registrados; no prueba independencia física entre especímenes ni descarta decisiones retrospectivas.",
                        "No es una réplica desde cero de todos los ajustes ni valida generalización a otra campaña.",
                        "Los rangos y el selector describen procedimientos adaptativos; la receta con toda la campaña carece de evaluación externa propia."]}


def main() -> None:
    """Publicar el resultado de QA únicamente tras comprobar el manifiesto completo."""
    report=audit()
    folder=ROOT/"reports/benchmark_regresion/auditoria"
    folder.mkdir(exist_ok=True)
    destination=folder/"revision_independiente.json"
    tmp=folder/".revision_independiente.tmp"
    tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    tmp.replace(destination)
    print(f"PASS: {report['numero_comprobaciones']} comprobaciones. {destination.relative_to(ROOT)}")


if __name__=="__main__":
    main()
