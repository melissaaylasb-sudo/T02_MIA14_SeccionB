"""Benchmark reproducible de seis regresores para F1, con cribado y tuning anidados.

Ejecutar desde la raíz: python -m src.benchmark_regresion.
La verificación y lectura de resultados existentes no ajustan ningún modelo.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.tree import export_text
from threadpoolctl import threadpool_limits

from .ajuste_modelos import ROOT, GROUP_COLUMNS, inner_splits, load_frozen_data, load_outer_splits
from .feature_engineering import GEOMETRY, TARGET, feature_names, nominal_groups
from .benchmark_regresion_modelos import (ADAPT, INITIAL, REFS, UNITS, Params, best_index,
                                          config, freeze_candidates, inner_metrics, pipeline, rank_families)

TABLES=("predicciones_oof","predicciones_train","metricas_folds","metricas_resumen",
        "cribado_interno","busqueda_interna","seleccion_tuning","predicciones_internas",
        "particiones_externas","particiones_internas","ajustes","diagnosticos_modelos",
        "coeficientes_ols","vecinos_knn","reglas_arbol","invariancia_historica")
SOURCES=("src/benchmark_regresion.py","src/benchmark_regresion_modelos.py","src/ajuste_modelos.py",
         "src/comparacion_modelos.py","src/feature_engineering.py","src/transformaciones_eda.py")
INPUTS=("config/benchmark_regresion.json","config/ajuste_modelos.json","config/comparacion_modelos.json",
        "data/processed/preliminary/model_table.csv","reports/feature_engineering/particiones.json",
        "reports/feature_engineering/tables/predicciones_modelos_oof.csv")
Rows=dict[str,list[dict[str,Any]]]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(value: Any) -> str:
    return json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False,
                      default=lambda x:x.item() if isinstance(x,np.generic) else str(x))


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json.loads(_object(value)),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def _ids(values: Any) -> str:
    return ";".join(str(int(v)) for v in values)


def _out(root: Path, output: str | Path | None) -> Path:
    return Path(output).resolve() if output else root/"reports/benchmark_regresion"


def _stage(proc: str) -> str:
    return "inicial" if proc in INITIAL else ("referencia" if proc in REFS else "optimizada")


def _group(proc: str) -> str:
    if proc in REFS:return "referencia"
    if proc in ADAPT:return "procedimiento_adaptativo"
    return "A_materiales" if proc in INITIAL[:3] else "B_clasicos"


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str,float]:
    """Métricas en N; R² solo definido para al menos dos respuestas diferentes."""
    y=np.asarray(y,dtype=float);p=np.asarray(pred,dtype=float)
    if y.shape!=p.shape or y.ndim!=1 or not len(y) or not np.isfinite([y,p]).all():raise ValueError("Respuestas/predicciones inválidas.")
    e=p-y;a=np.abs(e)
    return {"MAE_N":float(a.mean()),"RMSE_N":float(np.sqrt((e**2).mean())),"MedAE_N":float(np.median(a)),
            "R2_OOF":float(r2_score(y,p)) if len(y)>1 and np.ptp(y)>0 else np.nan,"sesgo_N":float(e.mean())}


def _fit(family: str, params: Params, cfg: Params, X: pd.DataFrame, y: pd.Series,
         context: Params, rows: Rows) -> tuple[Any,float]:
    if not X.index.equals(y.index) or not X.index.is_unique or set(X)!=set(GEOMETRY):raise ValueError("Contrato de entrenamiento inválido.")
    if family=="KNN" and params["n_neighbors"]>len(y):raise ValueError("k supera el train del pliegue.")
    obj=pipeline(family,params,cfg);start=perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always");obj.fit(X,y)
    seconds=perf_counter()-start
    record={**context,"familia_real":family,"n_train":len(y),"casos_train":_ids(X.index),
            "parametros_json":_object(params),"tiempo_ajuste_s":seconds,
            "avisos":" | ".join(sorted({str(w.message) for w in caught}))}
    for step,field in [("imputer","statistics_"),("scale","mean_"),("scale","scale_")]:
        if step in obj.named_steps:record[step+field]=_object(getattr(obj.named_steps[step],field).tolist())
    rows["ajustes"].append(record);return obj,seconds


def _predict(obj: Any, X: pd.DataFrame) -> tuple[np.ndarray,float]:
    start=perf_counter();p=np.asarray(obj.predict(X),dtype=float).reshape(-1);seconds=perf_counter()-start
    if len(p)!=len(X) or not np.isfinite(p).all():raise ValueError("Predicción no finita o de longitud incorrecta.")
    return p,seconds


def _diagnose(obj: Any, family: str, X: pd.DataFrame, Xt: pd.DataFrame, ctx: Params, rows: Rows) -> None:
    """Diagnósticos descriptivos del modelo ajustado; nunca intervienen en el cribado."""
    m=obj.named_steps["model"];rec={**ctx,"familia_real":family,"n_train":len(X)}
    if "variance" in obj.named_steps:
        names=list(feature_names(obj));rec["n_features_tras_varianza"]=len(names)
    if family=="Lineal_multiple":
        basis=obj.named_steps["basis"];scale=obj.named_steps["scale"];sing=basis.singular_values_
        z=obj[:-2].transform(X);zb=basis.transform(z);s=np.linalg.svd(zb-zb.mean(axis=0),compute_uv=False)
        keep=[names[i] for i in basis.indices_];phys=np.asarray(m.coef_)/scale.scale_[basis.indices_]
        intercept=float(m.intercept_-np.dot(phys,scale.mean_[basis.indices_]))
        rec.update(rango_centrado=basis.rank_,rango_base=len(keep),condicion_centrada=float("inf") if basis.rank_<len(names) else sing[0]/sing[-1],
                   condicion_base=s[0]/s[-1],variables_base=";".join(keep),intercepto_fisico_N=intercept,
                   tol_svd=basis.tol_,singulares_centrados_json=_object(sing.tolist()))
        for c in GEOMETRY:
            i=keep.index(c) if c in keep else None
            rows["coeficientes_ols"].append({**ctx,"variable":c,"unidad_variable":UNITS[c],
                "estado":"retenida" if i is not None else ("redundante_en_train" if c in names else "constante_en_train"),
                "coeficiente_N_por_unidad":phys[i] if i is not None else np.nan,
                "coeficiente_estandarizado_N":m.coef_[i] if i is not None else np.nan,"intercepto_fisico_N":intercept})
        np.testing.assert_allclose(intercept+X[keep].to_numpy()@phys,obj.predict(X),atol=1e-7,rtol=1e-10)
    elif family=="SVR_RBF":
        rec.update(n_vectores_soporte=len(m.support_),fraccion_vectores_soporte=len(m.support_)/len(X),
                   casos_soporte=_ids(X.index[m.support_]))
    elif family=="KNN":
        dist,idx=m.kneighbors(obj[:-1].transform(Xt))
        for case,ds,ix in zip(Xt.index,dist,idx):
            for i,(d,j) in enumerate(zip(ds,ix),1):
                rows["vecinos_knn"].append({**ctx,"Case_n":int(case),"orden_vecino":i,
                     "caso_vecino_train":int(X.index[j]),"distancia_escalada":float(d),"p":m.p,"weights":m.weights})
        rec.update(k=m.n_neighbors,distancia_media_vecinos=float(dist.mean()))
    elif family=="Arbol_decision":
        rec.update(profundidad=m.get_depth(),hojas=m.get_n_leaves())
        rows["reglas_arbol"].append({**ctx,"regla_descriptiva_train":export_text(m,feature_names=names,decimals=6)})
    elif family=="Media_por_nivel":
        rec.update(n_fallback_test=int((~Xt.n_cells_Y.isin(m.means_)).sum()),media_global_train_N=m.global_mean_)
    rows["diagnosticos_modelos"].append(rec)


def search_training(X: pd.DataFrame, y: pd.Series, groups: np.ndarray,
                    splits: list[tuple[np.ndarray,np.ndarray]], cfg: Params, draws: dict[str,list[Params]],
                    context: Params, rows: Rows) -> tuple[list[Params],int,float]:
    """Cribado/búsqueda únicamente sobre X,y del train externo; no recibe el test."""
    if not X.index.equals(y.index) or len(groups)!=len(X):raise ValueError("Índices internos incongruentes.")
    seen=[]
    for s,(tr,va) in enumerate(splits,1):
        if set(tr)&set(va) or set(groups[tr])&set(groups[va]) or set(tr)|set(va)!=set(range(len(X))):raise ValueError("Fuga en partición interna.")
        seen.extend(va);rows["particiones_internas"].append({**context,"split":s,"casos_train":_ids(X.index[tr]),
                     "casos_validacion":_ids(X.index[va]),"grupos_train":_object(sorted(set(groups[tr]))),
                     "grupos_validacion":_object(sorted(set(groups[va])))})
    if sorted(seen)!=list(range(len(X))):raise ValueError("Cobertura interna OOF incompleta.")
    def evaluate(family: str, order: int) -> tuple[tuple[float,float],float]:
        start=perf_counter();p=np.full(len(X),np.nan);params=draws[family][order];cid=f"{family}_{order:02d}"
        for s,(tr,va) in enumerate(splits,1):
            c={**context,"split":s,"etapa_ajuste":"cribado" if order==0 else "busqueda","config_id":cid}
            obj,_=_fit(family,params,cfg,X.iloc[tr],y.iloc[tr],c,rows);p[va],_=_predict(obj,X.iloc[va])
            for j in va:
                rows["predicciones_internas"].append({**c,"familia_real":family,"Case_n":int(X.index[j]),
                    "grupo_interno":groups[j],"observado_N":float(y.iloc[j]),"predicho_N":float(p[j])})
        return inner_metrics(y.to_numpy(),p,groups),perf_counter()-start
    started=perf_counter();screen={f:evaluate(f,0) for f in draws};ranking=rank_families({f:v[0] for f,v in screen.items()},cfg)
    shortlisted=ranking[:cfg["numero_familias_tuning"]]
    for rank,f in enumerate(ranking,1):
        (mae,mad),secs=screen[f];rows["cribado_interno"].append({**context,"familia_real":f,"MAE_interno_N":mae,
            "MAD_grupos_N":mad,"rango_cribado":rank,"elegible_tuning":True,"seleccionada_tuning":f in shortlisted,"tiempo_s":secs})
    winners=[]
    for rank,f in enumerate(shortlisted,1):
        values=[screen[f]]+[evaluate(f,i) for i in range(1,len(draws[f]))]
        idx=best_index([s for s,_ in values]);best=np.inf
        for i,((mae,mad),secs) in enumerate(values):
            best=min(mae,best);rows["busqueda_interna"].append({**context,"familia_real":f,"orden":i+1,
                "config_id":f"{f}_{i:02d}","parametros_json":_object(draws[f][i]),"MAE_interno_N":mae,
                "MAD_grupos_N":mad,"mejor_acumulado_N":best,"seleccionada":i==idx,
                "tiempo_s":secs if i else 0.0,"reuso_cribado":i==0})
        (mae,mad),_=values[idx];winners.append({"rango_cribado":rank,"familia_real":f,"config_id":f"{f}_{idx:02d}",
                        "parametros":draws[f][idx],"MAE_interno_N":mae,"MAD_grupos_N":mad})
    chosen=best_index([(w["MAE_interno_N"],w["MAD_grupos_N"]) for w in winners])
    for i,w in enumerate(winners):
        rows["seleccion_tuning"].append({**context,**{k:v for k,v in w.items() if k!="parametros"},
               "parametros_json":_object(w["parametros"]),"elegida_selector":i==chosen})
    return winners,chosen,perf_counter()-started


def _record(proc: str, family: str, stage_ctx: Params, X: pd.DataFrame, y: pd.Series, train: np.ndarray,
            test: np.ndarray, nominal: np.ndarray, p: np.ndarray, pt: np.ndarray, fit_s: float,predict_s: float,
            search_s: float, initial: np.ndarray, rows: Rows) -> None:
    ctx={**stage_ctx,"procedimiento":proc,"etapa":_stage(proc),"familia_real":family}
    yt=y.iloc[test].to_numpy();met=metrics(yt,p)
    rows["metricas_folds"].append({**ctx,"n_train":len(train),"n_test":len(test),
        "MAE_train_N":metrics(y.iloc[train].to_numpy(),pt)["MAE_N"],"MAE_validacion_N":met["MAE_N"],
        "RMSE_validacion_N":met["RMSE_N"],"MedAE_validacion_N":met["MedAE_N"],"sesgo_validacion_N":met["sesgo_N"],
        "tiempo_ajuste_s":fit_s,"tiempo_prediccion_s":predict_s,"tiempo_busqueda_s":search_s})
    for j,obs,pred,orig in zip(test,yt,p,initial):
        rows["predicciones_oof"].append({**ctx,"Case_n":int(X.index[j]),"n_cells_Y":int(X.iloc[j].n_cells_Y),
            "grupo_nominal":nominal[j],"observado_N":obs,"predicho_N":pred,"residuo_N":obs-pred,
            "error_absoluto_N":abs(obs-pred),"predicho_inicial_misma_familia_N":orig})
    for j,pred in zip(train,pt):
        rows["predicciones_train"].append({**ctx,"Case_n":int(X.index[j]),"observado_N":y.iloc[j],"predicho_N":pred})


def _summarize(rows: Rows) -> None:
    oof=pd.DataFrame(rows["predicciones_oof"]);folds=pd.DataFrame(rows["metricas_folds"])
    for (protocol,proc),g in oof.groupby(["protocolo","procedimiento"],sort=False):
        f=folds[(folds.protocolo==protocol)&(folds.procedimiento==proc)]
        summary={**metrics(g.observado_N.to_numpy(),g.predicho_N.to_numpy()),"protocolo":protocol,"procedimiento":proc,
            "etapa":_stage(proc),"grupo":_group(proc),"feature_set":"A" if proc not in REFS[:2] else ("sin_geometria" if proc==REFS[0] else "n_cells_Y"),
            "n":len(g),"n_folds":len(f),"DE_error_absoluto_N":g.error_absoluto_N.std(ddof=1),
            "DE_MAE_fold_N":f.MAE_validacion_N.std(ddof=1),"IQR_MAE_fold_N":f.MAE_validacion_N.quantile(.75)-f.MAE_validacion_N.quantile(.25),
            "MAE_train_medio_N":f.MAE_train_N.mean(),"tiempo_ajuste_s":f.tiempo_ajuste_s.sum(),
            "tiempo_prediccion_s":f.tiempo_prediccion_s.sum(),"tiempo_busqueda_s":f.tiempo_busqueda_s.sum()}
        for name,col in zip(REFS,["dummy","media_nivel","Ridge"]):
            ref=oof[(oof.protocolo==protocol)&(oof.procedimiento==name)]
            summary["delta_MAE_vs_"+col+"_N"]=summary["MAE_N"]-ref.error_absoluto_N.mean()
        summary["delta_MAE_vs_inicial_misma_familia_N"]=summary["MAE_N"]-np.abs(g.observado_N-g.predicho_inicial_misma_familia_N).mean()
        rows["metricas_resumen"].append(summary)


def _historical(root: Path, rows: Rows) -> None:
    old=pd.read_csv(root/INPUTS[-1]);new=pd.DataFrame(rows["predicciones_oof"])
    for proc,model in [("SVR_RBF","SVR_RBF"),("Ridge_historico","Ridge")]:
        g=new[new.procedimiento==proc].merge(old[old.modelo==model],on=["protocolo","fold","Case_n"],suffixes=("_nuevo","_historico"),validate="one_to_one")
        if len(g)!=52:raise ValueError("La referencia histórica no cubre ambos protocolos.")
        np.testing.assert_allclose(g.predicho_N_nuevo,g.predicho_N_historico,atol=1e-9,rtol=1e-12)
        for r in g.itertuples():rows["invariancia_historica"].append({"protocolo":r.protocolo,"fold":r.fold,"Case_n":r.Case_n,
                "procedimiento":proc,"predicho_N":r.predicho_N_nuevo,"predicho_historico_N":r.predicho_N_historico,
                "diferencia_N":r.predicho_N_nuevo-r.predicho_N_historico})


def run(root: Path=ROOT,output: str | Path | None=None) -> dict[str,pd.DataFrame]:
    """Ejecuta una vez el protocolo fijo; usa caché verificada si existe manifiesto."""
    root=Path(root).resolve();out=_out(root,output)
    if (out/"manifest.json").is_file():return ensure_outputs(root,out)
    cfg=config(root);draws=freeze_candidates(cfg);frame=load_frozen_data(root);splits,_,_=load_outer_splits(frame,root)
    X=frame.set_index("Case_n")[GEOMETRY];y=frame.set_index("Case_n")[TARGET];nominal=nominal_groups(frame,GROUP_COLUMNS).to_numpy()
    if {p:len(v) for p,v in splits.items()}!={"configuracion":24,"familia":3} or 2 in X.index:raise ValueError("Cohorte/particiones inesperadas.")
    tab=out/"tables";tab.mkdir(parents=True,exist_ok=True);stamp=out/"protocolo_congelado.json"
    frozen={"config":cfg,"candidatos":draws,"entradas_sha256":{p:_sha(root/p) for p in INPUTS},
            "fuentes_sha256":{p:_sha(root/p) for p in SOURCES}}
    if stamp.exists() and json.loads(stamp.read_text(encoding="utf-8"))!=frozen:raise ValueError("El directorio contiene un protocolo anterior distinto; utilice una salida nueva.")
    if not stamp.exists():_json(stamp,frozen)
    started=perf_counter();rows:Rows=defaultdict(list);cache={}
    with threadpool_limits(limits=1):
        for protocol,outer in splits.items():
            for fold,(tr,te) in enumerate(outer,1):
                ctx={"protocolo":protocol,"fold":fold};rows["particiones_externas"].append({**ctx,"casos_train":_ids(X.index[tr]),"casos_test":_ids(X.index[te])})
                for f in (*INITIAL,*REFS):
                    obj,fit_s=_fit(f,cfg["parametros_iniciales"][f],cfg,X.iloc[tr],y.iloc[tr],{**ctx,"etapa_ajuste":"externa_inicial","config_id":f+"_00"},rows)
                    p,pred_s=_predict(obj,X.iloc[te]);pt,_=_predict(obj,X.iloc[tr]);cache[(protocol,fold,f)]=p
                    _record(f,f,ctx,X,y,tr,te,nominal,p,pt,fit_s,pred_s,0,p,rows)
                    _diagnose(obj,f,X.iloc[tr],X.iloc[te],{**ctx,"procedimiento":f},rows)
        _historical(root,rows)
        print(f"Fase inicial completa: {len(rows['predicciones_oof'])} OOF; invarianza histórica verificada.",flush=True)
        for protocol,outer in splits.items():
            for fold,(tr,te) in enumerate(outer,1):
                ctx={"protocolo":protocol,"fold":fold};inner=inner_splits(frame,tr,protocol)
                grouping=nominal[tr] if protocol=="configuracion" else X.iloc[tr].n_cells_Y.to_numpy()
                winners,chosen,search_s=search_training(X.iloc[tr],y.iloc[tr],grouping,inner,cfg,draws,ctx,rows)
                for i,w in enumerate(winners):
                    proc=ADAPT[i];f=w["familia_real"];obj,fit_s=_fit(f,w["parametros"],cfg,X.iloc[tr],y.iloc[tr],
                             {**ctx,"etapa_ajuste":"externa_optimizada","config_id":w["config_id"]},rows)
                    pred,pred_s=_predict(obj,X.iloc[te]);pt,_=_predict(obj,X.iloc[tr]);initial=cache[(protocol,fold,f)]
                    _record(proc,f,ctx,X,y,tr,te,nominal,pred,pt,fit_s,pred_s,search_s,initial,rows)
                    _diagnose(obj,f,X.iloc[tr],X.iloc[te],{**ctx,"procedimiento":proc},rows)
                    if i==chosen:_record(ADAPT[2],f,ctx,X,y,tr,te,nominal,pred,pt,fit_s,pred_s,search_s,initial,rows)
                print(f"{protocol} {fold}/{len(outer)}: {', '.join(w['familia_real'] for w in winners)}; {perf_counter()-started:.1f}s",flush=True)
        # Receta final solo para un uso futuro con geometrías dentro del dominio primario.
        tr=np.arange(len(X));ctx={"protocolo":"campana_completa","fold":0}
        winners,chosen,_=search_training(X,y,nominal,inner_splits(frame,tr,"configuracion"),cfg,draws,ctx,rows)
        for w in winners:
            f=w["familia_real"];obj,_=_fit(f,w["parametros"],cfg,X,y,{**ctx,"etapa_ajuste":"reajuste_campana","config_id":w["config_id"]},rows)
            modeldir=out/"estimadores_locales";modeldir.mkdir(exist_ok=True);joblib.dump(obj,modeldir/(f+".joblib"),compress=3)
        _json(out/"campana_completa.json",{"protocolo":"configuracion","alcance":"Reajuste con todos los especímenes observados; sin evaluación externa nueva.",
               "casos_train":X.index.tolist(),"feature_set":"A","candidatas":winners,"selector":winners[chosen]})
    _summarize(rows);frames={name:pd.DataFrame(rows[name]) for name in TABLES}
    for name,d in frames.items():d.to_csv(tab/(name+".csv"),index=False,encoding="utf-8-sig")
    outputs=["tables/"+name+".csv" for name in TABLES]+["campana_completa.json","protocolo_congelado.json"]
    manifest={"version":cfg["version"],"utc":datetime.now(timezone.utc).isoformat(),"status":"complete","segundos":perf_counter()-started,
            "python":platform.python_version(),"plataforma":platform.platform(),"paquetes":{k:metadata.version(k) for k in ["numpy","pandas","scipy","scikit-learn","xgboost","catboost","threadpoolctl","joblib"]},
            "inputs":frozen["entradas_sha256"],"sources":frozen["fuentes_sha256"],"outputs":{p:_sha(out/p) for p in outputs},
            "conteos":{k:len(d) for k,d in frames.items()},"pesos_locales_opcionales":{p.name:_sha(p) for p in modeldir.glob("*.joblib")}}
    _json(out/"manifest.json",manifest);_json(out/"verificacion.json",verify_outputs(root,out))
    return frames


def _tables(out: Path) -> dict[str,pd.DataFrame]:
    return {n:pd.read_csv(out/"tables"/(n+".csv")) for n in TABLES}


def verify_outputs(root: Path=ROOT,output: str | Path | None=None) -> Params:
    """Comprueba hashes, reservas, métricas, búsquedas y contrastes históricos sin entrenar."""
    root=Path(root).resolve();out=_out(root,output);manifest=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
    if manifest["status"]!="complete":raise ValueError("Manifiesto incompleto.")
    count=0
    for name,base in [("inputs",root),("sources",root),("outputs",out)]:
        for rel,digest in manifest[name].items():
            if _sha(base/rel)!=digest:raise ValueError(f"SHA256 distinto: {rel}")
            count+=1
    cfg=config(root);frozen=json.loads((out/"protocolo_congelado.json").read_text(encoding="utf-8"))
    if frozen["candidatos"]!=freeze_candidates(cfg):raise ValueError("Candidatos diferentes del plan.")
    data=pd.read_csv(root/INPUTS[3]).set_index("Case_n");ids=set(data.index);frames=_tables(out)
    if manifest["conteos"]!={name:len(df) for name,df in frames.items()}:raise ValueError("Conteos de salida distintos.")
    oof=frames["predicciones_oof"];trainp=frames["predicciones_train"];ms=frames["metricas_resumen"]
    if len(oof)!=624 or len(ms)!=24 or oof.duplicated(["protocolo","procedimiento","Case_n"]).any():raise ValueError("Cobertura OOF inesperada.")
    np.testing.assert_allclose(oof.observado_N,data.loc[oof.Case_n,TARGET],rtol=0,atol=1e-12)
    np.testing.assert_allclose(oof.residuo_N,oof.observado_N-oof.predicho_N,rtol=0,atol=1e-8)
    np.testing.assert_allclose(oof.error_absoluto_N,np.abs(oof.residuo_N),rtol=0,atol=1e-8)
    def case_set(s: Any) -> set[int]:return {int(v) for v in str(s).split(";")}
    outer={(r.protocolo,r.fold):(case_set(r.casos_train),case_set(r.casos_test)) for r in frames["particiones_externas"].itertuples()}
    outer[("campana_completa",0)]=(ids,set())
    for (p,fold),(tr,te) in outer.items():
        if tr&te or tr|te!=ids:raise ValueError("Partición externa contaminada.")
        if p=="campana_completa":continue
        for _,g in oof[(oof.protocolo==p)&(oof.fold==fold)].groupby("procedimiento"):
            if set(g.Case_n)!=te or len(g)!=len(te):raise ValueError("Receptores OOF incorrectos.")
    for (p,fold,proc),g in trainp.groupby(["protocolo","fold","procedimiento"]):
        if set(g.Case_n)!=outer[(p,fold)][0]:raise ValueError("Predicciones de entrenamiento contaminadas.")
    for r in ms.itertuples():
        g=oof[(oof.protocolo==r.protocolo)&(oof.procedimiento==r.procedimiento)]
        if set(g.Case_n)!=ids:raise ValueError("Faltan receptores.")
        for k,v in metrics(g.observado_N.to_numpy(),g.predicho_N.to_numpy()).items():
            np.testing.assert_allclose(v,getattr(r,k),atol=1e-8,rtol=1e-9)
    inner={}
    for r in frames["particiones_internas"].itertuples():
        tr=case_set(r.casos_train);va=case_set(r.casos_validacion);allowed,forbidden=outer[(r.protocolo,r.fold)]
        if tr&va or tr|va!=allowed or (tr|va)&forbidden:raise ValueError("Fuga entrenamiento interno/externo.")
        inner[(r.protocolo,r.fold,r.split)]=(tr,va)
    for r in frames["ajustes"].itertuples():
        expected=inner[(r.protocolo,r.fold,int(r.split))][0] if r.etapa_ajuste in ("cribado","busqueda") else outer[(r.protocolo,r.fold)][0]
        if case_set(r.casos_train)!=expected:raise ValueError("Pipeline ajustado con casos ajenos al entrenamiento.")
    inp=frames["predicciones_internas"];losses={}
    for (p,fold,cid),g in inp.groupby(["protocolo","fold","config_id"]):
        if set(g.Case_n)!=outer[(p,fold)][0] or g.Case_n.duplicated().any():raise ValueError("OOF interno incompleto.")
        for split,v in g.groupby("split"):
            if set(v.Case_n)!=inner[(p,fold,split)][1]:raise ValueError("Predicción interna fuera de reserva.")
        losses[(p,fold,cid)]=inner_metrics(g.observado_N.to_numpy(),g.predicho_N.to_numpy(),g.grupo_interno.to_numpy())
    for (p,fold),c in frames["cribado_interno"].groupby(["protocolo","fold"]):
        scores={r.familia_real:losses[(p,fold,r.familia_real+"_00")] for r in c.itertuples()}
        ranking=rank_families(scores,cfg);np.testing.assert_array_equal(ranking,c.sort_values("rango_cribado").familia_real.to_list())
        selected=frames["seleccion_tuning"];selected=selected[(selected.protocolo==p)&(selected.fold==fold)].sort_values("rango_cribado")
        if selected.familia_real.tolist()!=ranking[:2]:raise ValueError("Familias elegidas no cumplen el cribado.")
        for r in selected.itertuples():
            b=frames["busqueda_interna"];b=b[(b.protocolo==p)&(b.fold==fold)&(b.familia_real==r.familia_real)].sort_values("orden")
            if len(b)!=cfg["presupuesto_por_candidata"]:raise ValueError("Presupuesto de búsqueda distinto.")
            vals=[losses[(p,fold,k)] for k in b.config_id]
            np.testing.assert_allclose(np.array(vals),b[["MAE_interno_N","MAD_grupos_N"]].to_numpy(),rtol=1e-9,atol=1e-8)
            i=best_index(vals)
            if r.config_id!=b.iloc[i].config_id or not b.iloc[i].seleccionada:raise ValueError("Configuración no reproduce la selección.")
        z=best_index(list(zip(selected.MAE_interno_N,selected.MAD_grupos_N)))
        if not selected.iloc[z].elegida_selector:raise ValueError("Selector adaptativo incongruente.")
    h=frames["invariancia_historica"]
    if len(h)!=104 or h.diferencia_N.abs().max()>1e-9:raise ValueError("Diferencias frente al histórico.")
    return {"status":"PASS","checks":{"hashes":count,"predicciones_oof":len(oof),"metricas_globales":len(ms),
            "reservas_externas":len(outer)-1,"reservas_internas_incluida_campana":len(inner),
            "ajustes_con_train_verificado":len(frames["ajustes"]),"predicciones_internas":len(inp),
            "configuraciones_buscadas":len(frames["busqueda_interna"]),"invariancia_historica":len(h)},
            "sin_reentrenamiento":True}


def ensure_outputs(root: Path=ROOT,output: str | Path | None=None) -> dict[str,pd.DataFrame]:
    """Carga tablas verificadas; ejecuta solo si no hay manifiesto terminado."""
    root=Path(root).resolve();out=_out(root,output)
    if (out/"manifest.json").exists():
        verify_outputs(root,out);return _tables(out)
    return run(root,out)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=ROOT);parser.add_argument("--output",type=Path)
    parser.add_argument("--verify",action="store_true");args=parser.parse_args()
    if args.verify:print(_object(verify_outputs(args.root,args.output)),flush=True)
    else:run(args.root,args.output)
