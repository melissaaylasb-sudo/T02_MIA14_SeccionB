"""Comparación anidada de seis regresores no lineales e importancia de permutación.

Cada búsqueda de hiperparámetros utiliza solo el entrenamiento externo.
Los donantes de X son globales al conjunto OOF correspondiente; cada receptor
se predice con su modelo reservado. No es permutación intrafold ni causal.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from .ajuste_modelos import ROOT, GROUP_COLUMNS, load_frozen_data, load_outer_splits
from .feature_engineering import GEOMETRY, GeometryContract, nominal_groups


FAMILIES = ("SVR_RBF", "RandomForest", "ExtraTrees", "GaussianProcess", "GradientBoosting", "XGBoost")
BASELINES = ("Mediana", "Media_por_nivel")
TARGET = "First_Failure_Load_N"
SEED = 20261030
N_REPEATS = 100
INNER_REPEATS = 20
TOP_K = 4
LABELS = {
    "n_cells_Y": "Número de celdas en Y", "Fiber_height": "Altura de fibra",
    "Fiber_base": "Ancho de fibra", "Fiber_total_length": "Longitud total de fibras",
    "Fiber_total_volume": "Volumen total de fibras", "Pivot_height": "Altura de pivote",
    "Pivot_radius": "Radio de pivote", "Pivot_total_number": "Número total de pivotes",
    "Pivot_total_volume": "Volumen total de pivotes", "Sample_total_volume": "Volumen declarado del espécimen",
}
GROUPS = {
    "arquitectura": ["n_cells_Y", "Fiber_total_length", "Pivot_total_number"],
    "seccion": ["Fiber_height", "Fiber_base"],
    "volumenes": ["Fiber_total_volume", "Pivot_total_volume", "Sample_total_volume"],
    "pivote": ["Pivot_height", "Pivot_radius"],
}
GROUPS["arquitectura_volumenes"] = GROUPS["arquitectura"] + GROUPS["volumenes"]
GROUP_LABELS = {"arquitectura": "Arquitectura", "seccion": "Sección de fibra",
                "volumenes": "Volúmenes", "pivote": "Geometría del pivote",
                "arquitectura_volumenes": "Arquitectura y volúmenes (conjunto)"}
TABLES = ("metricas_resumen", "comparacion_todas_prioritarias", "predicciones_oof",
          "metricas_folds", "importancia_resumen", "importancia_repeticiones",
          "seleccion_variables", "frecuencias_seleccion", "ablacion_bloque",
          "predicciones_ablacion", "particiones_externas", "particiones_internas",
          "importancia_interna_repeticiones", "ajustes_auditoria", "busquedas_hiperparametros")


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def _ids(values):
    return ";".join(str(int(v)) for v in values)


def _output(root, output):
    return Path(output).resolve() if output is not None else Path(root).resolve()/"reports/modelos_no_lineales"


class SelectOriginal(TransformerMixin, BaseEstimator):
    def __init__(self, columns=tuple(GEOMETRY)):
        self.columns = columns

    def fit(self, X, y=None):
        if not self.columns or not set(self.columns).issubset(GEOMETRY):
            raise ValueError("Subconjunto de geometrías originales requerido")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X):
        return X.loc[:, list(self.columns)].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns, dtype=object)


class ReferenceRegressor(RegressorMixin, BaseEstimator):
    """Mediana global o media por nivel, aprendida solo en entrenamiento."""
    def __init__(self, kind="Mediana"):
        self.kind=kind

    def fit(self, X, y):
        GeometryContract().fit(X)
        y=np.asarray(y, dtype=float)
        self.median_=float(np.median(y)); self.mean_=float(y.mean())
        self.level_means_=pd.DataFrame({"nivel":X.n_cells_Y.to_numpy(),"respuesta":y}).groupby("nivel").respuesta.mean().to_dict()
        self.fit_case_ids_=tuple(map(int,X.index))
        return self

    def predict(self, X):
        if self.kind=="Mediana": return np.full(len(X),self.median_)
        if self.kind=="Media_por_nivel": return np.array([self.level_means_.get(float(n),self.mean_) for n in X.n_cells_Y])
        raise ValueError("Referencia desconocida")


def fixed_specs(root=ROOT):
    """Read the frozen nonlinear catalog and bounded inner-search grids."""
    cfg=json.loads((Path(root)/"config/modelos_no_lineales.json").read_text(encoding="utf-8"))
    if tuple(cfg["modelos"])!=FAMILIES or cfg["version"]!="modelos_no_lineales_v1":
        raise ValueError("El catálogo debe contener exactamente las seis familias no lineales")
    for family, spec in cfg["modelos"].items():
        if not 1 <= len(ParameterGrid(spec["grid"])) <= 8:
            raise ValueError(f"Rejilla fuera de límite: {family}")
    return cfg["modelos"]


def fixed_pipeline(family, columns, specs, params=None):
    base_params=dict(specs[family]["parametros"])
    if family=="GaussianProcess":
        if base_params.pop("kernel")!="Matern": raise ValueError("Núcleo Matérn requerido")
        kernel=Matern(length_scale=1.0,length_scale_bounds="fixed",nu=base_params.pop("nu"))
        estimator=GaussianProcessRegressor(kernel=kernel,**base_params)
    else:
        cls={"SVR_RBF":SVR,"RandomForest":RandomForestRegressor,
             "ExtraTrees":ExtraTreesRegressor,"GradientBoosting":GradientBoostingRegressor,
             "XGBoost":XGBRegressor}[family]
        estimator=cls(**base_params)
    pipeline=Pipeline([
        ("contrato",GeometryContract()),
        ("imputacion",SimpleImputer(strategy="median",keep_empty_features=True).set_output(transform="pandas")),
        ("seleccion",SelectOriginal(tuple(columns))),
        ("varianza",VarianceThreshold(0.0)),("escala",StandardScaler()),("modelo",estimator)])
    if params is not None:
        pipeline.set_params(**params)
    return pipeline


def permutation_donors(n, repetitions, seed):
    """Uses only size and seed; y cannot affect donor assignments."""
    rng=np.random.default_rng(seed)
    return np.stack([rng.permutation(n) for _ in range(repetitions)])


def perturb_batch(X, receivers, donors, columns):
    """One donor row per receptor/repetition for every column in a block."""
    receivers=np.asarray(receivers,dtype=int); donors=np.asarray(donors,dtype=int)
    if donors.ndim!=2 or donors.shape[1]!=len(X): raise ValueError("Donantes incompatibles")
    if any(sorted(row.tolist())!=list(range(len(X))) for row in donors): raise ValueError("Cada repetición debe ser una permutación completa")
    locations=[X.columns.get_loc(c) for c in columns]
    batch=np.tile(X.iloc[receivers].to_numpy(),(len(donors),1))
    donor_values=X.to_numpy()[donors[:,receivers].ravel()]
    batch[:,locations]=donor_values[:,locations]
    return pd.DataFrame(batch,columns=X.columns)


def _prediction(model,X):
    result=np.asarray(model.predict(X),dtype=float).reshape(-1)
    if result.shape!=(len(X),) or not np.isfinite(result).all(): raise ValueError("Predicción no finita o mal dimensionada")
    return result


def crossfit_predictions(X, fitted):
    result=np.full(len(X),np.nan);seen=np.zeros(len(X),dtype=int)
    for indices,model,used in fitted:
        result[indices]=_prediction(model,X.iloc[indices]);seen[indices]+=1
    if not np.all(seen==1): raise ValueError("Cada receptor debe tener una predicción reservada")
    return result


def crossfit_permuted_predictions(X,fitted,donors,columns,original=None):
    """Hold models fixed; perturb only rows scored by their reserved model."""
    original=crossfit_predictions(X,fitted) if original is None else np.asarray(original)
    result=np.tile(original,(len(donors),1))
    for indices,model,used in fitted:
        if not set(columns)&set(used): continue
        batch=perturb_batch(X,indices,donors,columns)
        result[:,indices]=_prediction(model,batch).reshape(len(donors),len(indices))
    return result


def _fit(model,X,y,context,audit):
    started=perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always");model.fit(X,y)
    model.fit_case_ids_=tuple(map(int,X.index))
    row={**context,"n_entrenamiento":len(X),"casos_entrenamiento":_ids(X.index),
         "segundos":perf_counter()-started,"avisos":json.dumps([str(w.message) for w in caught],ensure_ascii=False)}
    if isinstance(model,Pipeline):
        row["medianas_train"]=json.dumps(model.named_steps["imputacion"].statistics_.tolist())
        row["variables_efectivas"]=json.dumps(np.asarray(model.named_steps["seleccion"].columns)[model.named_steps["varianza"].get_support()].tolist())
        row["medias_escalador_train"]=json.dumps(model.named_steps["escala"].mean_.tolist())
        row["escalas_train"]=json.dumps(model.named_steps["escala"].scale_.tolist())
    audit.append(row);return model


def select_top_variables(X,y,groups,family,specs,*,seed,context=None,audit=None,params=None):
    """Supervised selection sees only the supplied external-training sample.

    Five nominal GroupKFold splits in BOTH outer scenarios. The inner donor
    pool is this training sample only. A fixed family is refitted after choice.
    """
    if len(X)!=len(y) or len(groups)!=len(X): raise ValueError("Entradas internas desalineadas")
    if pd.Series(groups).nunique()<5: raise ValueError("Menos de cinco grupos nominales internos")
    context={} if context is None else context; audit=[] if audit is None else audit
    parts=list(GroupKFold(n_splits=5).split(X,groups=groups))
    fitted=[];partition_rows=[]
    for k,(train,valid) in enumerate(parts,1):
        if set(np.asarray(groups)[train])&set(np.asarray(groups)[valid]): raise ValueError("Grupos internos solapados")
        model=_fit(fixed_pipeline(family,GEOMETRY,specs,params),X.iloc[train],np.asarray(y)[train],
                   {**context,"modelo":family,"etapa":"seleccion_interna","fold_interno":k},audit)
        fitted.append((valid,model,tuple(GEOMETRY)))
        partition_rows.append({**context,"fold_interno":k,"casos_train":_ids(X.index[train]),"casos_validacion":_ids(X.index[valid])})
    original=crossfit_predictions(X,fitted); y=np.asarray(y,dtype=float)
    base=float(np.mean(np.abs(y-original)));donors=permutation_donors(len(X),INNER_REPEATS,seed)
    repetition_rows=[];summary=[]
    for j,feature in enumerate(GEOMETRY):
        variable=X[feature].dropna().nunique()>1
        if variable:
            pred=crossfit_permuted_predictions(X,fitted,donors,[feature],original)
            losses=np.mean(np.abs(y[None,:]-pred),axis=1);delta=losses-base
        else:delta=np.zeros(INNER_REPEATS)
        for repeat,value in enumerate(delta,1):
            repetition_rows.append({**context,"modelo":family,"variable":feature,"repeticion":repeat,
                                    "MAE_interno_original_N":base,"delta_MAE_N":float(value),"semilla":seed})
        summary.append({**context,"modelo":family,"variable":feature,"etiqueta":LABELS[feature],
                        "PFI_interna_media_N":float(delta.mean()),"PFI_interna_DE_N":float(delta.std(ddof=1)),
                        "no_constante_train":bool(variable),"orden_original":j})
    eligible=[r for r in summary if r["no_constante_train"]]
    if len(eligible)<TOP_K:raise ValueError("Menos de cuatro originales no constantes en entrenamiento")
    ordered=sorted(eligible,key=lambda r:(-r['PFI_interna_media_N'],r['orden_original']))
    chosen=tuple(r['variable'] for r in ordered[:TOP_K]);ranks={r['variable']:k for k,r in enumerate(ordered,1)}
    for row in summary:
        row.update(rango=ranks.get(row['variable']),seleccionada=row['variable'] in chosen,
                   seleccion_aun_con_PFI_no_positiva=bool(row['variable'] in chosen and row['PFI_interna_media_N']<=0))
    return chosen,summary,repetition_rows,partition_rows,donors


def tune_within_outer_train(X, y, groups, family, specs, protocol):
    """Select hyperparameters without exposing any external test observation.

    Configuration validation holds out nominal geometries. Family validation
    holds out one of the two training families at a time. The resulting
    parameters remain fixed for the full, reduced and ablated feature sets,
    isolating feature effects from changes in tuning.
    """
    groups=np.asarray(groups)
    if len(X)!=len(y) or len(groups)!=len(X):
        raise ValueError("Entrenamiento y grupos desalineados")
    if protocol=="configuracion":
        splitter=GroupKFold(n_splits=5)
    elif protocol=="familia":
        splitter=LeaveOneGroupOut()
    else:
        raise ValueError(f"Protocolo desconocido: {protocol}")
    parts=list(splitter.split(X, y, groups))
    for tr,va in parts:
        if set(groups[tr]) & set(groups[va]):
            raise ValueError("Fuga entre grupos internos")
    started=perf_counter()
    search=GridSearchCV(
        estimator=fixed_pipeline(family,GEOMETRY,specs),
        param_grid=specs[family]["grid"],
        scoring="neg_mean_absolute_error", cv=parts, refit=False,
        n_jobs=1, error_score="raise", return_train_score=False,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        search.fit(X, np.asarray(y,dtype=float))
    params=dict(search.best_params_)
    if not np.isfinite(search.best_score_):
        raise ValueError("La búsqueda interna no obtuvo MAE finito")
    return params, {"best_params_json":json.dumps(params,ensure_ascii=False,sort_keys=True),
                    "inner_MAE_N":float(-search.best_score_),
                    "n_candidates":int(len(search.cv_results_["params"])),
                    "n_inner_splits":len(parts),"segundos_busqueda":perf_counter()-started,
                    "casos_entrenamiento":_ids(X.index),
                    "criterio":"MAE interno medio entre grupos; minimizar"}


def scores(y,p):
    y=np.asarray(y);p=np.asarray(p);errors=y-p
    return {"n":len(y),"MAE_N":float(np.mean(np.abs(errors))),"MSE_N2":float(np.mean(errors**2)),
            "RMSE_N":float(np.sqrt(np.mean(errors**2))),"R2":float(r2_score(y,p)) if len(y)>1 else np.nan}


def _donor_rows(X,donors,context,seed):
    n=len(X);r=len(donors);case_ids=X.index.to_numpy(dtype=int)
    return pd.DataFrame({**{k:np.repeat(v,r*n) for k,v in context.items()},
                         "repeticion":np.repeat(np.arange(1,r+1),n),
                         "caso_receptor":np.tile(case_ids,r),"caso_donante":case_ids[donors.ravel()],
                         "semilla":seed})


def _load_tables(output):
    return {name:pd.read_csv(Path(output)/"tables"/(name+".csv")) for name in TABLES}


def _protocol(root,specs):
    return {"version":"modelos_no_lineales_v1","target":TARGET,"unidad":"N","modelos":specs,
            "familias":list(FAMILIES),"referencias":list(BASELINES),"seed":SEED,
            "entradas_originales":GEOMETRY,"protocolos_externos":["configuracion","familia"],
            "busqueda":{"metodo":"GridSearchCV","scoring":"neg_mean_absolute_error",
                        "configuracion":"GroupKFold(5) sobre train externo",
                        "familia":"LeaveOneGroupOut sobre train externo",
                        "mismos_parametros_en_todas_prioritarias_ablacion":True},
            "seleccion":{"top_k":TOP_K,"repeticiones":INNER_REPEATS,"inner":"GroupKFold5 nominal en ambos escenarios",
                         "metrica":"MAE pooled por receptor","donantes":"solo X de train externo; permutaciones globales internas",
                         "empate":"orden original de columnas","conservar_top4_si_no_positivas":True,
                         "distinta_por_modelo_y_fold":True,"no_optimiza_transferencia_leaveY":True},
            "permutacion_externa":{"repeticiones":N_REPEATS,"donantes":"permutación global de X OOF; receptores usan modelo externo propio",
                                    "misma_permutacion_modelos_columnas_grupos":True,"y_no_permutado":True,
                                    "autoasignaciones_permitidas":True,"no_es_PFI_intrafold":True},
            "grupos":GROUPS,"ablacion_refit":GROUPS["arquitectura_volumenes"],
            "sin_SHAP":True,"sin_escalamiento_y_adicional":True,
            "limites":["Donantes marginales pueden romper dependencias geométricas y salir del soporte.",
                       "El donante puede estar en entrenamiento del modelo del receptor; no se usa su y en la perturbación.",
                       "PFI cruzada global es un estimando de perturbación explícito, no PFI clásica intrafold.",
                       "Percentiles de repeticiones reflejan Monte Carlo, no IC poblacionales.",
                       "Configuraciones nominales no son réplicas físicas confirmadas.",
                       "Selección supervisada interna distinta para cada modelo: no hay cuatro variables universales."],
            "input_hashes":{rel:_hash(root/rel) for rel in ["config/modelos_no_lineales.json","config/ajuste_modelos.json",
                "data/processed/preliminary/model_table.csv","reports/feature_engineering/particiones.json"]}}


def run(root=ROOT,output=None):
    """Tune six nonlinear candidates inside external train; then evaluate OOF."""
    root=Path(root).resolve();out=_output(root,output)
    if (out/"manifest.json").exists():return ensure_outputs(root,out)
    start=perf_counter();specs=fixed_specs(root);frame=load_frozen_data(root)
    splits,groups,legacy_records=load_outer_splits(frame,root)
    if [len(splits[p]) for p in ['configuracion','familia']]!=[24,3]:raise ValueError('Particiones históricas inesperadas')
    X=frame.set_index('Case_n')[GEOMETRY];y=frame[TARGET].to_numpy(dtype=float)
    for folder in [out,out/'tables',out/'logs',out/'estimadores']:folder.mkdir(parents=True,exist_ok=True)
    protocol=_protocol(root,specs);frozen=out/'protocolo_congelado.json'
    if frozen.exists() and json.loads(frozen.read_text(encoding='utf-8'))!=protocol:raise ValueError('Protocolo congelado distinto; usar otra salida versionada')
    if not frozen.exists():_json(frozen,protocol)
    source_paths=[Path(__file__),root/'src/feature_engineering.py',root/'src/ajuste_modelos.py',root/'src/transformaciones_eda.py']
    source_hashes={str(p.relative_to(root)).replace('\\','/'):_hash(p) for p in source_paths}
    _json(out/'logs/inicio_ejecucion.json',{'utc':datetime.now(timezone.utc).isoformat(),'source_hashes':source_hashes,'protocolo_sha256':_hash(frozen)})
    rows={name:[] for name in TABLES};stored={};outer_donor_tables=[];inner_donor_tables=[]
    raw_path=out/'tables/predicciones_permutadas.csv.gz';raw_started=False
    with threadpool_limits(limits=1),gzip.open(raw_path,'wt',encoding='utf-8',newline='') as raw_stream:
        for pindex,protocol_name in enumerate(['configuracion','familia']):
            fitted={(family,subset):[] for family in FAMILIES for subset in ['todas','prioritarias']}
            fitted.update({(name,'referencia'):[] for name in BASELINES})
            baseline_used={'Mediana':(), 'Media_por_nivel':('n_cells_Y',)}
            fold_index=np.zeros(len(X),dtype=int)
            for fold,(train,test) in enumerate(splits[protocol_name],1):
                fold_index[test]=fold;xt=X.iloc[train];yt=y[train];inner_seed=SEED+10000*(pindex+1)+fold
                context={'protocolo':protocol_name,'fold':fold}
                rows['particiones_externas'].append({**context,'casos_train':_ids(xt.index),'casos_test':_ids(X.index[test]),
                                                    'n_train':len(train),'n_test':len(test)})
                for family in FAMILIES:
                    tuning_groups=(nominal_groups(frame.iloc[train],GROUP_COLUMNS).to_numpy()
                                   if protocol_name=='configuracion' else
                                   frame.iloc[train].n_cells_Y.astype(int).astype(str).to_numpy())
                    best_params,search_row=tune_within_outer_train(
                        xt,yt,tuning_groups,family,specs,protocol_name)
                    rows['busquedas_hiperparametros'].append(
                        {**context,'modelo':family,'conjunto':'todas',**search_row})
                    chosen,selection,inner_reps,inner_parts,donors=select_top_variables(
                        xt,yt,nominal_groups(frame.iloc[train],GROUP_COLUMNS).to_numpy(),family,specs,
                        seed=inner_seed,context=context,audit=rows['ajustes_auditoria'],params=best_params)
                    rows['seleccion_variables'].extend(selection);rows['importancia_interna_repeticiones'].extend(inner_reps)
                    if family==FAMILIES[0]:
                        rows['particiones_internas'].extend(inner_parts)
                        inner_donor_tables.append(_donor_rows(xt,donors,context,inner_seed))
                    for subset,columns in [('todas',tuple(GEOMETRY)),('prioritarias',chosen)]:
                        model=_fit(fixed_pipeline(family,columns,specs,best_params),xt,yt,
                                   {**context,'modelo':family,'etapa':subset,
                                    'best_params_json':search_row['best_params_json']},rows['ajustes_auditoria'])
                        fitted[(family,subset)].append((test,model,columns));pred=_prediction(model,X.iloc[test])
                        path=out/'estimadores'/f'{protocol_name}_{fold:02d}_{family}_{subset}.joblib';joblib.dump(model,path,compress=3);stored[str(path.relative_to(out))]=_hash(path)
                        for pos,value in zip(test,pred):
                            rows['predicciones_oof'].append({**context,'modelo':family,'conjunto':subset,'Case_n':int(X.index[pos]),
                                'observado_N':float(y[pos]),'predicho_N':float(value),'error_absoluto_N':float(abs(y[pos]-value)),
                                'variables_usadas':';'.join(columns),'estimador':str(path.relative_to(out)).replace('\\','/')})
                        rows['metricas_folds'].append({**context,'modelo':family,'conjunto':subset,**scores(y[test],pred)})
                    kept=tuple(c for c in GEOMETRY if c not in GROUPS['arquitectura_volumenes'])
                    ablated=_fit(fixed_pipeline(family,kept,specs,best_params),xt,yt,
                                 {**context,'modelo':family,'etapa':'ablacion_bloque',
                                  'best_params_json':search_row['best_params_json']},rows['ajustes_auditoria'])
                    pred=_prediction(ablated,X.iloc[test]);path=out/'estimadores'/f'{protocol_name}_{fold:02d}_{family}_ablacion.joblib'
                    joblib.dump(ablated,path,compress=3);stored[str(path.relative_to(out))]=_hash(path)
                    for pos,value in zip(test,pred):
                        rows['predicciones_ablacion'].append({**context,'modelo':family,'Case_n':int(X.index[pos]),'observado_N':float(y[pos]),
                            'predicho_N':float(value),'variables_retenidas':';'.join(kept),'estimador':str(path.relative_to(out)).replace('\\','/')})
                for name in BASELINES:
                    model=_fit(ReferenceRegressor(name),xt,yt,{**context,'modelo':name,'etapa':'referencia'},rows['ajustes_auditoria'])
                    fitted[(name,'referencia')].append((test,model,baseline_used[name]));pred=_prediction(model,X.iloc[test])
                    path=out/'estimadores'/f'{protocol_name}_{fold:02d}_{name}.joblib';joblib.dump(model,path,compress=3);stored[str(path.relative_to(out))]=_hash(path)
                    for pos,value in zip(test,pred):
                        rows['predicciones_oof'].append({**context,'modelo':name,'conjunto':'referencia','Case_n':int(X.index[pos]),'observado_N':float(y[pos]),
                            'predicho_N':float(value),'error_absoluto_N':float(abs(y[pos]-value)), 'variables_usadas':';'.join(baseline_used[name]),
                            'estimador':str(path.relative_to(out)).replace('\\','/'),
                            'fallback_media_global':bool(name=='Media_por_nivel' and float(X.iloc[pos].n_cells_Y) not in model.level_means_)})
                    rows['metricas_folds'].append({**context,'modelo':name,'conjunto':'referencia',**scores(y[test],pred)})
                print(json.dumps({'protocolo':protocol_name,'fold_completado':fold,'segundos':round(perf_counter()-start,2)}),flush=True)
            seed=SEED+100000*(pindex+1);donors=permutation_donors(len(X),N_REPEATS,seed)
            outer_donor_tables.append(_donor_rows(X,donors,{'protocolo':protocol_name},seed))
            definitions=[('variable',c,LABELS[c],[c]) for c in GEOMETRY]+[('grupo',g,GROUP_LABELS[g],v) for g,v in GROUPS.items()]
            for (family,subset),models in fitted.items():
                original=crossfit_predictions(X,models);original_mae=float(np.mean(np.abs(y-original)))
                rows['metricas_resumen'].append({'protocolo':protocol_name,'modelo':family,'conjunto':subset,**scores(y,original)})
                for kind,name,label,columns in definitions:
                    pred=crossfit_permuted_predictions(X,models,donors,columns,original)
                    loss=np.mean(np.abs(y[None,:]-pred),axis=1);delta=loss-original_mae
                    common={'protocolo':protocol_name,'modelo':family,'conjunto':subset,'tipo':kind,'variable_grupo':name,'etiqueta':label}
                    rows['importancia_resumen'].append({**common,'delta_MAE_media_N':float(delta.mean()),'delta_MAE_DE_N':float(delta.std(ddof=1)),
                        'p05_N':float(np.quantile(delta,.05)),'p95_N':float(np.quantile(delta,.95)),
                        'fraccion_delta_positiva':float(np.mean(delta>0)),'n_repeticiones':N_REPEATS,'interpretacion_intervalo':'variabilidad_MC_no_IC_poblacional'})
                    for repeat,(l,d) in enumerate(zip(loss,delta),1):
                        rows['importancia_repeticiones'].append({**common,'repeticion':repeat,'MAE_original_N':original_mae,'MAE_permutado_N':float(l),'delta_MAE_N':float(d),'semilla':seed})
                    n=len(X);r=N_REPEATS
                    raw=pd.DataFrame({**{k:np.repeat(v,n*r) for k,v in common.items() if k!='etiqueta'},
                                      'repeticion':np.repeat(np.arange(1,r+1),n),'fold':np.tile(fold_index,r),
                                      'Case_n':np.tile(X.index.to_numpy(dtype=int),r),'caso_donante':X.index.to_numpy(dtype=int)[donors.ravel()],
                                      'observado_N':np.tile(y,r),'predicho_original_N':np.tile(original,r),'predicho_permutado_N':pred.ravel(),'semilla':seed})
                    raw.to_csv(raw_stream,index=False,header=not raw_started);raw_started=True
    frames={name:pd.DataFrame(value) for name,value in rows.items()}
    metrics=frames['metricas_resumen']
    for p in ['configuracion','familia']:
        mask=metrics.protocolo==p;current=metrics[mask]
        med=current.loc[current.modelo=='Mediana','MAE_N'].iloc[0];mean=current.loc[current.modelo=='Media_por_nivel','MAE_N'].iloc[0]
        metrics.loc[mask,'delta_MAE_vs_mediana_N']=current.MAE_N-med
        metrics.loc[mask,'delta_MAE_vs_media_nivel_N']=current.MAE_N-mean
        for family in FAMILIES:
            full=current.loc[(current.modelo==family)&(current.conjunto=='todas')].iloc[0]
            reduced=current.loc[(current.modelo==family)&(current.conjunto=='prioritarias')].iloc[0]
            m=mask&(metrics.modelo==family);metrics.loc[m,'delta_MAE_vs_todas_N']=metrics.loc[m,'MAE_N']-full.MAE_N
            rows['comparacion_todas_prioritarias'].append({'protocolo':p,'modelo':family,'MAE_todas_N':full.MAE_N,'MAE_prioritarias_N':reduced.MAE_N,
                'delta_MAE_reducir_N':reduced.MAE_N-full.MAE_N,'RMSE_todas_N':full.RMSE_N,'RMSE_prioritarias_N':reduced.RMSE_N,'R2_todas':full.R2,'R2_prioritarias':reduced.R2})
            a=frames['predicciones_ablacion'];a=a[(a.protocolo==p)&(a.modelo==family)];sc=scores(a.observado_N,a.predicho_N)
            rows['ablacion_bloque'].append({'protocolo':p,'modelo':family,'bloque':'arquitectura_volumenes','MAE_todas_N':full.MAE_N,
                'MAE_ablacion_N':sc['MAE_N'],'delta_MAE_ablacion_N':sc['MAE_N']-full.MAE_N,'MSE_ablacion_N2':sc['MSE_N2'],
                'RMSE_ablacion_N':sc['RMSE_N'],'R2_ablacion':sc['R2'],'variables_retenidas':a.variables_retenidas.iloc[0]})
    frames['comparacion_todas_prioritarias']=pd.DataFrame(rows['comparacion_todas_prioritarias'])
    frames['ablacion_bloque']=pd.DataFrame(rows['ablacion_bloque'])
    select=frames['seleccion_variables']
    frames['frecuencias_seleccion']=select.groupby(['protocolo','modelo','variable','etiqueta'],sort=False).agg(n_selecciones=('seleccionada','sum'),n_folds=('fold','nunique')).reset_index()
    frames['frecuencias_seleccion']['frecuencia']=frames['frecuencias_seleccion'].n_selecciones/frames['frecuencias_seleccion'].n_folds
    for name in TABLES:frames[name].to_csv(out/'tables'/(name+'.csv'),index=False,encoding='utf-8-sig')
    pd.concat(outer_donor_tables,ignore_index=True).to_csv(out/'tables/donantes_externos.csv.gz',index=False,compression='gzip')
    pd.concat(inner_donor_tables,ignore_index=True).to_csv(out/'tables/donantes_internos.csv.gz',index=False,compression='gzip')
    outputs={str(p.relative_to(out)).replace('\\','/'):_hash(p) for p in sorted((out/'tables').glob('*')) if p.is_file()}
    manifest={'version':'modelos_no_lineales_v1','status':'complete','created_utc':datetime.now(timezone.utc).isoformat(),'seconds':perf_counter()-start,
              'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','scipy','joblib','threadpoolctl','xgboost']},
              'protocol_sha256':_hash(frozen),'input_hashes':protocol['input_hashes'],'source_hashes':source_hashes,
              'output_hashes':outputs,'estimator_hashes':stored,'counts':{'oof_rows':len(frames['predicciones_oof']),
              'outer_folds':len(frames['particiones_externas']),'fitted_models':len(rows['ajustes_auditoria']),
              'permutation_summaries':len(frames['importancia_resumen']),'permutation_repeats':len(frames['importancia_repeticiones'])}}
    _json(out/'manifest.json',manifest)
    verification=verify_outputs(root,out,check_raw=True,require_estimators=True);_json(out/'logs/verificacion.json',verification)
    print(json.dumps(verification,ensure_ascii=False),flush=True)
    return frames


def verify_outputs(root=ROOT,output=None,*,check_raw=False,require_estimators=False):
    """Verify public tables and any local weights without fitting.

    Every non-weight artifact remains mandatory. Missing private joblib files
    are reported explicitly; require_estimators=True makes their absence fatal.
    Hash verification does not assert that absent estimators were reloaded.
    """
    root=Path(root).resolve();out=_output(root,output);m=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    if m['status']!='complete' or _hash(out/'protocolo_congelado.json')!=m['protocol_sha256']:raise AssertionError('Protocolo/manifiesto no íntegro')
    counts={'hashes':0,'metricas':0,'selecciones':0,'donantes':0,'repeticiones_PFI':0}
    for group,base in [('input_hashes',root),('source_hashes',root),('output_hashes',out)]:
        for rel,expected in m[group].items():
            if _hash(base/rel)!=expected:raise AssertionError(f'Hash no coincide: {rel}')
            counts['hashes']+=1
    missing_estimators=[];present_estimators=0
    for rel,expected in m['estimator_hashes'].items():
        path=out/rel.replace('\\','/')
        if not path.is_file():missing_estimators.append(rel);continue
        if _hash(path)!=expected:raise AssertionError(f'Hash de estimador no coincide: {rel}')
        present_estimators+=1;counts['hashes']+=1
    if require_estimators and missing_estimators:
        raise FileNotFoundError(f'Faltan {len(missing_estimators)} estimadores locales; primer archivo: {missing_estimators[0]}')
    frames=_load_tables(out);oof=frames['predicciones_oof'];metrics=frames['metricas_resumen'];selection=frames['seleccion_variables']
    expected_oof=len(FAMILIES)*2*2*len(set(oof.Case_n))+len(BASELINES)*2*len(set(oof.Case_n))
    if len(oof)!=expected_oof or oof.duplicated(['protocolo','modelo','conjunto','Case_n']).any():raise AssertionError('OOF incompleto/duplicado')
    if 2 in set(oof.Case_n):raise AssertionError('Caso2 no observado en evaluación')
    for row in metrics.itertuples():
        g=oof[(oof.protocolo==row.protocolo)&(oof.modelo==row.modelo)&(oof.conjunto==row.conjunto)]
        expected=scores(g.observado_N,g.predicho_N)
        if len(g)!=26:raise AssertionError('Cobertura por procedimiento')
        for key in ['MAE_N','MSE_N2','RMSE_N','R2']:
            if not np.isclose(expected[key],getattr(row,key),atol=1e-9,rtol=1e-10):raise AssertionError('Métrica no reproducible')
        counts['metricas']+=1
    outer={(r.protocolo,r.fold):(set(map(int,r.casos_train.split(';'))),set(map(int,r.casos_test.split(';')))) for r in frames['particiones_externas'].itertuples()}
    searches=frames['busquedas_hiperparametros']
    if len(searches)!=len(outer)*len(FAMILIES) or searches.duplicated(['protocolo','fold','modelo']).any():
        raise AssertionError('Búsquedas internas incompletas o duplicadas')
    for row in searches.itertuples():
        train,test=outer[(row.protocolo,row.fold)]
        if set(map(int,row.casos_entrenamiento.split(';')))!=train or set(map(int,row.casos_entrenamiento.split(';')))&test:
            raise AssertionError('El tuning accedió al fold externo')
        params=json.loads(row.best_params_json)
        if set(params)!=set(fixed_specs(root)[row.modelo]['grid']) or row.n_candidates<1 or row.n_inner_splits<2:
            raise AssertionError('Hiperparámetros o particiones internas inválidos')
    for (p,model,fold),g in selection.groupby(['protocolo','modelo','fold']):
        eligible=g[g.no_constante_train].sort_values(['PFI_interna_media_N','orden_original'],ascending=[False,True])
        expected=set(eligible.head(TOP_K).variable);actual=set(g.loc[g.seleccionada,'variable'])
        if actual!=expected or len(actual)!=TOP_K:raise AssertionError('Selección no coincide con PFI interna')
        counts['selecciones']+=1
    for row in frames['particiones_internas'].itertuples():
        tr=set(map(int,row.casos_train.split(';')));va=set(map(int,row.casos_validacion.split(';')));ot,ov=outer[(row.protocolo,row.fold)]
        if tr&va or tr|va!=ot or (tr|va)&ov:raise AssertionError('Contaminación de selección interna')
    for kind in ['externos','internos']:
        d=pd.read_csv(out/'tables'/f'donantes_{kind}.csv.gz')
        keys=['protocolo']+(['fold'] if kind=='internos' else [])+['repeticion']
        for key,g in d.groupby(keys):
            if sorted(g.caso_receptor)!=sorted(g.caso_donante):raise AssertionError('Donantes no forman permutación')
            if kind=='internos':
                ot,ov=outer[(key[0],key[1])]
                if set(g.caso_donante)!=ot or set(g.caso_donante)&ov:raise AssertionError('Donantes internos fuera de train')
            counts['donantes']+=1
    reps=frames['importancia_repeticiones'];summaries=frames['importancia_resumen']
    keys=['protocolo','modelo','conjunto','tipo','variable_grupo']
    for key,g in reps.groupby(keys):
        if len(g)!=N_REPEATS or not np.allclose(g.MAE_permutado_N-g.MAE_original_N,g.delta_MAE_N,atol=1e-10):raise AssertionError('Repeticiones PFI inconsistentes')
        s=summaries
        for col,value in zip(keys,key):s=s[s[col]==value]
        if len(s)!=1 or not np.isclose(s.delta_MAE_media_N.iloc[0],g.delta_MAE_N.mean(),atol=1e-10):raise AssertionError('Resumen PFI inconsistente')
        if key[1]=='Mediana' and not np.allclose(g.delta_MAE_N,0,atol=1e-12):raise AssertionError('Mediana debe tener importancia nula')
        counts['repeticiones_PFI']+=len(g)
    if check_raw:
        chunks=[]
        for chunk in pd.read_csv(out/'tables/predicciones_permutadas.csv.gz',chunksize=60000):
            chunk['error_permutado']=np.abs(chunk.observado_N-chunk.predicho_permutado_N)
            ag=chunk.groupby(keys+['repeticion']).error_permutado.agg(['sum','count']).reset_index();chunks.append(ag)
        agg=pd.concat(chunks).groupby(keys+['repeticion'])[['sum','count']].sum().reset_index();agg['from_raw']=agg['sum']/agg['count']
        merged=reps.merge(agg,on=keys+['repeticion'],validate='one_to_one')
        if len(merged)!=len(reps) or not merged['count'].eq(26).all() or not np.allclose(merged.MAE_permutado_N,merged.from_raw,atol=1e-9):raise AssertionError('Predicciones por repetición no reproducen PFI')
        counts['predicciones_permutadas']=int(agg['count'].sum())
    return {'status':'PASS','checks':counts,'counts':m['counts'],'raw_predictions_checked':check_raw,
            'no_training_during_verification':True,'estimators_present':present_estimators,
            'estimators_missing':len(missing_estimators),'missing_estimator_files':missing_estimators,
            'require_estimators':require_estimators,'estimators_reloaded_during_verification':False}


def ensure_outputs(root=ROOT,output=None,*,require_estimators=False):
    """Verify/cacheload public tables; fit only when no completed cache exists."""
    root=Path(root).resolve();out=_output(root,output)
    if (out/'manifest.json').exists():
        verify_outputs(root,out,require_estimators=require_estimators);return _load_tables(out)
    return run(root,out)


if __name__=='__main__':
    # Fit through the importable module so joblib never records custom classes
    # as __main__.SelectOriginal / __main__.ReferenceRegressor.
    from importlib import import_module
    canonical=import_module('src.importancia_permutacion')
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--output',type=Path)
    parser.add_argument('--verify',action='store_true');parser.add_argument('--verify-raw',action='store_true')
    parser.add_argument('--require-estimators',action='store_true')
    args=parser.parse_args()
    if args.verify or args.verify_raw:print(json.dumps(canonical.verify_outputs(args.root,args.output,check_raw=args.verify_raw,require_estimators=args.require_estimators),ensure_ascii=False),flush=True)
    else:canonical.run(args.root,args.output)
