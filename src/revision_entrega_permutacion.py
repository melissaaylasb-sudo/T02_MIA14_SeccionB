"""Cotejo independiente y lectura de los contrastes de la actividad."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FAMILIES=['Ridge','ElasticNet','SVR_RBF','RandomForest','GaussianProcess']
NAMES=dict(zip(FAMILIES,['Ridge','Elastic Net','SVR RBF','Bosque aleatorio','Proceso gaussiano']))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(root,name):return pd.read_csv(Path(root)/'reports/permutation_importance/tables'/f'{name}.csv',float_precision='round_trip')

def evaluar_hipotesis(root=ROOT):
    root=Path(root);m=load(root,'metricas_resumen');compare=load(root,'comparacion_todas_prioritarias')
    criteria=json.loads((root/'config/criterios_entrega_permutacion.json').read_text(encoding='utf-8'))
    tol=float(criteria['tolerancia_numerica_N'])
    pfi=load(root,'importancia_resumen');abl=load(root,'ablacion_bloque')
    full=m[m.conjunto.eq('todas')].copy()
    good=full[(full.delta_MAE_vs_mediana_N < -tol)&(full.delta_MAE_vs_media_nivel_N < -tol)]
    hg=set(good.loc[good.protocolo.eq('configuracion'),'modelo'])
    h1={model for model in FAMILIES if set(good.loc[good.modelo.eq(model),'protocolo'])=={'configuracion','familia'}}
    h2={model for model in FAMILIES if len(compare[compare.modelo.eq(model)])==2 and (compare.loc[compare.modelo.eq(model),'delta_MAE_reducir_N']<=tol).all()}
    p=pfi[(pfi.protocolo=='configuracion')&(pfi.conjunto=='todas')&(pfi.tipo=='grupo')&(pfi.variable_grupo=='arquitectura_volumenes')]
    a=abl[(abl.protocolo=='configuracion')&(abl.bloque=='arquitectura_volumenes')]
    h3=set(p.loc[p.delta_MAE_media_N>tol,'modelo'])&set(a.loc[a.delta_MAE_ablacion_N>tol,'modelo'])
    def names(models):return ', '.join(NAMES[x] for x in FAMILIES if x in models) or 'Ninguno'
    records=[('General','Modelo completo con menor MAE que ambas referencias en configuraciones',hg,len(hg)>=1),
             ('H1','Mismo modelo completo supera ambas referencias en ambos protocolos',h1,len(h1)>=1),
             ('H2','Mismo modelo reducido no incrementa MAE en ambos protocolos',h2,len(h2)>=1),
             ('H3','PFI y ablación positivas simultáneamente en al menos tres modelos completos',h3,len(h3)>=3)]
    result=pd.DataFrame([{'Hipótesis':key,'Criterio':criterion,'Resultado':names(models),'Modelos que cumplen':len(models),
                          'Lectura':'Respaldada descriptivamente bajo el protocolo' if passes else 'No respaldada bajo el protocolo'} for key,criterion,models,passes in records])
    return result

def verify_and_document(root=ROOT):
    root=Path(root);out=root/'reports/permutation_importance';checks=[]
    def check(value,name):
        assert bool(value),name
        checks.append(name)
    data=pd.read_csv(root/'data/processed/preliminary/model_table.csv',float_precision='round_trip').set_index('Case_n')
    criteria=json.loads((root/'config/criterios_entrega_permutacion.json').read_text(encoding='utf-8'))
    check(criteria['modelos_fijos']==FAMILIES,'Catálogo de la matriz coincide con los cinco modelos evaluados')
    check(criteria['matriz_csv_sha256']==sha(root/'docs/matriz_consistencia.csv'),'Matriz canónica coincide con criterios registrados')
    check(criteria['configuracion_modelos_sha256']==sha(root/'config/comparacion_modelos.json'),'Parámetros fijos coinciden con criterios')
    pred=load(root,'predicciones_oof');metrics=load(root,'metricas_resumen');folds=load(root,'particiones_externas')
    check(len(pred)==624,'Cobertura externa completa')
    for row in folds.itertuples():
        tr=list(map(int,row.casos_train.split(';')));te=list(map(int,row.casos_test.split(';')))
        check(set(tr).isdisjoint(te),'Separación '+row.protocolo+'/'+str(row.fold))
        groupcols=['n_cells_Y','Fiber_height','Pivot_height'] if row.protocolo=='configuracion' else ['n_cells_Y']
        check(set(map(tuple,data.loc[tr,groupcols].to_numpy())).isdisjoint(map(tuple,data.loc[te,groupcols].to_numpy())),'Reserva de grupo '+row.protocolo+'/'+str(row.fold))
        for baseline in ['Mediana','Media_por_nivel']:
            g=pred[(pred.protocolo==row.protocolo)&(pred.fold==row.fold)&(pred.modelo==baseline)]
            expected=[]
            for case in g.Case_n:
                if baseline=='Mediana':value=data.loc[tr,'First_Failure_Load_N'].median()
                else:
                    same=[c for c in tr if data.loc[c,'n_cells_Y']==data.loc[case,'n_cells_Y']]
                    value=data.loc[same or tr,'First_Failure_Load_N'].mean()
                expected.append(value)
            check(np.allclose(g.predicho_N,expected,rtol=1e-12,atol=1e-10),'Referencia independiente '+row.protocolo+'/'+str(row.fold)+'/'+baseline)
    for row in metrics.itertuples():
        g=pred[(pred.protocolo==row.protocolo)&(pred.modelo==row.modelo)&(pred.conjunto==row.conjunto)]
        y=data.loc[g.Case_n,'First_Failure_Load_N'].to_numpy();e=y-g.predicho_N.to_numpy()
        check(np.allclose(y,g.observado_N,atol=1e-12,rtol=0),'Respuesta original '+row.protocolo+'/'+row.modelo+'/'+row.conjunto)
        expected=[np.abs(e).mean(),np.mean(e**2),np.sqrt(np.mean(e**2)),1-np.sum(e**2)/np.sum((y-y.mean())**2)]
        check(np.allclose(expected,[row.MAE_N,row.MSE_N2,row.RMSE_N,row.R2],rtol=1e-11,atol=1e-9),'Métricas independientes '+row.protocolo+'/'+row.modelo+'/'+row.conjunto)
    old=pd.read_csv(root/'reports/feature_engineering/tables/comparacion_modelos.csv')
    full=metrics[metrics.conjunto=='todas']
    matched=full.merge(old,on=['protocolo','modelo'],suffixes=('_actual','_previo'),validate='one_to_one')
    check(len(matched)==10,'Cinco modelos previos en los dos protocolos')
    check(np.allclose(matched.MAE_N_actual,matched.MAE_N_previo,atol=1e-9),'Mismos parámetros y métricas completas anteriores')
    selections=load(root,'seleccion_variables')
    check(not selections.loc[selections.seleccionada,'variable'].eq('Pivot_radius').any(),'No seleccionar el radio constante')
    pfi=load(root,'importancia_resumen');reps=load(root,'importancia_repeticiones')
    keys=['protocolo','modelo','conjunto','tipo','variable_grupo']
    for key,g in reps.groupby(keys):
        expected=g.delta_MAE_N.to_numpy();summary=pfi
        for col,value in zip(keys,key):summary=summary[summary[col]==value]
        r=summary.iloc[0]
        check(np.allclose([expected.mean(),expected.std(ddof=1),*np.quantile(expected,[.05,.95])],[r.delta_MAE_media_N,r.delta_MAE_DE_N,r.p05_N,r.p95_N],atol=1e-9),'Dispersión PFI '+str(key))
    contrasts=evaluar_hipotesis(root);contrasts.to_csv(out/'tables/contrastes_hipotesis.csv',index=False)
    matrix=pd.read_csv(root/'docs/matriz_consistencia.csv')
    check(len(matrix)==4 and matrix.notna().all().all(),'Matriz completa y alineada')
    check(not matrix.to_string().lower().count('shap'),'Matriz corresponde a permutación')
    report={'status':'PASS','checks':len(checks),'details':checks,'script_sha256':sha(__file__),
       'input_manifest_sha256':sha(out/'manifest.json'),'no_model_fitting':True,
       'criteria_sha256':sha(root/'config/criterios_entrega_permutacion.json'),'matrix_sha256':sha(root/'docs/matriz_consistencia.csv'),
       'backup':'C:/Users/MELISSA/Downloads/Respaldos_pantographic/antes_permutacion_20261030_5092a8e3.zip',
       'original_index_sha256':'5d1e5728fce934f22db746ae6ab7644071248dc21b12db08f04eddf1e12b4a72',
       'original_lock_sha256':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'}
    (out/'logs/revision_independiente.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rows=[]
    for r in metrics.itertuples():
        rows.append(f'| {r.protocolo} | {NAMES.get(r.modelo,r.modelo)} | {r.conjunto} | {r.MAE_N:.3f} | {r.MSE_N2:.3f} | {r.RMSE_N:.3f} | {r.R2:.3f} |')
    text='''# Análisis de importancia por permutación

Esta entrega compara cinco familias con los parámetros fijos ya evaluados en `config/comparacion_modelos.json`, junto con mediana y media por nivel estimadas en entrenamiento. No se ejecuta búsqueda de hiperparámetros. El notebook reúne resultados, interpretaciones y matriz.

La selección de cuatro entradas se realiza para cada modelo dentro de su entrenamiento externo: cinco particiones internas por configuración y veinte permutaciones marginales cruzadas. La misma regla se usa al reservar configuraciones y al reservar familias. Se conservan cuatro entradas no constantes incluso si una importancia interna no es positiva; no existe un subconjunto universal declarado relevante.

La evaluación de importancia utiliza cien permutaciones marginales cruzadas sobre la matriz de predicciones externas. Se mantiene el modelo de cada receptor, se permutan solo sus entradas y se conservan los valores negativos. El donante puede pertenecer al entrenamiento del modelo receptor; no se presenta como un test común independiente. La desviación estándar y los percentiles describen variación Monte Carlo, no intervalos poblacionales.

Los grupos se declaran antes del cálculo. Para arquitectura y volúmenes se añade una ablación con reajuste en las mismas particiones. Los descriptores correlacionados y las combinaciones geométricas sintéticas limitan cualquier interpretación causal.

## Resultados externos

| Protocolo | Modelo | Entradas | MAE [N] | MSE [N²] | RMSE [N] | R² |
|---|---|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'\n\n## Contrastes de la matriz\n\n'
    text+='| Hipótesis | Modelos que cumplen | Lectura |\n|---|---|---|\n'
    for _,r in contrasts.iterrows():text+=f'| {r["Hipótesis"]} | {r["Resultado"]} | {r["Lectura"]} |\n'
    text+='''
Los cortes de cero y la mayoría de tres de cinco expresan dirección y concordancia descriptivas. No son pruebas de significancia, equivalencia estadística ni umbrales de utilidad mecánica. Los criterios se documentaron después del EDA: no se presentan como prerregistro.

## Reproducción

```powershell
python -m pip install -r requirements-permutation-importance.txt
python -m src.importancia_permutacion
python -m src.revision_entrega_permutacion
python -m src.entrega_permutacion
python -m pytest tests/test_importancia_permutacion.py tests/test_entrega_permutacion.py tests/test_sprint_importancia.py -q
```

El módulo de análisis reutiliza una ejecución terminada solo si sus hashes y verificaciones concuerdan. Una modificación del protocolo requiere otra salida versionada. El notebook se ejecuta con kernel limpio y genera figuras desde los CSV comprobados. Los antecedentes de otras representaciones, tuning y explicaciones permanecen en sus rutas históricas; no se usan como resultados nuevos de esta actividad.

La procedencia declarada por la usuaria se conserva en `procedencia_declarada.json`. Las condiciones de uso y publicación y las aclaraciones de la fuente sobre carga terminal, geometría y ciclos continúan pendientes.

Referencias: [importancia por permutación](https://scikit-learn.org/1.7/modules/permutation_importance.html), [variables correlacionadas](https://scikit-learn.org/1.7/auto_examples/inspection/plot_permutation_importance_multicollinear.html).
'''
    (root/'docs/analisis_permutacion.md').write_text(text,encoding='utf-8')
    trace='# Trazabilidad de la actividad de permutación\n\nLas formulaciones vigentes son las de [la matriz](matriz_consistencia.md); se distinguen de los contrastes históricos de ingeniería y tuning. Todas las filas siguientes se calculan desde la nueva ejecución.\n\n| Hipótesis | Contraste | Resultado | Lectura |\n|---|---|---|---|\n'
    for _,r in contrasts.iterrows():trace+=f'| {r["Hipótesis"]} | {r["Criterio"]} | {r["Resultado"]} | {r["Lectura"]} |\n'
    trace+='\nEvidencia: `reports/permutation_importance/tables/contrastes_hipotesis.csv`, métricas OOF, selección interna, permutación repetida y ablación. El estudio sigue siendo exploratorio y la revisión no resuelve las aclaraciones pendientes de la fuente.\n'
    (root/'docs/trazabilidad_objetivos_resultados.md').write_text(trace,encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='details'},ensure_ascii=False,indent=2))
    return report

if __name__=='__main__':verify_and_document()
