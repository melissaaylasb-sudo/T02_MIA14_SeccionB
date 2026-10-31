"""Comprobaciones de la entrega y de decisiones que no deben mezclar modelos."""
from pathlib import Path
import json
import nbformat
import pandas as pd
from src.revision_entrega_permutacion import evaluar_hipotesis,FAMILIES

ROOT=Path(__file__).resolve().parents[1]

def test_hipotesis_exigen_mismo_modelo_e_interseccion(tmp_path):
    tables=tmp_path/'reports/permutation_importance/tables';tables.mkdir(parents=True)
    cfg=tmp_path/'config';cfg.mkdir()
    (cfg/'criterios_entrega_permutacion.json').write_text('{"tolerancia_numerica_N":1e-10}')
    metrics=[];reduced=[];pfi=[];abl=[]
    for i,model in enumerate(FAMILIES):
        for p in ['configuracion','familia']:
            # Different winners in each protocol cannot satisfy H1 or H2.
            win=(i==0 and p=='configuracion') or (i==1 and p=='familia')
            metrics.append({'protocolo':p,'modelo':model,'conjunto':'todas',
                'delta_MAE_vs_mediana_N':-1 if win else 1,'delta_MAE_vs_media_nivel_N':-1 if win else 1})
            reduced.append({'protocolo':p,'modelo':model,'delta_MAE_reducir_N':-1 if win else 1})
        # Each test separately has 3/5 positive; their intersection has only one.
        pfi.append({'protocolo':'configuracion','modelo':model,'conjunto':'todas','tipo':'grupo',
            'variable_grupo':'arquitectura_volumenes','delta_MAE_media_N':1 if i<3 else -1})
        abl.append({'protocolo':'configuracion','modelo':model,'bloque':'arquitectura_volumenes',
            'delta_MAE_ablacion_N':1 if i>=2 else -1})
    for name,rows in [('metricas_resumen',metrics),('comparacion_todas_prioritarias',reduced),
                       ('importancia_resumen',pfi),('ablacion_bloque',abl)]:
        pd.DataFrame(rows).to_csv(tables/f'{name}.csv',index=False)
    result=evaluar_hipotesis(tmp_path).set_index('Hipótesis')
    assert result.loc['General','Modelos que cumplen']==1
    assert result.loc['H1','Modelos que cumplen']==0
    assert result.loc['H2','Modelos que cumplen']==0
    assert result.loc['H3','Modelos que cumplen']==1
    assert all(result.loc[k,'Lectura'].startswith('No respaldada') for k in ['H1','H2','H3'])

def test_notebook_ejecutado_corresponde_a_permutacion():
    notebook=nbformat.read(ROOT/'notebooks/06_Modelado_Tuning_SHAP.ipynb',as_version=4)
    source='\n'.join(c.source for c in notebook.cells)
    assert 'shap' not in source.lower()
    assert 'GridSearchCV' not in source and 'RandomizedSearchCV' not in source
    assert "matriz_consistencia.csv" in source
    assert 'figura(18)' in source and 'figura(19)' in source
    assert source.index('## 3. Desempeño') < source.index('## 5. Importancia')
    for cell in notebook.cells:
        if cell.cell_type=='code':
            assert cell.execution_count is not None
            assert not any(o.output_type=='error' for o in cell.outputs)
    execution=json.loads((ROOT/'reports/permutation_importance/logs/notebook_ejecutado.json').read_text())
    assert execution['status']=='passed' and execution['kernel']=='clean'

def test_figuras_preservan_tablas_numericas_y_restricciones():
    manifest=json.loads((ROOT/'reports/permutation_importance/figures/manifest_figuras.json').read_text(encoding='utf-8'))
    text=json.dumps(manifest,ensure_ascii=False)
    assert '26 casos' not in text and 'Fuente:' not in text
    # The precise manifest schema is additionally checked by its figure generator.
    assert '18_modelos_todas_variables' in text and '19_modelos_variables_prioritarias' in text
    assert 'numeric_table' in text and 'MSE' in text and 'RMSE' in text
