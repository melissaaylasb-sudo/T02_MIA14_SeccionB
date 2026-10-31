"""Semantic checks: honest selection, global OOF perturbation and joint groups."""
import numpy as np
import pandas as pd

from src import importancia_permutacion as pfi


def test_module_cli_serializes_importable_classes_in_another_process(tmp_path):
    import subprocess
    import sys
    import joblib
    destination=tmp_path/'classes.joblib'
    script='''
import runpy,sys,joblib
from src import importancia_permutacion as canonical
destination=sys.argv[1]
canonical.run=lambda *args: joblib.dump([canonical.SelectOriginal(['n_cells_Y']),canonical.ReferenceRegressor('Mediana')],destination)
sys.argv=['src.importancia_permutacion','--root',str(destination)+'_nonexistent']
runpy.run_module('src.importancia_permutacion',run_name='__main__')
'''
    subprocess.run([sys.executable,'-X','utf8','-c',script,str(destination)],cwd=pfi.ROOT,check=True,capture_output=True,text=True)
    objects=joblib.load(destination)
    assert all(type(obj).__module__=='src.importancia_permutacion' for obj in objects)


def geometry(n=15):
    x=pd.DataFrame({c:np.arange(1,n+1,dtype=float)+(j+1)*.3 for j,c in enumerate(pfi.GEOMETRY)})
    x['n_cells_Y']=np.tile([4,5,6],int(np.ceil(n/3)))[:n]
    x['Pivot_total_number']=np.arange(n)+10
    x['Pivot_radius']=1.0
    x.index=np.arange(101,101+n)
    return x


def test_group_permutation_uses_one_joint_donor_and_preserves_other_columns():
    x=geometry(3);x['Fiber_total_volume']=[10.,20.,30.];x['Pivot_total_volume']=[90.,80.,70.]
    original=x.copy(deep=True);donors=np.array([[2,0,1],[1,2,0]])
    receivers=np.array([0,2]);columns=['Fiber_total_volume','Pivot_total_volume']
    batch=pfi.perturb_batch(x,receivers,donors,columns)
    for r in range(2):
        for k,i in enumerate(receivers):
            row=batch.iloc[r*len(receivers)+k]
            np.testing.assert_array_equal(row[columns],x.iloc[donors[r,i]][columns])
            rest=[c for c in x if c not in columns]
            np.testing.assert_array_equal(row[rest],x.iloc[i][rest])
            assert row.Fiber_total_volume+row.Pivot_total_volume==100
    pd.testing.assert_frame_equal(x,original)


def test_crossfit_global_permutation_is_nontrivial_for_singleton_test_folds():
    x=geometry(6);x['n_cells_Y']=np.arange(1,7)
    class Identity:
        def predict(self,frame):return frame.n_cells_Y.to_numpy()
    fitted=[(np.array([i]),Identity(),('n_cells_Y',)) for i in range(len(x))]
    original=pfi.crossfit_predictions(x,fitted)
    donors=np.arange(len(x)-1,-1,-1)[None,:]
    result=pfi.crossfit_permuted_predictions(x,fitted,donors,['n_cells_Y'],original)
    np.testing.assert_array_equal(original,x.n_cells_Y)
    np.testing.assert_array_equal(result[0],original[::-1])
    assert np.abs(result[0]-original).mean()==3


def test_inner_selection_fits_only_external_training_and_keeps_groups_separate():
    full=geometry(18);y=pd.Series(3*full.Fiber_total_length+2*full.n_cells_Y,index=full.index)
    external_train=full.index[:15];external_test=set(full.index[15:])
    groups=np.repeat(np.arange(9),2)
    specs={'Ridge':{'parametros':{'alpha':1.,'solver':'svd'}}}
    audit=[]
    first=pfi.select_top_variables(full.loc[external_train],y.loc[external_train],groups[:15],
                                   'Ridge',specs,seed=123,audit=audit)
    y.loc[list(external_test)]=1e12
    second=pfi.select_top_variables(full.loc[external_train],y.loc[external_train],groups[:15],
                                    'Ridge',specs,seed=123)
    assert first[0]==second[0] and len(first[0])==4 and 'Pivot_radius' not in first[0]
    for record in audit:
        used=set(map(int,record['casos_entrenamiento'].split(';')))
        assert used<=set(external_train) and not used&external_test
    lookup=dict(zip(full.index,groups))
    validation=[]
    for part in first[3]:
        tr=set(map(int,part['casos_train'].split(';')));va=set(map(int,part['casos_validacion'].split(';')))
        assert tr|va==set(external_train) and not tr&va
        assert not {lookup[i] for i in tr}&{lookup[i] for i in va}
        validation.extend(va)
    assert sorted(validation)==sorted(external_train)


def test_top_four_is_prespecified_even_when_all_scores_are_negative(monkeypatch):
    x=geometry(15);groups=np.arange(15);y=np.repeat(10.,15)
    class Zero:
        def fit(self,X,y):return self
        def predict(self,X):return np.zeros(len(X))
    monkeypatch.setattr(pfi,'fixed_pipeline',lambda *args:Zero())
    monkeypatch.setattr(pfi,'crossfit_permuted_predictions',lambda X,fitted,donors,columns,original=None:np.ones((len(donors),len(X))))
    chosen,rows,*_=pfi.select_top_variables(x,y,groups,'Ridge',{},seed=123)
    assert chosen==tuple(pfi.GEOMETRY[:4])
    selected=[r for r in rows if r['seleccionada']]
    assert all(r['PFI_interna_media_N']==-1 for r in selected)
    assert all(r['seleccion_aun_con_PFI_no_positiva'] for r in selected)


def test_reference_uses_training_mean_for_unknown_level_and_median_is_invariant():
    x=geometry(6);y=np.array([1.,3.,9.,11.,21.,23.])
    train=x.iloc[:4];yt=y[:4]
    mean=pfi.ReferenceRegressor('Media_por_nivel').fit(train,yt)
    new=x.iloc[[4]].copy();new['n_cells_Y']=99
    assert mean.predict(new)[0]==yt.mean()
    median=pfi.ReferenceRegressor('Mediana').fit(train,yt)
    assert np.all(median.predict(x)==np.median(yt))
    np.testing.assert_array_equal(pfi.permutation_donors(6,3,42),pfi.permutation_donors(6,3,42))
