"""Pruebas de límites estadísticos y del contrato geométrico del benchmark."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from src import benchmark_regresion as b
from src.benchmark_regresion_modelos import FullRankBasis, MeanByLevel, freeze_candidates, pipeline, rank_families


def test_rank_basis_preserves_training_predictions_but_not_causal_identification():
    x=np.array([[1.,0.],[2.,1.],[3.,4.],[4.,3.],[5.,2.]])
    design=np.column_stack((x[:,0],2*x[:,0],x[:,1]));centered=design-design.mean(axis=0)
    basis=FullRankBasis().fit(design);reduced=basis.transform(centered)
    assert basis.indices_.tolist()==[0,2]
    y=np.array([2.,-1.,1.,3.,5.])
    np.testing.assert_allclose(centered@np.linalg.lstsq(centered,y,rcond=None)[0],
                               reduced@np.linalg.lstsq(reduced,y,rcond=None)[0],atol=1e-12)


def test_mean_level_unseen_uses_train_global_mean_and_bias_sign():
    ref=MeanByLevel().fit(pd.DataFrame({"n_cells_Y":[4,4,5]}),np.array([2.,4.,12.]))
    np.testing.assert_array_equal(ref.predict(pd.DataFrame({"n_cells_Y":[4,6]})),[3.,6.])
    assert b.metrics(np.array([1.,2.]),np.array([2.,4.]))["sesgo_N"]==1.5


def test_one_newton_band_uses_stability_and_keeps_outside_band_out():
    cfg=b.config(b.ROOT)
    assert rank_families({"SVR_RBF":(10.,3.),"KNN":(10.9,1.),"XGBoost":(11.01,0.)},cfg)==["KNN","SVR_RBF","XGBoost"]


def test_candidates_are_frozen_without_outcomes_and_include_initial():
    cfg=b.config(b.ROOT);a=freeze_candidates(cfg);assert a==freeze_candidates(cfg) and len(a)==5
    for name,values in a.items():
        assert len({b._object(x) for x in values})==8 and values[0]==cfg["parametros_iniciales"][name]
    assert "Lineal_multiple" not in a


def test_leaky_column_and_false_inner_holdout_are_rejected_before_fitting():
    cfg=b.config(b.ROOT);raw=b.load_frozen_data(b.ROOT);X=raw.set_index("Case_n")[b.GEOMETRY];y=raw.set_index("Case_n")[b.TARGET]
    with pytest.raises(ValueError,match="geométricas"):pipeline("KNN",cfg["parametros_iniciales"]["KNN"],cfg).fit(X.assign(post_event=y),y)
    groups=np.repeat(np.arange(13),2);splits=list(GroupKFold(5).split(X,groups=groups));tr,va=splits[0]
    contaminated=[(np.r_[tr,va[0]],va),*splits[1:]]
    with pytest.raises(ValueError,match="Fuga"):b.search_training(X,y,groups,contaminated,cfg,{},{"fold":1},b.defaultdict(list))


def test_inner_folds_are_confined_to_external_train_for_all_protocols():
    raw=b.load_frozen_data(b.ROOT);splits,_,_=b.load_outer_splits(raw,b.ROOT)
    for protocol,outer in splits.items():
        for tr,te in outer:
            folds=b.inner_splits(raw,tr,protocol);valid=[]
            for it,iv in folds:
                assert not set(tr[it])&set(tr[iv]) and not set(tr[np.r_[it,iv]])&set(te)
                assert set(tr[np.r_[it,iv]])==set(tr);valid.extend(tr[iv])
            assert sorted(valid)==sorted(tr)
