import numpy as np
import pandas as pd
import pytest

from src.transformaciones_eda import GEOMETRY, geometry_features, make_preprocessor, representation_diagnostics


def fixture_geometry():
    frame = pd.DataFrame(np.ones((4, len(GEOMETRY))), columns=GEOMETRY)
    frame["Fiber_base"] = [2., 3., 4., 5.]
    frame["Fiber_height"] = [1., 2., 3., 4.]
    return frame


def test_geometric_formulas_and_temporal_isolation():
    x = fixture_geometry()
    original = x.copy()
    derived = geometry_features(x)
    np.testing.assert_allclose(derived.area_fibra_mm2, [2, 6, 12, 20])
    np.testing.assert_allclose(derived.I_fibra_bh3_mm4, x.Fiber_base * x.Fiber_height ** 3 / 12)
    np.testing.assert_allclose(derived.fraccion_volumen_pivotes, .5)
    x["First_Failure_Load_N"] = [10, 1000, -1, np.nan]
    pd.testing.assert_frame_equal(derived, geometry_features(x))
    pd.testing.assert_frame_equal(original, x[GEOMETRY])


def test_nonphysical_denominators_and_missing_features_rejected():
    x = fixture_geometry()
    x.loc[0, "Pivot_radius"] = 0
    with pytest.raises(ValueError, match="positivos"):
        geometry_features(x)
    with pytest.raises(ValueError, match="Faltan"):
        geometry_features(x.drop(columns="Fiber_base"))


@pytest.mark.parametrize("scale", ["standard", "robust"])
def test_reserved_extreme_does_not_change_fitted_preprocessor(scale):
    train = pd.DataFrame({"dimension": [1., 2., 3., np.nan], "constant": [1.] * 4})
    pipe = make_preprocessor(scale).fit(train)
    before = pipe.named_steps["imputer"].statistics_.copy()
    reserved = pd.DataFrame({"dimension": [1e9, np.nan], "constant": [1., 1.]})
    got = pipe.transform(reserved)
    np.testing.assert_array_equal(before, pipe.named_steps["imputer"].statistics_)
    assert got.shape[1] == 1 and np.isfinite(got).all()
    assert got[0, 0] > 1e6


def test_redundancy_detected_without_response():
    x = pd.DataFrame({"a": [1., 2., 3., 4.], "b": [3., 5., 7., 9.], "constant": [5.] * 4})
    d = representation_diagnostics(x)
    assert d["rango_centrado"] == 1 and d["constantes"] == 1
    assert np.isinf(d["condicion"])


def test_roundoff_from_large_offset_does_not_create_dimension():
    a = np.linspace(1, 2, 15)
    x = pd.DataFrame({"a": a, "b": 6500 - a})
    diagnostics = representation_diagnostics(x)
    assert diagnostics["rango_centrado"] == 1
    assert diagnostics["tolerancia_relativa"] == 1e-10
