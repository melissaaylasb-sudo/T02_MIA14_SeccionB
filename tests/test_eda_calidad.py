"""Controles del auditor: integridad, semántica de nulos y dependencias."""

import numpy as np
import pandas as pd
import pytest

from src.eda_calidad import (
    audit_matlab_literals, compare_tables, configuration_audit,
    duplicate_columns, missing_by_group, missing_patterns,
    profile_variables, robust_flags,
)


def test_comparison_matches_ids_not_positions_and_preserves_nan():
    left = pd.DataFrame({"Case_n": [1, 2], "x": [np.nan, 2.0]})
    right = pd.DataFrame({"Case_n": [2, 1], "x": [2, np.nan]})
    assert compare_tables(left, right).empty
    right.loc[1, "x"] = 0
    result = compare_tables(left, right)
    assert result["Case_n"].tolist() == [1]
    assert result["tipo"].tolist() == ["valor_distinto"]


def test_comparison_exposes_schema_and_identifier_changes():
    left = pd.DataFrame({"Case_n": [1], "x": [2]})
    right = pd.DataFrame({"Case_n": [2], "y": [2]})
    assert set(compare_tables(left, right)["tipo"]) == {
        "columna_solo_izquierda", "columna_solo_derecha",
        "caso_solo_izquierda", "caso_solo_derecha",
    }
    with pytest.raises(ValueError, match="identificadores"):
        compare_tables(pd.concat([left, left]), right)


def test_comparison_normalizes_numeric_case_ids_from_csv_strings():
    left = pd.DataFrame({"Case_n": pd.Series(["1", "2"], dtype="string"),
                         "x": pd.Series([1.25, 3.5], dtype="Float64")})
    right = pd.DataFrame({"Case_n": [2, 1], "x": [3.5, 1.25]})
    assert compare_tables(left, right).empty


def test_profile_separates_zero_negative_missing_and_nonfinite():
    profile = profile_variables(pd.DataFrame({"x": [0, -1, np.nan, np.inf, 2]})).iloc[0]
    assert profile.Ceros == 1
    assert profile.Negativos == 1
    assert profile.Ausentes == 1
    assert profile.No_finitos == 1
    assert profile.Mediana == 0


def test_missing_patterns_and_denominators_use_specimens():
    frame = pd.DataFrame({"a": [np.nan, 1, 2], "b": [np.nan, 1, np.nan]})
    patterns = missing_patterns(frame)
    assert set(patterns.Variables_ausentes) == {"a; b", "ninguna", "b"}
    assert patterns.Frecuencia.sum() == len(frame)
    grouped = missing_by_group(frame, pd.Series([4, 4, 6]))
    selected = grouped.loc[(grouped.Familia == 4) & (grouped.Variable == "a")].iloc[0]
    assert selected.Base_familia == 2
    assert selected["Ausencia_%"] == 50


def test_equal_columns_is_distinct_from_affine_redundancy():
    frame = pd.DataFrame({"a": [1, np.nan, 3], "b": [1, np.nan, 3], "c": [2, np.nan, 6]})
    result = duplicate_columns(frame)
    assert list(zip(result.Variable_1, result.Variable_2)) == [("a", "b")]


def test_partial_configuration_is_not_assumed_replica():
    geometry = ["n_cells_Y", "Fiber_height", "Pivot_height", "Fiber_base"]
    frame = pd.DataFrame({"Case_n": [1, 2], "n_cells_Y": [4, 4],
                          "Fiber_height": [1, 1], "Pivot_height": [1, 1], "Fiber_base": [2, 3]})
    result = configuration_audit(frame, geometry)
    assert result.Nivel.tolist() == ["combinación parcial de factores"]
    assert result.Geometria_que_difiere.tolist() == ["Fiber_base"]


def test_mad_zero_does_not_mean_no_outlier_and_nan_does_not_erase_mad():
    frame = pd.DataFrame({"Case_n": [1, 2, 3, 4], "x": [1, 1, 1, 7], "y": [1, 2, 3, np.nan]})
    result = robust_flags(frame, ["x", "y"])
    assert result.loc[result.Variable == "x", "Bandera_MAD"].isna().all()
    assert result.loc[result.Variable == "y", "MAD"].eq(1).all()


def test_matlab_lexical_audit_rejects_unconsumed_symbols(tmp_path):
    path = tmp_path / "source.m"
    path.write_text("Case_n = (1:2)'; x=[1;abc(2)];", encoding="utf-8")
    with pytest.raises(ValueError, match="no numérico"):
        audit_matlab_literals(path, pd.DataFrame({"Case_n": [1, 2], "x": [1, 2]}))
    path.write_text("Case_n = (1:2)'; x=[1;NaN];", encoding="utf-8")
    report = audit_matlab_literals(path, pd.DataFrame({"Case_n": [1, 2], "x": [1, np.nan]}))
    assert report.Valores_y_NaN_coinciden.all()
