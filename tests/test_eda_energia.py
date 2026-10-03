"""Pruebas de extracción energética sin ejecución de expresiones MATLAB."""

from pathlib import Path

import numpy as np
import pytest

from src.eda_energia import load_energy_tables


SOURCE = """
Case_n = (1:2)';
Energy_Dissipated = NaN(2,3);
Energy_Fracture = NaN(2,2);
Energy_Dissipated(1,1:3) = [-0.2, 1.3e+2 NaN]; % conservar signo
Energy_Dissipated(2,2) = [4];
Energy_Fracture(1,1:2) = [12 2.5];
"""


def test_numeric_extraction_preserves_nan_sign_and_index(tmp_path):
    source = tmp_path / "energy.m"
    source.write_text(SOURCE, encoding="utf-8")
    tables = load_energy_tables(source)
    dissipated = tables["Energy_Dissipated"]
    assert list(dissipated.index) == [1, 2]
    assert dissipated.index.name == "Case_n"
    assert list(dissipated.columns) == ["cycle_1", "cycle_2", "cycle_3"]
    assert dissipated.loc[1, "cycle_1"] == -0.2
    assert dissipated.loc[1, "cycle_2"] == 130
    assert np.isnan(dissipated.loc[1, "cycle_3"])
    assert np.isnan(dissipated.loc[2, "cycle_1"])
    assert dissipated.loc[2, "cycle_2"] == 4
    assert list(tables["Energy_Fracture"].columns) == ["event_1", "event_2"]
    assert dissipated.attrs["unit"] is None


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("NaN(2,3)", "NaN(3,3)"),
        ("NaN(2,3)", "NaN(2,0)"),
        ("(1,1:3)", "(3,1:3)"),
        ("(1,1:3)", "(1,0:2)"),
        ("(1,1:3)", "(1,3:1)"),
        ("(1,1:3)", "(1,1:4)"),
        ("[-0.2, 1.3e+2 NaN]", "[1 2]"),
        ("[-0.2, 1.3e+2 NaN]", "[1 2 system('anything')]"),
        ("[-0.2, 1.3e+2 NaN]", "[1+2 2 NaN]"),
        ("[-0.2, 1.3e+2 NaN]", "[1,,2 NaN]"),
        ("[-0.2, 1.3e+2 NaN]", "[1; 2; NaN]"),
        ("[-0.2, 1.3e+2 NaN]", "[1e999 2 NaN]"),
        ("(1:2)'", "[1;1]"),
        ("(1:2)'", "[1;1.5]"),
        ("(1:2)'", "[1 2;3 4]"),
        ("(1:2)'", "(999999999999999999999:999999999999999999999)'"),
    ],
)
def test_rejects_incompatible_or_nonliteral_data(tmp_path, old, new):
    source = tmp_path / "bad.m"
    source.write_text(SOURCE.replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError):
        load_energy_tables(source)


def test_rejects_repeated_cell_including_explicit_nan(tmp_path):
    source = tmp_path / "duplicate.m"
    source.write_text(SOURCE + "Energy_Dissipated(1,3) = [4];", encoding="utf-8")
    with pytest.raises(ValueError, match="repetida"):
        load_energy_tables(source)


def test_rejects_values_before_initialization(tmp_path):
    source = tmp_path / "order.m"
    source.write_text("Energy_Dissipated(1,1) = [1];\n" + SOURCE, encoding="utf-8")
    with pytest.raises(ValueError, match="inicializarse"):
        load_energy_tables(source)


def test_reads_campaign_known_cells_without_converting_missing_values():
    path = Path(__file__).resolve().parents[1] / "data/raw/dati_campagna_venditti.m"
    tables = load_energy_tables(path)
    dissipated = tables["Energy_Dissipated"]
    fracture = tables["Energy_Fracture"]
    assert dissipated.loc[3, "cycle_1"] == -0.2101
    assert dissipated.loc[20, "cycle_7"] == 372.1816
    assert dissipated.loc[2].isna().all()
    assert np.isnan(fracture.loc[5, "event_4"])
    assert fracture.loc[22, "event_1"] == 2939.3
    assert fracture.loc[7, "event_13"] == 64.2911
