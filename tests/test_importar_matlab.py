"""Contrato de ingesta conservado del flujo original del proyecto."""

import numpy as np

from src.importar_matlab import load_matlab_samples


def test_matlab_samples_parser_reads_only_declared_numeric_table(tmp_path):
    source = """
    Case_n = (1:3)';
    geometry = [1.5; 2.5; 3.5];
    response = [10; NaN; 30];
    constant = repmat(0.5,3,1);
    ignored = system('echo this must never run');
    samples = table(Case_n, geometry, response, constant);
    """
    path = tmp_path / "fixture.m"
    path.write_text(source, encoding="utf-8")
    frame = load_matlab_samples(path)
    assert frame.columns.tolist() == ["Case_n", "geometry", "response", "constant"]
    assert frame["Case_n"].tolist() == [1, 2, 3]
    assert np.isnan(frame.loc[1, "response"])
    assert frame["constant"].tolist() == [0.5, 0.5, 0.5]
