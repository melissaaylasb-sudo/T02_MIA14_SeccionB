"""Regresión de publicación atómica ante bloqueos transitorios de Windows."""

import os
from pathlib import Path
import time

import pytest

from src.eda import write_bytes_atomic


@pytest.mark.parametrize("failures_before_success", [1, 3])
def test_atomic_write_retries_and_cleans_temporary(tmp_path, monkeypatch, failures_before_success):
    destination = tmp_path / "manifest.json"
    previous = b"previous complete artifact"
    payload = bytes(range(256)) * 4
    destination.write_bytes(previous)
    real_replace = os.replace
    attempts, waits = [], []

    def temporarily_blocked(source, target):
        source, target = Path(source), Path(target)
        assert source != target and source.parent == destination.parent
        assert source.read_bytes() == payload
        assert destination.read_bytes() == previous
        attempts.append(source)
        if len(attempts) <= failures_before_success:
            raise OSError(22, "Simulated transient Windows lock")
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", temporarily_blocked)
    monkeypatch.setattr(time, "sleep", waits.append)
    write_bytes_atomic(destination, payload)

    assert len(attempts) == failures_before_success + 1
    assert len(waits) == failures_before_success and all(delay > 0 for delay in waits)
    assert destination.read_bytes() == payload
    assert all(not temporary.exists() for temporary in attempts)
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_write_preserves_destination_and_cleans_temporary_on_persistent_failure(tmp_path, monkeypatch):
    destination = tmp_path / "figure.png"
    previous, payload = b"previous complete figure", b"replacement figure"
    destination.write_bytes(previous)
    attempts = []

    def persistently_blocked(source, target):
        source = Path(source)
        assert Path(target) == destination
        assert source.read_bytes() == payload
        assert destination.read_bytes() == previous
        attempts.append(source)
        raise OSError(22, "Simulated persistent Windows lock")

    monkeypatch.setattr(os, "replace", persistently_blocked)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    with pytest.raises(OSError, match="persistent Windows lock") as error:
        write_bytes_atomic(destination, payload)

    assert error.value.errno == 22
    assert len(attempts) == 4
    assert destination.read_bytes() == previous
    assert all(not temporary.exists() for temporary in attempts)
    assert list(tmp_path.iterdir()) == [destination]
