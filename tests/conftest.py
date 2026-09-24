from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def nybble_home(tmp_path, monkeypatch) -> Path:
    """An isolated Nybble data directory. HOME is redirected so nothing on the
    real machine is probed."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    root = tmp_path / "nybble"
    monkeypatch.setenv("NYBBLE_HOME", str(root))
    return root
