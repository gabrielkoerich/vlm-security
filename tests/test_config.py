from pathlib import Path

import pytest

from vlmsec import config


def test_env_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DVR_HOST", "dvr.example.com")
    assert config.get("DVR_HOST", "dvr.local") == "dvr.example.com"


def test_file_format(tmp_path: Path) -> None:
    f = tmp_path / "conf"
    f.write_text("# comment\nA=1\n  B = two  \nnot a pair\n")
    assert config.load(f) == {"A": "1", "B": "two"}
    assert config.load(tmp_path / "missing") == {}


def test_require_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLMSEC_NOPE", raising=False)
    with pytest.raises(RuntimeError):
        config.require("VLMSEC_NOPE")


def test_passbox_only_when_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake(field: str, box: str = "", timeout: float = 0) -> str:
        _ = box, timeout
        calls.append(field)
        return "x"

    monkeypatch.delenv("HN_USE_PASSBOX", raising=False)
    monkeypatch.delenv("VLMSEC_UNSET", raising=False)
    monkeypatch.setattr(config, "passbox_field", fake)
    assert config.secret("VLMSEC_UNSET", field="f") == ""
    assert calls == [], "a daemon that never opts in can never reach passbox"
    assert config.secret("VLMSEC_UNSET", field="f", allow_passbox=True) == "x"
    assert calls == ["f"]
