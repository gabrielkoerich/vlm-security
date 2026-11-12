from pathlib import Path

from vlmsec.model_reload import after_reload, busy, judge, may_trim, runners

LIMIT = 7168


def test_judge_over_and_blind() -> None:
    files = {"big": "/m/big", "small": "/m/small"}
    assert judge({"/m/big": [11000], "/m/small": [3000]}, files, LIMIT) == (["big"], [])
    assert judge({"/m/big": [LIMIT], "/m/small": [None]}, files, LIMIT) == ([], ["small"])
    assert judge({}, files, LIMIT) == ([], ["big", "small"]), "an unmeasured runner is never small"
    assert judge({"/m/big": [3000, 9000]}, {"big": "/m/big"}, LIMIT) == (["big"], [])


def test_after_reload() -> None:
    assert after_reload([10], {10: 7700}, LIMIT) == "same", "the old runner still up is no reload"
    assert after_reload([10], {}, LIMIT) == "same", "no runner after the reload is no reload"
    assert after_reload([10], {11: 3400}, LIMIT) == "small"
    assert after_reload([10], {11: 7500}, LIMIT) == "big"
    assert after_reload([], {11: None}, LIMIT) == "small", "an unmeasured new runner is not taken as big"
    assert after_reload([10, 11], {10: 9000, 12: 3400}, LIMIT) == "same"


def test_may_trim() -> None:
    assert may_trim({}, "m")
    assert may_trim({"m": True}, "m")
    assert not may_trim({"m": False}, "m")


def test_busy_marker(tmp_path: Path) -> None:
    marker = tmp_path / "busy"
    assert not busy(marker), "no marker is not busy"
    marker.write_text("1000")
    assert busy(marker, now=1030)
    assert not busy(marker, now=1200), "a stale marker is a crashed bridge"
    marker.write_text("garbage")
    assert not busy(marker)


def test_runners_from_ps() -> None:
    out = (" 101 4194304 /usr/bin/ollama runner --model /blobs/sha-a --port 5000\n"
           + " 102 1048576 /usr/bin/ollama serve\n"
           + " 103 2097152 /opt/llama-server --model=/blobs/sha-b --ctx 4096\n"
           + "garbage line\n")
    assert runners(out) == {"/blobs/sha-a": {101: 4096}, "/blobs/sha-b": {103: 2048}}
