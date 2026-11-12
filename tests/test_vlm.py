import email.message
import urllib.error
import urllib.request
from pathlib import Path
from types import TracebackType

import pytest

from vlmsec import vlm
from vlmsec.vlm import Answer, Asker, parse, title_for, vehicle_color


def test_parse_five_lines() -> None:
    a = parse("PERSON: yes\nPERSON_WHERE: house\nVEHICLE: car\nCOLOR: red\nVEHICLE_WHERE: street")
    assert a == Answer(True, True, "car", "house", "red")


def test_parse_person_position_wins_over_vehicle() -> None:
    a = parse("PERSON: yes PERSON_WHERE: house VEHICLE: van COLOR: white VEHICLE_WHERE: street")
    assert a.where == "house"


def test_parse_vehicle_position_when_no_person() -> None:
    a = parse("PERSON: no PERSON_WHERE: none VEHICLE: blue SUV COLOR: blue VEHICLE_WHERE: street")
    assert a.vehicle == "suv"
    assert a.where == "street"


def test_parse_four_line_answer() -> None:
    a = parse("PERSON: no VEHICLE: motorcycle COLOR: black WHERE: house")
    assert a.present and a.vehicle == "motorcycle" and a.where == "house"


def test_parse_empty_is_believed_only_when_nothing_is_named() -> None:
    assert not parse("EMPTY There are no people or vehicles.").present
    assert parse("EMPTY PERSON: no VEHICLE: white car WHERE: street").present


def test_parse_nothing_named() -> None:
    assert not parse("PERSON: no VEHICLE: none WHERE: street").present


def test_colour() -> None:
    assert vehicle_color("VEHICLE: car COLOR: red VEHICLE_WHERE: street") == "red"
    assert vehicle_color("VEHICLE: none COLOR: none") == ""
    assert vehicle_color("PERSON: yes") == ""


def test_titles() -> None:
    assert title_for(Answer(True, False, "car", "street", "red")) == "Red Car (street)"
    assert title_for(Answer(True, True, "motorcycle", "street")) == "Motorcycle with rider (street)"
    assert title_for(Answer(True, False, "suv", "house")) == "SUV (house)"
    assert title_for(Answer(True, True, "none", "house")) == "Person (house)"


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body: bytes = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, t: type[BaseException] | None, e: BaseException | None, tb: TracebackType | None) -> None:
        pass

    def read(self) -> bytes:
        return self.body


class FakeOllama:
    """Plays a script of outcomes, advancing a fake clock by each call's duration."""

    def __init__(self, script: list[tuple[str, float]]) -> None:
        self.script: list[tuple[str, float]] = script
        self.now: float = 1000.0
        self.calls: int = 0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.now += s

    def urlopen(self, req: urllib.request.Request, timeout: float) -> FakeResponse:
        _ = timeout
        kind, took = self.script[self.calls]
        self.calls += 1
        self.now += took
        if kind == "http":
            raise urllib.error.HTTPError(req.full_url, 500, "runner gone", email.message.Message(), None)
        if kind == "refused":
            raise urllib.error.URLError(ConnectionRefusedError())
        if kind == "timeout":
            raise TimeoutError()
        return FakeResponse(b'{"response": "PERSON: yes"}')


@pytest.fixture
def image(tmp_path: Path) -> Path:
    p = tmp_path / "f.jpg"
    p.write_bytes(b"not really a jpeg")
    return p


def asker(fake: FakeOllama, monkeypatch: pytest.MonkeyPatch) -> Asker:
    monkeypatch.setattr(urllib.request, "urlopen", fake.urlopen)
    return Asker(url="http://ollama.example.com/api/generate", clock=fake.clock, sleep=fake.sleep)


def test_answer_on_first_try(image: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeOllama([("ok", 1.0)])
    assert asker(fake, monkeypatch).ask(image) == "PERSON: yes"
    assert fake.slept == []


def test_fast_http_error_is_retried_once(image: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeOllama([("http", 0.5), ("ok", 1.0)])
    assert asker(fake, monkeypatch).ask(image) == "PERSON: yes"
    assert fake.slept == [vlm.RETRY_S]


def test_slow_http_error_is_not_retried(image: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeOllama([("http", vlm.FAST_ERROR_S + 1)])
    assert asker(fake, monkeypatch).ask(image) is None
    assert fake.calls == 1


@pytest.mark.parametrize("kind", ["refused", "timeout"])
def test_dead_ollama_is_not_retried(kind: str, image: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeOllama([(kind, 0.1)])
    assert asker(fake, monkeypatch).ask(image) is None
    assert fake.calls == 1


def test_no_retry_for_a_minute_after_a_failed_retry(image: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeOllama([("http", 0.5), ("http", 0.5), ("http", 0.5), ("http", 0.5), ("ok", 0.5)])
    a = asker(fake, monkeypatch)
    assert a.ask(image) is None
    assert fake.calls == 2
    fake.now += 10
    assert a.ask(image) is None
    assert fake.calls == 3, "paused, so no retry"
    fake.now += vlm.RETRY_PAUSE_S
    assert a.ask(image) == "PERSON: yes"
    assert fake.calls == 5


def test_request_pins_the_model(image: Path) -> None:
    assert b'"keep_alive": -1' in vlm.body_for(image)
