from pathlib import Path

from vlmsec.bridge import cameras, evaluate, spread
from vlmsec.geometry import Line
from vlmsec.window import Detection, Seen

PERSON = Detection("person", 0.6, (500.0, 600.0, 50.0, 200.0), "best.jpg")
LINES = {4: Line((0.0, 500.0), (1000.0, 600.0), "below")}


class FakeModel:
    def __init__(self, answer: str | None) -> None:
        self.answer: str | None = answer
        self.asked: list[Path] = []

    def __call__(self, image: Path) -> str | None:
        self.asked.append(image)
        return self.answer


def test_scenery_is_never_asked() -> None:
    m = FakeModel("PERSON: yes")
    out = evaluate(4, Seen(PERSON, 20.0, 10), [], None, m, LINES)
    assert out.kind == "scenery"
    assert m.asked == []


def test_moving_subject_is_asked_and_the_line_places_it() -> None:
    m = FakeModel("PERSON: yes PERSON_WHERE: street VEHICLE: none COLOR: none VEHICLE_WHERE: none")
    out = evaluate(4, Seen(PERSON, 300.0, 10), [], None, m, LINES)
    assert out.kind == "asked"
    assert m.asked == [Path("best.jpg")]
    assert out.answer.where == "house", "the geometry line overrides the model"


def test_no_line_keeps_the_model_where() -> None:
    m = FakeModel("PERSON: yes PERSON_WHERE: street VEHICLE: none COLOR: none VEHICLE_WHERE: none")
    out = evaluate(2, Seen(PERSON, 300.0, 10), [], None, m, LINES)
    assert out.answer.where == "street"


def test_sure_call_skips_the_model_and_counts_as_inside_without_a_line() -> None:
    m = FakeModel("EMPTY")
    out = evaluate(2, Seen(PERSON, 300.0, 10), [], 0.95, m, LINES)
    assert out.kind == "sure"
    assert m.asked == []
    assert out.answer.person and out.answer.where == "house"


def test_sure_vehicle_on_the_street_side_of_a_line() -> None:
    car = Detection("car", 0.9, (500.0, 300.0, 200.0, 100.0), "car.jpg")
    out = evaluate(4, Seen(car, 300.0, 10), [], 0.95, FakeModel(None), LINES)
    assert out.answer.vehicle == "car" and out.answer.where == "street"


def test_hybrid_reopens_a_still_person() -> None:
    m = FakeModel("PERSON: yes PERSON_WHERE: house VEHICLE: none COLOR: none VEHICLE_WHERE: none")
    out = evaluate(2, Seen(PERSON, 20.0, 10), [], 0.4, m, LINES)
    assert out.kind == "asked" and out.answer.where == "house"


def test_model_down_reports_a_human() -> None:
    out = evaluate(2, Seen(PERSON, 300.0, 10), [], None, FakeModel(None), LINES)
    assert out.kind == "unreachable" and out.human


def test_confident_detection_stands_when_the_model_sees_nothing() -> None:
    sure = Detection("person", 0.9, (500.0, 600.0, 50.0, 200.0), "best.jpg")
    out = evaluate(4, Seen(sure, 300.0, 10), [], None, FakeModel("EMPTY"), LINES)
    assert out.answer.present and out.answer.where == "house"


def test_detector_down_asks_a_spread_of_frames() -> None:
    frames = [f"f{i}.jpg" for i in range(20)]
    m = FakeModel("EMPTY")
    out = evaluate(2, None, frames, None, m, LINES)
    assert out.kind == "asked" and not out.answer.present
    assert len(m.asked) == 6 and m.asked[0] == Path("f0.jpg") and m.asked[-1] == Path("f19.jpg")


def test_everything_down_counts_the_event_as_a_human() -> None:
    out = evaluate(2, None, ["f0.jpg"], None, FakeModel(None), LINES)
    assert out.kind == "unreachable" and out.human


def test_nothing_detected_is_empty() -> None:
    m = FakeModel("PERSON: yes")
    out = evaluate(2, Seen(None, 0.0, 10), [], None, m, LINES)
    assert out.kind == "empty" and m.asked == []


def test_spread_and_cameras() -> None:
    assert spread(["a", "b"]) == ["a", "b"]
    assert cameras('{"binary_sensor.gate_motion": [4, "Front gate"], "bad": ["x", 1]}') == {
        "binary_sensor.gate_motion": cameras('{"binary_sensor.gate_motion": [4, "Front gate"]}')[
            "binary_sensor.gate_motion"]}
