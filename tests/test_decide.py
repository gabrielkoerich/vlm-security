from vlmsec.decide import VisionDown, decide, place_sure
from vlmsec.vlm import NOTHING, Answer


def test_on_the_property_and_armed_is_an_alarm() -> None:
    a = Answer(True, True, "none", "house")
    assert decide(a, armed=True).action == "alarm"
    assert decide(a, armed=False).action == "notify"


def test_street_is_logged_and_nothing_is_ignored() -> None:
    assert decide(Answer(True, False, "car", "street"), armed=True).action == "log"
    assert decide(NOTHING, armed=True).action == "ignore"


def test_sure_call_without_placement_counts_as_inside() -> None:
    assert place_sure("none") == "house"
    assert place_sure("street") == "street"


def test_vision_down_first_alerts_second_fires() -> None:
    v = VisionDown()
    assert v.on_human(0.0, armed=True) == "alert"
    assert v.on_human(60.0, armed=True) == "alarm"


def test_vision_down_second_too_late_is_a_new_first() -> None:
    v = VisionDown()
    assert v.on_human(0.0, armed=True) == "alert"
    assert v.on_human(200.0, armed=True) == "alert"
    assert v.on_human(300.0, armed=True) == "alarm"


def test_vision_down_same_pass_is_not_a_second_detection() -> None:
    v = VisionDown()
    assert v.on_human(0.0, armed=True) == "alert"
    assert v.on_human(10.0, armed=True) == "ignore"
    assert v.on_human(50.0, armed=True) == "alarm"


def test_vision_down_after_an_alarm_starts_over() -> None:
    v = VisionDown()
    v.on_human(0.0, armed=True)
    assert v.on_human(60.0, armed=True) == "alarm"
    assert v.on_human(120.0, armed=True) == "alert"


def test_vision_down_disarmed_only_notifies() -> None:
    v = VisionDown()
    assert v.on_human(0.0, armed=False) == "notify"
    assert v.on_human(60.0, armed=False) == "notify"
