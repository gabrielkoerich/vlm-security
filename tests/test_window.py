from vlmsec.window import Detection, Seen, best_view, is_scenery, moved, parse_detector, sample_frames, summarise


def test_sample_keeps_every_fourth_and_the_last() -> None:
    paths = [f"f{i}.jpg" for i in range(10)]
    assert sample_frames(paths, 4) == ["f0.jpg", "f4.jpg", "f8.jpg", "f9.jpg"]


def test_sample_does_not_repeat_the_last() -> None:
    paths = [f"f{i}.jpg" for i in range(9)]
    assert sample_frames(paths, 4) == ["f0.jpg", "f4.jpg", "f8.jpg"]


def test_sample_short_window_is_kept_whole() -> None:
    assert sample_frames(["a", "b", "c"], 4) == ["a", "b", "c"]
    assert sample_frames(["a", "b", "c"], 1) == ["a", "b", "c"]


def test_best_view_keeps_top_confidence_and_the_whole_box() -> None:
    # the most confident box is cut by the right edge of a 1000 px frame
    cut = Detection("person", 0.9, (990.0, 300.0, 60.0, 120.0), "cut.jpg")
    small = Detection("person", 0.5, (400.0, 300.0, 40.0, 80.0), "small.jpg")
    large = Detection("person", 0.6, (500.0, 300.0, 80.0, 160.0), "large.jpg")
    car = Detection("car", 0.8, (500.0, 300.0, 400.0, 300.0), "car.jpg")
    got = best_view([cut, small, large, car], 1000.0)
    assert got.path == "large.jpg"
    assert got.conf == 0.9
    assert got.cls == "person"


def test_best_view_falls_back_to_top_when_every_box_touches_an_edge() -> None:
    a = Detection("car", 0.7, (5.0, 100.0, 40.0, 40.0), "a.jpg")
    b = Detection("car", 0.9, (995.0, 100.0, 40.0, 40.0), "b.jpg")
    got = best_view([a, b], 1000.0)
    assert got.conf == 0.9
    assert got.path in ("a.jpg", "b.jpg")


def test_moved_is_the_largest_span() -> None:
    boxes = [(100.0, 100.0, 20.0, 40.0), (130.0, 105.0, 20.0, 40.0), (160.0, 110.0, 20.0, 100.0)]
    assert moved(boxes, boxes[0]) == 60.0


def test_moved_ignores_boxes_far_from_the_anchor() -> None:
    near = [(100.0, 100.0, 20.0, 40.0), (104.0, 102.0, 20.0, 41.0)]
    far = [(900.0, 600.0, 20.0, 40.0)]
    assert moved(near + far, near[0]) == 4.0


def test_moved_needs_two_boxes() -> None:
    assert moved([(1.0, 1.0, 1.0, 1.0)], (1.0, 1.0, 1.0, 1.0)) == 0.0


def test_scenery_gate() -> None:
    d = Detection("person", 0.8, (100.0, 100.0, 20.0, 40.0), "f.jpg")
    assert is_scenery(Seen(d, 30.0, 10))
    assert not is_scenery(Seen(d, 150.0, 10))
    assert not is_scenery(Seen(None, 0.0, 10)), "nothing seen is empty, not scenery"


def test_detector_answer_to_one_subject() -> None:
    answer: object = {"results": [
        {"path": "a.jpg", "detections": [{"cls": "person", "conf": 0.4, "box": [100, 200, 30, 60]}]},
        {"path": "b.jpg", "detections": [{"cls": "person", "conf": 0.7, "box": [250, 210, 30, 62]}]},
        {"path": "c.jpg", "detections": [{"cls": "car", "conf": 0.3, "box": "bad"}]},
        {"path": "d.jpg"},
    ]}
    parsed = parse_detector(answer)
    assert len(parsed) == 4
    assert parsed[2][1] == [], "a malformed box is dropped"
    seen = summarise(parsed, 1280.0)
    assert seen.best is not None
    assert seen.best.conf == 0.7
    assert seen.moved == 150.0
    assert seen.frames == 4


def test_empty_window() -> None:
    seen = summarise([("a.jpg", [])], 1280.0)
    assert seen.best is None
