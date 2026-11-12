from vlmsec.geometry import Line, load, place, side_of

ACROSS = Line((0.0, 500.0), (1000.0, 600.0), "below")


def test_point_below_a_line_across_the_frame() -> None:
    # the bottom centre is what counts: centre 400 + half height 100 = 500, under the line at x=500 (550)
    assert side_of(ACROSS, (500.0, 400.0, 50.0, 200.0)) == "street"
    assert side_of(ACROSS, (500.0, 600.0, 50.0, 200.0)) == "house"


def test_line_direction_does_not_matter() -> None:
    flipped = Line((1000.0, 600.0), (0.0, 500.0), "below")
    assert side_of(flipped, (500.0, 600.0, 50.0, 200.0)) == "house"
    above = Line((0.0, 500.0), (1000.0, 600.0), "above")
    assert side_of(above, (500.0, 600.0, 50.0, 200.0)) == "street"


def test_vertical_line() -> None:
    right = Line((600.0, 0.0), (600.0, 720.0), "right")
    assert side_of(right, (800.0, 300.0, 40.0, 80.0)) == "house"
    assert side_of(right, (300.0, 300.0, 40.0, 80.0)) == "street"
    left = Line((600.0, 720.0), (600.0, 0.0), "left")
    assert side_of(left, (300.0, 300.0, 40.0, 80.0)) == "house"


def test_no_line_keeps_the_model_answer() -> None:
    lines = {4: ACROSS}
    assert place(4, (500.0, 600.0, 50.0, 200.0), "street", lines) == "house"
    assert place(2, (500.0, 600.0, 50.0, 200.0), "street", lines) == "street"
    assert place(4, None, "street", lines) == "street"


def test_load_skips_malformed_entries() -> None:
    got = load('{"4": {"a": [0, 520], "b": [1280, 610], "property": "below"},'
               + ' "5": {"a": [0], "b": [1, 2], "property": "below"},'
               + ' "6": {"a": [0, 1], "b": [1, 2], "property": "sideways"}}')
    assert list(got) == [4]
    assert got[4].b == (1280.0, 610.0)
    assert load("") == {}
