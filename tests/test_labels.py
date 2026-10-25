"""Tests for porkinspect.labels."""

import pytest

from porkinspect.labels import Box, frame_number, parse_label_line, read_label_file, source_stem


def test_parse_valid_line():
    assert parse_label_line("3 0.5 0.25 0.1 0.2") == Box(3, 0.5, 0.25, 0.1, 0.2)


@pytest.mark.parametrize(
    "line",
    [
        "3 0.5 0.5 0.1",  # too few fields
        "3 0.5 0.5 0.1 0.1 0.2 0.2",  # polygon, not a box
        "7 0.5 0.5 0.1 0.1",  # class id out of range
        "a 0.5 0.5 0.1 0.1",  # class id not an integer
        "1 1.5 0.5 0.1 0.1",  # coordinate > 1
        "1 0.5 0.5 0.0 0.1",  # zero width
    ],
)
def test_parse_rejects_bad_lines(line):
    with pytest.raises(ValueError):
        parse_label_line(line)


def test_read_label_file_sorts_and_handles_empty(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("3 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n\n")
    boxes = read_label_file(f)
    assert [b.cls for b in boxes] == [1, 3]
    g = tmp_path / "b.txt"
    g.write_text("")
    assert read_label_file(g) == ()


def test_missing_label_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_label_file(tmp_path / "missing.txt")


def test_box_round_trip():
    b = Box(2, 0.123456, 0.5, 0.25, 0.75)
    assert parse_label_line(b.to_line()) == b


def test_roboflow_names():
    name = "frame_with_red_10049_jpg.rf.cd786c4f9dd783e72bec7eed7637f629.jpg"
    assert source_stem(name) == "frame_with_red_10049"
    assert frame_number(name) == 10049
    # A plain file name falls back to its stem.
    assert source_stem("frame_7.jpg") == "frame_7"
    with pytest.raises(ValueError):
        frame_number("no_number_here.jpg")
