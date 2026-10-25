"""Tests for porkinspect.inspection: input validation and the output contract."""

import numpy as np
import pytest
from PIL import Image

from porkinspect.inspection import check_image, inspect_image


def belt(tmp_path, name="a.jpg", size=(720, 540), level=40, mode="RGB"):
    a = np.full((size[1], size[0], 3), level, np.uint8)
    im = Image.fromarray(a)
    im = im.convert(mode) if mode != "RGB" else im
    p = tmp_path / name
    im.save(p)
    return p


def test_valid_image_has_no_warnings(tmp_path):
    rgb, w = check_image(belt(tmp_path))
    assert rgb.mode == "RGB" and w == []


def test_missing_and_corrupt_files_rejected(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        check_image(tmp_path / "nope.jpg")
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"this is not an image")
    with pytest.raises(ValueError, match="not a readable image"):
        check_image(bad)
    # A truncated JPEG: valid header, missing data.
    good = belt(tmp_path, "g.jpg").read_bytes()
    trunc = tmp_path / "trunc.jpg"
    trunc.write_bytes(good[: len(good) // 3])
    with pytest.raises(ValueError):
        check_image(trunc)


def test_out_of_domain_inputs_warn(tmp_path):
    _, w = check_image(belt(tmp_path, "g.png", mode="L"))
    assert any("converted to RGB" in x for x in w)
    _, w = check_image(belt(tmp_path, "s.jpg", size=(640, 480)))
    assert any("differs" in x for x in w)
    _, w = check_image(belt(tmp_path, "d.jpg", level=5))
    assert any("brightness" in x for x in w)


def test_corrupt_file_rejected_even_after_ultralytics_import(tmp_path):
    """Regression test: Ultralytics patches Image.open, which changes the error raised."""
    pytest.importorskip("ultralytics")
    import ultralytics  # noqa: F401  (importing applies the patch)

    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="not a readable image"):
        check_image(bad)


class FakeBoxes:
    def __init__(self, cls, conf, xywhn):
        import types
        t = lambda v: types.SimpleNamespace(tolist=lambda: v)  # noqa: E731
        self.cls, self.conf, self.xywhn = t(cls), t(conf), t(xywhn)


class FakeModel:
    """Stands in for an Ultralytics model so the output contract can be tested without weights."""
    names = {0: "loose-meat", 1: "packaging-error", 2: "twisted-meat", 3: "unsealed"}

    def __init__(self, cls, conf):
        self._r = type("R", (), {"boxes": FakeBoxes(cls, conf, [[0.5, 0.5, 0.1, 0.1]] * len(cls)),
                                 "speed": {"inference": 4.2}})()

    def predict(self, img, **kw):
        assert img.shape == (540, 720, 3)
        return [self._r]


def test_inspect_output_contract(tmp_path):
    p = belt(tmp_path)
    r = inspect_image(p, None, threshold=0.16, model=FakeModel([3, 1], [0.05, 0.8]))
    assert r["defect_present"] is True
    assert [d["class"] for d in r["detections"]] == ["packaging-error"]  # 0.05 is below the threshold
    assert r["max_confidence"] == pytest.approx(0.8) and r["inference_ms"] == 4.2
    r = inspect_image(p, None, threshold=0.9, model=FakeModel([3], [0.5]))
    assert r["defect_present"] is False and r["detections"] == []
    with pytest.raises(ValueError):
        inspect_image(p, None, threshold=1.5, model=FakeModel([], []))


def test_draw_report_marks_boxes_and_verdict(tmp_path):
    from porkinspect.inspection import draw_report

    p = belt(tmp_path)
    rep = {"defect_present": True, "detections": [
        {"class": "unsealed", "confidence": 0.9, "box_xywhn": [0.5, 0.6, 0.2, 0.2]}]}
    out = draw_report(p, rep, truth=[("unsealed", (0.5, 0.6, 0.25, 0.25))], threshold=0.16)
    a = np.asarray(out)
    assert out.size == (720, 540)
    assert tuple(a[5, 360]) == (200, 30, 30)  # red DEFECT banner
    x0 = int((0.5 - 0.1) * 720)
    assert tuple(a[int(0.6 * 540), x0 + 1]) == (213, 94, 0)  # unsealed box outline colour
    ok = draw_report(p, {"defect_present": False, "detections": []})
    assert tuple(np.asarray(ok)[5, 360]) == (30, 140, 60)  # green banner
