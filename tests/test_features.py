"""Tests for porkinspect.features on synthetic images with known content."""

import numpy as np
import pytest
from PIL import Image

from porkinspect.features import COLOUR_SHAPE_NAMES, colour_shape_features, hog_features, meat_mask


def synthetic(tmp_path, meat_box=None, name="img.png"):
    """Dark grey 'belt' with an optional pink 'meat' rectangle (fractions of width/height)."""
    a = np.full((270, 360, 3), 40, np.uint8)
    if meat_box:
        x0, y0, x1, y1 = (int(v * s) for v, s in zip(meat_box, (360, 270, 360, 270)))
        a[y0:y1, x0:x1] = (200, 90, 100)
    p = tmp_path / name
    Image.fromarray(a).save(p)
    return p


def test_meat_mask_separates_pink_from_grey():
    rgb = np.zeros((2, 2, 3))
    rgb[0, 0] = (200 / 255, 90 / 255, 100 / 255)  # pink meat
    rgb[0, 1] = (0.16, 0.16, 0.16)  # belt
    rgb[1, 0] = (0.9, 0.9, 0.9)  # white glare
    assert meat_mask(rgb).tolist() == [[True, False], [False, False]]


def test_colour_shape_features_position_and_area(tmp_path):
    f = dict(zip(COLOUR_SHAPE_NAMES, colour_shape_features(synthetic(tmp_path, (0.75, 0.5, 1.0, 1.0)))))
    assert f["meat_fraction"] == pytest.approx(0.125, abs=0.01)  # 0.25 x 0.5 of the image
    assert f["meat_cx"] == pytest.approx(0.875, abs=0.01)
    assert f["meat_cy"] == pytest.approx(0.75, abs=0.01)
    # Red hue is shifted to about 0.5 so it averages correctly.
    assert 0.4 < f["meat_hue_mean"] < 0.6


def test_feature_lengths_and_no_meat_case(tmp_path):
    f = colour_shape_features(synthetic(tmp_path))
    assert len(f) == len(COLOUR_SHAPE_NAMES) and f[0] == 0.0 and np.isfinite(f).all()
    assert hog_features(synthetic(tmp_path, (0.5, 0.5, 1, 1), "b.png")).shape == (1944,)
