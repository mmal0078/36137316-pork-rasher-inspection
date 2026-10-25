"""Tests for porkinspect.evaluate, using hand-checkable cases."""

import numpy as np
import pytest

from porkinspect.evaluate import (
    Detection,
    FrameCounts,
    bootstrap_ci,
    box_recall_by_class,
    frame_counts,
    frame_flags,
    iou,
    match_image,
    threshold_sweep,
)
from porkinspect.labels import Box


def test_iou_known_values():
    a = (0.5, 0.5, 0.2, 0.2)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, (0.9, 0.9, 0.1, 0.1)) == 0.0
    # Shift by half a width: overlap 0.1 x 0.2 = 0.02; union 0.04 + 0.04 - 0.02 = 0.06.
    assert iou(a, (0.6, 0.5, 0.2, 0.2)) == pytest.approx(1 / 3)


def test_match_respects_class_threshold_and_one_to_one():
    gts = (Box(1, 0.5, 0.5, 0.2, 0.2),)
    dets = [Detection(1, 0.9, (0.5, 0.5, 0.2, 0.2)), Detection(1, 0.8, (0.51, 0.5, 0.2, 0.2))]
    found, correct = match_image(dets, gts, conf_thr=0.5)
    assert found == [True] and correct == [True, False]  # duplicate is a false positive
    found, _ = match_image(dets, gts, conf_thr=0.95)
    assert found == [False]
    wrong = [Detection(3, 0.9, (0.5, 0.5, 0.2, 0.2))]
    assert match_image(wrong, gts, 0.5, class_aware=True)[0] == [False]
    assert match_image(wrong, gts, 0.5, class_aware=False)[0] == [True]


def test_match_options_are_keyword_only():
    gts = (Box(1, 0.5, 0.5, 0.2, 0.2),)
    dets = [Detection(1, 0.9, (0.5, 0.5, 0.2, 0.2))]
    with pytest.raises(TypeError):
        match_image(dets, gts, 0.5, True)  # would otherwise mean iou_thr=1


def test_frame_counts_and_rates():
    flags = np.array([True, True, False, True, False])
    defect = np.array([True, True, True, False, False])
    c = frame_counts(flags, defect)
    assert (c.tp, c.fn, c.fp, c.tn) == (2, 1, 1, 1)
    assert c.recall == pytest.approx(2 / 3)
    assert c.false_alarm_rate == pytest.approx(0.5)
    assert np.isnan(FrameCounts(0, 0, 0, 3).recall)


def test_sweep_monotone():
    preds = {"a": [Detection(0, 0.9, (0.5, 0.5, 0.1, 0.1))], "b": [Detection(0, 0.3, (0.5, 0.5, 0.1, 0.1))], "c": []}
    gts = {"a": (Box(0, 0.5, 0.5, 0.1, 0.1),), "b": (), "c": (Box(0, 0.2, 0.2, 0.1, 0.1),)}
    rows = threshold_sweep(preds, gts, np.array([0.1, 0.5, 0.95]))
    assert [r["recall"] for r in rows] == [0.5, 0.5, 0.0]
    assert [r["false_alarm_rate"] for r in rows] == [1.0, 0.0, 0.0]
    assert list(frame_flags(preds, ["a", "b", "c"], 0.5)) == [True, False, False]


def test_box_recall_by_class():
    gts = {"a": (Box(0, 0.5, 0.5, 0.2, 0.2), Box(1, 0.2, 0.2, 0.1, 0.1))}
    preds = {"a": [Detection(2, 0.9, (0.5, 0.5, 0.2, 0.2))]}  # right place, wrong class
    r = box_recall_by_class(preds, gts, 0.5, n_classes=3)
    assert r["class_aware"][0] == (0, 1) and r["any_class"][0] == (1, 1)
    assert r["any_class"][1] == (0, 1)


def test_bootstrap_ci_brackets_mean_and_is_reproducible():
    x = np.random.default_rng(1).normal(10, 1, 200)
    lo, hi = bootstrap_ci(lambda idx: x[idx].mean(), len(x), n_boot=500, seed=3)
    assert lo < x.mean() < hi and hi - lo < 0.5
    assert (lo, hi) == bootstrap_ci(lambda idx: x[idx].mean(), len(x), n_boot=500, seed=3)
