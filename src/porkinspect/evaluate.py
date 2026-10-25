"""Evaluate saved detector predictions at the box level and the frame level.

The detector's raw predictions (every box with confidence >= 0.001) are saved
once on the GPU by ``scripts/train_yolo.py``. Everything here works from those
files, so thresholds and error bars can be analysed on any machine.

Two levels of evaluation:

* **Box level:** a prediction matches a ground-truth box if their IoU is at
  least ``iou_thr`` (0.5 by default). With ``class_aware=True`` the classes
  must also agree. Greedy matching in descending confidence, as in VOC/COCO.
* **Frame level:** a frame is *flagged* if any prediction has confidence
  >= the threshold. A frame is a *defect frame* if it has at least one
  ground-truth box. This is the quantity a production line would act on.

Uncertainty comes from a bootstrap over images: resample the evaluation images
with replacement, recompute the metric, and take percentiles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .labels import Box, read_label_file


@dataclass(frozen=True)
class Detection:
    """One predicted box.

    Attributes:
        cls: Predicted class id.
        conf: Confidence score in [0, 1].
        box: Normalised (x_centre, y_centre, w, h).
    """

    cls: int
    conf: float
    box: tuple[float, float, float, float]


def load_predictions(path: str | Path) -> dict[str, list[Detection]]:
    """Read a ``predictions_*.json`` file written by ``scripts/train_yolo.py``.

    Returns:
        Image file name -> detections, sorted by descending confidence.
    """
    out: dict[str, list[Detection]] = {}
    for r in json.loads(Path(path).read_text()):
        dets = [Detection(int(c), float(p), tuple(b)) for c, p, b in zip(r["cls"], r["conf"], r["xywhn"])]
        out[r["image"]] = sorted(dets, key=lambda d: -d.conf)
    return out


def load_ground_truth(label_dir: str | Path, images: list[str], n_classes: int) -> dict[str, tuple[Box, ...]]:
    """Read the label file for each image name in ``images``."""
    label_dir = Path(label_dir)
    return {im: read_label_file(label_dir / (Path(im).stem + ".txt"), n_classes) for im in images}


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two boxes given as (x_centre, y_centre, w, h).

    Returns a value in [0, 1]; 0 when the boxes do not overlap.
    """
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match_image(
    dets: list[Detection],
    gts: tuple[Box, ...],
    conf_thr: float,
    *,
    iou_thr: float = 0.5,
    class_aware: bool = True,
) -> tuple[list[bool], list[bool]]:
    """Greedily match detections above ``conf_thr`` to ground-truth boxes.

    ``iou_thr`` and ``class_aware`` are keyword-only: passing ``True`` positionally
    would otherwise be read silently as an IoU threshold of 1.

    Returns:
        ``(gt_found, det_correct)``: one flag per ground-truth box (was it
        matched?) and one per kept detection (was it a true positive?).
    """
    kept = [d for d in dets if d.conf >= conf_thr]
    gt_found = [False] * len(gts)
    det_correct = []
    for d in kept:  # already sorted by confidence
        best, best_j = iou_thr, -1
        for j, g in enumerate(gts):
            if gt_found[j] or (class_aware and g.cls != d.cls):
                continue
            v = iou(d.box, (g.x, g.y, g.w, g.h))
            if v >= best:
                best, best_j = v, j
        if best_j >= 0:
            gt_found[best_j] = True
        det_correct.append(best_j >= 0)
    return gt_found, det_correct


@dataclass
class FrameCounts:
    """Frame-level confusion counts at one threshold."""

    tp: int  # defect frames flagged
    fn: int  # defect frames missed
    fp: int  # defect-free frames flagged (false alarms)
    tn: int  # defect-free frames passed

    @property
    def recall(self) -> float:
        """Share of defect frames that were flagged (NaN if there are none)."""
        n = self.tp + self.fn
        return self.tp / n if n else float("nan")

    @property
    def false_alarm_rate(self) -> float:
        """Share of defect-free frames that were flagged (NaN if there are none)."""
        n = self.fp + self.tn
        return self.fp / n if n else float("nan")

    @property
    def precision(self) -> float:
        """Share of flagged frames that really contain a defect."""
        n = self.tp + self.fp
        return self.tp / n if n else float("nan")


def frame_flags(preds: dict[str, list[Detection]], images: list[str], conf_thr: float) -> np.ndarray:
    """Boolean array: is each image flagged (any detection >= ``conf_thr``)?"""
    return np.array([any(d.conf >= conf_thr for d in preds.get(im, [])) for im in images])


def frame_counts(flags: np.ndarray, is_defect: np.ndarray) -> FrameCounts:
    """Confusion counts from per-frame flags and ground truth."""
    return FrameCounts(
        tp=int((flags & is_defect).sum()),
        fn=int((~flags & is_defect).sum()),
        fp=int((flags & ~is_defect).sum()),
        tn=int((~flags & ~is_defect).sum()),
    )


def threshold_sweep(
    preds: dict[str, list[Detection]], gts: dict[str, tuple[Box, ...]], thresholds: np.ndarray
) -> list[dict]:
    """Frame-level recall, false-alarm rate, precision and F1 at each threshold."""
    images = sorted(gts)
    is_defect = np.array([len(gts[im]) > 0 for im in images])
    rows = []
    for t in thresholds:
        c = frame_counts(frame_flags(preds, images, float(t)), is_defect)
        p, r = c.precision, c.recall
        f1 = 2 * p * r / (p + r) if p == p and r == r and (p + r) > 0 else float("nan")
        rows.append({"threshold": float(t), "recall": r, "false_alarm_rate": c.false_alarm_rate,
                     "precision": p, "f1": f1, "tp": c.tp, "fn": c.fn, "fp": c.fp, "tn": c.tn})
    return rows


def box_recall_by_class(
    preds: dict[str, list[Detection]],
    gts: dict[str, tuple[Box, ...]],
    conf_thr: float,
    n_classes: int,
    iou_thr: float = 0.5,
) -> dict[str, dict[int, tuple[int, int]]]:
    """Ground-truth boxes found / total, per class, class-aware and class-agnostic.

    Returns:
        ``{"class_aware": {cls: (found, total)}, "any_class": {cls: (found, total)}}``.
        "any_class" counts a box as found if a detection of any class overlaps it,
        which separates "missed the defect" from "found it but named it wrongly".
    """
    out = {}
    for key, aware in (("class_aware", True), ("any_class", False)):
        found = np.zeros(n_classes, int)
        total = np.zeros(n_classes, int)
        for im, g in gts.items():
            hit, _ = match_image(preds.get(im, []), g, conf_thr, iou_thr=iou_thr, class_aware=aware)
            for b, h in zip(g, hit):
                total[b.cls] += 1
                found[b.cls] += int(h)
        out[key] = {k: (int(found[k]), int(total[k])) for k in range(n_classes)}
    return out


def bootstrap_ci(
    values_fn,
    n_items: int,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a statistic of resampled item indices.

    Args:
        values_fn: Function taking an index array (with repeats) and returning
            the statistic, or NaN if it is undefined for that resample.
        n_items: Number of items (images) to resample from.
        n_boot: Number of bootstrap resamples.
        seed: Random seed, for reproducibility.
        level: Coverage of the interval.

    Returns:
        ``(low, high)``. NaN resamples are ignored.
    """
    rng = np.random.default_rng(seed)
    stats = np.array([values_fn(rng.integers(0, n_items, n_items)) for _ in range(n_boot)], float)
    stats = stats[~np.isnan(stats)]
    a = (1 - level) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))
