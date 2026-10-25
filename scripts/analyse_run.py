"""Frame-level and box-level analysis of a saved detector run (no GPU needed).

The threshold is chosen on the **validation** images, then applied unchanged
to the **test** images, so the test numbers are not tuned on test data.

Two threshold rules are reported:
    * ``max_f1``: maximises frame-level F1 on validation (balances misses and false alarms);
    * ``recall_95``: the highest threshold whose validation defect-frame recall is >= 0.95
      (a safety-first rule: a missed defect costs more than a false alarm).

Usage::

    python scripts/analyse_run.py --run runs/yolov8n_640 --data data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from porkinspect.evaluate import (  # noqa: E402
    bootstrap_ci, box_recall_by_class, frame_counts, frame_flags, load_ground_truth,
    load_predictions, threshold_sweep,
)

NAMES = ["loose-meat", "packaging-error", "twisted-meat", "unsealed"]


def analyse_split(preds, gts, thr):
    """Frame counts, bootstrap CIs and per-class box recall at threshold ``thr``.

    Args:
        preds: Image name -> detections, from ``load_predictions``.
        gts: Image name -> labelled boxes, from ``load_ground_truth``.
        thr: Frame-flag confidence threshold.

    Returns:
        Dict of frame counts, recall and false-alarm rate with 95% bootstrap
        intervals, and per-class box recall (class-aware and any-class).
    """
    images = sorted(gts)
    defect = np.array([len(gts[i]) > 0 for i in images])
    flags = frame_flags(preds, images, thr)
    c = frame_counts(flags, defect)

    def rec(idx):
        """Defect-frame recall for a resample of image indices.

        Args:
            idx: Image indices (with repeats) from the bootstrap.

        Returns:
            Recall, or NaN if the resample has no defect frames.
        """
        d = defect[idx]
        return (flags[idx] & d).sum() / d.sum() if d.sum() else np.nan

    def far(idx):
        """False-alarm rate for a resample of image indices.

        Args:
            idx: Image indices (with repeats) from the bootstrap.

        Returns:
            False-alarm rate, or NaN if the resample has no defect-free frames.
        """
        n = ~defect[idx]
        return (flags[idx] & n).sum() / n.sum() if n.sum() else np.nan

    boxes = box_recall_by_class(preds, gts, thr, len(NAMES))
    return {
        "threshold": thr,
        "frames": len(images), "defect_frames": int(defect.sum()), "defect_free_frames": int((~defect).sum()),
        "tp": c.tp, "fn": c.fn, "fp": c.fp, "tn": c.tn,
        "recall": c.recall, "recall_ci95": bootstrap_ci(rec, len(images)),
        "false_alarm_rate": c.false_alarm_rate, "false_alarm_ci95": bootstrap_ci(far, len(images)),
        "precision": c.precision,
        "box_recall": {k: {NAMES[c_]: v for c_, v in d.items()} for k, d in boxes.items()},
    }


def main() -> None:
    """Choose thresholds on validation, evaluate on test, write JSON and a plot."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or ROOT / "results" / a.run.name
    out.mkdir(parents=True, exist_ok=True)

    sets = {}
    for split, folder in (("val", "valid"), ("test", "test")):
        preds = load_predictions(a.run / f"predictions_{split}.json")
        gts = load_ground_truth(a.data / folder / "labels", sorted(preds), len(NAMES))
        sets[split] = (preds, gts)

    ts = np.round(np.arange(0.01, 0.96, 0.01), 2)
    sweep = {s: threshold_sweep(p, g, ts) for s, (p, g) in sets.items()}
    v = sweep["val"]
    max_f1 = max(v, key=lambda r: (np.nan_to_num(r["f1"]), r["threshold"]))["threshold"]
    ok = [r["threshold"] for r in v if r["recall"] >= 0.95]
    recall_95 = max(ok) if ok else float(ts[0])

    result = {"thresholds_chosen_on_val": {"max_f1": max_f1, "recall_95": recall_95}}
    for rule, t in result["thresholds_chosen_on_val"].items():
        result[rule] = {s: analyse_split(*sets[s], t) for s in ("val", "test")}
    (out / "analysis.json").write_text(json.dumps(result, indent=2))
    (out / "threshold_sweep.json").write_text(json.dumps(sweep, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, s in zip(axes, ("val", "test")):
        r = sweep[s]
        ax.plot(ts, [x["recall"] for x in r], label="defect frames caught (recall)")
        ax.plot(ts, [x["false_alarm_rate"] for x in r], label="defect-free frames flagged (false alarms)")
        for rule, t in result["thresholds_chosen_on_val"].items():
            ax.axvline(t, color="grey", ls="--", lw=1)
            ax.text(t, 1.02, rule, rotation=90, va="bottom", ha="center", fontsize=8, color="grey")
        n = r[0]["tp"] + r[0]["fn"], r[0]["fp"] + r[0]["tn"]
        ax.set_title(f"{s}: {n[0]} defect / {n[1]} defect-free frames", fontsize=10)
        ax.set_xlabel("confidence threshold")
    axes[0].set_ylabel("rate")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "frame_threshold_sweep.png", dpi=130)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
