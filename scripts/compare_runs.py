"""Compare several detector runs (image size / model size sensitivity).

For each run, it reports:
    cost: parameters, GFLOPs at its input size, T4 inference time and training time;
    accuracy:
        mAP50 overall and per class (Ultralytics);
        frame-level ROC AUC on test, with the paired bootstrap difference vs the reference run;
        defect frames caught and false alarms at a threshold chosen on validation (recall >= 0.95);
        box recall for the two common classes, which are the only per-class numbers with enough test boxes to compare.

Usage::

    python scripts/compare_runs.py --runs runs/yolov8n_640 runs/yolov8n_320 runs/yolov8n_480 runs/yolov8s_640 \
        --reference yolov8n_640 --data data/processed --out results/sensitivity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from porkinspect.evaluate import box_recall_by_class, load_ground_truth, load_predictions  # noqa: E402

NAMES = ["loose-meat", "packaging-error", "twisted-meat", "unsealed"]


def frame_scores(preds, images):
    """Frame-level defect score from the detector.

    Args:
        preds: Image name -> detections.
        images: Image names, in the wanted order.

    Returns:
        Highest detection confidence per image (0 if none).
    """
    return np.array([max((d.conf for d in preds.get(i, [])), default=0.0) for i in images])


def main() -> None:
    """Load each run's saved predictions and metrics, compare, and plot accuracy vs cost."""
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--reference", default="yolov8n_640")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "sensitivity")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    gts, scores, preds_all, rows = {}, {}, {}, {}
    for split, folder in (("val", "valid"), ("test", "test")):
        imgs = sorted(p.name for p in (a.data / folder / "images").iterdir())
        gts[split] = load_ground_truth(a.data / folder / "labels", imgs, len(NAMES))
    y = {s: np.array([len(gts[s][i]) > 0 for i in sorted(gts[s])]) for s in gts}

    for run in a.runs:
        name = run.name
        m = json.loads((run / "metrics.json").read_text())
        p = {s: load_predictions(run / f"predictions_{s}.json") for s in ("val", "test")}
        preds_all[name] = p
        scores[name] = {s: frame_scores(p[s], sorted(gts[s])) for s in ("val", "test")}
        sv = scores[name]["val"]
        thr = float(np.sort(sv[y["val"]])[int(np.floor(0.05 * y["val"].sum()))])  # val recall >= 0.95
        st = scores[name]["test"]
        f = st >= thr
        br = box_recall_by_class(p["test"], gts["test"], thr, len(NAMES))["class_aware"]
        rows[name] = {
            "parameters": m["model"]["parameters"], "GFLOPs": m["model"]["GFLOPs_at_imgsz"],
            "imgsz": int(json.loads((run / "run_info.json").read_text())["args"]["imgsz"]),
            "t4_inference_ms": m["test"]["speed_ms_per_image"]["inference"],
            "t4_wall_ms": m["predict_timing_test"]["wall_ms_per_image"], "train_minutes": m["train_minutes"],
            "val_mAP50": m["val"]["mAP50"], "test_mAP50": m["test"]["mAP50"], "test_mAP50_95": m["test"]["mAP50_95"],
            "test_mAP50_per_class": {k: v["mAP50"] for k, v in m["test"]["per_class"].items()},
            "val_frame_AUC": float(roc_auc_score(y["val"], sv)), "test_frame_AUC": float(roc_auc_score(y["test"], st)),
            "threshold": thr, "test_caught": int((f & y["test"]).sum()), "test_defect_frames": int(y["test"].sum()),
            "test_false_alarms": int((f & ~y["test"]).sum()), "test_defect_free": int((~y["test"]).sum()),
            "test_box_recall": {NAMES[k]: list(v) for k, v in br.items()},
        }

    # Paired bootstrap: test frame-AUC difference of each run vs the reference.
    rng = np.random.default_rng(0)
    yt = y["test"]
    idx = [i for i in (rng.integers(0, len(yt), len(yt)) for _ in range(2000)) if yt[i].any() and not yt[i].all()]
    ref = scores[a.reference]["test"]
    for name in rows:
        s_ = scores[name]["test"]
        boot = np.array([roc_auc_score(yt[i], s_[i]) for i in idx])
        rows[name]["test_frame_AUC_ci95"] = [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]
    for name in rows:
        if name == a.reference:
            continue
        s = scores[name]["test"]
        d = np.array([roc_auc_score(yt[i], s[i]) - roc_auc_score(yt[i], ref[i]) for i in idx])
        rows[name]["test_AUC_diff_vs_reference"] = {"point": rows[name]["test_frame_AUC"] - rows[a.reference]["test_frame_AUC"],
                                                    "ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))]}
    (a.out / "sensitivity.json").write_text(json.dumps({"reference": a.reference, "runs": rows}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    def wilson(k, n, z=1.96):
        """Wilson score interval for a proportion k/n.

        Args:
            k: Number of successes.
            n: Number of trials (must be > 0).
            z: Normal quantile; 1.96 gives 95%.

        Returns:
            ``(low, high)``.
        """
        p_, d = k / n, 1 + z * z / n
        c = (p_ + z * z / (2 * n)) / d
        h = z * np.sqrt(p_ * (1 - p_) / n + z * z / (4 * n * n)) / d
        return c - h, c + h

    order = sorted(rows, key=lambda n: rows[n]["GFLOPs"])
    g = np.array([rows[n]["GFLOPs"] for n in order])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    auc = np.array([rows[n]["test_frame_AUC"] for n in order])
    ci = np.array([rows[n]["test_frame_AUC_ci95"] for n in order])
    axes[0].errorbar(g, auc, yerr=[auc - ci[:, 0], ci[:, 1] - auc], fmt="o", capsize=4, label="test (95% bootstrap CI)")
    axes[0].plot(g, [rows[n]["val_frame_AUC"] for n in order], "x", color="grey", label="validation")
    axes[0].set_ylim(0.6, 1.0)
    axes[0].set_ylabel("frame-level ROC AUC")
    axes[0].set_title("Defect-frame detection (threshold-free)", fontsize=10)
    for k, (cls, mk, col) in enumerate((("unsealed", "s", "C0"), ("packaging-error", "^", "C1"))):
        fr = np.array([rows[n]["test_box_recall"][cls] for n in order], float)
        r = fr[:, 0] / fr[:, 1]
        lo, hi = np.array([wilson(*f) for f in fr]).T
        x = g * (1.04 if k else 0.96)  # offset so the error bars do not overlap
        axes[1].errorbar(x, r, yerr=[r - lo, hi - r], fmt=mk, color=col, capsize=4,
                         label=f"{cls} ({int(fr[0, 1])} test boxes)")
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("test box recall (95% Wilson CI)")
    axes[1].set_title("Boxes found at each run's validation-chosen threshold", fontsize=10)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(g)
        ax.set_xticklabels([f"{n.replace('yolov8', 'v8')}\n{v:.1f} GFLOPs" for n, v in zip(order, g)], fontsize=8)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(g.min() / 1.5, g.max() * 1.5)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(a.out / "accuracy_vs_cost.png", dpi=140)
    for n, r in rows.items():
        print(n, json.dumps({k: r[k] for k in ("GFLOPs", "t4_inference_ms", "test_mAP50", "test_frame_AUC", "test_caught",
                                                "test_false_alarms", "threshold", "test_box_recall")}),
              r.get("test_AUC_diff_vs_reference", ""))


if __name__ == "__main__":
    main()
