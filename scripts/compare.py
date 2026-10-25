"""Compare the YOLO detector with the baseline on the same frame-level question.

The YOLO frame score is the highest detection confidence in the frame. The
baseline score is the classifier's decision value. Both are compared:

* threshold-free: ROC AUC on test, with a paired bootstrap interval for the
  difference (the same resampled images are used for both models);
* at the operating point: threshold chosen on validation so that >= 95% of
  defect frames are caught, then applied once to test.

It also times both methods end to end on this machine's CPU.

Usage::

    python scripts/compare.py --run runs/yolov8n_640 --data data/processed \
        --baseline results/baseline --features results/baseline/features.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from baseline import build_svm, evaluate, extract, feature_matrix, recall95_threshold, scores  # noqa: E402
from porkinspect.evaluate import load_predictions  # noqa: E402
from porkinspect.features import hog_features  # noqa: E402


def yolo_frame_scores(pred_json: Path, names: np.ndarray) -> np.ndarray:
    """Frame-level defect score from the detector.

    Args:
        pred_json: Saved predictions file.
        names: Image file names, in the wanted order.

    Returns:
        Highest detection confidence per image (0 if no detection).
    """
    p = load_predictions(pred_json)
    return np.array([max((d.conf for d in p.get(str(n), [])), default=0.0) for n in names])


def main() -> None:
    """Fit the chosen baseline, score both methods, compare and time them."""
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--baseline", type=Path, default=ROOT / "results" / "baseline")
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "comparison")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    D = extract(a.data, a.features or a.baseline / "features.npz")
    best = json.loads((a.baseline / "baseline_results.json").read_text())["best_by_val_AUC"]
    assert best["model"] == "svm", "this script refits the SVM chosen by baseline.py"
    C, gamma = (float(v.split("=")[1]) if v.split("=")[1] != "scale" else "scale"
                for v in best["params"].split(", "))
    Xtr = feature_matrix(D["train"], best["features"])
    model = build_svm(C, gamma, Xtr.shape[1]).fit(Xtr, D["train"]["y"])

    s = {"baseline": {}, "yolo": {}}
    for split in ("val", "test"):
        s["baseline"][split] = scores(model, feature_matrix(D[split], best["features"]))
        s["yolo"][split] = yolo_frame_scores(a.run / f"predictions_{split}.json", D[split]["names"])

    yt, yv = D["test"]["y"], D["val"]["y"]
    out = {"baseline_config": best}
    for m in ("baseline", "yolo"):
        t = recall95_threshold(s[m]["val"], yv)
        out[m] = {"val_AUC": float(roc_auc_score(yv, s[m]["val"])), "test_AUC": float(roc_auc_score(yt, s[m]["test"])),
                  "val": evaluate(s[m]["val"], yv, t), "test": evaluate(s[m]["test"], yt, t)}

    # Paired bootstrap of the test AUC difference (resamples lacking either class are skipped).
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, len(yt), len(yt))
        if yt[i].all() or not yt[i].any():
            continue
        diffs.append(roc_auc_score(yt[i], s["yolo"]["test"][i]) - roc_auc_score(yt[i], s["baseline"]["test"][i]))
    d = np.array(diffs)
    out["test_AUC_difference_yolo_minus_baseline"] = {
        "point": out["yolo"]["test_AUC"] - out["baseline"]["test_AUC"],
        "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
        "share_of_resamples_yolo_better": float((d > 0).mean())}

    # CPU timing, end to end from the image file, batch size 1, after one warm-up image.
    paths = sorted((a.data / "test" / "images").iterdir())
    hog_features(paths[0])
    t0 = time.perf_counter()
    for p in paths:
        scores(model, hog_features(p)[None])
    base_ms = (time.perf_counter() - t0) * 1000 / len(paths)

    from ultralytics import YOLO

    yolo = YOLO(a.run / "weights" / "best.pt")
    yolo.predict(str(paths[0]), imgsz=640, device="cpu", verbose=False)
    t0 = time.perf_counter()
    for p in paths:
        yolo.predict(str(p), imgsz=640, conf=0.001, device="cpu", verbose=False)
    yolo_ms = (time.perf_counter() - t0) * 1000 / len(paths)
    import platform

    out["cpu_timing_ms_per_image"] = {"baseline_hog_svm": round(base_ms, 1), "yolov8n_640": round(yolo_ms, 1),
                                      "cpu": platform.processor() or platform.machine(), "threads": __import__("torch").get_num_threads()}
    (a.out / "comparison.json").write_text(json.dumps(out, indent=2))
    np.savez(a.out / "frame_scores.npz", **{f"{m}_{sp}": v for m, d_ in s.items() for sp, v in d_.items()},
             y_val=yv, y_test=yt)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for m, lab in (("yolo", "YOLOv8n (max confidence)"), ("baseline", f"HOG + SVM baseline")):
        fpr, tpr, _ = roc_curve(yt, s[m]["test"])
        ax.step(fpr, tpr, where="post", label=f"{lab}, AUC {out[m]['test_AUC']:.2f}")
        te = out[m]["test"]
        ax.plot(te["false_alarm_rate"], te["recall"], "o", color=ax.lines[-1].get_color())
    ax.plot([0, 1], [0, 1], ":", color="grey", lw=1)
    ax.set_xlabel("false-alarm rate (defect-free frames flagged)")
    ax.set_ylabel("defect frames caught")
    ax.set_title(f"Test set: {int(yt.sum())} defect / {int((~yt).sum())} defect-free frames", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(a.out / "roc_test.png", dpi=140)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
