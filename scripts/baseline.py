"""Baseline: hand-crafted features + unit-taught classifiers for "defect in frame?".

Pipeline:
    1. Extract features for every image (cached to ``--cache``):
       20 colour/shape features and 1,944 HOG features.
    2. For each feature set (simple, hog, simple+hog) and model (decision tree,
       RBF SVM, MLP), fit on the training images (all augmented copies) and
       pick hyperparameters by validation ROC AUC (threshold-free). AUC is used rather than
       average precision because 88% of frames contain a defect, so even random scores
       get an AP of about 0.88, while random AUC is 0.5.
    3. For the best configuration of each model, choose a score threshold on
       validation with the same safety-first rule as the detector (highest
       threshold with defect-frame recall >= 0.95), then evaluate once on test
       with bootstrap 95% intervals.
    4. Time feature extraction + prediction per image on the CPU.

Usage::

    python scripts/baseline.py --data data/processed --out results/baseline
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

from porkinspect.evaluate import bootstrap_ci  # noqa: E402
from porkinspect.features import colour_shape_features, hog_features  # noqa: E402
from porkinspect.labels import read_label_file  # noqa: E402

SPLITS = {"train": "train", "val": "valid", "test": "test"}


def extract(data: Path, cache: Path) -> dict:
    """Features, labels and file names per split, cached as ``.npz``."""
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {s: {k: z[f"{s}_{k}"] for k in ("simple", "hog", "y", "names")} for s in SPLITS}
    out = {}
    for s, folder in SPLITS.items():
        imgs = sorted((data / folder / "images").iterdir())
        simple = np.stack([colour_shape_features(p) for p in imgs])
        hog = np.stack([hog_features(p) for p in imgs]).astype(np.float32)
        y = np.array([len(read_label_file(data / folder / "labels" / (p.stem + ".txt"), 4)) > 0 for p in imgs])
        out[s] = {"simple": simple, "hog": hog, "y": y, "names": np.array([p.name for p in imgs])}
        print(f"{s}: {len(imgs)} images, {y.mean():.0%} with a defect", flush=True)
    np.savez_compressed(cache, **{f"{s}_{k}": v for s, d in out.items() for k, v in d.items()})
    return out


def feature_matrix(d: dict, which: str) -> np.ndarray:
    """Select a feature set: 'simple', 'hog' or 'simple+hog'."""
    if which == "simple+hog":
        return np.hstack([d["simple"], d["hog"]])
    return d[which]


def candidates():
    """(model name, hyperparameter label, unfitted sklearn pipeline) for the grid."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    for depth in (3, 5, 8, 12):
        for leaf in (5, 20, 50):
            yield "tree", f"depth={depth}, leaf={leaf}", DecisionTreeClassifier(
                max_depth=depth, min_samples_leaf=leaf, class_weight="balanced", random_state=0)
    for C in (0.3, 1, 3, 10):
        for gamma in ("scale", 0.3, 3):
            g = gamma if gamma == "scale" else gamma  # scaled by n_features below
            yield "svm", f"C={C}, gamma={gamma}", (C, g)
    for hidden in ((32,), (128,), (128, 32)):
        for alpha in (1e-4, 1e-2, 1):
            yield "mlp", f"hidden={hidden}, alpha={alpha}", make_pipeline(
                StandardScaler(), MLPClassifier(hidden, alpha=alpha, early_stopping=True, max_iter=500, random_state=0))


def build_svm(C, gamma, n_features):
    """RBF SVM with standardised inputs; numeric gamma is divided by n_features."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    g = gamma if gamma == "scale" else gamma / n_features
    return make_pipeline(StandardScaler(), SVC(C=C, gamma=g, class_weight="balanced"))


def scores(model, X):
    """Continuous defect score: decision function if available, else P(defect)."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def recall95_threshold(s: np.ndarray, y: np.ndarray) -> float:
    """Highest threshold whose recall on (s, y) is >= 0.95."""
    pos = np.sort(s[y])
    k = int(np.floor(0.05 * len(pos)))  # may miss at most 5% of defect frames
    return float(pos[k])


def evaluate(s: np.ndarray, y: np.ndarray, t: float) -> dict:
    """Frame-level counts and bootstrap intervals at threshold ``t``."""
    f = s >= t
    rec = lambda i: (f[i] & y[i]).sum() / y[i].sum() if y[i].sum() else np.nan  # noqa: E731
    far = lambda i: (f[i] & ~y[i]).sum() / (~y[i]).sum() if (~y[i]).sum() else np.nan  # noqa: E731
    idx = np.arange(len(y))
    return {"threshold": t, "tp": int((f & y).sum()), "fn": int((~f & y).sum()),
            "fp": int((f & ~y).sum()), "tn": int((~f & ~y).sum()),
            "recall": float(rec(idx)), "recall_ci95": bootstrap_ci(rec, len(y)),
            "false_alarm_rate": float(far(idx)), "false_alarm_ci95": bootstrap_ci(far, len(y))}


def main() -> None:
    """Run the grid, choose by validation AUC, evaluate on test, time on CPU."""
    from sklearn.base import clone
    from sklearn.metrics import average_precision_score, roc_auc_score

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "baseline")
    ap.add_argument("--cache", type=Path, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    D = extract(a.data, a.cache or a.out / "features.npz")
    yv, yt = D["val"]["y"], D["test"]["y"]

    grid, best = [], {}
    for fs in ("simple", "hog", "simple+hog"):
        Xtr, Xv = feature_matrix(D["train"], fs), feature_matrix(D["val"], fs)
        for name, label, spec in candidates():
            m = build_svm(*spec, Xtr.shape[1]) if name == "svm" else clone(spec)
            t0 = time.perf_counter()
            m.fit(Xtr, D["train"]["y"])
            sv = scores(m, Xv)
            row = {"features": fs, "model": name, "params": label, "val_AP": float(average_precision_score(yv, sv)),
                   "val_AUC": float(roc_auc_score(yv, sv)), "fit_s": round(time.perf_counter() - t0, 2)}
            grid.append(row)
            key = (name, fs)
            if key not in best or row["val_AUC"] > best[key][0]["val_AUC"]:
                best[key] = (row, m)
            print(f"{fs:11s} {name:4s} {label:28s} val AP {row['val_AP']:.3f} AUC {row['val_AUC']:.3f}", flush=True)

    results = {}
    for (name, fs), (row, m) in best.items():
        Xv, Xt = feature_matrix(D["val"], fs), feature_matrix(D["test"], fs)
        sv, st = scores(m, Xv), scores(m, Xt)
        t = recall95_threshold(sv, yv)
        results[f"{name}|{fs}"] = {**row, "test_AP": float(average_precision_score(yt, st)),
                                   "test_AUC": float(roc_auc_score(yt, st)),
                                   "val": evaluate(sv, yv, t), "test": evaluate(st, yt, t)}

    # CPU timing of the overall best configuration (by validation AP), end to end from the image file.
    top = max(results.values(), key=lambda r: r["val_AUC"])
    (_, m) = best[(top["model"], top["features"])]
    paths = sorted((a.data / "test" / "images").iterdir())
    t0 = time.perf_counter()
    for p in paths:
        x = colour_shape_features(p) if top["features"] == "simple" else (
            hog_features(p) if top["features"] == "hog" else np.hstack([colour_shape_features(p), hog_features(p)]))
        scores(m, x[None])
    top_ms = (time.perf_counter() - t0) * 1000 / len(paths)

    summary = {"best_by_val_AUC": {k: top[k] for k in ("model", "features", "params", "val_AUC", "val_AP")},
               "cpu_ms_per_image_end_to_end": round(top_ms, 1), "results": results}
    (a.out / "baseline_results.json").write_text(json.dumps(summary, indent=2))
    (a.out / "baseline_grid.json").write_text(json.dumps(grid, indent=2))
    for k, r in results.items():
        v, te = r["val"], r["test"]
        print(f"{k:18s} {r['params']:28s} valAUC {r['val_AUC']:.3f} testAUC {r['test_AUC']:.3f} | test caught "
              f"{te['tp']}/{te['tp']+te['fn']} FA {te['fp']}/{te['fp']+te['tn']}")
    print("best:", summary["best_by_val_AUC"], "CPU ms/img", summary["cpu_ms_per_image_end_to_end"])


if __name__ == "__main__":
    main()
