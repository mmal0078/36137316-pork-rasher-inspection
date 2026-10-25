"""Robustness of a trained detector to global brightness changes.

Each test image is multiplied by a brightness factor (pixel values clipped to
0-255). The detector is re-run, and the results are compared with the
original images at a **fixed** threshold chosen on validation. A deployed
system cannot re-tune its threshold whenever the lighting changes.

For reference, the belt brightness measured in this dataset spans about 0.7x
to 1.4x its median (5th to 95th percentile, ``results/data_audit.json``), so
factors in that range are realistic; factors outside it are stress tests.

Reported per factor:
    mAP50 (Ultralytics validator on the perturbed copy of the test split),
    frame-level ROC AUC, defect frames caught and false alarms at the fixed threshold.

Usage::

    python scripts/brightness_test.py --run runs/yolov8n_640 --data data/processed --threshold 0.16
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from porkinspect.labels import read_label_file  # noqa: E402

FACTORS = (0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.4, 1.7, 2.0)


def perturb(src: Path, dst: Path, factor: float) -> None:
    """Write a copy of an image with every pixel multiplied by ``factor``.

    Args:
        src: Input image.
        dst: Output JPEG path.
        factor: Brightness multiplier; results are clipped to 0-255.
    """
    a = np.asarray(Image.open(src).convert("RGB"), dtype=float) * factor
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(dst, quality=95)


def plot(rows: list[dict], threshold: float, path: Path) -> None:
    """Plot mAP50 and frame catch rate against the brightness factor.

    Args:
        rows: One result dict per brightness factor.
        threshold: Frame-flag threshold, shown in the legend.
        path: Output PNG path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    fx = [r["factor"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axvspan(0.7, 1.4, color="0.92", label="brightness range seen in the data")
    ax.plot(fx, [r["mAP50"] for r in rows], "o-", label="mAP50 (all classes)")
    ax.plot(fx, [r["mAP50_per_class"].get("unsealed", np.nan) for r in rows], "s--", label="mAP50 unsealed")
    ax.plot(fx, [r["mAP50_per_class"].get("packaging-error", np.nan) for r in rows], "^--", label="mAP50 packaging-error")
    ax.plot(fx, [r["defect_frames_caught"] / r["defect_frames"] for r in rows], "d-",
            label=f"defect frames caught (threshold {threshold})")
    ax.set_xscale("log")
    ax.set_xticks(fx)
    ax.set_xticklabels([f"{v:g}x" for v in fx])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("brightness factor applied to test images")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7.5, loc="lower center")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    """Run the detector on brightness-scaled copies of the test split."""
    from sklearn.metrics import roc_auc_score
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--threshold", type=float, required=True, help="frame threshold chosen on validation")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    out = a.out or ROOT / "results" / a.run.name
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(a.run / "weights" / "best.pt")
    names = model.names
    imgs = sorted((a.data / "test" / "images").iterdir())
    y = np.array([len(read_label_file(a.data / "test" / "labels" / (p.stem + ".txt"), len(names))) > 0 for p in imgs])

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in FACTORS:
            d = tmp / f"b{f}"
            (d / "test" / "images").mkdir(parents=True)
            shutil.copytree(a.data / "test" / "labels", d / "test" / "labels")
            for p in imgs:
                perturb(p, d / "test" / "images" / p.name, f)
            (d / "data.yaml").write_text(
                f"path: {d}\ntrain: test/images\nval: test/images\ntest: test/images\n"
                f"nc: {len(names)}\nnames: {[names[i] for i in range(len(names))]}\n")
            m = model.val(data=str(d / "data.yaml"), split="test", imgsz=a.imgsz, conf=0.001, iou=0.6,
                          device=a.device, plots=False, verbose=False, project=str(tmp), name=f"v{f}")
            per_class = {names[c]: float(m.box.class_result(i)[2]) for i, c in enumerate(m.box.ap_class_index.tolist())}
            score = []
            for p in imgs:
                r = model.predict(str(d / "test" / "images" / p.name), imgsz=a.imgsz, conf=0.001,
                                  device=a.device, verbose=False)[0]
                score.append(float(r.boxes.conf.max()) if len(r.boxes) else 0.0)
            s = np.array(score)
            flag = s >= a.threshold
            rows.append({"factor": f, "mAP50": float(m.box.map50), "mAP50_per_class": per_class,
                         "frame_AUC": float(roc_auc_score(y, s)),
                         "defect_frames_caught": int((flag & y).sum()), "defect_frames": int(y.sum()),
                         "false_alarms": int((flag & ~y).sum()), "defect_free_frames": int((~y).sum())})
            print(json.dumps(rows[-1]), flush=True)

    (out / "brightness_test.json").write_text(json.dumps({"threshold": a.threshold, "rows": rows}, indent=2))

    plot(rows, a.threshold, out / "brightness_test.png")

if __name__ == "__main__":
    main()
