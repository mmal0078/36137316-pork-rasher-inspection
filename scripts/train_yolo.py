"""Fine-tune a YOLO detector, evaluate it and save raw predictions.

Designed to run on a GPU (e.g. Google Colab). Everything needed for later
analysis is written to ``<out>/<name>/``, so the analysis itself can run on any
machine without a GPU:

    weights/best.pt          best checkpoint (highest validation mAP50-95)
    results.csv              per-epoch training log (written by Ultralytics)
    metrics.json             validation and test metrics, model size, GFLOPs, timings
    predictions_val.json     every prediction on the validation images (conf >= 0.001)
    predictions_test.json    the same for the test images
    run_info.json            arguments, library versions and hardware

Usage::

    python scripts/train_yolo.py --data data/processed/data.yaml --model yolov8n.pt \
        --epochs 60 --imgsz 640 --name yolov8n_640
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path


def dump_predictions(model, image_dir: Path, imgsz: int, out_json: Path, conf: float = 0.001) -> dict:
    """Run the model on every image in ``image_dir`` and save all detections.

    A very low confidence floor is used so that any threshold can be applied
    later without re-running the model.

    Args:
        model: Loaded Ultralytics YOLO model.
        image_dir: Folder of images.
        imgsz: Inference image size in pixels.
        out_json: Where to write the predictions.
        conf: Confidence floor for saved detections.

    Returns:
        Timing summary: number of images and mean wall-clock milliseconds per
        image for the whole predict call (pre-processing + inference + NMS).
    """
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    records = []
    t0 = time.perf_counter()
    for r in model.predict(source=[str(p) for p in images], imgsz=imgsz, conf=conf, iou=0.7,
                           stream=True, verbose=False, batch=1):
        b = r.boxes
        records.append({
            "image": Path(r.path).name,
            "cls": b.cls.int().tolist(),
            "conf": [round(c, 5) for c in b.conf.tolist()],
            "xywhn": [[round(v, 6) for v in row] for row in b.xywhn.tolist()],
            "speed_ms": {k: round(v, 3) for k, v in r.speed.items()},
        })
    wall = (time.perf_counter() - t0) * 1000 / max(len(images), 1)
    out_json.write_text(json.dumps(records))
    return {"images": len(images), "wall_ms_per_image": round(wall, 2)}


def metrics_dict(m, names: dict) -> dict:
    """Convert an Ultralytics validation result into plain numbers.

    Args:
        m: Result object returned by ``model.val()``.
        names: Class id -> class name.

    Returns:
        JSON-serialisable dict of overall and per-class precision, recall,
        mAP50 and mAP50-95, plus Ultralytics' speed breakdown.
    """
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index.tolist()):
        p, r, ap50, ap = m.box.class_result(i)
        per_class[names[c]] = {"precision": p, "recall": r, "mAP50": ap50, "mAP50_95": ap}
    return {
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "per_class": per_class,
        "speed_ms_per_image": {k: round(v, 3) for k, v in m.speed.items()},
    }


def main() -> None:
    """Parse arguments, train, evaluate and save outputs."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="path to data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="pretrained weights to start from")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20, help="stop if val mAP has not improved for this many epochs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--name", default="yolov8n_640")
    ap.add_argument("--device", default=None, help="e.g. 0 for the first GPU, cpu for CPU")
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of training images to use (for quick tests)")
    a = ap.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO

    run_dir = (a.out / a.name).resolve()
    data_root = a.data.resolve().parent

    model = YOLO(a.model)
    t0 = time.perf_counter()
    model.train(data=str(a.data.resolve()), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                patience=a.patience, seed=a.seed, deterministic=True, project=str(run_dir.parent),
                name=run_dir.name, exist_ok=True, device=a.device, fraction=a.fraction, plots=True,
                verbose=False)
    train_minutes = (time.perf_counter() - t0) / 60

    best = YOLO(run_dir / "weights" / "best.pt")
    names = best.names
    out = {"train_minutes": round(train_minutes, 2)}
    for split in ("val", "test"):
        m = best.val(data=str(a.data.resolve()), split=split, imgsz=a.imgsz, batch=a.batch, conf=0.001,
                     iou=0.6, device=a.device, project=str(run_dir), name=f"eval_{split}", exist_ok=True,
                     plots=True, verbose=False)
        out[split] = metrics_dict(m, names)

    from ultralytics.utils.torch_utils import get_flops

    # get_flops profiles one forward pass with thop. Ultralytics reports 2 x MACs as "FLOPs".
    out["model"] = {
        "parameters": int(sum(p.numel() for p in best.model.parameters())),
        "GFLOPs_at_imgsz": round(float(get_flops(best.model, imgsz=a.imgsz)), 3),
        "GFLOPs_at_640": round(float(get_flops(best.model, imgsz=640)), 3),
    }
    for split, folder in (("val", "valid"), ("test", "test")):
        out[f"predict_timing_{split}"] = dump_predictions(
            best, data_root / folder / "images", a.imgsz, run_dir / f"predictions_{split}.json")
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2))

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    (run_dir / "run_info.json").write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(a).items()},
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "python": platform.python_version(), "gpu": gpu, "cpu": platform.processor() or platform.machine(),
    }, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
