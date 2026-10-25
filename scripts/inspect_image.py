"""Inspect one or more tray images from the command line.

Prints one JSON line per image: whether a defect is present, the detections
above the threshold, and any input warnings. Invalid files are reported and
skipped, not treated as "no defect".

Usage::

    python scripts/inspect_image.py --weights runs/yolov8n_640/weights/best.pt img1.jpg img2.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from porkinspect.inspection import inspect_image  # noqa: E402


def main() -> int:
    """Run the detector on each image; return a non-zero exit code if any image is invalid."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.16)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.weights)
    failed = 0
    for p in a.images:
        try:
            r = inspect_image(p, a.weights, a.threshold, device=a.device, model=model)
            print(json.dumps({"image": str(p), **r}))
        except ValueError as exc:
            failed += 1
            print(json.dumps({"image": str(p), "error": str(exc)}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
