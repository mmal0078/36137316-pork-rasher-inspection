"""Run the trained detector on one image, with input checks.

This is the top-level "use the system" entry point:

    >>> from porkinspect.inspection import inspect_image          # doctest: +SKIP
    >>> report = inspect_image("tray.jpg", "best.pt")          # doctest: +SKIP
    >>> report["defect_present"], report["detections"][0]      # doctest: +SKIP

Input handling (see :func:`check_image`):

* missing file, unreadable or corrupt image: ``ValueError`` (never silently
  "no defect");
* greyscale, RGBA or palette images: converted to RGB, with a warning;
* image size different from 720 x 540: accepted (the detector resizes to its
  input size), with a warning, because accuracy was only measured at 720 x 540;
* belt brightness outside the range the model saw in training (17-109 of 255,
  the minimum and maximum over all 3,349 training files including augmented
  copies): accepted with a warning. The brightness test showed stable accuracy
  from 0.4x to 2x, but behaviour outside the training range is not guaranteed.

Warnings are returned in the report rather than printed, so a production line
can log them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

#: Image size (width, height) of the training and test images.
EXPECTED_SIZE = (720, 540)
#: Belt brightness range (0-255, mean grey of the top 200 rows) treated as in-domain:
#: the minimum and maximum over all training files (measured 25 Oct 2026).
BELT_BRIGHTNESS_OK = (17.0, 109.0)


def check_image(path: str | Path) -> tuple[Image.Image, list[str]]:
    """Load and validate an input image.

    Args:
        path: Image file to inspect.

    Returns:
        ``(rgb_image, warnings)``: the image converted to RGB, and a list of
        human-readable warnings about out-of-domain properties.

    Raises:
        ValueError: If the file does not exist, is not a readable image, or is
            truncated or corrupt.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"input image not found: {path}")
    try:
        with Image.open(path) as im:
            im.load()  # forces full decoding, so a truncated file fails here
            mode, size = im.mode, im.size
            rgb = im.convert("RGB")
    # ImportError is included because Ultralytics patches Image.open after import: on an
    # unreadable file its fallback tries to import an optional HEIC plugin (pi_heif) and
    # raises ModuleNotFoundError instead of UnidentifiedImageError.
    except (UnidentifiedImageError, OSError, SyntaxError, ImportError) as exc:
        raise ValueError(f"not a readable image: {path} ({exc})") from exc

    warnings = []
    if mode != "RGB":
        warnings.append(f"image mode {mode} converted to RGB")
    if size != EXPECTED_SIZE:
        warnings.append(f"image size {size[0]}x{size[1]} differs from the {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]} "
                        "training images; accuracy at other sizes was not measured")
    top = np.asarray(rgb.convert("L").resize(EXPECTED_SIZE), dtype=float)[:200]
    b = float(top.mean())
    lo, hi = BELT_BRIGHTNESS_OK
    if not lo <= b <= hi:
        warnings.append(f"belt brightness {b:.0f}/255 is outside the training range {lo:.0f}-{hi:.0f}")
    return rgb, warnings


def inspect_image(path: str | Path, weights: str | Path, threshold: float = 0.16, imgsz: int = 640,
                  device: str = "cpu", model=None) -> dict:
    """Detect defects in one image and decide whether to flag the frame.

    Args:
        path: Image file.
        weights: Trained detector weights (``best.pt``). Ignored if ``model`` is given.
        threshold: Confidence threshold for flagging. The default, 0.16, was chosen on
            validation so that at least 95% of defect frames are caught.
        imgsz: Detector input size in pixels.
        device: ``"cpu"`` or a GPU index such as ``"0"``.
        model: An already-loaded Ultralytics ``YOLO`` model, to avoid reloading per image.

    Returns:
        A dict with:
            ``defect_present`` (bool): any detection with confidence >= ``threshold``;
            ``detections``: list of ``{"class", "confidence", "box_xywhn"}`` at or
            above the threshold, highest confidence first;
            ``max_confidence`` (float): highest confidence, 0 if none;
            ``warnings``: list of input warnings from :func:`check_image`;
            ``inference_ms`` (float): model time for this image.

    Raises:
        ValueError: If the image is invalid, or ``threshold`` is outside [0, 1].
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    rgb, warnings = check_image(path)
    if model is None:
        from ultralytics import YOLO

        model = YOLO(weights)
    # Ultralytics expects BGR arrays (OpenCV order), so convert explicitly.
    r = model.predict(np.asarray(rgb)[..., ::-1], imgsz=imgsz, conf=0.001, device=device, verbose=False)[0]
    dets = sorted(
        ({"class": model.names[int(c)], "confidence": float(p), "box_xywhn": [float(v) for v in b]}
         for c, p, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xywhn.tolist())),
        key=lambda d: -d["confidence"])
    kept = [d for d in dets if d["confidence"] >= threshold]
    return {"defect_present": bool(kept), "detections": kept,
            "max_confidence": dets[0]["confidence"] if dets else 0.0,
            "warnings": warnings, "inference_ms": float(r.speed["inference"])}
