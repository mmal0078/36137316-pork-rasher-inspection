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


#: Box colours per class for :func:`draw_report` (colour-blind-friendly hues, plus a distinct label strip).
CLASS_COLOURS = {"loose-meat": (230, 159, 0), "packaging-error": (86, 180, 233),
                 "twisted-meat": (204, 121, 167), "unsealed": (213, 94, 0)}


def draw_report(image: str | Path | Image.Image, report: dict, truth: list[tuple[str, tuple]] | None = None,
                threshold: float | None = None) -> Image.Image:
    """Draw the detections in ``report`` (from :func:`inspect_image`) on the image.

    Args:
        image: Image path or PIL image.
        report: Output of :func:`inspect_image`.
        truth: Optional labelled boxes as ``[(class_name, (x, y, w, h)), ...]`` in
            normalised coordinates. They are drawn as thin white dashed-look outlines,
            so predictions can be compared with the labels.
        threshold: Threshold used, shown in the verdict banner if given.

    Returns:
        A new RGB image with each detection outlined and labelled with its class
        and confidence, and a banner along the top: ``DEFECT`` (red) or
        ``NO DEFECT VISIBLE`` (green).
    """
    from PIL import ImageDraw, ImageFont

    img = (Image.open(image) if not isinstance(image, Image.Image) else image).convert("RGB").copy()
    w, h = img.size
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(14, h // 30))
    except OSError:
        font = ImageFont.load_default()

    def rect(b):
        """Convert a normalised (x_centre, y_centre, w, h) box to pixel corners.

        Args:
            b: Normalised box.

        Returns:
            ``[x0, y0, x1, y1]`` in pixels of the drawn image.
        """
        x, y, bw, bh = b
        return [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]

    for name, b in truth or []:
        x0, y0, x1, y1 = rect(b)
        for k in range(0, int(x1 - x0), 12):  # dashed outline: short segments along each edge
            d.line([x0 + k, y0, min(x0 + k + 6, x1), y0], fill="white", width=2)
            d.line([x0 + k, y1, min(x0 + k + 6, x1), y1], fill="white", width=2)
        for k in range(0, int(y1 - y0), 12):
            d.line([x0, y0 + k, x0, min(y0 + k + 6, y1)], fill="white", width=2)
            d.line([x1, y0 + k, x1, min(y0 + k + 6, y1)], fill="white", width=2)
    for det in report["detections"]:
        col = CLASS_COLOURS.get(det["class"], (255, 255, 0))
        x0, y0, x1, y1 = rect(det["box_xywhn"])
        d.rectangle([x0, y0, x1, y1], outline=col, width=4)
        text = f"{det['class']} {det['confidence']:.2f}"
        tw = d.textlength(text, font=font)
        ty = y0 - font.size - 6 if y0 - font.size - 6 > 40 else y0 + 2
        tx = min(x0, w - tw - 8)
        d.rectangle([tx, ty, tx + tw + 8, ty + font.size + 6], fill=col)
        d.text((tx + 4, ty + 2), text, fill="black", font=font)
    verdict = "DEFECT" if report["defect_present"] else "NO DEFECT VISIBLE"
    if threshold is not None:
        verdict += f"   (threshold {threshold:.2f})"
    d.rectangle([0, 0, w, font.size + 14], fill=(200, 30, 30) if report["defect_present"] else (30, 140, 60))
    d.text((8, 6), verdict, fill="white", font=font)
    return img
