"""Reading YOLO-format label files and Roboflow file names.

A YOLO label file has one line per bounding box::

    <class_id> <x_centre> <y_centre> <width> <height>

All four coordinates are normalised to [0, 1] by the image width and height,
so they have no units. An empty file means "no labelled defect in this image".

Roboflow names exported files ``<original-stem>_jpg.rf.<hash>.jpg``. Every
augmented copy of one original image shares ``<original-stem>``, so the stem is
the key used to keep copies together. In this dataset the stem is
``frame_with_red_<n>``, where ``n`` is the video frame number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Class names in the order of the class ids in the Roboflow ``data.yaml``.
CLASS_NAMES: tuple[str, ...] = (
    "loose-meat",
    "packaging-error",
    "twisted-meat",
    "unsealed",
    "wrinkle",
)

_ROBOFLOW_NAME = re.compile(
    r"^(?P<stem>.+)_(?:jpg|jpeg|png)\.rf\.(?P<hash>[0-9a-f]+)\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)
_FRAME_NUMBER = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class Box:
    """One labelled bounding box in normalised YOLO coordinates.

    Attributes:
        cls: Integer class id (index into :data:`CLASS_NAMES`).
        x: Box centre x, as a fraction of image width (0-1).
        y: Box centre y, as a fraction of image height (0-1).
        w: Box width, as a fraction of image width (0-1].
        h: Box height, as a fraction of image height (0-1].
    """

    cls: int
    x: float
    y: float
    w: float
    h: float

    def to_line(self) -> str:
        """Return the box as one YOLO label line (6 decimal places)."""
        return f"{self.cls} {self.x:.6f} {self.y:.6f} {self.w:.6f} {self.h:.6f}"


def parse_label_line(line: str, n_classes: int = len(CLASS_NAMES)) -> Box:
    """Parse and validate one line of a YOLO label file.

    Args:
        line: Text of the form ``"cls x y w h"``.
        n_classes: Number of valid classes; ids must lie in ``[0, n_classes)``.

    Returns:
        The parsed :class:`Box`.

    Raises:
        ValueError: If the line does not have 5 fields, the class id is not a
            valid integer, or a coordinate lies outside [0, 1] (or a width or
            height is zero). Segmentation polygons (more than 5 fields) are
            rejected rather than silently truncated.
    """
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"expected 5 fields (cls x y w h), got {len(parts)}: {line!r}")
    try:
        cls = int(parts[0])
    except ValueError as exc:
        raise ValueError(f"class id is not an integer: {line!r}") from exc
    if not 0 <= cls < n_classes:
        raise ValueError(f"class id {cls} outside [0, {n_classes}): {line!r}")
    x, y, w, h = (float(p) for p in parts[1:])
    # A small tolerance allows for rounding in the exported text.
    tol = 1e-6
    if not all(-tol <= v <= 1 + tol for v in (x, y, w, h)):
        raise ValueError(f"coordinate outside [0, 1]: {line!r}")
    if w <= 0 or h <= 0:
        raise ValueError(f"box has zero width or height: {line!r}")
    return Box(cls, x, y, w, h)


def read_label_file(path: str | Path, n_classes: int = len(CLASS_NAMES)) -> tuple[Box, ...]:
    """Read every box in a YOLO label file.

    Args:
        path: Path to the ``.txt`` label file.
        n_classes: Number of valid classes, passed to :func:`parse_label_line`.

    Returns:
        A tuple of boxes, sorted so that two files with the same boxes in a
        different order compare equal. Empty tuple for an empty file.

    Raises:
        FileNotFoundError: If the file does not exist. A missing label file is
            treated as an error, not as "no defect", because the two cannot be
            told apart.
    """
    text = Path(path).read_text()
    boxes = [parse_label_line(ln, n_classes) for ln in text.splitlines() if ln.strip()]
    return tuple(sorted(boxes, key=lambda b: (b.cls, b.x, b.y, b.w, b.h)))


def source_stem(filename: str) -> str:
    """Return the original-image stem of a Roboflow export file name.

    Example:
        >>> source_stem("frame_with_red_10049_jpg.rf.cd786c4f.jpg")
        'frame_with_red_10049'

    Args:
        filename: Base name of an exported image (no directory).

    Returns:
        The stem shared by all augmented copies of the same original image.
        Names that do not follow the Roboflow pattern are returned without
        their extension.
    """
    m = _ROBOFLOW_NAME.match(filename)
    return m.group("stem") if m else Path(filename).stem


def frame_number(filename: str) -> int:
    """Return the video frame number encoded at the end of the stem.

    Args:
        filename: Roboflow export file name.

    Returns:
        The frame number.

    Example:
        >>> frame_number("frame_with_red_10049_jpg.rf.cd786c4f.jpg")
        10049

    Raises:
        ValueError: If the stem does not end in a number.
    """
    stem = source_stem(filename)
    m = _FRAME_NUMBER.search(stem)
    if m is None:
        raise ValueError(f"no frame number at end of {stem!r}")
    return int(m.group(1))
