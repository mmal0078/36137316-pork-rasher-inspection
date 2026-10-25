"""Write a training-ready copy of the dataset on the official split.

The only change from the Roboflow export is the class list: dropped classes
(by default ``wrinkle``, which has too few examples to train or evaluate) are
removed from every label file and the remaining class ids are renumbered.

Output layout (``out``)::

    train/images, train/labels
    valid/images, valid/labels
    test/images,  test/labels
    data.yaml     Ultralytics dataset file (relative paths, so the folder can be moved)
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .index import OFFICIAL_SPLITS, SourceImage
from .labels import CLASS_NAMES, Box


def remap_boxes(boxes: tuple[Box, ...], keep: list[int]) -> list[Box]:
    """Remove boxes of classes not in ``keep`` and renumber the rest.

    Args:
        boxes: Original boxes.
        keep: Original class ids to keep, in their new order (new id = position).

    Returns:
        Remapped boxes, in their original order.
    """
    new_id = {old: new for new, old in enumerate(keep)}
    return [Box(new_id[b.cls], b.x, b.y, b.w, b.h) for b in boxes if b.cls in new_id]


def write_data_yaml(path: Path, names: list[str]) -> None:
    """Write an Ultralytics ``data.yaml`` with paths relative to its folder."""
    path.write_text(
        "path: .\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        f"nc: {len(names)}\n"
        "names: [" + ", ".join(f"'{n}'" for n in names) + "]\n"
    )


def export_dataset(images: list[SourceImage], out: str | Path, keep_classes: list[int]) -> dict[str, int]:
    """Copy every file into ``out`` with remapped labels, keeping the official split.

    Args:
        images: Output of :func:`porkinspect.index.build_index`.
        out: Destination folder (created if needed).
        keep_classes: Original class ids to keep, in their new order.

    Returns:
        Number of image files written per split.
    """
    out = Path(out)
    counts = {s: 0 for s in OFFICIAL_SPLITS}
    for s in OFFICIAL_SPLITS:
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "labels").mkdir(parents=True, exist_ok=True)
    for im in images:
        for c in im.copies:
            shutil.copy2(c.image, out / c.split / "images" / c.image.name)
            lines = "".join(b.to_line() + "\n" for b in remap_boxes(c.boxes, keep_classes))
            (out / c.split / "labels" / (c.image.stem + ".txt")).write_text(lines)
            counts[c.split] += 1
    write_data_yaml(out / "data.yaml", [CLASS_NAMES[k] for k in keep_classes])
    return counts
