"""Group the files of a Roboflow YOLO export into distinct source images.

Roboflow exports each training image three times (random exposure
augmentation), so one source image can appear as several files. Copies share
a file-name stem (``frame_with_red_<n>``), **but the stem is not unique**: in
this dataset 45 stems are each used by two different images, probably because
two recordings were both numbered from 1. File names alone therefore cannot
tell copies from different images.

This module groups files in two steps:

1. by file-name stem;
2. within a stem, by image content: two files are copies of the same image if
   the correlation of their normalised greyscale thumbnails exceeds
   ``threshold`` (default 0.9).

In this dataset copies correlate at 0.95-1.0 and different images below 0.8,
with no pairs in between, so the result does not depend on the exact
threshold. The grouping is used to count distinct images, check that no image
appears in two official splits, and find copies whose labels disagree.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .labels import CLASS_NAMES, Box, read_label_file, source_stem

#: Folder names used by Roboflow for the three official splits.
OFFICIAL_SPLITS: tuple[str, ...] = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
#: Thumbnail size (width, height) used to compare images.
THUMB_SIZE = (180, 135)


@dataclass(frozen=True)
class ImageFile:
    """One exported image file and its labels.

    Attributes:
        split: Official split folder the file came from.
        image: Path to the image file.
        label: Path to the matching label file.
        boxes: Boxes read from the label file.
    """

    split: str
    image: Path
    label: Path
    boxes: tuple[Box, ...]


@dataclass
class SourceImage:
    """All exported copies of one distinct source image.

    Attributes:
        key: Unique id, ``"<stem>#<k>"`` where ``k`` numbers the distinct
            images sharing that stem.
        copies: The exported files of this image.
    """

    key: str
    copies: list[ImageFile] = field(default_factory=list)

    @property
    def splits(self) -> set[str]:
        """Official splits that contain a copy of this image."""
        return {c.split for c in self.copies}

    @property
    def label_conflict(self) -> bool:
        """True if the copies do not all carry the same boxes."""
        return len({c.boxes for c in self.copies}) > 1

    @property
    def boxes(self) -> tuple[Box, ...]:
        """Boxes of the first copy (all copies agree unless :attr:`label_conflict`)."""
        return self.copies[0].boxes


def thumbnail_vector(path: str | Path) -> np.ndarray:
    """Return a zero-mean, unit-variance greyscale thumbnail as a flat vector.

    Normalising removes overall brightness and contrast, so exposure-augmented
    copies of one image give nearly identical vectors.

    Args:
        path: Image file.

    Returns:
        1-D float array of length ``THUMB_SIZE[0] * THUMB_SIZE[1]``.

    Raises:
        ValueError: If the image is a single flat colour (zero variance), since
            its correlation with anything is undefined.
    """
    a = np.asarray(Image.open(path).convert("L").resize(THUMB_SIZE), dtype=float).ravel()
    sd = a.std()
    if sd == 0:
        raise ValueError(f"image has no variation: {path}")
    return (a - a.mean()) / sd


def correlation(u: np.ndarray, v: np.ndarray) -> float:
    """Pearson correlation of two vectors from :func:`thumbnail_vector`.

    Because both vectors already have mean 0 and standard deviation 1, this
    is just their dot product divided by their length.

    Args:
        u: First normalised vector.
        v: Second normalised vector, same length.

    Returns:
        Correlation in [-1, 1]; above 0.9 is treated as the same image.
    """
    return float(u @ v) / len(u)


def cluster_by_content(vectors: list[np.ndarray], threshold: float = 0.9) -> list[list[int]]:
    """Group vectors whose correlation with a group's first member exceeds ``threshold``.

    Args:
        vectors: Normalised thumbnails, as returned by :func:`thumbnail_vector`.
        threshold: Correlation above which two images count as the same image.

    Returns:
        Lists of indices into ``vectors``, in order of first appearance.
    """
    groups: list[list[int]] = []
    for i, v in enumerate(vectors):
        for g in groups:
            if correlation(v, vectors[g[0]]) > threshold:
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


def build_index(root: str | Path, threshold: float = 0.9) -> list[SourceImage]:
    """Group every file of a Roboflow YOLO export into distinct source images.

    Args:
        root: Folder containing ``data.yaml`` and the split folders.
        threshold: Correlation threshold passed to :func:`cluster_by_content`.

    Returns:
        Source images sorted by key.

    Raises:
        FileNotFoundError: If a split folder or a label file is missing.
        ValueError: If a label file contains an invalid line.
    """
    root = Path(root)
    by_stem: dict[str, list[ImageFile]] = {}
    for split in OFFICIAL_SPLITS:
        img_dir = root / split / "images"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"missing folder {img_dir}")
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            lab = root / split / "labels" / (img.stem + ".txt")
            by_stem.setdefault(source_stem(img.name), []).append(
                ImageFile(split, img, lab, read_label_file(lab))
            )

    out: list[SourceImage] = []
    for stem, files in sorted(by_stem.items()):
        if len(files) == 1:
            out.append(SourceImage(f"{stem}#0", files))
            continue
        groups = cluster_by_content([thumbnail_vector(f.image) for f in files], threshold)
        out.extend(SourceImage(f"{stem}#{k}", [files[i] for i in g]) for k, g in enumerate(groups))
    return out


def summarise(images: list[SourceImage]) -> dict:
    """Return the headline counts used in the data audit.

    Box and image counts per class are per distinct image (one copy each), so
    augmentation does not inflate them.

    Args:
        images: Distinct source images from :func:`build_index`.

    Returns:
        JSON-serialisable dict of per-split file, image, box and class counts,
        plus counts of shared file names and label conflicts.
    """
    stems = Counter(im.key.split("#")[0] for im in images)
    per_split: dict[str, dict] = {}
    for s in OFFICIAL_SPLITS:
        ims = [im for im in images if s in im.splits]
        boxes = Counter(b.cls for im in ims for b in im.boxes)
        has = Counter(k for im in ims for k in {b.cls for b in im.boxes})
        per_split[s] = {
            "files": sum(c.split == s for im in images for c in im.copies),
            "distinct_images": len(ims),
            "images_without_boxes": sum(not im.boxes for im in ims),
            "boxes_per_class": {CLASS_NAMES[k]: boxes[k] for k in range(len(CLASS_NAMES))},
            "images_per_class": {CLASS_NAMES[k]: has[k] for k in range(len(CLASS_NAMES))},
        }
    return {
        "files": sum(len(im.copies) for im in images),
        "distinct_images": len(images),
        "copies_per_image": dict(sorted(Counter(len(im.copies) for im in images).items())),
        "stems_shared_by_different_images": sum(n > 1 for n in stems.values()),
        "images_in_more_than_one_split": sum(len(im.splits) > 1 for im in images),
        "images_with_conflicting_labels": sum(im.label_conflict for im in images),
        "per_split": per_split,
    }
