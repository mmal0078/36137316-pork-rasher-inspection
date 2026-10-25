"""Tests for porkinspect.index and porkinspect.export on a tiny fake export."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from porkinspect.export import export_dataset, remap_boxes
from porkinspect.index import build_index, cluster_by_content, summarise, thumbnail_vector
from porkinspect.labels import Box, read_label_file


def pattern(seed: int, gain: float = 1.0) -> Image.Image:
    """A random textured image; ``gain`` mimics an exposure change."""
    rng = np.random.default_rng(seed)
    base = rng.integers(20, 200, (54, 72), dtype=np.uint8)
    big = np.kron(base, np.ones((10, 10)))  # 540 x 720, blocky so resizing keeps it
    return Image.fromarray(np.clip(big * gain, 0, 255).astype(np.uint8)).convert("RGB")


def fname(frame: int, h: str) -> str:
    return f"frame_with_red_{frame}_jpg.rf.{h}.jpg"


def make_export(root: Path, files: dict[str, list[tuple[str, int, float, str]]]) -> None:
    """files: split -> [(file_name, pattern_seed, gain, label_text)]."""
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for name, seed, gain, lab in files.get(split, []):
            pattern(seed, gain).save(root / split / "images" / name)
            (root / split / "labels" / (Path(name).stem + ".txt")).write_text(lab)


BOX = "3 0.5 0.5 0.2 0.2\n"


def test_exposure_copies_match_and_different_images_do_not():
    a, b = pattern(1).resize((180, 135)), pattern(1, 0.8).resize((180, 135))
    a.save("/tmp/_a.png"), b.save("/tmp/_b.png"), pattern(2).save("/tmp/_c.png")
    va, vb, vc = (thumbnail_vector(f"/tmp/_{k}.png") for k in "abc")
    assert cluster_by_content([va, vb, vc]) == [[0, 1], [2]]


def test_flat_image_rejected(tmp_path):
    Image.new("RGB", (72, 54), (40, 40, 40)).save(tmp_path / "flat.png")
    with pytest.raises(ValueError):
        thumbnail_vector(tmp_path / "flat.png")


def test_shared_stem_with_different_images_is_split(tmp_path):
    # Frame 10: three exposure copies of image A, plus a different image B with the same name stem.
    make_export(
        tmp_path,
        {
            "train": [
                (fname(10, "a1"), 1, 1.0, BOX),
                (fname(10, "a2"), 1, 0.9, BOX),
                (fname(10, "a3"), 1, 1.1, BOX),
                (fname(10, "b1"), 2, 1.0, ""),
            ],
            "test": [(fname(10, "c1"), 3, 1.0, BOX)],  # yet another image, same stem
        },
    )
    images = build_index(tmp_path)
    assert sorted(len(im.copies) for im in images) == [1, 1, 3]
    s = summarise(images)
    assert s["distinct_images"] == 3 and s["files"] == 5
    assert s["stems_shared_by_different_images"] == 1
    # Same name in train and test, but different pixels: not leakage.
    assert s["images_in_more_than_one_split"] == 0
    assert s["images_with_conflicting_labels"] == 0


def test_real_leak_and_conflict_detected(tmp_path):
    make_export(
        tmp_path,
        {
            "train": [(fname(5, "a1"), 7, 1.0, BOX), (fname(5, "a2"), 7, 0.9, "")],
            "valid": [(fname(5, "a3"), 7, 1.1, BOX)],
        },
    )
    s = summarise(build_index(tmp_path))
    assert s["distinct_images"] == 1
    assert s["images_in_more_than_one_split"] == 1
    assert s["images_with_conflicting_labels"] == 1


def test_remap_drops_and_renumbers():
    boxes = (Box(0, .1, .1, .1, .1), Box(4, .2, .2, .1, .1), Box(3, .3, .3, .1, .1))
    assert [b.cls for b in remap_boxes(boxes, keep=[0, 1, 2, 3])] == [0, 3]
    assert [b.cls for b in remap_boxes(boxes, keep=[3, 0])] == [1, 0]


def test_export_keeps_split_and_drops_wrinkle(tmp_path):
    raw = tmp_path / "raw"
    make_export(
        raw,
        {
            "train": [(fname(1, "a1"), 1, 1.0, "4 0.5 0.5 0.2 0.2\n3 0.1 0.1 0.1 0.1\n")],
            "valid": [(fname(2, "b1"), 2, 1.0, "1 0.5 0.5 0.1 0.1\n")],
            "test": [(fname(3, "c1"), 3, 1.0, "")],
        },
    )
    out = tmp_path / "out"
    counts = export_dataset(build_index(raw), out, keep_classes=[0, 1, 2, 3])
    assert counts == {"train": 1, "valid": 1, "test": 1}
    lab = read_label_file(out / "train" / "labels" / (Path(fname(1, "a1")).stem + ".txt"), n_classes=4)
    assert [b.cls for b in lab] == [3]
    y = (out / "data.yaml").read_text()
    assert "nc: 4" in y and "wrinkle" not in y and "val: valid/images" in y
    # Absolute path: Ultralytics resolves relative paths from the working directory.
    assert f"path: {out.resolve()}" in y
