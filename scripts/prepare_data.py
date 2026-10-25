"""Audit the Roboflow export and write the training-ready dataset.

Usage (from the repo root)::

    python scripts/prepare_data.py --raw data/raw --out data/processed

``--raw`` may be the unzipped export folder or the ``.zip`` itself.

Writes:
    results/data_audit.json        counts of distinct images, copies, leakage and label conflicts
    results/brightness_over_time.png  belt brightness per image against frame number
    <out>/...                      dataset with dropped classes removed (official split)
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from porkinspect.export import export_dataset  # noqa: E402
from porkinspect.index import SourceImage, build_index, summarise  # noqa: E402
from porkinspect.labels import CLASS_NAMES, frame_number  # noqa: E402


def brightness_check(images: list[SourceImage], out_png: Path, block: int = 250) -> dict:
    """Test whether background brightness drifts over the recording.

    Brightness is the mean grey level (0-255) of the top 200 rows, which show
    only the belt. The median over a source image's copies is used, which
    cancels most of the random exposure augmentation.

    Args:
        images: Distinct source images from ``build_index``.
        out_png: Where to save the brightness-over-time plot.
        block: Number of consecutive frames per time block.

    Returns:
        Summary statistics: percentiles, block-mean range, share of variance
        explained by ``block``-frame time blocks, and lag-1 autocorrelation.
        Little variance explained and near-zero autocorrelation mean the
        variation is frame-to-frame noise, not drift over time.
    """
    rows = []
    for im in images:
        v = [np.asarray(Image.open(c.image).convert("L"), float)[:200].mean() for c in im.copies]
        rows.append((frame_number(im.copies[0].image.name), float(np.median(v))))
    a = np.array(sorted(rows))
    b = (a[:, 0] // block).astype(int)
    ks = np.unique(b)
    means = np.array([a[b == k, 1].mean() for k in ks])
    within = sum(a[b == k, 1].var() * (b == k).sum() for k in ks) / len(a)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.scatter(a[:, 0], a[:, 1], s=5, alpha=0.5, label="image")
    ax.step(ks * block + block / 2, means, where="mid", color="C3", lw=2, label=f"{block}-frame block mean")
    ax.set_xlabel("frame number")
    ax.set_ylabel("belt brightness (0-255)")
    ax.set_title("Background brightness over the recording")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return {
        "percentiles_5_50_95": np.percentile(a[:, 1], [5, 50, 95]).round(1).tolist(),
        "block_mean_min_max": [round(float(means.min()), 1), round(float(means.max()), 1)],
        "variance_explained_by_block": round(1 - within / a[:, 1].var(), 3),
        "lag1_autocorrelation": round(float(np.corrcoef(a[:-1, 1], a[1:, 1])[0, 1]), 3),
    }


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, run the audit and write the dataset."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--drop", nargs="*", default=["wrinkle"], help="class names to drop")
    a = p.parse_args(argv)

    raw = a.raw
    if raw.suffix == ".zip":
        dest = raw.with_suffix("")
        if not dest.exists():
            with zipfile.ZipFile(raw) as z:
                z.extractall(dest)
        raw = dest

    images = build_index(raw)
    audit = summarise(images)
    (ROOT / "results").mkdir(exist_ok=True)
    audit["brightness"] = brightness_check(images, ROOT / "results" / "brightness_over_time.png")

    keep = [i for i, n in enumerate(CLASS_NAMES) if n not in a.drop]
    audit["kept_classes"] = [CLASS_NAMES[k] for k in keep]
    audit["files_written"] = export_dataset(images, a.out, keep)

    (ROOT / "results" / "data_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
