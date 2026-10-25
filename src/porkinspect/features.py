"""Hand-crafted image features for the baseline classifier.

The baseline answers the same frame-level question as the detector ("does
this frame show a defect?") but without deep learning. Each image is reduced
to a fixed-length vector of numbers that a unit-taught model (decision tree,
SVM, MLP) can learn from.

Two feature sets:

* :func:`colour_shape_features`: 20 interpretable numbers. They describe how
  much meat is visible and where, its colour, the belt brightness, specular
  highlights (bright glints, which wrinkled film tends to produce) and edge
  density around the meat.
* :func:`hog_features`: a Histogram of Oriented Gradients on a 160 x 120
  greyscale thumbnail. This classic texture/shape descriptor summarises local
  edge directions in a grid of cells.

All features are computed on images resized to a fixed size, so the inputs
may be any resolution. Values are unscaled; standardise them before an SVM
or MLP.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

#: Working size (width, height) for the colour/shape features.
WORK_SIZE = (360, 270)

COLOUR_SHAPE_NAMES: tuple[str, ...] = (
    "meat_fraction", "meat_cx", "meat_cy", "meat_x_min", "meat_x_max", "meat_y_min", "meat_y_max",
    "meat_hue_mean", "meat_sat_mean", "meat_val_mean", "meat_sat_std", "meat_val_std",
    "belt_grey_mean", "belt_grey_std",
    "highlight_fraction", "highlight_near_meat", "highlight_on_meat",
    "edge_density", "edge_density_near_meat", "edge_density_on_meat",
)


def load_rgb(path: str | Path, size: tuple[int, int] = WORK_SIZE) -> np.ndarray:
    """Load an image as a float RGB array in [0, 1], resized to ``size``.

    Args:
        path: Image file.
        size: Output (width, height) in pixels.

    Returns:
        Array of shape (height, width, 3).

    Raises:
        OSError: If the file is not a readable image.
    """
    return np.asarray(Image.open(path).convert("RGB").resize(size), dtype=float) / 255.0


def meat_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels that look like raw pork: clearly red-dominant and not dark.

    The rule (R > 0.43, R - G > 0.14, R - B > 0.10 on a 0-1 scale) was chosen
    by inspecting images. It separates pink/red meat from the dark grey belt
    and the clear tray. It is a heuristic, not a calibrated segmentation.

    Args:
        rgb: Float RGB array in [0, 1], shape (height, width, 3).

    Returns:
        Boolean array of shape (height, width), True on meat pixels.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (r > 0.43) & (r - g > 0.14) & (r - b > 0.10)


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Binary dilation by a (2r+1) square, using shifted copies (no SciPy needed).

    Args:
        mask: Boolean image.
        r: Dilation radius in pixels.

    Returns:
        Boolean image of the same shape, True within ``r`` pixels of ``mask``.
    """
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ys, yd = slice(max(0, dy), h + min(0, dy)), slice(max(0, -dy), h + min(0, -dy))
            xs, xd = slice(max(0, dx), w + min(0, dx)), slice(max(0, -dx), w + min(0, -dx))
            out[yd, xd] |= mask[ys, xs]
    return out


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB -> HSV conversion.

    Args:
        rgb: Float RGB array in [0, 1], shape (..., 3).

    Returns:
        Array of the same shape holding (hue, saturation, value), all in [0, 1].
    """
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    nz = d > 0
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = (b - r)[gm] / d[gm] + 2
    h[bm] = (r - g)[bm] / d[bm] + 4
    s = np.where(mx > 0, d / np.where(mx > 0, mx, 1), 0)
    return np.stack([h / 6, s, mx], -1)


def colour_shape_features(path: str | Path) -> np.ndarray:
    """Compute the 20 colour/shape features of one image.

    Positions are fractions of image width/height; colour values are in
    [0, 1]. If no meat is visible, the meat statistics are 0 and the position
    is (0.5, 0.5); this case does not occur in this dataset. Hue is shifted by
    half a turn so that red, which wraps around 0/1, averages correctly.

    Args:
        path: Image file, any resolution (resized to :data:`WORK_SIZE`).

    Returns:
        A float array of length 20, ordered as :data:`COLOUR_SHAPE_NAMES`.

    Raises:
        OSError: If the file is not a readable image.
    """
    rgb = load_rgb(path)
    h, w = rgb.shape[:2]
    grey = rgb.mean(-1)
    meat = meat_mask(rgb)
    near = _dilate(meat, 8) & ~meat  # a band about 8 px around the meat: tray edge and film
    belt = ~_dilate(meat, 20)

    ys, xs = np.nonzero(meat)
    if len(xs):
        pos = [xs.mean() / w, ys.mean() / h, xs.min() / w, xs.max() / w, ys.min() / h, ys.max() / h]
        hsv = _rgb_to_hsv(rgb)[meat]
        # Red hue wraps around 0/1, so shift it by half a turn before averaging (red -> about 0.5).
        hue = (hsv[:, 0] + 0.5) % 1.0
        col = [hue.mean(), hsv[:, 1].mean(), hsv[:, 2].mean(), hsv[:, 1].std(), hsv[:, 2].std()]
    else:
        pos = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        col = [0.0] * 5

    # Specular highlights: near-white, low-saturation pixels, typical of light glinting off film.
    highlight = (grey > 0.75) & (rgb.max(-1) - rgb.min(-1) < 0.15)
    gy, gx = np.gradient(grey)
    edges = np.hypot(gx, gy) > 0.08

    def frac(m, region):
        """Share of ``region`` pixels that are also set in mask ``m``.

        Args:
            m: Boolean mask.
            region: Boolean mask of the same shape.

        Returns:
            Fraction in [0, 1]; 0 if the region is empty.
        """
        n = region.sum()
        return float((m & region).sum() / n) if n else 0.0

    feats = [meat.mean(), *pos, *col,
             grey[belt].mean() if belt.any() else 0.0, grey[belt].std() if belt.any() else 0.0,
             highlight.mean(), frac(highlight, near), frac(highlight, meat),
             edges.mean(), frac(edges, near), frac(edges, meat)]
    return np.asarray(feats, dtype=float)


def hog_features(path: str | Path, size: tuple[int, int] = (160, 120), cell: int = 16) -> np.ndarray:
    """Histogram of Oriented Gradients of a greyscale thumbnail.

    With the defaults (160 x 120 px, 16 px cells, 2 x 2 cells per block,
    9 orientations), this gives 6 x 9 blocks x 36 = 1,944 numbers.

    Args:
        path: Image file, any resolution.
        size: Thumbnail (width, height) in pixels.
        cell: Cell side length in pixels.

    Returns:
        1-D float array of HOG features (length 1,944 with the defaults).

    Raises:
        OSError: If the file is not a readable image.
    """
    from skimage.feature import hog

    grey = np.asarray(Image.open(path).convert("L").resize(size), dtype=float) / 255.0
    return hog(grey, orientations=9, pixels_per_cell=(cell, cell), cells_per_block=(2, 2),
               block_norm="L2-Hys", feature_vector=True)
