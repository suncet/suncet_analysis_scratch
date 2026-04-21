"""
Load moving-cutout circle test FITS from level0_5: four panels in one figure —
top row: both frames with default robust scaling (5th–99.5th percentile of both);
bottom row: same frames with vmin=4734, vmax=5009.

imshow ``origin`` vs numpy ``rot90``
------------------------------------
- ``origin`` does **not** move pixels in the array. It only chooses which **corner
  of the axes** gets ``img[0, 0]``, and which way row index increases on screen.
- ``origin='upper'`` (matplotlib default for imshow): row 0 at the **top** →
  ``img[0, 0]`` is the **upper-left** of the image; increasing row index goes **down**.
- ``origin='lower'``: row 0 at the **bottom** → ``img[0, 0]`` is the **lower-left**;
  increasing row index goes **up** (usual math / “x right, y up” feel).

``np.rot90`` **does** reorder pixels: after rotation, ``img[0, 0]`` is a **different**
detector pixel than before. Combining rot90 with ``origin`` is fine mathematically,
but you must think “rotate first, then place (0,0) at a corner” — easy to confuse.

Display orientation
--------------------
``imshow(..., origin='lower')`` keeps **array** ``(0, 0)`` at the **lower-left** of
the axes. To rotate the picture **90° clockwise** while **keeping that same FITS
pixel** at the lower-left corner, use ``flipud(rot90(img, k=1))``. That composition
is identical to ``fliplr(rot90(img, k=-1))`` (clockwise rotation then a mirror that
re-anchors the corner); a naive ``flipud(rot90(img, k=-1))`` would move ``(0,0)``
away from the lower-left.
"""

from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

# Fixed stretch for bottom row (DN).
FIXED_VMIN = 4734
FIXED_VMAX = 5009


def load_fits_image(path: str) -> np.ndarray:
    with fits.open(path) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float64)


def orient_for_display(img: np.ndarray) -> np.ndarray:
    """90° clockwise view + mirror so ``img[0,0]`` stays at lower-left with ``origin='lower'``."""
    return np.flipud(np.rot90(img, k=1))


def main() -> None:
    root = os.getenv("suncet_data")
    if not root:
        raise SystemExit("Environment variable suncet_data is not set.")

    data_dir = os.path.join(
        root, "test_data", "2026-04-21_moving_circles", "level0_5"
    )
    paths = sorted(glob.glob(os.path.join(data_dir, "*.fits")))
    if len(paths) != 2:
        raise SystemExit(
            f"Expected exactly 2 FITS files in {data_dir!r}, found {len(paths)}."
        )

    images = [orient_for_display(load_fits_image(p)) for p in paths]
    titles = [os.path.basename(p) for p in paths]

    flat = np.concatenate([img.ravel() for img in images])
    vmin0, vmax0 = np.percentile(flat, [5.0, 99.5])
    vmin0, vmax0 = float(vmin0), float(vmax0)
    if vmax0 <= vmin0:
        vmax0 = vmin0 + 1.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), layout="constrained")
    fig.suptitle("2026-04-21 moving circles (cutout) — level0_5")

    for col, (img, title) in enumerate(zip(images, titles)):
        ax_def = axes[0, col]
        im0 = ax_def.imshow(
            img,
            origin="lower",
            cmap="inferno",
            vmin=vmin0,
            vmax=vmax0,
            interpolation="nearest",
        )
        ax_def.set_title(f"{title}\n(5–99.5 pct of both)", fontsize=9)
        ax_def.set_xlabel("pixel x (column index)")
        ax_def.set_ylabel("pixel y (row index)")
        fig.colorbar(im0, ax=ax_def, fraction=0.046, pad=0.02)

        ax_fix = axes[1, col]
        im1 = ax_fix.imshow(
            img,
            origin="lower",
            cmap="inferno",
            vmin=FIXED_VMIN,
            vmax=FIXED_VMAX,
            interpolation="nearest",
        )
        ax_fix.set_title(
            f"{title}\n(vmin={FIXED_VMIN}, vmax={FIXED_VMAX})", fontsize=9
        )
        ax_fix.set_xlabel("pixel x (column index)")
        ax_fix.set_ylabel("pixel y (row index)")
        fig.colorbar(im1, ax=ax_fix, fraction=0.046, pad=0.02)

    plt.show()


if __name__ == "__main__":
    main()
