"""
Load CSIE test-pattern FITS from level0_5 and show all frames in one figure.

Display orientation matches ``moving_cutout_circle_analysis.py``:
``flipud(rot90(img, k=1))`` with ``imshow(..., origin='lower')`` so the FITS
array origin reads as lower-left on screen.
"""

from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def load_fits_image(path: str) -> np.ndarray:
    with fits.open(path) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float64)


def orient_for_display(img: np.ndarray) -> np.ndarray:
    """90 deg clockwise view + mirror so img[0,0] stays lower-left with origin='lower'."""
    return np.flipud(np.rot90(img, k=1))


def main() -> None:
    root = os.getenv("suncet_data")
    if not root:
        raise SystemExit("Environment variable suncet_data is not set.")

    data_dir = os.path.join(
        root, "test_data", "2026-04-22_test_patterns", "level0_5"
    )
    paths = sorted(glob.glob(os.path.join(data_dir, "*.fits")))
    if not paths:
        raise SystemExit(f"No FITS files found in {data_dir!r}.")

    images = [orient_for_display(load_fits_image(p)) for p in paths]
    image_ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]

    n = len(paths)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig_w = min(4.0 * ncols, 22.0)
    fig_h = min(3.5 * nrows, 18.0)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        layout="constrained",
        squeeze=False,
    )
    fig.suptitle("2026-04-22 CSIE test patterns - level0_5 (5-99.5 pct per frame)")

    axes_flat = np.atleast_1d(axes).ravel()
    for ax, img, image_id in zip(axes_flat, images, image_ids):
        vmin_i, vmax_i = np.percentile(img, [5.0, 99.5])
        vmin_i, vmax_i = float(vmin_i), float(vmax_i)
        if vmax_i <= vmin_i:
            vmax_i = vmin_i + 1.0

        im = ax.imshow(
            img,
            origin="lower",
            cmap="inferno",
            vmin=vmin_i,
            vmax=vmax_i,
            interpolation="nearest",
        )
        ax.set_title(image_id, fontsize=9)
        ax.set_xlabel("pixel x")
        ax.set_ylabel("pixel y")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.006)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.show()


if __name__ == "__main__":
    main()
