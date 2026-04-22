"""
Load default-loop test FITS from level0_5 and show all frames with interactive
vmin/vmax sliders.

Display orientation matches moving_cutout_circle_analysis.py:
flipud(rot90(img, k=1)) with imshow(..., origin="lower").
"""

from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.widgets import Slider


def load_fits_image(path: str) -> np.ndarray:
    with fits.open(path) as hdul:
        return np.asarray(hdul[0].data, dtype=np.float64)


def orient_for_display(img: np.ndarray) -> np.ndarray:
    """Match prior orientation convention for image display."""
    return np.flipud(np.rot90(img, k=1))


def main() -> None:
    root = os.getenv("suncet_data")
    if not root:
        raise SystemExit("Environment variable suncet_data is not set.")

    data_dir = os.path.join(
        root, "test_data", "2026-04-21_default_loop_10", "level0_5"
    )
    paths = sorted(glob.glob(os.path.join(data_dir, "*.fits")))
    if not paths:
        raise SystemExit(f"No FITS files found in {data_dir!r}.")

    images = [orient_for_display(load_fits_image(p)) for p in paths]
    titles = [os.path.splitext(os.path.basename(p))[0] for p in paths]

    n = len(images)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig_w = min(4.0 * ncols, 22.0)
    fig_h = min(3.5 * nrows + 1.4, 19.0)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )
    fig.subplots_adjust(bottom=0.14, wspace=0.28, hspace=0.32)
    fig.suptitle("2026-04-21 default_loop_10 — level0_5")

    flat = np.concatenate([img.ravel() for img in images])
    data_min = float(np.nanmin(flat))
    data_max = float(np.nanmax(flat))
    vmin0, vmax0 = np.nanpercentile(flat, [5.0, 99.5])
    vmin0 = float(vmin0)
    vmax0 = float(vmax0)
    if vmax0 <= vmin0:
        vmax0 = vmin0 + 1.0

    axes_flat = axes.ravel()
    imshows = []
    for ax, img, title in zip(axes_flat, images, titles):
        im = ax.imshow(
            img,
            origin="lower",
            cmap="inferno",
            interpolation="nearest",
            vmin=vmin0,
            vmax=vmax0,
        )
        imshows.append(im)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("pixel x")
        ax.set_ylabel("pixel y")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # Single shared colorbar for all panels.
    cbar = fig.colorbar(imshows[0], ax=axes_flat[:n], fraction=0.018, pad=0.01)
    cbar.set_label("DN")

    ax_vmin = fig.add_axes([0.15, 0.06, 0.7, 0.02])
    ax_vmax = fig.add_axes([0.15, 0.025, 0.7, 0.02])
    s_vmin = Slider(ax_vmin, "vmin", data_min, data_max, valinit=vmin0)
    s_vmax = Slider(ax_vmax, "vmax", data_min, data_max, valinit=vmax0)

    def update(_: float) -> None:
        vmin = float(s_vmin.val)
        vmax = float(s_vmax.val)
        if vmax <= vmin:
            return
        for im in imshows:
            im.set_clim(vmin=vmin, vmax=vmax)
        cbar.update_normal(imshows[0])
        fig.canvas.draw_idle()

    s_vmin.on_changed(update)
    s_vmax.on_changed(update)

    plt.show()


if __name__ == "__main__":
    main()
