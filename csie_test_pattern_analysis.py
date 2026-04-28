"""
Load CSIE test-pattern FITS from level0_5 and show all frames in one figure.

Display orientation matches ``moving_cutout_circle_analysis.py``:
``flipud(rot90(img, k=1))`` with ``imshow(..., origin='lower')`` so the FITS
array origin reads as lower-left on screen.

Per-frame figures compare each capture to the reference test pattern (image_id
1212 → 32, 1213 → 96): measured, reference, and difference.

Basename may be ``1212`` / ``1213`` or ``image_1212-...`` / ``image_1213-...``
(as in ``image_1212-hardline_playback_test.fits``).
"""

from __future__ import annotations

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from load_reference_test_patterns import (
    load_csie_fits_as_uint16,
    load_reference_test_pattern,
)

REF_BY_IMAGE_ID: dict[str, tuple[str, int]] = {
    "1212": ("reference_test_pattern_32.fits", 32),
    "1213": ("reference_test_pattern_96.fits", 96),
}


def ref_key_from_stem(stem: str) -> str | None:
    """Return reference dict key ``'1212'`` or ``'1213'``, or None if unknown."""
    if stem in REF_BY_IMAGE_ID:
        return stem
    m = re.match(r"^image_(1212|1213)(?:[^0-9]|$)", stem)
    if m:
        return m.group(1)
    return None


def load_fits_image(path: str) -> np.ndarray:
    """Level0_5 CSIE frame as ``uint16`` (same FITS handling as EM reference script)."""
    return load_csie_fits_as_uint16(path)


def orient_for_display(img: np.ndarray) -> np.ndarray:
    """90 deg clockwise view + mirror so img[0,0] stays lower-left with origin='lower'."""
    return np.flipud(np.rot90(img, k=1))


def _imshow_percentile(
    ax,
    img: np.ndarray,
    title: str,
    cmap: str,
    pct: tuple[float, float] = (5.0, 99.5),
    vmin: float | None = None,
    vmax: float | None = None,
    diverging: bool = False,
) -> None:
    if vmin is None or vmax is None:
        lo, hi = np.percentile(img, list(pct))
        lo, hi = float(lo), float(hi)
        if diverging:
            lim = float(np.nanpercentile(np.abs(img), pct[1]))
            if lim <= 0:
                lim = 1.0
            lo, hi = -lim, lim
        if hi <= lo:
            hi = lo + 1.0
        vmin, vmax = lo, hi
    im = ax.imshow(
        img,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


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

    allowed = ", ".join(sorted(REF_BY_IMAGE_ID))
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        if ref_key_from_stem(stem) is None:
            raise SystemExit(
                f"Unexpected image_id {stem!r} in {p!r}. "
                f"Expected stem {allowed!r} or names like "
                f"'image_1212-...' / 'image_1213-...'."
            )

    ref_raw: dict[str, np.ndarray] = {
        stem: load_reference_test_pattern(REF_BY_IMAGE_ID[stem][1])
        for stem in REF_BY_IMAGE_ID
    }

    images = [
        orient_for_display(load_fits_image(p).astype(np.float64)) for p in paths
    ]
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
        _imshow_percentile(
            ax, img, image_id, cmap="inferno", pct=(5.0, 99.5)
        )

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.show()

    for path, image_id in zip(paths, image_ids):
        ref_key = ref_key_from_stem(image_id)
        assert ref_key is not None
        ref_name, _binning = REF_BY_IMAGE_ID[ref_key]
        ref = ref_raw[ref_key]
        meas_u16 = load_fits_image(path)
        ref_u16 = ref
        meas = orient_for_display(meas_u16.astype(np.float64))
        ref_disp = orient_for_display(ref_u16.astype(np.float64))
        diff = meas - ref_disp

        fig_r, ax_r = plt.subplots(
            1, 3, figsize=(15.0, 4.2), layout="constrained", squeeze=False
        )
        fig_r.suptitle(
            f"{image_id}: measured vs {ref_name} (5–99.5 pct; diff symmetric 99.5 |·|)"
        )
        ax0, ax1, ax2 = ax_r[0]

        _imshow_percentile(
            ax0,
            meas,
            f"{image_id} (measured)",
            cmap="inferno",
            pct=(5.0, 99.5),
        )
        _imshow_percentile(
            ax1,
            ref_disp,
            f"{ref_name} (reference)",
            cmap="inferno",
            pct=(5.0, 99.5),
        )
        _imshow_percentile(
            ax2,
            diff,
            "difference (measured − reference)",
            cmap="RdBu_r",
            pct=(5.0, 99.5),
            diverging=True,
        )
        n_nz = int(np.count_nonzero(diff))
        n_tot = int(diff.size)
        ax2.annotate(
            f"{n_nz:,} of {n_tot:,} pixels\n≠ 0",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                alpha=0.88,
                edgecolor="0.35",
            ),
        )
        plt.show()


if __name__ == "__main__":
    main()
