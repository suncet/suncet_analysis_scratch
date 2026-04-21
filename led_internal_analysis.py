#!/usr/bin/env python3
"""Internal LED illumination: display level0_5 FITS and optional dust-structure checks."""

from __future__ import annotations

import glob
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def _data_dir() -> str:
    root = os.getenv("suncet_data")
    if not root:
        raise RuntimeError("Environment variable suncet_data is not set.")
    return os.path.join(
        root, "test_data", "2026-04-20_internal_led", "level0_5"
    )


def _intg_seconds(header: dict[str, Any]) -> float:
    ms = header.get("INTG_MS")
    if ms is None:
        raise KeyError("INTG_MS not found in FITS header")
    return float(ms) / 1000.0


def _load_fits(path: str) -> tuple[np.ndarray, dict[str, Any]]:
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        header = dict(hdul[0].header)
    return data, header


def _sorted_fits_paths(folder: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(folder, "*.fits")))
    if len(paths) < 2:
        raise RuntimeError(
            f"Expected at least 2 FITS files in {folder!r}, found {len(paths)}."
        )
    return paths[:2]


def _robust_vmin_vmax(img: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> tuple[float, float]:
    flat = img[np.isfinite(img)]
    if flat.size == 0:
        return 0.0, 1.0
    return float(np.percentile(flat, lo_pct)), float(np.percentile(flat, hi_pct))


def main() -> None:
    folder = _data_dir()
    paths = _sorted_fits_paths(folder)

    images: list[np.ndarray] = []
    headers: list[dict[str, Any]] = []
    times_s: list[float] = []
    for p in paths:
        data, hdr = _load_fits(p)
        images.append(data)
        headers.append(hdr)
        times_s.append(_intg_seconds(hdr))

    print("FITS integration times (INTG_MS → seconds):")
    for p, t in zip(paths, times_s):
        print(f"  {os.path.basename(p)}: {t:g} s")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, img, t, p in zip(axes, images, times_s, paths):
        vmin, vmax = _robust_vmin_vmax(img)
        im = ax.imshow(img, cmap="inferno", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"{os.path.basename(p)}\n{t:g} s")
        ax.set_xlabel("column")
        ax.set_ylabel("row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="DN")
    fig.suptitle("Internal LED — level0_5")
    plt.show()


if __name__ == "__main__":
    main()
