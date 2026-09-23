#!/usr/bin/env python3
"""Interactively compare a SunCET Level 1 image with its Level 2 product.

Both panels share the same physical display limits and intensity stretch.  The
default inputs are the provisional frame-300 products generated on 2026-09-23,
but any same-shaped Level 1/Level 2 FITS pair can be supplied on the command
line.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib import colors
from matplotlib.widgets import RadioButtons, Slider


DEFAULT_LEVEL1_RELATIVE = Path(
    "synthetic/level1/frame300_config_default_v1.0.4dev_20260923/"
    "config_default_OBS_2023-01-14T17:00:00.000_300_"
    "level1_v2.0.0_provisional.fits"
)
DEFAULT_LEVEL2_RELATIVE = Path(
    "synthetic/level2/frame300_config_default_v1.0.4dev_20260923/"
    "config_default_OBS_2023-01-14T17:00:00.000_300_"
    "level1_v2.0.0_provisional_level2_v2.0.0.fits"
)
SCALE_LABELS = ("Linear", "Asinh", "Log10", "Square root", "Fourth root")


class AsinhNormalize(colors.Normalize):
    """Normalize with a vmin-relative asinh stretch and an analytic inverse."""

    def __init__(
        self,
        *,
        vmin: float,
        vmax: float,
        linear_fraction: float = 0.03,
        clip: bool = True,
    ) -> None:
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        if not 0 < linear_fraction < 1:
            raise ValueError("linear_fraction must be between zero and one")
        self.linear_fraction = float(linear_fraction)
        self._normalizer = float(np.arcsinh(1.0 / self.linear_fraction))

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)
        if self.vmax <= self.vmin:
            raise ValueError("vmax must be greater than vmin")
        use_clip = self.clip if clip is None else clip
        scaled = (result - self.vmin) / (self.vmax - self.vmin)
        if use_clip:
            scaled = np.ma.clip(scaled, 0.0, 1.0)
        transformed = np.ma.arcsinh(scaled / self.linear_fraction)
        transformed /= self._normalizer
        return transformed[0] if is_scalar else transformed

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until vmin and vmax are set")
        value = np.asarray(value)
        scaled = self.linear_fraction * np.sinh(value * self._normalizer)
        return self.vmin + scaled * (self.vmax - self.vmin)


def default_paths() -> tuple[Path, Path]:
    data_root = os.environ.get("suncet_data")
    if not data_root:
        raise SystemExit(
            "Set suncet_data or supply both --level1 and --level2 explicitly."
        )
    root = Path(data_root).expanduser()
    return root / DEFAULT_LEVEL1_RELATIVE, root / DEFAULT_LEVEL2_RELATIVE


def load_primary_image(path: Path) -> tuple[np.ndarray, fits.Header]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with fits.open(path, checksum=True) as hdul:
        hdul.verify("exception")
        hdu = hdul[0]
        if "CHECKSUM" in hdu.header and hdu.verify_checksum() != 1:
            raise ValueError(f"FITS CHECKSUM failed for {path}")
        if "DATASUM" in hdu.header and hdu.verify_datasum() != 1:
            raise ValueError(f"FITS DATASUM failed for {path}")
        data = np.asarray(hdu.data, dtype=np.float64)
        header = hdu.header.copy()
    if data.ndim != 2:
        raise ValueError(f"Expected a 2-D primary image in {path}; got {data.shape}")
    if not np.any(np.isfinite(data)):
        raise ValueError(f"No finite pixels in {path}")
    return data, header


def scale_norm(
    label: str,
    *,
    vmin: float,
    vmax: float,
) -> colors.Normalize:
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("Finite display limits must satisfy vmin < vmax")
    if label == "Linear":
        return colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    if label == "Asinh":
        return AsinhNormalize(vmin=vmin, vmax=vmax, clip=True)
    if label == "Log10":
        if vmin <= 0:
            raise ValueError("Log10 scaling requires vmin > 0")
        return colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
    if label == "Square root":
        return colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax, clip=True)
    if label == "Fourth root":
        return colors.PowerNorm(gamma=0.25, vmin=vmin, vmax=vmax, clip=True)
    raise ValueError(f"Unknown scale: {label}")


def compare_images(
    level1: np.ndarray,
    level2: np.ndarray,
    *,
    level1_title: str = "Level 1 — exposure-normalized",
    level2_title: str = "Level 2 — PSF deconvolved",
    units: str = "DN/s",
) -> None:
    if level1.shape != level2.shape:
        raise ValueError(
            f"Level 1 and Level 2 shapes differ: {level1.shape} vs {level2.shape}"
        )

    finite = np.concatenate((level1[np.isfinite(level1)], level2[np.isfinite(level2)]))
    data_min = float(np.min(finite))
    data_max = float(np.max(finite))
    positive = finite[finite > 0]
    if positive.size == 0:
        raise ValueError("Log10 scaling is unavailable because no pixels are positive")
    smallest_positive = float(np.min(positive))
    vmin0, vmax0 = (float(value) for value in np.percentile(finite, [1.0, 99.5]))
    if vmax0 <= vmin0:
        vmax0 = np.nextafter(vmin0, np.inf)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.6), sharex=True, sharey=True)
    fig.canvas.manager.set_window_title("SunCET Level 1 / Level 2 comparison")
    fig.subplots_adjust(left=0.055, right=0.91, top=0.91, bottom=0.22, wspace=0.08)
    fig.suptitle("SunCET frame 300 — synchronized Level 1 / Level 2 viewer")

    initial_norm = scale_norm("Asinh", vmin=vmin0, vmax=vmax0)
    images = []
    for ax, data, title in zip(axes, (level1, level2), (level1_title, level2_title)):
        image = ax.imshow(
            data,
            origin="lower",
            cmap="inferno",
            norm=initial_norm,
            interpolation="nearest",
            aspect="equal",
        )
        images.append(image)
        ax.set_title(title)
        ax.set_xlabel("Detector column [pixel]")
    axes[0].set_ylabel("Detector row [pixel]")

    colorbar = fig.colorbar(images[0], ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label(units)

    slider_left = 0.16
    slider_width = 0.57
    vmin_axis = fig.add_axes([slider_left, 0.115, slider_width, 0.028])
    vmax_axis = fig.add_axes([slider_left, 0.065, slider_width, 0.028])
    vmin_slider = Slider(
        vmin_axis,
        f"vmin [{units}]",
        data_min,
        data_max,
        valinit=vmin0,
        valfmt="%0.5g",
    )
    vmax_slider = Slider(
        vmax_axis,
        f"vmax [{units}]",
        data_min,
        data_max,
        valinit=vmax0,
        valfmt="%0.5g",
        slidermin=vmin_slider,
    )
    vmin_slider.slidermax = vmax_slider

    radio_axis = fig.add_axes([0.765, 0.025, 0.13, 0.16])
    scale_buttons = RadioButtons(radio_axis, SCALE_LABELS, active=1)
    radio_axis.set_title("Intensity scale", loc="left")
    status = fig.text(0.16, 0.025, "", ha="left", va="center")
    state = {"updating": False}

    def update(_=None) -> None:
        if state["updating"]:
            return
        label = scale_buttons.value_selected
        vmin = float(vmin_slider.val)
        vmax = float(vmax_slider.val)
        message = ""
        if label == "Log10" and vmin <= 0:
            state["updating"] = True
            vmin_slider.set_val(smallest_positive)
            state["updating"] = False
            vmin = smallest_positive
            message = f"Log10 requires positive limits; vmin set to {vmin:.5g} {units}."
        if vmax <= vmin:
            return
        norm = scale_norm(label, vmin=vmin, vmax=vmax)
        for image in images:
            image.set_norm(norm)
        colorbar.update_normal(images[0])
        status.set_text(message)
        fig.canvas.draw_idle()

    vmin_slider.on_changed(update)
    vmax_slider.on_changed(update)
    scale_buttons.on_clicked(update)
    plt.show()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level1", type=Path, help="Level 1 FITS image")
    parser.add_argument("--level2", type=Path, help="Level 2 FITS image")
    args = parser.parse_args()
    if (args.level1 is None) != (args.level2 is None):
        parser.error("supply both --level1 and --level2, or neither")
    return args


def main() -> None:
    args = parse_arguments()
    level1_path, level2_path = (
        (args.level1, args.level2)
        if args.level1 is not None
        else default_paths()
    )
    level1, level1_header = load_primary_image(level1_path)
    level2, level2_header = load_primary_image(level2_path)
    level1_units = str(level1_header.get("BUNIT", "DN/s"))
    level2_units = str(level2_header.get("BUNIT", "DN/s"))
    if level1_units.casefold() != level2_units.casefold():
        raise ValueError(
            f"BUNIT mismatch: Level 1={level1_units!r}, Level 2={level2_units!r}"
        )
    compare_images(level1, level2, units=level1_units)


if __name__ == "__main__":
    main()
