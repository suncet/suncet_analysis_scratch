#!/usr/bin/env python3
"""Quicklook plot for compressed CSIE JPEG-LS streams."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from suncet_processing_pipeline.data_paths import data_path


def default_csie_dir() -> Path:
    return data_path(
        "test_data",
        "2026-05-20_xband_downlink_hk_sci_dsps_adcs",
        "csie_images",
    )


def image_id_from_path(path: Path) -> int:
    match = re.search(r"image_(\d+)", path.name)
    if not match:
        raise ValueError(f"Could not parse image id from {path.name}")
    return int(match.group(1))


def decode_with_imagecodecs(data: bytes):
    import imagecodecs

    return imagecodecs.jpegls_decode(data)


def try_decode_jpegls(path: Path) -> tuple[np.ndarray | None, str, str]:
    data = path.read_bytes()
    name = "imagecodecs.jpegls_decode"
    try:
        image = np.asarray(decode_with_imagecodecs(data))
    except Exception as exc:
        return None, "", f"{name}: {type(exc).__name__}: {exc}"
    if image.ndim != 2:
        return None, "", f"{name}: decoded array has unexpected shape {image.shape}"
    return image, name, ""


def plot_jpegls_quicklook(input_dir: Path, output_path: Path) -> None:
    paths = sorted(input_dir.glob("image_*.jls"), key=image_id_from_path)
    if not paths:
        raise FileNotFoundError(f"No image_*.jls files found in {input_dir}")

    results = []
    for path in paths:
        image, decoder_name, error = try_decode_jpegls(path)
        results.append((image_id_from_path(path), path, image, decoder_name, error))

    n = len(results)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 6.3 * rows))
    fig.subplots_adjust(wspace=0.24, hspace=0.24)
    axes = np.atleast_1d(axes).ravel()

    for ax, (image_id, path, image, decoder_name, error) in zip(axes, results):
        ax.set_title(str(image_id), fontsize=14, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])

        if image is None:
            ax.set_facecolor("black")
            ax.text(
                0.5,
                0.5,
                "decode failed",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="white",
                fontsize=12,
            )
            continue

        image = image.astype(np.float32, copy=False)
        finite = image[np.isfinite(image)]
        if finite.size:
            vmin, vmax = np.nanpercentile(finite, [1, 99.5])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin = float(np.nanmin(finite))
                vmax = float(np.nanmax(finite))
        else:
            vmin, vmax = 0.0, 1.0
        handle = ax.imshow(
            image,
            origin="lower",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        cbar = fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

    for ax in axes[len(results) :]:
        ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output_path}")
    for image_id, path, image, decoder_name, error in results:
        if image is None:
            print(f"{image_id}: decode failed: {error}")
        else:
            print(f"{image_id}: decoded with {decoder_name}; shape={image.shape}; file={path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_csie_dir(),
        help="Folder containing image_*.jls files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to <input-dir>/jpegls_decode_quicklook.png.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_dir / "jpegls_decode_quicklook.png"
    )
    plot_jpegls_quicklook(input_dir, output)


if __name__ == "__main__":
    main()
