"""
Light-leak test level0_5 (2026-04-22):

1. All FITS frames, oriented, shared stretch + sliders, stats boxes.
2. One 0.5 s reference subtracted from each 30 s frame (raw DN), then oriented;
   four difference panels with the same presentation.
3. Same as (2) but the 1219 panel has uniform datasheet Δdark(T1219 vs T1218) removed
   (empirical scale from post-vibe dark set, same as led_external_analysis).

Orientation matches ``moving_cutout_circle_analysis``:
``flipud(rot90(img, k=1))`` with ``imshow(..., origin='lower')``.
"""

from __future__ import annotations

import glob
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.widgets import Slider

from csie_det0_thermal import det0_temperature_deg_c_from_header
from led_external_analysis import (
    DARK_REF_IMAGE_TOKEN,
    GAIN_DN_PER_ELECTRON,
    _datasheet_delta_mean_dark_e,
    _empirical_mean_vs_datasheet_scale,
    process_folder,
)

# Integration labels from INTG_MS (milliseconds).
_IMG_ID_TEMP_REF = "1218"
_IMG_ID_TEMP_CORR = "1219"
_INTG_MS_REF = 500
_INTG_MS_LONG = 30_000


def intg_seconds(header: fits.Header) -> float:
    if "INTG_MS" not in header:
        raise KeyError("INTG_MS not found in FITS header")
    return float(header["INTG_MS"]) / 1000.0


def intg_ms(header: fits.Header) -> float:
    return float(header["INTG_MS"])


def img_id_str(header: fits.Header) -> str:
    if "IMG_ID" not in header:
        raise KeyError("IMG_ID not found in FITS header")
    return str(header["IMG_ID"])


def orient_for_display(img: np.ndarray) -> np.ndarray:
    """90° clockwise + mirror so ``img[0,0]`` stays lower-left with ``origin='lower'``."""
    return np.flipud(np.rot90(img, k=1))


def _stats_box_text(img: np.ndarray) -> str:
    flat = img[np.isfinite(img)]
    if flat.size == 0:
        return "mean = —\nmedian = —\nstd = —"
    mean = float(np.mean(flat))
    median = float(np.median(flat))
    std = float(np.std(flat))
    return f"mean = {mean:.5g}\nmedian = {median:.5g}\nstd = {std:.5g}"


def _summarize_intg(records: list[dict[str, Any]]) -> dict[float, int]:
    out: dict[float, int] = {}
    for r in records:
        ms = r["intg_ms"]
        out[ms] = out.get(ms, 0) + 1
    return out


def _plot_gridded_inferno_with_sliders(
    images: list[np.ndarray],
    titles: list[str],
    suptitle: str,
    *,
    fig_width_per_col: float = 4.0,
    fig_height_per_row: float = 3.5,
    slider_height: float = 0.12,
    title_fs: int = 9,
) -> None:
    """Inferno, per-panel colorbars, stats box, shared vmin/vmax + sliders."""
    n = len(images)
    if n != len(titles):
        raise ValueError("images and titles length mismatch")
    flat = np.concatenate([img.ravel() for img in images])
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        raise SystemExit("All pixels non-finite for figure.")

    dmin = float(np.min(flat))
    dmax = float(np.max(flat))
    if dmax <= dmin:
        dmax = dmin + 1.0

    p_lo, p_hi = np.percentile(flat, [5.0, 99.5])
    vmin0, vmax0 = float(p_lo), float(p_hi)
    if vmax0 <= vmin0:
        vmax0 = vmin0 + 1.0

    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig_h = fig_height_per_row * nrows + 2 * slider_height + 1.0
    fig_w = fig_width_per_col * ncols + 1.0
    fig = plt.figure(figsize=(fig_w, fig_h), layout=None)
    fig.suptitle(suptitle)
    gs = fig.add_gridspec(
        nrows + 2,
        ncols,
        height_ratios=[1.0] * nrows + [slider_height, slider_height],
        hspace=0.32,
        wspace=0.30,
        left=0.07,
        right=0.97,
        top=0.90,
        bottom=0.09,
    )

    axes: list[plt.Axes] = []
    for idx in range(n):
        r = idx // ncols
        c = idx % ncols
        axes.append(fig.add_subplot(gs[r, c]))

    ims: list[plt.AxesImage] = []
    bbox_kw = dict(boxstyle="round", facecolor="white", alpha=0.88, edgecolor="0.4")
    for ax, img, title in zip(axes, images, titles):
        im = ax.imshow(
            img,
            origin="lower",
            cmap="inferno",
            vmin=vmin0,
            vmax=vmax0,
            interpolation="nearest",
        )
        ims.append(im)
        ax.set_title(title, fontsize=title_fs)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.text(
            0.03,
            0.97,
            _stats_box_text(img),
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            family="monospace",
            bbox=bbox_kw,
        )

    ax_vmin = fig.add_subplot(gs[nrows, :])
    ax_vmax = fig.add_subplot(gs[nrows + 1, :])

    slider_vmin = Slider(
        ax_vmin,
        "vmin",
        dmin,
        dmax,
        valinit=vmin0,
        valfmt="%.1f",
    )
    slider_vmax = Slider(
        ax_vmax,
        "vmax",
        dmin,
        dmax,
        valinit=vmax0,
        valfmt="%.1f",
    )

    def apply_clim() -> None:
        lo = float(slider_vmin.val)
        hi = float(slider_vmax.val)
        if hi <= lo:
            hi = lo + 1e-6
        for im in ims:
            im.set_clim(lo, hi)
        fig.canvas.draw_idle()

    def on_vmin(_: float) -> None:
        if slider_vmin.val >= slider_vmax.val:
            slider_vmax.set_val(min(dmax, slider_vmin.val + 1e-6))
        apply_clim()

    def on_vmax(_: float) -> None:
        if slider_vmax.val <= slider_vmin.val:
            slider_vmin.set_val(max(dmin, slider_vmax.val - 1e-6))
        apply_clim()

    slider_vmin.on_changed(on_vmin)
    slider_vmax.on_changed(on_vmax)


def main() -> None:
    root = os.getenv("suncet_data")
    if not root:
        raise SystemExit("Environment variable suncet_data is not set.")

    data_dir = os.path.join(
        root, "test_data", "2026-04-22_light_leak_test", "level0_5"
    )
    paths = sorted(glob.glob(os.path.join(data_dir, "*.fits")))
    if not paths:
        raise SystemExit(f"No FITS files found in {data_dir!r}.")

    records: list[dict[str, Any]] = []
    for p in paths:
        with fits.open(p) as hdul:
            raw = np.asarray(hdul[0].data, dtype=np.float64)
            hdr = hdul[0].header
        ms = intg_ms(hdr)
        records.append({"path": p, "raw": raw, "header": hdr, "intg_ms": ms})

    # --- Figure 1: every frame (oriented raw) ---
    rec_sorted = sorted(
        records, key=lambda r: (r["intg_ms"], img_id_str(r["header"]))
    )
    all_images = [orient_for_display(r["raw"]) for r in rec_sorted]
    all_titles: list[str] = []
    for r in rec_sorted:
        h = r["header"]
        t_s = intg_seconds(h)
        t_c = det0_temperature_deg_c_from_header(h)
        all_titles.append(
            f"{img_id_str(h)} | INTG = {t_s:g} s | {t_c:.2f} °C"
        )
    _plot_gridded_inferno_with_sliders(
        all_images,
        all_titles,
        "2026-04-22 light leak | all frames | level0_5 (shared vmin/vmax)",
    )

    # --- Figure 2: 30 s − 0.5 s differences (four panels) ---
    ref_recs = [r for r in records if r["intg_ms"] == _INTG_MS_REF]
    long_recs = [r for r in records if r["intg_ms"] == _INTG_MS_LONG]

    if not ref_recs:
        found = sorted({r["intg_ms"] for r in records})
        raise SystemExit(
            f"No FITS with INTG_MS={_INTG_MS_REF} (0.5 s). Found INTG_MS values: {found}"
        )
    if len(long_recs) != 4:
        raise SystemExit(
            f"Expected exactly 4 FITS with INTG_MS={_INTG_MS_LONG} (30 s); "
            f"found {len(long_recs)}. INTG_MS counts: {_summarize_intg(records)}"
        )

    ref = ref_recs[0]
    ref_raw = ref["raw"]
    t_ref_s = intg_seconds(ref["header"])

    long_recs = sorted(long_recs, key=lambda r: img_id_str(r["header"]))

    diff_images: list[np.ndarray] = []
    diffs_raw: list[np.ndarray] = []
    diff_titles: list[str] = []
    for r in long_recs:
        raw = r["raw"]
        if raw.shape != ref_raw.shape:
            raise SystemExit(
                f"Shape mismatch: {os.path.basename(r['path'])} {raw.shape} vs ref "
                f"{ref_raw.shape}"
            )
        diff_raw = raw - ref_raw
        diffs_raw.append(diff_raw)
        diff_images.append(orient_for_display(diff_raw))
        t_long_s = intg_seconds(r["header"])
        delta_s = t_long_s - t_ref_s
        t_c = det0_temperature_deg_c_from_header(r["header"])
        diff_titles.append(
            f"{img_id_str(r['header'])} | Δt = {delta_s:g} s | {t_c:.2f} °C"
        )

    _plot_gridded_inferno_with_sliders(
        diff_images,
        diff_titles,
        "2026-04-22 light leak | 30 s − 0.5 s (one reference) | level0_5",
        fig_width_per_col=5.5,
        fig_height_per_row=4.0,
    )

    # --- Figure 3: same four diffs; 1219 panel minus empirically scaled datasheet
    #     Δdark for T(1219) vs T(1218) over Δt (same pipeline as led_external_analysis).
    r_1218 = next(
        (r for r in long_recs if img_id_str(r["header"]) == _IMG_ID_TEMP_REF), None
    )
    r_1219 = next(
        (r for r in long_recs if img_id_str(r["header"]) == _IMG_ID_TEMP_CORR), None
    )
    if r_1218 is None or r_1219 is None:
        ids = [img_id_str(r["header"]) for r in long_recs]
        raise SystemExit(
            f"Figure 3 needs IMG_ID {_IMG_ID_TEMP_REF!r} and {_IMG_ID_TEMP_CORR!r} among "
            f"long exposures; found {ids}."
        )

    dark_folder = os.path.join(
        root, "test_data", "2026-04-20_three_dark_images", "level0_5"
    )
    if not os.path.isdir(dark_folder):
        raise SystemExit(
            f"Figure 3 requires post-vibe dark folder for empirical scale: {dark_folder!r}."
        )
    dark_pkg = process_folder(
        dark_folder, label="dark (post-vibe)", ref_token=DARK_REF_IMAGE_TOKEN
    )
    empirical_scale = _empirical_mean_vs_datasheet_scale(dark_pkg)

    delta_t_s = float(intg_seconds(r_1219["header"]) - t_ref_s)
    t_hi = float(det0_temperature_deg_c_from_header(r_1219["header"]))
    t_lo = float(det0_temperature_deg_c_from_header(r_1218["header"]))
    offset_e = float(
        empirical_scale * _datasheet_delta_mean_dark_e(delta_t_s, t_hi, t_lo)
    )
    offset_dn = offset_e * GAIN_DN_PER_ELECTRON

    diff_images_tcorr: list[np.ndarray] = []
    diff_titles_tcorr: list[str] = []
    for r, d_raw, title in zip(long_recs, diffs_raw, diff_titles):
        if img_id_str(r["header"]) == _IMG_ID_TEMP_CORR:
            d_use = d_raw - offset_dn
            diff_titles_tcorr.append(
                f"{title} | −Δdark(T{_IMG_ID_TEMP_CORR} vs T{_IMG_ID_TEMP_REF}) "
                f"= {offset_dn:.4g} DN (scale={empirical_scale:.3g})"
            )
        else:
            d_use = d_raw
            diff_titles_tcorr.append(title)
        diff_images_tcorr.append(orient_for_display(d_use))

    _plot_gridded_inferno_with_sliders(
        diff_images_tcorr,
        diff_titles_tcorr,
        "2026-04-22 light leak | 30 s − 0.5 s | 1219: empirically scaled datasheet "
        f"Δdark vs {_IMG_ID_TEMP_REF} (obs/pred scale from post-vibe dark)",
        fig_width_per_col=5.5,
        fig_height_per_row=4.0,
    )

    plt.show()


if __name__ == "__main__":
    main()
