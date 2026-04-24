#!/usr/bin/env python3
"""LED external stim lamp vs post-vibration dark (same subtraction pipeline as post_vibe_dark_analysis)."""

from __future__ import annotations

import glob
import os
import re
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from astropy.io import fits

from csie_det0_thermal import det0_temperature_deg_c_from_header

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None

GAIN_DN_PER_ELECTRON = 1.8

# `_find_ref_path` picks the fixed-pattern frame; other *.fits in the folder are the integrations.
DARK_REF_IMAGE_TOKEN = "1185"  # post-vibe dark set
LED_REF_IMAGE_TOKEN = "1194"  # external stim lamp: ref=1194, 10 s≈1195, 20 s≈1196

_DATASHEET_MEAN_COEF = 20.0
_DATASHEET_STD_COEF = 12.0
_DATASHEET_TREF_C = 20.0
_DATASHEET_DOUBLING_K = 5.5


def _det0_temp_celsius(header: dict) -> float:
    return det0_temperature_deg_c_from_header(header)


def _datasheet_dark_mean_std(
    effective_integration_time_s: float, T_celsius: float
) -> tuple[float, float]:
    exp_term = 2 ** ((T_celsius - _DATASHEET_TREF_C) / _DATASHEET_DOUBLING_K)
    mean_pred = effective_integration_time_s * _DATASHEET_MEAN_COEF * exp_term
    std_pred = effective_integration_time_s * _DATASHEET_STD_COEF * exp_term
    return mean_pred, std_pred


def _datasheet_exp_term(T_celsius: float) -> float:
    return 2 ** ((T_celsius - _DATASHEET_TREF_C) / _DATASHEET_DOUBLING_K)


def _datasheet_delta_mean_dark_e(delta_t_s: float, T_hi: float, T_lo: float) -> float:
    """Datasheet-only extra mean integrated dark (e⁻) at ``T_hi`` vs ``T_lo`` over ``delta_t_s``."""
    return (
        delta_t_s
        * _DATASHEET_MEAN_COEF
        * (_datasheet_exp_term(T_hi) - _datasheet_exp_term(T_lo))
    )


def _empirical_mean_vs_datasheet_scale(dark_pkg: dict[str, Any]) -> float:
    """Mean(observed frame mean / datasheet pred. mean) from ``process_folder`` dark results."""
    ratios: list[float] = []
    for dr in dark_pkg["dark_results"]:
        pm = float(dr["pred_mean"])
        if pm <= 0.0:
            continue
        ratios.append(float(dr["mean"]) / pm)
    if not ratios:
        raise RuntimeError("Could not derive empirical vs. datasheet scale from dark_pkg.")
    return float(np.mean(ratios))


def _load_fits(path: str) -> tuple[np.ndarray, dict]:
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        header = dict(hdul[0].header)
    return data, header


def _intg_seconds(header: dict) -> float:
    ms = header.get("INTG_MS")
    if ms is None:
        raise KeyError("INTG_MS not found in FITS header")
    return float(ms) / 1000.0


def _find_ref_path(paths: list[str], ref_token: str = "1185") -> str:
    for p in paths:
        if re.search(rf"image_{ref_token}\b", os.path.basename(p)):
            return p
    raise FileNotFoundError(
        f"No FITS filename matching image_{ref_token} in {[os.path.basename(p) for p in paths]}"
    )


def process_folder(
    folder: str, label: str, *, ref_token: str = DARK_REF_IMAGE_TOKEN
) -> dict[str, Any]:
    """Same fixed-pattern subtraction and statistics as post_vibe_dark_analysis (no display)."""
    fits_paths = sorted(glob.glob(os.path.join(folder, "*.fits")))
    if len(fits_paths) < 3:
        raise RuntimeError(
            f"[{label}] Expected at least 3 FITS files in {folder!r}, found {len(fits_paths)}."
        )

    ref_path = _find_ref_path(fits_paths, ref_token=ref_token)
    other_paths = [p for p in fits_paths if p != ref_path]
    if len(other_paths) != 2:
        raise RuntimeError(f"[{label}] Expected exactly two non-reference FITS files.")

    ref_data, ref_header = _load_fits(ref_path)
    t_ref_s = _intg_seconds(ref_header)
    fixed_e = ref_data / GAIN_DN_PER_ELECTRON
    fixed_mean = float(np.mean(fixed_e))
    fixed_median = float(np.median(fixed_e))
    fixed_std = float(np.std(fixed_e))

    dark_results: list[dict[str, Any]] = []
    for p in sorted(other_paths):
        data, header = _load_fits(p)
        t_s = _intg_seconds(header)
        delta_t_s = t_s - t_ref_s
        if delta_t_s <= 0:
            raise ValueError(
                f"[{label}] Non-positive Δt for {os.path.basename(p)} vs reference; cannot form e⁻/s."
            )
        dn_sub = data - ref_data
        dark_e = dn_sub / GAIN_DN_PER_ELECTRON
        mean = float(np.mean(dark_e))
        median = float(np.median(dark_e))
        std = float(np.std(dark_e))
        det_temp_c = _det0_temp_celsius(header)
        pred_mean, pred_std = _datasheet_dark_mean_std(delta_t_s, det_temp_c)
        dark_rate = dark_e / delta_t_s
        rate_mean = float(np.mean(dark_rate))
        rate_median = float(np.median(dark_rate))
        rate_std = float(np.std(dark_rate))
        pred_mean_rate = pred_mean / delta_t_s
        pred_std_rate = pred_std / delta_t_s
        dark_results.append(
            {
                "path": p,
                "basename": os.path.basename(p),
                "intg_s": t_s,
                "delta_t_s": delta_t_s,
                "det_temp_c": det_temp_c,
                "pred_mean": pred_mean,
                "pred_std": pred_std,
                "pred_mean_rate": pred_mean_rate,
                "pred_std_rate": pred_std_rate,
                "dark_e": dark_e,
                "dark_rate_e_per_s": dark_rate,
                "mean": mean,
                "median": median,
                "std": std,
                "rate_mean": rate_mean,
                "rate_median": rate_median,
                "rate_std": rate_std,
                "raw_dn": data,
            }
        )

    return {
        "label": label,
        "folder": folder,
        "ref_token": ref_token,
        "ref_path": ref_path,
        "t_ref_s": t_ref_s,
        "fixed_e": fixed_e,
        "fixed_mean": fixed_mean,
        "fixed_median": fixed_median,
        "fixed_std": fixed_std,
        "dark_results": dark_results,
    }


def _match_by_intg(
    dark_list: list[dict[str, Any]], led_list: list[dict[str, Any]], tol_s: float = 1e-6
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for d in sorted(dark_list, key=lambda x: x["intg_s"]):
        led = next(
            (L for L in led_list if abs(L["intg_s"] - d["intg_s"]) <= tol_s),
            None,
        )
        if led is None:
            raise RuntimeError(
                f"No LED frame with INTG matching dark {d['basename']} (intg_s={d['intg_s']})."
            )
        pairs.append((d, led))
    return pairs


def _print_pipeline_summary(pkg: dict[str, Any]) -> None:
    print(f"\n=== {pkg['label']} ===")
    print(f"Folder: {pkg['folder']}")
    print(
        f"Reference (image_{pkg['ref_token']}): {os.path.basename(pkg['ref_path'])}  "
        f"INTG={pkg['t_ref_s']:.6g} s"
    )
    print(
        f"Fixed pattern (e⁻): mean={pkg['fixed_mean']:.5g} median={pkg['fixed_median']:.5g} "
        f"std={pkg['fixed_std']:.5g}"
    )
    for dr in pkg["dark_results"]:
        print(
            f"  {dr['basename']}: INTG={dr['intg_s']:.6g} s  Δt={dr['delta_t_s']:.6g} s  "
            f"mean={dr['mean']:.5g} e⁻  std={dr['std']:.5g} e⁻  "
            f"rate_mean={dr['rate_mean']:.5g} e⁻/s"
        )


def _annotate_residual_stats(ax, info: dict[str, Any], panel_label: str) -> None:
    """Same residual statistics box as post_vibe_dark_analysis `_annotate_dark`."""
    lines = [
        f"{panel_label}: {info['basename']}",
        f"INTG = {info['intg_s']:.6g} s",
        f"Δt (vs ref) = {info['delta_t_s']:.6g} s",
        f"DET0_TEM = {info['det_temp_c']:.4g} °C",
        f"mean = {info['mean']:.5g} e⁻",
        f"median = {info['median']:.5g} e⁻",
        f"std dev = {info['std']:.5g} e⁻",
        f"datasheet pred. mean = {info['pred_mean']:.5g} e⁻",
        f"datasheet pred. σ = {info['pred_std']:.5g} e⁻",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="black", alpha=0.55),
    )


def _annotate_raw_dn(ax, info: dict[str, Any], panel_label: str) -> None:
    """Annotation for unaltered primary (DN) frames."""
    raw = info["raw_dn"]
    mean = float(np.mean(raw))
    median = float(np.median(raw))
    std = float(np.std(raw))
    lines = [
        f"{panel_label}: {info['basename']}",
        f"INTG = {info['intg_s']:.6g} s",
        f"DET0_TEM = {info['det_temp_c']:.4g} °C",
        f"mean = {mean:.5g} DN",
        f"median = {median:.5g} DN",
        f"std dev = {std:.5g} DN",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="white",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="black", alpha=0.55),
    )


def _save_figure3_raw_comparisons(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    out_path: str,
) -> None:
    """Single figure: per integration, raw dark vs raw LED side-by-side + overlaid DN histograms."""
    n = len(pairs)
    ratios: list[float] = []
    for _ in range(n):
        ratios.append(1.0)
        ratios.append(0.48)
    fig = plt.figure(figsize=(13, 2.8 + 4.2 * n))
    gs = fig.add_gridspec(
        2 * n,
        2,
        height_ratios=ratios,
        hspace=0.38,
        wspace=0.28,
    )
    for i, (dark_r, led_r) in enumerate(pairs):
        r_img = 2 * i
        stack = np.concatenate(
            [dark_r["raw_dn"].ravel().astype(np.float64), led_r["raw_dn"].ravel().astype(np.float64)]
        )
        vmin = float(np.percentile(stack, 2))
        vmax = float(np.percentile(stack, 98))
        if vmin >= vmax:
            vmin, vmax = float(np.min(stack)), float(np.max(stack))
        t = dark_r["intg_s"]
        ax_d = fig.add_subplot(gs[r_img, 0])
        ax_l = fig.add_subplot(gs[r_img, 1])
        im0 = ax_d.imshow(
            dark_r["raw_dn"],
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax_d.set_xticks([])
        ax_d.set_yticks([])
        ax_d.set_title("Dark (raw DN)", fontsize=10)
        _annotate_raw_dn(ax_d, dark_r, "Dark")
        fig.colorbar(
            im0, ax=ax_d, location="left", fraction=0.046, pad=0.04, label="DN"
        )
        im1 = ax_l.imshow(
            led_r["raw_dn"],
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax_l.set_xticks([])
        ax_l.set_yticks([])
        ax_l.set_title("External LED (raw DN)", fontsize=10)
        _annotate_raw_dn(ax_l, led_r, "Ext. LED")
        fig.colorbar(
            im1, ax=ax_l, location="left", fraction=0.046, pad=0.04, label="DN"
        )

        ax_h = fig.add_subplot(gs[r_img + 1, :])
        combined = np.concatenate(
            [dark_r["raw_dn"].ravel().astype(np.float64), led_r["raw_dn"].ravel().astype(np.float64)]
        )
        lo = float(np.percentile(combined, 0.5))
        hi = float(np.percentile(combined, 99.5))
        if lo >= hi:
            lo, hi = float(combined.min()), float(combined.max())
        bins = np.linspace(lo, hi, 120)
        ax_h.hist(
            dark_r["raw_dn"].ravel(),
            bins=bins,
            color="C0",
            alpha=0.55,
            density=True,
            label=f"Dark ({dark_r['basename']})",
        )
        ax_h.hist(
            led_r["raw_dn"].ravel(),
            bins=bins,
            color="C1",
            alpha=0.55,
            density=True,
            label=f"LED ({led_r['basename']})",
        )
        ax_h.set_xlabel("DN (raw, full frame)")
        ax_h.set_ylabel("density")
        ax_h.set_title(f"Raw DN histograms | INTG = {t:.6g} s", fontsize=10)
        ax_h.legend(fontsize=7)

    fig.suptitle(
        "Figure 3 | Unaltered images: raw dark vs raw external stim lamp (no fixed pattern subtraction)",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=[0.06, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150)


def _first_matching_raw_shape_hw(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[int, int] | None:
    for dark_r, led_r in pairs:
        d = np.asarray(dark_r["raw_dn"])
        l = np.asarray(led_r["raw_dn"])
        if d.shape == l.shape and d.ndim == 2 and d.size:
            return (int(d.shape[0]), int(d.shape[1]))
    return None


def _set_imshow_box_aspect(ax: Any, image_2d: np.ndarray) -> None:
    """Match axes box aspect to the array so imshow does not letterbox empty bands."""
    if image_2d.ndim != 2 or image_2d.shape[0] < 1 or image_2d.shape[1] < 1:
        return
    h, w = int(image_2d.shape[0]), int(image_2d.shape[1])
    ax.set_box_aspect(h / w)


def _fig_size_led_raw_one_row(
    n: int,
    shape_hw: tuple[int, int] | None,
    *,
    fig_width_cap: float = 18.0,
) -> tuple[float, float]:
    """(width, height) in inches for a single row of ``n`` raw-DN image panels (sliders not included)."""
    if n <= 0:
        return 8.5, 4.0
    fig_w = min(fig_width_cap, max(9.0, 4.2 * float(n) + 1.5))
    if shape_hw is None:
        return fig_w, 4.2 + 0.35
    h_px, w_px = int(shape_hw[0]), max(int(shape_hw[1]), 1)
    per_col_w = (fig_w * 0.9) / float(n)
    plot_h = per_col_w * (h_px / w_px)
    row_h = plot_h + 0.7
    return fig_w, row_h + 0.35


def show_led_on_dark_vmin_vmax_interactive(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    dark_floor_dn: float = 1.0,
) -> None:
    """
    Two windows: (1) LED − dark raw DN, (2) LED / dark raw DN. Each arranges the ``n``
    integrations in a single row and has its own vmin / vmax sliders.
    """
    n = len(pairs)
    if n == 0:
        return

    ratio_data: list[np.ndarray] = []
    diff_data: list[np.ndarray] = []
    for dark_r, led_r in pairs:
        dark_raw = np.asarray(dark_r["raw_dn"], dtype=np.float64)
        led_raw = np.asarray(led_r["raw_dn"], dtype=np.float64)
        if dark_raw.shape != led_raw.shape:
            raise ValueError("Shape mismatch between dark and LED raw frames in a pair.")
        denom = np.maximum(dark_raw, dark_floor_dn)
        ratio_data.append((led_raw / denom).astype(np.float32))
        diff_data.append((led_raw - dark_raw).astype(np.float32))

    def _stack_range(arrays: list[np.ndarray]) -> tuple[float, float, float, float]:
        c = np.concatenate([a.ravel() for a in arrays])
        c = c[np.isfinite(c)]
        if c.size == 0:
            return 0.0, 1.0, 0.0, 1.0
        lo, hi = float(c.min()), float(c.max())
        p1, p99 = float(np.percentile(c, 1.0)), float(np.percentile(c, 99.0))
        span = hi - lo
        if span < 1e-12:
            span = 1.0
        m = 0.02 * span
        return lo - m, hi + m, p1, p99

    r_slo, r_shi, r_p1, r_p99 = _stack_range(ratio_data)
    d_slo, d_shi, d_p1, d_p99 = _stack_range(diff_data)

    sh = _first_matching_raw_shape_hw(pairs)
    fig_w, fig_h = _fig_size_led_raw_one_row(n, sh)
    fig_h = fig_h + 0.45

    def _clip_pair(
        slo: float, shi: float, v0: float, v1: float
    ) -> tuple[float, float]:
        if v0 >= v1 or not np.isfinite(v0) or not np.isfinite(v1):
            v0, v1 = slo, max(shi, slo + 1e-12 * max(abs(slo), 1.0))
        v0 = float(np.clip(v0, slo, shi))
        v1 = float(np.clip(v1, slo, shi))
        if v0 >= v1:
            v1 = min(shi, v0 + 1e-9 * (abs(v0) + 1.0))
        return v0, v1

    vd0, vd1 = _clip_pair(d_slo, d_shi, d_p1, d_p99)
    vr0, vr1 = _clip_pair(r_slo, r_shi, r_p1, r_p99)

    # --- Figure: LED − dark (one row, n columns) ---
    fig_d, axes_d = plt.subplots(
        1,
        n,
        figsize=(fig_w, fig_h),
        num="LED − dark, raw DN (interactive)",
        squeeze=False,
    )
    ax_d_list = np.asarray(axes_d).flatten().tolist()
    ims_d: list[Any] = []
    for i, ax in enumerate(ax_d_list):
        t = pairs[i][0]["intg_s"]
        v0, v1 = d_p1, d_p99
        if v0 >= v1 or not np.isfinite(v0) or not np.isfinite(v1):
            v0, v1 = d_slo, max(d_shi, d_slo + 1e-12)
        im = ax.imshow(
            diff_data[i],
            cmap="coolwarm",
            vmin=v0,
            vmax=v1,
            interpolation="nearest",
        )
        _set_imshow_box_aspect(ax, diff_data[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"LED − dark | INTG = {t:.6g} s", fontsize=10)
        ims_d.append(im)

    cbar_d = fig_d.colorbar(
        ims_d[0],
        ax=ax_d_list,
        location="left",
        fraction=0.03,
        pad=0.04,
        label="LED − dark (DN)",
    )
    fig_d.suptitle(
        "Raw DN | LED on minus dark | matched by INTG",
        fontsize=11,
        y=0.98,
    )
    fig_d.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.2)

    ax_d_vmin = fig_d.add_axes((0.45, 0.10, 0.5, 0.022))
    ax_d_vmax = fig_d.add_axes((0.45, 0.06, 0.5, 0.022))

    def _apply_diff_clim() -> None:
        v0, v1 = float(s_d_vmin.val), float(s_d_vmax.val)
        if v0 >= v1:
            return
        for im in ims_d:
            im.set_clim(v0, v1)
        cbar_d.update_normal(ims_d[0])
        fig_d.canvas.draw_idle()

    s_d_vmin = Slider(
        ax_d_vmin, "vmin", d_slo, d_shi, valinit=vd0, valfmt="%.5g", dragging=True
    )
    s_d_vmax = Slider(
        ax_d_vmax, "vmax", d_slo, d_shi, valinit=vd1, valfmt="%.5g", dragging=True
    )
    s_d_vmin.on_changed(lambda _: _apply_diff_clim())
    s_d_vmax.on_changed(lambda _: _apply_diff_clim())
    _apply_diff_clim()

    # --- Figure: LED / dark (one row, n columns) ---
    fig_r, axes_r = plt.subplots(
        1,
        n,
        figsize=(fig_w, fig_h),
        num="LED / dark, raw DN (interactive)",
        squeeze=False,
    )
    ax_r_list = np.asarray(axes_r).flatten().tolist()
    ims_r: list[Any] = []
    for i, ax in enumerate(ax_r_list):
        t = pairs[i][0]["intg_s"]
        v0, v1 = r_p1, r_p99
        if v0 >= v1 or not np.isfinite(v0) or not np.isfinite(v1):
            v0, v1 = r_slo, max(r_shi, r_slo + 1e-12)
        im = ax.imshow(
            ratio_data[i],
            cmap="viridis",
            vmin=v0,
            vmax=v1,
            interpolation="nearest",
        )
        _set_imshow_box_aspect(ax, ratio_data[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"LED / dark | INTG = {t:.6g} s", fontsize=10)
        ims_r.append(im)

    cbar_r = fig_r.colorbar(
        ims_r[0],
        ax=ax_r_list,
        location="left",
        fraction=0.03,
        pad=0.04,
        label="LED / dark (DN)",
    )
    fig_r.suptitle(
        "Raw DN | LED on divided by dark | matched by INTG",
        fontsize=11,
        y=0.98,
    )
    fig_r.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.2)

    ax_r_vmin = fig_r.add_axes((0.45, 0.10, 0.5, 0.022))
    ax_r_vmax = fig_r.add_axes((0.45, 0.06, 0.5, 0.022))

    def _apply_ratio_clim() -> None:
        v0, v1 = float(s_r_vmin.val), float(s_r_vmax.val)
        if v0 >= v1:
            return
        for im in ims_r:
            im.set_clim(v0, v1)
        cbar_r.update_normal(ims_r[0])
        fig_r.canvas.draw_idle()

    s_r_vmin = Slider(
        ax_r_vmin, "vmin", r_slo, r_shi, valinit=vr0, valfmt="%.5g", dragging=True
    )
    s_r_vmax = Slider(
        ax_r_vmax, "vmax", r_slo, r_shi, valinit=vr1, valfmt="%.5g", dragging=True
    )
    s_r_vmin.on_changed(lambda _: _apply_ratio_clim())
    s_r_vmax.on_changed(lambda _: _apply_ratio_clim())
    _apply_ratio_clim()


def _save_side_by_side_residuals(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    out_path: str,
) -> None:
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4.8 * n), squeeze=False)
    for i, (dark_r, led_r) in enumerate(pairs):
        stack = np.concatenate([dark_r["dark_e"].ravel(), led_r["dark_e"].ravel()])
        vmin = float(np.percentile(stack, 2))
        vmax = float(np.percentile(stack, 98))
        if vmin >= vmax:
            vmin, vmax = float(np.min(stack)), float(np.max(stack))
        for j, (dr, title_prefix) in enumerate(
            ((dark_r, "Dark"), (led_r, "External LED"))
        ):
            ax = axes[i, j]
            im = ax.imshow(dr["dark_e"], cmap="inferno", vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title_prefix, fontsize=10)
            _annotate_residual_stats(ax, dr, title_prefix)
            fig.colorbar(
                im, ax=ax, location="left", fraction=0.046, pad=0.04, label="e⁻ (residual)"
            )
    fig.suptitle(
        "Residual after fixed-pattern subtraction (electrons), matched by integration time",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=[0.07, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150)


def _save_histogram_overlays(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    empirical_scale: float,
    out_path: str,
) -> None:
    """
    Two rows × n columns: (top) dark vs raw LED residual e⁻ histograms; (bottom) same
    with uniform offset removed from LED = ``empirical_scale`` × datasheet Δdark(ΔT).
    """
    n = len(pairs)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 7.4), squeeze=False)

    for j, (dark_r, led_r) in enumerate(pairs):
        dt = float(dark_r["delta_t_s"])
        T_d = float(dark_r["det_temp_c"])
        T_l = float(led_r["det_temp_c"])
        offset_e = empirical_scale * _datasheet_delta_mean_dark_e(dt, T_l, T_d)
        led_corr = led_r["dark_e"] - offset_e

        for row, led_arr, subtitle in (
            (0, led_r["dark_e"], "raw LED residual"),
            (1, led_corr, f"LED − {offset_e:.4g} e⁻ | empir. Δdark for ΔT"),
        ):
            ax = axes[row, j]
            d_flat = dark_r["dark_e"].ravel()
            l_flat = np.asarray(led_arr, dtype=np.float64).ravel()
            combined = np.concatenate([d_flat, l_flat])
            lo = float(np.percentile(combined, 0.5))
            hi = float(np.percentile(combined, 99.5))
            if lo >= hi:
                lo, hi = float(combined.min()), float(combined.max())
            bins = np.linspace(lo, hi, 120)
            mean_d = float(np.mean(d_flat))
            mean_l = float(np.mean(l_flat))
            led_leg = (
                f"LED (uncorrected; µ = {int(round(mean_l))} e⁻)"
                if row == 0
                else f"LED (T corrected; µ = {int(round(mean_l))} e⁻)"
            )
            ax.hist(
                d_flat,
                bins=bins,
                color="C0",
                alpha=0.55,
                density=True,
                label=f"Dark (µ = {int(round(mean_d))} e⁻)",
            )
            ax.hist(
                l_flat,
                bins=bins,
                color="C1",
                alpha=0.55,
                density=True,
                label=led_leg,
            )
            ax.set_xlabel("e⁻ (residual, full frame)")
            ax.set_ylabel("density")
            t = dark_r["intg_s"]
            ax.set_title(
                f"INTG = {t:g} s | ΔT = {T_l - T_d:+.3g} °C\n{subtitle}",
                fontsize=9,
            )
            ax.legend(fontsize=7)

    fig.suptitle(
        "Residual e⁻ histograms | raw vs LED after empirically scaled temperature dark correction\n"
        f"(obs/pred scale = {empirical_scale:.3g} from post-vibe dark set)",
        fontsize=10,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def _print_pair_stats(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
    print("\n=== Dark vs LED (matched by INTG) ===")
    for dark_r, led_r in pairs:
        d_flat = dark_r["dark_e"].ravel()
        l_flat = led_r["dark_e"].ravel()
        if d_flat.shape != l_flat.shape:
            print(
                f"  INTG={dark_r['intg_s']:.6g} s: shape mismatch "
                f"{d_flat.shape} vs {l_flat.shape}"
            )
            continue
        diff = l_flat.astype(np.float64) - d_flat.astype(np.float64)
        rmse = float(np.sqrt(np.mean(diff**2)))
        mad = float(np.median(np.abs(diff)))
        print(
            f"  INTG={dark_r['intg_s']:.6g} s: mean_dark={dark_r['mean']:.6g} mean_led={led_r['mean']:.6g} "
            f"std_dark={dark_r['std']:.6g} std_led={led_r['std']:.6g} "
            f"RMSE(dark,led)={rmse:.6g} MAD={mad:.6g}"
        )
        if scipy_stats is not None:
            ks = scipy_stats.ks_2samp(d_flat, l_flat)
            print(
                f"    KS two-sample: statistic={float(ks.statistic):.6g} p-value={float(ks.pvalue):.6g}"
            )


def main() -> None:
    root = os.environ.get("suncet_data")
    if not root:
        print("Set environment variable suncet_data to your data root.", file=sys.stderr)
        sys.exit(1)

    dark_folder = os.path.join(root, "test_data", "2026-04-20_three_dark_images", "level0_5")
    led_folder = os.path.join(
        root, "test_data", "2026-04-20_external_stim_lamp", "level0_5"
    )

    dark_pkg = process_folder(dark_folder, label="dark (post-vibe)", ref_token=DARK_REF_IMAGE_TOKEN)
    led_pkg = process_folder(
        led_folder, label="external stim lamp", ref_token=LED_REF_IMAGE_TOKEN
    )

    _print_pipeline_summary(dark_pkg)
    _print_pipeline_summary(led_pkg)

    pairs = _match_by_intg(dark_pkg["dark_results"], led_pkg["dark_results"])
    _print_pair_stats(pairs)

    empirical_scale = _empirical_mean_vs_datasheet_scale(dark_pkg)

    out_dir = os.path.join(os.getcwd(), "led_external_analysis_output")
    os.makedirs(out_dir, exist_ok=True)
    side_path = os.path.join(out_dir, "dark_vs_led_residuals_side_by_side.png")
    hist_path = os.path.join(out_dir, "dark_vs_led_residual_histograms.png")
    fig3_path = os.path.join(out_dir, "dark_vs_led_raw_figure3.png")
    _save_side_by_side_residuals(pairs, side_path)
    _save_histogram_overlays(pairs, empirical_scale, hist_path)
    _save_figure3_raw_comparisons(pairs, fig3_path)
    show_led_on_dark_vmin_vmax_interactive(pairs)
    print(f"\nWrote:\n  {side_path}\n  {hist_path}\n  {fig3_path}")
    print(
        "Displaying interactive windows: LED − dark (raw DN) and LED / dark (raw DN); "
        "close windows to exit."
    )
    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
