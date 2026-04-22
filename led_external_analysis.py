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
from matplotlib.widgets import RadioButtons, Slider
from astropy.io import fits

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
    v = header.get("DET0_TEM")
    if v is None:
        raise KeyError("DET0_TEM not found in FITS header")
    return float(v) / 100.0


def _datasheet_dark_mean_std(
    effective_integration_time_s: float, T_celsius: float
) -> tuple[float, float]:
    exp_term = 2 ** ((T_celsius - _DATASHEET_TREF_C) / _DATASHEET_DOUBLING_K)
    mean_pred = effective_integration_time_s * _DATASHEET_MEAN_COEF * exp_term
    std_pred = effective_integration_time_s * _DATASHEET_STD_COEF * exp_term
    return mean_pred, std_pred


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
        f"DET0_TEM = {info['det_temp_c']:.4g} °C (hdr/100)",
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
        f"DET0_TEM = {info['det_temp_c']:.4g} °C (hdr/100)",
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
        ax_h.set_title(f"Raw DN histograms — INTG = {t:.6g} s", fontsize=10)
        ax_h.legend(fontsize=7)

    fig.suptitle(
        "Figure 3 — Unaltered images: raw dark vs raw external stim lamp (no fixed-pattern subtraction)",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=[0.06, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150)


def _save_led_on_over_dark_ratio(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    out_path: str,
    *,
    dark_floor_dn: float = 1.0,
) -> None:
    """
    Per matched integration: image of (external LED raw DN) / (dark raw DN).

    Denominator is clipped to at least `dark_floor_dn` to avoid divide-by-zero;
    only finite ratios are used for the displayed color scale (percentile robust).
    """
    n = len(pairs)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 3.2 * n), squeeze=False)
    axes_flat = axes.ravel()
    for ax, (dark_r, led_r) in zip(axes_flat, pairs):
        t = dark_r["intg_s"]
        dark_raw = np.asarray(dark_r["raw_dn"], dtype=np.float64)
        led_raw = np.asarray(led_r["raw_dn"], dtype=np.float64)
        if dark_raw.shape != led_raw.shape:
            ax.set_visible(False)
            continue
        denom = np.maximum(dark_raw, dark_floor_dn)
        ratio = led_raw / denom
        flat = ratio[np.isfinite(ratio)]
        if flat.size:
            vmin = float(np.percentile(flat, 1.0))
            vmax = float(np.percentile(flat, 99.0))
        else:
            vmin, vmax = 0.0, 1.0
        if vmin >= vmax:
            vmin, vmax = float(np.nanmin(flat)) if flat.size else 0.0, float(np.nanmax(flat)) if flat.size else 1.0
        r_mean = float(np.nanmean(ratio))
        r_med = float(np.nanmedian(ratio))
        r_std = float(np.nanstd(ratio))
        im = ax.imshow(
            ratio,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"LED on / dark (raw DN) — INTG = {t:.6g} s", fontsize=10)
        lines = [
            f"Dark: {dark_r['basename']}",
            f"LED: {led_r['basename']}",
            f"mean ratio = {r_mean:.5g}",
            f"median ratio = {r_med:.5g}",
            f"std ratio = {r_std:.5g}",
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
        fig.colorbar(
            im, ax=ax, location="left", fraction=0.046, pad=0.04, label="LED / dark (DN)"
        )

    fig.suptitle(
        "Raw DN ratio: external stim lamp (LED on) divided by post-vibration dark, matched by INTG",
        fontsize=11,
        y=0.998,
    )
    fig.tight_layout(rect=[0.07, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)


def _save_led_on_minus_dark(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    out_path: str,
) -> None:
    """Per matched integration: image of (external LED raw DN) − (dark raw DN)."""
    n = len(pairs)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 3.2 * n), squeeze=False)
    axes_flat = axes.ravel()
    for ax, (dark_r, led_r) in zip(axes_flat, pairs):
        t = dark_r["intg_s"]
        dark_raw = np.asarray(dark_r["raw_dn"], dtype=np.float64)
        led_raw = np.asarray(led_r["raw_dn"], dtype=np.float64)
        if dark_raw.shape != led_raw.shape:
            ax.set_visible(False)
            continue
        diff = led_raw - dark_raw
        flat = diff.ravel()
        p_lo = float(np.percentile(flat, 1.0))
        p_hi = float(np.percentile(flat, 99.0))
        lim = max(abs(p_lo), abs(p_hi))
        if lim < 1e-12:
            lim = 1.0
        vmin, vmax = -lim, lim
        d_mean = float(np.mean(diff))
        d_med = float(np.median(diff))
        d_std = float(np.std(diff))
        im = ax.imshow(
            diff,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"LED on − dark (raw DN) — INTG = {t:.6g} s", fontsize=10)
        lines = [
            f"Dark: {dark_r['basename']}",
            f"LED: {led_r['basename']}",
            f"mean Δ = {d_mean:.5g} DN",
            f"median Δ = {d_med:.5g} DN",
            f"std Δ = {d_std:.5g} DN",
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
        fig.colorbar(
            im, ax=ax, location="left", fraction=0.046, pad=0.04, label="LED − dark (DN)"
        )

    fig.suptitle(
        "Raw DN difference: external stim lamp (LED on) minus post-vibration dark, matched by INTG",
        fontsize=11,
        y=0.998,
    )
    fig.tight_layout(rect=[0.07, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)


def show_led_on_dark_vmin_vmax_interactive(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    dark_floor_dn: float = 1.0,
) -> None:
    """
    Open a matplotlib window: ratio (LED/dark) or difference (LED−dark) with interactive vmin/vmax.
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

    fig, axes = plt.subplots(
        n,
        1,
        figsize=(8.5, 0.4 + 2.9 * n),
        num="LED vs dark (interactive vmin / vmax)",
        squeeze=False,
    )
    ax_list = [axes.ravel()[i] for i in range(n)]
    ims: list[Any] = []
    for i, ax in enumerate(ax_list):
        t = pairs[i][0]["intg_s"]
        v0, v1 = r_p1, r_p99
        if v0 >= v1:
            v0, v1 = r_slo, r_shi
        im = ax.imshow(
            ratio_data[i],
            cmap="viridis",
            vmin=v0,
            vmax=v1,
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"INTG = {t:.6g} s", fontsize=10)
        ims.append(im)

    cbar = fig.colorbar(
        ims[0],
        ax=ax_list,
        location="left",
        fraction=0.03,
        pad=0.04,
        label="LED / dark (DN)",
    )
    fig.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

    ax_mode = fig.add_axes((0.08, 0.02, 0.3, 0.09))
    ax_vmin = fig.add_axes((0.45, 0.1, 0.5, 0.02))
    ax_vmax = fig.add_axes((0.45, 0.05, 0.5, 0.02))

    s_vmin: Any = None
    s_vmax: Any = None

    def _apply_clim() -> None:
        if s_vmin is None or s_vmax is None:
            return
        v0, v1 = s_vmin.val, s_vmax.val
        if v0 >= v1:
            return
        for im in ims:
            im.set_clim(v0, v1)
        cbar.update_normal(ims[0])
        fig.canvas.draw_idle()

    def _make_sliders(slo: float, shi: float, v0: float, v1: float) -> None:
        nonlocal s_vmin, s_vmax
        ax_vmin.clear()
        ax_vmax.clear()
        if v0 >= v1 or not np.isfinite(v0) or not np.isfinite(v1):
            v0, v1 = slo, max(shi, slo + 1e-12 * max(abs(slo), 1.0))
        v0 = float(np.clip(v0, slo, shi))
        v1 = float(np.clip(v1, slo, shi))
        if v0 >= v1:
            v1 = min(shi, v0 + 1e-9 * (abs(v0) + 1.0))
        s_vmin = Slider(ax_vmin, "vmin", slo, shi, valinit=v0, valfmt="%.5g", dragging=True)
        s_vmax = Slider(ax_vmax, "vmax", slo, shi, valinit=v1, valfmt="%.5g", dragging=True)
        s_vmin.on_changed(lambda _: _apply_clim())
        s_vmax.on_changed(lambda _: _apply_clim())
        _apply_clim()

    _make_sliders(r_slo, r_shi, r_p1, r_p99)

    def on_mode(label: str) -> None:
        use_ratio = "Ratio" in label
        arrays = ratio_data if use_ratio else diff_data
        cmap = "viridis" if use_ratio else "coolwarm"
        slo, shi, p1, p99 = (
            (r_slo, r_shi, r_p1, r_p99) if use_ratio else (d_slo, d_shi, d_p1, d_p99)
        )
        for i, im in enumerate(ims):
            im.set_data(arrays[i])
            im.set_cmap(cmap)
        cbar.mappable = ims[0]
        cbar.set_label("LED / dark" if use_ratio else "LED − dark (DN)")
        cbar.update_normal(ims[0])
        v0, v1 = p1, p99
        if v0 >= v1:
            v0, v1 = slo, max(shi, slo + 1e-12)
        _make_sliders(slo, shi, v0, v1)

    radio = RadioButtons(
        ax_mode,
        ("Ratio (LED / dark)", "Difference (LED − dark)"),
        active=0,
    )
    radio.on_clicked(on_mode)


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
    out_path: str,
) -> None:
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2), squeeze=False)
    axes_flat = axes.ravel()
    for ax, (dark_r, led_r) in zip(axes_flat, pairs):
        t = dark_r["intg_s"]
        combined = np.concatenate([dark_r["dark_e"].ravel(), led_r["dark_e"].ravel()])
        lo = float(np.percentile(combined, 0.5))
        hi = float(np.percentile(combined, 99.5))
        if lo >= hi:
            lo, hi = float(combined.min()), float(combined.max())
        bins = np.linspace(lo, hi, 120)
        ax.hist(
            dark_r["dark_e"].ravel(),
            bins=bins,
            color="C0",
            alpha=0.55,
            density=True,
            label=f"Dark ({dark_r['basename']})",
        )
        ax.hist(
            led_r["dark_e"].ravel(),
            bins=bins,
            color="C1",
            alpha=0.55,
            density=True,
            label=f"LED ({led_r['basename']})",
        )
        ax.set_xlabel("e⁻ (residual, full frame)")
        ax.set_ylabel("density")
        ax.set_title(f"INTG = {t:.6g} s", fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(
        "Residual electron histograms (normalized) — dark vs external stim lamp",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
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

    out_dir = os.path.join(os.getcwd(), "led_external_analysis_output")
    os.makedirs(out_dir, exist_ok=True)
    side_path = os.path.join(out_dir, "dark_vs_led_residuals_side_by_side.png")
    hist_path = os.path.join(out_dir, "dark_vs_led_residual_histograms.png")
    fig3_path = os.path.join(out_dir, "dark_vs_led_raw_figure3.png")
    ratio_path = os.path.join(out_dir, "led_on_over_dark_ratio.png")
    diff_path = os.path.join(out_dir, "led_on_minus_dark.png")
    _save_side_by_side_residuals(pairs, side_path)
    _save_histogram_overlays(pairs, hist_path)
    _save_figure3_raw_comparisons(pairs, fig3_path)
    _save_led_on_over_dark_ratio(pairs, ratio_path)
    _save_led_on_minus_dark(pairs, diff_path)
    show_led_on_dark_vmin_vmax_interactive(pairs)
    print(
        f"\nWrote:\n  {side_path}\n  {hist_path}\n  {fig3_path}\n  {ratio_path}\n  {diff_path}"
    )
    print("Displaying figures (close windows to exit).")
    plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
