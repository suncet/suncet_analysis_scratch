"""
Air bearing test: ADCS attitude-control sun point angle error vs time.

Reads level-0.5 mission-length telemetry HDF5 and plots
``adcs_att_ctrl_sun_point_ang_err`` versus the raw time column from the file
(``pktTimestamp`` or ``timestamp_seconds_since_boot``, whichever is present).
Rows where the angle error is NaN or non-finite are dropped from the plot.

Also plots ADCS CSS packet fields
(``adcs_css_num_diodes_used_*``, ``adcs_css_raw_sun_sensor_data_0`` … ``_7`` only,
``adcs_css_sun_sensor_used``, ``adcs_css_meas_sun_vld``) versus the same time
axis, one figure per channel.

Flood light phase: figures overplot ``adcs_css_raw_sun_sensor_data_0`` … ``_7``
on a shared time axis (ch0–3 black; ch4 green; ch5 blue; ch6 red; ch7 gold): full
flood window, plus a tighter zoom ``[1.7, 1.8]×10¹²`` s.

One mission-length figure overplots beacon wheel speeds ``beac_adcs_wheel_sp{1,2,3}``
(tomato / lime green / dodger blue) with the same test-region shading.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory

# ADCS attitude control telemetry — sun point angle error (see HDF5 field list)
SUN_POINT_ANGLE_ERR = "adcs_att_ctrl_sun_point_ang_err"
# Prefer primary packet timestamp; fall back if empty / all zero
TS_CANDIDATES = ("pktTimestamp", "timestamp_seconds_since_boot")

# Region fill colors: blue, green, same blue, yellow.
_REGION_BLUE = "#a8d5e2"
_REGION_GREEN = "#b8e0c4"
_REGION_YELLOW = "#f5e199"

# Test phases (x in same units as the HDF5 time column; ~1e12 scale here).
# (xmin, xmax_or_None, label, facecolor) — None xmax uses max(t).
_TEST_REGIONS: tuple[tuple[float, float | None, str, str], ...] = (
    (0.0, 0.67e12, "setup", _REGION_BLUE),
    (0.67e12, 1.3e12, "heliostat", _REGION_GREEN),
    (1.3e12, 1.7e12, "axis change", _REGION_BLUE),
    (1.7e12, None, "flood light", _REGION_YELLOW),
)

# Heliostat zoom plot y-range; stats below use only errors in _HELIOSTAT_STATS_Y_RANGE.
_HELIOSTAT_PLOT_YMAX = 5.0
_HELIOSTAT_STATS_Y_LO = 0.0
_HELIOSTAT_STATS_Y_HI = 2.0

# Flood light raw overlay: ch0–3 black; ch4 green; ch5 blue; ch6 red; ch7 gold.
_FLOOD_LIGHT_RAW_OVERLAY_COLORS: tuple[str, ...] = (
    "black",
    "black",
    "black",
    "black",
    "green",
    "blue",
    "red",
    "gold",
)

# Narrow flood-light zoom for raw overlay (secondary header seconds scale).
_FLOOD_LIGHT_RAW_ZOOM_X0 = 1.7e12
_FLOOD_LIGHT_RAW_ZOOM_X1 = 1.8e12

# Beacon packet wheel speeds (3 channels).
_BEACON_WHEEL_SP_KEYS: tuple[str, str, str] = (
    "beac_adcs_wheel_sp1",
    "beac_adcs_wheel_sp2",
    "beac_adcs_wheel_sp3",
)
_BEACON_WHEEL_LINE_COLORS: tuple[str, str, str] = ("tomato", "limegreen", "dodgerblue")
_BEACON_WHEEL_REF_RPM_LIGHT = 6000.0
_BEACON_WHEEL_REF_RPM_DARK = 8000.0
_REF_LINE_GREY_LIGHT = "#b8b8b8"
_REF_LINE_GREY_DARK = "#555555"
_LABEL_WHEEL_MOMENTUM = "limited momentum control remaining"
_LABEL_WHEEL_SATURATION = "wheel saturation"

# ADCS momentum packet: wheel/body momentum and total momentum magnitude.
_WHEEL_MOM_BODY_KEYS: tuple[str, str, str] = (
    "adcs_mom_wheel_mom_body_0",
    "adcs_mom_wheel_mom_body_1",
    "adcs_mom_wheel_mom_body_2",
)
_TOTAL_MOM_MAG_KEY = "adcs_mom_total_mom_mag"
_WHEEL_MOM_LINE_COLORS: tuple[str, str, str] = ("tomato", "limegreen", "dodgerblue")
_WHEEL_MOM_MAX_STORAGE_NMS = 0.015
_LABEL_WHEEL_MOM_MAX_STORAGE = "max wheel storage"
# Wheel momentum zoom plots (secondary header seconds).
_WHEEL_MOM_ZOOM_X0 = 2.2e12
_WHEEL_MOM_ZOOM_X1 = 2.4e12

# Torque-rod command indicators (duty cycle channels).
_TRQ_ROD_DUTY_CYC_KEYS: tuple[str, str, str] = (
    "adcs_mom_duty_cyc_0",
    "adcs_mom_duty_cyc_1",
    "adcs_mom_duty_cyc_2",
)
_TRQ_ROD_DUTY_LINE_COLORS: tuple[str, str, str] = ("tomato", "limegreen", "dodgerblue")


def _reinterpret_u32_counts_as_f32(x: np.ndarray) -> np.ndarray:
    """
    Some telemetry fields are defined as IEEE-754 'single' (float32) but may have
    been decoded upstream as an unsigned 32-bit integer before being written into
    the HDF5. This converts such uint32 *bit patterns* (stored as numbers) back
    into float32 values.
    """
    u = np.asarray(x)
    # Keep NaNs as NaNs (if any) while converting: NaN -> 0 bits -> 0.0f
    # We later rely on finite filtering in plots.
    if np.issubdtype(u.dtype, np.floating):
        fin = np.isfinite(u)
        u2 = np.zeros_like(u, dtype=np.uint32)
        u2[fin] = u[fin].astype(np.uint32, copy=False)
        return u2.view(np.float32).astype(np.float64)
    return u.astype(np.uint32, copy=False).view(np.float32).astype(np.float64)


def _looks_like_uint32_packed_float(x: np.ndarray) -> bool:
    """Heuristic: float array with integer-like values in uint32 range."""
    a = np.asarray(x)
    if not np.issubdtype(a.dtype, np.floating):
        return False
    fin = a[np.isfinite(a)]
    if fin.size == 0:
        return False
    # If most finite samples are whole numbers and in a plausible uint32 range,
    # assume this is a bit-pattern field that needs reinterpretation.
    frac = np.modf(fin)[0]
    whole = np.mean(np.isclose(frac, 0.0))
    if whole < 0.95:
        return False
    if np.nanmin(fin) < 0:
        return False
    if np.nanmax(fin) > 4294967295:
        return False
    # Exclude cases where it's already clearly a small physical float.
    if np.nanmedian(fin) < 1e6:
        return False
    return True


def _heliostat_stats_text(err_in_window: np.ndarray) -> str:
    """
    err_in_window: sun point error for rows in the heliostat time window (finite
    t/err pairs only). Mean/median/std use only finite err with
    _HELIOSTAT_STATS_Y_LO ≤ err ≤ _HELIOSTAT_STATS_Y_HI.
    """
    y0, y1 = _HELIOSTAT_STATS_Y_LO, _HELIOSTAT_STATS_Y_HI
    n_win = int(err_in_window.size)
    e = err_in_window[np.isfinite(err_in_window)]
    n_fin = int(e.size)
    e_band = e[(e >= y0) & (e <= y1)]
    n = int(e_band.size)
    hdr = (
        f"Stats: computed within [{y0:g}, {y1:g}º] only\n"
        f"(heliostat time window; y-axis 0–{_HELIOSTAT_PLOT_YMAX:g} for context)\n"
    )
    if n == 0:
        return hdr + f"n in window = {n_win}\nfinite err = {n_fin}\nNo points in [{y0:g}, {y1:g}]"
    mean = float(np.mean(e_band))
    med = float(np.median(e_band))
    if n == 1:
        std_s = "—"
    else:
        std_s = f"{float(np.std(e_band, ddof=1)):.6g}"
    return (
        hdr
        + f"n in window = {n_win}\n"
        + f"n in [{y0:g}, {y1:g}] = {n}\n"
        + f"mean = {mean:.6g}\n"
        + f"median = {med:.6g}\n"
        + f"std = {std_s}"
    )


def _region_bounds_by_label(label: str) -> tuple[float, float]:
    """Return (xmin, xmax) for a named region; xmax must be finite."""
    for xmin, xmax, lab, _ in _TEST_REGIONS:
        if lab == label:
            if xmax is None:
                raise ValueError(f"Region {label!r} has open xmax; need finite bounds")
            return (float(xmin), float(xmax))
    raise ValueError(f"No region named {label!r} in _TEST_REGIONS")


def _region_x_bounds(label: str, t_max: float) -> tuple[float, float]:
    """Return (xmin, xmax) for a named region; open xmax uses ``t_max``."""
    for xmin, xmax, lab, _ in _TEST_REGIONS:
        if lab == label:
            x1 = xmax if xmax is not None else t_max
            return (float(xmin), float(x1))
    raise ValueError(f"No region named {label!r} in _TEST_REGIONS")


def _annotate_test_regions(ax: plt.Axes, t: np.ndarray) -> None:
    """Shade named test regions and label them along the top of the axes."""
    t_max = float(np.nanmax(t)) if t.size else 0.0
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for xmin, xmax, label, color in _TEST_REGIONS:
        x1 = xmax if xmax is not None else t_max
        ax.axvspan(xmin, x1, facecolor=color, alpha=0.35, zorder=0, linewidth=0)
        mid = 0.5 * (xmin + x1)
        ax.text(
            mid,
            0.98,
            label,
            transform=trans,
            ha="center",
            va="top",
            fontsize=9,
            color="#333333",
            zorder=3,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.65),
        )


def _css_raw_sun_sensor_region_threshold_y(label: str) -> float | None:
    """Reference level for dashed lines on raw sun sensor plots (per test region)."""
    if label in ("setup", "heliostat"):
        return 300.0
    if label in ("axis change", "flood light"):
        return 10.0
    return None


def _annotate_css_raw_sun_sensor_thresholds(ax: plt.Axes, t: np.ndarray) -> None:
    """
    Per shaded region: dashed red horizontal line at the region's reference level
    (300 for setup/heliostat, 10 for axis change/flood light) and a small arrow
    pointing upward from the line.
    """
    t_max = float(np.nanmax(t)) if t.size else 0.0
    ylo, yhi = ax.get_ylim()
    span = max(yhi - ylo, 1e-9)
    dy = 0.02 * span
    for xmin, xmax, label, _ in _TEST_REGIONS:
        y_thr = _css_raw_sun_sensor_region_threshold_y(label)
        if y_thr is None:
            continue
        x1 = xmax if xmax is not None else t_max
        ax.plot(
            [xmin, x1],
            [y_thr, y_thr],
            color="red",
            linestyle="--",
            linewidth=1.0,
            zorder=3,
        )
        mid = 0.5 * (xmin + x1)
        ax.annotate(
            "",
            xy=(mid, y_thr + dy),
            xytext=(mid, y_thr),
            arrowprops=dict(
                arrowstyle="->",
                color="red",
                lw=0.8,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=3,
        )
    # Keep arrowheads inside the y-axis limits
    ylo, yhi = ax.get_ylim()
    span = max(yhi - ylo, 1e-9)
    dy2 = 0.02 * span
    tips: list[float] = []
    for xmin, xmax, label, _ in _TEST_REGIONS:
        y_thr = _css_raw_sun_sensor_region_threshold_y(label)
        if y_thr is None:
            continue
        tips.append(y_thr + dy2)
    if tips:
        ax.set_ylim(ylo, max(yhi, max(tips)))


def _default_h5_path() -> Path:
    root = os.environ.get("suncet_data")
    if not root:
        raise RuntimeError("Environment variable suncet_data is not set.")
    return (
        Path(root)
        / "test_data"
        / "2026-02-27_air_bearing_realtime"
        / "level0_5"
        / "suncet_telemetry_mission_length_v1.0.1-test_air_bearing.h5"
    )


def load_time_axis(f: h5py.File) -> tuple[np.ndarray, str]:
    """Return raw time array from the first usable timestamp dataset."""
    for name in TS_CANDIDATES:
        if name not in f:
            continue
        ts = np.asarray(f[name][:], dtype=float)
        if np.all(ts == 0) or np.all(~np.isfinite(ts)):
            continue
        return ts, name

    raise KeyError(f"No usable timestamp among {TS_CANDIDATES}")


def _h5_dataset_keys(h5: h5py.File) -> list[str]:
    names: list[str] = []

    def _visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset):
            names.append(name)

    h5.visititems(_visitor)
    return names


def _adcs_css_plot_keys(dataset_keys: Iterable[str]) -> list[str]:
    """Ordered list of ADCS CSS HDF5 datasets to plot (must exist in file)."""
    dk = set(dataset_keys)
    nums = sorted(
        (k for k in dk if k.startswith("adcs_css_num_diodes_used_")),
        key=lambda s: int(s.rsplit("_", 1)[-1]),
    )
    # Per-channel raw plots: channels 0–7 only (not _8 … _15).
    raws = sorted(
        (
            k
            for k in dk
            if k.startswith("adcs_css_raw_sun_sensor_data_")
            and int(k.rsplit("_", 1)[-1]) < 8
        ),
        key=lambda s: int(s.rsplit("_", 1)[-1]),
    )
    tail = [k for k in ("adcs_css_sun_sensor_used", "adcs_css_meas_sun_vld") if k in dk]
    return list(nums) + list(raws) + tail


def _plot_adcs_css_field(
    path: Path,
    key: str,
    t: np.ndarray,
    y: np.ndarray,
    out_path: Path | None,
) -> None:
    if len(t) != len(y):
        raise ValueError(f"{key}: time length {len(t)} != data length {len(y)}")
    ok = np.isfinite(t) & np.isfinite(y)
    t_p, y_p = t[ok], y[ok]
    fig, ax = plt.subplots(figsize=(10, 4))
    _annotate_test_regions(ax, t_p)
    ax.plot(
        t_p,
        y_p,
        linestyle="none",
        marker="o",
        markersize=2,
        markeredgewidth=0.3,
        zorder=2,
    )
    if y_p.size == 0:
        ax.set_ylim(0.0, 1.0)
        ax.text(0.5, 0.5, "No finite (time, value) pairs", transform=ax.transAxes, ha="center")
    else:
        ymin, ymax = float(np.nanmin(y_p)), float(np.nanmax(y_p))
        rng = max(ymax - ymin, 1e-6)
        ax.set_ylim(ymin - 0.05 * rng, ymax + 0.14 * rng)
        if key.startswith("adcs_css_raw_sun_sensor_data_"):
            thr: list[float] = []
            t_lo, t_hi = float(np.nanmin(t_p)), float(np.nanmax(t_p))
            t_max_r = t_hi
            for xmin, xmax, lab, _ in _TEST_REGIONS:
                x1 = xmax if xmax is not None else t_max_r
                if x1 < t_lo or xmin > t_hi:
                    continue
                y_thr = _css_raw_sun_sensor_region_threshold_y(lab)
                if y_thr is not None:
                    thr.append(y_thr)
            if thr:
                ylo, yhi = ax.get_ylim()
                ax.set_ylim(min(ylo, min(thr)), max(yhi, max(thr)))
            _annotate_css_raw_sun_sensor_thresholds(ax, t_p)
    ax.set_xlabel("secondary header seconds")
    ax.set_ylabel(key)
    ax.set_title(f"{path.name}\n{key} (adcs_css)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")
        plt.close(fig)
    # Interactive: leave figure open; caller calls plt.show() once at end


def _plot_raw_sun_sensor_overlay_window(
    path: Path,
    t: np.ndarray,
    css_series: dict[str, np.ndarray],
    x0: float,
    x1: float,
    title_subtitle: str,
    empty_message: str,
    out_path: Path | None,
) -> None:
    """Overplot raw_sun_sensor_data_0 … _7 for ``x0`` ≤ t ≤ ``x1`` (fixed colors)."""
    keys = [f"adcs_css_raw_sun_sensor_data_{i}" for i in range(8)]
    if not all(k in css_series for k in keys):
        print(
            "Skipping raw sun sensor overlay: missing one or more of "
            + ", ".join(keys)
        )
        return

    colors = _FLOOD_LIGHT_RAW_OVERLAY_COLORS
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axvspan(x0, x1, facecolor=_REGION_YELLOW, alpha=0.2, zorder=0, linewidth=0)
    all_y: list[float] = []
    for i, key in enumerate(keys):
        y = css_series[key]
        ok = np.isfinite(t) & np.isfinite(y) & (t >= x0) & (t <= x1)
        t_w, y_w = t[ok], y[ok]
        if t_w.size:
            all_y.extend((float(y_w.min()), float(y_w.max())))
        ax.plot(
            t_w,
            y_w,
            linestyle="none",
            marker="o",
            markersize=2,
            markeredgewidth=0.3,
            color=colors[i],
            label=f"{i}",
            zorder=2,
        )
    ax.axhline(10.0, color="red", linestyle="--", linewidth=1.0, zorder=3)
    ax.set_xlim(x0, x1)
    if not all_y:
        ax.set_ylim(0.0, 1.0)
        ax.text(
            0.5,
            0.5,
            empty_message,
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    else:
        ymin, ymax = min(all_y), max(all_y)
        rng = max(ymax - ymin, 1e-6)
        ax.set_ylim(ymin - 0.05 * rng, ymax + 0.14 * rng)
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(min(ylo, 10.0), max(yhi, 10.0))
    ax.set_xlabel("secondary header seconds")
    ax.set_ylabel("adcs_css_raw_sun_sensor_data (0–7)")
    ax.set_title(f"{path.name}\n{title_subtitle}")
    ax.legend(title="channel", ncol=4, fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")
        plt.close(fig)


def _plot_flood_light_raw_sun_sensor_overlay(
    path: Path,
    t: np.ndarray,
    css_series: dict[str, np.ndarray],
    t_max: float,
    out_path: Path | None,
) -> None:
    """Full flood-light phase window."""
    fx0, fx1 = _region_x_bounds("flood light", t_max)
    _plot_raw_sun_sensor_overlay_window(
        path,
        t,
        css_series,
        fx0,
        fx1,
        "Flood light — raw sun sensor 0–7 overlay",
        "No finite samples in flood light window",
        out_path,
    )


def _plot_flood_light_raw_sun_sensor_overlay_zoom(
    path: Path,
    t: np.ndarray,
    css_series: dict[str, np.ndarray],
    out_path: Path | None,
) -> None:
    """Flood-style overlay zoomed to [1.7, 1.8]×10¹² s."""
    _plot_raw_sun_sensor_overlay_window(
        path,
        t,
        css_series,
        _FLOOD_LIGHT_RAW_ZOOM_X0,
        _FLOOD_LIGHT_RAW_ZOOM_X1,
        "Flood light zoom [1.7, 1.8]×10¹² s — raw sun sensor 0–7 overlay",
        "No finite samples in [1.7, 1.8]×10¹² s window",
        out_path,
    )


def _plot_beacon_wheel_speeds(
    path: Path,
    t: np.ndarray,
    wheel: tuple[np.ndarray, np.ndarray, np.ndarray],
    out_path: Path | None,
) -> None:
    """Whole mission: ``beac_adcs_wheel_sp1`` … ``sp3`` vs time, shaded test regions."""
    w1, w2, w3 = wheel
    for i, arr in enumerate((w1, w2, w3), start=1):
        if len(arr) != len(t):
            raise ValueError(f"beac_adcs_wheel_sp{i}: length {len(arr)} != time length {len(t)}")

    fig, ax = plt.subplots(figsize=(10, 4))
    t_fin = t[np.isfinite(t)]
    _annotate_test_regions(ax, t_fin if t_fin.size else t)

    labels = ("beac_adcs_wheel_sp1", "beac_adcs_wheel_sp2", "beac_adcs_wheel_sp3")
    y_parts: list[np.ndarray] = []
    for color, y, lab in zip(_BEACON_WHEEL_LINE_COLORS, (w1, w2, w3), labels):
        ok = np.isfinite(t) & np.isfinite(y)
        ax.plot(t[ok], y[ok], color=color, linewidth=1.0, label=lab, zorder=2)
        y_parts.append(y[ok])

    y_all = np.concatenate(y_parts) if y_parts else np.array([])
    if y_all.size == 0:
        ax.set_ylim(-9000.0, 9000.0)
        ax.text(
            0.5,
            0.5,
            "No finite beacon wheel speed samples",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    else:
        ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
        rng = max(ymax - ymin, 1e-9)
        y_lo = ymin - 0.05 * rng
        y_hi = ymax + 0.14 * rng
        # Include ±8000 RPM reference band in view
        y_lo = min(y_lo, -_BEACON_WHEEL_REF_RPM_DARK - 0.05 * max(rng, 16000.0))
        y_hi = max(y_hi, _BEACON_WHEEL_REF_RPM_DARK + 0.05 * max(rng, 16000.0))
        ax.set_ylim(y_lo, y_hi)

    trans_r = blended_transform_factory(ax.transAxes, ax.transData)
    for y_ref in (_BEACON_WHEEL_REF_RPM_LIGHT, -_BEACON_WHEEL_REF_RPM_LIGHT):
        ax.axhline(
            y_ref,
            color=_REF_LINE_GREY_LIGHT,
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )
        ax.text(
            0.99,
            y_ref,
            f" {_LABEL_WHEEL_MOMENTUM}",
            transform=trans_r,
            ha="right",
            va="bottom" if y_ref > 0 else "top",
            fontsize=7,
            color=_REF_LINE_GREY_LIGHT,
            zorder=3,
        )
    for y_ref in (_BEACON_WHEEL_REF_RPM_DARK, -_BEACON_WHEEL_REF_RPM_DARK):
        ax.axhline(
            y_ref,
            color=_REF_LINE_GREY_DARK,
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )
        ax.text(
            0.99,
            y_ref,
            f" {_LABEL_WHEEL_SATURATION}",
            transform=trans_r,
            ha="right",
            va="bottom" if y_ref > 0 else "top",
            fontsize=7,
            color=_REF_LINE_GREY_DARK,
            zorder=3,
        )

    ax.set_xlabel("secondary header seconds")
    ax.set_ylabel("Beacon wheel speed [RPM]")
    ax.set_title(f"{path.name}\nBeacon wheel speeds (beac_adcs_wheel_sp1–sp3)")
    leg_h, _ = ax.get_legend_handles_labels()
    if leg_h:
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")
        plt.close(fig)


def _plot_wheel_momentum_body(
    path: Path,
    t: np.ndarray,
    wheel_mom: tuple[np.ndarray, np.ndarray, np.ndarray],
    out_path: Path | None,
    *,
    x0: float | None = None,
    x1: float | None = None,
) -> None:
    """Whole mission: ``adcs_mom_wheel_mom_body_0`` … ``_2`` vs time, shaded test regions."""
    m0, m1, m2 = wheel_mom
    for k, arr in zip(_WHEEL_MOM_BODY_KEYS, (m0, m1, m2)):
        if len(arr) != len(t):
            raise ValueError(f"{k}: length {len(arr)} != time length {len(t)}")

    fig, ax = plt.subplots(figsize=(10, 4))
    t_fin = t[np.isfinite(t)]
    _annotate_test_regions(ax, t_fin if t_fin.size else t)

    y_parts: list[np.ndarray] = []
    for color, y, lab in zip(_WHEEL_MOM_LINE_COLORS, (m0, m1, m2), _WHEEL_MOM_BODY_KEYS):
        ok = np.isfinite(t) & np.isfinite(y)
        if x0 is not None:
            ok &= t >= x0
        if x1 is not None:
            ok &= t <= x1
        ax.plot(t[ok], y[ok], color=color, linewidth=1.0, label=lab, zorder=2)
        y_parts.append(y[ok])

    y_all = np.concatenate(y_parts) if y_parts else np.array([])
    if y_all.size == 0:
        ax.set_ylim(-1.0, 1.0)
        ax.text(
            0.5,
            0.5,
            "No finite wheel momentum samples",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    else:
        ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
        rng = max(ymax - ymin, 1e-9)
        ax.set_ylim(ymin - 0.05 * rng, ymax + 0.14 * rng)

    if x0 is not None:
        if x1 is not None:
            ax.set_xlim(float(x0), float(x1))
        else:
            ax.set_xlim(float(x0), float(np.nanmax(t)) if t.size else float(x0))

    # Max wheel momentum storage reference lines.
    trans_r = blended_transform_factory(ax.transAxes, ax.transData)
    for y_ref in (_WHEEL_MOM_MAX_STORAGE_NMS, -_WHEEL_MOM_MAX_STORAGE_NMS):
        ax.axhline(
            y_ref,
            color=_REF_LINE_GREY_DARK,
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )
        ax.text(
            0.99,
            y_ref,
            f" {_LABEL_WHEEL_MOM_MAX_STORAGE}",
            transform=trans_r,
            ha="right",
            va="bottom" if y_ref > 0 else "top",
            fontsize=7,
            color=_REF_LINE_GREY_DARK,
            zorder=3,
        )

    ax.set_xlabel("secondary header seconds")
    ax.set_ylabel("Wheel momentum (body axes) [Nms]")
    ttl = "Wheel momentum (adcs_mom_wheel_mom_body_0–2)"
    if x0 is not None and x1 is not None:
        ttl = f"{ttl} (zoom x∈[{x0:.3g}, {x1:.3g}])"
    elif x0 is not None:
        ttl = f"{ttl} (zoom x≥{x0:.3g})"
    ax.set_title(f"{path.name}\n{ttl}")
    leg_h, _ = ax.get_legend_handles_labels()
    if leg_h:
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")
        plt.close(fig)


def _plot_trq_rod_duty_cycles(
    path: Path,
    t: np.ndarray,
    duty: tuple[np.ndarray, np.ndarray, np.ndarray],
    out_path: Path | None,
) -> None:
    """Whole mission: ``adcs_mom_duty_cyc_0`` … ``_2`` vs time, shaded test regions."""
    d0, d1, d2 = duty
    for k, arr in zip(_TRQ_ROD_DUTY_CYC_KEYS, (d0, d1, d2)):
        if len(arr) != len(t):
            raise ValueError(f"{k}: length {len(arr)} != time length {len(t)}")

    fig, ax = plt.subplots(figsize=(10, 4))
    t_fin = t[np.isfinite(t)]
    _annotate_test_regions(ax, t_fin if t_fin.size else t)

    y_parts: list[np.ndarray] = []
    for color, y, lab in zip(_TRQ_ROD_DUTY_LINE_COLORS, (d0, d1, d2), _TRQ_ROD_DUTY_CYC_KEYS):
        ok = np.isfinite(t) & np.isfinite(y)
        ax.plot(t[ok], y[ok], color=color, linewidth=1.0, label=lab, zorder=2)
        y_parts.append(y[ok])

    y_all = np.concatenate(y_parts) if y_parts else np.array([])
    if y_all.size == 0:
        ax.set_ylim(0.0, 100.0)
        ax.text(
            0.5,
            0.5,
            "No finite torque-rod duty cycle samples",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    else:
        ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
        rng = max(ymax - ymin, 1e-9)
        y_lo = min(0.0, ymin - 0.05 * rng)
        y_hi = max(100.0, ymax + 0.14 * rng) if ymax > 1.0 else (ymax + 0.14 * rng)
        ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel("secondary header seconds")
    ax.set_ylabel("Torque-rod duty cycle")
    ax.set_title(f"{path.name}\nTorque-rod duty cycles (adcs_mom_duty_cyc_0–2)")
    leg_h, _ = ax.get_legend_handles_labels()
    if leg_h:
        ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")
        plt.close(fig)


def _plot_wheel_momentum_zoom_stack_with_sun_point_err(
    path: Path,
    t: np.ndarray,
    wheel_mom: tuple[np.ndarray, np.ndarray, np.ndarray],
    sun_point_err: np.ndarray,
    out_path: Path | None,
    *,
    x0: float,
    x1: float,
) -> None:
    """Zoomed stacked plot: wheel momentum (top) + sun point error (bottom)."""
    m0, m1, m2 = wheel_mom
    for k, arr in zip(_WHEEL_MOM_BODY_KEYS, (m0, m1, m2)):
        if len(arr) != len(t):
            raise ValueError(f"{k}: length {len(arr)} != time length {len(t)}")
    if len(sun_point_err) != len(t):
        raise ValueError(
            f"{SUN_POINT_ANGLE_ERR}: length {len(sun_point_err)} != time length {len(t)}"
        )

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(10, 6),
        gridspec_kw={"height_ratios": (2.0, 1.3)},
    )
    # Single shared time axis: no vertical gap; x labels only on bottom panel.
    ax0.tick_params(axis="x", which="both", labelbottom=False)

    t_fin = t[np.isfinite(t)]
    for ax in (ax0, ax1):
        _annotate_test_regions(ax, t_fin if t_fin.size else t)
        ax.set_xlim(float(x0), float(x1))

    # Top: wheel momentum overlay.
    y_parts: list[np.ndarray] = []
    for color, y, lab in zip(_WHEEL_MOM_LINE_COLORS, (m0, m1, m2), _WHEEL_MOM_BODY_KEYS):
        ok = np.isfinite(t) & np.isfinite(y) & (t >= x0) & (t <= x1)
        ax0.plot(t[ok], y[ok], color=color, linewidth=1.0, label=lab, zorder=2)
        y_parts.append(y[ok])

    y_all = np.concatenate(y_parts) if y_parts else np.array([])
    if y_all.size:
        ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
        rng = max(ymax - ymin, 1e-9)
        ax0.set_ylim(ymin - 0.05 * rng, ymax + 0.14 * rng)

    trans_r = blended_transform_factory(ax0.transAxes, ax0.transData)
    for y_ref in (_WHEEL_MOM_MAX_STORAGE_NMS, -_WHEEL_MOM_MAX_STORAGE_NMS):
        ax0.axhline(
            y_ref,
            color=_REF_LINE_GREY_DARK,
            linestyle="--",
            linewidth=1.0,
            zorder=1,
        )
        ax0.text(
            0.99,
            y_ref,
            f" {_LABEL_WHEEL_MOM_MAX_STORAGE}",
            transform=trans_r,
            ha="right",
            va="bottom" if y_ref > 0 else "top",
            fontsize=7,
            color=_REF_LINE_GREY_DARK,
            zorder=3,
        )

    ax0.set_ylabel("Wheel momentum (body axes) [Nms]")
    leg_h, _ = ax0.get_legend_handles_labels()
    if leg_h:
        ax0.legend(loc="best", fontsize=8)
    ax0.grid(True, alpha=0.3)

    # Bottom: sun point angle error.
    ok1 = np.isfinite(t) & np.isfinite(sun_point_err) & (t >= x0) & (t <= x1)
    ax1.plot(t[ok1], sun_point_err[ok1], color="black", linewidth=1.0, zorder=2)
    if np.any(ok1):
        ymin, ymax = float(np.min(sun_point_err[ok1])), float(np.max(sun_point_err[ok1]))
        rng = max(ymax - ymin, 1e-9)
        ax1.set_ylim(ymin - 0.05 * rng, ymax + 0.14 * rng)
    ax1.set_xlabel("secondary header seconds")
    ax1.set_ylabel("Sun point angle error [º]")
    ax1.grid(True, alpha=0.3)

    fig.suptitle(
        f"{path.name}\nWheel momentum + sun point error (zoom x∈[{x0:.3g}, {x1:.3g}])",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=0)
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")
        plt.close(fig)


def _plot_total_momentum_mag(
    path: Path,
    t: np.ndarray,
    total_mom_mag: np.ndarray,
    out_path: Path | None,
) -> None:
    """Whole mission: ``adcs_mom_total_mom_mag`` vs time, shaded test regions."""
    if len(total_mom_mag) != len(t):
        raise ValueError(
            f"{_TOTAL_MOM_MAG_KEY}: length {len(total_mom_mag)} != time length {len(t)}"
        )

    ok = np.isfinite(t) & np.isfinite(total_mom_mag)
    fig, ax = plt.subplots(figsize=(10, 4))
    t_fin = t[np.isfinite(t)]
    _annotate_test_regions(ax, t_fin if t_fin.size else t)
    ax.plot(t[ok], total_mom_mag[ok], color="black", linewidth=1.0, zorder=2)

    if np.any(ok):
        ymin, ymax = float(np.min(total_mom_mag[ok])), float(np.max(total_mom_mag[ok]))
        rng = max(ymax - ymin, 1e-9)
        ax.set_ylim(ymin - 0.05 * rng, ymax + 0.14 * rng)
    else:
        ax.set_ylim(0.0, 1.0)
        ax.text(
            0.5,
            0.5,
            "No finite total momentum magnitude samples",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )

    ax.set_xlabel("secondary header seconds")
    ax.set_ylabel(f"{_TOTAL_MOM_MAG_KEY} [Nms]")
    ax.set_title(f"{path.name}\nTotal momentum magnitude ({_TOTAL_MOM_MAG_KEY})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        print(f"Wrote {out_path}")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "h5_path",
        nargs="?",
        type=Path,
        default=None,
        help="Telemetry HDF5 (default: air bearing file under $suncet_data)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save figure to this path instead of only showing interactively",
    )
    args = parser.parse_args()
    path = args.h5_path.expanduser() if args.h5_path else _default_h5_path()
    if not path.is_file():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        if SUN_POINT_ANGLE_ERR not in f:
            raise KeyError(
                f"Missing {SUN_POINT_ANGLE_ERR!r}; available keys sample: "
                f"{list(f.keys())[:20]} ..."
            )
        err = np.asarray(f[SUN_POINT_ANGLE_ERR][:], dtype=float)
        t, _ = load_time_axis(f)
        css_keys = _adcs_css_plot_keys(_h5_dataset_keys(f))
        css_series = {k: np.asarray(f[k][:], dtype=float) for k in css_keys}
        beacon_wheel: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        miss_bw = [k for k in _BEACON_WHEEL_SP_KEYS if k not in f]
        if miss_bw:
            print(
                "Skipping beacon wheel speed plot: missing dataset(s): "
                + ", ".join(miss_bw)
            )
        else:
            beacon_wheel = tuple(
                np.asarray(f[k][:], dtype=float) for k in _BEACON_WHEEL_SP_KEYS
            )

        wheel_mom: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        miss_wm = [k for k in _WHEEL_MOM_BODY_KEYS if k not in f]
        if miss_wm:
            print(
                "Skipping wheel momentum plot: missing dataset(s): " + ", ".join(miss_wm)
            )
        else:
            wheel_mom = tuple(np.asarray(f[k][:], dtype=float) for k in _WHEEL_MOM_BODY_KEYS)

        total_mom_mag: np.ndarray | None = None
        if _TOTAL_MOM_MAG_KEY not in f:
            print(f"Skipping total momentum magnitude plot: missing {_TOTAL_MOM_MAG_KEY}")
        else:
            total_mom_mag = np.asarray(f[_TOTAL_MOM_MAG_KEY][:], dtype=float)
            if _looks_like_uint32_packed_float(total_mom_mag):
                total_mom_mag = _reinterpret_u32_counts_as_f32(total_mom_mag)

        trq_rod_duty: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        miss_dc = [k for k in _TRQ_ROD_DUTY_CYC_KEYS if k not in f]
        if miss_dc:
            print(
                "Skipping torque-rod duty cycle plot: missing dataset(s): "
                + ", ".join(miss_dc)
            )
        else:
            trq_rod_duty = tuple(
                np.asarray(f[k][:], dtype=float) for k in _TRQ_ROD_DUTY_CYC_KEYS
            )

    if len(t) != len(err):
        raise ValueError(f"Time length {len(t)} != error length {len(err)}")

    ok_plot = np.isfinite(t) & np.isfinite(err)
    t_plot = t[ok_plot]
    err_plot = err[ok_plot]

    fig, ax = plt.subplots(figsize=(10, 4))
    _annotate_test_regions(ax, t_plot)
    ax.plot(
        t_plot,
        err_plot,
        linestyle="none",
        marker="o",
        markersize=2,
        markeredgewidth=0.3,
        zorder=2,
    )
    ymin, ymax = float(np.nanmin(err_plot)), float(np.nanmax(err_plot))
    rng = max(ymax - ymin, 1e-6)
    ax.set_ylim(ymin - 0.05 * rng, ymax + 0.14 * rng)
    ax.set_xlabel("secondary header seconds")
    ax.set_ylabel("Sun point angle error (ADCS att ctrl) [º]")
    ax.set_title(path.name)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    hx0, hx1 = _region_bounds_by_label("heliostat")
    helio = ok_plot & (t >= hx0) & (t <= hx1)
    err_h = err[helio]
    fig_h, ax_h = plt.subplots(figsize=(10, 4))
    ax_h.axvspan(hx0, hx1, facecolor=_REGION_GREEN, alpha=0.25, zorder=0, linewidth=0)
    ax_h.plot(
        t[helio],
        err[helio],
        linestyle="none",
        marker="o",
        markersize=2,
        markeredgewidth=0.3,
        zorder=2,
    )
    ax_h.set_xlim(hx0, hx1)
    ax_h.set_ylim(0.0, _HELIOSTAT_PLOT_YMAX)
    ax_h.set_xlabel("secondary header seconds")
    ax_h.set_ylabel("Sun point angle error (ADCS att ctrl) [º]")
    ax_h.set_title(f"{path.name} (heliostat)")
    ax_h.grid(True, alpha=0.3)
    ax_h.text(
        0.02,
        0.98,
        _heliostat_stats_text(err_h),
        transform=ax_h.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#888888", alpha=0.92),
        zorder=4,
    )
    fig_h.tight_layout()

    t_max_mission = float(np.nanmax(t)) if t.size else 0.0
    out_flood: Path | None = None
    if args.output is not None:
        out_flood = args.output.with_name(
            args.output.stem + "_flood_light_raw0_7" + args.output.suffix
        )
    _plot_flood_light_raw_sun_sensor_overlay(path, t, css_series, t_max_mission, out_flood)

    out_flood_zoom: Path | None = None
    if args.output is not None:
        out_flood_zoom = args.output.with_name(
            args.output.stem + "_flood_light_raw0_7_zoom" + args.output.suffix
        )
    _plot_flood_light_raw_sun_sensor_overlay_zoom(path, t, css_series, out_flood_zoom)

    out_bw: Path | None = None
    if args.output is not None:
        out_bw = args.output.with_name(
            args.output.stem + "_beacon_wheel_speeds" + args.output.suffix
        )
    if beacon_wheel is not None:
        _plot_beacon_wheel_speeds(path, t, beacon_wheel, out_bw)

    out_wm: Path | None = None
    if args.output is not None:
        out_wm = args.output.with_name(args.output.stem + "_wheel_mom_body" + args.output.suffix)
    if wheel_mom is not None:
        _plot_wheel_momentum_body(path, t, wheel_mom, out_wm)

    out_wm_zoom: Path | None = None
    if args.output is not None:
        out_wm_zoom = args.output.with_name(
            args.output.stem + "_wheel_mom_body_zoom" + args.output.suffix
        )
    if wheel_mom is not None:
        _plot_wheel_momentum_body(
            path,
            t,
            wheel_mom,
            out_wm_zoom,
            x0=_WHEEL_MOM_ZOOM_X0,
            x1=_WHEEL_MOM_ZOOM_X1,
        )

    out_tm: Path | None = None
    if args.output is not None:
        out_tm = args.output.with_name(
            args.output.stem + "_total_mom_mag" + args.output.suffix
        )
    if total_mom_mag is not None:
        _plot_total_momentum_mag(path, t, total_mom_mag, out_tm)

    out_dc: Path | None = None
    if args.output is not None:
        out_dc = args.output.with_name(
            args.output.stem + "_trq_rod_duty_cyc" + args.output.suffix
        )
    if trq_rod_duty is not None:
        _plot_trq_rod_duty_cycles(path, t, trq_rod_duty, out_dc)

    out_wm_zoom_stack: Path | None = None
    if args.output is not None:
        out_wm_zoom_stack = args.output.with_name(
            args.output.stem + "_wheel_mom_body_zoom_stack_err" + args.output.suffix
        )
    if wheel_mom is not None:
        _plot_wheel_momentum_zoom_stack_with_sun_point_err(
            path,
            t,
            wheel_mom,
            err,
            out_wm_zoom_stack,
            x0=_WHEEL_MOM_ZOOM_X0,
            x1=_WHEEL_MOM_ZOOM_X1,
        )

    for key in css_keys:
        out_css: Path | None = None
        if args.output is not None:
            short = key.removeprefix("adcs_css_").replace("/", "_")
            out_css = args.output.with_name(
                f"{args.output.stem}_css_{short}{args.output.suffix}"
            )
        _plot_adcs_css_field(path, key, t, css_series[key], out_css)

    if args.output:
        fig.savefig(args.output, dpi=150)
        out_h = args.output.with_name(args.output.stem + "_heliostat" + args.output.suffix)
        fig_h.savefig(out_h, dpi=150)
        print(f"Wrote {args.output}")
        print(f"Wrote {out_h}")
        plt.close(fig)
        plt.close(fig_h)
    else:
        plt.show()


if __name__ == "__main__":
    main()
