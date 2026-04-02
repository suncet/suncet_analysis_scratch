"""
Air bearing test: ADCS attitude-control sun point angle error vs time.

Reads level-0.5 mission-length telemetry HDF5 and plots
``adcs_att_ctrl_sun_point_ang_err`` versus the raw time column from the file
(``pktTimestamp`` or ``timestamp_seconds_since_boot``, whichever is present).
Rows where the angle error is NaN or non-finite are dropped from the plot.

Also plots ADCS CSS packet fields
(``adcs_css_num_diodes_used_*``, ``adcs_css_raw_sun_sensor_data_*``,
``adcs_css_sun_sensor_used``, ``adcs_css_meas_sun_vld``) versus the same time
axis, one figure per channel.
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


def _default_h5_path() -> Path:
    root = os.environ.get("suncet_data")
    if not root:
        raise RuntimeError("Environment variable suncet_data is not set.")
    return (
        Path(root)
        / "test_data"
        / "2026-03-09_air_bearing_realtime"
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
    raws = sorted(
        (k for k in dk if k.startswith("adcs_css_raw_sun_sensor_data_")),
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
