"""
Create quick-look beacon time-series plots from decoded packet CSVs.

This is intentionally downstream of ``make_level0_5.py``: it reads the
per-APID decoded CSV files and makes easy sanity-check plots without re-ingesting binary
data.
"""

from __future__ import annotations

import argparse
import os
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from suncet_processing_pipeline.config_parser import Config
from suncet_processing_pipeline.make_level0_5 import (
    DEFAULT_DECODED_DIR_BASENAME,
    resolve_config_data_folder,
)


J2000_UTC_EPOCH = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
DEFAULT_MIN_PLOT_TIME_UTC = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
DEFAULT_BEACON_CSV = "decoded_apid_0001_beacon.csv"
DEFAULT_CSIE_HK_CSV = "decoded_apid_0537_csie_hk.csv"
DEFAULT_DSPS_CSV = "decoded_apid_0035_dsps_data.csv"
DEFAULT_PLOT_DIR = "sanity_plots"
DEFAULT_TIME_COLUMN = None

# Leap seconds inserted after the J2000 UTC epoch. Bulletin C 71 says no leap second
# will be introduced at the end of June 2026, so this table is current for this test.
LEAP_SECOND_EFFECTIVE_UTC = [
    datetime(2006, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2009, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2012, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2015, 7, 1, 0, 0, 0, tzinfo=timezone.utc),
    datetime(2017, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
]

NUMERIC_BEACON_FIELDS = [
    "beac_ana_bat1_v",
    "beac_ana_bat2_v",
    "beac_ana_cdh_temp",
    "beac_ana_csie_curr",
    "beac_ana_csie_power",
    "beac_ana_eps_bus_i",
    "beac_ana_eps_bus_v",
    "beac_ana_xband_v",
    "beac_batt1_charge_current",
    "beac_batt1_temp",
    "beac_dsps_sensor_board_temp",
]

CATEGORICAL_BEACON_FIELDS = [
    "beac_uhf_alive",
]

CSIE_HK_NUMERIC_FIELDS = [
    "csie_det0_therm",
]

DSPS_QUAD_SUM_FIELDS = [
    "dsps_visible_sps_sum",
    "dsps_x_ray_sps_sum",
]

DSPS_VISIBLE_DIODE_FIELDS = [
    "dsps_visible_sps_diodes_0",
    "dsps_visible_sps_diodes_1",
    "dsps_visible_sps_diodes_2",
    "dsps_visible_sps_diodes_3",
]

DSPS_X_RAY_DIODE_FIELDS = [
    "dsps_x_ray_sps_diodes_0",
    "dsps_x_ray_sps_diodes_1",
    "dsps_x_ray_sps_diodes_2",
    "dsps_x_ray_sps_diodes_3",
]

PLOT_VALUE_LIMITS = {
    "beac_ana_bat1_v": {"min": 11.0, "max": 17.0},
    "beac_ana_bat2_v": {"min": 11.0, "max": 17.0},
    "beac_ana_cdh_temp": {"min": -100.0, "max": 100.0},
    "beac_ana_csie_curr": {"min": 0.0, "max": 5.0},
    "beac_ana_csie_volt": {"min": 0.0, "max": 17.0},
    "beac_ana_csie_power": {"min": 3.0},
    "beac_ana_eps_bus_v": {"sigma": 3.0},
    "beac_batt1_charge_current": {"sigma": 3.0},
    "beac_batt1_temp": {"sigma": 3.0},
    "beac_dsps_sensor_board_temp": {"sigma": 3.0},
    "csie_det0_therm": {"sigma": 3.0},
}

DERIVED_NUMERIC_FIELDS = {
    "beac_ana_csie_power": ("beac_ana_csie_volt", "beac_ana_csie_curr"),
}

SMOOTHED_NUMERIC_FIELDS = {
    "beac_ana_csie_power",
}


def _apply_numeric_limits(values: pd.Series, field: str) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    limits = PLOT_VALUE_LIMITS.get(field, {})
    if "min" in limits:
        values = values.where(values >= limits["min"])
    if "max" in limits:
        values = values.where(values <= limits["max"])
    if limits.get("drop_zero"):
        values = values.where(values != 0)
    return values


def leap_seconds_after_j2000(seconds: float) -> int:
    """Return leap seconds to add for an onboard no-leap J2000 second count."""
    leap_count = 0
    while True:
        corrected_time = J2000_UTC_EPOCH + timedelta(seconds=float(seconds) + leap_count)
        new_count = sum(
            1 for leap_time in LEAP_SECOND_EFFECTIVE_UTC if corrected_time >= leap_time
        )
        if new_count == leap_count:
            return leap_count
        leap_count = new_count


def j2000_seconds_to_datetime(
    seconds,
    *,
    add_leap_seconds: bool = True,
) -> datetime | None:
    if pd.isna(seconds):
        return None
    seconds_float = float(seconds)
    if add_leap_seconds:
        seconds_float += leap_seconds_after_j2000(seconds_float)
    return J2000_UTC_EPOCH + timedelta(seconds=seconds_float)


def parse_utc_datetime(value: str) -> datetime:
    """Parse an ISO-8601-ish UTC datetime string for plot filtering."""
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def find_time_column(df: pd.DataFrame, requested_time_column: str | None = None) -> str:
    if requested_time_column:
        if requested_time_column not in df.columns:
            raise KeyError(f"Requested time column not found: {requested_time_column}")
        return requested_time_column
    columns = [column for column in df.columns if column.startswith("ccsdsSecHeader2_sec")]
    if columns:
        if len(columns) == 1:
            return columns[0]
        beacon_columns = [column for column in columns if "beacon" in column.lower()]
        return beacon_columns[0] if beacon_columns else columns[0]
    if "SHCOARSE" in df.columns:
        return "SHCOARSE"
    raise KeyError("No ccsdsSecHeader2_sec* or SHCOARSE time column found.")


def prepare_beacon_dataframe(
    path: Path,
    *,
    time_column: str | None = DEFAULT_TIME_COLUMN,
    add_leap_seconds: bool = True,
) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    time_column = find_time_column(df, time_column)
    df["plot_time_utc"] = df[time_column].map(
        lambda seconds: j2000_seconds_to_datetime(
            seconds,
            add_leap_seconds=add_leap_seconds,
        )
    )
    if add_leap_seconds:
        df["plot_leap_seconds_added"] = df[time_column].map(
            lambda seconds: None
            if pd.isna(seconds)
            else leap_seconds_after_j2000(float(seconds))
        )
    df = df.dropna(subset=["plot_time_utc"]).sort_values("plot_time_utc").reset_index(drop=True)
    return df, time_column


def add_derived_beacon_fields(df: pd.DataFrame) -> pd.DataFrame:
    for derived_field, (left_field, right_field) in DERIVED_NUMERIC_FIELDS.items():
        if left_field not in df.columns or right_field not in df.columns:
            continue
        left = _apply_numeric_limits(df[left_field], left_field)
        right = _apply_numeric_limits(df[right_field], right_field)
        df[derived_field] = left * right
    return df


def filter_min_plot_time(
    df: pd.DataFrame,
    *,
    min_time_utc: datetime | None,
) -> tuple[pd.DataFrame, int]:
    """Drop rows before ``min_time_utc`` so boot-default timestamps do not dominate plots."""
    if min_time_utc is None or df.empty:
        return df, 0
    cutoff = min_time_utc.astimezone(timezone.utc)
    mask = df["plot_time_utc"] >= cutoff
    filtered = df.loc[mask].reset_index(drop=True)
    return filtered, int((~mask).sum())


def filter_to_median_time_window(
    df: pd.DataFrame,
    *,
    window_days: float,
) -> tuple[pd.DataFrame, int]:
    """Keep rows within +/- ``window_days`` of the median plotted UTC time."""
    if df.empty:
        return df, 0
    median_time = pd.Series(df["plot_time_utc"]).median()
    min_time = median_time - timedelta(days=window_days)
    max_time = median_time + timedelta(days=window_days)
    mask = (df["plot_time_utc"] >= min_time) & (df["plot_time_utc"] <= max_time)
    filtered = df.loc[mask].reset_index(drop=True)
    return filtered, int((~mask).sum())


def _setup_time_axis(ax) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%dT%H:%M:%SZ"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(True, alpha=0.28)


def _setup_sample_axis(ax) -> None:
    ax.grid(True, alpha=0.28)


def _filtered_numeric_series_for_x(
    df: pd.DataFrame,
    field: str,
    x_column: str,
) -> tuple[pd.Series, pd.Series, int]:
    raw_values = pd.to_numeric(df[field], errors="coerce")
    values = _apply_numeric_limits(raw_values, field)
    mask = values.notna()
    limits = PLOT_VALUE_LIMITS.get(field, {})
    sigma = limits.get("sigma")
    if sigma is not None:
        sigma_values = values.loc[mask].dropna()
        if len(sigma_values) > 1:
            mean = sigma_values.mean()
            std = sigma_values.std()
            if pd.notna(std) and std > 0:
                mask &= (values >= mean - float(sigma) * std) & (
                    values <= mean + float(sigma) * std
                )
    return df.loc[mask, x_column], values.loc[mask], int(raw_values.notna().sum() - mask.sum())


def _filtered_numeric_series(df: pd.DataFrame, field: str) -> tuple[pd.Series, pd.Series, int]:
    return _filtered_numeric_series_for_x(df, field, "plot_time_utc")


def _smooth_numeric_values(values: pd.Series, smooth_window_points: int | None) -> pd.Series:
    if smooth_window_points is None or smooth_window_points <= 1 or len(values) <= 1:
        return values
    window = min(int(smooth_window_points), len(values))
    min_periods = min(window, max(1, window // 5))
    return values.rolling(
        window=window,
        center=True,
        min_periods=min_periods,
    ).mean()


def _set_robust_ylim(ax, values: pd.Series) -> None:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return
    low = float(finite.quantile(0.01))
    high = float(finite.quantile(0.99))
    if low == high:
        padding = abs(low) * 0.05 or 1.0
        ax.set_ylim(low - padding, high + padding)
        return
    padding = 0.08 * (high - low)
    ax.set_ylim(low - padding, high + padding)


def _plot_numeric_series(
    ax,
    df: pd.DataFrame,
    field: str,
    *,
    smooth_window_points: int | None = None,
) -> None:
    limits = PLOT_VALUE_LIMITS.get(field, {})
    plot_times, plot_values, dropped = _filtered_numeric_series(df, field)
    ax.plot(
        plot_times,
        plot_values,
        marker=".",
        markersize=2.5,
        linewidth=0.9,
        color="#1f77b4",
        label="samples" if field in SMOOTHED_NUMERIC_FIELDS else None,
    )
    if (
        smooth_window_points is not None
        and smooth_window_points > 1
        and field in SMOOTHED_NUMERIC_FIELDS
        and len(plot_values) > 1
    ):
        window = min(int(smooth_window_points), len(plot_values))
        smoothed = _smooth_numeric_values(plot_values, smooth_window_points)
        ax.plot(
            plot_times,
            smoothed,
            linewidth=1.8,
            color="#d95f02",
            label=f"{window}-point rolling mean",
        )
        ax.legend(loc="best", fontsize=7)
    ax.set_ylabel("\n".join(textwrap.wrap(field, width=22)), fontsize=8)
    if dropped:
        label = []
        if "min" in limits:
            label.append(f"< {limits['min']:g}")
        if "max" in limits:
            label.append(f"> {limits['max']:g}")
        if limits.get("drop_zero"):
            label.append("= 0")
        if "sigma" in limits:
            label.append(f"> {limits['sigma']:g} sigma")
        label_text = " or ".join(label) if label else "filter"
        ax.text(
            0.99,
            0.92,
            f"{dropped} clipped ({label_text})",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#7f1d1d",
        )
    _setup_time_axis(ax)


def _plot_numeric_series_with_x(
    ax,
    df: pd.DataFrame,
    field: str,
    *,
    x_column: str,
    x_is_time: bool,
    color: str = "#1f77b4",
) -> pd.Series:
    plot_x, plot_values, dropped = _filtered_numeric_series_for_x(df, field, x_column)
    ax.plot(
        plot_x,
        plot_values,
        marker=".",
        markersize=1.4,
        linewidth=0.65,
        alpha=0.72,
        color=color,
    )
    ax.set_ylabel("\n".join(textwrap.wrap(field, width=22)), fontsize=8)
    if dropped:
        ax.text(
            0.99,
            0.92,
            f"{dropped} clipped",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#7f1d1d",
        )
    if x_is_time:
        _setup_time_axis(ax)
    else:
        _setup_sample_axis(ax)
    return plot_values


def _plot_overlaid_numeric_fields(
    ax,
    df: pd.DataFrame,
    fields: list[str],
    *,
    x_column: str,
    x_is_time: bool,
    ylabel: str,
) -> None:
    colors = ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd", "#8c564b"]
    plotted_values: list[pd.Series] = []
    for index, field in enumerate(fields):
        if field not in df.columns:
            continue
        plot_x, plot_values, _dropped = _filtered_numeric_series_for_x(
            df,
            field,
            x_column,
        )
        if plot_values.empty:
            continue
        plotted_values.append(plot_values)
        ax.plot(
            plot_x,
            plot_values,
            marker=".",
            markersize=1.2,
            linewidth=0.65,
            alpha=0.72,
            color=colors[index % len(colors)],
            label=field,
        )
    ax.set_ylabel(ylabel, fontsize=8)
    if plotted_values:
        _set_robust_ylim(ax, pd.concat(plotted_values, ignore_index=True))
    ax.legend(loc="best", fontsize=7)
    if x_is_time:
        _setup_time_axis(ax)
    else:
        _setup_sample_axis(ax)


def _category_mapping(values: pd.Series) -> dict[str, int]:
    categories = sorted(str(value) for value in values.dropna().unique())
    preferred_order = ["DEAD", "ALIVE"]
    ordered = [category for category in preferred_order if category in categories]
    ordered.extend(category for category in categories if category not in ordered)
    return {category: index for index, category in enumerate(ordered)}


def _plot_categorical_series(ax, df: pd.DataFrame, field: str) -> None:
    values = df[field].astype("string")
    mapping = _category_mapping(values)
    y = values.map(mapping)
    mask = y.notna()
    ax.step(
        df.loc[mask, "plot_time_utc"],
        y.loc[mask].astype(float),
        where="post",
        marker=".",
        markersize=2.5,
        linewidth=1.0,
        color="#c44e52",
    )
    ax.set_ylabel("\n".join(textwrap.wrap(field, width=22)), fontsize=8)
    ax.set_yticks(list(mapping.values()))
    ax.set_yticklabels(list(mapping.keys()))
    if mapping:
        ax.set_ylim(-0.4, max(mapping.values()) + 0.4)
    _setup_time_axis(ax)


def make_stacked_plot(
    df: pd.DataFrame,
    output_path: Path,
    *,
    numeric_fields: list[str],
    categorical_fields: list[str],
    time_column: str,
    add_leap_seconds: bool,
    smooth_window_points: int | None = None,
) -> None:
    plot_fields = [
        field
        for field in numeric_fields + categorical_fields
        if field in df.columns
    ]
    if not plot_fields:
        raise RuntimeError("None of the requested beacon fields were present in the CSV.")

    fig_height = max(7, 1.55 * len(plot_fields))
    fig, axes = plt.subplots(
        len(plot_fields),
        1,
        figsize=(13, fig_height),
        sharex=True,
        constrained_layout=True,
    )
    if len(plot_fields) == 1:
        axes = [axes]

    for ax, field in zip(axes, plot_fields):
        if field in categorical_fields:
            _plot_categorical_series(ax, df, field)
        else:
            _plot_numeric_series(
                ax,
                df,
                field,
                smooth_window_points=smooth_window_points,
            )

    axes[-1].set_xlabel("UTC time")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_individual_plots(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    numeric_fields: list[str],
    categorical_fields: list[str],
    time_column: str,
    add_leap_seconds: bool,
    filename_prefix: str = "beacon",
    smooth_window_points: int | None = None,
) -> list[Path]:
    output_paths: list[Path] = []
    for field in numeric_fields + categorical_fields:
        if field not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(11, 4.2), constrained_layout=True)
        if field in categorical_fields:
            _plot_categorical_series(ax, df, field)
        else:
            _plot_numeric_series(
                ax,
                df,
                field,
                smooth_window_points=smooth_window_points,
            )
        ax.set_title(field, fontsize=11)
        ax.set_xlabel("UTC time")
        output_path = output_dir / f"{filename_prefix}_{field}.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


def make_csie_power_temperature_stack(
    power_df: pd.DataFrame,
    therm_df: pd.DataFrame,
    output_path: Path,
    *,
    smooth_window_points: int | None = None,
) -> None:
    required_power = "beac_ana_csie_power"
    required_therm = "csie_det0_therm"
    if required_power not in power_df.columns:
        raise RuntimeError(f"Missing derived field: {required_power}")
    if required_therm not in therm_df.columns:
        raise RuntimeError(f"Missing CSIE HK field: {required_therm}")
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 9.6),
        sharex=False,
        constrained_layout=True,
    )
    _plot_numeric_series(
        axes[0],
        power_df,
        required_power,
        smooth_window_points=smooth_window_points,
    )
    _plot_numeric_series(axes[1], therm_df, required_therm)
    power_times, power_values, _dropped = _filtered_numeric_series(power_df, required_power)
    smoothed_power = _smooth_numeric_values(power_values, smooth_window_points)
    therm_times, therm_values, _dropped = _filtered_numeric_series(therm_df, required_therm)

    power_axis = axes[2]
    therm_axis = power_axis.twinx()
    power_axis.plot(
        power_times,
        smoothed_power,
        linewidth=1.8,
        color="#d95f02",
        label=required_power,
    )
    therm_axis.plot(
        therm_times,
        therm_values,
        linewidth=1.2,
        color="#1f77b4",
        alpha=0.9,
        label=required_therm,
    )
    power_axis.set_ylabel("\n".join(textwrap.wrap(required_power, width=22)), fontsize=8)
    therm_axis.set_ylabel("\n".join(textwrap.wrap(required_therm, width=22)), fontsize=8)
    power_axis.tick_params(axis="y", colors="#d95f02")
    therm_axis.tick_params(axis="y", colors="#1f77b4")
    power_axis.yaxis.label.set_color("#d95f02")
    therm_axis.yaxis.label.set_color("#1f77b4")
    _set_robust_ylim(power_axis, smoothed_power)
    therm_axis.set_ylim(-10, 0)
    _setup_time_axis(power_axis)
    axes[-1].set_xlabel("UTC time")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_eps_bus_voltage_det0_temperature_stack(
    beacon_df: pd.DataFrame,
    csie_hk_df: pd.DataFrame,
    output_path: Path,
) -> None:
    eps_bus_v_field = "beac_ana_eps_bus_v"
    det0_therm_field = "csie_det0_therm"
    if eps_bus_v_field not in beacon_df.columns:
        raise RuntimeError(f"Missing beacon field: {eps_bus_v_field}")
    if det0_therm_field not in csie_hk_df.columns:
        raise RuntimeError(f"Missing CSIE HK field: {det0_therm_field}")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 6.8),
        sharex=True,
        constrained_layout=True,
    )
    _plot_numeric_series(axes[0], beacon_df, eps_bus_v_field)
    _plot_numeric_series(axes[1], csie_hk_df, det0_therm_field)
    axes[-1].set_xlabel("UTC time")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def prepare_sample_index_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "packet_index" in df.columns:
        df = df.sort_values("packet_index").reset_index(drop=True)
        df["plot_sample_index"] = pd.to_numeric(
            df["packet_index"],
            errors="coerce",
        )
    else:
        df = df.reset_index(drop=True)
        df["plot_sample_index"] = df.index
    return df


def prepare_optional_packet_dataframe(
    path: Path,
    *,
    label: str,
    add_leap_seconds: bool,
    min_time_utc: datetime | None,
    median_window_days: float | None,
) -> tuple[pd.DataFrame, str, bool, str]:
    df, time_column = prepare_beacon_dataframe(
        path,
        time_column=None,
        add_leap_seconds=add_leap_seconds,
    )
    print(f"Read {len(df):,} timestamped {label} rows from {path}")
    print(f"Using {label} time column: {time_column}")
    if len(df):
        print(
            f"{label} time range UTC: "
            f"{df['plot_time_utc'].iloc[0].isoformat().replace('+00:00', 'Z')} to "
            f"{df['plot_time_utc'].iloc[-1].isoformat().replace('+00:00', 'Z')}"
        )
    filtered_df, dropped_early_rows = filter_min_plot_time(
        df,
        min_time_utc=min_time_utc,
    )
    if min_time_utc is not None:
        print(
            f"{label} minimum UTC filter: "
            f"{min_time_utc.isoformat().replace('+00:00', 'Z')}; "
            f"dropped {dropped_early_rows:,} row(s)"
        )
    if median_window_days is not None and not filtered_df.empty:
        filtered_df, dropped_rows = filter_to_median_time_window(
            filtered_df,
            window_days=median_window_days,
        )
        print(
            f"{label} median time window: +/- {median_window_days:g} day(s); "
            f"dropped {dropped_rows:,} row(s)"
        )
    if not filtered_df.empty:
        return filtered_df, "plot_time_utc", True, "UTC time"

    if df.empty:
        return filtered_df, "plot_time_utc", True, "UTC time"

    fallback_df = prepare_sample_index_dataframe(path)
    print(
        f"WARNING: no {label} rows remain after UTC filtering; writing {label} "
        "diagnostic plots versus packet_index instead."
    )
    return fallback_df, "plot_sample_index", False, "packet_index"


def make_dsps_quad_sum_plot(
    df: pd.DataFrame,
    output_path: Path,
    *,
    x_column: str,
    x_is_time: bool,
    x_label: str,
) -> None:
    plot_fields = [field for field in DSPS_QUAD_SUM_FIELDS if field in df.columns]
    if not plot_fields:
        raise RuntimeError("No DSPS quad-sum fields were present in the CSV.")
    fig, axes = plt.subplots(
        len(plot_fields),
        1,
        figsize=(13, max(5.4, 2.1 * len(plot_fields))),
        sharex=True,
        constrained_layout=True,
    )
    if len(plot_fields) == 1:
        axes = [axes]
    colors = ["#d95f02", "#1f77b4"]
    for index, (ax, field) in enumerate(zip(axes, plot_fields)):
        values = _plot_numeric_series_with_x(
            ax,
            df,
            field,
            x_column=x_column,
            x_is_time=x_is_time,
            color=colors[index % len(colors)],
        )
        _set_robust_ylim(ax, values)
    axes[-1].set_xlabel(x_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_dsps_visible_diodes_overplot(
    df: pd.DataFrame,
    output_path: Path,
    *,
    x_column: str,
    x_is_time: bool,
    x_label: str,
) -> None:
    plot_fields = [field for field in DSPS_VISIBLE_DIODE_FIELDS if field in df.columns]
    if not plot_fields:
        raise RuntimeError("No DSPS visible diode fields were present in the CSV.")
    fig, ax = plt.subplots(figsize=(13, 5.0), constrained_layout=True)
    _plot_overlaid_numeric_fields(
        ax,
        df,
        plot_fields,
        x_column=x_column,
        x_is_time=x_is_time,
        ylabel="dsps_visible_sps_diodes",
    )
    ax.set_title("dsps_visible_sps_diodes", fontsize=11)
    ax.set_xlabel(x_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_dsps_x_ray_diodes_overplot(
    df: pd.DataFrame,
    output_path: Path,
    *,
    x_column: str,
    x_is_time: bool,
    x_label: str,
) -> None:
    plot_fields = [field for field in DSPS_X_RAY_DIODE_FIELDS if field in df.columns]
    if not plot_fields:
        raise RuntimeError("No DSPS X-ray diode fields were present in the CSV.")
    fig, ax = plt.subplots(figsize=(13, 5.0), constrained_layout=True)
    _plot_overlaid_numeric_fields(
        ax,
        df,
        plot_fields,
        x_column=x_column,
        x_is_time=x_is_time,
        ylabel="dsps_x_ray_sps_diodes",
    )
    ax.set_title("dsps_x_ray_sps_diodes", fontsize=11)
    ax.set_xlabel(x_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def default_decoded_folder(config: Config) -> Path:
    return resolve_config_data_folder(config) / DEFAULT_DECODED_DIR_BASENAME


def main(argv: list[str] | None = None) -> None:
    default_config = Path(__file__).resolve().parent / "config_files" / "config_default.ini"
    parser = argparse.ArgumentParser(
        description="Create beacon sanity plots from decoded packet CSVs."
    )
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Path to processing config INI.",
    )
    parser.add_argument(
        "--decoded-folder",
        default=None,
        help=(
            "Folder containing decoded packet CSVs. Defaults to the configured data "
            f"folder's {DEFAULT_DECODED_DIR_BASENAME!r} folder."
        ),
    )
    parser.add_argument(
        "--beacon-csv",
        default=DEFAULT_BEACON_CSV,
        help=f"Beacon decoded CSV basename. Default: {DEFAULT_BEACON_CSV}.",
    )
    parser.add_argument(
        "--csie-hk-csv",
        default=DEFAULT_CSIE_HK_CSV,
        help=f"CSIE HK decoded CSV basename. Default: {DEFAULT_CSIE_HK_CSV}.",
    )
    parser.add_argument(
        "--dsps-csv",
        default=DEFAULT_DSPS_CSV,
        help=f"DSPS decoded CSV basename. Default: {DEFAULT_DSPS_CSV}.",
    )
    parser.add_argument(
        "--plot-dir",
        default=DEFAULT_PLOT_DIR,
        help=(
            "Output plot folder basename/path. Relative paths are placed under the "
            "decoded folder."
        ),
    )
    parser.add_argument(
        "--time-column",
        default=DEFAULT_TIME_COLUMN,
        help=(
            "Decoded CSV column to use as J2000 seconds for the x-axis. "
            "Default: auto-select the CCSDS secondary-header coarse seconds column."
        ),
    )
    parser.add_argument(
        "--use-secondary-header-time",
        action="store_true",
        help=(
            "Ignore --time-column and use the first ccsdsSecHeader2_sec* column. "
            "This is now the default unless --time-column is supplied."
        ),
    )
    parser.add_argument(
        "--no-leap-second-correction",
        action="store_true",
        help=(
            "Do not add post-J2000 leap seconds when converting the onboard no-leap "
            "second count to UTC."
        ),
    )
    parser.add_argument(
        "--median-window-days",
        type=float,
        default=None,
        help=(
            "Optional plotting filter: keep rows within this many days of the "
            "median plotted UTC time. Useful for a readable quick-look when one "
            "checksum-bypassed packet has a wild timestamp."
        ),
    )
    parser.add_argument(
        "--min-utc",
        default=DEFAULT_MIN_PLOT_TIME_UTC.isoformat().replace("+00:00", "Z"),
        help=(
            "Drop rows earlier than this UTC timestamp before plotting. Use 'none' "
            "to disable. Default: 2024-01-01T00:00:00Z, which removes boot-default "
            "J2000-zero timestamps."
        ),
    )
    parser.add_argument(
        "--stack-name",
        default="beacon_sanity_stack.png",
        help="Stacked plot output filename.",
    )
    parser.add_argument(
        "--csie-stack-name",
        default="csie_power_det0_therm_stack.png",
        help="CSIE power/detector-temperature stacked plot output filename.",
    )
    parser.add_argument(
        "--eps-det0-stack-name",
        default="eps_bus_v_det0_therm_stack.png",
        help="EPS bus voltage/detector-temperature stacked plot output filename.",
    )
    parser.add_argument(
        "--dsps-quad-sum-name",
        default="dsps_quad_sums.png",
        help="DSPS visible/X-ray quad-sum plot output filename.",
    )
    parser.add_argument(
        "--dsps-visible-diodes-name",
        default="dsps_visible_sps_diodes_overplot.png",
        help="DSPS visible diode overplot output filename.",
    )
    parser.add_argument(
        "--dsps-x-ray-diodes-name",
        default="dsps_x_ray_sps_diodes_overplot.png",
        help="DSPS X-ray diode overplot output filename.",
    )
    parser.add_argument(
        "--smooth-window-points",
        type=int,
        default=301,
        help="Rolling-mean window, in samples, for smoothed overplots. Default: 301.",
    )
    parser.add_argument(
        "--no-individual",
        action="store_true",
        help="Only write the stacked plot, not one PNG per field.",
    )
    args = parser.parse_args(argv)

    config = Config(os.path.abspath(os.path.expanduser(args.config)))
    decoded_folder = (
        Path(args.decoded_folder).expanduser().resolve()
        if args.decoded_folder
        else default_decoded_folder(config)
    )
    beacon_csv_path = decoded_folder / args.beacon_csv
    if not beacon_csv_path.is_file():
        raise FileNotFoundError(f"Beacon CSV not found: {beacon_csv_path}")

    plot_dir = Path(args.plot_dir).expanduser()
    if not plot_dir.is_absolute():
        plot_dir = decoded_folder / plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)

    requested_time_column = None if args.use_secondary_header_time else args.time_column
    add_leap_seconds = not args.no_leap_second_correction
    df, time_column = prepare_beacon_dataframe(
        beacon_csv_path,
        time_column=requested_time_column,
        add_leap_seconds=add_leap_seconds,
    )
    df = add_derived_beacon_fields(df)
    print(f"Read {len(df):,} timestamped beacon rows from {beacon_csv_path}")
    print(f"Using time column: {time_column}")
    print(
        "Leap-second correction: "
        f"{'enabled' if add_leap_seconds else 'disabled'}"
    )
    if len(df):
        print(
            "Time range UTC: "
            f"{df['plot_time_utc'].iloc[0].isoformat().replace('+00:00', 'Z')} to "
            f"{df['plot_time_utc'].iloc[-1].isoformat().replace('+00:00', 'Z')}"
        )
    min_time_utc = (
        None
        if args.min_utc.strip().lower() == "none"
        else parse_utc_datetime(args.min_utc)
    )
    df, dropped_early_rows = filter_min_plot_time(df, min_time_utc=min_time_utc)
    if min_time_utc is not None:
        print(
            "Minimum UTC filter: "
            f"{min_time_utc.isoformat().replace('+00:00', 'Z')}; "
            f"dropped {dropped_early_rows:,} row(s)"
        )
        if len(df):
            print(
                "Post-min-filter time range UTC: "
                f"{df['plot_time_utc'].iloc[0].isoformat().replace('+00:00', 'Z')} to "
                f"{df['plot_time_utc'].iloc[-1].isoformat().replace('+00:00', 'Z')}"
            )
    if args.median_window_days is not None:
        df, dropped_rows = filter_to_median_time_window(
            df,
            window_days=args.median_window_days,
        )
        print(
            f"Median time window: +/- {args.median_window_days:g} day(s); "
            f"dropped {dropped_rows:,} row(s)"
        )
        if len(df):
            print(
                "Filtered time range UTC: "
                f"{df['plot_time_utc'].iloc[0].isoformat().replace('+00:00', 'Z')} to "
                f"{df['plot_time_utc'].iloc[-1].isoformat().replace('+00:00', 'Z')}"
            )
    beacon_plots_available = not df.empty
    if df.empty:
        print("WARNING: no beacon rows remain after time filtering; beacon plots skipped.")
    else:
        if add_leap_seconds and "plot_leap_seconds_added" in df:
            leap_counts = sorted(
                int(value)
                for value in df["plot_leap_seconds_added"].dropna().unique()
            )
            print(f"Leap seconds added in plotted rows: {leap_counts}")

        stack_path = plot_dir / args.stack_name
        make_stacked_plot(
            df,
            stack_path,
            numeric_fields=NUMERIC_BEACON_FIELDS,
            categorical_fields=CATEGORICAL_BEACON_FIELDS,
            time_column=time_column,
            add_leap_seconds=add_leap_seconds,
            smooth_window_points=args.smooth_window_points,
        )
        print(f"Wrote stacked plot: {stack_path}")

        if not args.no_individual:
            individual_paths = make_individual_plots(
                df,
                plot_dir,
                numeric_fields=NUMERIC_BEACON_FIELDS,
                categorical_fields=CATEGORICAL_BEACON_FIELDS,
                time_column=time_column,
                add_leap_seconds=add_leap_seconds,
                smooth_window_points=args.smooth_window_points,
            )
            print(f"Wrote {len(individual_paths)} individual plot(s) to: {plot_dir}")

    csie_hk_csv_path = decoded_folder / args.csie_hk_csv
    if csie_hk_csv_path.is_file():
        csie_df, csie_time_column = prepare_beacon_dataframe(
            csie_hk_csv_path,
            time_column=None,
            add_leap_seconds=add_leap_seconds,
        )
        print(f"Read {len(csie_df):,} timestamped CSIE HK rows from {csie_hk_csv_path}")
        print(f"Using CSIE HK time column: {csie_time_column}")
        if len(csie_df):
            print(
                "CSIE HK time range UTC: "
                f"{csie_df['plot_time_utc'].iloc[0].isoformat().replace('+00:00', 'Z')} to "
                f"{csie_df['plot_time_utc'].iloc[-1].isoformat().replace('+00:00', 'Z')}"
            )
        csie_df, csie_dropped_early_rows = filter_min_plot_time(
            csie_df,
            min_time_utc=min_time_utc,
        )
        if min_time_utc is not None:
            print(
                "CSIE HK minimum UTC filter: "
                f"{min_time_utc.isoformat().replace('+00:00', 'Z')}; "
                f"dropped {csie_dropped_early_rows:,} row(s)"
            )
            if len(csie_df):
                print(
                    "CSIE HK post-min-filter time range UTC: "
                    f"{csie_df['plot_time_utc'].iloc[0].isoformat().replace('+00:00', 'Z')} to "
                    f"{csie_df['plot_time_utc'].iloc[-1].isoformat().replace('+00:00', 'Z')}"
                )
        if args.median_window_days is not None:
            csie_df, csie_dropped_rows = filter_to_median_time_window(
                csie_df,
                window_days=args.median_window_days,
            )
            print(
                f"CSIE HK median time window: +/- {args.median_window_days:g} day(s); "
                f"dropped {csie_dropped_rows:,} row(s)"
            )
        if csie_df.empty:
            print("WARNING: no CSIE HK rows remain after time filtering; CSIE plots skipped.")
        else:
            if not args.no_individual:
                csie_individual_paths = make_individual_plots(
                    csie_df,
                    plot_dir,
                    numeric_fields=CSIE_HK_NUMERIC_FIELDS,
                    categorical_fields=[],
                    time_column=csie_time_column,
                    add_leap_seconds=add_leap_seconds,
                    filename_prefix="csie_hk",
                    smooth_window_points=args.smooth_window_points,
                )
                print(
                    f"Wrote {len(csie_individual_paths)} CSIE HK individual plot(s) to: "
                    f"{plot_dir}"
                )
            if beacon_plots_available:
                csie_stack_path = plot_dir / args.csie_stack_name
                make_csie_power_temperature_stack(
                    df,
                    csie_df,
                    csie_stack_path,
                    smooth_window_points=args.smooth_window_points,
                )
                print(f"Wrote CSIE power/det0 stack plot: {csie_stack_path}")

                eps_det0_stack_path = plot_dir / args.eps_det0_stack_name
                make_eps_bus_voltage_det0_temperature_stack(
                    df,
                    csie_df,
                    eps_det0_stack_path,
                )
                print(f"Wrote EPS bus voltage/det0 stack plot: {eps_det0_stack_path}")
            else:
                print(
                    "WARNING: skipped CSIE power/det0 and EPS bus voltage/det0 "
                    "stacks because no beacon rows remain after time filtering."
                )
    else:
        print(f"CSIE HK CSV not found; skipped CSIE HK plots: {csie_hk_csv_path}")

    dsps_csv_path = decoded_folder / args.dsps_csv
    if dsps_csv_path.is_file():
        dsps_df, dsps_x_column, dsps_x_is_time, dsps_x_label = (
            prepare_optional_packet_dataframe(
                dsps_csv_path,
                label="DSPS",
                add_leap_seconds=add_leap_seconds,
                min_time_utc=min_time_utc,
                median_window_days=args.median_window_days,
            )
        )
        if dsps_df.empty:
            print("WARNING: no DSPS rows available; DSPS plots skipped.")
        else:
            dsps_quad_sum_path = plot_dir / args.dsps_quad_sum_name
            make_dsps_quad_sum_plot(
                dsps_df,
                dsps_quad_sum_path,
                x_column=dsps_x_column,
                x_is_time=dsps_x_is_time,
                x_label=dsps_x_label,
            )
            print(f"Wrote DSPS quad-sum plot: {dsps_quad_sum_path}")

            dsps_diodes_path = plot_dir / args.dsps_visible_diodes_name
            make_dsps_visible_diodes_overplot(
                dsps_df,
                dsps_diodes_path,
                x_column=dsps_x_column,
                x_is_time=dsps_x_is_time,
                x_label=dsps_x_label,
            )
            print(f"Wrote DSPS visible diode overplot: {dsps_diodes_path}")

            dsps_x_ray_diodes_path = plot_dir / args.dsps_x_ray_diodes_name
            make_dsps_x_ray_diodes_overplot(
                dsps_df,
                dsps_x_ray_diodes_path,
                x_column=dsps_x_column,
                x_is_time=dsps_x_is_time,
                x_label=dsps_x_label,
            )
            print(f"Wrote DSPS X-ray diode overplot: {dsps_x_ray_diodes_path}")
    else:
        print(f"DSPS CSV not found; skipped DSPS plots: {dsps_csv_path}")


if __name__ == "__main__":
    main()
