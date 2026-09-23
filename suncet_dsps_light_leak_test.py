"""Plot DSPS 1/2 SPS positions for the September 17 light-leak test.

The APID 35 DSPS packet's own coarse timestamp is not J2000 in this capture.
Times here are inferred from the bracketing APID 1 beacons in the same ordered
CCSDS stream. The beacon cadence is three seconds, so times near a requested
window boundary are approximate. The decoded X/Y values are arcseconds and
are divided by 3,600 for display in degrees; the source CSV is unchanged.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, DateFormatter
import numpy as np
import pandas as pd


DEFAULT_FOLDER = Path(
    "/Users/masonjp2/Dropbox/suncet_dropbox/9000 Processing/data/test_data/"
    "2026-09-17_dsps_light_leak_check"
)
DSPS_CSV = "decoded_apid_0035_dsps_data.csv"
BEACON_CSV = "decoded_apid_0001_beacon.csv"
BEACON_TIME = "ccsdsSecHeader2_sec_beacon"
SOURCE = "input_file_relative_path"
OFFSET = "source_offset"
POSITION_FIELDS = (
    "dsps_visible_sps_x_pos",
    "dsps_visible_sps_y_pos",
    "dsps_x_ray_sps_x_pos",
    "dsps_x_ray_sps_y_pos",
)
MAX_BEACON_GAP_SECONDS = 6
ARCSECONDS_PER_DEGREE = 3_600.0
J2000_UTC_EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)
# Match the pipeline's UTC display convention. This post-J2000 leap-second
# adjustment is still a display policy under review, not a wire-format rule.
LEAP_SECOND_EFFECTIVE_UTC = (
    datetime(2006, 1, 1, tzinfo=timezone.utc),
    datetime(2009, 1, 1, tzinfo=timezone.utc),
    datetime(2012, 7, 1, tzinfo=timezone.utc),
    datetime(2015, 7, 1, tzinfo=timezone.utc),
    datetime(2017, 1, 1, tzinfo=timezone.utc),
)


def j2000_seconds_to_utc(seconds: float) -> datetime:
    """Use the same midnight epoch and leap adjustment as pipeline displays."""
    leap_count = 0
    while True:
        corrected = J2000_UTC_EPOCH + timedelta(seconds=seconds + leap_count)
        next_count = sum(corrected >= effective for effective in LEAP_SECOND_EFFECTIVE_UTC)
        if next_count == leap_count:
            return corrected
        leap_count = next_count


def _read_csv(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    for name in required:
        if name not in (SOURCE, "dsps_data_unknown_tail_nonzero"):
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def infer_dsps_j2000(dsps: pd.DataFrame, beacons: pd.DataFrame) -> pd.DataFrame:
    """Interpolate between same-file beacons by stream offset, never bad DSPS time."""
    result = dsps.copy()
    result["j2000_beacon_inferred"] = np.nan
    valid_beacons = beacons.loc[
        beacons[BEACON_TIME].between(800_000_000, 900_000_000)
    ]

    for source, dsps_rows in result.groupby(SOURCE):
        anchors = valid_beacons.loc[valid_beacons[SOURCE] == source].sort_values(OFFSET)
        anchors = anchors.dropna(subset=[OFFSET, BEACON_TIME])
        if len(anchors) < 2:
            continue
        anchor_offsets = anchors[OFFSET].to_numpy(dtype=np.float64)
        anchor_times = anchors[BEACON_TIME].to_numpy(dtype=np.float64)
        if np.any(np.diff(anchor_offsets) <= 0) or np.any(np.diff(anchor_times) <= 0):
            raise ValueError(f"Beacon offsets/times are not strictly increasing: {source}")

        sample_offsets = dsps_rows[OFFSET].to_numpy(dtype=np.float64)
        left = np.searchsorted(anchor_offsets, sample_offsets, side="right") - 1
        inferred = np.full(len(sample_offsets), np.nan)
        bracketed = (left >= 0) & (left < len(anchor_offsets) - 1)
        sample_indices = np.flatnonzero(bracketed)
        left_indices = left[sample_indices]
        right_indices = left_indices + 1
        spans_seconds = anchor_times[right_indices] - anchor_times[left_indices]
        usable = spans_seconds <= MAX_BEACON_GAP_SECONDS
        sample_indices = sample_indices[usable]
        left_indices = left_indices[usable]
        right_indices = right_indices[usable]
        fraction = (
            sample_offsets[sample_indices] - anchor_offsets[left_indices]
        ) / (anchor_offsets[right_indices] - anchor_offsets[left_indices])
        inferred[sample_indices] = (
            anchor_times[left_indices]
            + fraction * (anchor_times[right_indices] - anchor_times[left_indices])
        )

        # A couple of DSPS packets follow the final beacon. Retain only packets
        # within one normal beacon interval, at the final anchor's known second.
        trailing = left == len(anchor_offsets) - 1
        recent_steps = np.diff(anchor_offsets[-101:])
        max_tail_bytes = float(np.median(recent_steps))
        near_tail = trailing & (
            sample_offsets - anchor_offsets[-1] <= max_tail_bytes
        )
        inferred[near_tail] = anchor_times[-1]
        result.loc[dsps_rows.index, "j2000_beacon_inferred"] = inferred

    return result


def _limits(frame: pd.DataFrame, fields: tuple[str, str]) -> tuple[float, float]:
    values = frame[list(fields)].to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError(f"No finite positions in {fields}")
    lower, upper = float(values.min()), float(values.max())
    padding = max((upper - lower) * 0.04, 1.0 / ARCSECONDS_PER_DEGREE)
    return lower - padding, upper + padding


def plot_window(
    frame: pd.DataFrame,
    *,
    start: int,
    stop: int | None,
    limits: tuple[tuple[float, float], tuple[float, float]],
    flagged_omitted: int,
    output: Path,
) -> None:
    if frame.empty:
        raise ValueError(f"No unflagged DSPS samples for window beginning {start}")
    j2000 = frame["j2000_beacon_inferred"].to_numpy(dtype=float)
    x = [j2000_seconds_to_utc(value) for value in j2000]
    start_utc = j2000_seconds_to_utc(start)
    stop_utc = j2000_seconds_to_utc(stop if stop is not None else float(np.nanmax(j2000)))
    panels = (
        ("DSPS 1 / visible SPS", POSITION_FIELDS[:2]),
        ("DSPS 2 / X-ray SPS", POSITION_FIELDS[2:]),
    )
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.88, top=0.89, bottom=0.18, hspace=0.20)
    range_label = (
        f"{start_utc:%Y-%m-%dT%H:%M:%SZ} — {stop_utc:%Y-%m-%dT%H:%M:%SZ}"
    )
    fig.suptitle(f"DSPS X/Y positions · {range_label}", fontsize=13)
    for ax, (panel_title, fields), ylim in zip(axes, panels, limits):
        for field, label, color in zip(fields, ("X", "Y"), ("#1f77b4", "#d95f02")):
            ax.scatter(
                x,
                frame[field].to_numpy(dtype=float),
                s=5,
                alpha=0.62,
                color=color,
                label=label,
                rasterized=True,
            )
        ax.set_title(panel_title, loc="left", fontsize=11)
        ax.set_ylabel("Angular position (degrees)")
        ax.set_ylim(*ylim)
        ax.grid(color="0.85", linewidth=0.6)
        ax.legend(
            title="Axis",
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            frameon=True,
            facecolor="white",
            edgecolor="0.8",
            framealpha=1,
        )
    axes[-1].set_xlabel("UTC time (ISO 8601, inferred from neighboring beacons)")
    axes[-1].set_xlim(start_utc, stop_utc)
    axes[-1].xaxis.set_major_locator(AutoDateLocator(minticks=3, maxticks=5))
    axes[-1].xaxis.set_major_formatter(
        DateFormatter("%Y-%m-%dT%H:%M:%SZ", tz=timezone.utc)
    )
    axes[-1].tick_params(axis="x", rotation=20)
    fig.text(
        0.01,
        0.02,
        f"APID 1 beacon timing (~3 s cadence); {flagged_omitted} flagged DSPS packet(s) omitted. "
        "Points are not connected across acquisition gaps.\n"
        "UTC uses the pipeline's post-J2000 leap-second convention (+5 s for this test).",
        ha="left",
        va="bottom",
        fontsize=9,
        color="0.35",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--first-start", type=int, default=842_984_450)
    # Stop just after the last DSPS sample before the 17-minute acquisition gap.
    parser.add_argument("--first-stop", type=int, default=842_986_220)
    parser.add_argument("--second-start", type=int, default=842_988_791)
    args = parser.parse_args()
    if not args.first_start < args.first_stop <= args.second_start:
        parser.error("require --first-start < --first-stop <= --second-start")

    decoded_dir = args.folder / "decoded_packets"
    dsps = _read_csv(
        decoded_dir / DSPS_CSV,
        (SOURCE, OFFSET, *POSITION_FIELDS, "dsps_data_unknown_tail_nonzero"),
    )
    beacons = _read_csv(decoded_dir / BEACON_CSV, (SOURCE, OFFSET, BEACON_TIME))
    dsps = infer_dsps_j2000(dsps, beacons)
    dsps = dsps.dropna(subset=["j2000_beacon_inferred", *POSITION_FIELDS])
    dsps[list(POSITION_FIELDS)] = (
        dsps[list(POSITION_FIELDS)] / ARCSECONDS_PER_DEGREE
    )
    flagged = (
        dsps["dsps_data_unknown_tail_nonzero"].astype("string").str.lower() == "true"
    ).fillna(False)
    dsps["flagged_packet"] = flagged.to_numpy(dtype=bool)
    first_all = dsps.loc[
        (dsps["j2000_beacon_inferred"] >= args.first_start)
        & (dsps["j2000_beacon_inferred"] < args.first_stop)
    ]
    second_all = dsps.loc[dsps["j2000_beacon_inferred"] >= args.second_start]
    first = first_all.loc[~first_all["flagged_packet"]]
    second = second_all.loc[~second_all["flagged_packet"]]
    plotted = pd.concat((first, second))
    limits = (
        _limits(plotted, POSITION_FIELDS[:2]),
        _limits(plotted, POSITION_FIELDS[2:]),
    )

    output_dir = args.output_dir or (args.folder / "analysis_plots")
    outputs = (
        (
            first,
            args.first_start,
            args.first_stop,
            first_all,
            output_dir / f"dsps_xy_j2000_{args.first_start}_to_{args.first_stop}.png",
        ),
        (
            second,
            args.second_start,
            None,
            second_all,
            output_dir / f"dsps_xy_j2000_{args.second_start}_to_end.png",
        ),
    )
    for window, start, stop, original, output in outputs:
        omitted = int(original["flagged_packet"].sum())
        plot_window(
            window,
            start=start,
            stop=stop,
            limits=limits,
            flagged_omitted=omitted,
            output=output,
        )
        print(f"Saved {output} ({len(window):,} plotted, {omitted} flagged omitted)")


if __name__ == "__main__":
    main()
