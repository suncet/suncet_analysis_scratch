"""Plot X-band temperature and power telemetry from a decoded beacon CSV.

Adapted from the older ``suncet_analysis_scratch/plot_bluefin_tvac.py`` workflow.
The current CTDB exposes the equivalent measurements in the beacon packet instead
of a dedicated ``xband_hk_pkt`` CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_BEACON_CSV = "decoded_apid_0001_beacon.csv"
TIME_FIELD = "ccsdsSecHeader2_sec_beacon"
SUBSECOND_FIELD = "ccsdsSecHeader2_sub_beacon"
PA_TEMPERATURE_FIELD = "beac_xband_pa_temp"
PA_CURRENT_FIELD = "beac_xband_pa_curr"
INPUT_CURRENT_FIELD = "beac_ana_xband_i"
INPUT_VOLTAGE_FIELD = "beac_ana_xband_v"
POWER_STATE_FIELD = "beac_eps_pwr_state_xband"

REQUIRED_FIELDS = [
    TIME_FIELD,
    PA_TEMPERATURE_FIELD,
    PA_CURRENT_FIELD,
    INPUT_CURRENT_FIELD,
    INPUT_VOLTAGE_FIELD,
    POWER_STATE_FIELD,
]


def fix_timestamps(timestamps: pd.Series) -> np.ndarray:
    """Make reset-prone onboard seconds monotonic while retaining sample cadence."""
    raw = pd.to_numeric(timestamps, errors="coerce").to_numpy(dtype=float)
    if not len(raw):
        return raw
    positive_steps = np.diff(raw)
    positive_steps = positive_steps[np.isfinite(positive_steps) & (positive_steps > 0)]
    cadence = float(np.median(positive_steps)) if len(positive_steps) else 1.0
    corrected = np.empty_like(raw)
    corrected[0] = raw[0]
    offset = 0.0
    for index in range(1, len(raw)):
        candidate = raw[index] + offset
        if candidate <= corrected[index - 1]:
            offset += corrected[index - 1] + cadence - candidate
        corrected[index] = raw[index] + offset
    return corrected


def prepare_xband_dataframe(beacon_csv: Path) -> pd.DataFrame:
    """Read beacon telemetry and add plotting time and derived X-band power."""
    df = pd.read_csv(beacon_csv)
    missing = [field for field in REQUIRED_FIELDS if field not in df.columns]
    if missing:
        raise KeyError(f"Beacon CSV is missing required field(s): {', '.join(missing)}")

    prepared = pd.DataFrame()
    prepared["onboard_seconds"] = pd.to_numeric(df[TIME_FIELD], errors="coerce")
    if SUBSECOND_FIELD in df:
        prepared["onboard_subseconds"] = pd.to_numeric(
            df[SUBSECOND_FIELD], errors="coerce"
        )
    corrected_seconds = fix_timestamps(prepared["onboard_seconds"])
    prepared["elapsed_minutes"] = (corrected_seconds - corrected_seconds[0]) / 60.0
    prepared["xband_pa_temperature_degC"] = pd.to_numeric(
        df[PA_TEMPERATURE_FIELD], errors="coerce"
    )
    prepared["xband_pa_current_A"] = pd.to_numeric(
        df[PA_CURRENT_FIELD], errors="coerce"
    )
    prepared["xband_input_current_A"] = pd.to_numeric(
        df[INPUT_CURRENT_FIELD], errors="coerce"
    )
    prepared["xband_input_voltage_V"] = pd.to_numeric(
        df[INPUT_VOLTAGE_FIELD], errors="coerce"
    )
    prepared["xband_input_power_W"] = (
        prepared["xband_input_current_A"] * prepared["xband_input_voltage_V"]
    )
    prepared["xband_power_state"] = df[POWER_STATE_FIELD].astype("string")
    prepared["xband_power_state_on"] = (
        prepared["xband_power_state"].str.upper() == "ON"
    ).astype(int)
    return prepared


def select_elapsed_window(
    df: pd.DataFrame,
    *,
    start_minute: float | None,
    reset_time_zero: bool,
) -> pd.DataFrame:
    """Select a trailing elapsed-time window and optionally rebase it to zero."""
    selected = df
    if start_minute is not None:
        selected = selected.loc[selected["elapsed_minutes"] >= start_minute].copy()
    if selected.empty:
        raise RuntimeError("No samples remain after the elapsed-time selection.")
    if reset_time_zero:
        selected = selected.copy()
        selected["elapsed_minutes"] -= float(selected["elapsed_minutes"].iloc[0])
    return selected.reset_index(drop=True)


def _plot_stack(df: pd.DataFrame, output_path: Path) -> None:
    time = df["elapsed_minutes"]
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(13, 13),
        sharex=True,
        constrained_layout=True,
    )
    fig.suptitle("X-Band Temperature and Power Quick Look", fontsize=15)

    axes[0].plot(time, df["xband_pa_temperature_degC"], color="#d95f02")
    axes[0].set_ylabel("PA temperature\n(°C)")

    axes[1].plot(time, df["xband_input_power_W"], color="#c62828")
    axes[1].set_ylabel("Input power\n(W)")

    axes[2].plot(
        time,
        df["xband_input_current_A"],
        label="total X-band input current",
        color="#2e7d32",
    )
    axes[2].plot(
        time,
        df["xband_pa_current_A"],
        label="PA current",
        color="#00796b",
        alpha=0.85,
    )
    axes[2].set_ylabel("Current\n(A)")
    axes[2].legend(loc="best", fontsize=8)

    axes[3].plot(time, df["xband_input_voltage_V"], color="#6a1b9a")
    axes[3].set_ylabel("Input voltage\n(V)")

    axes[4].step(
        time,
        df["xband_power_state_on"],
        where="post",
        color="#1565c0",
    )
    axes[4].set_ylabel("EPS X-band\npower state")
    axes[4].set_yticks([0, 1], labels=["OFF", "ON"])
    axes[4].set_ylim(-0.15, 1.15)
    axes[4].set_xlabel("Elapsed time (minutes)")

    for axis in axes:
        axis.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_temperature_vs_power(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    points = ax.scatter(
        df["xband_input_power_W"],
        df["xband_pa_temperature_degC"],
        c=df["elapsed_minutes"],
        cmap="viridis",
        s=13,
        alpha=0.8,
    )
    ax.set_title("X-Band PA Temperature vs. Input Power")
    ax.set_xlabel("X-band input power (W)")
    ax.set_ylabel("PA temperature (°C)")
    ax.grid(True, alpha=0.3)
    colorbar = fig.colorbar(points, ax=ax)
    colorbar.set_label("Elapsed time (minutes)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def print_summary(df: pd.DataFrame) -> None:
    duration = float(df["elapsed_minutes"].max()) if len(df) else 0.0
    print(f"Samples: {len(df):,}")
    print(f"Duration: {duration:.3f} minutes")
    for state, rows in df.groupby("xband_power_state", dropna=False):
        print(
            f"{state}: {len(rows):,} samples; "
            f"temperature {rows['xband_pa_temperature_degC'].min():.3f} to "
            f"{rows['xband_pa_temperature_degC'].max():.3f} degC; "
            f"input power {rows['xband_input_power_W'].min():.3f} to "
            f"{rows['xband_input_power_W'].max():.3f} W"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "beacon_csv",
        type=Path,
        help=f"Decoded beacon CSV (normally {DEFAULT_BEACON_CSV}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <beacon CSV directory>/xband_analysis",
    )
    parser.add_argument(
        "--start-minute",
        type=float,
        default=None,
        help="Keep samples at or after this elapsed minute in the full capture.",
    )
    parser.add_argument(
        "--reset-time-zero",
        action="store_true",
        help="Reset the first selected sample to elapsed time zero.",
    )
    args = parser.parse_args()

    beacon_csv = args.beacon_csv.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else beacon_csv.parent / "xband_analysis"
    )
    df = prepare_xband_dataframe(beacon_csv)
    df = select_elapsed_window(
        df,
        start_minute=args.start_minute,
        reset_time_zero=args.reset_time_zero,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_csv = output_dir / "xband_temperature_power_samples.csv"
    stack_plot = output_dir / "xband_temperature_power_vs_time.png"
    diagnostic_plot = output_dir / "xband_temperature_vs_power.png"
    df.to_csv(analysis_csv, index=False)
    _plot_stack(df, stack_plot)
    _plot_temperature_vs_power(df, diagnostic_plot)
    print_summary(df)
    print(f"Wrote {analysis_csv}")
    print(f"Wrote {stack_plot}")
    print(f"Wrote {diagnostic_plot}")


if __name__ == "__main__":
    main()
