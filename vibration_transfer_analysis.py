import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DB_EPS = 1e-16

# Stage colors (PSD lines) and transmissibility line colors (blends = numerator/denominator).
COLOR_CONTROL = "k"
COLOR_DISPENSER = "tomato"
COLOR_BATTERY = "dodgerblue"
COLOR_DSPS = "limegreen"
# dodgerblue + tomato (blue/red blend) → purple tone
COLOR_BATTERY_OVER_DISPENSER = "#8E79A3"
# limegreen + tomato (green/red blend) → olive / yellow-green tone
COLOR_DSPS_OVER_DISPENSER = "#99A33D"

RATIO_LINE_STYLES: List[Tuple[str, str]] = [
    ("Dispenser Combined / Control", COLOR_DISPENSER),
    ("Battery / Dispenser Combined", COLOR_BATTERY_OVER_DISPENSER),
    ("DSPS / Dispenser Combined", COLOR_DSPS_OVER_DISPENSER),
]


@dataclass
class RunConfig:
    run_number: int
    axis_label: str


RUNS_DEFAULT = [
    RunConfig(5, "Y"),
    RunConfig(11, "Z"),
    RunConfig(17, "X"),
]


def get_default_data_dir() -> str:
    suncet_data = os.getenv("suncet_data")
    if not suncet_data:
        raise RuntimeError("Environment variable 'suncet_data' is not set.")
    return os.path.join(
        suncet_data,
        "test_data",
        "2026-04-01_vibration_test",
        "PUS58001595",
    )


def get_default_output_dir() -> str:
    username = os.getenv("USER", "unknown_user")
    return (
        f"/Users/{username}/Dropbox/suncet_dropbox/"
        "7000 Testing/7190 Vibration Tests/2026-04-01  SunCET FM1 Vibe/analysis/"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze vibration transmissibility from Control1 to payload sensors."
    )
    parser.add_argument(
        "--data-dir",
        default=get_default_data_dir(),
        help="Directory containing Run N.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default=get_default_output_dir(),
        help="Directory to save plots and summary tables.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        default=[r.run_number for r in RUNS_DEFAULT],
        help="Run numbers to process.",
    )
    return parser.parse_args()


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _vector_psd(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [f"{prefix}-X (G²/Hz)", f"{prefix}-Y (G²/Hz)", f"{prefix}-Z (G²/Hz)"]
    return df[cols].sum(axis=1)


def load_run_psd(run_csv: str) -> pd.DataFrame:
    df = pd.read_csv(run_csv)

    required_cols = [
        "Frequency",
        "Control1 (G²/Hz)",
        "C1 (G²/Hz)",
        "C2 (G²/Hz)",
        "Dispenser Top Back-X (G²/Hz)",
        "Dispenser Top Back-Y (G²/Hz)",
        "Dispenser Top Back-Z (G²/Hz)",
        "Dispenser Top Front-X (G²/Hz)",
        "Dispenser Top Front-Y (G²/Hz)",
        "Dispenser Top Front-Z (G²/Hz)",
        "Battery-X (G²/Hz)",
        "Battery-Y (G²/Hz)",
        "Battery-Z (G²/Hz)",
        "DSPS-X (G²/Hz)",
        "DSPS-Y (G²/Hz)",
        "DSPS-Z (G²/Hz)",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {run_csv}: {missing}")

    out = pd.DataFrame()
    out["Frequency_Hz"] = _safe_numeric(df["Frequency"])
    out["Control_psd"] = _safe_numeric(df["Control1 (G²/Hz)"])
    out["C1_psd"] = _safe_numeric(df["C1 (G²/Hz)"])
    out["C2_psd"] = _safe_numeric(df["C2 (G²/Hz)"])

    out["Dispenser_Back_psd"] = _vector_psd(df, "Dispenser Top Back")
    out["Dispenser_Front_psd"] = _vector_psd(df, "Dispenser Top Front")
    out["Battery_psd"] = _vector_psd(df, "Battery")
    out["DSPS_psd"] = _vector_psd(df, "DSPS")

    out = out.dropna(subset=["Frequency_Hz", "Control_psd"])
    out = out[out["Frequency_Hz"] > 0].copy()
    out = out.sort_values("Frequency_Hz")

    # PSD energy average of the two dispenser sensors.
    out["Dispenser_Combined_psd"] = 0.5 * (
        out["Dispenser_Back_psd"] + out["Dispenser_Front_psd"]
    )

    # Drop duplicated frequency rows if present; keep first value.
    out = out.drop_duplicates(subset=["Frequency_Hz"], keep="first")
    return out


def transmissibility(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / np.clip(denominator, DB_EPS, None)


def db10(x: pd.Series) -> pd.Series:
    return 10.0 * np.log10(np.clip(x, DB_EPS, None))


def summarize_curve(
    freq: pd.Series, ratio_linear: pd.Series, label: str, run_number: int, axis: str
) -> Dict[str, float]:
    ratio_db = db10(ratio_linear)
    idx_peak = ratio_db.idxmax()
    idx_notch = ratio_db.idxmin()

    row = {
        "run": run_number,
        "axis": axis,
        "metric": label,
        "peak_amp_db": float(ratio_db.loc[idx_peak]),
        "peak_amp_freq_hz": float(freq.loc[idx_peak]),
        "max_damping_db": float(ratio_db.loc[idx_notch]),
        "max_damping_freq_hz": float(freq.loc[idx_notch]),
        "median_db": float(ratio_db.median()),
        "pct_points_gt_3db": float((ratio_db > 3.0).mean() * 100.0),
        "pct_points_lt_minus3db": float((ratio_db < -3.0).mean() * 100.0),
    }
    return row


def make_run_plots(
    run_df: pd.DataFrame, run_number: int, axis: str, output_dir: str
) -> Tuple[List[str], List[Dict[str, float]]]:
    f = run_df["Frequency_Hz"]
    c = run_df["Control_psd"]

    ratios = {
        "Dispenser Combined / Control": transmissibility(
            run_df["Dispenser_Combined_psd"], c
        ),
        "Battery / Dispenser Combined": transmissibility(
            run_df["Battery_psd"], run_df["Dispenser_Combined_psd"]
        ),
        "DSPS / Dispenser Combined": transmissibility(
            run_df["DSPS_psd"], run_df["Dispenser_Combined_psd"]
        ),
    }

    created_files = []
    summary_rows = []

    # Plot 1: Energy PSD comparison.
    plt.figure(figsize=(11, 7))
    curves = [
        ("Control", c, COLOR_CONTROL),
        ("Dispenser Combined (mean PSD)", run_df["Dispenser_Combined_psd"], COLOR_DISPENSER),
        ("Battery (vector PSD)", run_df["Battery_psd"], COLOR_BATTERY),
        ("DSPS (vector PSD)", run_df["DSPS_psd"], COLOR_DSPS),
    ]
    for label, y, color in curves:
        plt.loglog(f, np.clip(y, DB_EPS, None), label=label, color=color)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (G²/Hz)")
    plt.title(f"Run {run_number} ({axis}-axis): PSD Energy Transfer Overview")
    plt.grid(which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    p1 = os.path.join(output_dir, f"run_{run_number:02d}_{axis}_psd_overview.png")
    plt.savefig(p1, dpi=180)
    plt.close()
    created_files.append(p1)

    # Plot 1b: Control vs table accelerometer channels C1 and C2 (PSD).
    plt.figure(figsize=(11, 7))
    for label, y, color in [
        ("Control1", c, COLOR_CONTROL),
        ("C1", run_df["C1_psd"], "0.45"),
        ("C2", run_df["C2_psd"], "0.65"),
    ]:
        plt.loglog(f, np.clip(y, DB_EPS, None), label=label, color=color)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (G²/Hz)")
    plt.title(
        f"Run {run_number} ({axis}-axis): Control1 vs C1 and C2 (table accelerometers)"
    )
    plt.grid(which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    p1b = os.path.join(
        output_dir, f"run_{run_number:02d}_{axis}_control_c1_c2_psd.png"
    )
    plt.savefig(p1b, dpi=180)
    plt.close()
    created_files.append(p1b)

    # Plot 2: Transmissibility linear.
    plt.figure(figsize=(11, 7))
    for label, color in RATIO_LINE_STYLES:
        plt.semilogx(f, ratios[label], label=label, color=color)
    plt.axhline(1.0, color="k", linestyle=":", linewidth=1.0)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Transmissibility (PSD ratio)")
    plt.title(f"Run {run_number} ({axis}-axis): Stage-to-Stage Transmissibility")
    plt.grid(which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    p2 = os.path.join(
        output_dir, f"run_{run_number:02d}_{axis}_transmissibility_linear.png"
    )
    plt.savefig(p2, dpi=180)
    plt.close()
    created_files.append(p2)

    # Plot 3: Transmissibility dB with stage-to-stage relationships.
    plt.figure(figsize=(11, 7))
    for label, color in RATIO_LINE_STYLES:
        plt.semilogx(f, db10(ratios[label]), label=label, color=color)
    plt.axhline(0.0, color="k", linestyle=":", linewidth=1.0)
    plt.axhline(3.0, color="gray", linestyle="--", linewidth=0.8)
    plt.axhline(-3.0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Transmissibility (dB)")
    plt.title(f"Run {run_number} ({axis}-axis): Transmissibility (dB)")
    plt.grid(which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    p3 = os.path.join(
        output_dir, f"run_{run_number:02d}_{axis}_transmissibility_db.png"
    )
    plt.savefig(p3, dpi=180)
    plt.close()
    created_files.append(p3)

    for metric, values in ratios.items():
        summary_rows.append(summarize_curve(f, values, metric, run_number, axis))

    return created_files, summary_rows


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    axis_lookup = {r.run_number: r.axis_label for r in RUNS_DEFAULT}
    all_summaries: List[Dict[str, float]] = []
    all_outputs: List[str] = []

    for run in args.runs:
        axis = axis_lookup.get(run, "Unknown")
        run_csv = os.path.join(args.data_dir, f"Run {run}.csv")
        if not os.path.exists(run_csv):
            print(f"Skipping Run {run}: file not found at {run_csv}")
            continue

        print(f"Processing Run {run} ({axis}) from {run_csv}")
        run_df = load_run_psd(run_csv)
        created, summaries = make_run_plots(run_df, run, axis, args.output_dir)
        all_outputs.extend(created)
        all_summaries.extend(summaries)

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries).sort_values(["run", "metric"])
        summary_csv = os.path.join(args.output_dir, "vibration_transmissibility_summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        all_outputs.append(summary_csv)

    print("\nCreated files:")
    for path in all_outputs:
        print(f" - {path}")


if __name__ == "__main__":
    main()
