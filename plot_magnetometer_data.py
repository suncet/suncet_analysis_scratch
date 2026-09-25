"""Plot legacy timestamp/X/Y/Z or elapsed-time/Bx/By/Bz magnetometer CSVs.

Field values are plotted in microtesla, as in the original plotting script.
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


ROD_COLUMNS = [f"torq_rod_{i}_state" for i in range(1, 4)]
STATE_COLORS = {
    "OFF": "#e4e7ec", "ON_POS": "#e9a343", "ON_NEG": "#74a9dc",
    "AUTO": "#b9a4d5", "UNKNOWN": "#ffffff",
}


def load_rod_states(csv_file):
    states = pd.read_csv(csv_file, skipinitialspace=True)
    states.columns = states.columns.str.strip()
    required = ["utc_time", *ROD_COLUMNS]
    if not set(required).issubset(states.columns) or states.empty:
        raise ValueError(f"State CSV must contain nonempty {required}")
    states["utc"] = pd.to_datetime(
        states["utc_time"].str.strip(), format="%y/%j-%H:%M:%S", utc=True
    )
    if states["utc"].isna().any() or not (states["utc"].diff().dropna() > pd.Timedelta(0)).all():
        raise ValueError("State UTC timestamps must be valid and strictly increasing")
    for col in ROD_COLUMNS:
        states[col] = states[col].str.strip()
        if not states[col].isin(set(STATE_COLORS) - {"UNKNOWN"}).all():
            raise ValueError(f"Unrecognized or missing state in {col}")
    return states


def find_field_transitions(df):
    """Detect changes in both the field mean and short-timescale variability.

    Bin by supplied elapsed time for analysis only; raw plots retain sample order.
    Compare adjacent 1 s windows of 0.1 s bin means/standard deviations.
    Thresholds are in the source field units (assumed microtesla).
    """
    bins = df.groupby(np.floor(df["time"] / 0.1).astype(int))[["X", "Y", "Z"]]
    features = bins.agg(["mean", "std"]).fillna(0)
    # Leave gaps missing so windows cannot bridge a missing interval.
    features = features.reindex(range(features.index.min(), features.index.max() + 1))
    left = features.rolling(10).mean().shift(1).to_numpy()
    right = features.iloc[::-1].rolling(10).mean().iloc[::-1].to_numpy()
    strength = np.linalg.norm(right - left, axis=1)
    peaks, _ = find_peaks(
        np.nan_to_num(strength), height=0.65, prominence=0.4, distance=30
    )
    return features.index.to_numpy()[peaks] * 0.1


def align_rod_states(df, states, mag_start_utc=None):
    """Fit one clock offset by consensus; never warp time or alter logged states."""
    if not pd.api.types.is_numeric_dtype(df["time"]):
        raise ValueError("Rod-state alignment currently requires elapsed-time magnetometer data")
    observed = find_field_transitions(df)
    command_time = (states["utc"] - states["utc"].iloc[0]).dt.total_seconds().to_numpy()
    tolerance = 1.5  # Whole-second command log; allow roughly one second of response scatter.
    estimated = mag_start_utc is None
    alternative = None
    if estimated:
        if min(len(observed), len(states)) < 4:
            raise ValueError("Too few transitions to estimate alignment; supply --mag-start-utc")
        solutions = []
        for offset in (observed[:, None] - command_time).ravel():
            for _ in range(3):
                nearest = np.abs(observed[:, None] - (command_time + offset)).argmin(axis=0)
                offsets = observed[nearest] - command_time
                inliers = np.abs(offsets - offset) <= tolerance
                if inliers.any():
                    offset = np.median(offsets[inliers])
            residual = np.min(np.abs(observed[:, None] - (command_time + offset)), axis=0)
            solutions.append((int((residual <= tolerance).sum()),
                              -np.minimum(residual, tolerance).sum(), float(offset)))
        best = max(solutions)
        alternatives = [s for s in solutions if abs(s[2] - best[2]) > 3]
        alternative = max(alternatives) if alternatives else None
        if best[0] < max(4, np.ceil(0.65 * len(states))):
            raise ValueError("Insufficient timing agreement; supply --mag-start-utc")
        if alternative and alternative[0] == best[0]:
            raise ValueError("Ambiguous timing alignment; supply --mag-start-utc")
        offset = best[2]
        origin = states["utc"].iloc[0] - pd.to_timedelta(offset, unit="s")
    else:
        origin = pd.Timestamp(mag_start_utc)
        origin = origin.tz_localize("UTC") if origin.tzinfo is None else origin.tz_convert("UTC")
        if pd.isna(origin):
            raise ValueError("Invalid --mag-start-utc")
        offset = (states["utc"].iloc[0] - origin).total_seconds()
    aligned = states.copy()
    aligned["elapsed_s"] = command_time + offset
    aligned["observed_s"] = np.nan
    aligned["residual_s"] = np.nan
    if len(observed):
        nearest = np.abs(observed[:, None] - aligned["elapsed_s"].to_numpy()).argmin(axis=0)
        residual = observed[nearest] - aligned["elapsed_s"].to_numpy()
        matched = np.abs(residual) <= tolerance
        aligned.loc[matched, "observed_s"] = observed[nearest[matched]]
        aligned.loc[matched, "residual_s"] = residual[matched]
    return aligned, origin, observed, estimated, alternative


def load_magnetometer(csv_file):
    df = pd.read_csv(csv_file, comment="#")
    df.columns = df.columns.str.strip()
    if {"time", "Bx", "By", "Bz"}.issubset(df.columns):
        df = df.rename(columns={"Bx": "X", "By": "Y", "Bz": "Z"})
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df.loc[~np.isfinite(df["time"]), "time"] = np.nan
        time_label = "Elapsed time (s)"
    elif {"X", "Y", "Z"}.issubset(df.columns):
        df["time"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        time_label = "Time"
    else:
        raise ValueError("Expected time/Bx/By/Bz or timestamp/X/Y/Z columns")

    has_total = "Btotal" in df.columns
    fields = ["X", "Y", "Z"] + (["Btotal"] if has_total else [])
    for col in fields:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[~np.isfinite(df[col]), col] = np.nan
    original_count = len(df)
    df = df.dropna(subset=["time", *fields]).copy()
    if len(df) < original_count:
        print(f"Dropped {original_count - len(df):,} invalid rows")
    if df.empty:
        raise ValueError("No valid magnetometer samples found")
    if not has_total:
        df["Btotal"] = np.sqrt((df[["X", "Y", "Z"]] ** 2).sum(axis=1))
    if not df["time"].is_monotonic_increasing:
        print("Warning: time steps backward; retaining the original sample order")
    return df, time_label, has_total


def plot_magnetometer(csv_file, output_dir, show=False):
    df, time_label, has_total = load_magnetometer(csv_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    title = f"SunCET magnetometer\n{csv_file.name}"
    notes = ["Units: µT (following the original script; CSV does not specify units)."]
    if not df["time"].is_monotonic_increasing:
        notes.append("Backward timestamp step present; acquisition order preserved.")
    note = "\n".join(notes)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(title)
    for ax, component, color in zip(
        axes, ["X", "Y", "Z"], ["tomato", "limegreen", "dodgerblue"]
    ):
        ax.plot(df["time"], df[component], color=color, linewidth=0.7)
        ax.set_ylabel(f"{component} (µT)")
        ax.grid(alpha=0.25)
        ax.margins(x=0)
    axes[-1].set_xlabel(time_label)
    fig.text(0.01, 0.015, note, fontsize=8, color="dimgray")
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    total_fig, ax = plt.subplots(figsize=(12, 4.5))
    total_label = "Recorded total field" if has_total else "Calculated total field"
    ax.plot(df["time"], df["Btotal"], color="purple", linewidth=0.7)
    ax.set_title(f"{title}\n{total_label}")
    ax.set_ylabel("Total field (µT)")
    ax.set_xlabel(time_label)
    ax.grid(alpha=0.25)
    ax.margins(x=0)
    total_fig.text(0.01, 0.025, note, fontsize=8, color="dimgray")
    total_fig.tight_layout(rect=(0, 0.10, 1, 1))

    for figure, suffix in [(fig, "components"), (total_fig, "total_field")]:
        output_path = output_dir / f"{csv_file.stem}_{suffix}.png"
        figure.savefig(output_path, dpi=180)
        print(f"Wrote {output_path}")
    print(f"Plotted {len(df):,} samples: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    if show:
        plt.show()
    plt.close(fig)
    plt.close(total_fig)


def plot_with_rod_states(csv_file, states_file, output_dir, mag_start_utc=None, show=False):
    df, _, _ = load_magnetometer(csv_file)
    states = load_rod_states(states_file)
    states, origin, observed, estimated, alternative = align_rod_states(df, states, mag_start_utc)
    output_dir.mkdir(parents=True, exist_ok=True)
    matched = states["observed_s"].notna()
    extra = observed[np.all(
        np.abs(observed[:, None] - states["elapsed_s"].to_numpy()) > 1.5, axis=1
    )]
    start, end = df["time"].min(), df["time"].max()
    origin_label = origin.strftime("%Y-%m-%d %H:%M:%S.%f")[:-5] + " UTC"
    mode = "PROVISIONAL transition fit" if estimated else "Supplied UTC origin"

    fig, axes = plt.subplots(
        5, 1, figsize=(15, 12), sharex=True,
        gridspec_kw={"height_ratios": [1.7, 1.7, 1.7, 1.4, 1.1]},
    )
    fig.suptitle(
        f"SunCET magnetometer and logged torque rod states\n"
        f"{mode}: t = 0 at {origin_label}", fontsize=14, y=0.985,
    )
    for ax, col, color in zip(axes[:4], ["X", "Y", "Z", "Btotal"],
                              ["#d24d38", "#288443", "#207cc1", "#7b4397"]):
        ax.plot(df["time"], df[col], color=color, linewidth=0.55)
        ax.set_ylabel(f"{'Total' if col == 'Btotal' else col} (µT)")
        ax.grid(axis="y", alpha=0.2)
        for row in states.itertuples():
            missing = pd.isna(row.observed_s)
            ax.axvline(row.elapsed_s, color="#bc3030" if missing else "#667085",
                       linestyle="--", alpha=0.9 if missing else 0.35,
                       linewidth=1.2 if missing else 0.65)
        for t in extra:
            ax.axvline(t, color="#aa6900", linestyle=":", linewidth=1.2)
    for row in states.loc[~matched].itertuples():
        if start <= row.elapsed_s <= end:
            changed = [i for i, col in enumerate(ROD_COLUMNS, 1)
                       if row.Index == 0 or getattr(row, col) != states.loc[row.Index - 1, col]]
            label = ", ".join(f"rod {i} {getattr(row, ROD_COLUMNS[i - 1])}" for i in changed)
            axes[0].text(row.elapsed_s + 1, 0.94,
                         f"Logged {label}\nNo matched field change",
                         transform=axes[0].get_xaxis_transform(), va="top",
                         fontsize=8, color="#a52525")
    secondary = axes[0].secondary_xaxis("top")
    secondary.xaxis.set_major_formatter(FuncFormatter(
        lambda seconds, _: (origin + pd.to_timedelta(seconds, unit="s")).strftime("%H:%M:%S")
    ))
    secondary.set_xlabel(f"{'Estimated' if estimated else 'Supplied'} UTC on {origin:%Y-%m-%d}")

    lane = axes[-1]
    short = {"OFF": "Off", "ON_POS": "+", "ON_NEG": "−", "AUTO": "Auto", "UNKNOWN": "?"}
    for rod, col in enumerate(ROD_COLUMNS):
        # Merge adjacent intervals with the same state for legible labels.
        boundaries = [(start, "UNKNOWN")]
        for row in states.itertuples():
            t, state = row.elapsed_s, getattr(row, col)
            if t <= start:
                boundaries = [(start, state)]
            elif t <= end and state != boundaries[-1][1]:
                boundaries.append((t, state))
        y = 2 - rod
        for i, (left, state) in enumerate(boundaries):
            right = boundaries[i + 1][0] if i + 1 < len(boundaries) else end
            lane.broken_barh([(left, right - left)], (y - 0.38, 0.76),
                             facecolors=STATE_COLORS[state], edgecolors="#777777",
                             linewidth=0.5, hatch="///" if state == "UNKNOWN" else None)
            if right - left >= 5:
                lane.text((left + right) / 2, y, short[state], ha="center", va="center", fontsize=8)
    lane.set_yticks([2, 1, 0], ["Rod 1", "Rod 2", "Rod 3"])
    lane.set_ylim(-0.6, 2.6)
    lane.set_xlabel("Magnetometer elapsed time (s)")
    lane.set_xlim(start, end)
    fig.legend(handles=[Patch(facecolor=color, edgecolor="#777777", label=state,
                              hatch="///" if state == "UNKNOWN" else None)
                        for state, color in STATE_COLORS.items()],
               loc="lower center", bbox_to_anchor=(0.5, 0.092), ncol=5, frameon=False)
    notes = [
        f"Timing agreement: {matched.sum()}/{len(states)} log entries within ±1.5 s of detected field changes. "
        "Dashed lines: logged transitions; red: unmatched log; dotted amber: unmatched field change.",
        "State bands show the log, not independently verified actuation. States persist until the next entry; the last state is carried to the recording end.",
        "Units follow the original script (µT). Raw timing and sample order retained, including the backward timestamp step."
        if not df["time"].is_monotonic_increasing else "Units follow the original script (µT). Raw timing and sample order retained.",
    ]
    if estimated:
        notes.insert(1, "Alignment is inferred from transition timing, not measured absolute timestamps. Treat the origin as approximate to about a second.")
    fig.text(0.09, 0.025, "\n".join(notes), fontsize=8, color="#475467", linespacing=1.5)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.865, bottom=0.18, hspace=0.18)
    plot_path = output_dir / f"{csv_file.stem}_with_rod_states.png"
    fig.savefig(plot_path, dpi=180)
    print(f"Wrote {plot_path}")

    report = [
        "# Magnetometer / torque rod alignment", "",
        f"- Magnetometer source: `{csv_file}`", f"- State source: `{states_file}`",
        f"- {mode}: magnetometer t = 0 corresponds to **{origin_label}**.",
        f"- UTC = origin + the recorded elapsed seconds; one constant offset, no drift correction.",
        f"- {matched.sum()}/{len(states)} entries match detected changes within ±1.5 s.",
        "- States are displayed as logged. Before the first entry they are unknown; afterwards each entry is held until the next, including the last entry to the end.",
        "- Field units are assumed µT, following the original script. Original timing and sample order are retained in the plot.",
        "", "## Method", "",
        "Detect changes using adjacent 1 s windows of 0.1 s bin means and standard deviations of all three components. "
        "The change score is the Euclidean norm of the six feature differences; peaks require height 0.65, prominence 0.4, and separation 3 s. "
        "Analysis bins use the supplied elapsed times; the raw data are not corrected. "
        "For an estimated origin, try event-pair offsets, refine with the median of matches within 1.5 s, and maximize the number of matching entries "
        "(break ties by summed absolute residuals capped at 1.5 s). The first log row is a timing candidate, although its preceding state is unknown. "
        "Timing matches do not establish which rod caused a change. Residual = detected time minus predicted time.",
    ]
    if estimated:
        report += ["", "This is a provisional alignment with roughly one-second practical precision, not an absolute clock calibration or a statistical confidence interval."]
    if alternative:
        report += [f"The best other solution separated by more than 3 s matches {alternative[0]}/{len(states)} entries (first log entry at magnetometer t={alternative[2]:.1f} s)."]
    report += ["", "## Event comparison", "",
               "| Logged UTC | Rod 1 | Rod 2 | Rod 3 | Predicted elapsed (s) | Detected (s) | Residual (s) |",
               "|---|---|---|---|---:|---:|---:|"]
    for row in states.itertuples():
        found = "unmatched" if pd.isna(row.observed_s) else f"{row.observed_s:.1f}"
        residual = "—" if pd.isna(row.residual_s) else f"{round(row.residual_s, 1) + 0.0:+.1f}"
        report.append(f"| {row.utc:%H:%M:%S} | {row.torq_rod_1_state} | {row.torq_rod_2_state} | {row.torq_rod_3_state} | {row.elapsed_s:.1f} | {found} | {residual} |")
    report += ["", "Detected field changes with no log entry within 1.5 s: " +
               (", ".join(f"{t:.1f} s" for t in extra) or "none") + ".",
               "Unmatched log entries and field changes are not automatically paired or used to rewrite the log.", ""]
    report_path = output_dir / f"{csv_file.stem}_alignment.md"
    report_path.write_text("\n".join(report))
    print(f"Wrote {report_path}")
    print(f"{mode}: {origin_label}; matched {matched.sum()}/{len(states)}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", nargs="?", type=Path, help="Magnetometer CSV")
    parser.add_argument("--output-dir", type=Path, help="Directory for PNG plots")
    parser.add_argument("--show", action="store_true", help="Also open interactive plots")
    parser.add_argument("--rod-states", type=Path, help="CSV of UTC torque rod states")
    alignment = parser.add_mutually_exclusive_group()
    alignment.add_argument("--estimate-alignment", action="store_true", help="Infer a provisional UTC offset from field transitions")
    alignment.add_argument("--mag-start-utc", help="Supplied UTC corresponding to magnetometer t=0, e.g. 2026-09-23T22:32:36.3Z")
    args = parser.parse_args()
    if args.rod_states and not (args.estimate_alignment or args.mag_start_utc):
        parser.error("Use --estimate-alignment or --mag-start-utc with --rod-states")
    if (args.estimate_alignment or args.mag_start_utc) and not args.rod_states:
        parser.error("Alignment options require --rod-states")
    csv_file = args.csv_file
    if csv_file is None:
        data_root = os.getenv("suncet_data")
        if not data_root:
            parser.error("Provide a CSV path or set suncet_data for the legacy default")
        csv_file = (
            Path(data_root) / "test_data" / "2026-04-06_post_vibe_cpt"
            / "suncet_tr_20260406_125316.csv"
        )
    csv_file = csv_file.expanduser().resolve()
    output_dir = args.output_dir or (
        Path(__file__).resolve().parent / "magnetometer_analysis_output" / csv_file.parent.name
    )
    output_dir = output_dir.expanduser().resolve()
    if args.rod_states:
        plot_with_rod_states(csv_file, args.rod_states.expanduser().resolve(), output_dir,
                             mag_start_utc=args.mag_start_utc, show=args.show)
    else:
        plot_magnetometer(csv_file, output_dir, show=args.show)


if __name__ == "__main__":
    main()
