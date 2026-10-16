#!/usr/bin/env python3
"""Reproducible TVAC2 light-leak analysis; original data are read only.

Capture-source META packets define integration intervals. EPS activity is sampled
at 1 Hz, so the primary closed/open split brackets the first active timer sample
by [-1, +0.5] seconds. Every complete exposure remains in transient diagnostics.
The cold baseline uses the final continuous acquisition run, not the earlier run
separated by a nine-minute gap. Units remain DN; no gain calibration is assumed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import gzip
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tvac2_light_leak_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy import stats

DEFAULT_DATA = Path("/Users/masonjp2/Dropbox/suncet_dropbox/9000 Processing/data/test_data/2026-10-07_tvac2")
DEFAULT_OUT = Path(__file__).resolve().parent / "tvac2_light_leak_output"
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)
TESTS = {"cold": ("2026-10-14", "19:17"), "hot": ("2026-10-15", "17:10")}


def utc(coarse, fine=0):
    # Match the exported image inventory / telemetry dashboard UTC convention.
    return EPOCH + timedelta(seconds=int(coarse) + 5, milliseconds=int(fine))


def block_mean(a, size):
    """Include partial edge blocks with their actual pixel counts."""
    y, x = np.arange(0, a.shape[-2], size), np.arange(0, a.shape[-1], size)
    sums = np.add.reduceat(np.add.reduceat(a, y, axis=-2), x, axis=-1)
    counts = np.diff(np.r_[y, a.shape[-2]])[:, None] * np.diff(np.r_[x, a.shape[-1]])[None, :]
    return sums / counts


def load_records(data):
    rows = list(csv.DictReader((data / "csie_images/csie_image_inventory.csv").open()))
    eps_file = next((data / "public_dashboard").glob("*apid-0030-*.json.gz"))
    with gzip.open(eps_file, "rt") as f:
        eps = json.load(f)
    cols = {k: i for i, k in enumerate(eps["columns"])}
    result = {}
    for name, (day, hm) in TESTS.items():
        events = [utc(r[cols["packet_time_sec"]], r[cols["packet_time_sub"]])
                  for r in eps["rows"]
                  if r[cols["eps_deploy_select"]] == "DOOR" and r[cols["eps_deploy_time"]] > 0
                  and utc(r[cols["packet_time_sec"]]).strftime("%Y-%m-%dT%H:%M") == f"{day}T{hm}"]
        if len(events) != 1:
            raise ValueError(f"Expected one EPS deployment marker for {name}: {events}")
        event = events[0]
        records = []
        for row in rows:
            if not row["metadata_time_iso8601"].startswith(day):
                continue
            meta = json.loads(Path(row["meta_json_path"]).read_text())
            packets = meta.get("metadata_packets", [])
            caps = {(p["SHCOARSE"], p["SHFINE"], p["csie_meta_intg_ms"]) for p in packets
                    if p.get("csie_meta_meta_src") == 0}
            if not caps:
                raise ValueError(f"No capture timestamp for image {row['image_id']}")
            starts = [utc(c, f) for c, f, ms in caps]
            ends = [utc(c, f) + timedelta(milliseconds=ms) for c, f, ms in caps]
            start, end = min(starts), max(ends)
            complete = (row["complete_uncompressed_image"] == "True"
                        and int(row["selected_checksum_valid_rows"]) == int(row["expected_rows"])
                        and int(row["selected_checksum_failed_rows"]) == 0
                        and int(row["selected_checksum_missing_rows"]) == 0
                        and int(row["zero_filled_total_rows"]) == 0
                        and int(row["boundary_fill_repaired_rows"]) == 0)
            dt = (start - event).total_seconds()
            dt_end = (end - event).total_seconds()
            # The first cold acquisition segment is an independent older control.
            phase = "earlier_closed" if dt < -120 else (
                "closed" if dt_end <= -1 else "open" if dt >= .5 else "transition")
            records.append(dict(image_id=int(row["image_id"]), path=row["fits_path"],
                                capture_start_utc=start.isoformat(), capture_end_utc=end.isoformat(),
                                dt=dt, dt_mid=(dt + dt_end) / 2, dt_end=dt_end,
                                exposure_s=(end - start).total_seconds(),
                                complete=complete, phase=phase,
                                temp=meta["detector_temp"], adc_temp=meta["csie_meta_adc_thermistor_temp"],
                                fpga_temp=meta["csie_meta_fpga_temp"],
                                integration_ms=meta["csie_meta_intg_ms"],
                                row_bin=meta["fpm_proc_cfg_row_bin_meta"],
                                col_bin=meta["fpm_proc_cfg_col_bin_meta"],
                                right_shift=meta["fpm_proc_cfg_right_shift_meta"],
                                stim_lamp=meta["csie_meta_stim_lamp_stat"],
                                test_mode=meta["csie_meta_test_mode"],
                                warnings=row["warnings"]))
        result[name] = dict(event_utc=event.isoformat(), records=sorted(records, key=lambda r: r["dt"]))
    return result


class Moments:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)

    def add(self, a):
        self.n += 1
        delta = a - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (a - self.mean)


def first_pass(name, test, out, refresh=False):
    cache = out / f"{name}_cache.npz"
    metrics_file = out / f"{name}_frames.csv"
    # Cache is tied to the image file metadata, configuration, and deployment split.
    import hashlib
    fingerprint = hashlib.sha256(json.dumps(
        [(r, Path(r["path"]).stat().st_size, Path(r["path"]).stat().st_mtime_ns)
         for r in test["records"]], sort_keys=True).encode()).hexdigest()
    if cache.exists() and not refresh:
        z = np.load(cache)
        if str(z["fingerprint"]) == fingerprint:
            rows = list(csv.DictReader(metrics_file.open()))
            for r in rows:
                for k in r:
                    if k not in ["phase", "capture_start_utc", "capture_end_utc"]:
                        r[k] = float(r[k])
            return dict(z), rows
    good = [r for r in test["records"] if r["complete"]]
    shape = fits.getdata(good[0]["path"]).shape
    groups = {p: Moments(shape) for p in ["closed", "open", "earlier_closed", "transition"]}
    cube, row_profiles, col_profiles, metrics = [], [], [], []
    for i, r in enumerate(good):
        with fits.open(r["path"], memmap=False) as hdul:
            raw = hdul[0].data
            if raw.shape != shape or raw.dtype != np.uint16:
                raise ValueError(f"Unexpected shape/dtype: {r['path']}")
            a = raw.astype(np.float64)
        if (r["exposure_s"] != 1 or r["integration_ms"] != 1000
            or r["right_shift"] or r["row_bin"] or r["col_bin"]
            or r["stim_lamp"] or r["test_mode"]):
            raise ValueError(f"Unexpected acquisition settings in {r['image_id']}")
        h = np.bincount(raw.ravel(), minlength=65536)
        quantiles = np.searchsorted(h.cumsum(), np.array([.01, .5, .99, .999]) * raw.size)
        groups[r["phase"]].add(a)
        cube.append(block_mean(a, 8).astype(np.float32))
        row_profiles.append(a.mean(axis=1).astype(np.float32))
        col_profiles.append(a.mean(axis=0).astype(np.float32))
        m = {k: r[k] for k in ["image_id", "capture_start_utc", "capture_end_utc", "dt", "dt_mid", "dt_end", "temp", "adc_temp", "fpga_temp", "phase"]}
        m.update(mean=float(a.mean()), std=float(a.std()), q01=int(quantiles[0]),
                 median=int(quantiles[1]), q99=int(quantiles[2]), q999=int(quantiles[3]),
                 minimum=int(raw.min()), maximum=int(raw.max()), zero_pixels=int(h[0]),
                 at_65535=int(h[-1]), above_16000=int(h[16000:].sum()), above_50000=int(h[50000:].sum()))
        metrics.append(m)
        if i % 80 == 0:
            print(f"{name}: {i + 1}/{len(good)} complete images", flush=True)
    arrays = dict(fingerprint=np.array(fingerprint), cube8=np.stack(cube),
                  row_profiles=np.stack(row_profiles), col_profiles=np.stack(col_profiles))
    for phase, moments in groups.items():
        arrays[f"{phase}_n"] = np.array(moments.n)
        arrays[f"{phase}_mean"] = moments.mean.astype(np.float32)
        arrays[f"{phase}_var"] = (moments.m2 / max(1, moments.n - 1)).astype(np.float32)
    np.savez_compressed(cache, **arrays)
    with metrics_file.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    return arrays, metrics


def linear_fit(t, y, hac_lags=3):
    """OLS with a small-sample adjusted Newey-West covariance for scalar series."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    x = np.column_stack([np.ones(len(t)), t])
    inv = np.linalg.inv(x.T @ x)
    beta = inv @ x.T @ y
    resid = y - x @ beta
    scores = x * resid[:, None]
    meat = scores.T @ scores
    for lag in range(1, min(hac_lags, len(t) - 3) + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (hac_lags + 1)) * (cross + cross.T)
    hac = inv @ meat @ inv * len(t) / (len(t) - 2)
    iid = inv * (resid @ resid) / (len(t) - 2)
    # Do not claim more precision from a noisy short-segment HAC estimate.
    intercept_var = max(float(hac[0, 0]), float(iid[0, 0]))
    return dict(beta=beta, cov=hac, iid_cov=iid, intercept_var=intercept_var,
                residual=resid, rms=float(np.sqrt(np.mean(resid**2))), n=len(t))


def estimate_step(t, y, before, after):
    left, right = linear_fit(t[before], y[before]), linear_fit(t[after], y[after])
    delta = float(right["beta"][0] - left["beta"][0])
    v1, v2 = left["intercept_var"], right["intercept_var"]
    se = float(np.sqrt(v1 + v2))
    df = (v1 + v2)**2 / (v1**2 / (left["n"] - 2) + v2**2 / (right["n"] - 2))
    upper95 = delta + stats.t.ppf(.95, df) * se
    return dict(step_dn=delta, se_dn=se, df=float(df),
                upper95_dn=float(max(0, delta) + stats.t.ppf(.95, df) * se),
                ci95_dn=[float(delta - stats.t.ppf(.975, df) * se),
                         float(delta + stats.t.ppf(.975, df) * se)],
                p_positive=float(stats.t.sf(delta / se, df)),
                closed_slope_dn_s=float(left["beta"][1]), open_slope_dn_s=float(right["beta"][1]),
                closed_rms_dn=left["rms"], open_rms_dn=right["rms"],
                n_closed=left["n"], n_open=right["n"])


def inspect_statistics(name, arrays, rows, out):
    t = np.array([r["dt_mid"] for r in rows])
    y = np.array([r["mean"] for r in rows])
    phase = np.array([r["phase"] for r in rows])
    before = phase == "closed"
    after = phase == "open"
    summary = dict(n_complete=len(rows), counts={p: int(np.sum(phase == p)) for p in np.unique(phase)},
                   mean_closed_dn=float(y[before].mean()), mean_open_dn=float(y[after].mean()),
                   raw_difference_dn=float(y[after].mean() - y[before].mean()),
                   max_pixel_value=int(max(r["maximum"] for r in rows)),
                   zero_pixels=int(sum(r["zero_pixels"] for r in rows)),
                   at_65535=int(sum(r["at_65535"] for r in rows)), steps={})
    for window in [15, 20, 30, 40, 60, 90, 400]:
        b, a = before & (t > -window), after & (t < window)
        if min(b.sum(), a.sum()) >= 8:
            summary["steps"][str(window)] = estimate_step(t, y, b, a)
    # Prospective prediction: use ONLY closed frames for the drift correction.
    lf = linear_fit(t[before], y[before])
    shift = float(t[after].mean() - t[before].mean())
    predicted_change = float(lf["beta"][1] * shift)
    summary["closed_only_prediction"] = dict(predicted_drift_dn=predicted_change,
        remaining_difference_dn=summary["raw_difference_dn"] - predicted_change,
        mean_time_separation_s=shift)
    # Timeline: each subplot retains all complete frames, including transitions.
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, constrained_layout=True)
    recent = t > -120
    colors = {"closed": "#2563eb", "open": "#e87925", "transition": "#b5179e", "earlier_closed": "#64748b"}
    fields = [("mean", "Whole-image mean (DN)"), ("median", "Whole-image median (DN)"),
              ("q999", "99.9th percentile (DN)"), ("std", "Spatial standard deviation (DN)"),
              ("temp", "Detector temperature (°C)"), ("adc_temp", "ADC temperature (°C)")]
    for ax, (field, label) in zip(axes.flat, fields):
        for p in ["closed", "transition", "open"]:
            sel = (phase == p) & recent
            ax.scatter(t[sel], [rows[i][field] for i in np.flatnonzero(sel)], s=13, color=colors[p], label=p)
        ax.axvspan(-1, .5, color="#b5179e", alpha=.15)
        ax.axvline(0, color="black", ls=":", lw=1)
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    xx = np.linspace(t[before].min(), min(t.max(), 90), 200)
    axes[0, 0].plot(xx, lf["beta"][0] + lf["beta"][1] * xx, color="#2563eb", ls="--", lw=1.5, label="closed-only trend")
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Seconds from EPS door-deploy marker (capture midpoint)")
    fig.suptitle(f"TVAC2 {name}: all complete images, including door motion", fontsize=16)
    fig.savefig(out / f"{name}_timeline.png", dpi=160)
    plt.close(fig)
    # The raw difference and the prospectively drift-corrected difference are BOTH kept.
    pre, post = arrays["closed_mean"], arrays["open_mean"]
    difference = post.astype(float) - pre
    corrected = difference - predicted_change
    np.savez_compressed(out / f"{name}_differences.npz", closed_mean_dn=pre,
                        open_mean_dn=post, raw_difference_dn=difference.astype(np.float32),
                        closed_trend_corrected_dn=corrected.astype(np.float32))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    lo, hi = np.percentile(pre, [1, 99])
    for ax, image, title in zip(axes[0, :2], [pre, post], ["Closed-door mean", "Open-door mean"]):
        im = ax.imshow(image, origin="lower", cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(title + " · identical stretch")
        fig.colorbar(im, ax=ax, shrink=.7, label="DN")
    for ax, image, title in [(axes[0, 2], difference, "Raw open − closed"),
                              (axes[1, 0], block_mean(difference, 8), "Raw difference · 8×8 mean"),
                              (axes[1, 1], block_mean(difference, 32), "Raw difference · 32×32 mean"),
                              (axes[1, 2], block_mean(corrected, 32), "32×32 · closed-only drift removed")]:
        vmax = max(.5, float(np.percentile(np.abs(image), 99)))
        im = ax.imshow(image, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, shrink=.7, label="DN / pixel")
    for ax in axes.flat:
        ax.set_xlabel("Detector column / bin index")
        ax.set_ylabel("Detector row / bin index")
    fig.suptitle(f"TVAC2 {name}: averages and differences · no image normalization", fontsize=16)
    fig.savefig(out / f"{name}_differences.png", dpi=160)
    plt.close(fig)
    return summary


def field_step(t, values, before, after, lags=3):
    """Intercept difference at deploy, retaining each cell's own linear drift.

    Both fitted intercepts are retained; this cannot normalize away a uniform
    step. The two slopes may differ. HAC variances use temporal residuals from
    whole images, not the spatial scatter divided by a pixel count.
    """
    shape = values.shape[1:]
    intercepts, variances, counts, residuals = [], [], [], []
    for sel in [before, after]:
        ts = t[sel]
        ys = values[sel].reshape(sel.sum(), -1).astype(float)
        x = np.column_stack([np.ones(len(ts)), ts])
        inv = np.linalg.inv(x.T @ x)
        beta = inv @ x.T @ ys
        resid = ys - x @ beta
        w = x @ inv[:, 0]
        score = w[:, None] * resid
        hac = np.sum(score**2, axis=0)
        for lag in range(1, min(lags, len(ts)-3) + 1):
            hac += 2 * (1 - lag / (lags + 1)) * np.sum(score[lag:] * score[:-lag], axis=0)
        hac *= len(ts) / (len(ts) - 2)
        iid = np.sum(resid**2, axis=0) / (len(ts) - 2) * inv[0, 0]
        intercepts.append(beta[0])
        variances.append(np.maximum(hac, iid))
        counts.append(len(ts))
        residuals.append(resid)
    effect = intercepts[1] - intercepts[0]
    v1, v2 = variances
    se = np.sqrt(v1 + v2)
    df = (v1 + v2)**2 / (v1**2 / (counts[0] - 2) + v2**2 / (counts[1] - 2))
    return effect.reshape(shape), se.reshape(shape), df.reshape(shape), residuals


def multiscale_checks(name, arrays, rows, out):
    t = np.array([r["dt_mid"] for r in rows])
    y = np.array([r["mean"] for r in rows])
    phase = np.array([r["phase"] for r in rows])
    before, after = (phase == "closed") & (t > -40), (phase == "open") & (t < 40)
    # Native cube is reduced to 8x8 cells; partial edges are handled explicitly
    # for all larger scales by reducing original weighted pixel sums.
    cube = arrays["cube8"].astype(float)
    scales = [8, 32, 128]
    n_tests = sum(int(np.ceil(2000/s) * np.ceil(1504/s)) for s in scales) + 2000 + 1504 + 1
    diagnostics, maps = {}, {}
    for size in scales:
        if size == 8:
            data = cube
        else:
            # Size 8 divides both dimensions exactly; each base cell has 64 pixels.
            data = block_mean(cube, size // 8)
        step, se, df, _ = field_step(t, data, before, after)
        p = stats.t.sf(step / se, df)
        # Split the 1% spatial-screen budget equally between native and larger
        # regions, then between cold and hot. Bonferroni needs no spatial
        # independence assumption, although the temporal noise model still does.
        cut = stats.t.ppf(1 - .005 / (2*n_tests), df)
        rejected = (step > 0) & (step / se > cut)
        j = np.unravel_index(np.argmin(p), p.shape)
        diagnostics[str(size)] = dict(n_cells=int(step.size),
            positive_cells_at_global_1pct=int(rejected.sum()),
            strongest_cell_yx=[int(j[0]*size), int(j[1]*size)],
            strongest_step_dn=float(step[j]), strongest_se_dn=float(se[j]),
            smallest_one_sided_p=float(p[j]),
            median_se_dn=float(np.median(se)),
            median_threshold_dn=float(np.median(cut * se)),
            median_pointwise_upper95_dn=float(np.median(np.maximum(0, step) + stats.t.ppf(.95, df)*se)))
        maps[f"step_{size}"] = step.astype(np.float32)
        maps[f"se_{size}"] = se.astype(np.float32)
        maps[f"z_{size}"] = (step/se).astype(np.float32)
    for key, data in [("rows", arrays["row_profiles"]), ("columns", arrays["col_profiles"])]:
        step, se, df, _ = field_step(t, data, before, after)
        p = stats.t.sf(step/se, df)
        diagnostics[key] = dict(positive_at_global_1pct=int(np.sum(p < .005 / (2*n_tests))),
                                maximum_step_dn=float(step.max()), maximum_z=float(np.max(step/se)))
        maps[f"step_{key}"] = step.astype(np.float32)
        maps[f"se_{key}"] = se.astype(np.float32)
    # The source can illuminate any chosen region. Recovery is calculated for a
    # constant signal in one tested cell with all correlations/noise held fixed.
    # 95%-power approximation = search threshold + 1.645 SE, not a measured leak.
    recovery = {}
    for size in scales:
        step, se = maps[f"step_{size}"], maps[f"se_{size}"]
        data = cube if size == 8 else block_mean(cube, size // 8)
        _, _, df, _ = field_step(t, data, before, after)
        threshold = stats.t.ppf(1 - .005 / (2*n_tests), df)
        recovery[str(size)] = dict(approx_median_95pct_recovery_dn_per_pixel=float(np.median((threshold+1.645)*se)),
                                   approx_p90_95pct_recovery_dn_per_pixel=float(np.percentile((threshold+1.645)*se,90)))
    # Placebo deployments inside a single door state. These overlap and are
    # descriptive controls, not independent replicates or calibrated p-values.
    controls = {}
    for window in [12, 15, 20]:
        values = []
        for center in np.arange(t.min()+window, t.max()-window, 2):
            b = (t > center-window) & (t < center-1)
            a = (t > center+1) & (t < center+window)
            if min(b.sum(), a.sum()) < 8:
                continue
            if any(np.all(phase[b] == p) and np.all(phase[a] == p)
                   for p in ["closed", "earlier_closed", "open"]):
                s = estimate_step(t-center, y, b, a)
                values.append(dict(center_s=float(center), phase=str(phase[b][0]),
                                   step_dn=s["step_dn"], se_dn=s["se_dn"]))
        controls[str(window)] = values
    # Second time differences cancel a locally linear drift. Every complete
    # image in a continuous run, INCLUDING transition frames, is checked.
    previous_weight=(t[2:]-t[1:-1])/(t[2:]-t[:-2])
    next_weight=1-previous_weight
    d2=cube[1:-1]-previous_weight[:,None,None]*cube[:-2]-next_weight[:,None,None]*cube[2:]
    # Ordinary cadence is ~1.27 seconds. Do not bridge rejected/missing frames.
    continuous = (np.diff(t)[:-1] < 1.8) & (np.diff(t)[1:] < 1.8)
    stable_pre = continuous & (phase[1:-1] == "closed") & (phase[:-2] == "closed") & (phase[2:] == "closed")
    noise = np.std(d2[stable_pre], axis=0, ddof=1)
    # A robust floor prevents unusually quiet short-baseline cells dominating.
    noise = np.maximum(noise, np.median(noise)*.5)
    flash_z = np.max(d2 / noise, axis=(1, 2))
    flash_mean = np.mean(d2, axis=(1, 2))
    transition = (phase[1:-1] == "transition") & continuous
    flash = dict(transition_frame_ids=[int(rows[i+1]["image_id"]) for i in np.flatnonzero(transition)],
                 maximum_transition_cell_z=float(np.max(flash_z[transition])),
                 maximum_closed_control_cell_z=float(np.max(flash_z[stable_pre])),
                 transition_mean_d2_dn=[float(v) for v in flash_mean[transition]],
                 closed_mean_d2_rms_dn=float(np.std(flash_mean[stable_pre],ddof=1)),
                 explanation="8x8 cell second differences; isolated particle hits are not automatically classified as light leaks")
    np.savez_compressed(out / f"{name}_spatial_checks.npz", **maps)
    fig, ax = plt.subplots(2, 3, figsize=(15,9), constrained_layout=True)
    for k, size in enumerate(scales):
        effect = maps[f"step_{size}"]
        vmax = max(.3, float(np.percentile(np.abs(effect), 99)))
        im = ax[0,k].imshow(effect,origin="lower",cmap="RdBu_r",vmin=-vmax,vmax=vmax,interpolation="nearest")
        ax[0,k].set_title(f"{size}×{size} cells: step at deployment")
        fig.colorbar(im,ax=ax[0,k],label="DN / pixel",shrink=.75)
        im = ax[1,k].imshow(maps[f"z_{size}"],origin="lower",cmap="RdBu_r",vmin=-5,vmax=5,interpolation="nearest")
        ax[1,k].set_title("Step / temporal uncertainty")
        fig.colorbar(im,ax=ax[1,k],label="Standard errors",shrink=.75)
    for a in ax.flat:
        a.set_xlabel("Column bin")
        a.set_ylabel("Row bin")
    fig.suptitle(f"TVAC2 {name}: local before/after trends retain any uniform step",fontsize=15)
    fig.savefig(out / f"{name}_spatial_checks.png",dpi=160)
    plt.close(fig)
    return dict(spatial=diagnostics, approximate_recovery=recovery, placebo_controls=controls, flash_checks=flash)


def native_pixel_checks(name, test, arrays, out):
    cache = out / f"{name}_native_checks.npz"
    selected = [r for r in test["records"] if r["complete"] and abs(r["dt_mid"]) < 40
                and r["phase"] in ["closed", "open"]]
    signature = json.dumps(dict(method="native-v2-alpha0.005",fingerprint=str(arrays["fingerprint"]),
                               records=[(r["image_id"],r["phase"],r["dt_mid"]) for r in selected]))
    if cache.exists():
        z = np.load(cache)
        if str(z["signature"]) == signature:
            return json.loads(str(z["summary"]))
    shape = arrays["closed_mean"].shape
    intercepts, variances, counts = [], [], []
    for phase in ["closed", "open"]:
        records = [r for r in selected if r["phase"] == phase]
        t = np.array([r["dt_mid"] for r in records])
        centered = t - t.mean()
        sxx = np.sum(centered**2)
        moments = Moments(shape)
        sy = np.zeros(shape, float)
        for r, ti in zip(records, centered):
            a = fits.getdata(r["path"], memmap=False).astype(float)
            moments.add(a)
            sy += ti * a
        slope = sy / sxx
        intercept = moments.mean - slope*t.mean()
        iid = np.maximum(0, moments.m2 - slope**2*sxx) / (len(t)-2) * (1/len(t) + t.mean()**2/sxx)
        meat = np.zeros(shape, float)
        history = []
        for i, r in enumerate(records):
            a = fits.getdata(r["path"], memmap=False).astype(float)
            residual = a - intercept - slope*t[i]
            weight = 1/len(t) - centered[i]*t.mean()/sxx
            score = residual*weight
            meat += score**2
            for lag, prev in enumerate(reversed(history), start=1):
                meat += 2*(1-lag/4)*score*prev
            history.append(score)
            history = history[-3:]
        hac = meat * len(t)/(len(t)-2)
        intercepts.append(intercept)
        variances.append(np.maximum(iid, hac))
        counts.append(len(t))
    effect = intercepts[1]-intercepts[0]
    v1, v2 = variances
    se = np.sqrt(v1+v2)
    df = (v1+v2)**2 / (v1**2/(counts[0]-2) + v2**2/(counts[1]-2))
    p = stats.t.sf(effect/se, df)
    # Include native pixels and all previous spatial tests, for both tests.
    threshold = .005 / (2*(effect.size + 47000 + 2961 + 192 + 2000 + 1504 + 1))
    premean = arrays["closed_mean"]
    prevar = arrays["closed_var"]
    meanmad = np.median(np.abs(premean-np.median(premean)))*1.4826
    varmad = np.median(np.abs(prevar-np.median(prevar)))*1.4826
    baseline_bad = (premean > np.median(premean)+10*meanmad) | (prevar > np.median(prevar)+10*varmad)
    positives = (effect > 0) & (p < threshold)
    negatives = (effect < 0) & (stats.t.cdf(effect/se,df) < threshold)
    top = np.argpartition(p.ravel(), 10)[:10]
    top = top[np.argsort(p.ravel()[top])]
    candidates = []
    for index in top:
        y, x = np.unravel_index(index, shape)
        candidates.append(dict(row=int(y),column=int(x),step_dn=float(effect[y,x]),se_dn=float(se[y,x]),
                               p_positive=float(p[y,x]),baseline_bad=bool(baseline_bad[y,x])))
    # Full-frame means with a mask defined using CLOSED frames only. Keep the
    # all-pixel test alongside this; no post-deployment feature can enter the mask.
    masked_y=[]
    for r in selected:
        a=fits.getdata(r["path"],memmap=False)
        masked_y.append(float(a[~baseline_bad].mean()))
    t=np.array([r["dt_mid"] for r in selected]); phases=np.array([r["phase"] for r in selected])
    masked_step=estimate_step(t,np.array(masked_y),phases=="closed",phases=="open")
    summary=dict(n_pixels=int(effect.size),baseline_bad_pixels=int(baseline_bad.sum()),
                 positive_pixels_global_1pct=int(positives.sum()),
                 positive_unmasked_pixels_global_1pct=int((positives & ~baseline_bad).sum()),
                 negative_pixels_global_1pct=int(negatives.sum()),
                 negative_unmasked_pixels_global_1pct=int((negatives & ~baseline_bad).sum()),
                 symmetric_tail_counts={str(k):dict(positive=int(((effect/se>k)&~baseline_bad).sum()),
                                                    negative=int(((effect/se< -k)&~baseline_bad).sum())) for k in [4,5,6,7,8]},
                 strongest_candidates=candidates,masked_whole_frame_step=masked_step,
                 median_se_dn=float(np.median(se[~baseline_bad])),
                 median_95pct_recovery_dn=float(np.median((stats.t.ppf(1-threshold,df[~baseline_bad])+1.645)*se[~baseline_bad])))
    np.savez_compressed(cache,signature=np.array(signature),summary=np.array(json.dumps(summary)),
                        step_dn=effect.astype(np.float32),se_dn=se.astype(np.float32),baseline_bad=baseline_bad)
    return summary


def candidate_checks(name, test, arrays, rows, out):
    """Inspect increases and decreases without removing them from the results."""
    z=np.load(out/f"{name}_native_checks.npz")
    effect,se=z["step_dn"],z["se_dn"]
    zz=effect/se
    n=5
    pos=np.argpartition(-zz.ravel(),n)[:n]
    pos=pos[np.argsort(-zz.ravel()[pos])]
    neg=np.argpartition(zz.ravel(),n)[:n]
    neg=neg[np.argsort(zz.ravel()[neg])]
    coordinates=[tuple(map(int,np.unravel_index(i,effect.shape))) for i in np.r_[pos,neg]]
    good=[r for r in test["records"] if r["complete"] and r["dt_mid"]>-120]
    values,patches=[],[]
    for r in good:
        raw=fits.getdata(r["path"],memmap=False)
        values.append([raw[q] for q in coordinates])
        patches.append(np.array([raw[max(0,y-2):y+3,max(0,x-2):x+3] for y,x in coordinates]))
    values=np.array(values);patches=np.array(patches)
    t=np.array([r["dt_mid"] for r in good]);phase=np.array([r["phase"] for r in good])
    np.savez_compressed(out/f"{name}_candidate_series.npz",coordinates=np.array(coordinates),
                        t=t,values=values,patches=patches)
    result=[]
    fig,axes=plt.subplots(5,2,figsize=(13,13),sharex=True,constrained_layout=True)
    # First column increases; second column decreases.
    order=[v for pair in zip(range(5),range(5,10)) for v in pair]
    for ax,j in zip(axes.flat,order):
        y,x=coordinates[j]
        ax.plot(t,values[:,j],"o-",ms=3,lw=.6,color="#b45309" if j<5 else "#2563eb")
        ax.axvspan(-1,.5,color="gray",alpha=.2);ax.axvline(0,color="k",ls=":")
        ax.set_title(f"Row {y}, col {x} · fitted step {effect[y,x]:+.0f} DN",fontsize=10)
        ax.set_ylabel("DN");ax.grid(alpha=.2)
        result.append(dict(row=y,column=x,step_dn=float(effect[y,x]),se_dn=float(se[y,x]),
            first_half_open_change_dn=float(values[(t>1)&(t<20),j].mean()-values[phase=="closed",j].mean()),
            late_open_change_dn=float(values[t>40,j].mean()-values[phase=="closed",j].mean())))
    for ax in axes[-1]:ax.set_xlabel("Seconds from EPS deploy marker")
    fig.suptitle(f"TVAC2 {name}: strongest pixel increases and decreases",fontsize=15)
    fig.savefig(out/f"{name}_candidate_pixels.png",dpi=150);plt.close(fig)
    # Timing sensitivity includes the unknown subsecond EPS sample phase.
    all_t=np.array([r["dt_mid"] for r in rows]);starts=np.array([r["dt"] for r in rows]);ends=np.array([r["dt_end"] for r in rows])
    all_y=np.array([r["mean"] for r in rows]);timing={}
    for shift in [-2,-1,0,1,2]:
        b=(all_t>shift-40)&(ends<=shift-1)
        a=(all_t<shift+40)&(starts>=shift+.5)
        timing[str(shift)]=estimate_step(all_t-shift,all_y,b,a)
    # Synthetic signal recovery exercises the estimator with the actual temporal
    # noise and unchanged slopes. This is performed on analysis arrays only.
    before=(phase=="closed")&(t>-40);after=(phase=="open")&(t<40)
    # Use an ordinary 8x8 cell away from the candidate selection for this check.
    cube=arrays["cube8"];all_phase=np.array([r["phase"] for r in rows])
    b=(all_phase=="closed")&(all_t>-40);a=(all_phase=="open")&(all_t<40)
    steps,se_cell,_,_=field_step(all_t,cube[:,125:126,94:95],b,a)
    injected=cube[:,125:126,94:95].astype(float).copy()
    injected[all_phase=="open"]+=20
    recovered,_,_,_=field_step(all_t,injected,b,a)
    recovery=float((recovered-steps).item())
    if not np.isclose(recovery,20,atol=1e-6):
        raise AssertionError(f"Injected signal was not preserved: {recovery}")
    return dict(candidate_pixels=result,timing_sensitivity=timing,
                injected_20dn_cell_recovered_dn=recovery)


def write_report(summaries, tests, out):
    """A prominent verdict with all positive findings and caveats visible."""
    import base64
    import html
    # These are assessments of this dataset, not a claim of absolute zero flux.
    for name,d in summaries.items():
        flagged=(d["native_pixels"]["positive_pixels_global_1pct"]>0 or
                 any(v.get("positive_cells_at_global_1pct",v.get("positive_at_global_1pct",0))>0
                     for v in d["spatial"].values()) or d["steps"]["40"]["p_positive"]<.001)
        d["assessment"]="UNRESOLVED: changes need attribution" if flagged else "NO light leak detected"
        d["model_conditional_upper95_envelope_dn_per_pixel_s"] = max(s["upper95_dn"] for s in d["steps"].values())
        d["incomplete_excluded"] = sum(not r["complete"] for r in tests[name]["records"])
        d["light_source_timing_assumption"] = "The described source illuminated the telescope while the door was open; exact on/off and aiming history was not supplied."
    # Compact visual suitable for sharing without opening the technical report.
    fig=plt.figure(figsize=(13,8.5),layout="constrained")
    gs=fig.add_gridspec(3,2,height_ratios=[.5,1.4,.4])
    for col,name in enumerate(["cold","hot"]):
        d=summaries[name]
        ax=fig.add_subplot(gs[0,col]);ax.axis("off")
        color="#166534" if name=="cold" else "#92400e"
        title="NO leak detected" if name=="cold" else "UNRESOLVED"
        ax.text(.02,.75,f"{name.upper()} · {TESTS[name][0]}",fontsize=13,color="#475569",transform=ax.transAxes)
        ax.text(.02,.25,title,fontsize=30,fontweight="bold",color=color,transform=ax.transAxes)
        rows=list(csv.DictReader((out/f"{name}_frames.csv").open()))
        t=np.array([float(r["dt_mid"]) for r in rows]);y=np.array([float(r["mean"]) for r in rows]);phase=np.array([r["phase"] for r in rows])
        before=phase=="closed";zero=y[before].mean();ax=fig.add_subplot(gs[1,col])
        for p,c in [("closed","#2563eb"),("transition","#a21caf"),("open","#ea580c")]:
            mask=phase==p;ax.scatter(t[mask],y[mask]-zero,s=18,color=c,label=p)
        ax.axvspan(-1,.5,color="#a21caf",alpha=.12);ax.axvline(0,color="black",ls=":",lw=1)
        lf=linear_fit(t[before],y[before]);xx=np.linspace(t[before].min(),min(t.max(),85),100)
        ax.plot(xx,lf["beta"][0]+lf["beta"][1]*xx-zero,"--",color="#2563eb",lw=1,label="Closed-door trend")
        ax.set_xlabel("Seconds from EPS deployment marker")
        ax.set_ylabel("Mean change from closed baseline (DN / pixel)")
        ax.grid(alpha=.2);ax.legend(fontsize=9)
        ax=fig.add_subplot(gs[2,col]);ax.axis("off")
        text=("No positive pixel, tile, row or column detection.\nNo image-wide flash during motion." if name=="cold" else
              "3 pixels brighten persistently; other pixels dim.\nNo broad leak pattern; cause of small changes unresolved.")
        ax.text(.02,.9,text,va="top",fontsize=11,linespacing=1.6,transform=ax.transAxes)
    fig.suptitle("TVAC2 light-leak check · cold: no detection; hot: unresolved",fontsize=19,fontweight="bold")
    fig.savefig(out/"verdict.png",dpi=160);plt.close(fig)
    def embedded(filename):
        payload=base64.b64encode((out/filename).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{payload}" alt="{html.escape(filename)}">'
    c,h=summaries["cold"],summaries["hot"]
    table=[]
    for name,d in summaries.items():
        s=d["steps"]["40"];n=d["native_pixels"]
        table.append(f'<tr><th>{name.title()}</th><td>{d["n_complete"]}</td><td>{d["incomplete_excluded"]}</td>'
                     f'<td>{d["counts"]["closed"]} / {d["counts"]["transition"]} / {d["counts"]["open"]}</td>'
                     f'<td>{s["step_dn"]:+.3f} ± {s["se_dn"]:.3f}</td>'
                     f'<td>{n["positive_pixels_global_1pct"]} / {n["negative_pixels_global_1pct"]}</td></tr>')
    recovery=[]
    for size in [8,32,128]:
        recovery.append(f'<tr><th>{size} × {size} pixels</th>' + ''.join(
            f'<td>{summaries[name]["approximate_recovery"][str(size)]["approx_median_95pct_recovery_dn_per_pixel"]:.1f}</td>' for name in ["cold","hot"]) + '</tr>')
    figures=[]
    for name in ["cold","hot"]:
        figures.append(f'<h2>{name.title()} test: supporting plots</h2>')
        for suffix in ["timeline","differences","spatial_checks","candidate_pixels"]:
            figures.append(embedded(f"{name}_{suffix}.png"))
    report=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TVAC2 light-leak analysis · 2026-10-15</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#172033;background:#f3f5f7;margin:0}}main{{max-width:1120px;margin:36px auto;padding:36px;background:white;border-radius:16px}}h1{{font-size:38px;line-height:1.15}}h2{{margin-top:38px}}.cards{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.card{{padding:22px;border-radius:12px;background:#ecfdf3;border-left:6px solid #15803d}}.amber{{background:#fffbeb;border-color:#b45309}}.verdict{{font-size:32px;font-weight:800;line-height:1.25}}.meta{{color:#596779}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #ddd}}th{{background:#f5f7fa}}img{{width:100%;height:auto;margin:18px 0}}a{{color:#1d4ed8}}.note{{padding:18px;background:#f0f4f8;border-radius:10px}}code{{font-size:13px}}@media(max-width:700px){{main{{margin:0;padding:18px}}.cards{{grid-template-columns:1fr}}table{{font-size:12px}}th,td{{padding:7px}}}}</style>
<main><p class="meta">SunCET · TVAC 2 · Analysis dated October 15, 2026 · Raw data preserved</p>
<h1>Light-leak verdict</h1><div class="cards"><div class="card"><p>Cold · Sun cannon · approximately −16.6 °C</p><div class="verdict">NO light leak detected</div><p>No detected positive deployment step in native pixels or tested larger regions. No image-wide flash during door motion.</p></div>
<div class="card amber"><p>Hot · Phone flashlight · approximately +42.5 °C</p><div class="verdict">UNRESOLVED</div><p>No confirmed optical leak. Three native pixels brighten significantly near deployment; other pixels dim. A clean no-leak pass is not established.</p></div></div>
<p><strong>The data do not justify saying “yes, there is an optical leak,” or an unconditional “no” for the hot test.</strong> Real localized changes are present and remain in the results. Their cause needs to be established.</p>
{embedded('verdict.png')}
<h2>What changed</h2><table><thead><tr><th>Test</th><th>Complete images</th><th>Incomplete excluded</th><th>Nearby closed / motion / open</th><th>Image-wide step ± uncertainty, DN/pixel</th><th>Flagged native pixels, brighter / dimmer</th></tr></thead><tbody>{''.join(table)}</tbody></table>
<p>The cold dataset also has {c['counts']['earlier_closed']} earlier closed-door frames, separated from the deployment sequence by a nine-minute gap. They are controls rather than the primary baseline. Every exposure is one second, unbinned, internal stim lamp off, with no right shift. The displayed ± value is one estimated standard error, not the full detection limit.</p>
<p>The hot open-minus-closed image mean is {h['raw_difference_dn']:.2f} DN/pixel, but most of that is a gradual rise already present before deployment. The local deployment step is approximately +0.27 DN/pixel; it ranges from +0.27 to +0.54 DN/pixel when the fitting window is changed, with the 30-second fit compatible with zero. Similar apparent steps occur at invented event times inside a fixed door state, so a single straight-line model understates drift uncertainty.</p>
<p>The hot native-pixel screen flags 3 brightening pixels at native (row, column) coordinates (737, 661), (623, 771), and (623, 772), using zero-based coordinates in the stored array. Fitted changes are about +376, +350, and +314 DN. The adjacent pair is one localized site. These changes persist through the remainder of the sequence. The screen also flags {h['native_pixels']['negative_pixels_global_1pct']} dimming pixels. Among unmasked pixels, the 5-standard-error tails contain 15 increases and 14 decreases. This symmetry and the observed level switching support a detector-effect hypothesis, but do not prove it or rule out an optical contribution.</p>
<h2>What the sensitivity numbers mean</h2>
<p>A detection limit answers: <em>How much added light could remain hidden in the fluctuations?</em> Averaging many pixels makes the image-wide test sensitive to a small uniform glow. A tiny spot has fewer pixels to average and requires a larger local signal.</p>
<p>Under the fitted local linear-drift model, the conservative one-sided 95% upper bounds on a uniform sudden brightening span up to <strong>{c['model_conditional_upper95_envelope_dn_per_pixel_s']:.2f} DN/pixel/s for cold</strong> and <strong>{h['model_conditional_upper95_envelope_dn_per_pixel_s']:.2f} DN/pixel/s for hot</strong> across the tested time windows. These are conditional statistical bounds, not drift-independent guarantees. They do not describe the anomalous individual hot pixels.</p>
<table><thead><tr><th>Region receiving a constant added signal</th><th>Cold: approximate 95% recovery signal</th><th>Hot: approximate 95% recovery signal</th></tr></thead><tbody>{''.join(recovery)}</tbody></table>
<p class="meta">Values are DN/pixel/s at a typical tested cell, after allowing for the number of searched locations. They are approximate sensitivity calculations using temporal noise and nominal familywise thresholds. A 20-DN injected step was recovered as 20 DN in both tests; this checks that the estimator retains a real signal. It does not independently validate the noise model or establish 95% recovery at every location.</p>
<h2>Door motion and timing</h2>
<p>EPS first shows an active DOOR deploy timer at <strong>{html.escape(c['event_utc'])}</strong> and <strong>{html.escape(h['event_utc'])}</strong>, in the export's UTC convention. The EPS sampling interval is about one second; the reported physical actuation lasts about half a second. Capture-source META packets, rather than process-source timestamps or downlink times, define the exposure intervals.</p>
<p>Only two complete frames in each test are set aside from the averaged closed/open comparison because they intersect the timing bracket [one second before the EPS sample, half a second after]. All four are retained in the transient checks. The detector should indeed be insensitive to door position if filtering is complete. Keeping these frames in the flash check tests exactly that expectation. Shifting the deployment reference by ±2 seconds leaves the cold image-wide result negative; the hot fitted step varies from +0.13 to +0.29 DN/pixel.</p>
<h2>What would resolve the hot result</h2>
<p>First establish the phone flashlight's on/off and aiming history, and the relative timing of the CSIE clock, EPS clock, and video. If an illuminated/unilluminated interval already exists after the door is open, compare the flagged pixels across it. Otherwise a short source-off / source-on / source-off sequence at stable detector temperature with the door open would distinguish illumination that follows the light from detector state changes. Without that discrimination, the hot sequence remains unresolved.</p>
<h2>Method and checks</h2>
<ul><li>Only fully reconstructed images with all selected rows checksum-valid and no filled or repaired rows enter the analysis. Original FITS, metadata, and telemetry were read only.</li>
<li>Raw means, medians, upper percentiles, row/column profiles, spatial variance, temperatures, and storage-rail counts are retained. No frame is normalized to its own median or mean.</li>
<li>Raw average differences, differences with the closed-only uniform trend prediction, and local regression step maps are all saved. The long closed-only extrapolation is a diagnostic, not the basis for the verdict.</li>
<li>Local fits estimate the closed and open intercepts at deployment with separate slopes. Their difference retains any uniform step. Primary spatial screening uses a 40-second window, with sensitivity checks at 15, 20, 30, 60, 90, and 400 seconds.</li>
<li>Temporal residuals determine uncertainty, with a three-lag Newey–West estimate and an ordinary temporal-variance floor. Pixels are not treated as independent repeated exposures. Native-pixel, 8×8, 32×32, 128×128, row, and column screens use conservative nominal multiple-comparison thresholds.</li>
<li>Bad-pixel masks are defined only from closed-door averages/variances. All-pixel results remain alongside masked results; the three hot positive candidates are not removed by this mask.</li>
<li>Flash checks use second differences between adjacent complete frames, canceling locally linear drift. Transition-frame image-wide excursions are smaller than the normal closed-door fluctuation scale. Bright isolated cells are not automatically attributed to light.</li>
<li>Fictitious events within one door state are descriptive drift controls. Their overlapping windows are not independent trials; open-door controls cannot exclude source variability.</li>
<li>Synthetic checks verify a known step in a drifting series, a 20-DN step in a real cell series, and partial-edge spatial binning.</li></ul>
<p>Level-switching behavior is known in CMOS pixels; see <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC6960882/">the primary study of random telegraph noise sources</a>. That literature motivates a possible explanation, not a diagnosis of these specific pixels. Confidence-limit interpretation follows <a href="https://www.itl.nist.gov/div898/handbook/eda/section3/eda3521.htm">NIST guidance</a>.</p>
<p class="note">This assesses the combined entrance and focal-plane filter system under the described test sources. Individual filter integrity and optical attenuation are not independently measured. No source irradiance or applicable gain calibration was supplied, so no transmission ratio or electron conversion is inferred.</p>
{''.join(figures)}
<h2>Reproduce and inspect</h2><p>Run <code>python3 tvac2_light_leak_analysis.py</code> in the analysis repository. Use <code>--data</code>, <code>--out</code>, or <code>--refresh</code> as needed. <code>selection.json</code> records every selected or rejected image; <code>cold_frames.csv</code> and <code>hot_frames.csv</code> retain frame metrics; <code>summary.json</code> retains statistics. NPZ files preserve full-resolution averages, variances, differences, masks, step maps, and candidate time series.</p></main></html>'''
    (out/"report.html").write_text(report)
    with (out/"results.csv").open("w") as f:
        writer=csv.writer(f);writer.writerow(["test","assessment","complete_images","incomplete_excluded","step_40s_dn_per_pixel","se_dn_per_pixel","conditional_upper95_envelope_dn_per_pixel_s","flagged_brighter_native_pixels","flagged_dimmer_native_pixels"])
        for name,d in summaries.items():
            writer.writerow([name,d["assessment"],d["n_complete"],d["incomplete_excluded"],d["steps"]["40"]["step_dn"],d["steps"]["40"]["se_dn"],d["model_conditional_upper95_envelope_dn_per_pixel_s"],d["native_pixels"]["positive_pixels_global_1pct"],d["native_pixels"]["negative_pixels_global_1pct"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tests = load_records(args.data)
    serial = {name: {"event_utc": test["event_utc"], "records": test["records"]} for name, test in tests.items()}
    (args.out / "selection.json").write_text(json.dumps(serial, indent=2))
    summaries = {}
    for name, test in tests.items():
        arrays, rows = first_pass(name, test, args.out, args.refresh)
        summaries[name] = inspect_statistics(name, arrays, rows, args.out)
        summaries[name].update(multiscale_checks(name, arrays, rows, args.out))
        summaries[name]["native_pixels"] = native_pixel_checks(name, test, arrays, args.out)
        summaries[name].update(candidate_checks(name,test,arrays,rows,args.out))
        summaries[name]["event_utc"] = test["event_utc"]
        print(name, "steps",json.dumps(summaries[name]["steps"]), "native pixels",json.dumps(summaries[name]["native_pixels"]),flush=True)
    write_report(summaries,tests,args.out)
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2))
    # The physical thermal check uses independent housekeeping and earlier dark
    # exposure pairs; it does not fit a correction to the open-door brightness.
    from tvac2_thermal_followup import main as thermal_followup
    thermal_followup(args.data,args.out)


if __name__ == "__main__":
    main()
