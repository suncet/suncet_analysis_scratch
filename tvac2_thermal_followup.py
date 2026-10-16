#!/usr/bin/env python3
"""Independent thermal checks for the TVAC2 hot light-leak sequence.

No image signal after deployment is used to fit the temperature history or the
dark-current temperature law. The earlier nominal 0.5/1-second exposure pairs
are a consistency check, with extrapolation limits explicitly reported.
"""
from __future__ import annotations

import base64
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
import matplotlib.dates as mdates
import numpy as np
from astropy.io import fits
from scipy import stats

from tvac2_light_leak_analysis import DEFAULT_DATA, DEFAULT_OUT, EPOCH, utc

COORDS = [(737,661),(623,771),(623,772),(1823,539),(1704,908)]
DOUBLE_C = 5.5
DATASHEET = "https://www.teledyneimaging.com/media/1359/cis115.pdf"


def raw_seconds(t):
    return (t-EPOCH).total_seconds()-5


def housekeeping(data, out):
    start=datetime(2026,10,15,16,20,tzinfo=timezone.utc)
    stop=datetime(2026,10,15,17,25,tzinfo=timezone.utc)
    lo,hi=raw_seconds(start),raw_seconds(stop)
    rows=[]
    f=next((data/"public_dashboard").glob("*apid-0537-*.json.gz"))
    with gzip.open(f,"rt")as stream:d=json.load(stream)
    ix={k:i for i,k in enumerate(d["columns"])}
    fields=["csie_detBackside_therm","csie_detBoard_therm","csie_adc_thermistor_temp","csie_fpga_therm"]
    seen=set()
    for r in d["rows"]:
        if not lo<=r[ix["packet_time_sec"]]<=hi:continue
        key=(r[ix["packet_time_sec"]],r[ix["packet_time_sub"]])
        if key in seen:continue
        seen.add(key)
        tm=utc(*key)
        rows.append(dict(utc=tm.isoformat(),**{k:float(r[ix[k]])for k in fields}))
    rows.sort(key=lambda r:r["utc"])
    with (out/"hot_thermal_housekeeping.csv").open("w")as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    f=next((data/"public_dashboard").glob("*apid-0030-*.json.gz"))
    with gzip.open(f,"rt")as stream:d=json.load(stream)
    ix={k:i for i,k in enumerate(d["columns"])}
    events=[];last_power=None
    for r in d["rows"]:
        if not lo<=r[ix["packet_time_sec"]]<=hi:continue
        tm=utc(r[ix["packet_time_sec"]],r[ix["packet_time_sub"]])
        power=r[ix["eps_pwr_state_csie"]]
        if power!=last_power:
            events.append(dict(utc=tm.isoformat(),event=f"CSIE power {power}"));last_power=power
        if r[ix["eps_deploy_time"]]>0 and r[ix["eps_deploy_select"]] in ["SA_MINUS_Y","SA_PLUS_Y","DOOR"]:
            events.append(dict(utc=tm.isoformat(),event=r[ix["eps_deploy_select"]]))
    return rows,events


def earlier_dark_pairs(data,out):
    path=out/"earlier_dark_pair_metrics.csv"
    if path.exists():
        records=list(csv.DictReader(path.open()))
        source_paths=[Path(r[k])for r in records for k in ["path_short","path_long"]]
        valid=bool(records) and all(p.is_relative_to(data) and p.exists() and p.stat().st_mtime_ns<=path.stat().st_mtime_ns for p in source_paths)
        if valid:
            for r in records:
                for k in r:
                    if k not in ["time","path_short","path_long"]:r[k]=float(r[k])
            return records
    inventory=list(csv.DictReader((data/"csie_images/csie_image_inventory.csv").open()))
    candidates=[]
    for r in inventory:
        if not r["metadata_time_iso8601"].startswith("2026-10-11") or r["complete_uncompressed_image"]!="True":continue
        if (int(r["selected_checksum_valid_rows"])!=int(r["expected_rows"])
            or any(int(r[k]) for k in ["selected_checksum_failed_rows","selected_checksum_missing_rows","zero_filled_total_rows","boundary_fill_repaired_rows"])):continue
        m=json.loads(Path(r["meta_json_path"]).read_text())
        if m.get("detector_temp",-100)<30 or m.get("csie_meta_intg_ms")not in [500,1000]:continue
        if any(m.get(k) for k in ["csie_meta_stim_lamp_stat","csie_meta_test_mode","fpm_proc_cfg_right_shift_meta","fpm_proc_cfg_row_bin_meta","fpm_proc_cfg_col_bin_meta"]):continue
        if m.get("fpm_proc_cfg_dark_negate_meta")!="DIS" or m.get("fpm_proc_cfg_pix_clean_meta")!="DIS":continue
        candidates.append(dict(time=datetime.fromisoformat(r["metadata_time_iso8601"].replace("Z","+00:00")),
            path=r["fits_path"],image_id=int(r["image_id"]),meta=m))
    candidates.sort(key=lambda r:r["time"])
    records=[]
    for i,long in enumerate(candidates):
        if long["meta"]["csie_meta_intg_ms"]!=1000:continue
        previous=[r for r in candidates[:i] if r["meta"]["csie_meta_intg_ms"]==500 and 0<(long["time"]-r["time"]).total_seconds()<2]
        if not previous:continue
        short=previous[-1]
        if abs(short["meta"]["detector_temp"]-long["meta"]["detector_temp"])>.1:continue
        a=fits.getdata(short["path"],memmap=False).astype(float)
        b=fits.getdata(long["path"],memmap=False).astype(float)
        if a.shape!=(2000,1504) or b.shape!=a.shape:continue
        rate=(b-a)/.5
        record=dict(time=long["time"].isoformat(),image_short=short["image_id"],image_long=long["image_id"],
            path_short=short["path"],path_long=long["path"],temp=(short["meta"]["detector_temp"]+long["meta"]["detector_temp"])/2,
            adc_temp=(short["meta"]["csie_meta_adc_thermistor_temp"]+long["meta"]["csie_meta_adc_thermistor_temp"])/2,
            mean_short=float(a.mean()),mean_long=float(b.mean()),dark_rate_dn_s=float(rate.mean()),
            extrapolated_bias_dn=float((2*a-b).mean()))
        for y,x in COORDS:
            record[f"dark_rate_{y}_{x}"]=float(rate[y,x])
            record[f"signal_1s_{y}_{x}"]=float(b[y,x])
        records.append(record)
    with path.open("w")as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    return records


def main(data=DEFAULT_DATA,out=DEFAULT_OUT):
    hk,events=housekeeping(data,out)
    pairs=earlier_dark_pairs(data,out)
    frames=list(csv.DictReader((out/"hot_frames.csv").open()))
    summary=json.loads((out/"summary.json").read_text())
    door=datetime.fromisoformat(summary["hot"]["event_utc"])
    ht=np.array([(datetime.fromisoformat(r["utc"])-door).total_seconds()for r in hk])
    td=np.array([r["csie_detBackside_therm"]for r in hk])
    ta=np.array([r["csie_adc_thermistor_temp"]for r in hk])
    ft=np.array([float(r["dt_mid"])for r in frames]);signal=np.array([float(r["mean"])for r in frames])
    det=np.array([float(r["temp"])for r in frames]);adc=np.array([float(r["adc_temp"])for r in frames])
    # Smooth quantization using only HK data in a preselected 10-minute window.
    mask=(ht>-330)&(ht<270)
    thermal_slope,thermal_intercept=np.polyfit(ht[mask],td[mask],1)
    smoothed=thermal_intercept+thermal_slope*ft
    alpha=np.log(2)/DOUBLE_C
    dt_raw=float(det[-1]-det[0]);dt_smooth=float(smoothed[-1]-smoothed[0])
    measured=float(signal[-1]-signal[0])
    # Explicitly approximate bias: cold-frame average is only a reference, not a
    # calibrated hot-electronics bias frame. Keep this check separate from fits.
    bias_reference=summary["cold"]["mean_closed_dn"]
    apparent_dark=float(signal[0]-bias_reference)
    rough_prediction=apparent_dark*np.expm1(alpha*dt_raw)
    rough_smooth=apparent_dark*np.expm1(alpha*dt_smooth)
    temp=np.array([r["temp"]for r in pairs]);rates=np.array([r["dark_rate_dn_s"]for r in pairs])
    empirical_slope,empirical_intercept=np.polyfit(temp,np.log(rates),1)
    empirical_dark=np.exp(empirical_intercept+empirical_slope*smoothed)
    empirical_change=float(empirical_dark[-1]-empirical_dark[0])
    nominal_ref=np.median(rates*np.exp(-alpha*(temp-np.median(temp))))
    nominal_dark=nominal_ref*np.exp(alpha*(smoothed-np.median(temp)))
    nominal_change=float(nominal_dark[-1]-nominal_dark[0])
    # Per-pixel ordinary dark-current response: multiply measured mean rate at
    # 30–32 C by the same temperature scaling, without fitting hot pixel signals.
    candidate=[]
    for y,x in COORDS:
        normalizer=float(np.median(np.array([r[f"dark_rate_{y}_{x}"]for r in pairs])*np.exp(-alpha*(temp-np.median(temp)))))
        predicted=float(normalizer*np.exp(alpha*(smoothed[0]-np.median(temp)))*np.expm1(alpha*dt_smooth))
        c=next((q for q in summary["hot"]["candidate_pixels"]if q["row"]==y and q["column"]==x),None)
        candidate.append(dict(row=y,column=x,ordinary_dark_rise_dn=predicted,
            fitted_deployment_step_dn=c["step_dn"]if c else None,
            note="This smooth-temperature calculation does not model random telegraph switching, thermal stress, or electronics offsets."))
    actual_firstlast=float(np.mean(signal[-10:])-np.mean(signal[:10]))
    result=dict(events=events,detector_start_C=float(det[0]),detector_end_C=float(det[-1]),
        measured_detector_change_C=dt_raw,adc_change_C=float(adc[-1]-adc[0]),
        image_first_last_change_dn=measured,image_first10_last10_change_dn=actual_firstlast,
        hk_fit_window_relative_door_s=[-330,270],hk_detector_rate_C_per_min=float(thermal_slope*60),
        hk_smoothed_image_interval_delta_C=dt_smooth,
        datasheet_dark_doubling_C=DOUBLE_C,datasheet_fractional_dark_increase_raw=float(np.expm1(alpha*dt_raw)),
        cold_reference_bias_dn=bias_reference,approximate_hot_dark_dn=apparent_dark,
        rough_cold_bias_prediction_raw_temperature_dn=float(rough_prediction),
        rough_cold_bias_prediction_smoothed_temperature_dn=float(rough_smooth),
        independent_exposure_pairs=len(pairs),calibration_temperature_range_C=[float(temp.min()),float(temp.max())],
        independent_pair_rate_range_dn_s=[float(rates.min()),float(rates.max())],
        empirical_pair_doubling_C=float(np.log(2)/empirical_slope),
        independent_pair_prediction_nominal_5p5C_dn=nominal_change,
        independent_pair_prediction_fitted_exponential_dn=empirical_change,
        candidate_smooth_dark_predictions=candidate,
        interpretation="Warming is quantitatively consistent with most or all of the gradual image-wide rise. The sparse persistent positive/negative pixel steps remain unattributed; small smooth warming alone does not reproduce their amplitude.",
        limitations=["Detector thermistor quantization is about 0.0275 C per code here; smoothing is inferred from independent housekeeping, not image brightness.",
            "The 30–32 C exposure-pair calibration is extrapolated about 10 C and assumes nominal process-source exposure settings; it is a plausibility check, not a precision 42 C dark calibration.",
            "A cold detector image is not a measured hot-electronics bias calibration.",
            "The first relevant CSIE temperature telemetry is after -Y deployment; radiator exposure and instrument self-heating cannot be separated from these records alone."],
        datasheet_url=DATASHEET)
    (out/"hot_thermal_summary.json").write_text(json.dumps(result,indent=2))
    summary["hot"]["thermal_followup"]=result
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    # The overview makes timing and the independent prediction easy to compare.
    fig,ax=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    dates=[datetime.fromisoformat(r["utc"])for r in hk]
    ax[0,0].plot(dates,td,color="#2563eb",lw=1,label="Detector backside")
    ax[0,0].plot(dates,[r["csie_detBoard_therm"]for r in hk],color="#b45309",lw=1,label="Detector board")
    ax[0,0].set_ylabel("Temperature (°C)");ax[0,0].set_title("Detector was already warming before door deployment")
    ax[0,0].legend()
    ax[0,1].plot(dates,ta,color="#0f766e",label="ADC")
    ax[0,1].plot(dates,[r["csie_fpga_therm"]for r in hk],color="#7e22ce",label="FPGA")
    ax[0,1].set_ylabel("Temperature (°C)");ax[0,1].set_title("Electronics were also warming");ax[0,1].legend()
    for a in ax[0]:
        for e in events:
            if e["event"] in ["SA_MINUS_Y","CSIE power ON"] or e["utc"]==door.isoformat():
                date=datetime.fromisoformat(e["utc"])
                a.axvline(date,color="#64748b",ls=":",lw=1)
        a.set_xlim(datetime(2026,10,15,16,28,tzinfo=timezone.utc),datetime(2026,10,15,17,24,tzinfo=timezone.utc))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M",tz=timezone.utc));a.set_xlabel("October 15, UTC")
    ax[0,0].text(.02,.04,"−Y: 16:31:35   ·   CSIE on: 16:44:03   ·   Door: 17:10:32",transform=ax[0,0].transAxes,fontsize=8)
    ax[1,0].scatter(ft,det,s=13,label="Recorded detector temperature",color="#2563eb")
    ax[1,0].plot(ft,smoothed,label="Independent HK trend",color="#2563eb",ls="--")
    ax[1,0].set_xlabel("Seconds from door deployment marker");ax[1,0].set_ylabel("Detector temperature (°C)");ax[1,0].legend(fontsize=8)
    ax[1,0].set_title("Thermistor steps are one sensor-code wide")
    ax[1,1].scatter(ft,signal-signal[0],s=14,color="#ea580c",label="Observed image mean")
    ax[1,1].plot(ft,apparent_dark*np.expm1(alpha*(smoothed-smoothed[0])),label="5.5 °C law + approximate bias",color="#2563eb")
    ax[1,1].plot(ft,nominal_dark-nominal_dark[0],label="Earlier exposure pairs + 5.5 °C law",color="#0f766e")
    ax[1,1].plot(ft,empirical_dark-empirical_dark[0],label="Earlier exposure pairs + fitted law",color="#7e22ce",ls="--")
    ax[1,1].set_xlabel("Seconds from door deployment marker");ax[1,1].set_ylabel("Change in image mean (DN/pixel)")
    ax[1,1].set_title("Heating predicts a rise of the observed order");ax[1,1].legend(fontsize=8)
    for a in ax[1]:a.axvline(0,color="k",ls=":",lw=1)
    for a in ax.flat:a.grid(alpha=.2)
    fig.suptitle("TVAC2 hot test: thermal follow-up",fontsize=18,fontweight="bold")
    fig.savefig(out/"hot_thermal_followup.png",dpi=160);plt.close(fig)
    payload=base64.b64encode((out/"hot_thermal_followup.png").read_bytes()).decode()
    section=f'''<section id="thermal-followup"><h2>Thermal follow-up: warming explains the broad rise</h2>
<p>The −Y deploy timer was active at 16:31:35.549 UTC, and CSIE power changed to ON at 16:44:03.549 UTC. CSIE housekeeping then records a sustained detector and electronics warm-up before the door deployment at 17:10:32.550 UTC. This is consistent with the reported exposure of the radiator to the hot shroud; instrument self-heating is also present and cannot be separated here.</p>
<p>Across the images, the detector thermistor changes from {det[0]:.5f} to {det[-1]:.5f} °C (+{dt_raw:.5f} °C), while ADC temperature rises {adc[-1]-adc[0]:.3f} °C. The detector change is about one thermistor code. An independent 10-minute housekeeping fit gives a smoother rise of {dt_smooth:.5f} °C during the image sequence ({thermal_slope*60:.4f} °C/min).</p>
<p>The <a href="{DATASHEET}">CIS115 manufacturer datasheet</a> gives a typical dark-current doubling interval of 5.5 °C. The measured one-code rise corresponds to a {100*np.expm1(alpha*dt_raw):.3f}% dark-current increase. For roughly {apparent_dark:.0f} DN of dark signal above an approximate cold-image bias reference, that predicts {rough_prediction:.1f} DN of brightening, or {rough_smooth:.1f} DN using the smoothed temperature change. The observed first-to-last rise is {measured:.1f} DN ({actual_firstlast:.1f} DN comparing the first and last ten frames).</p>
<p>As an independent check, {len(pairs)} earlier nominal 0.5/1-second image pairs at {temp.min():.2f}–{temp.max():.2f} °C predict a rise of {nominal_change:.1f} DN using the manufacturer's 5.5 °C scaling, or {empirical_change:.1f} DN using their fitted exponential temperature law. These involve extrapolation to approximately 42.5 °C; they establish plausibility rather than a precise correction.</p>
<p><strong>The broad hot-sequence brightening is consistent with warming and should not be called evidence of an optical leak.</strong> The earlier analysis already allowed time drift, but had not quantified this physical thermal explanation. Ordinary smooth heating predicts changes of only a few to tens of DN at the flagged pixels, not their approximately 300–400 DN deployment steps. Those sparse positive and negative changes remain unexplained; thermal detector/electronics state switching is possible but is not established by this calculation. The hot result therefore remains unresolved at the individual-pixel level.</p>
    <img src="data:image/png;base64,{payload}" alt="Hot thermal follow-up"><p class="meta">Thermal checks are reproduced by <code>tvac2_thermal_followup.py</code>, also called by the main analysis script. <code>hot_thermal_summary.json</code>, <code>hot_thermal_housekeeping.csv</code>, and <code>earlier_dark_pair_metrics.csv</code> retain the numerical evidence.</p></section>'''
    path=out/"report.html";report=path.read_text()
    marker='<section id="thermal-followup">'
    if marker in report:
        before,rest=report.split(marker,1);_,after=rest.split('</section>',1);report=before+after
    report=report.replace('<h2>What would resolve the hot result</h2>',section+'<h2>What would resolve the hot result</h2>')
    path.write_text(report)
    print(json.dumps(result,indent=2))
    return result


if __name__=="__main__":main()
