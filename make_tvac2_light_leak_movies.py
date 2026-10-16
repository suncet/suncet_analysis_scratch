#!/usr/bin/env python3
"""Golden Hour movies with fixed per-test stretches and deployment alignment.

All inputs are read only. Complete exposures spanning door motion are included.
Images are held at their measured times; no temporal interpolation is applied.
The comparison uses the common recent cold/hot acquisition window. Earlier cold
images across the nine-minute acquisition gap are not included in these movies.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from astropy.io import fits
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
DEFAULT_ANALYSIS = ROOT / "tvac2_light_leak_output"
DEFAULT_LUT = ROOT.parent / "suncet_instrument_simulator/docs/colortables/luts/13-sunset-gold.csv"
WIDTH, HEIGHT = 1920, 1080
IMAGE_W, IMAGE_H, IMAGE_Y = 800, 602, 195
PANEL_X = {"cold": 32, "hot": 976}
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
BG, FG, MUTED = "#0c1018", "#f1f5f9", "#a8b5c7"
BLUE, ORANGE = "#67a9ff", "#ffb46c"


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ("Arial Bold.ttf" if bold else "Arial.ttf")), size)


FONTS = {size: font(size) for size in [14, 16, 17, 18, 19, 20, 26, 28, 34]}
BOLD = {size: font(size, True) for size in [18, 20, 26, 28, 34]}


def spatial_mean(a, size):
    y, x = a.shape
    if y % size or x % size:
        raise ValueError("Expected binning to divide both detector dimensions")
    return a.reshape(y//size, size, x//size, size).mean(axis=(1,3))


def fixed_limits(reference, step=25):
    lo, hi = np.percentile(reference, [.5, 99.5])
    return float(np.floor(lo/step)*step), float(np.ceil(hi/step)*step)


class MovieData:
    def __init__(self, name, selection, analysis, lut, args):
        self.name, self.event, self.lut = name, selection["event_utc"], lut
        self.records = [r for r in selection["records"] if r["complete"] and r["dt_mid"] > -120]
        self.times = np.array([r["dt_mid"] for r in self.records])
        metrics = {int(float(r["image_id"])): r for r in csv.DictReader((analysis/f"{name}_frames.csv").open())}
        self.means = np.array([float(metrics[r["image_id"]]["mean"]) for r in self.records])
        self.temps = np.array([float(metrics[r["image_id"]]["temp"]) for r in self.records])
        self.ids = np.array([r["image_id"] for r in self.records])
        z = np.load(analysis/f"{name}_cache.npz")
        self.reference = z["closed_mean"].astype(np.float32)
        self.reference8 = spatial_mean(self.reference,8)
        self.closed_mean = float(self.means[[r["phase"]=="closed" for r in self.records]].mean())
        limits = fixed_limits(self.reference)
        self.raw_limits = (getattr(args,f"{name}_vmin") if getattr(args,f"{name}_vmin") is not None else limits[0],
                           getattr(args,f"{name}_vmax") if getattr(args,f"{name}_vmax") is not None else limits[1])
        if self.raw_limits[0]>=self.raw_limits[1]:raise ValueError("vmin must be below vmax")
        full_metrics=list(csv.DictReader((analysis/f"{name}_frames.csv").open()))
        self.cube_index={int(float(r["image_id"])):i for i,r in enumerate(full_metrics)}
        self.cube=z["cube8"]
        closed=np.array([r["phase"]=="closed" for r in full_metrics])
        p=np.percentile(np.abs(self.cube[closed]-self.reference8),99.5)
        radius=float(np.ceil(p/5)*5)
        self.difference_limits=(-radius,radius)
        self.cache_key=None;self.cache_image=None

    def at(self,t):
        i=int(np.searchsorted(self.times,t,side="right")-1)
        i=max(0,min(i,len(self.times)-1))
        gap=(i+1<len(self.times) and self.times[i+1]-self.times[i]>1.8 and t-self.times[i]>1.45)
        return i,gap

    def display_image(self,i,mode):
        key=(i,mode)
        if key==self.cache_key:return self.cache_image
        record=self.records[i]
        if mode=="raw":
            a=fits.getdata(record["path"],memmap=False).astype(np.float32)
            a=spatial_mean(a,2)
            limits=self.raw_limits
        else:
            a=self.cube[self.cube_index[record["image_id"]]]-self.reference8
            limits=self.difference_limits
        # Matches flipud(rot90(data,k=1)) with imshow(origin='lower') in the
        # existing instrument analysis. That display array equals data.T.
        # Raster row zero is at the top, hence the final vertical reversal.
        raster=np.flipud(a.T)
        # Average DN before applying the LUT. Avoid averaging false-color RGBs.
        raster=np.asarray(Image.fromarray(raster).resize((IMAGE_W,IMAGE_H),Image.Resampling.BOX))
        normalized=np.clip((raster-limits[0])/(limits[1]-limits[0]),0,1)
        index=np.minimum((normalized*256).astype(np.uint16),255)
        self.cache_image=Image.fromarray(self.lut[index])
        self.cache_key=key
        return self.cache_image


def number(value):
    return f"{value:g}"


def chart_bounds(data,start,end):
    sel=(data.times>=start-2)&(data.times<=end+2)
    values=data.means[sel]-data.closed_mean
    lo,hi=float(values.min()),float(values.max())
    pad=max(.2,(hi-lo)*.15)
    unit=.2 if hi-lo<3 else 1
    return float(np.floor((min(lo,0)-pad)/unit)*unit),float(np.ceil((max(hi,0)+pad)/unit)*unit)


def draw_chart(draw,data,t,start,end,bounds):
    x=PANEL_X[data.name]+56;y=913;w=IMAGE_W;h=96
    lo,hi=bounds
    px=lambda a: x+(a-start)/(end-start)*w
    py=lambda a: y+h-(a-lo)/(hi-lo)*h
    draw.rectangle((x,y,x+w,y+h),fill="#131c29",outline="#334155")
    for v in sorted(set([lo,0,hi])):
        if lo<=v<=hi:
            yy=py(v);draw.line((x,yy,x+w,yy),fill="#283549")
            draw.text((x-8,yy),number(v),font=FONTS[14],fill=MUTED,anchor="rm")
    if start<=0<=end:
        draw.line((px(0),y,px(0),y+h),fill="#8995a5",width=1)
        draw.text((px(0)+5,y+4),"door",font=FONTS[14],fill="#8592a4")
    points=[]
    for tm,mean in zip(data.times,data.means):
        if start<=tm<=min(t,end):
            points.append((px(tm),py(mean-data.closed_mean)))
    if len(points)>1:draw.line(points,fill=ORANGE,width=2)
    i,_=data.at(t)
    if start<=data.times[i]<=end:
        xx,yy=px(data.times[i]),py(data.means[i]-data.closed_mean)
        draw.ellipse((xx-4,yy-4,xx+4,yy+4),fill=FG)
    ticks=[start,0,end] if start<0<end else [start,end]
    for tick in ticks:
        anchor="la" if tick==start else "ra" if tick==end else "ma"
        draw.text((px(tick),y+h+6),f"{tick:+.1f} s",font=FONTS[14],fill=MUTED,anchor=anchor)


def compose(data,t,mode,speed,title,start,end,bounds):
    canvas=Image.new("RGB",(WIDTH,HEIGHT),BG);draw=ImageDraw.Draw(canvas)
    draw.text((32,18),title,font=BOLD[34],fill=FG)
    draw.text((32,65),f"Golden Hour  |  Deployment-aligned time {t:+07.2f} s  |  Playback {speed:g}×",font=FONTS[20],fill=MUTED)
    frame_records=[]
    for name,sequence in data.items():
        x=PANEL_X[name];ix=x+56;i,gap=sequence.at(t);r=sequence.records[i]
        title=f"COLD · SUN CANNON · 2026-10-14" if name=="cold" else "HOT · PHONE FLASHLIGHT · 2026-10-15"
        draw.text((x,106),title,font=BOLD[26],fill=FG)
        timestamp=datetime.fromisoformat(r["capture_start_utc"]).strftime("%H:%M:%S.%f")[:-3]
        draw.text((x,141),f"{timestamp} UTC   |   Image {r['image_id']}   |   1 s exposure",font=FONTS[19],fill=MUTED)
        phase={"closed":"BEFORE DEPLOY","transition":"DEPLOY TIMING","open":"AFTER DEPLOY"}[r["phase"]]
        phase_color=BLUE if r["phase"]=="closed" else "#dfa0ed" if r["phase"]=="transition" else ORANGE
        draw.text((x,168),phase,font=BOLD[18],fill=phase_color)
        draw.text((x+240,168),f"Detector {sequence.temps[i]:+.3f} °C   |   Mean {sequence.means[i]:.2f} DN",font=FONTS[18],fill=MUTED)
        # Place the brief deployment cue outside detector pixels, so the cue
        # cannot be mistaken for an actual flash or change in image brightness.
        if -.5<=t<=.75:
            draw.text((x+912,168),"DOOR DEPLOY",font=BOLD[18],fill="#9ba7b5",anchor="ra")
        canvas.paste(sequence.display_image(i,mode),(ix,IMAGE_Y))
        draw.rectangle((ix,IMAGE_Y,ix+IMAGE_W,IMAGE_Y+IMAGE_H),outline="#334155",width=1)
        # Constant, faint time annotation in each image pane. The actual UTC
        # times differ; both explicitly correspond to the same aligned t=0.
        door_time=datetime.fromisoformat(sequence.event).strftime("%H:%M:%S.%f")[:-3]
        overlay=ImageDraw.Draw(canvas,"RGBA")
        stamp=f"Door deploy {door_time} UTC  |  t = 0"
        overlay.text((ix+IMAGE_W-14,IMAGE_Y+IMAGE_H-27),stamp,font=FONTS[16],fill=(0,0,0,100),anchor="ra",stroke_width=1,stroke_fill=(0,0,0,100))
        overlay.text((ix+IMAGE_W-14,IMAGE_Y+IMAGE_H-27),stamp,font=FONTS[16],fill=(238,244,250,135),anchor="ra")
        if gap:
            draw.rectangle((ix+12,IMAGE_Y+12,ix+IMAGE_W-12,IMAGE_Y+43),fill="#111827")
            draw.text((ix+22,IMAGE_Y+17),"Incomplete-frame gap · holding last valid image",font=FONTS[18],fill="#ffcf8b")
        limits=sequence.raw_limits if mode=="raw" else sequence.difference_limits
        gradient=sequence.lut[np.minimum((np.linspace(0,1,IMAGE_W)*256).astype(int),255)]
        canvas.paste(Image.fromarray(np.repeat(gradient[None],14,axis=0)),(ix,818))
        draw.text((ix,839),f"vmin {number(limits[0])}",font=FONTS[17],fill=MUTED)
        draw.text((ix+IMAGE_W,839),f"vmax {number(limits[1])}",font=FONTS[17],fill=MUTED,anchor="ra")
        label="DN · fixed linear stretch" if mode=="raw" else "ΔDN / pixel · fixed linear stretch"
        draw.text((ix+IMAGE_W/2,839),label,font=FONTS[17],fill=FG,anchor="ma")
        if mode!="raw":draw.line((ix+IMAGE_W/2,815,ix+IMAGE_W/2,835),fill=FG,width=2)
        delta=sequence.means[i]-sequence.closed_mean
        draw.text((ix,877),f"Image mean − closed baseline: {delta:+.3f} DN / pixel",font=BOLD[18],fill=FG)
        draw_chart(draw,sequence,t,start,end,bounds[name])
        frame_records.append(dict(test=name,image_id=r["image_id"],capture_midpoint_relative_s=r["dt_mid"],
                                  phase=r["phase"],gap_hold=gap,vmin=limits[0],vmax=limits[1]))
    footer=("Full detector · DN averaged down for display · fixed limits throughout · no temporal interpolation" if mode=="raw" else
            "Fixed closed-door reference subtracted · 8×8 spatial means · thermal drift retained · fixed limits throughout")
    draw.text((32,1050),footer,font=FONTS[17],fill=MUTED)
    return canvas,frame_records


def encode(data,out,stem,mode,speed,fps,start,end,title):
    step=speed/fps
    # Anchor the common playback grid at EXACTLY t=0. Both deployment markers
    # therefore occur in the same encoded video frame. The endpoints move by
    # less than one video frame and retain the first/last measured exposures.
    first_tick=int(np.ceil(start/step));last_tick=int(np.ceil(end/step))
    times=[k*step for k in range(first_tick,last_tick+1)]
    start,end=times[0],times[-1]
    # One-second endpoint holds make first and last images easy to inspect.
    times=[start]*fps+times+[end]*fps
    bounds={name:chart_bounds(seq,start,end)for name,seq in data.items()}
    path=out/f"{stem}.mp4"
    args=["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-vcodec","rawvideo",
          "-pix_fmt","rgb24","-s",f"{WIDTH}x{HEIGHT}","-r",str(fps),"-i","-","-an",
          "-c:v","libx264","-preset","fast","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",
          "-metadata",f"title={title}",str(path)]
    process=subprocess.Popen(args,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    records=[];previews=[]
    preview_targets=[start,min(max(2,start),end),end]
    preview_indices={min(range(len(times)),key=lambda i:abs(times[i]-t)):label for t,label in zip(preview_targets,["before","after","last"])}
    try:
        for i,t in enumerate(times):
            frame,fr=compose(data,float(t),mode,speed,title,start,end,bounds)
            process.stdin.write(frame.tobytes())
            for r in fr:records.append(dict(video_frame=i,video_seconds=i/fps,relative_seconds=float(t),**r))
            if i in preview_indices:
                name=f"{stem}_{preview_indices[i]}.png";frame.save(out/name);previews.append(name)
            if i%120==0:print(f"{stem}: {i+1}/{len(times)} frames",flush=True)
        process.stdin.close()
        error=process.stderr.read().decode();code=process.wait()
        if code:raise RuntimeError(f"ffmpeg failed ({code}): {error}")
    except BaseException:
        if process.poll()is None:process.kill();process.wait()
        raise
    with (out/f"{stem}_frame_map.csv").open("w")as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    return dict(file=path.name,title=title,mode=mode,playback_speed=speed,fps=fps,frames=len(times),
                duration_s=len(times)/fps,relative_time_range_s=[start,end],endpoint_holds_s=1,
                deployment_video_frame=times.index(0.0),deployment_video_seconds=times.index(0.0)/fps,
                door_time_annotations="Faint UTC timestamp in each pane; simultaneous brief header cue at aligned t=0; labeled vertical line in each mean plot",
                mean_plot_limits=bounds,previews=previews)


def write_gallery(out,movies,manifest):
    import html
    cards=[]
    for m in movies:
        poster=next((p for p in m["previews"]if p.endswith("_after.png")),m["previews"][0])
        cards.append(f'<section><h2>{html.escape(m["title"])}</h2><p>{m["playback_speed"]:g}× playback · {m["duration_s"]:.1f} seconds</p><video controls loop playsinline preload="metadata" poster="{poster}" src="{m["file"]}"></video><p><a href="{m["file"]}">Open / download MP4</a></p></section>')
    c,h=manifest["raw_limits_dn"]["cold"],manifest["raw_limits_dn"]["hot"]
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TVAC2 light-leak movies</title>
<style>body{{background:#0c1018;color:#e2e8f0;font:17px/1.6 system-ui,sans-serif;max-width:1200px;margin:auto;padding:30px}}h1{{font-size:36px}}section{{margin:40px 0}}video{{width:100%;background:black;border:1px solid #334155;border-radius:10px}}a{{color:#93c5fd}}p{{color:#b7c4d5}}code{{color:white}}</style>
<h1>TVAC2 · cold and hot deployment movies</h1><p>Exact <strong>Sunset / Golden Hour</strong> lookup table from the SunCET instrument simulator. Cold is on the left, hot on the right. Acquisition times are aligned relative to each EPS door-deployment marker.</p>
<p>Raw-image fixed limits: <strong>cold {c[0]:g}–{c[1]:g} DN</strong>; <strong>hot {h[0]:g}–{h[1]:g} DN</strong>. These limits were selected once from the closed-door reference and are unchanged across time and both raw movies. The baseline-subtracted movie has its own fixed limits. No thermal correction, frame normalization, or temporal interpolation is applied.</p>
<p>The shared window runs from {movies[0]["relative_time_range_s"][0]:+.1f} to {movies[0]["relative_time_range_s"][1]:+.1f} seconds around deployment. Complete transition frames are included. Incomplete images are omitted, and gaps are labeled when the last valid exposure is held. The earlier cold block across the nine-minute acquisition gap and the later cold-only tail are outside this shared comparison.</p>
{''.join(cards)}<p>Each MP4 has a matching <code>_frame_map.csv</code> linking every rendered frame to its original image IDs, timestamps, and fixed stretch. <code>manifest.json</code> records the exact LUT hash and rendering settings.</p></html>'''
    (out/"index.html").write_text(text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis",type=Path,default=DEFAULT_ANALYSIS)
    p.add_argument("--out",type=Path)
    p.add_argument("--lut",type=Path,default=DEFAULT_LUT)
    for name in ["cold","hot"]:
        p.add_argument(f"--{name}-vmin",type=float)
        p.add_argument(f"--{name}-vmax",type=float)
    args=p.parse_args();out=args.out or args.analysis/"movies";out.mkdir(parents=True,exist_ok=True)
    if not shutil.which("ffmpeg"):raise RuntimeError("ffmpeg is required")
    lut=np.loadtxt(args.lut,delimiter=",",skiprows=1,dtype=np.uint8)
    if lut.shape!=(256,3):raise ValueError("Expected exact 256-entry RGB lookup table")
    shutil.copyfile(args.lut,out/"golden_hour_lut.csv")
    selections=json.loads((args.analysis/"selection.json").read_text())
    data={name:MovieData(name,selections[name],args.analysis,lut,args)for name in ["cold","hot"]}
    start=max(seq.times[0]for seq in data.values());end=min(seq.times[-1]for seq in data.values())
    movies=[]
    for stem,mode,speed,a,b,title in [
        ("cold_hot_fixed_stretch","raw",4,start,end,"TVAC2 · cold / hot · fixed stretch"),
        ("cold_hot_deployment_slow","raw",1,-8,12,"TVAC2 · door deployment · 1× playback"),
        ("cold_hot_baseline_difference","difference",4,start,end,"TVAC2 · change from closed-door reference")]:
        movies.append(encode(data,out,stem,mode,speed,12,float(a),float(b),title))
    manifest=dict(colormap="Sunset / Golden Hour",lut_source=str(args.lut.resolve()),
        lut_sha256=hashlib.sha256(args.lut.read_bytes()).hexdigest(),
        raw_limits_dn={k:list(v.raw_limits)for k,v in data.items()},
        difference_limits_dn_per_pixel={k:list(v.difference_limits)for k,v in data.items()},
        fixed_limit_selection="Raw: closed-door mean image 0.5/99.5 percentiles rounded out to 25 DN. Difference: closed-door residual absolute 99.5 percentile rounded out to 5 DN.",
        dimensions=[WIDTH,HEIGHT],orientation="Instrument convention: flipud(rot90(data,k=1)), origin lower; raster flipud(data.T)",
        raw_spatial_display="2x2 DN means, then BOX resampling in DN to 800x602 pixels per panel",
        difference_spatial_display="8x8 DN means minus fixed closed-door mean, then BOX resampling in DN",
        temporal_interpolation=False,frame_selection="Latest complete exposure midpoint at or before the deployment-aligned playback time; incomplete-frame holds are labeled.",
        deployment_markers_utc={k:v.event for k,v in data.items()},movies=movies)
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    write_gallery(out,movies,manifest)
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":main()
