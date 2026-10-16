# suncet_analysis_scratch
 Basically scratch paper but its code. Don't expect especially well documented things here. 

## TVAC2 light-leak analysis

Analyze the October 14–15, 2026 door deployments using the processed TVAC2
`csie_images` inventory, FITS/META products, and exported telemetry:

```sh
python3 tvac2_light_leak_analysis.py --data /path/to/2026-10-07_tvac2
python3 make_tvac2_light_leak_movies.py
```

The analysis also runs `tvac2_thermal_followup.py`, which compares the hot
sequence with temperature telemetry and earlier exposure pairs. Reports,
statistics, caches, and movies go into the ignored `tvac2_light_leak_output/`
directory. Use `--out` to choose another destination; the movie script's
`--analysis` option selects an existing analysis output directory.

Movies require `ffmpeg` on `PATH` and use macOS Arial fonts. The Golden Hour
lookup table defaults to the sibling `suncet_instrument_simulator` checkout;
use `--lut` to supply another location. Cold and hot stretches remain fixed
throughout playback, and their deployment markers share the same video frame.
