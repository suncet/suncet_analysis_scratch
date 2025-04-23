import astropy.time
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import time_support

from aiapy.calibrate import degradation
from aiapy.calibrate.util import get_correction_table

# This lets you pass `astropy.time.Time` objects directly to matplotlib
time_support(format="jyear")

correction_table = get_correction_table()

aia_channels = [171, 193] * u.angstrom

start_time = astropy.time.Time("2010-03-25T00:00:00", scale="utc")
end_time = astropy.time.Time("2013-03-25T00:00:00", scale="utc")
time_range = start_time + np.arange(0, (end_time - start_time).to(u.day).value, 7) * u.day

degradations = {
    channel: degradation(channel, time_range, correction_table=correction_table) for channel in aia_channels
}

fig = plt.figure()
ax = fig.gca()

for channel in aia_channels:
    ax.plot(time_range, degradations[channel], label=f"{channel:latex}")

ax.set_xlim(time_range[[0, -1]])
ax.legend(frameon=False, ncol=4, bbox_to_anchor=(0.5, 1), loc="lower center")
ax.set_xlabel("Time")
ax.set_ylabel("Degradation")


pass