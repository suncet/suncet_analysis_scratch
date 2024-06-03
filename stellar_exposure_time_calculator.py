import os
import pandas as pd
import numpy as np
from scipy.integrate import simps
import astropy.units as u

# Load the SunCET effective area data
suncet = pd.read_csv(os.getenv('suncet_data') + '/effective_area/suncet_fm1_effective_area.csv')
suncet['wavelength'] = suncet['wavelength [nm]'] * 10.0 * u.angstrom
suncet['effective area'] = suncet['effective area [cm2]'] * u.cm**2

# Load the Procyon data
procyon_data = pd.read_csv(os.getenv('suncet_data') + '/reference_solar_spectrum/F5V_inactive_1Ang_smex24_Procyon.csv')
procyon_data['wavelength'] = procyon_data['wavelength [angstrom]'] * u.angstrom
procyon_data['photon_count'] = procyon_data['photon_count [ph/cm2/s/angstrom]'] * u.ph / (u.cm**2 * u.s * u.angstrom)

# Interpolate Procyon photon flux to SunCET wavelengths
photon_flux_interp = np.interp(suncet['wavelength'], procyon_data['wavelength'], procyon_data['photon_count']) * procyon_data['photon_count'][0].unit

# Constants
quantum_yield = 19 * u.electron / u.photon # electrons/photon
quantum_efficiency = 0.85  # 85%

# Calculate the expected signal (product of effective area, interpolated photon flux, quantum efficiency, and quantum yield)
spectral_signal = (suncet['effective area'] * photon_flux_interp * quantum_efficiency * quantum_yield) * quantum_yield.unit

# Integrate the expected signal over the SunCET wavelength range
integrated_signal = simps(spectral_signal, suncet['wavelength']) * spectral_signal[0].unit * suncet['wavelength'][0].unit

# Assume a point spread function that will spread out the signal -- for now assuming that the PSF is narrower than a single pixel
integrated_signal *= 1/u.pix

# Estimate noise
read_noise = 5 * u.electron / u.pix
dark_noise_20C = 20 * u.electron / (u.pixel * u.s)  # electrons/pixel/second
temperature_decrease = 20 - (-10)  # 30°C decrease
dark_noise_factor = 2 ** (temperature_decrease / 5.5)
dark_noise = dark_noise_20C / dark_noise_factor  # electrons/pixel/second

# Desired SNR (e.g., 10 for a good signal-to-noise ratio)
desired_snr = 10

# Calculate total noise (quadrature sum of read noise, dark noise, and photon noise)
def calculate_total_noise(signal_per_second, exposure_time):
    photon_noise = np.sqrt(signal_per_second * exposure_time)
    total_noise = np.sqrt(read_noise + dark_noise * exposure_time + photon_noise**2)
    total_noise = total_noise.value * (u.electron/u.pixel) # Force units to not be square root... dunno why the math doesn't work out to be same units as signal
    return total_noise

# Estimate the required exposure time iteratively
def estimate_exposure_time(signal_per_second, desired_snr):
    exposure_time = 1 * u.s  # initial guess in seconds
    signal = signal_per_second * exposure_time
    noise = calculate_total_noise(signal_per_second, exposure_time)

    snr = signal / noise
    while snr < desired_snr:
        exposure_time += 1 * u.s
        signal = signal_per_second * exposure_time
        noise = calculate_total_noise(signal_per_second, exposure_time)
        snr = signal / noise
    
    return exposure_time

# Calculate the exposure time needed to achieve the desired SNR
required_exposure_time = estimate_exposure_time(integrated_signal, desired_snr)
print(f"Estimated exposure time for SNR of {desired_snr}: {required_exposure_time.to(u.s).value} seconds")

pass