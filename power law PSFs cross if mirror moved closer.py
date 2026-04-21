import numpy as np
import matplotlib.pyplot as plt
from scipy import special

# ------------------------------------------------------------------
# PSD parameters (Martínez‑Galarce SUVI style)
A = 610.322
B = 120                    # break spatial frequency (mm⁻¹)
C = 1.089
K = 1.0 / (2 * np.sqrt(np.pi)) * special.gamma((C + 1) / 2) / special.gamma(C / 2)

lambda_ang = 185.0          # Å
lambda_mm  = lambda_ang * 1e-7  # mm

# ------------------------------------------------------------------
# Angular grid (degrees → radians)
th_deg = np.logspace(-6, 1, 2000)
th_rad = np.deg2rad(th_deg)

# 2‑D PSD evaluated at f = θ/λ  (Martínez‑Galarce Eq. 5)
f_mirror = th_rad / lambda_mm          # mm⁻¹
PSD_2D   = K * (A * B) / (1 + (B * f_mirror) ** 2) ** ((C + 1) / 2)

# Single‑mirror PSF wing  (θ · PSD₂D)
PSF_far = th_rad * PSD_2D
area_far = np.trapz(PSF_far, th_rad)

# ------------------------------------------------------------------
# Move detector 4× closer (z0 → z1) to illustrate crossing
z0, z1 = 1000.0, 250.0     # mm  (scale = 4)
scale  = z0 / z1

# Near‑plane PSF expressed on far‑plane θ grid (raw)
PSF_near_raw = scale * th_rad * K * (A * B) / (1 + (B * (scale * th_rad) / lambda_mm) ** 2) ** ((C + 1) / 2)
area_near_raw = np.trapz(PSF_near_raw, th_rad)

# Renormalise so total scatter matches far‑plane PSF
equal_factor = area_far / area_near_raw
PSF_near = PSF_near_raw * equal_factor
area_near = np.trapz(PSF_near, th_rad)

# ------------------------------------------------------------------
# Plot
plt.figure(figsize=(6, 4))
plt.loglog(th_deg, PSF_far,      label=f"far‑plane PSF   ∫={area_far:.3e}", color='tab:blue')
plt.loglog(th_deg, PSF_near_raw, label=f"near‑plane RAW  ∫={area_near_raw:.3e}", color='tab:green', ls='--')
plt.loglog(th_deg, PSF_near,     label=f"near‑plane renorm ∫={area_near:.3e}", color='tab:orange')

# mark the break angle θ_c = λ/B
th_c_deg = (lambda_mm / B) * 180 / np.pi
plt.axvline(th_c_deg, ls='--', color='grey', label="θ_c = λ/B")

plt.xlabel("θ  (degrees)")
plt.ylabel("PSF (arb units)")
plt.ylim(1e-5, 1e-2)
plt.xlim(1e-6, 1e-3)
plt.title("Far vs near detector PSF wings (SunCET parameters)")
plt.legend(fontsize='8')
plt.tight_layout()
plt.show()
