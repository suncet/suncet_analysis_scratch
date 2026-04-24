"""CSIE detector 0 thermistor: FITS ``DET0_TEM`` is DN; map to °C via polynomial."""

from __future__ import annotations

from typing import Any, Mapping


def CONVERT_csie_det0_therm(val: float) -> float:
    """DET0 thermistor DN → temperature (°C)."""
    v = float(val)
    acc = 0.0
    acc += 1.243100e02 * pow(v, 0)
    acc += -1.339600e-01 * pow(v, 1)
    acc += 9.685700e-05 * pow(v, 2)
    acc += -4.346300e-08 * pow(v, 3)
    acc += 9.978400e-12 * pow(v, 4)
    acc += -9.270400e-16 * pow(v, 5)
    return acc


def det0_temperature_deg_c_from_header(header: Mapping[str, Any]) -> float:
    """Read ``DET0_TEM`` (DN) from a FITS-like header mapping; return °C."""
    if "DET0_TEM" not in header:
        raise KeyError("DET0_TEM not found in FITS header")
    return CONVERT_csie_det0_therm(float(header["DET0_TEM"]))
