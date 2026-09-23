"""Export the SunCET plasma-temperature response as a CSV table.

The response is the wavelength integral of the CHIANTI emissivity spectrum and
the saved SunCET wavelength response.  That wavelength response encodes the
instrument effective area plus the detector and pixel conversion needed to
produce DN.  The default log-temperature range matches the existing
``SunCET temperature responsivity Comparison to SUVI.png`` figure.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.integrate import trapezoid
from sunpy.io.special.genx import read_genx


DEFAULT_SPECTRAL_RESPONSE = Path("calibration/suncet_spectral_resp.genx")
DEFAULT_EMISSIVITY = Path("ancillary/emissivity/aia_V9_fullemiss.nc")
DEFAULT_OUTPUT = Path("suncet_temperature_response.csv")


def default_data_path(relative_path: Path) -> Path:
    """Resolve a data path below the ``suncet_data`` environment variable."""
    data_root = os.environ.get("suncet_data")
    if not data_root:
        raise RuntimeError(
            "suncet_data is not set; pass explicit --spectral-response and "
            "--emissivity paths or set the environment variable."
        )
    return Path(data_root).expanduser() / relative_path


def calculate_temperature_response(
    spectral_response_path: Path,
    emissivity_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return log10(temperature/K) and response in DN cm^5 s^-1 pix^-1."""
    spectral_data = read_genx(spectral_response_path)
    response_wavelength = np.asarray(spectral_data["SAVEGEN0"], dtype=float)
    spectral_response = np.asarray(spectral_data["SAVEGEN1"], dtype=float)

    if response_wavelength.shape != spectral_response.shape:
        raise ValueError("Spectral-response wavelength and value arrays differ in shape.")
    if response_wavelength.ndim != 1 or len(response_wavelength) < 2:
        raise ValueError("Spectral response must contain a one-dimensional wavelength grid.")
    if np.any(np.diff(response_wavelength) <= 0):
        raise ValueError("Spectral-response wavelengths must be strictly increasing.")

    with xr.open_dataset(emissivity_path) as emissivity_data:
        for required_name in ("logte", "wave", "total"):
            if required_name not in emissivity_data:
                raise KeyError(f"Emissivity file is missing {required_name!r}.")
        log_temperature = np.asarray(emissivity_data["logte"].values, dtype=float)
        emissivity_wavelength = np.asarray(
            emissivity_data["wave"].values,
            dtype=float,
        )
        total_emissivity = np.asarray(emissivity_data["total"].values, dtype=float)

    expected_shape = (len(log_temperature), len(emissivity_wavelength))
    if total_emissivity.shape != expected_shape:
        raise ValueError(
            f"Expected emissivity shape {expected_shape}, got {total_emissivity.shape}."
        )

    interpolated_response = np.interp(
        emissivity_wavelength,
        response_wavelength,
        spectral_response,
        left=0.0,
        right=0.0,
    )
    temperature_response = trapezoid(
        total_emissivity * interpolated_response[np.newaxis, :],
        x=emissivity_wavelength,
        axis=1,
    )
    return log_temperature, temperature_response


def response_dataframe(
    log_temperature: np.ndarray,
    temperature_response: np.ndarray,
    *,
    min_log_temperature: float | None,
    max_log_temperature: float | None,
) -> pd.DataFrame:
    """Create the flat, unit-labelled output table for the requested range."""
    selected = np.ones(len(log_temperature), dtype=bool)
    if min_log_temperature is not None:
        selected &= log_temperature >= min_log_temperature
    if max_log_temperature is not None:
        selected &= log_temperature <= max_log_temperature
    if not np.any(selected):
        raise ValueError("The requested log-temperature range contains no samples.")

    # The NetCDF grid is float32 but is physically defined in exact 0.05-dex
    # increments.  Remove representation noise such as 5.05000019 in the CSV.
    plotted_log_temperature = np.round(log_temperature[selected], decimals=2)
    return pd.DataFrame(
        {
            "log10_electron_temperature_K": plotted_log_temperature,
            "electron_temperature_K": 10.0**plotted_log_temperature,
            "suncet_temperature_response_DN_cm5_s-1_pix-1": (
                temperature_response[selected]
            ),
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spectral-response",
        type=Path,
        default=None,
        help=(
            "SunCET spectral-response GENX file. Default: "
            "$suncet_data/calibration/suncet_spectral_resp.genx"
        ),
    )
    parser.add_argument(
        "--emissivity",
        type=Path,
        default=None,
        help=(
            "CHIANTI emissivity NetCDF file. Default: "
            "$suncet_data/ancillary/emissivity/aia_V9_fullemiss.nc"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--min-log-temperature",
        type=float,
        default=5.0,
        help="Minimum log10 electron temperature in K. Default: 5.0.",
    )
    parser.add_argument(
        "--max-log-temperature",
        type=float,
        default=7.0,
        help="Maximum log10 electron temperature in K. Default: 7.0.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spectral_response_path = (
        args.spectral_response.expanduser()
        if args.spectral_response is not None
        else default_data_path(DEFAULT_SPECTRAL_RESPONSE)
    )
    emissivity_path = (
        args.emissivity.expanduser()
        if args.emissivity is not None
        else default_data_path(DEFAULT_EMISSIVITY)
    )
    output_path = args.output.expanduser()

    log_temperature, temperature_response = calculate_temperature_response(
        spectral_response_path,
        emissivity_path,
    )
    output = response_dataframe(
        log_temperature,
        temperature_response,
        min_log_temperature=args.min_log_temperature,
        max_log_temperature=args.max_log_temperature,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    peak_row = output.loc[
        output["suncet_temperature_response_DN_cm5_s-1_pix-1"].idxmax()
    ]
    print(f"Wrote {len(output):,} rows to {output_path.resolve()}")
    print(
        "Peak response: "
        f"{peak_row['suncet_temperature_response_DN_cm5_s-1_pix-1']:.12e} "
        "DN cm^5 s^-1 pix^-1 at log10(T/K)="
        f"{peak_row['log10_electron_temperature_K']:.2f}"
    )


if __name__ == "__main__":
    main()
