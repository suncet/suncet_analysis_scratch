"""Load CSIE reference test-pattern FITS as uint16 from ``$suncet_data/test_data/``."""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
from astropy.io import fits


def _suncet_data_root() -> str:
    root = os.getenv("suncet_data")
    if not root:
        raise SystemExit("Environment variable suncet_data is not set.")
    return root


def load_csie_fits_as_uint16(path: str) -> np.ndarray:
    """
    Same semantics as ``load_reference_fits_array`` in
    ``read_em_bla_ingest_and_display_csie_em_image.py``:
    require a 16-bit integer primary array and reinterpret bit patterns as ``uint16``
    (FITS is often ``int16``).
    """
    p = os.path.expanduser(path)
    data = fits.getdata(p)
    if data is None:
        raise SystemExit(f"No data array in FITS: {p}")
    a = np.asanyarray(data)
    if a.itemsize != 2 or a.dtype.kind not in "iu":
        raise SystemExit(
            f"FITS must be 16-bit integer for CSIE diff; got {a.dtype!r} in {p}"
        )
    return np.asanyarray(a, dtype=np.uint16)


def load_reference_test_pattern(
    binning: Literal[32, 96],
) -> np.ndarray:
    """
    Load ``reference_test_pattern_{binning}.fits`` from ``$suncet_data/test_data/``.

    Uses :func:`load_csie_fits_as_uint16` only (no alternate read path).
    """
    if binning not in (32, 96):
        raise ValueError(f"binning must be 32 or 96, got {binning!r}")
    path = os.path.join(
        _suncet_data_root(),
        "test_data",
        f"reference_test_pattern_{binning}.fits",
    )
    return load_csie_fits_as_uint16(path)


if __name__ == "__main__":
    a32 = load_reference_test_pattern(32)
    a96 = load_reference_test_pattern(96)
    print("32:", a32.shape, a32.dtype, a32.min(), a32.max())
    print("96:", a96.shape, a96.dtype, a96.min(), a96.max())
