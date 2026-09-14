"""Tests for biotek_to_excel.py with inline Gen5-format fixtures.

Each fixture is a tiny but structurally faithful slice of a real Gen5 export:
two wavelengths (734, 950), one labeled well (A1), one blank well (B1), so the
parser exercises every section without us shipping large data files.

Run with:
    pytest -q test_biotek_to_excel.py
"""
from __future__ import annotations

import io
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from pchemapps import biotek_to_excel as b2x


# --- Fixtures: stringified Gen5 .txt content ---------------------------------

FULL_RUN = """\
Procedure Summary

Plate Type\tCostar 96 flat bottom
Set Temperature\tTemperature:  Setpoint 25 \xb0C
Plate In/Out\tPlate Out,In   load sample
Read\tRead:  Intial Read before antioxidant (A) 734, 950
Plate In/Out\tPlate Out,In   Add Antioxidant
Start Kinetic\tStart Kinetic [Run 0:06:20, Interval 0:01:19]
Read\tRead:  734,950 (A) 734, 950
End Kinetic\tEnd Kinetic

Layout
\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12
A\tSPL1\t\t\t\t\t\t\t\t\t\t\t\tWell ID
\t\t\t\t\t\t\t\t\t\t\t\t\tName
B\tSPL2\t\t\t\t\t\t\t\t\t\t\t\tWell ID
\t\t\t\t\t\t\t\t\t\t\t\t\tName

Well IDs

Well ID\tName
SPL1\t
SPL2\t

Intial Read before antioxidant:734
\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12
A\t0.500\t\t\t\t\t\t\t\t\t\t\t\tIntial Read before antioxidant:734
B\t0.490\t\t\t\t\t\t\t\t\t\t\t\tIntial Read before antioxidant:734

Intial Read before antioxidant:950
\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12
A\t0.080\t\t\t\t\t\t\t\t\t\t\t\tIntial Read before antioxidant:950
B\t0.082\t\t\t\t\t\t\t\t\t\t\t\tIntial Read before antioxidant:950

734,950:734

Kinetic read\tA1\tA2\tA3\tA4\tA5\tA6\tA7\tA8\tA9\tA10\tA11\tA12\tB1\tB2\tB3\tB4\tB5\tB6\tB7\tB8\tB9\tB10\tB11\tB12\tC1\tC2\tC3\tC4\tC5\tC6\tC7\tC8\tC9\tC10\tC11\tC12\tD1\tD2\tD3\tD4\tD5\tD6\tD7\tD8\tD9\tD10\tD11\tD12\tE1\tE2\tE3\tE4\tE5\tE6\tE7\tE8\tE9\tE10\tE11\tE12\tF1\tF2\tF3\tF4\tF5\tF6\tF7\tF8\tF9\tF10\tF11\tF12\tG1\tG2\tG3\tG4\tG5\tG6\tG7\tG8\tG9\tG10\tG11\tG12\tH1\tH2\tH3\tH4\tH5\tH6\tH7\tH8\tH9\tH10\tH11\tH12
0:00:00\t0.400\t\t\t\t\t\t\t\t\t\t\t\t0.390\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t
0:01:19\t0.300\t\t\t\t\t\t\t\t\t\t\t\t0.380\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t
0:02:38\t0.200\t\t\t\t\t\t\t\t\t\t\t\t0.375\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t

734,950:950

Kinetic read\tA1\tA2\tA3\tA4\tA5\tA6\tA7\tA8\tA9\tA10\tA11\tA12\tB1\tB2\tB3\tB4\tB5\tB6\tB7\tB8\tB9\tB10\tB11\tB12\tC1\tC2\tC3\tC4\tC5\tC6\tC7\tC8\tC9\tC10\tC11\tC12\tD1\tD2\tD3\tD4\tD5\tD6\tD7\tD8\tD9\tD10\tD11\tD12\tE1\tE2\tE3\tE4\tE5\tE6\tE7\tE8\tE9\tE10\tE11\tE12\tF1\tF2\tF3\tF4\tF5\tF6\tF7\tF8\tF9\tF10\tF11\tF12\tG1\tG2\tG3\tG4\tG5\tG6\tG7\tG8\tG9\tG10\tG11\tG12\tH1\tH2\tH3\tH4\tH5\tH6\tH7\tH8\tH9\tH10\tH11\tH12
0:00:00\t0.050\t\t\t\t\t\t\t\t\t\t\t\t0.060\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t
0:01:19\t0.052\t\t\t\t\t\t\t\t\t\t\t\t0.061\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t
0:02:38\t0.054\t\t\t\t\t\t\t\t\t\t\t\t0.062\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t
"""

SPECTRUM_ONLY = """\
Spectrum
Wavelength 1 (300 nm)
\t12
A\t0.987\tSpectrum Read#1
"""

# Initial reads filled, kinetic blocks all blank (like Trolox.txt was)
INITIAL_ONLY = """\
Procedure Summary

Plate Type\tCostar 96 flat bottom

Layout
\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12
A\tSPL1\t\t\t\t\t\t\t\t\t\t\t\tWell ID

Well IDs

Well ID\tName
SPL1\t

Intial Read before antioxidant:734
\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12
A\t0.597\t\t\t\t\t\t\t\t\t\t\t\tIntial Read before antioxidant:734

Intial Read before antioxidant:950
\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12
A\t0.097\t\t\t\t\t\t\t\t\t\t\t\tIntial Read before antioxidant:950
"""

# 2023 Demmi-era Gen5 dialect: lowercase 'intial read', no '734,950:wl' banner,
# bare 'Kinetic read\t...' blocks ordered by wavelength, protocol duplicated.
DEMMI_DIALECT = """\
Procedure Summary

Plate Type\t96 WELL PLATE
Read\tRead:  intial read before antioxidant (A) 734, 950
Start Kinetic\tStart Kinetic [Run 0:00:18, Interval 0:00:09]

Layout
\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12
A\tSPL1\t\t\t\t\t\t\t\t\t\t\t\tWell ID

intial read before antioxidant:734
\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12
A\t0.600\t\t\t\t\t\t\t\t\t\t\t\tintial read before antioxidant:734

intial read before antioxidant:950
\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12
A\t0.090\t\t\t\t\t\t\t\t\t\t\t\tintial read before antioxidant:950

Kinetic read\tA1\tA2\tA3\tA4\tA5\tA6\tA7\tA8\tA9\tA10\tA11\tA12\tB1\tB2\tB3\tB4\tB5\tB6\tB7\tB8\tB9\tB10\tB11\tB12\tC1\tC2\tC3\tC4\tC5\tC6\tC7\tC8\tC9\tC10\tC11\tC12\tD1\tD2\tD3\tD4\tD5\tD6\tD7\tD8\tD9\tD10\tD11\tD12\tE1\tE2\tE3\tE4\tE5\tE6\tE7\tE8\tE9\tE10\tE11\tE12\tF1\tF2\tF3\tF4\tF5\tF6\tF7\tF8\tF9\tF10\tF11\tF12\tG1\tG2\tG3\tG4\tG5\tG6\tG7\tG8\tG9\tG10\tG11\tG12\tH1\tH2\tH3\tH4\tH5\tH6\tH7\tH8\tH9\tH10\tH11\tH12
0:00:00\t0.500\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t
0:00:09\t0.400\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t

Kinetic read\tA1\tA2\tA3\tA4\tA5\tA6\tA7\tA8\tA9\tA10\tA11\tA12\tB1\tB2\tB3\tB4\tB5\tB6\tB7\tB8\tB9\tB10\tB11\tB12\tC1\tC2\tC3\tC4\tC5\tC6\tC7\tC8\tC9\tC10\tC11\tC12\tD1\tD2\tD3\tD4\tD5\tD6\tD7\tD8\tD9\tD10\tD11\tD12\tE1\tE2\tE3\tE4\tE5\tE6\tE7\tE8\tE9\tE10\tE11\tE12\tF1\tF2\tF3\tF4\tF5\tF6\tF7\tF8\tF9\tF10\tF11\tF12\tG1\tG2\tG3\tG4\tG5\tG6\tG7\tG8\tG9\tG10\tG11\tG12\tH1\tH2\tH3\tH4\tH5\tH6\tH7\tH8\tH9\tH10\tH11\tH12
0:00:00\t0.070\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t
0:00:09\t0.072\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t

Procedure Summary

Plate Type\t96 WELL PLATE

Layout
\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12
A\tSPL1\t\t\t\t\t\t\t\t\t\t\t\tWell ID

intial read before antioxidant:734
\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12
A\t0.600\t\t\t\t\t\t\t\t\t\t\t\tintial read before antioxidant:734

intial read before antioxidant:950
\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12
A\t0.090\t\t\t\t\t\t\t\t\t\t\t\tintial read before antioxidant:950

Kinetic read\tA1\tA2\tA3\tA4\tA5\tA6\tA7\tA8\tA9\tA10\tA11\tA12\tB1\tB2\tB3\tB4\tB5\tB6\tB7\tB8\tB9\tB10\tB11\tB12\tC1\tC2\tC3\tC4\tC5\tC6\tC7\tC8\tC9\tC10\tC11\tC12\tD1\tD2\tD3\tD4\tD5\tD6\tD7\tD8\tD9\tD10\tD11\tD12\tE1\tE2\tE3\tE4\tE5\tE6\tE7\tE8\tE9\tE10\tE11\tE12\tF1\tF2\tF3\tF4\tF5\tF6\tF7\tF8\tF9\tF10\tF11\tF12\tG1\tG2\tG3\tG4\tG5\tG6\tG7\tG8\tG9\tG10\tG11\tG12\tH1\tH2\tH3\tH4\tH5\tH6\tH7\tH8\tH9\tH10\tH11\tH12
0:00:00\t0.500\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t
0:00:09\t0.400\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t

Kinetic read\tA1\tA2\tA3\tA4\tA5\tA6\tA7\tA8\tA9\tA10\tA11\tA12\tB1\tB2\tB3\tB4\tB5\tB6\tB7\tB8\tB9\tB10\tB11\tB12\tC1\tC2\tC3\tC4\tC5\tC6\tC7\tC8\tC9\tC10\tC11\tC12\tD1\tD2\tD3\tD4\tD5\tD6\tD7\tD8\tD9\tD10\tD11\tD12\tE1\tE2\tE3\tE4\tE5\tE6\tE7\tE8\tE9\tE10\tE11\tE12\tF1\tF2\tF3\tF4\tF5\tF6\tF7\tF8\tF9\tF10\tF11\tF12\tG1\tG2\tG3\tG4\tG5\tG6\tG7\tG8\tG9\tG10\tG11\tG12\tH1\tH2\tH3\tH4\tH5\tH6\tH7\tH8\tH9\tH10\tH11\tH12
0:00:00\t0.070\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t
0:00:09\t0.072\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t
"""

# Initial reads = #N/A, kinetic blocks have values (like Trolox kinetic read.txt was)
KINETIC_ONLY_NA_INIT = """\
Procedure Summary

Plate Type\tCostar 96 flat bottom

Intial Read before antioxidant:734
\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12
A\t#N/A\t\t\t\t\t\t\t\t\t\t\t\tIntial Read before antioxidant:734

Intial Read before antioxidant:950
\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12
A\t#N/A\t\t\t\t\t\t\t\t\t\t\t\tIntial Read before antioxidant:950

734,950:734

Kinetic read\tA1\tA2\tA3\tA4\tA5\tA6\tA7\tA8\tA9\tA10\tA11\tA12\tB1\tB2\tB3\tB4\tB5\tB6\tB7\tB8\tB9\tB10\tB11\tB12\tC1\tC2\tC3\tC4\tC5\tC6\tC7\tC8\tC9\tC10\tC11\tC12\tD1\tD2\tD3\tD4\tD5\tD6\tD7\tD8\tD9\tD10\tD11\tD12\tE1\tE2\tE3\tE4\tE5\tE6\tE7\tE8\tE9\tE10\tE11\tE12\tF1\tF2\tF3\tF4\tF5\tF6\tF7\tF8\tF9\tF10\tF11\tF12\tG1\tG2\tG3\tG4\tG5\tG6\tG7\tG8\tG9\tG10\tG11\tG12\tH1\tH2\tH3\tH4\tH5\tH6\tH7\tH8\tH9\tH10\tH11\tH12
0:00:00\t0.700\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t
0:01:19\t0.600\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t
"""


# --- Helpers ----------------------------------------------------------------


def _write_temp(content: str, suffix: str = ".txt") -> str:
    """Write content as cp1252 to a NamedTemporaryFile-compatible path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(content.encode("cp1252"))
    return path


def _plate_info_with_a1_labeled() -> pd.DataFrame:
    """PlateInfo where A1 is labeled 'ABTS-Trolox' @ 2.5 mM, everything else blank."""
    df = b2x.make_blank_plate_info()
    df.loc[df["Loc"] == "A1", ["Type", "Sample", "Conc_mM"]] = ["Trolox", "ABTS-Trolox", 2.5]
    return df


# --- Parser unit tests -------------------------------------------------------


def test_split_sections_blank_separator():
    text = "Header1\nbody1\n\nHeader2\nbody2\n"
    secs = b2x.split_sections(text)
    assert [s[0] for s in secs] == ["Header1", "Header2"]
    assert secs[0][1] == ["body1"]
    assert secs[1][1] == ["body2"]


def test_parse_8x12_grid_handles_na_and_blank():
    body = [
        "\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12",
        "A\t0.5\t#N/A\t\t\t\t\t\t\t\t\t\t",
    ]
    grid = b2x.parse_8x12_grid(body)
    assert grid["A1"] == "0.5"
    assert grid["A2"] == ""  # #N/A
    assert grid["A3"] == ""  # blank
    assert grid["B1"] == ""  # missing row


def test_parse_grid_float():
    body = [
        "\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12",
        "A\t0.500\t#N/A\t\t\t\t\t\t\t\t\t\t",
    ]
    out = b2x.parse_grid_float(body)
    assert out["A1"] == 0.5
    assert out["A2"] is None
    assert out["B1"] is None


def test_hms_to_seconds():
    assert b2x.hms_to_seconds("0:00:00") == 0
    assert b2x.hms_to_seconds("0:01:19") == 79
    assert b2x.hms_to_seconds("1:00:00") == 3600


def test_parse_kinetic_basic():
    body = [
        "Kinetic read\t" + "\t".join(b2x.WELLS),
        "0:00:00\t" + "0.400\t" + "0.000\t" * 95 + "0.000",
        "0:01:19\t" + "0.300\t" + "0.000\t" * 95 + "0.000",
    ]
    times, wells, A = b2x.parse_kinetic(body)
    assert times == ["0:00:00", "0:01:19"]
    assert wells == b2x.WELLS
    assert A.shape == (2, 96)
    assert A[0, 0] == pytest.approx(0.400)
    assert A[1, 0] == pytest.approx(0.300)


def test_parse_full_run_picks_up_both_kinetics():
    """Regression: section split puts '734,950:734' banner alone, with the
    'Kinetic read\\t...' table in the next section. Parser must stitch."""
    path = _write_temp(FULL_RUN)
    try:
        pf = b2x.parse_file(path)
        assert set(pf.kinetics.keys()) == {"734", "950"}
        t734, A734 = pf.kinetics["734"]
        assert len(t734) == 3
        assert A734.shape == (3, 96)
        # A1 column is index 0
        np.testing.assert_allclose(A734[:, 0], [0.400, 0.300, 0.200])
        # B1 column (B=row 1, col=1 -> index 12)
        np.testing.assert_allclose(A734[:, 12], [0.390, 0.380, 0.375])
        t950, A950 = pf.kinetics["950"]
        np.testing.assert_allclose(A950[:, 0], [0.050, 0.052, 0.054])
    finally:
        os.unlink(path)


def test_parse_initial_reads():
    path = _write_temp(FULL_RUN)
    try:
        pf = b2x.parse_file(path)
        assert pf.init_reads["734"]["A1"] == pytest.approx(0.500)
        assert pf.init_reads["734"]["B1"] == pytest.approx(0.490)
        assert pf.init_reads["950"]["A1"] == pytest.approx(0.080)
        # an unfilled well is None
        assert pf.init_reads["734"]["H12"] is None
    finally:
        os.unlink(path)


def test_parse_spectrum_only_flag():
    path = _write_temp(SPECTRUM_ONLY)
    try:
        pf = b2x.parse_file(path)
        assert pf.is_spectrum_only
        assert pf.spectrum["wavelength_nm"] == 300.0
        assert pf.spectrum["readings"] == [
            {"Loc": "A12", "Letter": "A", "Number": 12, "A": 0.987},
        ]
    finally:
        os.unlink(path)


def test_parse_initial_only_has_no_kinetics():
    path = _write_temp(INITIAL_ONLY)
    try:
        pf = b2x.parse_file(path)
        assert pf.init_reads["734"]["A1"] == pytest.approx(0.597)
        # no kinetic blocks were present
        assert pf.kinetics == {}
        assert not pf.is_spectrum_only
    finally:
        os.unlink(path)


def test_parse_demmi_dialect():
    """2023 dialect: lowercase 'intial read', no banner, bare Kinetic read
    blocks ordered by wavelength, protocol duplicated. Second copy overwrites
    the first cleanly, A734 and A950 each get one kinetic block."""
    path = _write_temp(DEMMI_DIALECT)
    try:
        pf = b2x.parse_file(path)
        assert "734" in pf.init_reads and "950" in pf.init_reads
        assert pf.init_reads["734"]["A1"] == pytest.approx(0.600)
        assert set(pf.kinetics.keys()) == {"734", "950"}
        t734, A734 = pf.kinetics["734"]
        assert len(t734) == 2
        np.testing.assert_allclose(A734[:, 0], [0.500, 0.400])
        t950, A950 = pf.kinetics["950"]
        np.testing.assert_allclose(A950[:, 0], [0.070, 0.072])
    finally:
        os.unlink(path)


def test_parse_demmi_dialect_with_typo_fixed():
    """04-13 run: same dialect, but the protocol step is spelled 'initial'.

    The section header is a user-typed protocol step name, so both spellings
    must land in init_reads -- otherwise the initial grid is misread as a
    kinetic banner and the real kinetic blocks fall back to k0/k1 names.
    """
    path = _write_temp(DEMMI_DIALECT.replace("intial read", "initial read"))
    try:
        pf = b2x.parse_file(path)
        assert "734" in pf.init_reads and "950" in pf.init_reads
        assert pf.init_reads["734"]["A1"] == pytest.approx(0.600)
        assert set(pf.kinetics.keys()) == {"734", "950"}
        np.testing.assert_allclose(pf.kinetics["734"][1][:, 0], [0.500, 0.400])
        np.testing.assert_allclose(pf.kinetics["950"][1][:, 0], [0.070, 0.072])
    finally:
        os.unlink(path)


def test_parse_kinetic_only_with_na_initials():
    path = _write_temp(KINETIC_ONLY_NA_INIT)
    try:
        pf = b2x.parse_file(path)
        # #N/A initials become None across the grid
        assert all(v is None for v in pf.init_reads["734"].values())
        # kinetic block parsed
        t, A = pf.kinetics["734"]
        assert len(t) == 2
        np.testing.assert_allclose(A[:, 0], [0.700, 0.600])
    finally:
        os.unlink(path)


# --- PlateInfo tests ---------------------------------------------------------


def test_make_blank_plate_info_is_96_rows():
    df = b2x.make_blank_plate_info()
    assert len(df) == 96
    assert list(df.columns) == b2x.PLATEINFO_COLS
    assert df.iloc[0]["Loc"] == "A1"
    assert df.iloc[-1]["Loc"] == "H12"


def test_load_plate_info_tolerates_leading_blank_col(tmp_path):
    """Demmi's 'Blank' template had a leading empty col + 2 empty rows."""
    df = pd.DataFrame({
        "_pad": [None, None, None, None],
        "A": [None, None, "Loc", "A1"],
        "B": [None, None, "Sample", "Trolox"],
        "C": [None, None, "Conc_mM", 2.5],
        "D": [None, None, "Letter", "A"],
        "E": [None, None, "Number", 1],
    })
    p = tmp_path / "pi.xlsx"
    df.to_excel(p, index=False, header=False)
    pi = b2x.load_plate_info(str(p))
    assert pi.iloc[0]["Loc"] == "A1"
    assert pi.iloc[0]["Sample"] == "Trolox"
    assert pi.iloc[0]["Conc_mM"] == 2.5


# --- Data assembly tests -----------------------------------------------------


def test_build_data_long_combines_initial_and_kinetic():
    path = _write_temp(FULL_RUN)
    try:
        pf = b2x.parse_file(path)
        df = b2x.build_data_long(pf, b2x.make_blank_plate_info())
        # 2 wavelengths x 2 wells x (1 initial + 3 kinetic) = 16
        assert len(df) == 16
        # Initial reads have Time_index == -1
        init = df[df["Time_index"] == -1]
        assert len(init) == 4  # 2 wavelengths x 2 wells
        # First kinetic timepoint has Time_index == 0 and Time_s == 0
        first = df[(df["Time_index"] == 0) & (df["Wavelength"] == 734.0)]
        assert (first["Time_s"] == 0).all()
        # Last kinetic timepoint has Time_s == 158 (0:02:38)
        last = df[(df["Time_index"] == 2) & (df["Wavelength"] == 734.0)]
        assert (last["Time_s"] == 158).all()
    finally:
        os.unlink(path)


def test_add_dA_appends_reference_corrected_signal():
    path = _write_temp(FULL_RUN)
    try:
        pf = b2x.parse_file(path)
        df = b2x.build_data_long(pf, b2x.make_blank_plate_info())
        df_dA = b2x.add_dA(df)
        dA = df_dA[df_dA["Wavelength"].astype(str) == "dA"]
        # A1 @ t_index=0 was A_734=0.400, A_950=0.050 -> dA=0.350
        a1_t0 = dA[(dA["Loc"] == "A1") & (dA["Time_index"] == 0)]
        assert a1_t0["A"].iloc[0] == pytest.approx(0.350)
        # initial reads also get dA: A_734=0.500, A_950=0.080 -> 0.420
        a1_init = dA[(dA["Loc"] == "A1") & (dA["Time_index"] == -1)]
        assert a1_init["A"].iloc[0] == pytest.approx(0.420)
    finally:
        os.unlink(path)


def test_build_wide_labels_columns_with_sample_and_conc():
    path = _write_temp(FULL_RUN)
    try:
        pf = b2x.parse_file(path)
        pi = _plate_info_with_a1_labeled()
        df = b2x.build_data_long(pf, pi)
        wide = b2x.build_wide(df, 734.0, ["A1"], pi)
        cols = list(wide.columns)
        assert "Time_index" in cols and "Time_s" in cols
        # The labeled-well column carries 'Sample | Conc mM'
        labeled = [c for c in cols if "ABTS-Trolox" in c]
        assert labeled, f"expected ABTS-Trolox column in {cols}"
        assert "2.5 mM" in labeled[0]
    finally:
        os.unlink(path)


# --- End-to-end workbook tests ----------------------------------------------


def test_write_workbook_full_run(tmp_path):
    path = _write_temp(FULL_RUN)
    try:
        pf = b2x.parse_file(path)
        out = tmp_path / "out.xlsx"
        b2x.write_workbook(pf, _plate_info_with_a1_labeled(), str(out))
        assert out.exists()
        wb = load_workbook(out)
        # Expected sheets
        assert {"Summary", "PlateInfo", "Layout", "Data_long",
                "Abs_734_wide", "Abs_950_wide", "dA_wide"} <= set(wb.sheetnames)
        # Data_long has more than just the 4 initial rows -> kinetic landed
        ws = wb["Data_long"]
        assert ws.max_row > 5  # 16 data rows + 1 header
    finally:
        os.unlink(path)


def test_write_workbook_spectrum_only(tmp_path):
    path = _write_temp(SPECTRUM_ONLY)
    try:
        pf = b2x.parse_file(path)
        out = tmp_path / "spec.xlsx"
        b2x.write_workbook(pf, b2x.make_blank_plate_info(), str(out))
        wb = load_workbook(out)
        assert "Spectrum" in wb.sheetnames
        assert "Data_long" not in wb.sheetnames  # spectrum-only skips kinetic sheets
        # Spectrum sheet has the one reading
        ws = wb["Spectrum"]
        assert ws.max_row == 2  # header + 1 reading
    finally:
        os.unlink(path)


def test_write_workbook_kinetic_only_with_na_initials(tmp_path):
    """The kinetic data should still land even though initial reads are #N/A."""
    path = _write_temp(KINETIC_ONLY_NA_INIT)
    try:
        pf = b2x.parse_file(path)
        out = tmp_path / "k.xlsx"
        b2x.write_workbook(pf, b2x.make_blank_plate_info(), str(out))
        wb = load_workbook(out)
        ws = wb["Data_long"]
        # 2 timepoints x 1 wavelength x 1 well (A1) = 2 rows; add dA = 4 rows
        assert ws.max_row >= 3
    finally:
        os.unlink(path)


def test_write_workbook_initial_only_does_not_crash(tmp_path):
    """No kinetic blocks at all -- the wide sheets are skipped silently."""
    path = _write_temp(INITIAL_ONLY)
    try:
        pf = b2x.parse_file(path)
        out = tmp_path / "i.xlsx"
        b2x.write_workbook(pf, b2x.make_blank_plate_info(), str(out))
        wb = load_workbook(out)
        # Initial reads alone produce a 1-well-each Data_long
        assert "Data_long" in wb.sheetnames
        assert wb["Data_long"].max_row >= 2
    finally:
        os.unlink(path)


READ_N_DIALECT = """Procedure Summary

Plate Type\t96 WELL PLATE
Read\tRead:  (A) 734, 950
Start Kinetic\tStart Kinetic [Run 0:06:00, Interval 0:00:30]
Read\tRead:  (A) 734, 950
End Kinetic\tEnd Kinetic

Read 1:734
\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12
A\t\t0.957\t\t\t\t\t\t\t\t\t\t\tRead 1:734
B\t\t0.969\t\t\t\t\t\t\t\t\t\t\tRead 1:734

Read 1:950
\t1\t2\t3\t4\t5\t6\t7\t8\t9\t10\t11\t12
A\t\t0.172\t\t\t\t\t\t\t\t\t\t\tRead 1:950
B\t\t0.176\t\t\t\t\t\t\t\t\t\t\tRead 1:950

Read 2:734

Kinetic read\tA1\tA2\tB1\tB2
0:00:00\t\t0.171\t\t0.272
0:00:30\t\t0.236\t\t0.233

Read 2:950

Kinetic read\tA1\tA2\tB1\tB2
0:00:05\t\t0.114\t\t0.114
0:00:35\t\t0.112\t\t0.095
"""


def test_parse_read_n_dialect():
    """2026-09 protocol ('ABTS assay.prt'): initial reads exported as
    'Read 1:<wl>' grids and kinetics as 'Read 2:<wl>' banners. The grid
    sections must land in init_reads, not be swallowed as empty kinetics."""
    path = _write_temp(READ_N_DIALECT)
    try:
        pf = b2x.parse_file(path)
        assert set(pf.init_reads) == {"734", "950"}
        assert pf.init_reads["734"]["A2"] == pytest.approx(0.957)
        assert pf.init_reads["950"]["B2"] == pytest.approx(0.176)
        assert pf.init_reads["734"]["A1"] is None
        assert set(pf.kinetics) == {"734", "950"}
        t734, A734 = pf.kinetics["734"]
        assert t734 == ["0:00:00", "0:00:30"]
        np.testing.assert_allclose(A734[:, b2x.WELLS.index("A2")], [0.171, 0.236])
        np.testing.assert_allclose(A734[:, b2x.WELLS.index("B2")], [0.272, 0.233])
        assert np.isnan(A734[:, 0]).all()
    finally:
        os.unlink(path)
