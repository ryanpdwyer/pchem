"""Tests for the OMNIC .SPA reader used by the Combine Raman / IR tool."""
import io
import os
import struct
import sys
import types
import glob

import numpy as np
import pytest

# util.py imports streamlit and sigfig at module level; stub them so the reader
# can be tested without a Streamlit install.
sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))
_sig = types.ModuleType("sigfig"); _sig.round = round
sys.modules.setdefault("sigfig", _sig)

from pchemapps.util import read_spa, process_raman  # noqa: E402
from pchemapps.combineRaman import combine_spectra, standardize  # noqa: E402

SAMPLE_DIR = os.path.expanduser("~/Dropbox/__mu/pchem1-2020/Alaina Major Project")


class FakeUpload(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name


def make_spa(y, firstx=4000.0, lastx=400.0, title="synthetic", xunits=1, yunits=17):
    """Build a minimal OMNIC-style .SPA byte blob."""
    y = np.asarray(y, dtype="<f4")
    hdr_pos, data_pos = 560, 1024
    buf = bytearray(data_pos + y.nbytes)
    buf[0:20] = b"Spectral Data File\r\n"
    buf[30:30 + len(title)] = title.encode()
    struct.pack_into("<H", buf, 294, 2)
    # directory entries at 304: header (key 2) then data (key 3), then terminator
    struct.pack_into("<HII", buf, 304, 2, hdr_pos, 32)
    struct.pack_into("<HII", buf, 320, 3, data_pos, y.nbytes)
    struct.pack_into("<i", buf, hdr_pos + 4, len(y))
    struct.pack_into("<i", buf, hdr_pos + 8, xunits)
    struct.pack_into("<i", buf, hdr_pos + 12, yunits)
    struct.pack_into("<ff", buf, hdr_pos + 16, firstx, lastx)
    buf[data_pos:data_pos + y.nbytes] = y.tobytes()
    return bytes(buf)


def test_read_spa_synthetic():
    y = np.linspace(0, 1, 11)
    df, header = read_spa(make_spa(y))
    assert list(df.columns) == ["Wavenumber (cm-1)", "Absorbance"]
    assert header["Title"] == "synthetic"
    assert header["Points"] == 11
    # x is returned ascending, y reversed to match
    assert df.iloc[0, 0] == pytest.approx(400.0)
    assert df.iloc[-1, 0] == pytest.approx(4000.0)
    np.testing.assert_allclose(df.iloc[:, 1].values, y[::-1], rtol=1e-6)


def test_read_spa_rejects_non_spa():
    with pytest.raises(ValueError):
        read_spa(b"not a spectral data file")


@pytest.mark.parametrize("code, expected", [
    (11, "Reflectance (%)"),
    (12, "Log(1/R)"),
    (20, "Kubelka-Munk"),
])
def test_read_spa_y_unit_codes(code, expected):
    df, _ = read_spa(make_spa([1, 2, 3], yunits=code))
    assert df.columns[1] == expected


def test_read_spa_rejects_truncated_payload():
    raw = make_spa([1, 2, 3])[:-1]
    with pytest.raises(ValueError, match="truncated intensity block"):
        read_spa(raw)


def test_process_raman_dispatch():
    spa = process_raman(FakeUpload(make_spa([1, 2, 3], yunits=16), "a.SPA"))
    assert spa.kind == "OMNIC IR"
    assert list(spa.df.columns) == ["Wavenumber (cm-1)", "Transmittance (%)"]

    omnic_csv = process_raman(FakeUpload(b"4.0e2,1.0\n4.1e2,2.0\n", "b.csv"))
    assert omnic_csv.kind == "OMNIC IR"
    assert omnic_csv.df.shape == (2, 2)

    with pytest.raises(NotImplementedError):
        process_raman(FakeUpload(b"", "c.xlsx"))


def test_spa_and_omnic_csv_use_positional_columns_and_export_tolerance():
    spa = process_raman(FakeUpload(make_spa([1, 2, 3]), "same.spa"))
    # OMNIC text exports round the x axis slightly compared with SPA float endpoints.
    csv = process_raman(FakeUpload(b"400.0005,3\n2200.0005,2\n4000.0005,1\n", "same.csv"))

    data = [standardize(item.df, 0, 1, "x", "y") for item in (spa, csv)]
    combined = combine_spectra(data, ["SPA", "CSV"], "x", "y")

    assert combined.shape == (3, 3)
    assert list(combined.columns) == ["x", "SPA", "CSV"]


@pytest.mark.skipif(not os.path.isdir(SAMPLE_DIR), reason="sample OMNIC files not available")
def test_spa_matches_omnic_csv_export():
    pairs = [(p, p[:-4] + ".CSV") for p in glob.glob(os.path.join(SAMPLE_DIR, "*.SPA"))]
    pairs = [(s, c) for s, c in pairs if os.path.exists(c)]
    assert pairs
    for spa_path, csv_path in pairs:
        spa = process_raman(FakeUpload(open(spa_path, "rb").read(), os.path.basename(spa_path)))
        ref = process_raman(FakeUpload(open(csv_path, "rb").read(), os.path.basename(csv_path)))
        assert spa.df.shape == ref.df.shape
        np.testing.assert_allclose(spa.df.iloc[:, 0].values, ref.df.iloc[:, 0].values, atol=1e-2)
        np.testing.assert_allclose(spa.df.iloc[:, 1].values, ref.df.iloc[:, 1].values, atol=1e-3)
