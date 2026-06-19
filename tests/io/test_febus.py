"""Tests for the Febus HDF5 engine."""

import h5py
import numpy as np
import pytest

import xdas as xd
from xdas.io.febus import FebusEngine


def make_febus_file(path, use_freqres=False):
    """Create a minimal valid febus HDF5 file.

    Spacing layout (from febus engine): delta = (Spacing[1]/1000, Spacing[0])
    so Spacing[0] = distance step (m), Spacing[1] = time scale (dt*1000).
    """
    nchunks, nt, nx = 3, 12, 5
    dx_m = 5.0
    dt_s = 0.001  # 1 ms per sample → Spacing[1] = 1.0
    block_rate_hz = 100.0  # 100 Hz → 10ms per block → nt_real=10, noverlap=2
    data = np.zeros((nchunks, nt, nx), dtype=np.float32)
    # chunk starts spaced by 1 real block = 10ms = 0.01s
    times = np.arange(nchunks, dtype=np.float64) * 0.01

    with h5py.File(path, "w") as f:
        device = f.create_group("DeviceName")
        source = device.create_group("Source1")
        source.create_dataset("time", data=times)
        zone = source.create_group("Zone1")
        if use_freqres:
            zone.attrs["FreqRes"] = np.array([block_rate_hz])
        else:
            zone.attrs["BlockRate"] = np.array([block_rate_hz])
        # Spacing[0]=dx_m, Spacing[1]=dt_s*1000 → delta=(dt_s, dx_m)
        zone.attrs["Spacing"] = np.array([dx_m, dt_s * 1000.0])
        zone.attrs["Extent"] = np.array([0.0, (nx - 1) * dx_m])
        zone.attrs["Origin"] = np.array([0.0, 0.0])
        zone.create_dataset("StrainRate", data=data)


class TestFebusEngine:
    def test_open_with_freqres_attr(self, tmp_path):
        path = tmp_path / "febus_freqres.h5"
        make_febus_file(path, use_freqres=True)
        da = xd.open(str(path), engine="febus", overlaps=(1, 1), offset=0)
        assert isinstance(da, xd.DataArray)

    def test_invalid_overlaps_raises(self, tmp_path):
        path = tmp_path / "febus.h5"
        make_febus_file(path)
        with pytest.raises(ValueError, match="overlaps must be"):
            FebusEngine().open_dataarray(str(path), overlaps="bad")

    def test_invalid_offset_raises(self, tmp_path):
        path = tmp_path / "febus.h5"
        make_febus_file(path)
        with pytest.raises(ValueError, match="offset must be an integer"):
            FebusEngine().open_dataarray(str(path), overlaps=(1, 1), offset="bad")

    def test_missing_block_rate_raises(self, tmp_path):
        path = tmp_path / "febus_no_blockrate.h5"
        nchunks, nt, nx = 2, 10, 5
        with h5py.File(path, "w") as f:
            device = f.create_group("Dev")
            source = device.create_group("Source1")
            source.create_dataset("time", data=np.zeros(nchunks))
            zone = source.create_group("Zone1")
            zone.attrs["Spacing"] = np.array([1.0, 5.0])
            zone.attrs["Extent"] = np.array([0.0, 20.0])
            zone.attrs["Origin"] = np.array([0.0, 0.0])
            zone.create_dataset("Data", data=np.zeros((nchunks, nt, nx)))
        with pytest.raises(KeyError, match="Could not find the block size"):
            FebusEngine().open_dataarray(str(path), overlaps=(0, 0), offset=0)
