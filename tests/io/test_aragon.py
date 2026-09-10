"""Tests for the Aragon Photonics HDAS HDF5 engine."""

import h5py
import numpy as np
import numpy.testing as npt
import pytest

import xdas as xd
from xdas.io.aragon import AragonEngine

NT, NX = 30, 8
TRIGGER_HZ = 10000.0
NAVG_DAQ = 1.0
NAVG_STRAIN = 4.0
UNDERSAMPLING = 50.0
X0 = 48.0
DX = 3.0
EPOCH = 1788815410.0
DT_S = NAVG_DAQ * NAVG_STRAIN * UNDERSAMPLING / TRIGGER_HZ  # 0.02 s


def make_header(data_type=0, device_type=1):
    h = np.zeros(200, dtype="f8")
    h[1] = TRIGGER_HZ
    h[44] = DX  # spatial sampling (m)
    h[49] = NAVG_DAQ
    h[72] = X0  # processed-fiber start (m)
    h[76] = NAVG_STRAIN
    h[101] = UNDERSAMPLING
    h[129] = data_type  # 0 = StrainRate
    h[199] = device_type  # 1 = HDAS 3.0
    return h


def make_aragon_file(
    path, data=None, data_type=0, device_type=1, timestamps=None, drop=None
):
    if data is None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((NT, NX)).astype("f4")
    if timestamps is None:
        timestamps = EPOCH + np.arange(NT, dtype="f8") * DT_S
    timestamps = np.asarray(timestamps, dtype="f8")
    groups = {
        "Header/Header_Data": make_header(data_type, device_type),
        "StrainRate/StrainRate_Data": data,
        "Timestamps/Timestamps_Data": timestamps.reshape(-1, 1),
        "Reference/Reference_Data": np.zeros(NX, dtype="f8"),
    }
    with h5py.File(path, "w") as f:
        for name, value in groups.items():
            if name != drop:
                f.create_dataset(name, data=value)
    return data


class TestAragonEngine:
    def test_open_hdf5(self, tmp_path):
        path = tmp_path / "aragon.h5"
        data = make_aragon_file(path)
        da = xd.open(str(path), engine="aragon")
        assert isinstance(da, xd.DataArray)
        assert da.dims == ("time", "distance")
        assert da.shape == (NT, NX)
        assert da.dtype == np.float32
        assert da.name is None
        npt.assert_array_equal(da.values, data)

    def test_open_tiles(self, tmp_path):
        path = tmp_path / "aragon_tiles.h5"
        data = make_aragon_file(path)
        da = xd.open(str(path), engine="aragon", vtype="tiles")
        assert da.shape == (NT, NX)
        assert da.dtype == np.float32
        npt.assert_array_equal(da.values, data)

    def test_time_coordinate(self, tmp_path):
        path = tmp_path / "aragon.h5"
        make_aragon_file(path)
        da = xd.open(str(path), engine="aragon")
        t = da["time"].values
        # float64 seconds near epoch carry ~100 ns of quantisation
        assert abs(t[0] - np.datetime64(round(EPOCH * 1e9), "ns")) < np.timedelta64(
            1, "us"
        )
        npt.assert_array_equal(np.diff(t), np.timedelta64(20_000_000, "ns"))

    def test_distance_coordinate(self, tmp_path):
        path = tmp_path / "aragon.h5"
        make_aragon_file(path)
        da = xd.open(str(path), engine="aragon")
        npt.assert_array_equal(da["distance"].values, X0 + DX * np.arange(NX))

    @pytest.mark.parametrize("ctype", ["interpolated", "sampled", "dense"])
    def test_distance_ctypes(self, tmp_path, ctype):
        path = tmp_path / "aragon.h5"
        make_aragon_file(path)
        da = xd.open(str(path), engine="aragon", ctype={"distance": ctype})
        npt.assert_array_equal(da["distance"].values, X0 + DX * np.arange(NX))

    def test_time_ctype_sampled_rejected(self):
        with pytest.raises(NotImplementedError, match="sampled"):
            AragonEngine(ctype={"time": "sampled"})

    def test_time_axis_from_stamps(self, tmp_path):
        # time comes from Timestamps, not header[69]
        path = tmp_path / "aragon.h5"
        shifted = EPOCH + 0.37 + np.arange(NT, dtype="f8") * DT_S
        make_aragon_file(path, timestamps=shifted)
        da = xd.open(str(path), engine="aragon")
        assert abs(
            da["time"].values[0] - np.datetime64(round((EPOCH + 0.37) * 1e9), "ns")
        ) < np.timedelta64(1, "us")

    def test_time_anchor_minimizes_deviation(self, tmp_path):
        # a lone early first stamp must not drag the whole grid with it: the
        # anchor sits at the Chebyshev centre of the residuals
        path = tmp_path / "aragon.h5"
        ts = EPOCH + np.arange(NT, dtype="f8") * DT_S
        ts[0] -= 4e-3  # 4 ms early outlier
        make_aragon_file(path, timestamps=ts)
        da = xd.open(str(path), engine="aragon")
        stamps = (ts * 1e9).round().astype("int64")
        residual = stamps - 20_000_000 * np.arange(NT)
        spread = int(residual.max() - residual.min())
        assert da["time"].values[0] == np.datetime64(
            int(residual.min()) + spread // 2, "ns"
        )
        assert da["time"].tolerance == np.timedelta64((spread + 1) // 2, "ns")

    def test_dense_time_keeps_jitter(self, tmp_path):
        path = tmp_path / "aragon.h5"
        rng = np.random.default_rng(1)
        jittery = EPOCH + np.arange(NT) * DT_S + rng.uniform(-1e-3, 1e-3, NT)
        make_aragon_file(path, timestamps=jittery)
        da = xd.open(str(path), engine="aragon", ctype={"time": "dense"})
        expected = (jittery * 1e9).round().astype("datetime64[ns]")
        npt.assert_array_equal(da["time"].values, expected)

    def test_interpolated_time_tolerates_jitter(self, tmp_path):
        path = tmp_path / "aragon.h5"
        rng = np.random.default_rng(2)
        jittery = EPOCH + np.arange(NT) * DT_S + rng.uniform(-1e-3, 1e-3, NT)
        make_aragon_file(path, timestamps=jittery)
        da = xd.open(str(path), engine="aragon")
        stamps = (jittery * 1e9).round().astype("int64")
        spread = int(np.ptp(stamps - 20_000_000 * np.arange(NT)))
        npt.assert_array_equal(
            np.diff(da["time"].values), np.timedelta64(20_000_000, "ns")
        )
        assert da["time"].tolerance == np.timedelta64((spread + 1) // 2, "ns")

    def test_gap_warns(self, tmp_path):
        path = tmp_path / "aragon.h5"
        gappy = EPOCH + np.arange(NT, dtype="f8") * DT_S
        gappy[NT // 2 :] += 5 * DT_S  # five dropped samples mid-file
        make_aragon_file(path, timestamps=gappy)
        with pytest.warns(RuntimeWarning, match="gap"):
            xd.open(str(path), engine="aragon")

    def test_missing_group_raises(self, tmp_path):
        path = tmp_path / "notaragon.h5"
        make_aragon_file(path, drop="Timestamps/Timestamps_Data")
        with pytest.raises(NotImplementedError, match="not an HDAS file"):
            AragonEngine().open_dataarray(str(path))

    def test_timestamp_count_mismatch_raises(self, tmp_path):
        path = tmp_path / "aragon.h5"
        make_aragon_file(path, timestamps=EPOCH + np.arange(NT - 1) * DT_S)
        with pytest.raises(ValueError, match="timestamps for"):
            AragonEngine().open_dataarray(str(path))

    def test_not_strainrate_raises(self, tmp_path):
        path = tmp_path / "temp.h5"
        make_aragon_file(path, data_type=1)
        with pytest.raises(NotImplementedError, match="StrainRate"):
            AragonEngine().open_dataarray(str(path))

    def test_not_hdas3_raises(self, tmp_path):
        path = tmp_path / "nothdas.h5"
        make_aragon_file(path, device_type=0)
        with pytest.raises(ValueError, match="not an HDAS 3.0 file"):
            AragonEngine().open_dataarray(str(path))

    def test_autodetect(self, tmp_path):
        path = tmp_path / "aragon.h5"
        make_aragon_file(path)
        da = xd.open(str(path))
        assert da.equals(xd.open(str(path), engine="aragon"))
