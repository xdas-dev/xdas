"""Tests for the Aragon Photonics HDAS HDF5 engine."""

import warnings

import h5py
import numpy as np
import numpy.testing as npt
import pytest

import xdas as xd
from xdas.coordinates import InterpCoordinate
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


DT_NS = 20_000_000
MTB = 10  # samples between two sync stamps
EPS_NS = 2 * np.spacing(EPOCH) * 1e9  # two float64 spacings of the stamps


def hdas_like_stamps(nt, mtb=MTB, late=(), seed=0):
    """NTP-locked HDAS stamps: a sync every *mtb* samples with ms jitter,
    interpolated in between, except before each sync in *late*, where the block
    keeps stepping at the header dt and the stamps jump at the sync."""
    rng = np.random.default_rng(seed)
    nsync = nt // mtb + 1
    syncs = EPOCH + np.arange(nsync) * mtb * DT_S + rng.uniform(-1e-3, 1e-3, nsync)
    stamps = np.empty(nsync * mtb)
    for k in range(nsync - 1):
        block = slice(k * mtb, (k + 1) * mtb)
        if k + 1 in late:
            syncs[k + 1] = syncs[k] + mtb * DT_S + 3e-3
            stamps[block] = syncs[k] + np.arange(mtb) * DT_S
        else:
            stamps[block] = np.linspace(syncs[k], syncs[k + 1], mtb + 1)[:-1]
    return stamps[:nt]


def as_ns(stamps):
    return (stamps * 1e9).round().astype("int64")


def assert_within_eps(time, stamps):
    deviation = np.abs(time.values.astype("int64") - as_ns(stamps))
    assert deviation.max() <= EPS_NS


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

    def test_time_axis_from_stamps(self, tmp_path):
        # time comes from Timestamps, not header[69]
        path = tmp_path / "aragon.h5"
        shifted = EPOCH + 0.37 + np.arange(NT, dtype="f8") * DT_S
        make_aragon_file(path, timestamps=shifted)
        da = xd.open(str(path), engine="aragon")
        assert abs(
            da["time"].values[0] - np.datetime64(round((EPOCH + 0.37) * 1e9), "ns")
        ) < np.timedelta64(1, "us")

    def test_dense_time_keeps_jitter(self, tmp_path):
        path = tmp_path / "aragon.h5"
        rng = np.random.default_rng(1)
        jittery = EPOCH + np.arange(NT) * DT_S + rng.uniform(-1e-3, 1e-3, NT)
        make_aragon_file(path, timestamps=jittery)
        da = xd.open(str(path), engine="aragon", ctype={"time": "dense"})
        expected = (jittery * 1e9).round().astype("datetime64[ns]")
        npt.assert_array_equal(da["time"].values, expected)

    def test_locked_stamps_interpolated(self, tmp_path):
        path = tmp_path / "aragon.h5"
        nt, late = 20 * MTB, (4, 11)
        stamps = hdas_like_stamps(nt, late=late)
        make_aragon_file(path, data=np.zeros((nt, NX), "f4"), timestamps=stamps)
        time = xd.open(str(path), engine="aragon")["time"]
        assert isinstance(time, InterpCoordinate)
        assert_within_eps(time, stamps)
        assert len(time.tie_indices) <= nt // MTB + 1 + len(late)
        for k in late:
            assert k * MTB - 1 in time.tie_indices  # the pre-jump stamp

    def test_sampling_interval_and_minimal_tolerance(self, tmp_path):
        path = tmp_path / "aragon.h5"
        nt = 20 * MTB
        stamps = hdas_like_stamps(nt, late=(7,))
        make_aragon_file(path, data=np.zeros((nt, NX), "f4"), timestamps=stamps)
        time = xd.open(str(path), engine="aragon")["time"]
        assert time.get_sampling_interval() == DT_S
        assert time.tolerance > np.timedelta64(0, "ns")
        data = {
            "tie_indices": time.tie_indices,
            "tie_values": time.tie_values,
            "sampling_interval": time.sampling_interval,
        }
        InterpCoordinate({**data, "tolerance": time.tolerance}, dim="time")
        with pytest.raises(ValueError, match="not consistent"):
            InterpCoordinate(
                {**data, "tolerance": time.tolerance - np.timedelta64(1, "ns")},
                dim="time",
            )

    def test_regular_stamps_two_ties(self, tmp_path):
        path = tmp_path / "aragon.h5"
        make_aragon_file(path)
        time = xd.open(str(path), engine="aragon")["time"]
        npt.assert_array_equal(time.tie_indices, [0, NT - 1])
        assert time.get_sampling_interval() == DT_S

    def test_free_running_stamps(self, tmp_path):
        # not locked: one stamp per sample, sawtooth drift and us noise
        path = tmp_path / "aragon.h5"
        nt = 300
        rng = np.random.default_rng(3)
        sawtooth = 2e-4 * ((np.arange(nt) % 70) / 70)
        stamps = EPOCH + np.arange(nt) * DT_S + sawtooth + rng.uniform(-1e-6, 1e-6, nt)
        make_aragon_file(path, data=np.zeros((nt, NX), "f4"), timestamps=stamps)
        time = xd.open(str(path), engine="aragon")["time"]
        assert_within_eps(time, stamps)
        assert time.get_sampling_interval() == DT_S

    def test_sampled_time(self, tmp_path):
        path = tmp_path / "aragon.h5"
        nt = 20 * MTB
        stamps = hdas_like_stamps(nt, late=(4,))
        make_aragon_file(path, data=np.zeros((nt, NX), "f4"), timestamps=stamps)
        ties = xd.open(str(path), engine="aragon")["time"]
        time = xd.open(str(path), engine="aragon", ctype={"time": "sampled"})["time"]
        npt.assert_array_equal(time.tie_indices, ties.tie_indices[:-1])
        npt.assert_array_equal(time.tie_values, ties.tie_values[:-1])
        assert time.tie_lengths.sum() == nt
        assert time.sampling_interval == np.timedelta64(DT_NS, "ns")

    @pytest.mark.parametrize("ctype", ["interpolated", "sampled", "dense"])
    def test_no_warning(self, tmp_path, ctype):
        path = tmp_path / "aragon.h5"
        nt = 20 * MTB
        stamps = hdas_like_stamps(nt, late=(3, 9))
        make_aragon_file(path, data=np.zeros((nt, NX), "f4"), timestamps=stamps)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            xd.open(str(path), engine="aragon", ctype={"time": ctype})

    def test_gap_sides_are_ties(self, tmp_path):
        path = tmp_path / "aragon.h5"
        stamps = EPOCH + np.arange(NT, dtype="f8") * DT_S
        stamps[NT // 2 :] += 3 * DT_S  # three dropped samples mid-file
        make_aragon_file(path, timestamps=stamps)
        time = xd.open(str(path), engine="aragon")["time"]
        assert {NT // 2 - 1, NT // 2} <= set(time.tie_indices)
        assert_within_eps(time, stamps)

    @pytest.mark.parametrize("nt", [1, 2])
    @pytest.mark.parametrize("ctype", ["interpolated", "sampled", "dense"])
    def test_tiny_files(self, tmp_path, nt, ctype):
        path = tmp_path / "aragon.h5"
        stamps = EPOCH + np.arange(nt) * DT_S
        make_aragon_file(path, data=np.zeros((nt, NX), "f4"), timestamps=stamps)
        time = xd.open(str(path), engine="aragon", ctype={"time": ctype})["time"]
        npt.assert_array_equal(time.values.astype("int64"), as_ns(stamps))
        if ctype == "interpolated":
            assert time.tolerance == np.timedelta64(0, "ns")
            assert time.get_sampling_interval() == DT_S

    @pytest.mark.parametrize("ctype", ["interpolated", "sampled", "dense"])
    def test_tiles_time_matches_hdf5(self, tmp_path, ctype):
        path = tmp_path / "aragon.h5"
        nt = 20 * MTB
        stamps = hdas_like_stamps(nt, late=(5,))
        make_aragon_file(path, data=np.zeros((nt, NX), "f4"), timestamps=stamps)
        ctype = {"time": ctype}
        da = xd.open(str(path), engine="aragon", ctype=ctype)
        tiles = xd.open(str(path), engine="aragon", vtype="tiles", ctype=ctype)
        assert tiles["time"].equals(da["time"])

    def make_consecutive_files(self, tmp_path, nfile=3, nt=20 * MTB):
        stamps = hdas_like_stamps(nfile * nt, late=(6, 27, 44))
        paths = []
        for k in range(nfile):
            path = tmp_path / f"aragon_{k}.h5"
            chunk = stamps[k * nt : (k + 1) * nt]
            make_aragon_file(path, data=np.zeros((nt, NX), "f4"), timestamps=chunk)
            paths.append(str(path))
        return paths, stamps

    @pytest.mark.parametrize("ctype", ["interpolated", "sampled", "dense"])
    def test_consecutive_files_exact(self, tmp_path, ctype):
        paths, stamps = self.make_consecutive_files(tmp_path)
        da = xd.open(paths, engine="aragon", ctype={"time": ctype}, tolerance=0)
        assert da.sizes["time"] == stamps.size
        if ctype == "sampled":
            parts = [
                xd.open(path, engine="aragon", ctype={"time": ctype})["time"].values
                for path in paths
            ]
            npt.assert_array_equal(da["time"].values, np.concatenate(parts))
        else:
            assert_within_eps(da["time"], stamps)
        if ctype == "interpolated":
            assert da["time"].get_sampling_interval() == DT_S

    def test_consecutive_files_default_tolerance(self, tmp_path):
        paths, stamps = self.make_consecutive_files(tmp_path)
        tolerance = max(
            xd.open(path, engine="aragon")["time"].tolerance for path in paths
        )
        time = xd.open(paths, engine="aragon")["time"]
        assert time.get_sampling_interval() == DT_S
        assert time.tolerance == tolerance
        deviation = np.abs(time.values.astype("int64") - as_ns(stamps))
        assert deviation.max() <= tolerance.astype("i8") + EPS_NS

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
