"""Cross-format tests of the optional ``tiles`` vtype of the io engines."""

import h5py
import numpy as np
import numpy.testing as npt
import pytest

import xdas as xd
from xdas.tiles import TileArray


def ramp(shape, dtype="float32"):
    """Distinct values everywhere, so misplaced reads cannot cancel out."""
    return np.arange(np.prod(shape), dtype=dtype).reshape(shape)


def make_asn_file(path, nt=20, nd=4, t0=0.0):
    data = ramp((nt, nd))
    with h5py.File(path, "w") as file:
        header = file.create_group("header")
        header["time"] = t0
        header["dt"] = 0.1
        header["dx"] = 10.0
        file.create_dataset("data", data=data)
        cable_spec = file.create_group("cableSpec")
        cable_spec["sensorDistances"] = 10.0 * np.arange(nd)
        demod_spec = file.create_group("demodSpec")
        demod_spec["roiStart"] = np.array([0])
        demod_spec["roiEnd"] = np.array([nd])
    return data


def make_febus_file(path, nchunks=3, nt=12, nx=5):
    data = ramp((nchunks, nt, nx))
    times = np.arange(nchunks, dtype=np.float64) * 0.01
    with h5py.File(path, "w") as file:
        source = file.create_group("DeviceName").create_group("Source1")
        source.create_dataset("time", data=times)
        zone = source.create_group("Zone1")
        zone.attrs["BlockRate"] = np.array([100.0])
        zone.attrs["Spacing"] = np.array([5.0, 1.0])
        zone.attrs["Extent"] = np.array([0.0, (nx - 1) * 5.0])
        zone.attrs["Origin"] = np.array([0.0, 0.0])
        zone.create_dataset("StrainRate", data=data)
    return data


def make_terra15_file(path, nt=15, nd=6):
    data = ramp((nt, nd))
    with h5py.File(path, "w") as file:
        product = file.create_group("data_product")
        # small epoch offsets stay exact in float64 down to the nanosecond
        product.create_dataset("gps_time", data=0.001 * np.arange(nt))
        product.create_dataset("data", data=data)
        file.attrs["sensing_range_start"] = 12.0
        file.attrs["dx"] = 2.0
    return data


def make_apsensing_file(path, nt=10, nd=8):
    data = ramp((nt, nd))
    with h5py.File(path, "w") as file:
        file.create_dataset("DAS", data=data)
        meta = file.create_group("Metadata")
        meta.create_dataset("Timestamp", data=np.bytes_(b"2020-01-01T00:00:00.000Z"))
        proc = file.create_group("ProcessingServer")
        proc["DataRate"] = 1000.0
        proc["SpatialSampling"] = 2.0
        file.create_group("DAQ")["PositionStart"] = 0.0
    return data


def make_prodml_file(path, nt=10, nd=5, swapped=False):
    data = ramp((nd, nt) if swapped else (nt, nd))
    with h5py.File(path, "w") as file:
        acquisition = file.create_group("Acquisition")
        acquisition.attrs["SpatialSamplingInterval"] = 2.0
        acquisition.attrs["StartLocusIndex"] = 0
        rawdata = acquisition.create_group("Raw[0]").create_dataset(
            "RawData", data=data
        )
        rawdata.attrs["PartStartTime"] = np.bytes_(b"2020-01-01T00:00:00.000+00:00")
        rawdata.attrs["PartEndTime"] = np.bytes_(b"2020-01-01T00:00:00.900+00:00")
    return data


MAKERS = {
    "asn": (make_asn_file, {}),
    "febus": (make_febus_file, {"overlaps": (1, 1), "offset": 0}),
    "terra15": (make_terra15_file, {}),
    "apsensing": (make_apsensing_file, {}),
    "prodml": (make_prodml_file, {}),
}


@pytest.mark.parametrize("fmt", sorted(MAKERS))
def test_tiles_vtype_matches_hdf5(tmp_path, fmt):
    """The opt-in tiles backing yields the exact same array as the VDS one."""
    maker, kwargs = MAKERS[fmt]
    path = str(tmp_path / f"{fmt}.h5")
    maker(path)
    expected = xd.open_dataarray(path, engine=fmt, **kwargs)
    result = xd.open_dataarray(path, engine=fmt, vtype="tiles", **kwargs)
    assert isinstance(result.data, TileArray)
    assert result.data.engine["name"] == fmt
    assert result.equals(expected)
    sliced = result[3:9:2, 1:3]
    assert isinstance(sliced.data, TileArray)
    npt.assert_array_equal(sliced.values, expected.values[3:9:2, 1:3])


def test_febus_block_crossing_reads(tmp_path):
    """Row ranges spanning block boundaries fuse the right trimmed parts."""
    path = str(tmp_path / "febus.h5")
    make_febus_file(path)
    kwargs = {"overlaps": (1, 1), "offset": 0}
    expected = xd.open_dataarray(path, engine="febus", **kwargs).values
    result = xd.open_dataarray(path, engine="febus", vtype="tiles", **kwargs)
    assert result.data.engine["block_size"] == 12
    assert result.data.engine["overlaps"] == [1, 1]
    npt.assert_array_equal(result[8:12].values, expected[8:12])
    npt.assert_array_equal(result[::3].values, expected[::3])
    npt.assert_array_equal(result[25:].values, expected[25:])


def test_prodml_transpose_param(tmp_path):
    """The shared contract reads distance-major files time-major on request.

    The 0.2 engine never writes ``transpose`` (its manifests keep the
    on-disk layout), but manifests written by the 0.3 line use it.
    """
    path = str(tmp_path / "prodml_swapped.h5")
    data = make_prodml_file(path, swapped=True)
    manifest = TileArray.from_tiles(
        path, data.T.shape, data.dtype, {"name": "prodml", "transpose": True}
    )
    npt.assert_array_equal(np.asarray(manifest), data.T)
    npt.assert_array_equal(np.asarray(manifest[2:7:2, 1:4]), data.T[2:7:2, 1:4])


def test_tiles_view_roundtrip(tmp_path):
    """A tile-backed view persists (spec params included) and reopens lazily."""
    path = str(tmp_path / "febus.h5")
    make_febus_file(path)
    da = xd.open_dataarray(
        path, engine="febus", vtype="tiles", overlaps=(1, 1), offset=0
    )
    out = str(tmp_path / "view.nc")
    da.to_netcdf(out)
    reopened = xd.open_dataarray(out)
    assert isinstance(reopened.data, TileArray)
    assert reopened.data.engine == da.data.engine
    assert reopened.equals(da)


def test_open_mfdataarray_fuses_tiles(tmp_path):
    """Multi-file opening fuses at the manifest level and stays lazy."""
    paths, parts = [], []
    for k in range(2):
        path = str(tmp_path / f"asn{k}.h5")
        parts.append(make_asn_file(path, t0=k * 2.0))
        paths.append(path)
    da = xd.open_mfdataarray(paths, engine="asn", vtype="tiles", parallel=False)
    assert isinstance(da, xd.DataArray)
    assert isinstance(da.data, TileArray)
    assert da.data.ntiles == 2
    npt.assert_array_equal(da.values, np.concatenate(parts))


def test_xdas_engine_tiles_vtype(tmp_path):
    """Materialized native files reopen lazily as tile arrays on request."""
    da = xd.testing.dummy(shape=(10, 5), step=(1.0, 10.0), dtype=np.float32)
    path = str(tmp_path / "da.nc")
    da.to_netcdf(path)
    result = xd.open_dataarray(path, engine="xdas", vtype="tiles")
    assert isinstance(result.data, TileArray)
    assert result.data.engine == {"name": "xdas", "dataset": "/__values__"}
    assert result.equals(da)


def test_tiles_datacollection_roundtrip(tmp_path):
    """Collections of tile-backed arrays reopen as collections, not as errors.

    The manifest lives in a sibling group of the data array's variables, which
    used to make the array look like a nested collection to the reader.
    """
    das = [
        xd.testing.dummy(shape=(10, 5), step=(1.0, 10.0), dtype=np.float32)
        for _ in range(2)
    ]
    paths = []
    for k, da in enumerate(das):
        path = str(tmp_path / f"da{k}.nc")
        da.to_netcdf(path)
        paths.append(path)
    tiled = [xd.open_dataarray(path, engine="xdas", vtype="tiles") for path in paths]

    sequence = xd.DataCollection(tiled, name="acquisition")
    fname = str(tmp_path / "sequence.nc")
    sequence.to_netcdf(fname, virtual=True)
    result = xd.open_datacollection(fname)
    assert len(result) == 2
    for expected, actual in zip(das, result):
        assert isinstance(actual.data, TileArray)
        npt.assert_array_equal(actual.values, expected.values)

    mapping = xd.DataCollection({"a": sequence, "b": sequence}, name="node")
    fname = str(tmp_path / "mapping.nc")
    mapping.to_netcdf(fname, virtual=True)
    result = xd.open_datacollection(fname)
    assert sorted(result) == ["a", "b"]
    assert isinstance(result["a"][1].data, TileArray)
    npt.assert_array_equal(result["b"][0].values, das[0].values)


def test_default_vtypes():
    """The backing each engine picks when the caller does not say.

    Pinned because it is what the I/O guide documents: formats that store
    several blocks per file, or that HDF5 virtual datasets cannot serve at
    all, default to tiles.
    """
    from xdas.io import Engine

    expected = {
        "apsensing": "hdf5",
        "asn": "hdf5",
        "febus": "tiles",
        "miniseed": "tiles",
        "prodml": "hdf5",
        "silixa": "tiles",
        "terra15": "hdf5",
        "xdas": "hdf5",
    }
    assert {name: Engine[name]().vtype for name in expected} == expected
