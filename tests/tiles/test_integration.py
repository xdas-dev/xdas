"""Tile-backed data inside the 0.2 DataArray and the native file format."""

import dask.array as da_
import numpy as np
import numpy.testing as npt
import pytest

import xdas as xd
from xdas.tiles import TileArray, extract_array

NX = 5

DIMS = ("time", "distance")


def wrap(manifest):
    """Wrap *manifest* in a DataArray with regular time/distance coordinates."""
    nt, nx = manifest.shape
    # ns resolution: the netCDF round trip casts datetimes to M8[ns]
    time = xd.Coordinate["interpolated"].from_block(
        np.datetime64("2020-01-01T00:00:00", "ns"),
        nt,
        np.timedelta64(10_000_000, "ns"),
        dim="time",
    )
    distance = xd.Coordinate["interpolated"].from_block(0.0, nx, 4.0, dim="distance")
    return xd.DataArray(manifest, {"time": time, "distance": distance})


class TestDataArray:
    def test_data_and_extract(self, stack):
        manifest, _ = stack
        da = wrap(manifest)
        assert da.data is manifest
        assert extract_array(da) is manifest
        assert "TileArray" in repr(da)

    def test_extract_rejections(self, stack):
        manifest, _ = stack
        with pytest.raises(TypeError, match="in-memory numpy array"):
            extract_array(wrap(manifest).load())
        with pytest.raises(TypeError, match="not backed by"):
            extract_array("something else")

    def test_isel_stays_virtual(self, stack, engine_calls):
        manifest, reference = stack
        da = wrap(manifest)
        view = da.isel(time=slice(9, 13), distance=slice(1, 4))
        assert isinstance(view.data, TileArray)
        assert engine_calls == []
        npt.assert_array_equal(view.values, reference[9:13, 1:4])

    def test_sel_stays_virtual(self, stack, engine_calls):
        manifest, reference = stack
        da = wrap(manifest)
        t0 = da["time"][2].values
        t1 = da["time"][20].values
        view = da.sel(time=slice(t0, t1))
        assert isinstance(view.data, TileArray)
        assert engine_calls == []
        npt.assert_array_equal(view.values, reference[2:21])

    def test_load_materializes(self, stack):
        manifest, reference = stack
        loaded = wrap(manifest).load()
        assert isinstance(loaded.data, np.ndarray)
        npt.assert_array_equal(loaded.values, reference)

    def test_concat_along_existing_dim_stays_virtual(self, stack, engine_calls):
        manifest, reference = stack
        da = wrap(manifest)
        head = da.isel(time=slice(0, 10))
        tail = da.isel(time=slice(10, None))
        out = xd.concat([head, tail], "time")
        assert isinstance(out.data, TileArray)
        assert engine_calls == []
        npt.assert_array_equal(out.values, reference)

    def test_concat_along_new_dim_stays_virtual(self, stack, engine_calls):
        manifest, reference = stack
        objs = [wrap(manifest), wrap(manifest)]
        out = xd.concat(objs, "station")
        assert out.dims == ("station", "time", "distance")
        assert isinstance(out.data, TileArray)
        assert engine_calls == []
        npt.assert_array_equal(out.values, np.stack([reference, reference]))

    def test_mean_streams(self, stack, engine_calls):
        manifest, reference = stack
        da = wrap(manifest)
        npt.assert_allclose(da.mean("time").values, reference.mean(0))
        assert len(engine_calls) > 0
        assert manifest._cache is None


class TestPersistence:
    def test_round_trip(self, stack, tmp_path):
        manifest, reference = stack
        da = wrap(manifest)
        path = str(tmp_path / "view.nc")
        da.to_netcdf(path)
        reopened = xd.open_dataarray(path)
        assert isinstance(reopened.data, TileArray)
        assert reopened.data.equals(manifest)
        assert reopened.coords["time"].equals(da.coords["time"])
        npt.assert_array_equal(reopened.values, reference)

    def test_sliced_view_round_trip(self, stack, tmp_path):
        manifest, reference = stack
        view = wrap(manifest).isel(time=slice(9, 13))
        path = str(tmp_path / "sliced.nc")
        view.to_netcdf(path)
        reopened = xd.open_dataarray(path)
        assert isinstance(reopened.data, TileArray)
        npt.assert_array_equal(reopened.values, reference[9:13])

    def test_grouped_round_trip(self, stack, tmp_path):
        manifest, reference = stack
        da = wrap(manifest)
        path = str(tmp_path / "grouped.nc")
        da.to_netcdf(path, group="acquisition")
        reopened = xd.open_dataarray(path, engine="xdas", group="acquisition")
        assert isinstance(reopened.data, TileArray)
        npt.assert_array_equal(reopened.values, reference)

    def test_eager_save_writes_values(self, stack, tmp_path):
        manifest, reference = stack
        da = wrap(manifest)
        path = str(tmp_path / "eager.nc")
        da.to_netcdf(path, virtual=False)
        reopened = xd.open_dataarray(path)
        assert not isinstance(reopened.data, TileArray)
        npt.assert_array_equal(reopened.values, reference)

    def test_dask_write_deprecated(self, tmp_path):
        import dask

        data = da_.from_delayed(dask.delayed(np.zeros)((4, NX)), (4, NX), np.float64)
        da = xd.DataArray(data, dims=DIMS)
        path = str(tmp_path / "dask.nc")
        with pytest.warns(FutureWarning, match="dask-backed"):
            da.to_netcdf(path, virtual=True)
        reopened = xd.open_dataarray(path)
        npt.assert_array_equal(reopened.values, np.zeros((4, NX)))
