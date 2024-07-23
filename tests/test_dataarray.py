import os
from tempfile import TemporaryDirectory

import hdf5plugin
import numpy as np
import pytest

import xdas
from xdas.core.coordinates import Coordinates, DenseCoordinate, InterpCoordinate
from xdas.core.dataarray import DataArray
from xdas.synthetics import wavelet_wavefronts


class TestDataArray:
    def generate(self, dense=False):
        if dense:
            coords = {"dim": 100.0 * (1 + np.arange(9))}
        else:
            coords = {"dim": {"tie_indices": [0, 8], "tie_values": [100.0, 900.0]}}
        data = 0.1 * np.arange(9)
        da = xdas.DataArray(data, coords)
        return da

    def test_init_without_coords(self):
        data = np.arange(2 * 3 * 5).reshape(2, 3, 5)
        da = xdas.DataArray(data)
        assert np.array_equal(da.data, data)
        assert da.dims == ("dim_0", "dim_1", "dim_2")
        assert da.coords == {}
        assert da.isel(dim_0=0).equals(da[0])
        assert da.isel(dim_1=slice(1)).equals(da[:, :1])
        assert da.isel(dim_2=[1, 2, 3]).equals(da[:, :, [1, 2, 3]])
        assert da.sizes == {"dim_0": 2, "dim_1": 3, "dim_2": 5}
        with pytest.raises(KeyError, match="no coordinate"):
            da["dim_0"]
        with pytest.raises(KeyError, match="no coordinate"):
            da.sel(dim_0=0)

    def test_init_and_properties(self):
        da = self.generate()
        assert isinstance(da["dim"], InterpCoordinate)
        assert da.dims == ("dim",)
        assert da.ndim == 1
        assert da.shape == (9,)
        assert da.sizes == {"dim": 9}
        assert da.sizes["first"] == 9
        assert da.sizes["last"] == 9
        assert np.allclose(da.data, 0.1 * np.arange(9))
        assert np.all(np.equal(da.values, da.data))
        assert da.get_axis_num("dim") == 0
        assert da.dtype == np.float64
        da = self.generate(dense=True)
        assert isinstance(da["dim"], DenseCoordinate)
        da = DataArray()
        assert np.array_equal(da.values, np.array(np.nan), equal_nan=True)
        assert da.coords == {}
        assert da.dims == tuple()
        da = DataArray([[]])
        assert da.dims == ("dim_0", "dim_1")
        assert da.ndim == 2
        da = DataArray(1)
        assert da.dims == tuple()
        assert da.ndim == 0

    def test_raises_on_data_and_coords_mismatch(self):
        with pytest.raises(ValueError, match="different number of dimensions"):
            DataArray(np.zeros(3), dims=("time", "distance"))
        with pytest.raises(ValueError, match="infered dimension number from `coords`"):
            DataArray(np.zeros(3), coords={"time": [1], "distance": [1]})
        with pytest.raises(ValueError, match="conflicting sizes for dimension"):
            DataArray(np.zeros((2, 3)), coords={"time": [1, 2], "distance": [1, 2]})

    def test_coords_setter(self):
        da = xdas.DataArray(np.arange(3 * 11).reshape(3, 11))
        da["dim_0"] = [1, 2, 4]
        da["dim_1"] = {"tie_indices": [0, 10], "tie_values": [0.0, 100.0]}
        da["dim_0"] = [1, 2, 3]
        da["metadata"] = 0
        da["non-dimensional"] = ("dim_0", [-1, -1, -1])
        assert da.dims == ("dim_0", "dim_1")
        assert list(da.coords) == ["dim_0", "dim_1", "metadata", "non-dimensional"]
        with pytest.raises(KeyError, match="cannot add new dimension"):
            da["dim_2"] = [1, 2, 3]
        with pytest.raises(ValueError, match="conflicting sizes"):
            da["dim_0"] = [1, 2, 3, 4]
        with pytest.raises(ValueError, match="conflicting sizes"):
            da["dim_1"] = [1]
        coords = da.coords.copy()
        assert coords._parent is None
        da.coords = coords
        assert da.coords._parent is da
        coords = da.coords.copy()
        del coords["dim_1"]
        da.coords = coords
        assert list(da.coords.keys()) == ["dim_0", "metadata", "non-dimensional"]
        assert da.dims == ("dim_0", "dim_1")
        coords = da.coords.copy()
        coords = coords.drop_dims("dim_0")
        with pytest.raises(ValueError, match="replacement coords must have the same"):
            da.coords = coords

    def test_cannot_set_dims(self):
        da = self.generate()
        with pytest.raises(AttributeError):
            da.dims = ("other_dim",)

    def test_getitem(self):
        da = self.generate()
        # assert da[0].data == 0.0
        # assert da[1].data == 0.1
        # assert da[0]["dim"] == 100.0
        sel = da[2:4]
        assert np.allclose(sel.data, [0.2, 0.3])
        assert np.allclose(sel["dim"].tie_indices, [0, 1])
        assert np.allclose(sel["dim"].tie_values, [300.0, 400.0])
        da = self.generate(dense=True)
        assert np.allclose(sel.data, [0.2, 0.3])
        assert np.allclose(sel["dim"].values, [300.0, 400.0])

    def test_setitem(self):
        # da = self.generate()
        # da[0] = -100.0
        # assert da[0].data == -100.0
        ...

    def test_data_setter(self):
        da = wavelet_wavefronts()
        data = np.arange(np.prod(da.shape)).reshape(da.shape)
        da.data = data
        assert np.array_equal(da.data, data)
        with pytest.raises(ValueError, match="replacement data must match"):
            da.data = [1, 2, 3]

    def test_sel(self):
        # interp
        da = self.generate()
        da.sel(dim=slice(2, 4))
        assert da.sel(dim=225, method="nearest").values == 0.1
        assert da.sel(dim=225, method="ffill").values == 0.1
        assert da.sel(dim=225, method="bfill").values == 0.2
        with pytest.raises(KeyError):
            da.sel(dim=225, method=None)
        assert da.sel(dim=slice(100.0, 300.0)).equals(da[0:3])
        assert da.sel(dim=slice(100.0, 300.0), endpoint=False).equals(da[0:2])
        # dense
        da = self.generate(dense=True)
        da.sel(dim=slice(2, 4))
        assert da.sel(dim=225, method="nearest").values == 0.1
        assert da.sel(dim=225, method="ffill").values == 0.1
        assert da.sel(dim=225, method="bfill").values == 0.2
        with pytest.raises(KeyError):
            da.sel(dim=225, method=None)
        assert da.sel(dim=slice(100.0, 300.0)).equals(da[0:3])
        assert da.sel(dim=slice(100.0, 300.0), endpoint=False).equals(da[0:2])
        # drop
        da = wavelet_wavefronts()
        result = da.sel(distance=0, method="nearest", drop=True)
        assert "distance" not in result.coords

    def test_isel(self):
        da = wavelet_wavefronts()
        result = da.isel(first=0)
        excepted = da.isel(time=0)
        assert result.equals(excepted)
        result = da.isel(last=0)
        excepted = da.isel(distance=0)
        assert result.equals(excepted)
        # drop
        da = wavelet_wavefronts()
        result = da.sel(distance=0, drop=True)
        assert "distance" not in result.coords

    def test_to_xarray(self):
        for dense in [True, False]:
            da = self.generate(dense=dense)
            result = da.to_xarray()
            assert np.array_equal(result.values, da.values)
            assert np.array_equal(result["dim"].values, da["dim"].values)
            da = da.sel(dim=slice(1000, 2000))  # empty dataarray
            result = da.to_xarray()
            assert np.array_equal(result.values, da.values)
            assert np.array_equal(result["dim"].values, da["dim"].values)

    def test_from_xarray(self):
        da = self.generate()
        da = da.to_xarray()
        result = DataArray.from_xarray(da)
        assert np.array_equal(result.values, da.values)
        assert np.array_equal(result["dim"].values, da["dim"].values)

    def test_stream(self):
        da = wavelet_wavefronts()
        st = da.to_stream(dim={"distance": "time"})
        assert st[0].id == "NET.DAS00001.00.BN1"
        assert len(st) == da.sizes["distance"]
        assert st[0].stats.npts == da.sizes["time"]
        assert np.datetime64(st[0].stats.starttime.datetime) == da["time"][0].values
        assert np.datetime64(st[0].stats.endtime.datetime) == da["time"][-1].values
        result = DataArray.from_stream(st)
        assert np.array_equal(result.values.T, da.values)
        assert result.sizes == {
            "channel": da.sizes["distance"],
            "time": da.sizes["time"],
        }
        assert result["time"].equals(da["time"])

    def test_dense_str(self):
        coord = [f"D{k}" for k in range(9)]
        coords = Coordinates({"dim": coord})
        data = 0.1 * np.arange(9)
        DataArray(data, coords)

    def test_single_index_selection(self):
        da = DataArray(
            np.arange(12).reshape(3, 4),
            {
                "time": {"tie_values": [0.0, 1.0], "tie_indices": [0, 2]},
                "distance": [0.0, 10.0, 20.0, 30.0],
            },
        )
        da_getitem = da[1]
        da_isel = da.isel(time=1)
        da_sel = da.sel(time=0.5)
        da_expected = DataArray(
            np.array([4, 5, 6, 7]),
            {"time": (None, 0.5), "distance": [0.0, 10.0, 20.0, 30.0]},
        )
        assert da_getitem.equals(da_expected)
        assert da_isel.equals(da_expected)
        assert da_sel.equals(da_expected)
        da_getitem = da[:, 1]
        da_isel = da.isel(distance=1)
        da_sel = da.sel(distance=10.0)
        da_expected = DataArray(
            np.array([1, 5, 9]),
            {
                "time": {"tie_values": [0.0, 1.0], "tie_indices": [0, 2]},
                "distance": (None, 10.0),
            },
        )
        assert da_getitem.equals(da_expected)
        assert da_isel.equals(da_expected)
        assert da_sel.equals(da_expected)

    def test_assign_coords(self):
        da = DataArray(
            data=np.zeros(3),
            coords={"time": np.array([3, 4, 5])},
        )
        result = da.assign_coords(time=[0, 1, 2])
        assert np.array_equal(result["time"].values, [0, 1, 2])
        assert result.equals(da.assign_coords({"time": [0, 1, 2]}))
        result = da.assign_coords(relative_time=("time", [0, 1, 2]))
        assert np.array_equal(result["relative_time"].values, [0, 1, 2])

    def test_swap_dims(self):
        da = DataArray(
            data=[0, 1],
            coords={"x": ["a", "b"], "y": ("x", [0, 1])},
        )
        result = da.swap_dims({"x": "y"})
        assert result.dims == ("y",)
        assert result["x"].dim == "y"
        assert result["y"].dim == "y"
        assert da.swap_dims({"x": "y"}).equals(result)
        result = da.swap_dims(x="z")
        assert result.dims == ("z",)
        assert result["x"].dim == "z"
        assert result["y"].dim == "z"
        with pytest.raises(KeyError, match="not found in current object with dims"):
            da.swap_dims({"z": "x"})

    def test_to_xarray_non_dimensional(self):
        da = DataArray(
            data=np.zeros(3),
            coords={
                "time": np.array([3, 4, 5]),
                "relative_time": ("time", np.array([0, 1, 2])),
                "channel": (None, "DAS000001"),
            },
        )
        result = da.to_xarray()
        assert np.array_equal(result["time"], [3, 4, 5])
        assert np.array_equal(result["relative_time"], [0, 1, 2])
        assert result.dims == da.dims

    def test_netcdf_non_dimensional(self):
        da = DataArray(
            data=np.zeros(3),
            coords={
                "time": np.array([3, 4, 5]),
                "relative_time": ("time", np.array([0, 1, 2])),
            },
        )
        with TemporaryDirectory() as dirpath:
            path = os.path.join(dirpath, "tmp.nc")
            da.to_netcdf(path)
            result = xdas.open_dataarray(path)
            assert result.equals(da)
        with TemporaryDirectory() as dirpath:
            path = os.path.join(dirpath, "da.nc")
            da = wavelet_wavefronts().assign_coords(lon=("distance", np.arange(401)))
            da.to_netcdf(path)
            tmp = xdas.open_dataarray(path)
            path = path = os.path.join(dirpath, "vds.nc")
            tmp.to_netcdf(path)
            result = xdas.open_dataarray(path)
            assert result.equals(da)

    def test_transpose(self):
        da = wavelet_wavefronts()
        result = da.transpose("distance", "time")
        assert result.dims == ("distance", "time")
        assert np.array_equal(result.values, da.values.T)
        assert result.equals(da.transpose())
        assert result.equals(da.transpose(..., "time"))
        assert result.equals(da.transpose("distance", ...))
        assert result.equals(da.T)
        with pytest.raises(ValueError, match="must be a permutation of"):
            da.transpose("distance")
        with pytest.raises(ValueError, match="must be a permutation of"):
            da.transpose("space", "frequency")

    def test_expand_dims(self):
        da = DataArray([1.0, 2.0, 3.0], {"x": [0, 1, 2]})
        result = da.expand_dims("y", 0)
        assert result.dims == ("y", "x")
        assert result.shape == (1, 3)

    def test_io(self):
        # both coords interpolated
        da = wavelet_wavefronts()
        with TemporaryDirectory() as dirpath:
            path = os.path.join(dirpath, "tmp.nc")
            da.to_netcdf(path)
            da_recovered = DataArray.from_netcdf(path)
            assert da.equals(da_recovered)

        # mixed interpolated and dense
        da["time"] = np.asarray(da["time"])
        with TemporaryDirectory() as dirpath:
            path = os.path.join(dirpath, "tmp.nc")
            da.to_netcdf(path)
            da_recovered = DataArray.from_netcdf(path)
            assert da.equals(da_recovered)

        # only dense coords
        da["distance"] = np.asarray(da["distance"])
        with TemporaryDirectory() as dirpath:
            path = os.path.join(dirpath, "tmp.nc")
            da.to_netcdf(path)
            da_recovered = DataArray.from_netcdf(path)
            assert da.equals(da_recovered)

    def test_io_with_zfp_compression(self):
        da = DataArray(np.random.rand(101, 101))
        with TemporaryDirectory() as tmpdir:
            tmpfile_uncompressed = os.path.join(tmpdir, "uncompressed.nc")
            da.to_netcdf(tmpfile_uncompressed)
            tmpfile_compressed = os.path.join(tmpdir, "compressed.nc")
            da.to_netcdf(tmpfile_compressed, encoding=hdf5plugin.Zfp(accuracy=0.001))
            tmpfile_chunk_compressed = os.path.join(tmpdir, "chunk_compressed.nc")
            da.to_netcdf(
                tmpfile_chunk_compressed,
                encoding={"chunks": (10, 10), **hdf5plugin.Zfp(accuracy=0.001)},
            )
            uncompressed_size = os.path.getsize(tmpfile_uncompressed)
            compressed_size = os.path.getsize(tmpfile_compressed)
            chunk_compressed_size = os.path.getsize(tmpfile_chunk_compressed)
            assert chunk_compressed_size < uncompressed_size
            assert compressed_size < chunk_compressed_size
            _da = DataArray.from_netcdf(tmpfile_compressed)
            assert np.abs(da - _da).max().values < 0.001

    def test_ufunc(self):
        da = wavelet_wavefronts()
        result = np.add(da, 1)
        assert np.array_equal(result.data, da.data + 1)
        result = np.add(da, np.ones(da.shape[-1]))
        assert np.array_equal(result.data, da.data + 1)
        result = np.add(da, da)
        assert np.array_equal(result.data, da.data + da.data)
        result = np.add(da, da.isel(time=0, drop=True))
        assert np.array_equal(result.data, da.data + da.data[0])

    def test_arithmetics(self):
        da = wavelet_wavefronts()
        result = da + 1
        assert np.array_equal(result.data, da.data + 1)
        result = da + np.array(1)
        assert np.array_equal(result.data, da.data + np.array(1))
        result = np.array(1) + da
        assert np.array_equal(result.data, np.array(1) + da.data)
        result = 1 + da
        assert np.array_equal(result.data, 1 + da.data)
        result = da + np.ones(da.shape[-1])
        assert np.array_equal(result.data, da.data + 1)
        result = da + da
        assert np.array_equal(result.data, da.data + da.data)
        result = da + da.isel(time=0, drop=True)
        assert np.array_equal(result.data, da.data + da.data[0])
