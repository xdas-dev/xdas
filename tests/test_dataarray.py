import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest

import xdas
from xdas.core.coordinates import Coordinates, DenseCoordinate, InterpCoordinate
from xdas.core.dataarray import DataArray
from xdas.synthetics import generate


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

    def test_cannot_set_dims(self):
        da = self.generate()
        with pytest.raises(AttributeError):
            da.dims = ("other_dim",)

    def test_getitem(self):
        da = self.generate()
        # assert da[0].data == 0.0
        # assert da[1].data == 0.1
        # assert da[0]["dim"] == 100.0
        subda = da[2:4]
        assert np.allclose(subda.data, [0.2, 0.3])
        assert np.allclose(subda["dim"].tie_indices, [0, 1])
        assert np.allclose(subda["dim"].tie_values, [300.0, 400.0])
        da = self.generate(dense=True)
        assert np.allclose(subda.data, [0.2, 0.3])
        assert np.allclose(subda["dim"].values, [300.0, 400.0])

    def test_setitem(self):
        da = self.generate()
        # da[0] = -100.0
        # assert da[0].data == -100.0

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

    def test_isel(self):
        da = generate()
        result = da.isel(first=0)
        excepted = da.isel(time=0)
        assert result.equals(excepted)
        result = da.isel(last=0)
        excepted = da.isel(distance=0)
        assert result.equals(excepted)

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
        da = generate()
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
        da = DataArray(data, coords)

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

    def test_io(self):
        # both coords interpolated
        da = generate()
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
