import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest

import xdas
from xdas.core.coordinates import Coordinates, DenseCoordinate, InterpCoordinate
from xdas.core.database import DataArray
from xdas.synthetics import generate


class TestDatabase:
    def generate(self, dense=False):
        if dense:
            coords = {"dim": 100.0 * (1 + np.arange(9))}
        else:
            coords = {"dim": {"tie_indices": [0, 8], "tie_values": [100.0, 900.0]}}
        data = 0.1 * np.arange(9)
        db = xdas.DataArray(data, coords)
        return db

    def test_init_and_properties(self):
        db = self.generate()
        assert isinstance(db["dim"], InterpCoordinate)
        assert db.dims == ("dim",)
        assert db.ndim == 1
        assert db.shape == (9,)
        assert db.sizes == {"dim": 9}
        assert db.sizes["first"] == 9
        assert db.sizes["last"] == 9
        assert np.allclose(db.data, 0.1 * np.arange(9))
        assert np.all(np.equal(db.values, db.data))
        assert db.get_axis_num("dim") == 0
        assert db.dtype == np.float64
        db = self.generate(dense=True)
        assert isinstance(db["dim"], DenseCoordinate)
        db = DataArray()
        assert np.array_equal(db.values, np.array(np.nan), equal_nan=True)
        assert db.coords == {}
        assert db.dims == tuple()
        db = DataArray([[]])
        assert db.dims == ("dim_0", "dim_1")
        assert db.ndim == 2
        db = DataArray(1)
        assert db.dims == tuple()
        assert db.ndim == 0

    def test_getitem(self):
        db = self.generate()
        # assert db[0].data == 0.0
        # assert db[1].data == 0.1
        # assert db[0]["dim"] == 100.0
        subdb = db[2:4]
        assert np.allclose(subdb.data, [0.2, 0.3])
        assert np.allclose(subdb["dim"].tie_indices, [0, 1])
        assert np.allclose(subdb["dim"].tie_values, [300.0, 400.0])
        db = self.generate(dense=True)
        assert np.allclose(subdb.data, [0.2, 0.3])
        assert np.allclose(subdb["dim"].values, [300.0, 400.0])

    def test_setitem(self):
        db = self.generate()
        # db[0] = -100.0
        # assert db[0].data == -100.0

    def test_sel(self):
        # interp
        db = self.generate()
        db.sel(dim=slice(2, 4))
        assert db.sel(dim=225, method="nearest").values == 0.1
        assert db.sel(dim=225, method="ffill").values == 0.1
        assert db.sel(dim=225, method="bfill").values == 0.2
        with pytest.raises(KeyError):
            db.sel(dim=225, method=None)
        assert db.sel(dim=slice(100.0, 300.0)).equals(db[0:3])
        assert db.sel(dim=slice(100.0, 300.0), endpoint=False).equals(db[0:2])
        # dense
        db = self.generate(dense=True)
        db.sel(dim=slice(2, 4))
        assert db.sel(dim=225, method="nearest").values == 0.1
        assert db.sel(dim=225, method="ffill").values == 0.1
        assert db.sel(dim=225, method="bfill").values == 0.2
        with pytest.raises(KeyError):
            db.sel(dim=225, method=None)
        assert db.sel(dim=slice(100.0, 300.0)).equals(db[0:3])
        assert db.sel(dim=slice(100.0, 300.0), endpoint=False).equals(db[0:2])

    def test_isel(self):
        db = generate()
        result = db.isel(first=0)
        excepted = db.isel(time=0)
        assert result.equals(excepted)
        result = db.isel(last=0)
        excepted = db.isel(distance=0)
        assert result.equals(excepted)

    def test_to_xarray(self):
        for dense in [True, False]:
            db = self.generate(dense=dense)
            da = db.to_xarray()
            assert np.array_equal(da.values, db.values)
            assert np.array_equal(da["dim"].values, db["dim"].values)
            db = db.sel(dim=slice(1000, 2000))  # empty database
            da = db.to_xarray()
            assert np.array_equal(da.values, db.values)
            assert np.array_equal(da["dim"].values, db["dim"].values)

    def test_from_xarray(self):
        db = self.generate()
        da = db.to_xarray()
        out = DataArray.from_xarray(da)
        assert np.array_equal(db.values, out.values)
        assert np.array_equal(db["dim"].values, out["dim"].values)

    def test_stream(self):
        db = generate()
        st = db.to_stream(dim={"distance": "time"})
        assert st[0].id == "NET.DAS00001.00.BN1"
        assert len(st) == db.sizes["distance"]
        assert st[0].stats.npts == db.sizes["time"]
        assert np.datetime64(st[0].stats.starttime.datetime) == db["time"][0].values
        assert np.datetime64(st[0].stats.endtime.datetime) == db["time"][-1].values
        result = DataArray.from_stream(st)
        assert np.array_equal(result.values.T, db.values)
        assert result.sizes == {
            "channel": db.sizes["distance"],
            "time": db.sizes["time"],
        }
        assert result["time"].equals(db["time"])

    def test_dense_str(self):
        coord = [f"D{k}" for k in range(9)]
        coords = Coordinates({"dim": coord})
        data = 0.1 * np.arange(9)
        db = DataArray(data, coords)

    def test_single_index_selection(self):
        db = DataArray(
            np.arange(12).reshape(3, 4),
            {
                "time": {"tie_values": [0.0, 1.0], "tie_indices": [0, 2]},
                "distance": [0.0, 10.0, 20.0, 30.0],
            },
        )
        db_getitem = db[1]
        db_isel = db.isel(time=1)
        db_sel = db.sel(time=0.5)
        db_expected = DataArray(
            np.array([4, 5, 6, 7]),
            {"time": (None, 0.5), "distance": [0.0, 10.0, 20.0, 30.0]},
        )
        assert db_getitem.equals(db_expected)
        assert db_isel.equals(db_expected)
        assert db_sel.equals(db_expected)
        db_getitem = db[:, 1]
        db_isel = db.isel(distance=1)
        db_sel = db.sel(distance=10.0)
        db_expected = DataArray(
            np.array([1, 5, 9]),
            {
                "time": {"tie_values": [0.0, 1.0], "tie_indices": [0, 2]},
                "distance": (None, 10.0),
            },
        )
        assert db_getitem.equals(db_expected)
        assert db_isel.equals(db_expected)
        assert db_sel.equals(db_expected)

    def test_io(self):
        # both coords interpolated
        db = generate()
        with TemporaryDirectory() as dirpath:
            path = os.path.join(dirpath, "tmp.nc")
            db.to_netcdf(path)
            db_recovered = DataArray.from_netcdf(path)
            assert db.equals(db_recovered)

        # mixed interpolated and dense
        db["time"] = np.asarray(db["time"])
        with TemporaryDirectory() as dirpath:
            path = os.path.join(dirpath, "tmp.nc")
            db.to_netcdf(path)
            db_recovered = DataArray.from_netcdf(path)
            assert db.equals(db_recovered)

        # only dense coords
        db["distance"] = np.asarray(db["distance"])
        with TemporaryDirectory() as dirpath:
            path = os.path.join(dirpath, "tmp.nc")
            db.to_netcdf(path)
            db_recovered = DataArray.from_netcdf(path)
            assert db.equals(db_recovered)
