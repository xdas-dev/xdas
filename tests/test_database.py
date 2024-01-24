import os
from tempfile import TemporaryDirectory

import numpy as np

import xdas
from xdas.coordinates import Coordinates, DenseCoordinate, InterpCoordinate
from xdas.database import Database


class TestDatabase:
    def generate(self, dense=False):
        coord = xdas.Coordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        if dense:
            coord = coord.values
        coords = xdas.Coordinates(dim=coord)
        data = 0.1 * np.arange(9)
        db = xdas.Database(data, coords)
        return db

    def test_init_and_properties(self):
        db = self.generate()
        assert isinstance(db["dim"], InterpCoordinate)
        assert db.dims == ("dim",)
        assert db.ndim == 1
        assert db.shape == (9,)
        assert db.sizes == {"dim": 9}
        assert np.allclose(db.data, 0.1 * np.arange(9))
        assert np.all(np.equal(db.values, db.data))
        assert db.get_axis_num("dim") == 0
        assert db.dtype == np.float64
        db = self.generate(dense=True)
        assert isinstance(db["dim"], DenseCoordinate)

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
        db = self.generate()
        db.sel(dim=slice(2, 4))
        db = self.generate(dense=True)
        db.sel(dim=slice(2, 4))

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
        out = Database.from_xarray(da)
        assert np.array_equal(db.values, out.values)
        assert np.array_equal(db["dim"].values, out["dim"].values)

    def test_dense_str(self):
        coord = [f"D{k}" for k in range(9)]
        coords = Coordinates(dim=coord)
        data = 0.1 * np.arange(9)
        db = Database(data, coords)

    def test_single_index_selection(self):
        db = Database(
            np.arange(12).reshape(3, 4),
            {
                "time": {"tie_values": [0.0, 1.0], "tie_indices": [0, 2]},
                "distance": [0.0, 10.0, 20.0, 30.0],
            },
        )
        db_getitem = db[1]
        db_isel = db.isel(time=1)
        db_sel = db.sel(time=0.5)
        db_expected = Database(
            np.array([4, 5, 6, 7]), {"time": 0.5, "distance": [0.0, 10.0, 20.0, 30.0]}
        )
        assert db_getitem.equals(db_expected)
        assert db_isel.equals(db_expected)
        assert db_sel.equals(db_expected)
        db_getitem = db[:, 1]
        db_isel = db.isel(distance=1)
        db_sel = db.sel(distance=10.0)
        db_expected = Database(
            np.array([1, 5, 9]),
            {
                "time": {"tie_values": [0.0, 1.0], "tie_indices": [0, 2]},
                "distance": 10.0,
            },
        )
        assert db_getitem.equals(db_expected)
        assert db_isel.equals(db_expected)
        assert db_sel.equals(db_expected)

    def test_io(self):
        db = Database(
            data=np.arange(12).reshape(3, 4),
            coords={
                "time": {
                    "tie_indices": [0, 2],
                    "tie_values": np.array(
                        ["2000-01-01T00:00:00", "2000-01-01T00:00:02"],
                        dtype="datetime64[s]",
                    ),
                },
                "distance": [0.0, 100.0, 200.0, 300.0],
            },
        )
        with TemporaryDirectory() as dirpath:
            path = os.path.join(dirpath, "tmp.nc")
            db.to_netcdf(path)
            db_recovered = Database.from_netcdf(path)
            assert db.equals(db_recovered)
