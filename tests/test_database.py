import numpy as np
import pandas as pd

from xdas.coordinates import InterpolatedCoordinate, Coordinates, DenseCoordinate
from xdas.database import Database


class TestDatabase:
    def generate(self, dense=False):
        coord = InterpolatedCoordinate([0, 8], [100.0, 900.0])
        if dense:
            coord = coord.values
        coords = Coordinates(dim=coord)
        data = 0.1 * np.arange(9)
        db = Database(data, coords)
        return db

    def test_init_and_properties(self):
        db = self.generate()
        assert isinstance(db["dim"], InterpolatedCoordinate)
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