import numpy as np

from xdas.coordinates import Coordinate, Coordinates
from xdas.database import Database


class TestDatabase:
    def generate(self):
        coord = Coordinate([0, 8], [100.0, 900.0])
        coords = Coordinates(dim=coord)
        data = 0.1 * np.arange(9)
        database = Database(data, coords)
        return database

    def test_init_and_properties(self):
        database = self.generate()
        assert database.dims == ("dim",)
        assert database.ndim == 1
        assert database.shape == (9,)
        assert database.sizes == {"dim": 9}
        assert np.allclose(database.data, 0.1 * np.arange(9))
        assert np.all(np.equal(database.values, database.data))
        assert database.get_axis_num("dim") == 0
        assert database.dtype == np.float64

    def test_getitem(self):
        database = self.generate()
        assert database[0].data == 0.0
        assert database[1].data == 0.1
        assert database[0]["dim"] == 100.0
        subdatabase = database[2:4]
        assert np.allclose(subdatabase.data, [0.2, 0.3])
        assert np.allclose(subdatabase["dim"].tie_indices, [0, 1])
        assert np.allclose(subdatabase["dim"].tie_values, [300.0, 400.0])

    def test_setitem(self):
        database = self.generate()
        database[0] = -100.0
        assert database[0].data == -100.0

    def test_sel(self):
        database = self.generate()
        sub = database.sel(dim=slice(2, 4))

    def test_to_xarray(self):
        database = self.generate()
        database.to_xarray()
