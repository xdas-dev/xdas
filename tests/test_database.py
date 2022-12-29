import numpy as np
import pytest

from xdas.database import Coordinate, Coordinates, Database


class TestCoordinate:
    def generate(self):
        return Coordinate([0, 8], [100.0, 900.0])

    def test_init(self):
        coord = self.generate()
        assert np.allclose(coord.tie_indices, [0, 8])
        assert np.allclose(coord.tie_values, [100.0, 900.0])
        assert coord.kind == "linear"

    def test_len(self):
        coord = self.generate()
        assert len(coord) == 9

    def test_eq(self):
        coord1 = self.generate()
        coord2 = self.generate()
        assert coord1 == coord2

    def test_dtype(self):
        coord = self.generate()
        assert coord.dtype == np.float64

    def test_ndim(self):
        coord = self.generate()
        assert coord.ndim == 1
        assert isinstance(coord.ndim, int)

    def test_shape(self):
        coord = self.generate()
        assert coord.shape == (9,)

    def test_get_value(self):
        coord = self.generate()
        assert coord.get_value(0) == 100.0
        assert coord.get_value(4) == 500.0
        assert coord.get_value(8) == 900.0
        assert coord.get_value(-1) == 900.0
        assert coord.get_value(-9) == 100.0
        assert np.allclose(coord.get_value([1, 2, 3, -2]), [200.0, 300.0, 400.0, 800.0])
        with pytest.raises(IndexError):
            coord.get_value(-10)
            coord.get_value(9)
            coord.get_value(0.5)

    def test_get_index(self):
        coord = self.generate()
        assert coord.get_index(100.0) == 0
        assert coord.get_index(900.0) == 8
        assert coord.get_index(0.0, "nearest") == 0
        assert coord.get_index(1000.0, "nearest") == 8
        assert coord.get_index(125.0, "nearest") == 0
        assert coord.get_index(175.0, "nearest") == 1
        assert coord.get_index(175.0, "before") == 0
        assert coord.get_index(200.0, "before") == 1
        assert coord.get_index(200.0, "after") == 1
        assert coord.get_index(125.0, "after") == 1
        assert np.all(np.equal(coord.get_index([100.0, 900.0]), [0, 8]))
        with pytest.raises(KeyError):
            assert coord.get_index(0.0) == 0
            assert coord.get_index(1000.0) == 8
            assert coord.get_index(150.0) == 0
            assert coord.get_index(1000.0, "after") == 8
            assert coord.get_index(0.0, "before") == 0

    def test_indices(self):
        coord = self.generate()
        assert np.all(np.equal(coord.indices(), [0, 1, 2, 3, 4, 5, 6, 7, 8]))

    def test_values(self):
        coord = self.generate()
        assert np.all(
            np.equal(
                coord.values(),
                [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0],
            )
        )

    def test_get_index_slice(self):
        coord = self.generate()
        assert coord.get_index_slice(slice(100.0, 200.0)) == slice(0, 2)
        assert coord.get_index_slice(slice(150.0, 250.0)) == slice(1, 2)
        assert coord.get_index_slice(slice(300.0, 500.0)) == slice(2, 5)
        assert coord.get_index_slice(slice(0.0, 500.0)) == slice(0, 5)
        assert coord.get_index_slice(slice(125.0, 175.0)) == slice(1, 1)
        assert coord.get_index_slice(slice(0.0, 50.0)) == slice(0, 0)
        assert coord.get_index_slice(slice(1000.0, 1100.0)) == slice(9, 9)
        assert coord.get_index_slice(slice(1000.0, 500.0)) == slice(9, 5)
        assert coord.get_index_slice(slice(None, None)) == slice(None, None)


    def test_slice(self):
        coord = self.generate()
        assert coord.slice(slice(0, 2)) == Coordinate([0, 1], [100.0, 200.0])

    def test_getitem(self):
        coord = self.generate()
        assert coord[0] == 100.0
        assert coord[4] == 500.0
        assert coord[8] == 900.0
        assert np.allclose(coord.indices(), np.arange(9))
        assert np.allclose(coord.values(), np.arange(100.0, 1000.0, 100.0))
        subcoord = coord[0:2]
        assert np.allclose(subcoord.tie_indices, [0, 1])
        assert np.allclose(subcoord.tie_values, [100.0, 200.0])

    def test_asarray(self):
        coord = self.generate()
        assert np.allclose(np.asarray(coord), coord.values())


class TestDatabase:
    def generate(self):
        coord = Coordinate([0, 8], [100.0, 900.0])
        coords = Coordinates(dim=coord)
        data = 0.1 * np.arange(9)
        database = Database(data, coords)
        return database

    def test_init(self):
        database = self.generate()
        assert database.dims == ("dim",)
        assert database.ndim == 1
        assert database.shape == (9,)
        assert database.sizes == {"dim": 9}

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
