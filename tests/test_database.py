import numpy as np
import pytest

from xdas.database import ScaleOffset, Coordinate, Coordinates, Database


class TestScaleOffset:
    def test_init(self):
        transform = ScaleOffset(10.0, 100.0)
        assert transform.scale == 10.0
        assert transform.offset == 100.0

    def test_direct(self):
        transform = ScaleOffset(10.0, 100.0)
        assert transform.direct(150.0) == 5.0
        assert np.allclose(
            transform.direct(np.array([150.0, 160.0])), np.array([5.0, 6.0])
        )
        transform = ScaleOffset(
            np.timedelta64(1, "s"), np.datetime64("2000-01-01T00:00:00")
        )
        assert transform.direct(np.datetime64("2000-01-01T00:00:10")) == 10.0

    def test_inverse(self):
        transform = ScaleOffset(10.0, 100.0)
        assert transform.inverse(5.0) == 150
        assert np.allclose(
            transform.inverse(np.array([5.0, 6.0])), np.array([150.0, 160.0])
        )
        transform = ScaleOffset(
            np.timedelta64(1, "s"), np.datetime64("2000-01-01T00:00:00")
        )
        assert transform.inverse(10.0) == np.datetime64("2000-01-01T00:00:10")
        assert transform.inverse(10.3) == np.datetime64("2000-01-01T00:00:10")
        assert transform.inverse(10.7) == np.datetime64("2000-01-01T00:00:11")
        assert transform.inverse(11.0) == np.datetime64("2000-01-01T00:00:11")
        transform = ScaleOffset(
            np.timedelta64(1_000_000, "us"), np.datetime64("2000-01-01T00:00:00")
        )
        assert transform.inverse(10.111111111) == np.datetime64(
            "2000-01-01T00:00:10.111111"
        )
        assert transform.inverse(10.777777777) == np.datetime64(
            "2000-01-01T00:00:10.777778"
        )


class TestCoordinate:
    def test_init(self):
        coord = Coordinate([0, 8], [100.0, 900.0])
        assert np.allclose(coord.tie_indices, [0, 8])
        assert np.allclose(coord.tie_values, [100.0, 900.0])
        assert coord.kind == "linear"

    def test_bool(self):
        coord = Coordinate([0, 8], [100.0, 900.0])
        assert coord
        assert not Coordinate([], [])

    def test_len(self):
        coord = Coordinate([0, 8], [100.0, 900.0])
        assert len(coord) == 9
        assert len(Coordinate([], [])) == 0

    def test_repr(self):
        # TODO
        pass

    def test_eq(self):
        coord1 = Coordinate([0, 8], [100.0, 900.0])
        coord2 = Coordinate([0, 8], [100.0, 900.0])
        assert coord1 == coord2

    def test_getitem(self):
        coord = Coordinate([0, 8], [100.0, 900.0])
        assert coord[0] == 100.0
        assert coord[4] == 500.0
        assert coord[8] == 900.0
        assert coord[-1] == 900.0
        assert coord[-2] == 800.0
        assert np.allclose(coord[[1, 2, 3]], [200.0, 300.0, 400.0])
        with pytest.raises(IndexError):
            coord[9]
            coord[-9]
        coord[0:2] == Coordinate([0, 1], [100.0, 200.0])
        coord[:] == coord
        coord[6:3] == Coordinate([], [])
        coord[1:2] == Coordinate([0], [200.0])
        coord[-3:-1] == Coordinate([0, 1], [700.0, 800.0])

    def test_setitem(self):
        coord = Coordinate([0, 8], [100.0, 900.0])
        with pytest.raises(TypeError):
            coord[1] = 0
            coord[:] = 0

    def test_asarray(self):
        coord = Coordinate([0, 8], [100.0, 900.0])
        assert np.allclose(np.asarray(coord), coord.values())

    def test_dtype(self):
        coord = Coordinate([0, 8], [100.0, 900.0])
        assert coord.dtype == np.float64

    def test_ndim(self):
        coord = Coordinate([0, 8], [100.0, 900.0])
        assert coord.ndim == 1
        assert isinstance(coord.ndim, int)

    def test_shape(self):
        coord = Coordinate([0, 8], [100.0, 900.0])
        assert coord.shape == (9,)

    def test_format_index(self):
        # TODO
        pass

    def test_format_index_slice(self):
        # TODO
        pass

    def test_get_value(self):
        coord = Coordinate([0, 8], [100.0, 900.0])
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
        coord = Coordinate([0, 8], [100.0, 900.0])
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
        coord = Coordinate([0, 8], [100.0, 900.0])
        assert np.all(np.equal(coord.indices(), np.arange(9)))

    def test_values(self):
        coord = Coordinate([0, 8], [100.0, 900.0])
        assert np.allclose(coord.values(), np.arange(100.0, 1000.0, 100.0))

    def test_get_index_slice(self):
        coord = Coordinate([0, 8], [100.0, 900.0])
        assert coord.get_index_slice(slice(100.0, 200.0)) == slice(0, 2)
        assert coord.get_index_slice(slice(150.0, 250.0)) == slice(1, 2)
        assert coord.get_index_slice(slice(300.0, 500.0)) == slice(2, 5)
        assert coord.get_index_slice(slice(0.0, 500.0)) == slice(0, 5)
        assert coord.get_index_slice(slice(125.0, 175.0)) == slice(1, 1)
        assert coord.get_index_slice(slice(0.0, 50.0)) == slice(0, 0)
        assert coord.get_index_slice(slice(1000.0, 1100.0)) == slice(9, 9)
        assert coord.get_index_slice(slice(1000.0, 500.0)) == slice(9, 5)
        assert coord.get_index_slice(slice(None, None)) == slice(None, None)

    def test_slice_index(self):
        coord = Coordinate([0, 8], [100.0, 900.0])
        assert coord.slice_index(slice(0, 2)) == Coordinate([0, 1], [100.0, 200.0])
        assert coord.slice_index(slice(7, None)) == Coordinate([0, 1], [800.0, 900.0])
        assert coord.slice_index(slice(None, None)) == coord
        assert coord.slice_index(slice(0, 0)) == Coordinate([], [])
        assert coord.slice_index(slice(4, 2)) == Coordinate([], [])
        assert coord.slice_index(slice(9, 9)) == Coordinate([], [])
        assert coord.slice_index(slice(3, 3)) == Coordinate([], [])
        assert coord.slice_index(slice(0, -1)) == Coordinate([0, 7], [100.0, 800.0])
        assert coord.slice_index(slice(0, -2)) == Coordinate([0, 6], [100.0, 700.0])
        assert coord.slice_index(slice(-2, None)) == Coordinate([0, 1], [800.0, 900.0])
        assert coord.slice_index(slice(1, 2)) == Coordinate([0], [200.0])

    def test_to_index(self):
        # TODO
        pass

    def test_simplify(self):
        # TODO
        pass


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
