import numpy as np
import pytest

from xdas.coordinates import Coordinate, ScaleOffset


class TestScaleOffset:
    def test_init(self):
        transform = ScaleOffset(10.0, 100.0)
        assert transform.scale == 10.0
        assert transform.offset == 100.0

    def test_eq(self):
        transform1 = ScaleOffset(10.0, 100.0)
        transform2 = ScaleOffset(
            np.timedelta64(1, "s"), np.datetime64("2000-01-01T00:00:00")
        )
        assert transform1 == transform1
        assert transform2 == transform2
        assert transform1 != transform2

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
            np.timedelta64(1000, "us"), np.datetime64("2000-01-01T00:00:00")
        )
        assert transform.inverse(1000.0) == np.datetime64("2000-01-01T00:00:01")
        assert transform.inverse(1000.3) == np.datetime64("2000-01-01T00:00:01")
        assert transform.inverse(1000.7) == np.datetime64("2000-01-01T00:00:01.001")

    def test_floatize(self):
        assert ScaleOffset.floatize(np.linspace(0, 1)) == ScaleOffset(1.0, 0.0)
        assert ScaleOffset.floatize(
            np.array(
                [
                    np.datetime64("2000-01-01T00:00:00"),
                    np.datetime64("2000-01-02T00:00:00"),
                ]
            )
        ) == ScaleOffset(np.timedelta64(1, "s"), np.datetime64("2000-01-01T12:00:00"))

    def test_accuracy(self):
        x = np.random.rand(1000)
        transform = ScaleOffset.floatize(x)
        assert np.all(np.abs(x - transform.inverse(transform.direct(x))) < 1e-15)

        t0 = np.datetime64("2000-01-01T00:00:00.000000000")
        three_years_ns = t0 + np.random.randint(0, int(1e9) * 3600 * 24 * 1000, 1000)
        three_month_ns = t0 + np.random.randint(0, int(1e9) * 3600 * 24 * 100, 1000)

        t = three_years_ns.astype("datetime64[us]")
        transform = ScaleOffset.floatize(t)
        assert np.all(t == transform.inverse(transform.direct(t)))

        t = three_month_ns
        transform = ScaleOffset.floatize(t)
        assert np.all(t == transform.inverse(transform.direct(t)))

        t = three_years_ns
        with pytest.warns(UserWarning):
            transform = ScaleOffset.floatize(t)
            error = np.abs(t - transform.inverse(transform.direct(t)))
            assert np.all(error < np.timedelta64(1, "us"))


class TestCoordinate:
    def test_init(self):
        coord = Coordinate([0, 8], [100.0, 900.0])
        assert np.allclose(coord.tie_indices, [0, 8])
        assert np.allclose(coord.tie_values, [100.0, 900.0])
        assert coord.kind == "linear"
        with pytest.raises(ValueError):
            Coordinate(0, 100.0)
        with pytest.raises(ValueError):
            Coordinate([1, 9], [100.0, 900.0])
        with pytest.raises(ValueError):
            Coordinate([0, 8, 9], [100.0, 900.0])
        with pytest.raises(ValueError):
            Coordinate([[0], [8], [9]], [100.0, 900.0])

    def test_len(self):
        assert len(Coordinate([0, 8], [100.0, 900.0])) == 9
        assert len(Coordinate([], [])) == 0

    def test_repr(self):
        # TODO
        pass

    def test_equals(self):
        coord1 = Coordinate([0, 8], [100.0, 900.0])
        coord2 = Coordinate([0, 8], [100.0, 900.0])
        assert coord1.equals(coord2)

    def test_arithmetic(self):
        coord1 = Coordinate([0, 8], [100.0, 900.0])
        coord2 = Coordinate([0, 8], [150.0, 950.0])
        assert coord2.equals(coord1 + 50.0)
        assert coord1.equals(coord2 - 50.0)

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
        assert coord[0:2].equals(Coordinate([0, 1], [100.0, 200.0]))
        assert coord[:].equals(coord)
        assert coord[6:3].equals(Coordinate([], []))
        assert coord[1:2].equals(Coordinate([0], [200.0]))
        assert coord[-3:-1].equals(Coordinate([0, 1], [700.0, 800.0]))

    def test_setitem(self):
        coord = Coordinate([0, 8], [100.0, 900.0])
        with pytest.raises(TypeError):
            coord[1] = 0
            coord[:] = 0

    def test_asarray(self):
        coord = Coordinate([0, 8], [100.0, 900.0])
        assert np.allclose(np.asarray(coord), coord.values)

    def test_empty(self):
        assert not Coordinate([0, 8], [100.0, 900.0]).empty
        assert Coordinate([], []).empty

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
        starttime = np.datetime64("2000-01-01T00:00:00")
        endtime = np.datetime64("2000-01-01T00:00:08")
        coord = Coordinate([0, 8], [starttime, endtime])
        assert coord.get_value(0) == starttime
        assert coord.get_value(4) == np.datetime64("2000-01-01T00:00:04")
        assert coord.get_value(8) == endtime
        assert coord.get_value(-1) == endtime
        assert coord.get_value(-9) == starttime

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

        starttime = np.datetime64("2000-01-01T00:00:00")
        endtime = np.datetime64("2000-01-01T00:00:08")
        coord = Coordinate([0, 8], [starttime, endtime])
        assert coord.get_index(starttime) == 0
        assert coord.get_index(endtime) == 8
        assert coord.get_index(str(starttime)) == 0
        assert coord.get_index(str(endtime)) == 8
        assert coord.get_index("2000-01-01T00:00:04.1", "nearest") == 4

    def test_indices(self):
        coord = Coordinate([0, 8], [100.0, 900.0])
        assert np.all(np.equal(coord.indices, np.arange(9)))

    def test_values(self):
        coord = Coordinate([0, 8], [100.0, 900.0])
        assert np.allclose(coord.values, np.arange(100.0, 1000.0, 100.0))

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
        assert coord.slice_index(slice(0, 2)).equals(Coordinate([0, 1], [100.0, 200.0]))
        assert coord.slice_index(slice(7, None)).equals(
            Coordinate([0, 1], [800.0, 900.0])
        )
        assert coord.slice_index(slice(None, None)).equals(coord)
        assert coord.slice_index(slice(0, 0)).equals(Coordinate([], []))
        assert coord.slice_index(slice(4, 2)).equals(Coordinate([], []))
        assert coord.slice_index(slice(9, 9)).equals(Coordinate([], []))
        assert coord.slice_index(slice(3, 3)).equals(Coordinate([], []))
        assert coord.slice_index(slice(0, -1)).equals(
            Coordinate([0, 7], [100.0, 800.0])
        )
        assert coord.slice_index(slice(0, -2)).equals(
            Coordinate([0, 6], [100.0, 700.0])
        )
        assert coord.slice_index(slice(-2, None)).equals(
            Coordinate([0, 1], [800.0, 900.0])
        )
        assert coord.slice_index(slice(1, 2)).equals(Coordinate([0], [200.0]))
        assert coord.slice_index(slice(1, 3, 2)).equals(Coordinate([0], [200.0]))
        assert coord.slice_index(slice(None, None, 2)).equals(
            Coordinate([0, 4], [100.0, 900.0])
        )
        assert coord.slice_index(slice(None, None, 3)).equals(
            Coordinate([0, 2], [100.0, 700.0])
        )
        assert coord.slice_index(slice(None, None, 4)).equals(
            Coordinate([0, 2], [100.0, 900.0])
        )
        assert coord.slice_index(slice(None, None, 5)).equals(
            Coordinate([0, 1], [100.0, 600.0])
        )
        assert coord.slice_index(slice(2, 7, 3)).equals(
            Coordinate([0, 1], [300.0, 600.0])
        )

    def test_to_index(self):
        # TODO
        pass

    def test_simplify(self):
        # TODO
        pass
