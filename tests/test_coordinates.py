import numpy as np
import pandas as pd
import pytest

import xdas
from xdas.coordinates import (
    DenseCoordinate,
    InterpCoordinate,
    ScalarCoordinate,
    ScaleOffset,
)


class TestScalarCoordinate:
    valid = [
        1,
        np.array(1),
        1.0,
        np.array(1.0),
        "label",
        np.array("label"),
        np.datetime64(1, "s"),
    ]
    invalid = [[1], np.array([1]), {"key": "value"}]

    def test_isvalid(self):
        for data in self.valid:
            assert ScalarCoordinate.isvalid(data)
        for data in self.invalid:
            assert not ScalarCoordinate.isvalid(data)

    def test_init(self):
        coord = ScalarCoordinate(1)
        assert coord.data == 1
        assert coord.dim is None
        coord = ScalarCoordinate(1, None)
        assert coord.dim is None
        with pytest.raises(ValueError):
            ScalarCoordinate(1, "dim")
        for data in self.valid:
            assert ScalarCoordinate(data).data == np.array(data)
        for data in self.invalid:
            with pytest.raises(TypeError):
                ScalarCoordinate(data)

    def test_getitem(self):
        assert ScalarCoordinate(1)[...].equals(ScalarCoordinate(1))
        with pytest.raises(IndexError):
            ScalarCoordinate(1)[:]
        with pytest.raises(IndexError):
            ScalarCoordinate(1)[0]

    def test_len(self):
        with pytest.raises(TypeError):
            len(ScalarCoordinate(1))

    def test_repr(self):
        for data in self.valid:
            assert ScalarCoordinate(data).__repr__() == str(np.array(data))

    def test_array(self):
        for data in self.valid:
            assert ScalarCoordinate(data).__array__() == np.array(data)

    def test_dtype(self):
        for data in self.valid:
            assert ScalarCoordinate(data).dtype == np.array(data).dtype

    def test_values(self):
        for data in self.valid:
            assert ScalarCoordinate(data).values == np.array(data)

    def test_equals(self):
        for data in self.valid:
            coord = ScalarCoordinate(data)
            assert coord.equals(coord)
        assert ScalarCoordinate(1).equals(ScalarCoordinate(np.array(1)))

    def test_to_index(self):
        with pytest.raises(NotImplementedError):
            ScalarCoordinate(1).to_index("item")

    def test_isinstance(self):
        assert ScalarCoordinate(1).isscalar()
        assert not ScalarCoordinate(1).isdense()
        assert not ScalarCoordinate(1).isinterp()


class TestDenseCoordinate:
    valid = [
        [1, 2, 3],
        np.array([1, 2, 3]),
        [1.0, 2.0, 3.0],
        np.array([1.0, 2.0, 3.0]),
        ["a", "b", "c"],
        np.array(["a", "b", "c"]),
        np.array([1, 2, 3], dtype="datetime64[s]"),
    ]
    invalid = [
        1,
        np.array(1),
        1.0,
        np.array(1.0),
        "label",
        np.array("label"),
        np.datetime64(1, "s"),
        {"key": "value"},
    ]

    def test_isvalid(self):
        for data in self.valid:
            assert DenseCoordinate.isvalid(data)
        for data in self.invalid:
            assert not DenseCoordinate.isvalid(data)

    def test_init(self):
        coord = DenseCoordinate([1, 2, 3])
        assert np.array_equiv(coord.data, [1, 2, 3])
        assert coord.dim is None
        coord = DenseCoordinate([1, 2, 3], "dim")
        assert coord.dim == "dim"
        for data in self.valid:
            assert np.array_equiv(DenseCoordinate(data).data, data)
        for data in self.invalid:
            with pytest.raises(TypeError):
                DenseCoordinate(data)

    def test_getitem(self):
        assert np.array_equiv(DenseCoordinate([1, 2, 3])[...].values, [1, 2, 3])
        assert isinstance(DenseCoordinate([1, 2, 3])[...], DenseCoordinate)
        assert np.array_equiv(DenseCoordinate([1, 2, 3])[:].values, [1, 2, 3])
        assert isinstance(DenseCoordinate([1, 2, 3])[:], DenseCoordinate)
        assert np.array_equiv(DenseCoordinate([1, 2, 3])[1].values, 2)
        assert isinstance(DenseCoordinate([1, 2, 3])[1], ScalarCoordinate)
        assert np.array_equiv(DenseCoordinate([1, 2, 3])[1:].values, [2, 3])
        assert isinstance(DenseCoordinate([1, 2, 3])[1:], DenseCoordinate)

    def test_len(self):
        for data in self.valid:
            assert len(DenseCoordinate(data)) == 3

    def test_repr(self):
        for data in self.valid:
            assert DenseCoordinate(data).__repr__() == str(np.array(data))

    def test_array(self):
        for data in self.valid:
            assert np.array_equiv(DenseCoordinate(data).__array__(), data)

    def test_dtype(self):
        for data in self.valid:
            assert DenseCoordinate(data).dtype == np.array(data).dtype

    def test_values(self):
        for data in self.valid:
            assert np.array_equiv(DenseCoordinate(data).values, data)

    def test_index(self):
        for data in self.valid:
            assert DenseCoordinate(data).index.equals(pd.Index(data))

    def test_equals(self):
        for data in self.valid:
            coord = DenseCoordinate(data)
            assert coord.equals(coord)
        assert DenseCoordinate([1, 2, 3]).equals(DenseCoordinate([1, 2, 3]))

    def test_isinstance(self):
        assert not DenseCoordinate([1, 2, 3]).isscalar()
        assert DenseCoordinate([1, 2, 3]).isdense()
        assert not DenseCoordinate([1, 2, 3]).isinterp()

    def test_get_indexer(self):
        assert DenseCoordinate([1, 2, 3]).get_indexer(2) == 1
        assert np.array_equiv(DenseCoordinate([1, 2, 3]).get_indexer([2, 3]), [1, 2])
        assert DenseCoordinate([1, 2, 3]).get_indexer(2.1, method="nearest") == 1
        assert DenseCoordinate([1, 2, 3]).get_indexer(2.1, method="ffill") == 1
        assert DenseCoordinate([1, 2, 3]).get_indexer(2.1, method="bfill") == 2

    def test_get_slice_indexer(self):
        assert np.array_equiv(
            DenseCoordinate([1, 2, 3]).slice_indexer(start=2), slice(1, 3)
        )

    def test_to_index(self):
        assert DenseCoordinate([1, 2, 3]).to_index(2) == 1
        assert np.array_equiv(DenseCoordinate([1, 2, 3]).to_index([2, 3]), [1, 2])
        assert np.array_equiv(
            DenseCoordinate([1, 2, 3]).to_index(slice(2, None)), slice(1, 3)
        )


class TestInterpCoordinate:
    valid = [
        {"tie_indices": [], "tie_values": []},
        {"tie_indices": [0], "tie_values": [100.0]},
        {"tie_indices": [0, 8], "tie_values": [100.0, 900.0]},
        {"tie_indices": [0, 8], "tie_values": [100, 900]},
        {
            "tie_indices": [0, 8],
            "tie_values": [
                np.datetime64("2000-01-01T00:00:00"),
                np.datetime64("2000-01-01T00:00:08"),
            ],
        },
        {"tie_indices": np.array([0, 8], dtype="int16"), "tie_values": [100.0, 900.0]},
    ]
    invalid = [
        1,
        np.array(1),
        1.0,
        np.array(1.0),
        "label",
        np.array("label"),
        np.datetime64(1, "s"),
        [1, 2, 3],
        np.array([1, 2, 3]),
        [1.0, 2.0, 3.0],
        np.array([1.0, 2.0, 3.0]),
        ["a", "b", "c"],
        np.array(["a", "b", "c"]),
        np.array([1, 2, 3], dtype="datetime64[s]"),
    ]
    error = [
        {"key": "value"},
        {"tie_indices": 0, "tie_values": [100.0]},
        {"tie_indices": [0], "tie_values": 100.0},
        {"tie_indices": [0, 7, 8], "tie_values": [100.0, 900.0]},
        {"tie_indices": [0.0, 8.0], "tie_values": [100.0, 900.0]},
        {"tie_indices": [1, 9], "tie_values": [100.0, 900.0]},
        {"tie_indices": [8, 0], "tie_values": [100.0, 900.0]},
        {"tie_indices": [8, 0], "tie_values": ["a", "b"]},
    ]

    def test_isvalid(self):
        for data in self.valid:
            assert InterpCoordinate.isvalid(data)
        for data in self.invalid:
            assert not InterpCoordinate.isvalid(data)

    def test_init(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert np.array_equiv(coord.data["tie_indices"], [0, 8])
        assert np.array_equiv(coord.data["tie_values"], [100.0, 900.0])
        assert coord.dim is None
        coord = InterpCoordinate(
            {"tie_indices": [0, 8], "tie_values": [100.0, 900.0]}, "dim"
        )
        assert coord.dim == "dim"
        for data in self.valid:
            coord = InterpCoordinate(data)
            assert np.array_equiv(coord.data["tie_indices"], data["tie_indices"])
            assert np.array_equiv(coord.data["tie_values"], data["tie_values"])
        for data in self.invalid:
            with pytest.raises(TypeError):
                InterpCoordinate(data)
        for data in self.error:
            with pytest.raises(ValueError):
                InterpCoordinate(data)

    def test_len(self):
        assert (
            len(InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]}))
            == 9
        )
        assert len(InterpCoordinate(dict(tie_indices=[], tie_values=[]))) == 0

    def test_repr(self):
        # TODO
        pass

    def test_equals(self):
        coord1 = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        coord2 = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert coord1.equals(coord2)

    def test_getitem(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert isinstance(coord[0], ScalarCoordinate)
        assert coord[0].values == 100.0
        assert coord[4].values == 500.0
        assert coord[8].values == 900.0
        assert coord[-1].values == 900.0
        assert coord[-2].values == 800.0
        assert np.allclose(coord[[1, 2, 3]].values, [200.0, 300.0, 400.0])
        with pytest.raises(IndexError):
            coord[9]
            coord[-9]
        assert coord[0:2].equals(
            InterpCoordinate(dict(tie_indices=[0, 1], tie_values=[100.0, 200.0]))
        )
        assert coord[:].equals(coord)
        assert coord[6:3].equals(InterpCoordinate(dict(tie_indices=[], tie_values=[])))
        assert coord[1:2].equals(
            InterpCoordinate(dict(tie_indices=[0], tie_values=[200.0]))
        )
        assert coord[-3:-1].equals(
            InterpCoordinate(dict(tie_indices=[0, 1], tie_values=[700.0, 800.0]))
        )

    def test_setitem(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        with pytest.raises(TypeError):
            coord[1] = 0
            coord[:] = 0

    def test_asarray(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert np.allclose(np.asarray(coord), coord.values)

    def test_empty(self):
        assert not InterpCoordinate(
            {"tie_indices": [0, 8], "tie_values": [100.0, 900.0]}
        ).empty
        assert InterpCoordinate(dict(tie_indices=[], tie_values=[])).empty

    def test_dtype(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert coord.dtype == np.float64

    def test_ndim(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert coord.ndim == 1
        assert isinstance(coord.ndim, int)

    def test_shape(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert coord.shape == (9,)

    def test_format_index(self):
        # TODO
        pass

    def test_format_index_slice(self):
        # TODO
        pass

    def test_get_value(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
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
        coord = InterpCoordinate(
            dict(tie_indices=[0, 8], tie_values=[starttime, endtime])
        )
        assert coord.get_value(0) == starttime
        assert coord.get_value(4) == np.datetime64("2000-01-01T00:00:04")
        assert coord.get_value(8) == endtime
        assert coord.get_value(-1) == endtime
        assert coord.get_value(-9) == starttime

    def test_get_index(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert coord.get_indexer(100.0) == 0
        assert coord.get_indexer(900.0) == 8
        assert coord.get_indexer(0.0, "nearest") == 0
        assert coord.get_indexer(1000.0, "nearest") == 8
        assert coord.get_indexer(125.0, "nearest") == 0
        assert coord.get_indexer(175.0, "nearest") == 1
        assert coord.get_indexer(175.0, "ffill") == 0
        assert coord.get_indexer(200.0, "ffill") == 1
        assert coord.get_indexer(200.0, "bfill") == 1
        assert coord.get_indexer(125.0, "bfill") == 1
        assert np.all(np.equal(coord.get_indexer([100.0, 900.0]), [0, 8]))
        with pytest.raises(KeyError):
            assert coord.get_indexer(0.0) == 0
            assert coord.get_indexer(1000.0) == 8
            assert coord.get_indexer(150.0) == 0
            assert coord.get_indexer(1000.0, "bfill") == 8
            assert coord.get_indexer(0.0, "ffill") == 0

        starttime = np.datetime64("2000-01-01T00:00:00")
        endtime = np.datetime64("2000-01-01T00:00:08")
        coord = InterpCoordinate(
            dict(tie_indices=[0, 8], tie_values=[starttime, endtime])
        )
        assert coord.get_indexer(starttime) == 0
        assert coord.get_indexer(endtime) == 8
        assert coord.get_indexer(str(starttime)) == 0
        assert coord.get_indexer(str(endtime)) == 8
        assert coord.get_indexer("2000-01-01T00:00:04.1", "nearest") == 4

    def test_indices(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert np.all(np.equal(coord.indices, np.arange(9)))

    def test_values(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert np.allclose(coord.values, np.arange(100.0, 1000.0, 100.0))

    def test_get_index_slice(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert coord.slice_indexer(100.0, 200.0) == slice(0, 2)
        assert coord.slice_indexer(150.0, 250.0) == slice(1, 2)
        assert coord.slice_indexer(300.0, 500.0) == slice(2, 5)
        assert coord.slice_indexer(0.0, 500.0) == slice(0, 5)
        assert coord.slice_indexer(125.0, 175.0) == slice(1, 1)
        assert coord.slice_indexer(0.0, 50.0) == slice(0, 0)
        assert coord.slice_indexer(1000.0, 1100.0) == slice(9, 9)
        assert coord.slice_indexer(1000.0, 500.0) == slice(9, 5)
        assert coord.slice_indexer(None, None) == slice(None, None)

    def test_slice_index(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert coord.slice_index(slice(0, 2)).equals(
            InterpCoordinate(dict(tie_indices=[0, 1], tie_values=[100.0, 200.0]))
        )
        assert coord.slice_index(slice(7, None)).equals(
            InterpCoordinate(dict(tie_indices=[0, 1], tie_values=[800.0, 900.0]))
        )
        assert coord.slice_index(slice(None, None)).equals(coord)
        assert coord.slice_index(slice(0, 0)).equals(
            InterpCoordinate(dict(tie_indices=[], tie_values=[]))
        )
        assert coord.slice_index(slice(4, 2)).equals(
            InterpCoordinate(dict(tie_indices=[], tie_values=[]))
        )
        assert coord.slice_index(slice(9, 9)).equals(
            InterpCoordinate(dict(tie_indices=[], tie_values=[]))
        )
        assert coord.slice_index(slice(3, 3)).equals(
            InterpCoordinate(dict(tie_indices=[], tie_values=[]))
        )
        assert coord.slice_index(slice(0, -1)).equals(
            InterpCoordinate(dict(tie_indices=[0, 7], tie_values=[100.0, 800.0]))
        )
        assert coord.slice_index(slice(0, -2)).equals(
            InterpCoordinate(dict(tie_indices=[0, 6], tie_values=[100.0, 700.0]))
        )
        assert coord.slice_index(slice(-2, None)).equals(
            InterpCoordinate(dict(tie_indices=[0, 1], tie_values=[800.0, 900.0]))
        )
        assert coord.slice_index(slice(1, 2)).equals(
            InterpCoordinate(dict(tie_indices=[0], tie_values=[200.0]))
        )
        assert coord.slice_index(slice(1, 3, 2)).equals(
            InterpCoordinate(dict(tie_indices=[0], tie_values=[200.0]))
        )
        assert coord.slice_index(slice(None, None, 2)).equals(
            InterpCoordinate(dict(tie_indices=[0, 4], tie_values=[100.0, 900.0]))
        )
        assert coord.slice_index(slice(None, None, 3)).equals(
            InterpCoordinate(dict(tie_indices=[0, 2], tie_values=[100.0, 700.0]))
        )
        assert coord.slice_index(slice(None, None, 4)).equals(
            InterpCoordinate(dict(tie_indices=[0, 2], tie_values=[100.0, 900.0]))
        )
        assert coord.slice_index(slice(None, None, 5)).equals(
            InterpCoordinate(dict(tie_indices=[0, 1], tie_values=[100.0, 600.0]))
        )
        assert coord.slice_index(slice(2, 7, 3)).equals(
            InterpCoordinate(dict(tie_indices=[0, 1], tie_values=[300.0, 600.0]))
        )

    def test_to_index(self):
        # TODO
        pass

    def test_simplify(self):
        # TODO
        pass


class TestCoordinate:
    def test_new(self):
        assert xdas.Coordinate(1).isscalar()
        assert xdas.Coordinate([1]).isdense()
        assert xdas.Coordinate({"tie_values": [], "tie_indices": []}).isinterp()
        coord = xdas.Coordinate(xdas.Coordinate([1]), "dim")
        assert coord.isdense()
        assert coord.dim == "dim"


class TestCoordinates:
    def test_init(self):
        coords = xdas.Coordinates(
            {"dim": ("dim", {"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})}
        )
        coord = coords["dim"]
        assert coord.isinterp()
        assert np.allclose(coord.tie_indices, [0, 8])
        assert np.allclose(coord.tie_values, [100.0, 900.0])
        assert coords.isdim("dim")
        coords = xdas.Coordinates({"dim": [1.0, 2.0, 3.0]})
        coord = coords["dim"]
        assert coord.isdense()
        assert np.allclose(coord.values, [1.0, 2.0, 3.0])
        assert coords.isdim("dim")
        coords = xdas.Coordinates(
            {
                "dim_0": (
                    "dim_0",
                    {"tie_indices": [0, 8], "tie_values": [100.0, 900.0]},
                ),
                "dim_1": (
                    "dim_0",
                    {"tie_indices": [0, 8], "tie_values": [100.0, 900.0]},
                ),
            }
        )
        assert coords.isdim("dim_0")
        assert not coords.isdim("dim_1")


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
