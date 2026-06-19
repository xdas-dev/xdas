import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xdas.coordinates import DenseCoordinate, ScalarCoordinate


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
            assert DenseCoordinate._isvalid(data)
        for data in self.invalid:
            assert not DenseCoordinate._isvalid(data)

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
            assert DenseCoordinate(data).__repr__() == np.array2string(
                np.asarray(data), threshold=0, edgeitems=1
            )

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
        assert not DenseCoordinate([1, 2, 3]).equals(42)

    def test_get_indexer(self):
        assert DenseCoordinate([1, 2, 3])._get_indexer(2) == 1
        assert np.array_equiv(DenseCoordinate([1, 2, 3])._get_indexer([2, 3]), [1, 2])
        assert DenseCoordinate([1, 2, 3])._get_indexer(2.1, method="nearest") == 1
        assert DenseCoordinate([1, 2, 3])._get_indexer(2.1, method="ffill") == 1
        assert DenseCoordinate([1, 2, 3])._get_indexer(2.1, method="bfill") == 2

    def test_get_slice_indexer(self):
        assert DenseCoordinate([1, 2, 3]).slice_indexer(start=2) == slice(1, None)

    def test_to_index(self):
        assert DenseCoordinate([1, 2, 3]).to_index(2) == 1
        assert np.array_equiv(DenseCoordinate([1, 2, 3]).to_index([2, 3]), [1, 2])
        assert DenseCoordinate([1, 2, 3]).to_index(slice(2, None)) == slice(1, None)

    def test_empty(self):
        coord = DenseCoordinate()
        assert coord.empty

    def test_concat(self):
        coord0 = DenseCoordinate()
        coord1 = DenseCoordinate([1, 2, 3])
        coord2 = DenseCoordinate([4, 5, 6])

        result = coord1._concat(coord2)
        expected = DenseCoordinate([1, 2, 3, 4, 5, 6])
        assert result.equals(expected)

        result = coord2._concat(coord1)
        expected = DenseCoordinate([4, 5, 6, 1, 2, 3])
        assert result.equals(expected)

        assert coord0._concat(coord0).empty
        assert coord0._concat(coord1).equals(coord1)
        assert coord1._concat(coord0).equals(coord1)

        with pytest.raises(TypeError):
            coord1._concat(ScalarCoordinate(1))
        with pytest.raises(ValueError, match="different dimension"):
            DenseCoordinate([1, 2, 3], "x")._concat(DenseCoordinate([4, 5, 6], "y"))
        with pytest.raises(ValueError, match="different dtype"):
            DenseCoordinate(np.array([1, 2, 3], dtype=np.int32))._concat(
                DenseCoordinate(np.array([4.0, 5.0, 6.0], dtype=np.float64))
            )

    def test_get_div_points(self):
        coord = DenseCoordinate([1, 2, 3, 10, 11, 12])
        div_points = coord.get_div_points(tolerance=3.0)
        assert np.array_equal(div_points, [0, 3, 6])
        with pytest.raises(NotImplementedError):
            coord.get_div_points()

    def test_from_block(self):
        coord = DenseCoordinate.from_block(0, 5, 1, dim="x")
        expected = DenseCoordinate([0, 1, 2, 3, 4], dim="x")
        assert coord.equals(expected)

    def test_is_monotonic_increasing(self):
        assert DenseCoordinate([1, 2, 3])._is_monotonic_increasing()
        assert not DenseCoordinate([1, 3, 2])._is_monotonic_increasing()
        t0 = np.datetime64("2000-01-01T00:00:00")
        times = np.array([t0, t0 + np.timedelta64(1, "s"), t0 + np.timedelta64(2, "s")])
        assert DenseCoordinate(times)._is_monotonic_increasing()
        times_bad = np.array(
            [t0, t0 + np.timedelta64(2, "s"), t0 + np.timedelta64(1, "s")]
        )
        assert not DenseCoordinate(times_bad)._is_monotonic_increasing()

    def test_add(self):
        coord = DenseCoordinate([1.0, 2.0, 3.0], "x")
        result = coord + 1.0
        expected = DenseCoordinate([2.0, 3.0, 4.0], "x")
        assert result.equals(expected)

    def test_get_indexer_missing(self):
        with pytest.raises(KeyError):
            DenseCoordinate([1, 2, 3])._get_indexer(99)

    def test_to_dataset(self):
        coord = DenseCoordinate([1.0, 2.0, 3.0], "x")
        coord.dim = "x"
        import xdas as xd

        da = xd.DataArray([0, 0, 0], {"x": coord})
        dataset = xr.Dataset()
        dataset, attrs = da.coords["x"]._to_dataset(dataset, {})
        assert "x" in dataset.coords

    def test_to_dataset_no_name(self):
        coord = DenseCoordinate([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="no name"):
            coord._to_dataset(xr.Dataset(), {})

    def test_collect_from_dataset_object_dtype(self):
        coord = DenseCoordinate([1.0, 2.0, 3.0], "x")
        import xdas as xd

        da = xd.DataArray([0, 0, 0], {"x": coord})
        dataset = xr.Dataset()
        dataset, _ = da.coords["x"]._to_dataset(dataset, {})
        dataset["x"] = dataset["x"].astype(object)
        result = DenseCoordinate._collect_from_dataset(dataset, "x")
        assert "x" in result
