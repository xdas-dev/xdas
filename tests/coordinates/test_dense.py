import numpy as np
import pandas as pd
import pytest

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

    def test_to_from_dict(self):
        for data in self.valid:
            coord = DenseCoordinate(data)
            assert DenseCoordinate.from_dict(coord.to_dict()).equals(coord)

    def test_empty(self):
        coord = DenseCoordinate()
        assert coord.empty

    def test_append(self):
        coord0 = DenseCoordinate()
        coord1 = DenseCoordinate([1, 2, 3])
        coord2 = DenseCoordinate([4, 5, 6])

        result = coord1.append(coord2)
        expected = DenseCoordinate([1, 2, 3, 4, 5, 6])
        assert result.equals(expected)

        result = coord2.append(coord1)
        expected = DenseCoordinate([4, 5, 6, 1, 2, 3])
        assert result.equals(expected)

        assert coord0.append(coord0).empty
        assert coord0.append(coord1).equals(coord1)
        assert coord1.append(coord0).equals(coord1)
