import numpy as np
import pytest

from xdas.coordinates import Coordinate
from xdas.coordinates.default import DefaultCoordinate


class TestDefaultCoordinate:
    def test_isvalid(self):
        assert DefaultCoordinate.isvalid({"size": 5})
        assert DefaultCoordinate.isvalid({"size": None})
        assert not DefaultCoordinate.isvalid({"size": 1.5})
        assert not DefaultCoordinate.isvalid({"length": 5})
        assert not DefaultCoordinate.isvalid([1, 2, 3])
        assert not DefaultCoordinate.isvalid(5)

    def test_init_default(self):
        coord = DefaultCoordinate()
        assert coord.empty
        assert len(coord) == 0

    def test_init_with_size(self):
        coord = DefaultCoordinate({"size": 5}, "x")
        assert not coord.empty
        assert len(coord) == 5
        assert coord.dim == "x"

    def test_init_invalid_data(self):
        with pytest.raises(TypeError, match="must be a mapping"):
            DefaultCoordinate([1, 2, 3])

    def test_init_dtype_rejected(self):
        with pytest.raises(ValueError, match="dtype"):
            DefaultCoordinate({"size": 3}, dtype=np.int32)

    def test_empty_property(self):
        assert DefaultCoordinate({"size": 0}).empty
        assert not DefaultCoordinate({"size": 1}).empty

    def test_dtype(self):
        assert DefaultCoordinate({"size": 3}).dtype == np.int64

    def test_ndim(self):
        assert DefaultCoordinate({"size": 3}).ndim == 1

    def test_shape(self):
        assert DefaultCoordinate({"size": 5}).shape == (5,)

    def test_len_with_none(self):
        coord = DefaultCoordinate({"size": None})
        assert len(coord) == 0

    def test_len_with_size(self):
        assert len(DefaultCoordinate({"size": 7})) == 7

    def test_getitem_scalar(self):
        coord = DefaultCoordinate({"size": 5}, "x")
        result = coord[2]
        assert isinstance(result, Coordinate)
        assert result.dim is None  # scalar → no dim

    def test_getitem_slice(self):
        coord = DefaultCoordinate({"size": 5}, "x")
        result = coord[1:3]
        assert len(result) == 2
        assert result.dim == "x"

    def test_array(self):
        coord = DefaultCoordinate({"size": 4})
        arr = np.asarray(coord)
        np.testing.assert_array_equal(arr, np.arange(4))

    def test_isdefault(self):
        assert DefaultCoordinate({"size": 3}).isdefault()

    def test_get_sampling_interval(self):
        assert DefaultCoordinate({"size": 3}).get_sampling_interval() == 1

    def test_equals_same(self):
        assert DefaultCoordinate({"size": 3}).equals(DefaultCoordinate({"size": 3}))

    def test_equals_different_size(self):
        assert not DefaultCoordinate({"size": 3}).equals(DefaultCoordinate({"size": 5}))

    def test_equals_wrong_type(self):
        from xdas.coordinates import DenseCoordinate

        result = DefaultCoordinate({"size": 3}).equals(
            DenseCoordinate(np.arange(3), "x")
        )
        assert result is None

    def test_get_indexer(self):
        coord = DefaultCoordinate({"size": 5})
        assert coord.get_indexer(3) == 3

    def test_slice_indexer(self):
        coord = DefaultCoordinate({"size": 5})
        s = coord.slice_indexer(1, 4, 2)
        assert s == slice(1, 4, 2)

    def test_concat(self):
        a = DefaultCoordinate({"size": 3}, "x")
        b = DefaultCoordinate({"size": 2}, "x")
        c = a.concat(b)
        assert len(c) == 5
        assert c.dim == "x"

    def test_concat_type_error(self):
        from xdas.coordinates import DenseCoordinate

        a = DefaultCoordinate({"size": 3}, "x")
        b = DenseCoordinate(np.array([0, 1, 2]), "x")
        with pytest.raises(TypeError):
            a.concat(b)

    def test_concat_dim_mismatch(self):
        a = DefaultCoordinate({"size": 3}, "x")
        b = DefaultCoordinate({"size": 2}, "y")
        with pytest.raises(ValueError):
            a.concat(b)

    def test_to_from_dict(self):
        coord = DefaultCoordinate({"size": 5}, "x")
        dct = coord.to_dict()
        assert dct["dim"] == "x"
        assert dct["data"] == {"size": 5}
        restored = DefaultCoordinate.from_dict(dct)
        assert restored.equals(coord)
        assert restored.dim == coord.dim
