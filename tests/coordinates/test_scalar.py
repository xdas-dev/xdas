import numpy as np
import pytest

from xdas.coordinates import ScalarCoordinate


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
            assert ScalarCoordinate._isvalid(data)
        for data in self.invalid:
            assert not ScalarCoordinate._isvalid(data)

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
        with pytest.raises(TypeError):
            ScalarCoordinate(1)[...]
        with pytest.raises(TypeError):
            ScalarCoordinate(1)[:]
        with pytest.raises(TypeError):
            ScalarCoordinate(1)[0]

    def test_len(self):
        assert len(ScalarCoordinate(1)) == 1

    def test_repr(self):
        for data in self.valid:
            assert ScalarCoordinate(data).__repr__() == np.array2string(
                np.asarray(data), threshold=0, edgeitems=1
            )

    def test_array(self):
        for data in self.valid:
            assert ScalarCoordinate(data).__array__() == np.array(data)

    def test_array_dtype(self):
        coord = ScalarCoordinate(1)
        arr = coord.__array__(np.float64)
        assert arr.dtype == np.float64
        assert arr == np.array(1.0)

    def test_dtype(self):
        for data in self.valid:
            assert ScalarCoordinate(data).dtype == np.array(data).dtype

    def test_ndim_shape_size(self):
        coord = ScalarCoordinate(1)
        assert coord.ndim == 0
        assert coord.shape == ()
        assert coord.size == 1

    def test_values(self):
        for data in self.valid:
            assert ScalarCoordinate(data).values == np.array(data)

    def test_dim_setter(self):
        coord = ScalarCoordinate(1)
        coord.dim = None  # allowed
        with pytest.raises(ValueError):
            coord.dim = "x"

    def test_equals(self):
        for data in self.valid:
            coord = ScalarCoordinate(data)
            assert coord.equals(coord)
        assert ScalarCoordinate(1).equals(ScalarCoordinate(np.array(1)))
        assert not ScalarCoordinate(1).equals(42)

    def test_to_index(self):
        with pytest.raises(NotImplementedError):
            ScalarCoordinate(1).to_index("item")

    def test_is_monotonic_increasing(self):
        with pytest.raises(TypeError):
            ScalarCoordinate(1)._is_monotonic_increasing()

    def test_concat(self):
        with pytest.raises(TypeError):
            ScalarCoordinate(1)._concat(ScalarCoordinate(2))

    def test_from_block(self):
        with pytest.raises(TypeError):
            ScalarCoordinate.from_block(0, 5, 1)

    def test_empty(self):
        with pytest.raises(TypeError, match="cannot be empty"):
            ScalarCoordinate()
