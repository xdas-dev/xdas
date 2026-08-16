import numpy as np
import pytest
import xarray as xr

import xdas as xd
from xdas.coordinates import AxisCoordinate, ScalarCoordinate


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

    def test_not_axis_coordinate(self):
        # a scalar coordinate is not an axis coordinate and carries no axis API
        coord = ScalarCoordinate(1)
        assert not isinstance(coord, AxisCoordinate)
        assert not hasattr(coord, "from_block")
        assert not hasattr(coord, "_get_value")
        assert not hasattr(coord, "to_index")

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

    def test_empty(self):
        with pytest.raises(TypeError, match="cannot be empty"):
            ScalarCoordinate()

    def test_to_dataset_with_name(self):
        da = xd.DataArray([1, 2, 3], {"x": [1.0, 2.0, 3.0], "meta": 42})
        sc = da.coords["meta"]
        dataset = xr.Dataset()
        dataset, _attrs = sc._to_dataset(dataset, {})
        assert "meta" in dataset.coords

    def test_add(self):
        result = ScalarCoordinate(1.0) + 1.0
        assert isinstance(result, ScalarCoordinate)
        assert result.data == 2.0

    def test_sub(self):
        result = ScalarCoordinate(2.0) - 1.0
        assert isinstance(result, ScalarCoordinate)
        assert result.data == 1.0

    def test_sub_scalar_coordinate(self):
        t0 = ScalarCoordinate(np.datetime64("2026-01-01T00:00:01"))
        t1 = ScalarCoordinate(np.datetime64("2026-01-01T00:00:02"))
        result = t1 - t0
        assert isinstance(result, ScalarCoordinate)
        assert result.values / np.timedelta64(1, "s") == 1.0


class TestScalarCoordinateRegularity:
    def test_never_regular(self):
        assert not ScalarCoordinate(42).isregular()
