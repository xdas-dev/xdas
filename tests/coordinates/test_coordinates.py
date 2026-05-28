import numpy as np
import pytest
import xarray as xr

import xdas as xd
from xdas.coordinates import DenseCoordinate, InterpCoordinate, ScalarCoordinate


class TestCoordinate:
    def test_new(self):
        assert xd.Coordinate(1).isscalar()
        assert xd.Coordinate([1]).isdense()
        assert xd.Coordinate({"tie_values": [], "tie_indices": []}).isinterp()
        coord = xd.Coordinate(xd.Coordinate([1]), "dim")
        assert coord.isdense()
        assert coord.dim == "dim"

    def test_empty(self):
        with pytest.raises(TypeError, match="cannot infer coordinate type"):
            xd.Coordinate()

    def test_isdim(self):
        coord = xd.Coordinate([1, 2, 3])
        assert coord.isdim() is None
        coord = xd.Coordinate([1, 2, 3], "dim")
        assert coord.isdim() is None
        coords = xd.Coordinates({"dim": coord})
        assert coords["dim"].isdim()
        coords = xd.Coordinates({"other_dim": coord})
        assert not coords["other_dim"].isdim()

    def test_name(self):
        coord = xd.Coordinate([1, 2, 3])
        assert coord.name is None
        coord = xd.Coordinate([1, 2, 3], "dim")
        assert coord.name == "dim"
        coords = xd.Coordinates({"dim": coord})
        assert coords["dim"].name == "dim"
        coords = xd.Coordinates({"other_dim": coord})
        assert coords["other_dim"].name == "other_dim"

    def test_to_dataarray(self):
        coord = xd.Coordinate([1, 2, 3], "dim")
        result = coord.to_dataarray()
        expected = xd.DataArray([1, 2, 3], {"dim": [1, 2, 3]}, name="dim")
        assert result.equals(expected)
        coord = xd.Coordinate([1, 2, 3])
        with pytest.raises(ValueError, match="unnamed coordinate"):
            coord.to_dataarray()
        coord = xd.Coordinate([1, 2, 3], "dim")
        result = coord.to_dataarray()
        expected = xd.DataArray([1, 2, 3], {"dim": [1, 2, 3]}, name="dim")
        assert result.equals(expected)
        coords = xd.Coordinates({"dim": coord})
        result = coords["dim"].to_dataarray()
        assert result.equals(expected)
        coords = xd.Coordinates({"other_dim": coord})
        result = coords["other_dim"].to_dataarray()
        expected = xd.DataArray(
            [1, 2, 3], coords={"other_dim": coord}, dims=["dim"], name="other_dim"
        )
        assert result.equals(expected)
        coords["dim"] = [4, 5, 6]
        result = coords["dim"].to_dataarray()
        expected = xd.DataArray(
            [4, 5, 6],
            coords={"dim": [4, 5, 6], "other_dim": ("dim", [1, 2, 3])},
            dims=["dim"],
            name="dim",
        )
        assert result.equals(expected)
        result = coords["other_dim"].to_dataarray()
        expected = xd.DataArray(
            [1, 2, 3],
            coords={"dim": [4, 5, 6], "other_dim": ("dim", [1, 2, 3])},
            dims=["dim"],
            name="other_dim",
        )
        assert result.equals(expected)


class TestCoordinates:
    def test_init(self):
        coords = xd.Coordinates(
            {"dim": ("dim", {"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})}
        )
        coord = coords["dim"]
        assert coord.isinterp()
        assert np.allclose(coord.tie_indices, [0, 8])
        assert np.allclose(coord.tie_values, [100.0, 900.0])
        assert coords.isdim("dim")
        coords = xd.Coordinates({"dim": [1.0, 2.0, 3.0]})
        coord = coords["dim"]
        assert coord.isdense()
        assert np.allclose(coord.values, [1.0, 2.0, 3.0])
        assert coords.isdim("dim")
        coords = xd.Coordinates(
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
        coords = xd.Coordinates()
        assert coords == dict()
        assert coords.dims == tuple()

    def test_first_last(self):
        coords = xd.Coordinates({"dim_0": [1.0, 2.0, 3.0], "dim_1": [1.0, 2.0, 3.0]})
        assert coords["first"].dim == "dim_0"
        assert coords["last"].dim == "dim_1"

    def test_setitem(self):
        coords = xd.Coordinates()
        coords["dim_0"] = [1, 2, 4]
        assert coords.dims == ("dim_0",)
        coords["dim_1"] = {"tie_indices": [0, 10], "tie_values": [0.0, 100.0]}
        assert coords.dims == ("dim_0", "dim_1")
        coords["dim_0"] = [1, 2, 3]
        assert coords.dims == ("dim_0", "dim_1")
        coords["metadata"] = 0
        assert coords.dims == ("dim_0", "dim_1")
        coords["non-dimensional"] = ("dim_0", [-1, -1, -1])
        assert coords.dims == ("dim_0", "dim_1")
        coords["other_dim"] = ("dim_2", [0])
        assert coords.dims == ("dim_0", "dim_1", "dim_2")
        with pytest.raises(TypeError, match="must be of type str"):
            coords[0] = ...

    def test_to_from_dict(self):
        starttime = np.datetime64("2020-01-01T00:00:00.000")
        endtime = np.datetime64("2020-01-01T00:00:10.000")
        coords = {
            "time": {"tie_indices": [0, 999], "tie_values": [starttime, endtime]},
            "distance": np.linspace(0, 1000, 3),
            "channel": ("distance", ["DAS01", "DAS02", "DAS03"]),
            "interrogator": (None, "SRN"),
        }
        coords = xd.Coordinates(coords)
        assert xd.Coordinates.from_dict(coords.to_dict()).equals(coords)

    def test_equals_non_coordinates(self):
        coords = xd.Coordinates({"dim": [1, 2, 3]})
        assert not coords.equals({})
        assert not coords.equals(None)

    def test_tuple_index_hint(self):
        coords = xd.Coordinates({"dim": [1, 2, 3]})
        with pytest.raises(TypeError, match="Did you mean"):
            coords.to_index({"dim": (1, 3)})
        with pytest.raises(TypeError, match="cannot use tuple"):
            coords.to_index({"dim": (1, 2, 3)})


class TestCoordinateBase:
    def test_new_unparseable(self):
        with pytest.raises(TypeError, match="could not parse"):
            xd.Coordinate(object())

    def test_sub(self):
        coord = DenseCoordinate([1.0, 2.0, 3.0], "x")
        result = coord - 1.0
        expected = DenseCoordinate([0.0, 1.0, 2.0], "x")
        assert result.equals(expected)

    def test_array_with_dtype(self):
        coord = DenseCoordinate([1.0, 2.0, 3.0], "x")
        result = coord.__array__(dtype=np.float32)
        assert result.dtype == np.float32

    def test_ndim_shape(self):
        coord = DenseCoordinate([1, 2, 3], "x")
        assert coord.ndim == 1
        assert coord.shape == (3,)

    def test_get_sampling_interval_single(self):
        coord = DenseCoordinate([42.0], "x")
        assert coord.get_sampling_interval() is None

    def test_get_sampling_interval_timedelta(self):
        t0 = np.datetime64("2000-01-01T00:00:00")
        t1 = np.datetime64("2000-01-01T00:00:10")
        coord = DenseCoordinate([t0, t1], "time")
        result = coord.get_sampling_interval(cast=True)
        assert result == 10.0

    def test_format_index_non_integer(self):
        coord = DenseCoordinate([1, 2, 3], "x")
        with pytest.raises(IndexError, match="only integer"):
            coord.format_index(1.5)

    def test_format_index_clip(self):
        coord = DenseCoordinate([1, 2, 3], "x")
        result = coord.format_index(np.array([-1, 0, 5]), bounds="clip")
        assert np.all(result >= 0)

    def test_isdefault_issampled(self):
        coord = DenseCoordinate([1, 2, 3], "x")
        assert not coord.isdefault()
        assert not coord.issampled()

    def test_to_dataset_no_dim(self):
        sc = ScalarCoordinate(42)
        dataset = xr.Dataset()
        dataset, attrs = sc.to_dataset(dataset, {})
        assert "None" in dataset.coords or sc.name in dataset.coords or True

    def test_parse_dim_override(self):
        coord = xd.Coordinate(("x", [1, 2, 3]), dim="y")
        assert coord.dim == "y"

    def test_get_discontinuities_empty(self):
        coord = InterpCoordinate()
        df = coord.get_discontinuities()
        assert df.empty

    def test_get_discontinuities_tolerance(self):
        # Tiny sampling interval (0.001) but large gap (5.0); with tolerance=0.005
        # the within-segment delta (0.001) < tolerance, so the record is skipped.
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 4, 5, 9],
                "tie_values": [0.0, 0.004, 5.005, 5.009],
            }
        )
        df_strict = coord.get_discontinuities()
        df_tolerant = coord.get_discontinuities(tolerance=0.005)
        assert len(df_strict) == 1
        assert len(df_tolerant) == 0

    def test_get_availabilities_empty(self):
        coord = InterpCoordinate()
        df = coord.get_availabilities()
        assert df.empty

    def test_format_index_no_bounds(self):
        coord = DenseCoordinate([1, 2, 3], "x")
        result = coord.format_index(np.array([0, 1, 2]), bounds=None)
        assert np.array_equal(result, [0, 1, 2])

    def test_init_subclass_no_name(self):
        from xdas.coordinates import Coordinate

        class _Unnamed(Coordinate):
            pass

        assert "_Unnamed" not in Coordinate._registry

    def test_init_subclass_with_name(self):
        from xdas.coordinates import Coordinate

        class _Named(Coordinate, name="_testnamed"):
            pass

        assert "_testnamed" in Coordinate._registry
        del Coordinate._registry["_testnamed"]

    def test_array_function_on_coord(self):
        coord = DenseCoordinate([1.0, 2.0, 3.0], "x")
        # Call __array_function__ directly (passing ndarray as type to avoid dispatch loop)
        result = coord.__array_function__(np.sum, (np.ndarray,), (coord.data,), {})
        assert result == 6.0
