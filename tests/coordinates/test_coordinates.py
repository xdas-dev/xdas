import warnings

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

import xdas as xd
from xdas.coordinates import (
    AxisCoordinate,
    DenseCoordinate,
    InterpCoordinate,
    ScalarCoordinate,
)
from xdas.coordinates.core import format_datetime


class TestCoordinate:
    def test_new(self):
        assert isinstance(xd.Coordinate(1), ScalarCoordinate)
        assert not isinstance(xd.Coordinate(1), AxisCoordinate)
        coord = xd.Coordinate(xd.Coordinate([1]), "dim")
        assert coord.dim == "dim"
        assert isinstance(coord, AxisCoordinate)

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
        assert np.allclose(coord.tie_indices, [0, 8])
        assert np.allclose(coord.tie_values, [100.0, 900.0])
        assert coords.isdim("dim")
        coords = xd.Coordinates({"dim": [1.0, 2.0, 3.0]})
        coord = coords["dim"]
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
        assert coords == {}
        assert coords.dims == ()

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

    def test_init_from_coordinates(self):
        original = xd.Coordinates({"dim": [1.0, 2.0, 3.0]})
        copy = xd.Coordinates(original)
        assert copy.dims == original.dims
        assert copy.equals(original)

    def test_getitem_dim_without_coord(self):
        coords = xd.Coordinates(dims=("dim",))
        with pytest.raises(KeyError, match="has no coordinate"):
            coords["dim"]

    def test_repr(self):
        coords = xd.Coordinates(
            {
                "dim": [1.0, 2.0, 3.0],
                "meta": 0,
                "other": ("dim", [4.0, 5.0, 6.0]),
            }
        )
        r = repr(coords)
        assert "Coordinates:" in r
        assert "* dim" in r
        assert "meta" in r
        assert "other (dim)" in r

    def test_reduce(self):
        import pickle

        coords = xd.Coordinates({"dim": [1.0, 2.0, 3.0]})
        restored = pickle.loads(pickle.dumps(coords))
        assert restored.equals(coords)

    def test_parent(self):
        da = xd.DataArray(np.ones(3), {"dim": [0.0, 1.0, 2.0]})
        assert da.coords.parent is da

    def test_get_query_first_last(self):
        coords = xd.Coordinates({"dim_0": [1.0, 2.0, 3.0], "dim_1": [1.0, 2.0, 3.0]})
        q = coords._get_query({"first": slice(0, 1)})
        assert q["dim_0"] == slice(0, 1)
        assert q["dim_1"] == slice(None)
        q = coords._get_query({"last": slice(1, 2)})
        assert q["dim_0"] == slice(None)
        assert q["dim_1"] == slice(1, 2)

    def test_get_query_tuple(self):
        coords = xd.Coordinates({"dim_0": [1.0, 2.0, 3.0], "dim_1": [1.0, 2.0, 3.0]})
        q = coords._get_query((slice(0, 1), slice(1, 2)))
        assert q["dim_0"] == slice(0, 1)
        assert q["dim_1"] == slice(1, 2)

    def test_get_query_else_and_return(self):
        coords = xd.Coordinates({"dim_0": [1.0, 2.0, 3.0], "dim_1": [1.0, 2.0, 3.0]})
        q = coords._get_query(slice(0, 1))
        assert q["dim_0"] == slice(0, 1)
        assert q["dim_1"] == slice(None)

    def test_to_index(self):
        coords = xd.Coordinates({"dim": [1.0, 2.0, 3.0]})
        idx = coords.to_index(2.0)
        assert idx == {"dim": 1}

    def test_equals_different_names(self):
        assert not xd.Coordinates({"dim": [1.0, 2.0, 3.0]}).equals(
            xd.Coordinates({"other": [1.0, 2.0, 3.0]})
        )

    def test_equals_different_values(self):
        assert not xd.Coordinates({"dim": [1.0, 2.0, 3.0]}).equals(
            xd.Coordinates({"dim": [4.0, 5.0, 6.0]})
        )

    def test_copy(self):
        coords = xd.Coordinates({"dim": [1.0, 2.0, 3.0]})
        copy = coords.copy()
        assert copy.equals(coords)
        assert copy is not coords

    def test_setitem_with_parent(self):
        class FakeParent:
            ndim = 1
            shape = (3,)
            sizes = {"dim": 3}

        coords = xd.Coordinates({"dim": [1.0, 2.0, 3.0]})
        parent = FakeParent()
        coords._assign_parent(parent)
        with pytest.raises(KeyError, match="cannot add new dimension"):
            coords["other_dim"] = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="conflicting sizes"):
            coords["dim"] = [1.0, 2.0, 3.0, 4.0]
        # scalar coord: coord.dim is None → skips the dim check block
        coords["meta"] = 42
        # correctly-sized coord: sizes match → no error
        coords["dim"] = [4.0, 5.0, 6.0]

    def test_assign_parent_ndim_mismatch(self):
        class FakeParent:
            ndim = 1
            shape = (3,)
            sizes = {"dim_0": 3}

        coords = xd.Coordinates({"dim_0": [1.0, 2.0, 3.0], "dim_1": [4.0, 5.0, 6.0]})
        with pytest.raises(ValueError, match="number of dimensions"):
            coords._assign_parent(FakeParent())

    def test_assign_parent_size_mismatch(self):
        class FakeParent:
            ndim = 1
            shape = (3,)
            sizes = {"dim": 3}

        coords = xd.Coordinates({"dim": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(ValueError, match="conflicting sizes"):
            coords._assign_parent(FakeParent())


class TestCoordinateBase:
    def test_new_unparseable(self):
        with pytest.raises(TypeError, match="could not parse"):
            xd.Coordinate(object())

    def test_sub(self):
        coord = DenseCoordinate([1.0, 2.0, 3.0], "x")
        result = coord - 1.0
        expected = DenseCoordinate([0.0, 1.0, 2.0], "x")
        assert result.equals(expected)

    def test_deprecated_type_queries_default_false(self):
        coord = ScalarCoordinate(1)
        for name in ("isdense", "isinterp", "issampled"):
            with pytest.warns(FutureWarning, match=name):
                assert getattr(coord, name)() is False

    def test_get_value_deprecated(self):
        coord = DenseCoordinate([1.0, 2.0, 3.0], "x")
        with pytest.warns(FutureWarning, match="get_value"):
            assert coord.get_value(1) == 2.0

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
        assert coord.get_sampling_interval(cast=True) is None
        assert not coord.isregular()
        assert coord.to_regular().get_sampling_interval(cast=True) == 10.0

    def test_format_index_non_integer(self):
        coord = DenseCoordinate([1, 2, 3], "x")
        with pytest.raises(IndexError, match="only integer"):
            coord._format_index(1.5)

    def test_format_index_clip(self):
        coord = DenseCoordinate([1, 2, 3], "x")
        result = coord._format_index(np.array([-1, 0, 5]), bounds="clip")
        assert np.all(result >= 0)

    def test_to_dataset_no_name(self):
        sc = ScalarCoordinate(42)
        with pytest.raises(ValueError, match="no name"):
            sc._to_dataset(xr.Dataset(), {})

    def test_scalar_array_copy(self):
        sc = ScalarCoordinate(42)
        result = np.array(sc, copy=True)
        assert result == 42

    def test_parse_dim_override(self):
        coord = xd.Coordinate(("x", [1, 2, 3]), dim="y")
        assert coord.dim == "y"

    def test_get_discontinuities_empty(self):
        coord = InterpCoordinate()
        df = coord.get_discontinuities()
        assert df.empty

    def test_get_discontinuities_values(self):
        # a 5.0 gap then a 2.0 overlap, on a sampling interval of 1.0: the
        # boundary straddles the last sample of one segment and the first of
        # the next, and `delta` is the jump between them, one sampling interval
        # less than that step
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 4, 5, 9, 10, 14],
                "tie_values": [0.0, 4.0, 10.0, 14.0, 13.0, 17.0],
            }
        )
        df = coord.get_discontinuities()
        npt.assert_array_equal(df["start_index"], [4, 9])
        npt.assert_array_equal(df["end_index"], [5, 10])
        npt.assert_array_equal(df["start_value"], [4.0, 14.0])
        npt.assert_array_equal(df["end_value"], [10.0, 13.0])
        npt.assert_array_equal(df["delta"], [5.0, -2.0])
        npt.assert_array_equal(df["type"], ["gap", "overlap"])

    def test_get_discontinuities_subsample_overlap_is_an_overlap(self):
        # the axis still moves forward, by 0.4 of a 1.0 sampling interval, so
        # the step is positive while the jump is not: obspy calls this an
        # overlap and so do we
        coord = InterpCoordinate(
            {"tie_indices": [0, 4, 5, 9], "tie_values": [0.0, 4.0, 4.4, 8.4]}
        )
        df = coord.get_discontinuities()
        npt.assert_array_equal(df["start_value"], [4.0])
        npt.assert_array_equal(df["end_value"], [4.4])
        npt.assert_allclose(df["delta"], [-0.6])
        npt.assert_array_equal(df["type"], ["overlap"])
        npt.assert_array_equal(coord.get_split_indices("overlaps"), df["end_index"])

    def test_get_discontinuities_datetime(self):
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 4, 5, 9],
                "tie_values": np.array(
                    ["2024-01-01T00:00:00", "2024-01-01T00:00:04"]
                    + ["2024-01-01T00:00:10", "2024-01-01T00:00:14"],
                    dtype="datetime64[ms]",
                ),
            }
        )
        df = coord.get_discontinuities()
        npt.assert_array_equal(df["start_index"], [4])
        npt.assert_array_equal(df["delta"], [np.timedelta64(5, "s")])
        npt.assert_array_equal(df["type"], ["gap"])

    def test_get_discontinuities_tolerance(self):
        # `tolerance` filters the very quantity reported as `delta`: the jump,
        # 5.0 and -2.0 here
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 4, 5, 9, 10, 14],
                "tie_values": [0.0, 4.0, 10.0, 14.0, 13.0, 17.0],
            }
        )
        npt.assert_array_equal(coord.get_discontinuities()["type"], ["gap", "overlap"])
        npt.assert_array_equal(
            coord.get_discontinuities(tolerance=1.5)["type"], ["gap", "overlap"]
        )
        npt.assert_array_equal(
            coord.get_discontinuities(tolerance=3.0)["type"], ["gap"]
        )
        assert len(coord.get_discontinuities(tolerance=5.5)) == 0

    def test_get_availabilities_empty(self):
        coord = InterpCoordinate()
        df = coord.get_availabilities()
        assert df.empty

    def test_format_index_no_bounds(self):
        coord = DenseCoordinate([1, 2, 3], "x")
        result = coord._format_index(np.array([0, 1, 2]), bounds=None)
        assert np.array_equal(result, [0, 1, 2])

    def test_init_subclass_no_name(self):
        from xdas.coordinates import Coordinate

        class _Unnamed(Coordinate):
            pass

        assert "_Unnamed" not in Coordinate._registry

    def test_init_subclass_with_name(self):
        from xdas.coordinates import Coordinate

        class _Named(Coordinate, ctype="_testnamed"):
            pass

        assert "_testnamed" in Coordinate._registry
        del Coordinate._registry["_testnamed"]

    def test_coordinate_copy_deep(self):
        coord = DenseCoordinate([1.0, 2.0, 3.0], "x")
        deep = coord.copy(deep=True)
        assert deep.equals(coord)
        assert deep.data is not coord.data

    def test_coordinate_copy_shallow(self):
        coord = DenseCoordinate([1.0, 2.0, 3.0], "x")
        shallow = coord.copy(deep=False)
        assert shallow.equals(coord)

    def test_get_sampling_interval_helper(self):
        from xdas.coordinates import get_sampling_interval

        da = xd.DataArray([1, 2, 3], {"x": [10.0, 20.0, 30.0]})
        with pytest.warns(FutureWarning, match="implicit inference is deprecated"):
            assert get_sampling_interval(da, "x") == 10.0
        da["x"] = da["x"].to_regular()
        assert get_sampling_interval(da, "x") == 10.0

    def test_get_sampling_interval_helper_jittery_dense_raises(self):
        from xdas.coordinates import get_sampling_interval

        da = xd.DataArray([1, 2, 3], {"x": [0.0, 1.0, 5.0]})
        with pytest.raises(ValueError, match="none could be inferred"):
            get_sampling_interval(da, "x")

    def test_get_sampling_interval_helper_single_sample_raises(self):
        from xdas.coordinates import SampledCoordinate, get_sampling_interval

        coord = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [1], "sampling_interval": 5.0}, "x"
        )
        da = xd.DataArray([1], {"x": coord})
        with pytest.raises(ValueError, match="none could be inferred"):
            get_sampling_interval(da, "x")

    def test_get_sampling_interval_helper_regular(self):
        from xdas.coordinates import SampledCoordinate, get_sampling_interval

        coord = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 5.0}
        )
        da = xd.DataArray([1, 2, 3], {"x": coord})
        assert get_sampling_interval(da, "x") == 5.0

    def test_format_datetime_no_fractional(self):
        x = np.datetime64("2000-01-01T00:00:00", "s")
        assert format_datetime(x) == "2000-01-01T00:00:00"

    def test_format_datetime_truncates_sub_ms(self):
        x = np.datetime64("2000-01-01T00:00:00.123456789", "ns")
        result = format_datetime(x)
        assert result == "2000-01-01T00:00:00.123"

    def test_drop_dims_variadic_first_last(self):
        coords = xd.Coordinates(
            {
                "dim_0": [1.0, 2.0, 3.0],
                "dim_1": [4.0, 5.0, 6.0],
                "dim_2": [7.0, 8.0, 9.0],
            }
        )
        result = coords.drop_dims("first", "last")
        assert list(result.dims) == ["dim_1"]

    def test_drop_coords_variadic_first_last(self):
        coords = xd.Coordinates(
            {
                "dim_0": [1.0, 2.0, 3.0],
                "dim_1": [4.0, 5.0, 6.0],
                "dim_2": [7.0, 8.0, 9.0],
            }
        )
        result = coords.drop_coords("first", "last")
        assert "dim_0" not in result
        assert "dim_2" not in result
        assert "dim_1" in result

    def test_reduce(self):
        import pickle

        coord = DenseCoordinate([1.0, 2.0, 3.0], "x")
        restored = pickle.loads(pickle.dumps(coord))
        assert restored.equals(coord)

    def test_equals_returns_false_different_values(self):
        c1 = DenseCoordinate([1.0, 2.0, 3.0], "x")
        c2 = DenseCoordinate([4.0, 5.0, 6.0], "x")
        assert not c1.equals(c2)

    def test_slice_indexer_endpoint_false(self):
        coord = DenseCoordinate([1.0, 2.0, 3.0], "x")
        slc = coord._slice_indexer(stop=3.0, endpoint=False)
        assert slc == slice(None, 2)


class TestEncodeDelta:
    def test_generic_timedelta_promoted_to_ns(self):
        from xdas.coordinates.core import decode_delta, encode_delta

        attrs = encode_delta("tolerance", np.timedelta64(0))
        assert attrs == {
            "tolerance": 0,
            "tolerance_units": "nanoseconds",
            "tolerance_dtype": "timedelta64[ns]",
        }
        assert decode_delta("tolerance", attrs) == np.timedelta64(0, "ns")

    def test_none_is_omitted(self):
        from xdas.coordinates.core import encode_delta

        assert encode_delta("tolerance", None) == {}


class TestGetSamplingIntervalHelperNonAxis:
    def test_scalar_coordinate_returns_none(self):
        from xdas.coordinates import get_sampling_interval

        da = xd.DataArray(np.zeros(3), {"x": [0.0, 1.0, 2.0], "meta": 0})
        assert get_sampling_interval(da, "meta") is None


class TestReversals:
    """A reversal is an overlap of a full sampling interval or more."""

    def test_a_subsample_overlap_is_not_a_reversal(self):
        # the axis advances by 0.4 of a 1.0 sampling interval: an overlap that
        # leaves the order intact, so there is nothing to cut
        coord = InterpCoordinate(
            {"tie_indices": [0, 4, 5, 9], "tie_values": [0.0, 4.0, 4.4, 8.4]}
        )
        npt.assert_array_equal(coord.get_split_indices("overlaps"), [5])
        npt.assert_array_equal(coord.get_split_indices("reversals"), [])

    def test_a_full_sample_overlap_is_a_reversal(self):
        # 0.5 of a sampling interval short of a whole one: the axis stops
        coord = InterpCoordinate(
            {"tie_indices": [0, 4, 5, 9], "tie_values": [0.0, 4.0, 3.5, 7.5]}
        )
        npt.assert_array_equal(coord.get_split_indices("reversals"), [5])
        assert not coord._is_monotonic_increasing()

    def test_repeated_values_are_a_reversal(self):
        # an axis that stands still is not strictly increasing either, and the
        # default threshold is exact rather than the epsilon used on jumps
        coord = DenseCoordinate([10.0, 20.0, 20.0, 30.0])
        npt.assert_array_equal(coord.get_split_indices("reversals"), [2])

    def test_reversals_are_the_boundaries_monotonicity_trips_on(self):
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 4, 5, 9, 10, 14],
                "tie_values": [0.0, 4.0, 10.0, 14.0, 13.0, 17.0],
            }
        )
        # a gap, then an overlap of two sampling intervals
        npt.assert_array_equal(coord.get_split_indices("discontinuities"), [5, 10])
        npt.assert_array_equal(coord.get_split_indices("reversals"), [10])
        assert not coord._is_monotonic_increasing()
        for chunk in xd.split(
            xd.DataArray(np.arange(15.0), {"dim_0": coord}), "reversals"
        ):
            assert chunk["dim_0"]._is_monotonic_increasing()

    def test_tolerance_drops_the_short_reversals(self):
        coord = DenseCoordinate([0.0, 1.0, 0.5, 1.5, 2.5, 0.5])
        npt.assert_array_equal(coord.get_split_indices("reversals"), [2, 5])
        npt.assert_array_equal(coord.get_split_indices("reversals", 1.0), [5])


class TestOrderedSelectionOnUnsortedAxes:
    @staticmethod
    def dataarray(coord):
        return xd.DataArray(np.arange(len(coord), dtype=float), {"dim_0": coord})

    def test_a_subsample_overlap_slices_without_splitting(self):
        coord = InterpCoordinate(
            {"tie_indices": [0, 4, 5, 9], "tie_values": [0.0, 4.0, 4.4, 8.4]}
        )
        da = self.dataarray(coord)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = da.sel(dim_0=slice(2.0, 6.0))
        npt.assert_array_equal(result.values, [2.0, 3.0, 4.0, 5.0, 6.0])

    def test_a_reversal_splits_and_concatenates(self):
        coord = InterpCoordinate(
            {"tie_indices": [0, 4, 5, 9], "tie_values": [0.0, 4.0, 3.0, 7.0]}
        )
        da = self.dataarray(coord)
        with pytest.warns(match="not monotonic increasing"):
            result = da.sel(dim_0=slice(2.0, 5.0))
        npt.assert_array_equal(result.values, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    def test_an_axis_that_decreases_within_a_segment_is_refused(self):
        # nothing marks the turn, so no splitting can order it: the selection
        # says so instead of recursing on itself forever
        coord = InterpCoordinate(
            {"tie_indices": [0, 4, 8], "tie_values": [0.0, 4.0, 2.0]}
        )
        da = self.dataarray(coord)
        with (
            pytest.warns(match="not monotonic increasing"),
            pytest.raises(ValueError, match="decreases along its axis"),
        ):
            da.sel(dim_0=slice(1.0, 3.0))
