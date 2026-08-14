import numpy as np
import pandas as pd
import pytest
import xarray as xr

import xdas as xd
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
        assert DenseCoordinate([1, 2, 3])._slice_indexer(start=2) == slice(1, None)

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

    def test_get_split_indices(self):
        coord = DenseCoordinate([1, 2, 3, 10, 11, 12])
        # local spacing is 1; only the jump 3->10 stands out as a gap, and the
        # normal step 10->11 right after it must not be reported as an overlap
        np.testing.assert_array_equal(
            coord.get_split_indices("discontinuities", tolerance=3.0), [3]
        )
        np.testing.assert_array_equal(
            coord.get_split_indices("gaps", tolerance=None), [3]
        )
        np.testing.assert_array_equal(
            coord.get_split_indices("overlaps", tolerance=None), []
        )
        # with no tolerance filtering every consecutive pair is a candidate boundary
        np.testing.assert_array_equal(coord.get_split_indices(), [1, 2, 3, 4, 5])

    def test_get_split_indices_rate_change(self):
        # A continuous axis whose sampling rate changes (step 1 then step 2) is
        # not a discontinuity: the baseline follows the new rate, so only the
        # single transition is reported and the sustained run stays clean.
        coord = DenseCoordinate([0, 1, 2, 3, 5, 7, 9])
        np.testing.assert_array_equal(
            coord.get_split_indices("gaps", tolerance=0.5), [4]
        )
        np.testing.assert_array_equal(
            coord.get_split_indices("discontinuities", tolerance=1.5), []
        )

    def test_get_split_indices_leading_gap(self):
        # A discontinuity in the very first step is still detected.
        coord = DenseCoordinate([0, 10, 11, 12])
        np.testing.assert_array_equal(
            coord.get_split_indices("gaps", tolerance=3.0), [1]
        )

    def test_get_split_indices_empty(self):
        coord = DenseCoordinate([])
        np.testing.assert_array_equal(coord.get_split_indices(), [])
        np.testing.assert_array_equal(
            coord.get_split_indices("gaps", tolerance=None), []
        )

    def test_simplify_is_noop(self):
        coord = DenseCoordinate([1, 2, 3, 10, 11, 12], "x")
        result = coord.simplify(tolerance=5.0)
        assert result.equals(coord)
        assert result is not coord

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

    def test_is_monotonic_increasing_duplicates(self):
        # the check is strict: repeated values are not monotonic increasing.
        assert not DenseCoordinate([1, 2, 2, 3])._is_monotonic_increasing()

    def test_is_monotonic_increasing_strings(self):
        # string dtypes have no `subtract` loop, so the check cannot be
        # arithmetic; it must still work on labels.
        assert DenseCoordinate(["N", "P", "S"])._is_monotonic_increasing()
        assert not DenseCoordinate(["P", "N", "S"])._is_monotonic_increasing()
        assert not DenseCoordinate(["N", "P", "P"])._is_monotonic_increasing()

    @pytest.mark.parametrize(
        "values",
        [
            [1, 2, 3],
            [1, 2, 2, 3],
            [3, 2, 1],
            [1],
            [],
            np.array([0, 1, 2], dtype="datetime64[s]"),
            np.array([0, 1, 1, 2], dtype="datetime64[s]"),
            np.array([2, 1, 0], dtype="datetime64[s]"),
        ],
    )
    def test_is_monotonic_increasing_matches_arithmetic(self, values):
        # the pandas-based check must agree with the arithmetic one it replaced
        # wherever the latter was defined.
        coord = DenseCoordinate(values)
        zero = np.timedelta64(0) if np.issubdtype(coord.dtype, np.datetime64) else 0
        expected = bool(np.all(np.diff(coord.values) > zero))
        assert coord._is_monotonic_increasing() == expected

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
        dataset, _attrs = da.coords["x"]._to_dataset(dataset, {})
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


class TestDenseCoordinateStringSelection:
    @staticmethod
    def dataarray():
        return xd.DataArray(
            np.arange(12).reshape(4, 3),
            {"time": [0.0, 1.0, 2.0, 3.0], "phase": ["N", "P", "S"]},
        )

    def test_sel_scalar(self):
        result = self.dataarray().sel(phase="P")
        assert np.array_equal(result.values, [1, 4, 7, 10])
        assert result.coords["phase"].values == "P"

    def test_sel_list(self):
        result = self.dataarray().sel(phase=["P", "S"])
        assert np.array_equal(result.values, [[1, 2], [4, 5], [7, 8], [10, 11]])
        assert np.array_equal(result.coords["phase"].values, ["P", "S"])

    def test_sel_list_reorders(self):
        result = self.dataarray().sel(phase=["S", "P"])
        assert np.array_equal(result.values, [[2, 1], [5, 4], [8, 7], [11, 10]])
        assert np.array_equal(result.coords["phase"].values, ["S", "P"])

    def test_sel_slice(self):
        result = self.dataarray().sel(phase=slice("N", "P"))
        assert np.array_equal(result.values, [[0, 1], [3, 4], [6, 7], [9, 10]])
        assert np.array_equal(result.coords["phase"].values, ["N", "P"])

    def test_sel_missing_label(self):
        with pytest.raises(KeyError):
            self.dataarray().sel(phase="Q")


class TestDenseCoordinateUnsortedSelection:
    """
    Label selection on an axis whose labels are not in sorted order.

    A categorical axis is unordered by nature — a SeisBench phase axis is
    ``"PSN"`` on 14 of the 17 cached ``PhaseNet`` weight sets and ``"NPS"`` on
    the other three — yet every label still designates exactly one position, so
    an exact look-up is well defined. Ordered look-ups (a slice, or ``method``)
    are not, and stay guarded.
    """

    @staticmethod
    def dataarray():
        return xd.DataArray(
            np.arange(12).reshape(4, 3),
            {"time": [0.0, 1.0, 2.0, 3.0], "phase": ["P", "S", "N"]},
        )

    def test_the_axis_is_not_monotonic_increasing(self):
        assert not self.dataarray()["phase"]._is_monotonic_increasing()

    def test_sel_scalar(self):
        result = self.dataarray().sel(phase="P")
        assert np.array_equal(result.values, [0, 3, 6, 9])
        assert result.coords["phase"].values == "P"

    def test_sel_list(self):
        result = self.dataarray().sel(phase=["P", "S"])
        assert np.array_equal(result.values, [[0, 1], [3, 4], [6, 7], [9, 10]])
        assert np.array_equal(result.coords["phase"].values, ["P", "S"])

    def test_sel_list_preserves_the_requested_order(self):
        result = self.dataarray().sel(phase=["S", "P"])
        assert np.array_equal(result.values, [[1, 0], [4, 3], [7, 6], [10, 9]])
        assert np.array_equal(result.coords["phase"].values, ["S", "P"])

    def test_sel_missing_label(self):
        with pytest.raises(KeyError):
            self.dataarray().sel(phase="Q")

    def test_sel_slice_is_still_refused(self):
        # a slice resolves its bounds by searching, so the guard still catches
        # it and sends it down the split-on-reversals path, which cannot order
        # these labels either: the selection fails rather than returning
        # something arbitrary. The `TypeError` is numpy's, from differencing
        # strings while looking for the reversals.
        with (
            pytest.raises(TypeError),
            pytest.warns(match="not monotonic increasing"),
        ):
            self.dataarray().sel(phase=slice("P", "S"))

    def test_sel_with_method_is_still_refused(self):
        # a neighbour search is an ordered look-up too.
        with pytest.raises(NotImplementedError, match="overlaps"):
            self.dataarray().sel(phase="P", method="nearest")


class TestDenseCoordinateUnsortedNumericSelection:
    """Exact selection needs no order on numeric axes either."""

    @staticmethod
    def dataarray():
        return xd.DataArray(np.arange(4), {"x": [30.0, 10.0, 40.0, 20.0]})

    def test_sel_scalar(self):
        assert self.dataarray().sel(x=40.0).values == 2

    def test_sel_list_preserves_the_requested_order(self):
        result = self.dataarray().sel(x=[40.0, 10.0])
        assert np.array_equal(result.values, [2, 1])
        assert np.array_equal(result.coords["x"].values, [40.0, 10.0])

    def test_sel_datetime_scalar(self):
        da = xd.DataArray(
            np.arange(3),
            {"time": np.array(["2000-01-03", "2000-01-01", "2000-01-02"], "M8[s]")},
        )
        assert da.sel(time="2000-01-01").values == 1

    def test_sorted_axes_are_untouched(self):
        da = xd.DataArray(np.arange(4), {"x": [10.0, 20.0, 30.0, 40.0]})
        assert da.sel(x=30.0).values == 2
        assert np.array_equal(da.sel(x=slice(20.0, 30.0)).values, [1, 2])
        assert da.sel(x=24.0, method="nearest").values == 1


class TestDenseCoordinateDuplicatedSelection:
    """
    Duplicated labels are ambiguous, whatever the guard does.

    The ``is_unique`` half of the monotonicity check keeps routing slices
    through the split-on-reversals path — a repeated value is a boundary the
    axis does not advance across — and an exact look-up is refused by
    ``pandas`` because it cannot name a single position.
    """

    @staticmethod
    def dataarray():
        return xd.DataArray(np.arange(4), {"x": [10.0, 20.0, 20.0, 30.0]})

    def test_the_axis_is_not_monotonic_increasing(self):
        assert not self.dataarray()["x"]._is_monotonic_increasing()

    def test_sel_slice_still_takes_the_guarded_path(self):
        with pytest.warns(match="not monotonic increasing"):
            self.dataarray().sel(x=slice(10.0, 30.0))

    def test_sel_scalar_is_refused(self):
        with pytest.raises(pd.errors.InvalidIndexError):
            self.dataarray().sel(x=20.0)


class TestDenseCoordinateToRegular:
    def test_never_regular(self):
        coord = DenseCoordinate([0.0, 1.0, 2.0], "x")
        assert coord.get_sampling_interval() is None
        assert not coord.isregular()

    def test_to_regular_uniform(self):
        coord = DenseCoordinate([0.0, 1.0, 2.0, 3.0], "x")
        reg = coord.to_regular()
        assert reg.isregular()
        assert reg.get_sampling_interval() == 1.0
        assert reg.dim == "x"
        np.testing.assert_array_equal(reg.values, coord.values)

    def test_to_regular_explicit_args(self):
        coord = DenseCoordinate([0.0, 1.05, 2.0], "x")
        reg = coord.to_regular(sampling_interval=1.0, tolerance=0.1)
        assert reg.get_sampling_interval() == 1.0

    def test_to_regular_irregular_raises(self):
        coord = DenseCoordinate([0.0, 1.0, 5.0], "x")
        with pytest.raises(ValueError, match="not evenly spaced"):
            coord.to_regular()

    def test_to_regular_too_short_raises(self):
        with pytest.raises(ValueError, match="fewer than two"):
            DenseCoordinate([1.0], "x").to_regular()

    def test_to_regular_datetime(self):
        t0 = np.datetime64("2000-01-01T00:00:00")
        values = t0 + np.timedelta64(1, "s") * np.arange(5)
        coord = DenseCoordinate(values, "time")
        reg = coord.to_regular()
        assert reg.get_sampling_interval() == 1.0
        np.testing.assert_array_equal(reg.values, coord.values)
