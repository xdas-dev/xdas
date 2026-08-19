import tempfile

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import xdas as xd
from xdas.coordinates import (
    DenseCoordinate,
    SampledCoordinate,
    ScalarCoordinate,
)


class TestSampledCoordinateBasics:
    def test_isvalid(self):
        assert SampledCoordinate._isvalid(
            {"tie_values": [0.0], "tie_lengths": [1], "sampling_interval": 1.0}
        )
        assert SampledCoordinate._isvalid(
            {
                "tie_values": [np.datetime64("2000-01-01T00:00:00")],
                "tie_lengths": [1],
                "sampling_interval": np.timedelta64(1, "s"),
            }
        )
        assert not SampledCoordinate._isvalid({"tie_values": [0.0], "tie_lengths": [1]})
        assert not SampledCoordinate._isvalid({})

    def test_init_and_empty(self):
        empty = SampledCoordinate()
        assert empty.empty
        assert len(empty) == 0
        assert empty.dtype is not None
        assert empty.shape == (0,)
        assert empty.ndim == 1
        assert empty.values.size == 0
        assert empty.indices.size == 0

    def test_init_validation_numeric(self):
        # valid numeric
        coord = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 1.0}
        )
        assert len(coord) == 3
        assert coord.start == 0.0
        assert coord.end == 2.0
        with pytest.warns(FutureWarning, match="issampled"):
            assert coord.issampled() is True
        assert coord.get_sampling_interval() == 1.0

        # mismatched lengths
        with pytest.raises(ValueError):
            SampledCoordinate(
                {
                    "tie_values": [0.0, 10.0],
                    "tie_lengths": [3],
                    "sampling_interval": 1.0,
                }
            )
        # non-integer lengths
        with pytest.raises(ValueError):
            SampledCoordinate(
                {"tie_values": [0.0], "tie_lengths": [1.5], "sampling_interval": 1.0}
            )
        # non-positive lengths
        with pytest.raises(ValueError):
            SampledCoordinate(
                {"tie_values": [0.0], "tie_lengths": [0], "sampling_interval": 1.0}
            )
        # sampling interval must be scalar
        with pytest.raises(ValueError):
            SampledCoordinate(
                {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": [1.0]}
            )

        # non-numeric tie_values
        with pytest.raises(ValueError):
            SampledCoordinate(
                {"tie_values": ["a"], "tie_lengths": [3], "sampling_interval": 1.0}
            )

    def test_init_validation_datetime(self):
        # valid datetime with timedelta sampling interval
        t0 = np.datetime64("2000-01-01T00:00:00")
        coord = SampledCoordinate(
            {
                "tie_values": [t0],
                "tie_lengths": [2],
                "sampling_interval": np.timedelta64(1, "s"),
            }
        )
        assert coord.start == t0
        assert coord.end == t0 + np.timedelta64(1, "s")
        assert coord.get_sampling_interval() == 1
        assert coord.get_sampling_interval(cast=False) == np.timedelta64(1, "s")

        # invalid: datetime with numeric sampling interval
        with pytest.raises(ValueError):
            SampledCoordinate(
                {"tie_values": [t0], "tie_lengths": [2], "sampling_interval": 1}
            )

    def test_invalid_data(self):
        # lack of required keys
        with pytest.raises(ValueError):
            SampledCoordinate({"tie_values": [0.0], "tie_lengths": [3]})
        with pytest.raises(ValueError):
            SampledCoordinate({"tie_lengths": [3], "sampling_interval": 1.0})
        with pytest.raises(ValueError):
            SampledCoordinate({"tie_values": [0.0], "sampling_interval": 1.0})

    def test_invalid_shapes(self):
        # tie_values and tie_lengths must be 1D
        with pytest.raises(ValueError):
            SampledCoordinate(
                {
                    "tie_values": [[0.0, 10.0]],
                    "tie_lengths": [3, 2],
                    "sampling_interval": 1.0,
                }
            )
        with pytest.raises(ValueError):
            SampledCoordinate(
                {
                    "tie_values": [0.0, 10.0],
                    "tie_lengths": [[3], [2]],
                    "sampling_interval": 1.0,
                }
            )


class TestSamplingRatio:
    """SampledCoordinate's numerator/denominator spelling, alongside the
    legacy ``sampling_interval`` one it stays interchangeable with."""

    def test_numerator_denominator_spelling_equals_legacy_spelling(self):
        legacy = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [10], "sampling_interval": 2.0}
        )
        ratio = SampledCoordinate(
            {
                "tie_values": [0.0],
                "tie_lengths": [10],
                "sampling_numerator": 2.0,
                "sampling_denominator": 1,
            }
        )
        assert legacy.equals(ratio)
        assert legacy._sampling_ratio == ratio._sampling_ratio

    def test_integer_ratio_is_gcd_reduced(self):
        coord = SampledCoordinate(
            {
                "tie_values": np.array([0], dtype="int64"),
                "tie_lengths": [10],
                "sampling_numerator": 14,
                "sampling_denominator": 6,
            }
        )
        numerator, denominator = coord._sampling_ratio
        assert numerator == 7
        assert denominator == 3
        assert coord.sampling_interval == 7 // 3

    def test_both_spellings_together_raise(self):
        with pytest.raises(ValueError, match="cannot pass both"):
            SampledCoordinate(
                {
                    "tie_values": [0.0],
                    "tie_lengths": [10],
                    "sampling_interval": 2.0,
                    "sampling_numerator": 2.0,
                    "sampling_denominator": 1,
                }
            )

    def test_partial_pair_raises(self):
        with pytest.raises(ValueError, match="provided together"):
            SampledCoordinate(
                {"tie_values": [0.0], "tie_lengths": [10], "sampling_numerator": 2.0}
            )
        with pytest.raises(ValueError, match="provided together"):
            SampledCoordinate(
                {"tie_values": [0.0], "tie_lengths": [10], "sampling_denominator": 1}
            )

    def test_non_scalar_denominator_raises(self):
        with pytest.raises(ValueError, match="scalar value"):
            SampledCoordinate(
                {
                    "tie_values": [0.0],
                    "tie_lengths": [10],
                    "sampling_numerator": 2.0,
                    "sampling_denominator": [1, 2],
                }
            )

    def test_non_positive_denominator_raises(self):
        with pytest.raises(ValueError, match="strictly positive"):
            SampledCoordinate(
                {
                    "tie_values": [0.0],
                    "tie_lengths": [10],
                    "sampling_numerator": 2.0,
                    "sampling_denominator": 0,
                }
            )


class TestSampledCoordinateIndexing:
    def make_coord(self):
        # Two segments: [0,1,2] and [10,11]
        return SampledCoordinate(
            {"tie_values": [0.0, 10.0], "tie_lengths": [3, 2], "sampling_interval": 1.0}
        )

    def test_len_indices_values(self):
        coord = self.make_coord()
        assert len(coord) == 5
        assert np.array_equal(coord.indices, np.arange(5))
        assert np.array_equal(coord.values, np.array([0.0, 1.0, 2.0, 10.0, 11.0]))

    def test_get_value_scalar_and_vector(self):
        coord = self.make_coord()
        # scalar
        assert coord._get_value(0) == 0.0
        assert coord._get_value(1) == 1.0
        assert coord._get_value(2) == 2.0
        assert coord._get_value(3) == 10.0
        assert coord._get_value(4) == 11.0
        # negative index
        assert coord[-1].data == 11.0
        assert coord[-2].data == 10.0
        assert coord[-3].data == 2.0
        assert coord[-4].data == 1.0
        assert coord[-5].data == 0.0
        # vectorized
        vals = coord[[0, 1, 2, 3, 4, -5, -4, -3, -2, -1]].values
        assert np.array_equal(
            vals, np.array([0.0, 1.0, 2.0, 10.0, 11.0, 0.0, 1.0, 2.0, 10.0, 11.0])
        )
        # bounds
        with pytest.raises(IndexError):
            coord[-6]
        with pytest.raises(IndexError):
            coord[5]
        with pytest.raises(IndexError):
            coord[[0, 5]]
        with pytest.raises(IndexError):
            coord[[-6, 0]]

    def test_values(self):
        coord = self.make_coord()
        expected = np.array([0.0, 1.0, 2.0, 10.0, 11.0])
        assert np.array_equal(coord.values, expected)
        assert np.array_equal(coord.__array__(), expected)
        assert np.array_equal(coord.__array__(dtype=expected.dtype), expected)

    def test_getitem(self):
        coord = self.make_coord()
        # scalar -> ScalarCoordinate
        item = coord[1]
        assert isinstance(item, ScalarCoordinate)
        assert item.values == 1.0
        # slice -> SampledCoordinate or compatible
        sub = coord[1:4]
        assert isinstance(sub, SampledCoordinate)
        assert np.array_equal(sub.values, np.array([1.0, 2.0, 10.0]))
        # slice negative
        sub_neg = coord[-4:-1]
        assert isinstance(sub_neg, SampledCoordinate)
        assert np.array_equal(sub_neg.values, np.array([1.0, 2.0, 10.0]))
        # full slice
        full = coord[:]
        assert full.equals(coord)
        # None bound indexing
        none_start = coord[None:3]
        assert isinstance(none_start, SampledCoordinate)
        assert np.array_equal(none_start.values, np.array([0.0, 1.0, 2.0]))
        none_end = coord[2:None]
        assert isinstance(none_end, SampledCoordinate)
        assert np.array_equal(none_end.values, np.array([2.0, 10.0, 11.0]))
        # step slice -> SampledCoordinate
        step = coord[::2]
        assert isinstance(step, SampledCoordinate)
        assert np.array_equal(step.values, np.array([0.0, 2.0, 11.0]))
        # step slice with start/stop
        step_ss = coord[1:5:2]
        assert isinstance(step_ss, SampledCoordinate)
        assert np.array_equal(step_ss.values, np.array([1.0, 10.0]))
        # negative step slice with start/stop
        step_ss_neg = coord[-4:-1:2]
        assert isinstance(step_ss_neg, SampledCoordinate)
        assert np.array_equal(step_ss_neg.values, np.array([1.0, 10.0]))
        # negative step slice -> raise NotImplementedError
        with pytest.raises(NotImplementedError):
            coord[::-1]
        # array -> DenseCoordinate of values
        arr = coord[[0, 4]]
        assert isinstance(arr, DenseCoordinate)
        assert np.array_equal(arr.values, np.array([0.0, 11.0]))
        # negative step is not implemented yet
        with pytest.raises(NotImplementedError):
            coord[4:0:-1]

    def test_repr(self):
        # floating coord
        floating = self.make_coord()
        assert isinstance(repr(floating), str)
        # integer coord
        integer = SampledCoordinate(
            {"tie_values": [0], "tie_lengths": [3], "sampling_interval": 1}
        )
        assert isinstance(repr(integer), str)
        # empty coord
        empty = SampledCoordinate()
        assert repr(empty) == "empty coordinate"
        # singleton
        singleton = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [1], "sampling_interval": 1.0}
        )
        assert isinstance(repr(singleton), str)
        # numeric coord
        datetime = SampledCoordinate(
            {
                "tie_values": [np.datetime64("2000-01-01T00:00:00")],
                "tie_lengths": [3],
                "sampling_interval": np.timedelta64(1, "s"),
            }
        )
        assert isinstance(repr(datetime), str)


class TestSampledCoordinateSliceEdgeCases:
    def make_coord(self):
        return SampledCoordinate(
            {"tie_values": [0.0, 10.0], "tie_lengths": [3, 2], "sampling_interval": 1.0}
        )

    def test_slice_negative_and_out_of_bounds(self):
        coord = self.make_coord()
        # negative slice indices
        s = coord[-4:-1]
        assert isinstance(s, SampledCoordinate)
        assert np.array_equal(s.values, np.array([1.0, 2.0, 10.0]))
        # slice that extends beyond bounds should clip
        s2 = coord[-10:10]
        assert s2.equals(coord)

    def test_slice_step(self):
        coord = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [10], "sampling_interval": 1.0}
        )
        stepped = coord[::2]
        assert isinstance(stepped, SampledCoordinate)
        assert stepped.sampling_interval == 2.0
        assert stepped.tie_lengths[0] == 5


class TestSampledCoordinateValueBasedIndexing:
    def make_coord(self):
        return SampledCoordinate(
            {"tie_values": [0.0, 10.0], "tie_lengths": [3, 2], "sampling_interval": 1.0}
        )  # two segments: [0, 1, 2] and [10, 11]

    def make_coord_datetime(self):
        t0 = np.datetime64("2000-01-01T00:00:00")
        return SampledCoordinate(
            {
                "tie_values": [t0, t0 + np.timedelta64(10, "s")],
                "tie_lengths": [3, 2],
                "sampling_interval": np.timedelta64(1, "s"),
            }
        )

    def test_get_indexer_exact(self):
        # float
        coord = self.make_coord()
        assert coord._get_indexer(0.0, method=None) == 0
        assert coord._get_indexer(10.0, method=None) == 3
        with pytest.raises(KeyError):
            coord._get_indexer(1.5, method=None)
        with pytest.raises(KeyError):
            coord._get_indexer(5.0, method=None)

        # datetime
        coord = self.make_coord_datetime()
        t0 = coord[0].values
        assert coord._get_indexer(t0, method=None) == 0
        assert coord._get_indexer(t0 + np.timedelta64(10, "s"), method=None) == 3
        with pytest.raises(KeyError):
            coord._get_indexer(t0 + np.timedelta64(1500, "ms"), method=None)
        with pytest.raises(KeyError):
            coord._get_indexer(t0 + np.timedelta64(5, "s"), method=None)

    def test_get_indexer_nearest(self):
        # float
        coord = self.make_coord()
        vals = [0.0, 0.4, 0.6, 1.0, 10.4, 10.6, -10.0, 20.0, 5.9, 6.0, 6.1]
        expected = [0, 0, 1, 1, 3, 4, 0, 4, 2, 3, 3]
        # scalar
        for v, e in zip(vals, expected):
            idx = coord._get_indexer(v, method="nearest")
            assert idx == e
        # vectorized
        idxs = coord._get_indexer(vals, method="nearest")
        assert np.array_equal(idxs, np.array(expected))

        # datetime
        coord = self.make_coord_datetime()
        t0 = coord[0].values
        vals = t0 + np.rint(1000 * np.array(vals)).astype("timedelta64[ms]")
        # scalar
        for v, e in zip(vals, expected):
            idx = coord._get_indexer(v, method="nearest")
            assert idx == e
        # vectorized
        idxs = coord._get_indexer(vals, method="nearest")
        assert np.array_equal(idxs, np.array(expected))

    def test_get_indexer_ffill(self):
        # float
        coord = self.make_coord()
        vals = [0.0, 0.4, 0.6, 1.0, 10.4, 10.6, 20.0, 5.9, 6.0, 6.1]
        expected = [0, 0, 0, 1, 3, 3, 4, 2, 2, 2]
        # scalar
        for v, e in zip(vals, expected):
            idx = coord._get_indexer(v, method="ffill")
            assert idx == e
        with pytest.raises(KeyError):
            coord._get_indexer(-10.0, method="ffill")
        # vectorized
        idxs = coord._get_indexer(vals, method="ffill")
        assert np.array_equal(idxs, np.array(expected))
        with pytest.raises(KeyError):
            coord._get_indexer([-10.0, 0.0], method="ffill")

        # datetime
        coord = self.make_coord_datetime()
        t0 = coord[0].values
        vals = t0 + np.rint(1000 * np.array(vals)).astype("timedelta64[ms]")
        # scalar
        for v, e in zip(vals, expected):
            idx = coord._get_indexer(v, method="ffill")
            assert idx == e
        with pytest.raises(KeyError):
            coord._get_indexer(t0 - np.timedelta64(10, "s"), method="ffill")
        # vectorized
        idxs = coord._get_indexer(vals, method="ffill")
        assert np.array_equal(idxs, np.array(expected))
        with pytest.raises(KeyError):
            coord._get_indexer([t0 - np.timedelta64(10, "s"), t0], method="ffill")

    def test_get_indexer_bfill(self):
        # float
        coord = self.make_coord()
        vals = [0.0, 0.4, 0.6, 1.0, 10.4, 10.6, -10.0, 5.9, 6.0, 6.1]
        expected = [0, 1, 1, 1, 4, 4, 0, 3, 3, 3]
        # scalar
        for v, e in zip(vals, expected):
            idx = coord._get_indexer(v, method="bfill")
            assert idx == e
        with pytest.raises(KeyError):
            coord._get_indexer(20.0, method="bfill")
        # vectorized
        idxs = coord._get_indexer(vals, method="bfill")
        assert np.array_equal(idxs, np.array(expected))
        with pytest.raises(KeyError):
            coord._get_indexer([11.0, 20.0], method="bfill")

        # datetime
        coord = self.make_coord_datetime()
        t0 = coord[0].values
        vals = t0 + np.rint(1000 * np.array(vals)).astype("timedelta64[ms]")
        # scalar
        for v, e in zip(vals, expected):
            idx = coord._get_indexer(v, method="bfill")
            assert idx == e
        with pytest.raises(KeyError):
            coord._get_indexer(t0 + np.timedelta64(20, "s"), method="bfill")
        # vectorized
        idxs = coord._get_indexer(vals, method="bfill")
        assert np.array_equal(idxs, np.array(expected))
        with pytest.raises(KeyError):
            coord._get_indexer([t0, t0 + np.timedelta64(20, "s")], method="bfill")

    def test_get_indexer_overlap(self):
        coord = SampledCoordinate(
            {"tie_values": [0.0, 2.0], "tie_lengths": [3, 3], "sampling_interval": 1.0}
        )  # segments: [0,1,2] and [2,3,4]
        assert coord._get_indexer(1.0) == 1
        assert coord._get_indexer(3.0) == 4
        with pytest.raises(KeyError):
            coord._get_indexer(2.0)
        coord = SampledCoordinate(
            {"tie_values": [0.0, 2.0], "tie_lengths": [5, 5], "sampling_interval": 1.0}
        )  # segments: [0,1,2,3,4] and [2,3,4,5,6]
        assert coord._get_indexer(1.0) == 1
        assert coord._get_indexer(6.0) == 9
        with pytest.raises(KeyError):
            coord._get_indexer(2.0)
        with pytest.raises(KeyError):
            coord._get_indexer(2.5, method="nearest")
        with pytest.raises(KeyError):
            coord._get_indexer(4.0)

    def test_get_indexer_invalid_method(self):
        coord = self.make_coord()
        with pytest.raises(ValueError):
            coord._get_indexer(0.0, method="invalid")


class TestSampledCoordinateConcat:
    def test_concat_two_coords(self):
        coord1 = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 1.0}
        )
        coord2 = SampledCoordinate(
            {"tie_values": [10.0], "tie_lengths": [2], "sampling_interval": 1.0}
        )
        expected = SampledCoordinate(
            {"tie_values": [0.0, 10.0], "tie_lengths": [3, 2], "sampling_interval": 1.0}
        )
        result = coord1._concat(coord2)
        assert result.equals(expected)

    def test_concat_two_datetime_coords(self):
        coord1 = SampledCoordinate(
            {
                "tie_values": [np.datetime64("2000-01-01T00:00:00")],
                "tie_lengths": [3],
                "sampling_interval": np.timedelta64(1, "s"),
            }
        )
        coord2 = SampledCoordinate(
            {
                "tie_values": [np.datetime64("2000-01-01T00:00:10")],
                "tie_lengths": [2],
                "sampling_interval": np.timedelta64(1, "s"),
            }
        )
        expected = SampledCoordinate(
            {
                "tie_values": [
                    np.datetime64("2000-01-01T00:00:00"),
                    np.datetime64("2000-01-01T00:00:10"),
                ],
                "tie_lengths": [3, 2],
                "sampling_interval": np.timedelta64(1, "s"),
            }
        )
        result = coord1._concat(coord2)
        assert result.equals(expected)

    def test_concat_empty(self):
        coord1 = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 1.0}
        )
        coord2 = SampledCoordinate()
        assert coord1._concat(coord2).equals(coord1)
        assert coord2._concat(coord1).equals(coord1)

    def test_concat_sampling_interval_mismatch(self):
        coord1 = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 1.0}
        )
        coord2 = SampledCoordinate(
            {"tie_values": [10.0], "tie_lengths": [2], "sampling_interval": 2.0}
        )
        with pytest.raises(ValueError):
            coord1._concat(coord2)

    def test_concat_dtype_mismatch(self):
        coord1 = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 1.0}
        )
        coord2 = SampledCoordinate(
            {
                "tie_values": [np.datetime64("2000-01-01T00:00:00")],
                "tie_lengths": [1],
                "sampling_interval": np.timedelta64(1, "s"),
            }
        )
        with pytest.raises(ValueError):
            coord1._concat(coord2)

    def test_concat_type_mismatch(self):
        coord1 = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 1.0}
        )
        coord2 = DenseCoordinate(np.array([10.0, 11.0]))
        with pytest.raises(TypeError):
            coord1._concat(coord2)

    def test_concat_dimension_mismatch(self):
        coord1 = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 1.0},
            dim="time",
        )
        coord2 = SampledCoordinate(
            {"tie_values": [10.0], "tie_lengths": [2], "sampling_interval": 1.0},
            dim="depth",
        )
        with pytest.raises(ValueError):
            coord1._concat(coord2)


class TestSampledCoordinateDiscontinuitiesAvailabilities:
    def test_discontinuities_and_availabilities(self):
        # two segments, [0, 1, 2] then [5, 6]: one 2.0 gap between them
        coord = SampledCoordinate(
            {"tie_values": [0.0, 5.0], "tie_lengths": [3, 2], "sampling_interval": 1.0}
        )
        dis = coord.get_discontinuities()
        avail = coord.get_availabilities()
        for df in (dis, avail):
            assert isinstance(df, pd.DataFrame)
            assert set(df.columns) >= {
                "start_index",
                "end_index",
                "start_value",
                "end_value",
                "delta",
                "type",
            }
        # the discontinuity straddles the two segments, which the
        # availabilities bound on either side
        npt.assert_array_equal(dis["start_index"], [2])
        npt.assert_array_equal(dis["end_index"], [3])
        npt.assert_array_equal(dis["start_value"], [2.0])
        npt.assert_array_equal(dis["end_value"], [5.0])
        npt.assert_array_equal(dis["delta"], [2.0])
        npt.assert_array_equal(dis["type"], ["gap"])
        npt.assert_array_equal(avail["start_index"], [0, 3])
        npt.assert_array_equal(avail["end_index"], [2, 4])
        npt.assert_array_equal(avail["type"], ["data", "data"])


class TestSampledCoordinateSlicing:
    def make_coord(self):
        # Two segments: [0,1,2] and [10,11]
        return SampledCoordinate(
            {"tie_values": [0.0, 10.0], "tie_lengths": [3, 2], "sampling_interval": 1.0}
        )

    def test_slice_within_segment(self):
        coord = self.make_coord()
        sliced = coord[0:2]
        assert isinstance(sliced, SampledCoordinate)
        assert len(sliced) == 2
        assert np.array_equal(sliced.values, np.array([0.0, 1.0]))

    def test_slice_cross_segments(self):
        coord = self.make_coord()
        sliced = coord[1:4]
        assert isinstance(sliced, SampledCoordinate)
        assert len(sliced) == 3
        assert np.array_equal(sliced.values, np.array([1.0, 2.0, 10.0]))

    def test_slice_full(self):
        coord = self.make_coord()
        sliced = coord[:]
        assert sliced.equals(coord)


class TestSampledCoordinateDecimate:
    def test_decimate(self):
        coord = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [10], "sampling_interval": 1.0}
        )
        decimated = coord[::2]
        assert decimated.sampling_interval == 2.0
        assert decimated.tie_lengths[0] == 5  # (10 + 2 - 1) // 2 = 5


class TestSampledCoordinateSimplify:
    def test_simplify_continuous(self):
        # Two continuous segments should merge
        coord = SampledCoordinate(
            {
                "tie_values": [0.0, 3.0],
                "tie_lengths": [3, 2],
                "sampling_interval": 1.0,
            }
        )
        result = coord.simplify()
        expected = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [5], "sampling_interval": 1.0}
        )
        assert result.equals(expected)

    def test_simplify_with_tolerance(self):
        # Two nearly continuous segments should merge with tolerance
        coord = SampledCoordinate(
            {
                "tie_values": [0.0, 3.1],
                "tie_lengths": [3, 2],
                "sampling_interval": 1.0,
            }
        )
        result = coord.simplify(tolerance=0.2)
        expected = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [5], "sampling_interval": 1.0}
        )
        assert result.equals(expected)
        # more advanced test
        coord = SampledCoordinate(
            {
                "tie_values": 10 * np.arange(100) + np.random.rand(100) * 0.2 - 0.1,
                "tie_lengths": 10 * np.ones(100, dtype=int),
                "sampling_interval": 1.0,
            }
        )
        result = coord.simplify(tolerance=0.2)
        assert len(result.tie_values) == 1
        # extra test
        coord = SampledCoordinate(
            {
                "tie_values": 10 * np.arange(100) + np.random.rand(100) * 0.2 - 0.1,
                "tie_lengths": 10 * np.ones(100, dtype=int),
                "sampling_interval": 1.0,
            }
        )
        result = coord.simplify(tolerance=0.1)
        assert np.all(np.abs(result.values - coord.values) <= 0.1)

    def test_simplify_with_tolerance_on_datetime(self):
        t0 = np.datetime64("2000-01-01T00:00:00")
        jitter = np.random.rand(100) * 0.2 - 0.1
        jitter = jitter.astype("timedelta64[ms]")  # convert to timedelta
        coord = SampledCoordinate(
            {
                "tie_values": t0 + 10 * np.arange(100) + jitter,
                "tie_lengths": 10 * np.ones(100, dtype=int),
                "sampling_interval": np.timedelta64(1, "s"),
            }
        )
        result = coord.simplify(tolerance=np.timedelta64(200, "ms"))
        assert len(result.tie_values) == 1
        # float tolerance should be treated as seconds
        result = coord.simplify(tolerance=0.2)
        assert len(result.tie_values) == 1


class TestSampledCoordinateGetIndexer:
    def make_coord(self):
        return SampledCoordinate(
            {"tie_values": [0.0, 10.0], "tie_lengths": [3, 2], "sampling_interval": 1.0}
        )

    def test_get_indexer_exact(self):
        coord = self.make_coord()
        idx = coord._get_indexer(0.0, method="nearest")
        assert idx == 0
        idx = coord._get_indexer(10.0, method="nearest")
        assert idx == 3

    def test_get_indexer_nearest(self):
        coord = self.make_coord()
        idx = coord._get_indexer(0.5, method="nearest")
        assert idx in [0, 1]

    def test_get_indexer_out_of_bounds(self):
        coord = self.make_coord()
        with pytest.raises(KeyError):
            coord._get_indexer(100.0)


class TestSampledCoordinateArithmetic:
    def test_add(self):
        coord = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 1.0}
        )
        result = coord + 10.0
        assert result.tie_values[0] == 10.0
        assert np.array_equal(result.values, np.array([10.0, 11.0, 12.0]))

    def test_sub(self):
        coord = SampledCoordinate(
            {"tie_values": [10.0], "tie_lengths": [3], "sampling_interval": 1.0}
        )
        result = coord - 5.0
        assert result.tie_values[0] == 5.0
        assert np.array_equal(result.values, np.array([5.0, 6.0, 7.0]))


class TestSampledCoordinateDatetime:
    def make_dt_coord(self):
        t0 = np.datetime64("2000-01-01T00:00:00")
        return SampledCoordinate(
            {
                "tie_values": [t0, t0 + np.timedelta64(10, "s")],
                "tie_lengths": [3, 2],
                "sampling_interval": np.timedelta64(1, "s"),
            }
        )

    def test_datetime_values_and_dtype(self):
        coord = self.make_dt_coord()
        assert np.issubdtype(coord.dtype, np.datetime64)
        vals = coord.values
        assert np.issubdtype(vals.dtype, np.datetime64)
        assert vals[0] == np.datetime64("2000-01-01T00:00:00")
        assert vals[3] == np.datetime64("2000-01-01T00:00:10")

    def test_get_value_datetime(self):
        coord = self.make_dt_coord()
        assert coord._get_value(1) == np.datetime64("2000-01-01T00:00:01")
        assert coord._get_value(4) == np.datetime64("2000-01-01T00:00:11")
        with pytest.raises(IndexError):
            coord[5]

    def test_get_indexer_datetime_methods(self):
        coord = self.make_dt_coord()
        t = np.datetime64("2000-01-01T00:00:01.500")
        # exact required when method=None -> should raise
        with pytest.raises(KeyError):
            coord._get_indexer(t)
        # method variants
        assert coord._get_indexer(t, method="nearest") in [1, 2]
        assert coord._get_indexer(t, method="ffill") == 1
        assert coord._get_indexer(t, method="bfill") == 2
        # bounds
        with pytest.raises(KeyError):
            coord._get_indexer(np.datetime64("1999-12-31T23:59:59"))
        with pytest.raises(KeyError):
            coord._get_indexer(np.datetime64("2000-01-01T00:00:12"))
        # string input
        assert coord._get_indexer("2000-01-01T00:00:01.500", method="nearest") in [1, 2]
        # invalid method
        with pytest.raises(ValueError):
            coord._get_indexer(t, method="bad")

    def test_start_end_properties_datetime(self):
        coord = self.make_dt_coord()
        assert coord.start == np.datetime64("2000-01-01T00:00:00")
        assert coord.end == np.datetime64("2000-01-01T00:00:11")


class TestSampledCoordinateIndexerEdgeCases:
    def test_invalid_method_raises(self):
        coord = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 1.0}
        )
        with pytest.raises(ValueError):
            coord._get_indexer(0.0, method="bad")

    def test_non_increasing_tie_values_raises(self):
        coord = SampledCoordinate(
            {"tie_values": [2.0, 1.0], "tie_lengths": [3, 2], "sampling_interval": 1.0}
        )
        with pytest.raises(ValueError):
            coord._get_indexer(2.0)


class TestSampledCoordinateToNetCDF:
    def make_dataarray(self):
        return xd.DataArray(
            np.random.rand(20, 30),
            {
                "time": {
                    "tie_values": [
                        np.datetime64("2000-01-01T00:00:00.000000000"),
                        np.datetime64("2000-01-01T00:00:10.000000000"),
                    ],
                    "tie_lengths": [5, 15],
                    "sampling_interval": np.timedelta64(1_000_000_000, "ns").astype(
                        "timedelta64[ns]"
                    ),
                },
                "distance": {
                    "tie_values": [0.0],
                    "tie_lengths": [30],
                    "sampling_interval": 1.0,
                },
            },
        )

    def test_to_dataset_and_back(self):
        import xarray as xr

        da = self.make_dataarray()
        dataset = xr.Dataset()
        variable_attrs = {}

        # prepare metadata
        for coord in da.coords.values():
            dataset, variable_attrs = coord._to_dataset(dataset, variable_attrs)

        dataset["data"] = xr.DataArray(attrs=variable_attrs)
        coords = xd.Coordinates._from_dataset(dataset, "data")

        assert coords.equals(da.coords)

    def test_to_dataset_and_back_preserves_the_denominator(self):
        # F: a 999-sample rate has a denominator that does not reduce to 1;
        # the round trip must carry the exact pair, not the floored scalar.
        import xarray as xr

        coord = SampledCoordinate(
            {
                "tie_values": [0],
                "tie_lengths": [1000],
                "sampling_numerator": 30_000_000_000,
                "sampling_denominator": 999,
            },
            "time",
            dtype="timedelta64[ns]",
        )
        dataset = xr.Dataset()
        dataset, attrs = coord._to_dataset(dataset, {})
        assert dataset["time_sampling"].attrs["sampling_interval_denominator"] == 333
        dataset["__values__"] = xr.DataArray(np.zeros(1000), dims=["time"], attrs=attrs)
        recovered = SampledCoordinate._collect_from_dataset(dataset, "__values__")["time"]
        assert recovered._sampling_ratio == coord._sampling_ratio

    def test_dataset_written_before_the_denominator_existed_loads_unchanged(self):
        # A file written before this round trip existed has no
        # `sampling_interval_denominator` attribute at all -- exactly what a
        # whole-tick rate (denominator 1) still writes today -- and must
        # load exactly as before, denominator implicitly 1.
        import xarray as xr

        coord = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [30], "sampling_interval": 2.5}, "distance"
        )
        dataset = xr.Dataset()
        dataset, attrs = coord._to_dataset(dataset, {})
        assert "sampling_interval_denominator" not in dataset["distance_sampling"].attrs
        dataset["__values__"] = xr.DataArray(np.zeros(30), dims=["distance"], attrs=attrs)
        recovered = SampledCoordinate._collect_from_dataset(dataset, "__values__")["distance"]
        assert recovered.equals(coord)

    def test_to_netcdf_and_back(self):
        expected = self.make_dataarray()

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as file:
            expected.to_netcdf(file.name)
            result = xd.open(file.name)
            assert result.equals(expected)

    def test_collect_legacy_spelling(self):
        # the spelling that predates the CF-shaped grammar: the group named the
        # coordinate, the mapping listed both tie point variables, and the
        # interval was the sampling variable's own value
        import xarray as xr

        dataset = xr.Dataset(
            {
                "time_values": ("time_points", np.array([0, 1_000_000_000])),
                "time_lengths": ("time_points", np.array([100, 100])),
                "time_sampling": (
                    (),
                    8,
                    {
                        "tie_point_mapping": "time: time_values time_lengths",
                        "dtype": "timedelta64[ns]",
                        "units": "milliseconds",
                    },
                ),
                "__values__": (
                    ("time",),
                    np.zeros(200),
                    {"coordinate_sampling": "time: time_sampling"},
                ),
            }
        )
        recovered = SampledCoordinate._collect_from_dataset(dataset, "__values__")
        coord = recovered["time"]
        assert coord.dim == "time"
        assert coord.sampling_interval == np.timedelta64(8, "ms")
        npt.assert_array_equal(coord.tie_lengths, [100, 100])

    def test_collect_legacy_spelling_numeric(self):
        # the same, on a numeric axis: no units/dtype attributes to decode, the
        # sampling variable's value is the interval as it stands
        import xarray as xr

        dataset = xr.Dataset(
            {
                "distance_values": ("distance_points", np.array([0.0])),
                "distance_lengths": ("distance_points", np.array([30])),
                "distance_sampling": (
                    (),
                    2.5,
                    {"tie_point_mapping": "distance: distance_values distance_lengths"},
                ),
                "__values__": (
                    ("distance",),
                    np.zeros(30),
                    {"coordinate_sampling": "distance: distance_sampling"},
                ),
            }
        )
        recovered = SampledCoordinate._collect_from_dataset(dataset, "__values__")
        assert recovered["distance"].sampling_interval == 2.5


class TestGetSplitIndices:
    def test_no_tolerance(self):
        coord = SampledCoordinate(
            {"tie_values": [0.0, 10.0], "tie_lengths": [3, 2], "sampling_interval": 1.0}
        )
        div_points = coord.get_split_indices()
        expected = np.array([3])  # indices where segments end
        assert np.array_equal(div_points, expected)

    def test_with_tolerance(self):
        coord = SampledCoordinate(
            {
                "tie_values": [0.0, 3.1, 10.0],
                "tie_lengths": [3, 2, 2],
                "sampling_interval": 1.0,
            }
        )
        div_points = coord.get_split_indices(tolerance=0.2)
        expected = np.array([5])  # only the second gap exceeds tolerance
        assert np.array_equal(div_points, expected)

    def test_with_tolerance_on_datetime(self):
        t0 = np.datetime64("2000-01-01T00:00:00")
        coord = SampledCoordinate(
            {
                "tie_values": [
                    t0,
                    t0 + np.timedelta64(3, "s") + np.timedelta64(100, "ms"),
                    t0 + np.timedelta64(10, "s"),
                ],
                "tie_lengths": [3, 2, 2],
                "sampling_interval": np.timedelta64(1, "s"),
            }
        )
        div_points = coord.get_split_indices(tolerance=np.timedelta64(200, "ms"))
        expected = np.array([5])  # only the second gap exceeds tolerance
        assert np.array_equal(div_points, expected)
        # float tolerance should be treated as seconds
        div_points = coord.get_split_indices(tolerance=0.2)
        assert np.array_equal(div_points, expected)


class TestFromBlock:
    def test_from_block(self):
        result = SampledCoordinate.from_block(start=0.0, size=5, step=1.0)
        expected = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [5], "sampling_interval": 1.0}
        )
        assert result.equals(expected)


class TestNotImplementedMethods:
    def test_raises(self):
        coord = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 1.0}
        )
        with pytest.raises(NotImplementedError):
            coord[::-1]


class TestSampledCoordinateMissingBranches:
    def make_coord(self):
        return SampledCoordinate(
            {"tie_values": [0.0, 10.0], "tie_lengths": [3, 2], "sampling_interval": 1.0}
        )

    def make_coord_with_overlap(self):
        return SampledCoordinate(
            {"tie_values": [0.0, 5.0], "tie_lengths": [3, 2], "sampling_interval": 1.0}
        )

    def test_simplify_false(self):
        coord = self.make_coord()
        result = coord.simplify(False)
        assert result is not coord
        assert result.equals(coord)

    def test_get_split_indices_gaps(self):
        coord = self.make_coord()
        gaps = coord.get_split_indices(kind="gaps")
        assert isinstance(gaps, np.ndarray)

    def test_get_split_indices_overlaps(self):
        coord = self.make_coord()
        overlaps = coord.get_split_indices(kind="overlaps")
        assert isinstance(overlaps, np.ndarray)

    def test_get_indexer_bfill_in_bounds(self):
        coord = self.make_coord()
        assert coord._get_indexer(0.0, method="bfill") == 0
        assert coord._get_indexer(0.5, method="bfill") == 1

    def test_get_split_indices_overlaps_tolerance_false(self):
        # Build a coord with an actual overlap (segment 2 starts before segment 1 ends)
        coord = SampledCoordinate(
            {"tie_values": [0.0, 2.0], "tie_lengths": [5, 5], "sampling_interval": 1.0}
        )
        result = coord.get_split_indices(kind="overlaps", tolerance=False)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [5], strict=True)

    def test_get_split_indices_overlaps_with_tolerance(self):
        # Build a coord with an actual overlap (segment 2 starts before segment 1 ends)
        coord = SampledCoordinate(
            {"tie_values": [0.0, 2.0], "tie_lengths": [5, 5], "sampling_interval": 1.0}
        )
        result = coord.get_split_indices(kind="overlaps", tolerance=1.0)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [5], strict=True)

    def test_is_monotonic_increasing_true(self):
        coord = SampledCoordinate(
            {"tie_values": [0.0, 5.0], "tie_lengths": [5, 5], "sampling_interval": 1.0}
        )
        assert coord._is_monotonic_increasing() is True

    def test_is_monotonic_increasing_false(self):
        coord = SampledCoordinate(
            {"tie_values": [0.0, 2.0], "tie_lengths": [5, 5], "sampling_interval": 1.0}
        )
        assert coord._is_monotonic_increasing() is False

    def test_is_monotonic_increasing_multi_segment(self):
        # Three segments all increasing — must not raise ValueError from bool()
        coord = SampledCoordinate(
            {
                "tie_values": [0.0, 5.0, 11.0],
                "tie_lengths": [5, 5, 5],
                "sampling_interval": 1.0,
            }
        )
        assert coord._is_monotonic_increasing() is True

    def test_is_monotonic_increasing_subsample_overlap(self):
        # the seam advances by 0.4 of a 1.0 sampling interval: an overlap, but
        # the values keep increasing, so the axis stays sorted
        coord = SampledCoordinate(
            {"tie_values": [0.0, 4.4], "tie_lengths": [5, 5], "sampling_interval": 1.0}
        )
        npt.assert_array_equal(coord.get_split_indices("overlaps"), [5])
        assert coord._is_monotonic_increasing() is True

    def test_is_monotonic_increasing_negative_interval(self):
        # a regular axis running backwards has no seam to report it
        coord = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [5], "sampling_interval": -1.0}
        )
        npt.assert_array_equal(coord.get_split_indices("discontinuities"), [])
        assert coord._is_monotonic_increasing() is False

    def test_is_monotonic_increasing_single_sample_segments(self):
        # segments of one sample have no interior, so the interval never falls
        # between two samples and its sign cannot disorder anything
        coord = SampledCoordinate(
            {
                "tie_values": [0.0, 1.0, 2.0],
                "tie_lengths": [1, 1, 1],
                "sampling_interval": -1.0,
            }
        )
        assert coord._is_monotonic_increasing() is True

    def test_is_monotonic_increasing_empty(self):
        assert SampledCoordinate()._is_monotonic_increasing() is True

    def test_get_sampling_interval_singleton(self):
        coord = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [1], "sampling_interval": 1.0}
        )
        assert coord.get_sampling_interval() is None

    def test_collect_from_dataset_no_sampling(self):
        import xarray as xr

        dataset = xr.Dataset({"data": xr.DataArray(np.zeros(3))})
        result = SampledCoordinate._collect_from_dataset(dataset, "data")
        assert result == {}


class TestSampledCoordinateToRegular:
    def test_returns_copy(self):
        coord = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [5], "sampling_interval": 2.0}, "x"
        )
        reg = coord.to_regular()
        assert reg.equals(coord)
        assert reg is not coord

    def test_matching_explicit_interval(self):
        coord = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [5], "sampling_interval": 2.0}, "x"
        )
        assert coord.to_regular(sampling_interval=2.0).equals(coord)

    def test_mismatching_interval_raises(self):
        coord = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [5], "sampling_interval": 2.0}, "x"
        )
        with pytest.raises(ValueError, match="does not match"):
            coord.to_regular(sampling_interval=3.0)

    def test_mismatch_within_tolerance(self):
        coord = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [5], "sampling_interval": 2.0}, "x"
        )
        assert coord.to_regular(sampling_interval=2.05, tolerance=0.1).equals(coord)


class TestSampledCoordinateExactRate:
    """Phase B acceptance: fixes F1 (unbounded drift from a rounded rate),
    F3 (two rates that only agree once rounded), F7 (overflow at large tick
    counts) -- see /ssd/trabatto/claude/exact-rate-ratio/PLAN.html section 07.
    """

    NUM, DEN = 30_000_000, 999  # F1/F3's rate, in ticks

    def test_forward_step_exact_to_half_tick_at_every_index(self):
        # the plan's acceptance size: a 3e7-sample segment at 30000000/999
        n = 30_000_000
        coord = SampledCoordinate(
            {
                "tie_values": np.array([0], dtype="int64"),
                "tie_lengths": [n],
                "sampling_numerator": self.NUM,
                "sampling_denominator": self.DEN,
            }
        )
        k = np.arange(n, dtype="int64")
        # k * NUM tops out at 9e14, comfortably under 2**53 (~9.007e15), so a
        # float64 round-half-to-even division is an exact reference here
        exact = k.astype("f8") * self.NUM / self.DEN
        expected = np.round(exact).astype("int64")
        actual = coord._get_value(k)
        assert np.array_equal(actual, expected)
        assert np.max(np.abs(actual.astype("f8") - exact)) <= 0.5

    def test_get_value_exact_datetime_fractional_rate(self):
        from fractions import Fraction

        t0 = np.datetime64("2024-01-01T00:00:00", "us")
        coord = SampledCoordinate(
            {
                "tie_values": np.array([t0], dtype="datetime64[us]"),
                "tie_lengths": [50],
                "sampling_numerator": np.timedelta64(self.NUM, "us"),
                "sampling_denominator": self.DEN,
            }
        )
        k = np.arange(50)
        actual = (coord._get_value(k) - t0).astype("timedelta64[us]").astype("i8")
        # python's round() on a Fraction ties to even, matching xinterp
        expected = np.array([round(Fraction(int(ki) * self.NUM, self.DEN)) for ki in k])
        assert np.array_equal(actual, expected)

    def test_two_rates_that_round_to_the_same_interval_no_longer_concat(self):
        # both 30000000/999 and 30000300/999 floor to sampling_interval
        # 30030 -- laundered as the same rate before phase A/B -- yet are
        # exact, different rates and must not be allowed to concatenate
        a = SampledCoordinate(
            {
                "tie_values": np.array([0], dtype="int64"),
                "tie_lengths": [999],
                "sampling_numerator": self.NUM,
                "sampling_denominator": self.DEN,
            }
        )
        b = SampledCoordinate(
            {
                "tie_values": np.array([a.values[-1] + a.sampling_interval]),
                "tie_lengths": [999],
                "sampling_numerator": 30_000_300,
                "sampling_denominator": self.DEN,
            }
        )
        assert a.sampling_interval == b.sampling_interval == 30030
        with pytest.raises(ValueError, match="different sampling intervals"):
            a._concat(b)

    @pytest.mark.parametrize("step", [1, 7, 100])
    def test_slice_step_exact(self, step):
        from fractions import Fraction

        n = 10_000
        coord = SampledCoordinate(
            {
                "tie_values": np.array([0], dtype="int64"),
                "tie_lengths": [n],
                "sampling_numerator": self.NUM,
                "sampling_denominator": self.DEN,
            }
        )
        sliced = coord[::step]
        numerator, denominator = sliced._sampling_ratio
        assert (
            Fraction(int(numerator), int(denominator))
            == Fraction(self.NUM, self.DEN) * step
        )

        parent_values = coord.values
        sliced_values = sliced.values
        expected_indices = np.arange(len(sliced_values)) * step
        # the anchor is read straight off the parent, so it is bit-identical
        assert sliced_values[0] == parent_values[0]
        # later samples, re-stepped from that anchor, land within a tick of
        # a full re-derivation against the parent -- accepted as
        # sub-resolution (D3)
        assert np.all(np.abs(sliced_values - parent_values[expected_indices]) <= 1)

    def test_slice_anchor_is_bit_identical_to_parent_at_offset(self):
        coord = SampledCoordinate(
            {
                "tie_values": np.array([0], dtype="int64"),
                "tie_lengths": [10_000],
                "sampling_numerator": self.NUM,
                "sampling_denominator": self.DEN,
            }
        )
        sliced = coord[537:9000:13]
        assert sliced.tie_values[0] == coord.values[537]

    def test_simplify_preserves_exact_ratio_instead_of_flooring(self):
        coord = SampledCoordinate(
            {
                "tie_values": np.array([0], dtype="int64"),
                "tie_lengths": [10],
                "sampling_numerator": self.NUM,
                "sampling_denominator": self.DEN,
            }
        )
        # NUM/DEN does not divide evenly: a floored `sampling_interval`
        # would have silently rounded this away before phase B
        assert coord._sampling_ratio[1] != 1
        simplified = coord.simplify(tolerance=0)
        assert simplified._sampling_ratio == coord._sampling_ratio

    def test_concat_preserves_exact_ratio(self):
        a = SampledCoordinate(
            {
                "tie_values": np.array([0], dtype="int64"),
                "tie_lengths": [10],
                "sampling_numerator": self.NUM,
                "sampling_denominator": self.DEN,
            }
        )
        b = SampledCoordinate(
            {
                "tie_values": np.array([a.values[-1] + a.sampling_interval]),
                "tie_lengths": [5],
                "sampling_numerator": self.NUM,
                "sampling_denominator": self.DEN,
            }
        )
        combined = a._concat(b)
        assert combined._sampling_ratio == a._sampling_ratio

    def test_add_and_sub_preserve_exact_ratio(self):
        coord = SampledCoordinate(
            {
                "tie_values": np.array([0], dtype="int64"),
                "tie_lengths": [10],
                "sampling_numerator": self.NUM,
                "sampling_denominator": self.DEN,
            }
        )
        assert (coord + 5)._sampling_ratio == coord._sampling_ratio
        assert (coord - 5)._sampling_ratio == coord._sampling_ratio

    def test_get_indexer_round_trip_on_plain_integer_dtype(self):
        coord = SampledCoordinate(
            {
                "tie_values": np.array([0], dtype="int64"),
                "tie_lengths": [30],
                "sampling_numerator": self.NUM,
                "sampling_denominator": self.DEN,
            }
        )
        values = coord.values
        for i, value in enumerate(values):
            assert coord._get_indexer(value, method=None) == i

    def test_get_indexer_on_single_sample_block(self):
        coord = SampledCoordinate(
            {"tie_values": [5.0], "tie_lengths": [1], "sampling_interval": 2.0}
        )
        assert coord._get_indexer(5.0, method=None) == 0
        with pytest.raises(KeyError):
            coord._get_indexer(6.0, method=None)

        assert coord._get_indexer(6.0, method="nearest") == 0
        assert coord._get_indexer(4.0, method="nearest") == 0

        assert coord._get_indexer(6.0, method="ffill") == 0
        with pytest.raises(KeyError):
            coord._get_indexer(4.0, method="ffill")

        assert coord._get_indexer(4.0, method="bfill") == 0
        with pytest.raises(KeyError):
            coord._get_indexer(6.0, method="bfill")
