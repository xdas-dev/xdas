import tempfile

import numpy as np
import pandas as pd
import pytest

import xdas as xd
from xdas.coordinates import (
    Coordinate,
    DenseCoordinate,
    SampledCoordinate,
    ScalarCoordinate,
)


class TestSampledCoordinateBasics:
    def test_isvalid(self):
        assert SampledCoordinate.isvalid(
            {"tie_values": [0.0], "tie_lengths": [1], "sampling_interval": 1.0}
        )
        assert SampledCoordinate.isvalid(
            {
                "tie_values": [np.datetime64("2000-01-01T00:00:00")],
                "tie_lengths": [1],
                "sampling_interval": np.timedelta64(1, "s"),
            }
        )
        assert not SampledCoordinate.isvalid({"tie_values": [0.0], "tie_lengths": [1]})
        assert not SampledCoordinate.isvalid({})

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
        assert coord.end == 3.0
        assert coord.issampled()
        coord.get_sampling_interval() == 1.0

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
        assert coord.end == t0 + np.timedelta64(2, "s")
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
        assert coord.get_value(0) == 0.0
        assert coord.get_value(1) == 1.0
        assert coord.get_value(2) == 2.0
        assert coord.get_value(3) == 10.0
        assert coord.get_value(4) == 11.0
        # negative index
        assert coord.get_value(-1) == 11.0
        assert coord.get_value(-2) == 10.0
        assert coord.get_value(-3) == 2.0
        assert coord.get_value(-4) == 1.0
        assert coord.get_value(-5) == 0.0
        # vectorized
        vals = coord.get_value([0, 1, 2, 3, 4, -5, -4, -3, -2, -1])
        assert np.array_equal(
            vals, np.array([0.0, 1.0, 2.0, 10.0, 11.0, 0.0, 1.0, 2.0, 10.0, 11.0])
        )
        # bounds
        with pytest.raises(IndexError):
            coord.get_value(-6)
        with pytest.raises(IndexError):
            coord.get_value(5)
        with pytest.raises(IndexError):
            coord.get_value([0, 5])
        with pytest.raises(IndexError):
            coord.get_value([-6, 0])

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

    def test_slice_step_decimate(self):
        coord = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [10], "sampling_interval": 1.0}
        )
        stepped = coord[::2]
        decimated = coord.decimate(2)
        assert isinstance(stepped, SampledCoordinate)
        assert decimated.equals(stepped)


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
        assert coord.get_indexer(0.0, method=None) == 0
        assert coord.get_indexer(10.0, method=None) == 3
        with pytest.raises(KeyError):
            coord.get_indexer(1.5, method=None)
        with pytest.raises(KeyError):
            coord.get_indexer(5.0, method=None)

        # datetime
        coord = self.make_coord_datetime()
        t0 = coord[0].values
        assert coord.get_indexer(t0, method=None) == 0
        assert coord.get_indexer(t0 + np.timedelta64(10, "s"), method=None) == 3
        with pytest.raises(KeyError):
            coord.get_indexer(t0 + np.timedelta64(1500, "ms"), method=None)
        with pytest.raises(KeyError):
            coord.get_indexer(t0 + np.timedelta64(5, "s"), method=None)

    def test_get_indexer_nearest(self):
        # float
        coord = self.make_coord()
        vals = [0.0, 0.4, 0.6, 1.0, 10.4, 10.6, -10.0, 20.0, 5.9, 6.0, 6.1]
        expected = [0, 0, 1, 1, 3, 4, 0, 4, 2, 3, 3]
        # scalar
        for v, e in zip(vals, expected):
            idx = coord.get_indexer(v, method="nearest")
            assert idx == e
        # vectorized
        idxs = coord.get_indexer(vals, method="nearest")
        assert np.array_equal(idxs, np.array(expected))

        # datetime
        coord = self.make_coord_datetime()
        t0 = coord[0].values
        vals = t0 + np.rint(1000 * np.array(vals)).astype("timedelta64[ms]")
        # scalar
        for v, e in zip(vals, expected):
            idx = coord.get_indexer(v, method="nearest")
            assert idx == e
        # vectorized
        idxs = coord.get_indexer(vals, method="nearest")
        assert np.array_equal(idxs, np.array(expected))

    def test_get_indexer_ffill(self):
        # float
        coord = self.make_coord()
        vals = [0.0, 0.4, 0.6, 1.0, 10.4, 10.6, 20.0, 5.9, 6.0, 6.1]
        expected = [0, 0, 0, 1, 3, 3, 4, 2, 2, 2]
        # scalar
        for v, e in zip(vals, expected):
            idx = coord.get_indexer(v, method="ffill")
            assert idx == e
        with pytest.raises(KeyError):
            coord.get_indexer(-10.0, method="ffill")
        # vectorized
        idxs = coord.get_indexer(vals, method="ffill")
        assert np.array_equal(idxs, np.array(expected))
        with pytest.raises(KeyError):
            coord.get_indexer([-10.0, 0.0], method="ffill")

        # datetime
        coord = self.make_coord_datetime()
        t0 = coord[0].values
        vals = t0 + np.rint(1000 * np.array(vals)).astype("timedelta64[ms]")
        # scalar
        for v, e in zip(vals, expected):
            idx = coord.get_indexer(v, method="ffill")
            assert idx == e
        with pytest.raises(KeyError):
            coord.get_indexer(t0 - np.timedelta64(10, "s"), method="ffill")
        # vectorized
        idxs = coord.get_indexer(vals, method="ffill")
        assert np.array_equal(idxs, np.array(expected))
        with pytest.raises(KeyError):
            coord.get_indexer([t0 - np.timedelta64(10, "s"), t0], method="ffill")

    def test_get_indexer_bfill(self):
        # float
        coord = self.make_coord()
        vals = [0.0, 0.4, 0.6, 1.0, 10.4, 10.6, -10.0, 5.9, 6.0, 6.1]
        expected = [0, 1, 1, 1, 4, 4, 0, 3, 3, 3]
        # scalar
        for v, e in zip(vals, expected):
            idx = coord.get_indexer(v, method="bfill")
            assert idx == e
        with pytest.raises(KeyError):
            coord.get_indexer(20.0, method="bfill")
        # vectorized
        idxs = coord.get_indexer(vals, method="bfill")
        assert np.array_equal(idxs, np.array(expected))
        with pytest.raises(KeyError):
            coord.get_indexer([11.0, 20.0], method="bfill")

        # datetime
        coord = self.make_coord_datetime()
        t0 = coord[0].values
        vals = t0 + np.rint(1000 * np.array(vals)).astype("timedelta64[ms]")
        # scalar
        for v, e in zip(vals, expected):
            idx = coord.get_indexer(v, method="bfill")
            assert idx == e
        with pytest.raises(KeyError):
            coord.get_indexer(t0 + np.timedelta64(20, "s"), method="bfill")
        # vectorized
        idxs = coord.get_indexer(vals, method="bfill")
        assert np.array_equal(idxs, np.array(expected))
        with pytest.raises(KeyError):
            coord.get_indexer([t0, t0 + np.timedelta64(20, "s")], method="bfill")

    def test_get_indexer_overlap(self):
        coord = SampledCoordinate(
            {"tie_values": [0.0, 2.0], "tie_lengths": [3, 3], "sampling_interval": 1.0}
        )  # segments: [0,1,2] and [2,3,4]
        assert coord.get_indexer(1.0) == 1
        assert coord.get_indexer(3.0) == 4
        with pytest.raises(KeyError):
            coord.get_indexer(2.0)
        coord = SampledCoordinate(
            {"tie_values": [0.0, 2.0], "tie_lengths": [5, 5], "sampling_interval": 1.0}
        )  # segments: [0,1,2,3,4] and [2,3,4,5,6]
        assert coord.get_indexer(1.0) == 1
        assert coord.get_indexer(6.0) == 9
        with pytest.raises(KeyError):
            coord.get_indexer(2.0)
        with pytest.raises(KeyError):
            coord.get_indexer(2.5, method="nearest")
        with pytest.raises(KeyError):
            coord.get_indexer(4.0)

    def test_get_indexer_invalid_method(self):
        coord = self.make_coord()
        with pytest.raises(ValueError):
            coord.get_indexer(0.0, method="invalid")


class TestSampledCoordinateAppend:
    def test_append_two_coords(self):
        coord1 = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 1.0}
        )
        coord2 = SampledCoordinate(
            {"tie_values": [10.0], "tie_lengths": [2], "sampling_interval": 1.0}
        )
        expected = SampledCoordinate(
            {"tie_values": [0.0, 10.0], "tie_lengths": [3, 2], "sampling_interval": 1.0}
        )
        result = coord1.append(coord2)
        assert result.equals(expected)

    def test_append_two_datetime_coords(self):
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
        result = coord1.append(coord2)
        assert result.equals(expected)

    def test_append_empty(self):
        coord1 = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 1.0}
        )
        coord2 = SampledCoordinate()
        assert coord1.append(coord2).equals(coord1)
        assert coord2.append(coord1).equals(coord1)

    def test_append_sampling_interval_mismatch(self):
        coord1 = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 1.0}
        )
        coord2 = SampledCoordinate(
            {"tie_values": [10.0], "tie_lengths": [2], "sampling_interval": 2.0}
        )
        with pytest.raises(ValueError):
            coord1.append(coord2)

    def test_append_dtype_mismatch(self):
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
            coord1.append(coord2)

    def test_append_type_mismatch(self):
        coord1 = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 1.0}
        )
        coord2 = DenseCoordinate(np.array([10.0, 11.0]))
        with pytest.raises(TypeError):
            coord1.append(coord2)

    def test_append_dimension_mismatch(self):
        coord1 = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 1.0},
            dim="time",
        )
        coord2 = SampledCoordinate(
            {"tie_values": [10.0], "tie_lengths": [2], "sampling_interval": 1.0},
            dim="depth",
        )
        with pytest.raises(ValueError):
            coord1.append(coord2)


class TestSampledCoordinateDiscontinuitiesAvailabilities:
    def test_discontinuities_and_availabilities(self):
        # tie_lengths set to create 2 segments
        coord = SampledCoordinate(
            {"tie_values": [0.0, 5.0], "tie_lengths": [3, 2], "sampling_interval": 1.0}
        )
        dis = coord.get_discontinuities()
        avail = coord.get_availabilities()
        # expect DataFrame with specific columns
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
        # availabilities should list segments (2 segments -> 2 records)
        assert len(avail) >= 1


class TestSampledCoordinateToDatasetAndDict:
    def test_to_dict_contains_expected_keys(self):
        coord = SampledCoordinate(
            {
                "tie_values": [0.0, 10.0],
                "tie_lengths": [3, 2],
                "sampling_interval": 1.0,
            },
            dim="time",
        )
        d = coord.to_dict()
        assert "dim" in d
        assert "data" in d
        assert set(d["data"].keys()) >= {
            "tie_values",
            "tie_lengths",
            "sampling_interval",
        }

    def test_to_dict_with_datetime(self):
        t0 = np.datetime64("2000-01-01T00:00:00")
        coord = SampledCoordinate(
            {
                "tie_values": [t0, t0 + np.timedelta64(10, "s")],
                "tie_lengths": [3, 2],
                "sampling_interval": np.timedelta64(1, "s"),
            },
            dim="time",
        )
        d = coord.to_dict()
        assert "dim" in d
        assert "data" in d
        assert set(d["data"].keys()) >= {
            "tie_values",
            "tie_lengths",
            "sampling_interval",
        }


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
        decimated = coord.decimate(2)
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
        idx = coord.get_indexer(0.0, method="nearest")
        assert idx == 0
        idx = coord.get_indexer(10.0, method="nearest")
        assert idx == 3

    def test_get_indexer_nearest(self):
        coord = self.make_coord()
        idx = coord.get_indexer(0.5, method="nearest")
        assert idx in [0, 1]

    def test_get_indexer_out_of_bounds(self):
        coord = self.make_coord()
        with pytest.raises(KeyError):
            coord.get_indexer(100.0)


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


class TestSampledCoordinateSerialization:
    def test_to_from_dict(self):
        coord = SampledCoordinate(
            {
                "tie_values": [0.0, 10.0],
                "tie_lengths": [3, 2],
                "sampling_interval": 1.0,
            },
            dim="time",
        )
        d = coord.to_dict()
        # round-trip via Coordinate factory
        back = Coordinate.from_dict(d)
        assert isinstance(back, SampledCoordinate)
        assert back.equals(coord)


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
        assert coord.get_value(1) == np.datetime64("2000-01-01T00:00:01")
        assert coord.get_value(4) == np.datetime64("2000-01-01T00:00:11")
        with pytest.raises(IndexError):
            coord.get_value(5)

    def test_get_indexer_datetime_methods(self):
        coord = self.make_dt_coord()
        t = np.datetime64("2000-01-01T00:00:01.500")
        # exact required when method=None -> should raise
        with pytest.raises(KeyError):
            coord.get_indexer(t)
        # method variants
        assert coord.get_indexer(t, method="nearest") in [1, 2]
        assert coord.get_indexer(t, method="ffill") == 1
        assert coord.get_indexer(t, method="bfill") == 2
        # bounds
        with pytest.raises(KeyError):
            coord.get_indexer(np.datetime64("1999-12-31T23:59:59"))
        with pytest.raises(KeyError):
            coord.get_indexer(np.datetime64("2000-01-01T00:00:12"))
        # string input
        assert coord.get_indexer("2000-01-01T00:00:01.500", method="nearest") in [1, 2]
        # invalid method
        with pytest.raises(ValueError):
            coord.get_indexer(t, method="bad")

    def test_start_end_properties_datetime(self):
        coord = self.make_dt_coord()
        assert coord.start == np.datetime64("2000-01-01T00:00:00")
        # end is last tie_value + sampling_interval * last_length
        assert coord.end == np.datetime64("2000-01-01T00:00:12")


class TestSampledCoordinateIndexerEdgeCases:
    def test_invalid_method_raises(self):
        coord = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 1.0}
        )
        with pytest.raises(ValueError):
            coord.get_indexer(0.0, method="bad")

    def test_non_increasing_tie_values_raises(self):
        coord = SampledCoordinate(
            {"tie_values": [2.0, 1.0], "tie_lengths": [3, 2], "sampling_interval": 1.0}
        )
        with pytest.raises(ValueError):
            coord.get_indexer(2.0)


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
            dataset, variable_attrs = coord.to_dataset(dataset, variable_attrs)

        dataset["data"] = xr.DataArray(attrs=variable_attrs)
        coords = xd.Coordinates.from_dataset(dataset, "data")

        assert coords.equals(da.coords)

    def test_to_netcdf_and_back(self):
        expected = self.make_dataarray()

        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as file:
            expected.to_netcdf(file.name)
            result = xd.open_dataarray(file.name)
            assert result.equals(expected)


class TestGetSplitIndices:
    def test_get_split_indices_no_tolerance(self):
        coord = SampledCoordinate(
            {"tie_values": [0.0, 10.0], "tie_lengths": [3, 2], "sampling_interval": 1.0}
        )
        div_points = coord.get_split_indices()
        expected = np.array([3])  # indices where segments end
        assert np.array_equal(div_points, expected)

    def test_get_split_indices_with_tolerance(self):
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
            coord.__array_ufunc__(None, None)
        with pytest.raises(NotImplementedError):
            coord.__array_function__(None, None, None, None)
        with pytest.raises(NotImplementedError):
            coord.from_array(None)
