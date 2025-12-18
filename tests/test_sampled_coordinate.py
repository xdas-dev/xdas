import numpy as np
import pytest

from xdas.core.coordinates import SampledCoordinate, ScalarCoordinate, DenseCoordinate
import pandas as pd


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

    def test_init_validation_numeric(self):
        # valid numeric
        coord = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 1.0}
        )
        assert len(coord) == 3
        assert coord.start == 0.0
        assert coord.end == 3.0

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

        # invalid: datetime with numeric sampling interval
        with pytest.raises(ValueError):
            SampledCoordinate(
                {"tie_values": [t0], "tie_lengths": [2], "sampling_interval": 1}
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
        # vectorized
        vals = coord.get_value([0, 2, 3, 4])
        assert np.array_equal(vals, np.array([0.0, 2.0, 10.0, 11.0]))
        # bounds
        with pytest.raises(IndexError):
            coord.get_value(-6)
        with pytest.raises(IndexError):
            coord.get_value(5)

    def test_getitem(self):
        coord = self.make_coord()
        # scalar -> ScalarCoordinate
        item = coord[1]
        assert isinstance(item, ScalarCoordinate)
        assert item.values == 1.0
        # slice -> SampledCoordinate or compatible
        sub = coord[1:4]
        assert isinstance(sub, SampledCoordinate)
        # array -> DenseCoordinate of values
        arr = coord[[0, 4]]
        assert isinstance(arr, DenseCoordinate)
        assert np.array_equal(arr.values, np.array([0.0, 11.0]))

    def test_repr(self):
        # Just ensure it returns a string
        coord = self.make_coord()
        assert isinstance(repr(coord), str)


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


class TestSampledCoordinateAppendErrors:
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


class TestSampledCoordinateAppend:
    def test_append_two_coords(self):
        coord1 = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 1.0}
        )
        coord2 = SampledCoordinate(
            {"tie_values": [10.0], "tie_lengths": [2], "sampling_interval": 1.0}
        )
        result = coord1.append(coord2)
        assert len(result) == 5
        assert result.tie_values[0] == 0.0
        assert result.tie_values[1] == 10.0

    def test_append_empty(self):
        coord1 = SampledCoordinate(
            {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 1.0}
        )
        coord2 = SampledCoordinate()
        assert coord1.append(coord2).equals(coord1)
        assert coord2.append(coord1).equals(coord1)


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
        simplified = coord.simplify(tolerance=0.1)
        # If continuous (end of first == start of second), should merge
        assert len(simplified.tie_values) <= 2


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
        from xdas.core.coordinates import Coordinate

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
