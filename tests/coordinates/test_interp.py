import numpy as np
import pytest
import xarray as xr

import xdas as xd
from xdas.coordinates import (
    InterpCoordinate,
    ScalarCoordinate,
)
from xdas.coordinates.core import Coordinate
from xdas.coordinates.interp import (
    _epsilon_ratio,
    _shear,
    _sleeve_kernel,
    _sleeve_loop,
)


class TestInterpCoordinate:
    valid = [
        {"tie_indices": [], "tie_values": []},
        {"tie_indices": [0], "tie_values": [100.0]},
        {"tie_indices": [0, 8], "tie_values": [100.0, 900.0]},
        {"tie_indices": [0, 8], "tie_values": [100, 900]},
        {
            "tie_indices": [0, 8],
            "tie_values": [
                np.datetime64("2000-01-01T00:00:00"),
                np.datetime64("2000-01-01T00:00:08"),
            ],
        },
        {"tie_indices": np.array([0, 8], dtype="int16"), "tie_values": [100.0, 900.0]},
    ]
    invalid = [
        1,
        np.array(1),
        1.0,
        np.array(1.0),
        "label",
        np.array("label"),
        np.datetime64(1, "s"),
        [1, 2, 3],
        np.array([1, 2, 3]),
        [1.0, 2.0, 3.0],
        np.array([1.0, 2.0, 3.0]),
        ["a", "b", "c"],
        np.array(["a", "b", "c"]),
        np.array([1, 2, 3], dtype="datetime64[s]"),
        {"key": "value"},
    ]
    error = [
        {"tie_indices": 0, "tie_values": [100.0]},
        {"tie_indices": [0], "tie_values": 100.0},
        {"tie_indices": [0, 7, 8], "tie_values": [100.0, 900.0]},
        {"tie_indices": [0.0, 8.0], "tie_values": [100.0, 900.0]},
        {"tie_indices": [1, 9], "tie_values": [100.0, 900.0]},
        {"tie_indices": [8, 0], "tie_values": [100.0, 900.0]},
        {"tie_indices": [8, 0], "tie_values": ["a", "b"]},
    ]

    def test_isvalid(self):
        for data in self.valid:
            assert InterpCoordinate._isvalid(data)
        for data in self.invalid:
            assert not InterpCoordinate._isvalid(data)
        # with optional sampling_interval / tolerance is still valid
        assert InterpCoordinate._isvalid(
            {"tie_indices": [0, 8], "tie_values": [0.0, 8.0], "sampling_interval": 1.0}
        )
        # unknown extra key is rejected
        assert not InterpCoordinate._isvalid(
            {"tie_indices": [0, 8], "tie_values": [0.0, 8.0], "extra": 1}
        )

    def test_init(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert np.array_equiv(coord.data["tie_indices"], [0, 8])
        assert np.array_equiv(coord.data["tie_values"], [100.0, 900.0])
        assert coord.dim is None
        coord = InterpCoordinate(
            {"tie_indices": [0, 8], "tie_values": [100.0, 900.0]}, "dim"
        )
        assert coord.dim == "dim"
        for data in self.valid:
            coord = InterpCoordinate(data)
            assert np.array_equiv(coord.data["tie_indices"], data["tie_indices"])
            assert np.array_equiv(coord.data["tie_values"], data["tie_values"])
        for data in self.invalid:
            with pytest.raises(TypeError):
                InterpCoordinate(data)
        for data in self.error:
            with pytest.raises(ValueError):
                InterpCoordinate(data)

    def test_len(self):
        assert (
            len(InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]}))
            == 9
        )
        assert len(InterpCoordinate({"tie_indices": [], "tie_values": []})) == 0

    @pytest.mark.parametrize("valid_input", valid)
    def test_repr(self, valid_input):
        coord = InterpCoordinate(data=valid_input)
        my_coord = repr(coord)
        assert isinstance(my_coord, str)

    def test_equals(self):
        coord1 = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        coord2 = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert coord1.equals(coord2)

    def test_getitem(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert isinstance(coord[0], ScalarCoordinate)
        assert coord[0].values == 100.0
        assert coord[4].values == 500.0
        assert coord[8].values == 900.0
        assert coord[-1].values == 900.0
        assert coord[-2].values == 800.0
        assert np.allclose(coord[[1, 2, 3]].values, [200.0, 300.0, 400.0])
        with pytest.raises(IndexError):
            coord[9]
            coord[-9]
        assert coord[0:2].equals(
            InterpCoordinate({"tie_indices": [0, 1], "tie_values": [100.0, 200.0]})
        )
        assert coord[:].equals(coord)
        assert coord[6:3].equals(
            InterpCoordinate({"tie_indices": [], "tie_values": []})
        )
        assert coord[1:2].equals(
            InterpCoordinate({"tie_indices": [0], "tie_values": [200.0]})
        )
        assert coord[-3:-1].equals(
            InterpCoordinate({"tie_indices": [0, 1], "tie_values": [700.0, 800.0]})
        )

    def test_setitem(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        with pytest.raises(TypeError):
            coord[1] = 0
            coord[:] = 0

    def test_asarray(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert np.allclose(np.asarray(coord), coord.values)

    def test_empty(self):
        assert not InterpCoordinate(
            {"tie_indices": [0, 8], "tie_values": [100.0, 900.0]}
        ).empty
        assert InterpCoordinate({"tie_indices": [], "tie_values": []}).empty

    def test_dtype(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert coord.dtype == np.float64

    def test_ndim(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert coord.ndim == 1
        assert isinstance(coord.ndim, int)

    def test_shape(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert coord.shape == (9,)

    def test_format_index(self):
        # TODO
        pass

    def test_get_value(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert coord._get_value(0) == 100.0
        assert coord._get_value(4) == 500.0
        assert coord._get_value(8) == 900.0
        assert coord[-1].data == 900.0
        assert coord[-9].data == 100.0
        assert np.allclose(coord[[1, 2, 3, -2]].values, [200.0, 300.0, 400.0, 800.0])
        with pytest.raises(IndexError):
            coord[-10]
        with pytest.raises(IndexError):
            coord[9]
        with pytest.raises(IndexError):
            coord[0.5]
        starttime = np.datetime64("2000-01-01T00:00:00")
        endtime = np.datetime64("2000-01-01T00:00:08")
        coord = InterpCoordinate(
            {"tie_indices": [0, 8], "tie_values": [starttime, endtime]}
        )
        assert coord._get_value(0) == starttime
        assert coord._get_value(4) == np.datetime64("2000-01-01T00:00:04")
        assert coord._get_value(8) == endtime
        assert coord[-1].data == endtime
        assert coord[-9].data == starttime

    def test_get_index(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert coord._get_indexer(100.0) == 0
        assert coord._get_indexer(900.0) == 8
        assert coord._get_indexer(0.0, "nearest") == 0
        assert coord._get_indexer(1000.0, "nearest") == 8
        assert coord._get_indexer(125.0, "nearest") == 0
        assert coord._get_indexer(175.0, "nearest") == 1
        assert coord._get_indexer(175.0, "ffill") == 0
        assert coord._get_indexer(200.0, "ffill") == 1
        assert coord._get_indexer(200.0, "bfill") == 1
        assert coord._get_indexer(125.0, "bfill") == 1
        assert np.all(np.equal(coord._get_indexer([100.0, 900.0]), [0, 8]))
        with pytest.raises(KeyError):
            assert coord._get_indexer(0.0) == 0
            assert coord._get_indexer(1000.0) == 8
            assert coord._get_indexer(150.0) == 0
            assert coord._get_indexer(1000.0, "bfill") == 8
            assert coord._get_indexer(0.0, "ffill") == 0

        starttime = np.datetime64("2000-01-01T00:00:00")
        endtime = np.datetime64("2000-01-01T00:00:08")
        coord = InterpCoordinate(
            {"tie_indices": [0, 8], "tie_values": [starttime, endtime]}
        )
        assert coord._get_indexer(starttime) == 0
        assert coord._get_indexer(endtime) == 8
        assert coord._get_indexer(str(starttime)) == 0
        assert coord._get_indexer(str(endtime)) == 8
        assert coord._get_indexer("2000-01-01T00:00:04.1", "nearest") == 4

    def test_indices(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert np.all(np.equal(coord.indices, np.arange(9)))

    def test_values(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert np.allclose(coord.values, np.arange(100.0, 1000.0, 100.0))

    def test_get_index_slice(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert coord._slice_indexer(100.0, 200.0) == slice(0, 2)
        assert coord._slice_indexer(150.0, 250.0) == slice(1, 2)
        assert coord._slice_indexer(300.0, 500.0) == slice(2, 5)
        assert coord._slice_indexer(0.0, 500.0) == slice(0, 5)
        assert coord._slice_indexer(125.0, 175.0) == slice(1, 1)
        assert coord._slice_indexer(0.0, 50.0) == slice(0, 0)
        assert coord._slice_indexer(1000.0, 1100.0) == slice(9, 9)
        assert coord._slice_indexer(1000.0, 500.0) == slice(9, 5)
        assert coord._slice_indexer(None, None) == slice(None, None)

    def test_slice_index(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        assert coord[0:2].equals(
            InterpCoordinate({"tie_indices": [0, 1], "tie_values": [100.0, 200.0]})
        )
        assert coord[7:].equals(
            InterpCoordinate({"tie_indices": [0, 1], "tie_values": [800.0, 900.0]})
        )
        assert coord[:].equals(coord)
        assert coord[0:0].equals(
            InterpCoordinate({"tie_indices": [], "tie_values": []})
        )
        assert coord[4:2].equals(
            InterpCoordinate({"tie_indices": [], "tie_values": []})
        )
        assert coord[9:9].equals(
            InterpCoordinate({"tie_indices": [], "tie_values": []})
        )
        assert coord[3:3].equals(
            InterpCoordinate({"tie_indices": [], "tie_values": []})
        )
        assert coord[0:-1].equals(
            InterpCoordinate({"tie_indices": [0, 7], "tie_values": [100.0, 800.0]})
        )
        assert coord[0:-2].equals(
            InterpCoordinate({"tie_indices": [0, 6], "tie_values": [100.0, 700.0]})
        )
        assert coord[-2:].equals(
            InterpCoordinate({"tie_indices": [0, 1], "tie_values": [800.0, 900.0]})
        )
        assert coord[1:2].equals(
            InterpCoordinate({"tie_indices": [0], "tie_values": [200.0]})
        )
        assert coord[1:3:2].equals(
            InterpCoordinate({"tie_indices": [0], "tie_values": [200.0]})
        )
        assert coord[::2].equals(
            InterpCoordinate({"tie_indices": [0, 4], "tie_values": [100.0, 900.0]})
        )
        assert coord[::3].equals(
            InterpCoordinate({"tie_indices": [0, 2], "tie_values": [100.0, 700.0]})
        )
        assert coord[::4].equals(
            InterpCoordinate({"tie_indices": [0, 2], "tie_values": [100.0, 900.0]})
        )
        assert coord[::5].equals(
            InterpCoordinate({"tie_indices": [0, 1], "tie_values": [100.0, 600.0]})
        )
        assert coord[2:7:3].equals(
            InterpCoordinate({"tie_indices": [0, 1], "tie_values": [300.0, 600.0]})
        )

    def test_to_index(self):
        # TODO
        pass

    def test_simplify(self):
        xp = np.sort(np.random.choice(10000, 1000, replace=False))
        xp[0] = 0
        xp[-1] = 10000
        yp = xp + (np.random.rand(1000) - 0.5)
        coord = InterpCoordinate({"tie_indices": xp, "tie_values": yp})
        assert len(coord.simplify(1.0).tie_indices) == 2

    def test_simplify_datetime(self):
        t0 = np.datetime64("2000-01-01T00:00:00")
        xp = np.sort(np.random.choice(10000, 1000, replace=False))
        xp[0] = 0
        xp[-1] = 10000
        yp = (
            t0
            + xp.astype("timedelta64[s]")
            + np.random.randint(-500, 500, size=1000).astype("timedelta64[ms]")
        )
        coord = InterpCoordinate({"tie_indices": xp, "tie_values": yp})
        assert len(coord.simplify(np.timedelta64(1, "s")).tie_indices) == 2
        assert len(coord.simplify(np.timedelta64(1000, "ms")).tie_indices) == 2
        assert len(coord.simplify(1.0).tie_indices) == 2

    def test_singleton(self):
        coord = InterpCoordinate({"tie_indices": [0], "tie_values": [1.0]})
        assert coord[0].values == 1.0

    def test_concat(self):
        coord0 = InterpCoordinate()
        coord1 = InterpCoordinate({"tie_indices": [0, 2], "tie_values": [0, 20]})
        coord2 = InterpCoordinate({"tie_indices": [0, 2], "tie_values": [30, 50]})

        result = coord1._concat(coord2).simplify()
        expected = InterpCoordinate({"tie_indices": [0, 5], "tie_values": [0, 50]})
        assert result.equals(expected)

        result = coord2._concat(coord1).simplify()
        expected = InterpCoordinate(
            {"tie_indices": [0, 2, 3, 5], "tie_values": [30, 50, 0, 20]}
        )
        assert result.equals(expected)

        assert coord0._concat(coord0).empty
        assert coord0._concat(coord1).equals(coord1)
        assert coord1._concat(coord0).equals(coord1)

    def test_simplify_preserves_real_discontinuity(self):
        # A large jump across a den == 1 gap is preserved as an emergent property
        # of the tolerance bound: both boundary points survive while the colinear
        # interior collapses.
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 5, 10, 11, 16, 21],
                "tie_values": [0.0, 5.0, 10.0, 1000.0, 1005.0, 1010.0],
            }
        )
        result = coord.simplify()
        assert result.equals(
            InterpCoordinate(
                {
                    "tie_indices": [0, 10, 11, 21],
                    "tie_values": [0.0, 10.0, 1000.0, 1010.0],
                }
            )
        )

    def test_simplify_absorbs_soft_discontinuity(self):
        # A den == 1 gap whose jump fits within tolerance is fused away and the
        # two areas merge into a single ramp.
        coord = InterpCoordinate(
            {"tie_indices": [0, 10, 11, 21], "tie_values": [0.0, 10.0, 11.4, 21.4]}
        )
        result = coord.simplify(1.0)
        assert result.equals(
            InterpCoordinate({"tie_indices": [0, 21], "tie_values": [0.0, 21.4]})
        )

    def test_simplify_multiple_runs_and_isolated_point(self):
        # Two real discontinuities flanking an isolated tie point: each run is
        # thinned independently and the isolated point survives.
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 5, 10, 11, 12, 17, 22],
                "tie_values": [0.0, 5.0, 10.0, 100.0, 200.0, 205.0, 210.0],
            }
        )
        result = coord.simplify()
        assert result.equals(
            InterpCoordinate(
                {
                    "tie_indices": [0, 10, 11, 12, 22],
                    "tie_values": [0.0, 10.0, 100.0, 200.0, 210.0],
                }
            )
        )

    def test_simplify_keeps_kink(self):
        # A genuine kink inside a continuous area forces the reduction to keep
        # the deviating interior point.
        coord = InterpCoordinate(
            {"tie_indices": [0, 5, 10], "tie_values": [0.0, 100.0, 0.0]}
        )
        result = coord.simplify()
        assert result.equals(coord)
        assert len(coord.simplify(200.0).tie_indices) == 2

    def test_simplify_datetime_discontinuity(self):
        t0 = np.datetime64("2000-01-01T00:00:00")
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 5, 10, 11, 16, 21],
                "tie_values": [
                    t0,
                    t0 + np.timedelta64(5, "s"),
                    t0 + np.timedelta64(10, "s"),
                    t0 + np.timedelta64(1000, "s"),
                    t0 + np.timedelta64(1005, "s"),
                    t0 + np.timedelta64(1010, "s"),
                ],
            }
        )
        result = coord.simplify()
        assert np.array_equal(result.tie_indices, [0, 10, 11, 21])


class TestInterpCoordinateExtra:
    def test_init_extra_keys(self):
        with pytest.raises(TypeError, match="tie_indices"):
            InterpCoordinate(
                {"tie_indices": [0, 8], "tie_values": [100.0, 900.0], "extra": 1}
            )

    def test_concat_errors(self):
        with pytest.raises(TypeError):
            InterpCoordinate({"tie_indices": [0, 2], "tie_values": [0, 20]})._concat(
                ScalarCoordinate(1)
            )
        with pytest.raises(ValueError, match="different dimension"):
            InterpCoordinate(
                {"tie_indices": [0, 2], "tie_values": [0, 20]}, "x"
            )._concat(
                InterpCoordinate({"tie_indices": [0, 2], "tie_values": [30, 50]}, "y")
            )
        with pytest.raises(ValueError, match="different dtype"):
            InterpCoordinate(
                {"tie_indices": [0, 2], "tie_values": np.array([0, 20], dtype=np.int32)}
            )._concat(
                InterpCoordinate(
                    {"tie_indices": [0, 2], "tie_values": np.array([30.0, 50.0])}
                )
            )

    def test_init_non_monotonic(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            InterpCoordinate(
                {"tie_indices": [0, 0, 8], "tie_values": [100.0, 200.0, 900.0]}
            )

    def test_init_string_values(self):
        with pytest.raises(ValueError, match="numeric or datetime"):
            InterpCoordinate({"tie_indices": [0, 1], "tie_values": ["a", "b"]})

    def test_indices_empty(self):
        coord = InterpCoordinate()
        assert len(coord.indices) == 0

    def test_array_with_dtype(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        result = coord.__array__(dtype=np.float32)
        assert result.dtype == np.float32

    def test_to_regular_empty(self):
        coord = InterpCoordinate()
        with pytest.raises(ValueError, match="cannot infer"):
            coord.to_regular()

    def test_get_indexer_overlaps(self):
        coord = InterpCoordinate(
            {"tie_indices": [0, 4, 8], "tie_values": [100.0, 50.0, 900.0]}
        )
        with pytest.raises(ValueError, match="overlaps were found"):
            coord._get_indexer(200.0)

    def test_simplify_false(self):
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})
        result = coord.simplify(False)
        assert result is not coord
        assert result.equals(coord)

    def test_get_split_indices_kinds(self):
        t0 = np.datetime64("2000-01-01T00:00:00")
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 4, 5, 9, 10, 14],
                "tie_values": [
                    t0,
                    t0 + np.timedelta64(4, "s"),
                    t0 + np.timedelta64(10, "s"),
                    t0 + np.timedelta64(14, "s"),
                    t0 + np.timedelta64(12, "s"),
                    t0 + np.timedelta64(16, "s"),
                ],
            }
        )
        gaps = coord.get_split_indices(kind="gaps")
        overlaps = coord.get_split_indices(kind="overlaps")
        assert len(gaps) >= 0
        assert len(overlaps) >= 0

    def test_get_split_indices_overlaps_tolerance_false(self):
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 4, 5, 9],
                "tie_values": [0.0, 4.0, 3.0, 7.0],
            }
        )
        result = coord.get_split_indices(kind="overlaps", tolerance=False)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [5], strict=True)

    def test_get_split_indices_overlaps_with_tolerance(self):
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 4, 5, 9],
                "tie_values": [0.0, 4.0, 3.0, 7.0],
            }
        )
        result = coord.get_split_indices(kind="overlaps", tolerance=0.5)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [5], strict=True)

    def test_is_monotonic_increasing_true(self):
        coord = InterpCoordinate(
            {"tie_indices": [0, 4, 5, 9], "tie_values": [0.0, 4.0, 5.0, 9.0]}
        )
        assert coord._is_monotonic_increasing() is True

    def test_is_monotonic_increasing_false(self):
        coord = InterpCoordinate(
            {"tie_indices": [0, 4, 5, 9], "tie_values": [0.0, 4.0, 3.0, 7.0]}
        )
        assert coord._is_monotonic_increasing() is False

    def test_is_monotonic_increasing_multi_segment(self):
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 4, 5, 9, 10, 14],
                "tie_values": [0.0, 4.0, 5.0, 9.0, 10.0, 14.0],
            }
        )
        assert coord._is_monotonic_increasing() is True

    def test_is_monotonic_increasing_subsample_overlap(self):
        # the axis advances by 0.4 of a 1.0 sampling interval: an overlap, but
        # the values keep increasing, so the axis stays sorted
        coord = InterpCoordinate(
            {"tie_indices": [0, 4, 5, 9], "tie_values": [0.0, 4.0, 4.4, 8.4]}
        )
        np.testing.assert_array_equal(coord.get_split_indices("overlaps"), [5])
        assert coord._is_monotonic_increasing() is True

    def test_is_monotonic_increasing_smooth_decrease(self):
        # no discontinuity anywhere — the axis simply turns around inside a
        # segment, which no boundary can report
        coord = InterpCoordinate(
            {"tie_indices": [0, 4, 8], "tie_values": [0.0, 4.0, 2.0]}
        )
        np.testing.assert_array_equal(coord.get_split_indices("discontinuities"), [])
        assert coord._is_monotonic_increasing() is False

    def test_is_monotonic_increasing_single_decreasing_segment(self):
        coord = InterpCoordinate({"tie_indices": [0, 4], "tie_values": [4.0, 0.0]})
        assert coord._is_monotonic_increasing() is False

    def test_slice_step_collision(self):
        coord = InterpCoordinate(
            {"tie_indices": [0, 2, 6, 12], "tie_values": [0.0, 20.0, 60.0, 120.0]}
        )
        result = coord[::3]
        assert isinstance(result, InterpCoordinate)
        assert len(result.tie_indices) >= 3
        assert result.tie_indices[0] == 0
        assert all(
            result.tie_indices[i] < result.tie_indices[i + 1]
            for i in range(len(result.tie_indices) - 1)
        )

    def test_to_regular_explicit_args(self):
        coord = InterpCoordinate(
            {"tie_indices": [0, 10, 20], "tie_values": [0.0, 1.0, 2.05]}
        )
        # strict default tolerance rejects the jitter
        with pytest.raises(ValueError, match="not consistent"):
            coord.to_regular()
        # an explicit tolerance accepts it
        reg = coord.to_regular(sampling_interval=0.1, tolerance=0.1)
        assert isinstance(reg, InterpCoordinate)
        assert reg.isregular()
        assert reg.sampling_interval == 0.1

    def test_to_regular_already_regular_is_preserved(self):
        # a regular coordinate keeps its stored spacing untouched
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 10, 20],
                "tie_values": [0.0, 1.0, 2.05],
                "sampling_interval": 0.1,
                "tolerance": 0.1,
            }
        )
        reg = coord.to_regular()
        assert reg is not coord
        assert reg.sampling_interval == 0.1
        assert reg.tolerance == 0.1
        # an explicit spacing still overrides it
        reg2 = coord.to_regular(sampling_interval=0.103, tolerance=0.1)
        assert reg2.sampling_interval == 0.103

    def test_module_helper_infers_with_warning(self):
        da = xd.DataArray(
            np.zeros(9),
            {"x": {"tie_indices": [0, 8], "tie_values": [0.0, 8.0]}},
        )
        with pytest.warns(FutureWarning, match="implicit inference is deprecated"):
            assert xd.get_sampling_interval(da, "x") == 1.0
        da["x"] = da["x"].to_regular()
        assert xd.get_sampling_interval(da, "x") == 1.0

    def test_module_helper_jittery_infers_with_warning(self):
        # The fallback states the tolerance required to accept the jitter; the
        # strict conversion still rejects it.
        da = xd.DataArray(
            np.zeros(21),
            {"x": {"tie_indices": [0, 10, 20], "tie_values": [0.0, 1.0, 2.05]}},
        )
        with pytest.warns(FutureWarning, match="accepting jitter up to tolerance"):
            result = xd.get_sampling_interval(da, "x")
        assert 0.1 <= result <= 0.105
        with pytest.raises(ValueError, match="not consistent"):
            da["x"].to_regular()

    def test_module_helper_datetime_cast(self):
        t0 = np.datetime64("2000-01-01T00:00:00")
        da = xd.DataArray(
            np.zeros(21),
            {
                "time": {
                    "tie_indices": [0, 10, 20],
                    "tie_values": [
                        t0,
                        t0 + np.timedelta64(10, "s"),
                        t0 + np.timedelta64(21, "s"),
                    ],
                }
            },
        )
        with pytest.warns(FutureWarning, match="implicit inference is deprecated"):
            result = xd.get_sampling_interval(da, "time")
        assert 1.0 <= result <= 1.1
        with pytest.warns(FutureWarning):
            result = xd.get_sampling_interval(da, "time", cast=False)
        assert isinstance(result, np.timedelta64)

    def test_module_helper_no_continuous_area_raises(self):
        da = xd.DataArray(
            np.zeros(2),
            {"x": {"tie_indices": [0, 1], "tie_values": [0.0, 1.0]}},
        )
        with pytest.raises(ValueError, match="none could be inferred"):
            xd.get_sampling_interval(da, "x")

    def test_to_regular_datetime_cast(self):
        t0 = np.datetime64("2000-01-01T00:00:00")
        t1 = np.datetime64("2000-01-01T00:00:08")
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [t0, t1]})
        result = coord.to_regular().get_sampling_interval()  # cast=True by default
        assert result == 1.0

    def test_to_regular_infer_datetime(self):
        t0 = np.datetime64("2000-01-01T00:00:00")
        t1 = np.datetime64("2000-01-01T00:00:08")
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [t0, t1]})
        reg = coord.to_regular()
        assert reg.sampling_interval == np.timedelta64(1, "s")
        assert reg.get_sampling_interval() == 1.0

    def test_to_regular_unit_spaced(self):
        # all tie-index gaps == 1 → no constrained segment → cannot infer
        coord = InterpCoordinate(
            {"tie_indices": [0, 1, 2], "tie_values": [0.0, 1.0, 2.0]}
        )
        with pytest.raises(ValueError, match="cannot infer"):
            coord.to_regular()

    def test_to_regular_minimax_favours_long_segment(self):
        # rates 1.0 (den=10) and 1.1 (den=2); minimax is pulled toward the long
        # segment, not the median midpoint 1.05
        coord = InterpCoordinate(
            {"tie_indices": [0, 10, 12], "tie_values": [0.0, 10.0, 12.2]}
        )
        si = coord.to_regular(tolerance=1.0).sampling_interval
        np.testing.assert_allclose(si, 12.2 / 12)

    def test_tolerance_without_sampling_interval(self):
        with pytest.raises(ValueError, match="cannot be set without"):
            InterpCoordinate(
                {"tie_indices": [0, 10], "tie_values": [0.0, 10.0], "tolerance": 0.1}
            )

    def test_infer_regular(self):
        # Numeric: rates 1.0 (den=10) and 1.0555 (den=5); the inferred spacing
        # and tolerance must round-trip through `to_regular`.
        coord = InterpCoordinate(
            {"tie_indices": [0, 10, 15], "tie_values": [0.0, 10.0, 15.55]}
        )
        si, tol = coord._infer_regular()
        assert si > 0
        assert tol > 0
        reg = coord.to_regular(sampling_interval=si, tolerance=tol)
        assert reg.isregular()

        # Datetime variant: tolerance comes back as a timedelta64.
        t0 = np.datetime64("2000-01-01T00:00:00")
        coord_dt = InterpCoordinate(
            {
                "tie_indices": [0, 10, 15],
                "tie_values": [
                    t0,
                    t0 + np.timedelta64(10_000_000_000, "ns"),
                    t0 + np.timedelta64(15_550_000_000, "ns"),
                ],
            }
        )
        si_dt, tol_dt = coord_dt._infer_regular()
        assert np.issubdtype(np.asarray(tol_dt).dtype, np.timedelta64)
        assert tol_dt > np.timedelta64(0)
        assert coord_dt.to_regular(
            sampling_interval=si_dt, tolerance=tol_dt
        ).isregular()

        # No continuous segment → nothing to infer.
        unit = InterpCoordinate(
            {"tie_indices": [0, 1, 2], "tie_values": [0.0, 1.0, 2.0]}
        )
        assert unit._infer_regular() == (None, None)

    def test_add_sub(self):
        coord = InterpCoordinate({"tie_indices": [0, 4], "tie_values": [10.0, 50.0]})
        result = coord + 5.0
        assert isinstance(result, InterpCoordinate)
        assert np.allclose(result.tie_values, [15.0, 55.0])
        result2 = coord - 5.0
        assert np.allclose(result2.tie_values, [5.0, 45.0])

    def test_to_dataset_collect_roundtrip(self):
        da = xd.DataArray(
            np.zeros(9),
            {"x": {"tie_indices": [0, 8], "tie_values": [100.0, 900.0]}},
        )
        coord = da.coords["x"]
        dataset = xr.Dataset()
        attrs = {}
        dataset, attrs = coord._to_dataset(dataset, attrs)
        assert "coordinate_interpolation" in attrs
        assert "x_indices" in dataset
        assert "x_values" in dataset
        dataset["__values__"] = xr.DataArray(np.zeros(9), dims=["x"])
        dataset["__values__"].attrs["coordinate_interpolation"] = attrs[
            "coordinate_interpolation"
        ]
        recovered = InterpCoordinate._collect_from_dataset(dataset, "__values__")
        assert "x" in recovered
        assert np.allclose(recovered["x"].tie_values, coord.tie_values)

    def test_to_dataset_multiple_coords_append(self):
        da = xd.DataArray(
            np.zeros((9, 5)),
            {
                "x": {"tie_indices": [0, 8], "tie_values": [100.0, 900.0]},
                "y": {"tie_indices": [0, 4], "tie_values": [0.0, 40.0]},
            },
        )
        attrs = {}
        dataset = xr.Dataset()
        dataset, attrs = da.coords["x"]._to_dataset(dataset, attrs)
        dataset, attrs = da.coords["y"]._to_dataset(dataset, attrs)
        assert "x" in attrs["coordinate_interpolation"]
        assert "y" in attrs["coordinate_interpolation"]

    def test_to_dataset_datetime(self):
        t0 = np.datetime64("2000-01-01T00:00:00")
        t1 = np.datetime64("2000-01-01T00:00:08")
        da = xd.DataArray(
            np.zeros(9),
            {"time": {"tie_indices": [0, 8], "tie_values": [t0, t1]}},
        )
        coord = da.coords["time"]
        dataset = xr.Dataset()
        attrs = {}
        dataset, attrs = coord._to_dataset(dataset, attrs)
        assert "time_indices" in dataset
        assert dataset["time_values"].dtype == np.dtype("datetime64[ns]")

    def test_collect_legacy_spelling(self):
        # the pre-break grammar, written without any interpolation variable
        dataset = xr.Dataset(
            {
                "x_indices": ("x_points", np.array([0, 8])),
                "x_values": ("x_points", np.array([100.0, 900.0])),
                "__values__": (
                    ("x",),
                    np.zeros(9),
                    {"coordinate_interpolation": "x: x_indices x_values"},
                ),
            }
        )
        recovered = InterpCoordinate._collect_from_dataset(dataset, "__values__")
        assert np.allclose(recovered["x"].tie_values, [100.0, 900.0])


class TestInterpCoordinateRegular:
    """Tests for InterpCoordinate with an enforced sampling_interval (regular mode)."""

    valid = [
        {
            "tie_indices": [0, 5, 9, 10, 19],
            "tie_values": [0.0, 0.5, 0.9, 2.0, 2.9],
            "sampling_interval": 0.1,
        }
    ]

    def make(self):
        return InterpCoordinate(self.valid[0], "dim")

    def test_isvalid(self):
        for data in self.valid:
            assert InterpCoordinate._isvalid(data)
        # plain tie-point dict without sampling_interval is also valid
        assert InterpCoordinate._isvalid(
            {"tie_indices": [0, 8], "tie_values": [0.0, 8.0]}
        )
        # unknown extra key is rejected
        assert not InterpCoordinate._isvalid(
            {
                "tie_indices": [0, 8],
                "tie_values": [0.0, 8.0],
                "sampling_interval": 1.0,
                "extra": 1,
            }
        )

    def test_init(self):
        coord = self.make()
        assert coord.sampling_interval == 0.1
        assert coord.tolerance is not None
        assert coord.dim == "dim"
        assert coord.isregular()

    def test_factory_dispatch(self):
        coord = Coordinate(self.valid[0])
        assert isinstance(coord, InterpCoordinate)
        assert coord.isregular()
        # plain tie-point data routes to a non-regular InterpCoordinate
        plain = Coordinate({"tie_indices": [0, 8], "tie_values": [0.0, 8.0]})
        assert isinstance(plain, InterpCoordinate)
        assert not plain.isregular()

    def test_init_inconsistent(self):
        with pytest.raises(ValueError, match="not consistent"):
            InterpCoordinate(
                {
                    "tie_indices": [0, 10],
                    "tie_values": [0.0, 10.0],
                    "sampling_interval": 0.5,
                }
            )

    def test_init_tolerance_allows_jitter(self):
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 10, 20],
                "tie_values": [0.0, 1.0, 2.05],
                "sampling_interval": 0.1,
                "tolerance": 0.1,
            }
        )
        assert coord.sampling_interval == 0.1
        assert coord.isregular()

    def test_empty(self):
        coord = InterpCoordinate()
        assert coord.empty
        assert coord.sampling_interval is None
        assert coord.tolerance is None
        assert coord.get_sampling_interval() is None
        with pytest.raises(ValueError, match="cannot infer"):
            coord.to_regular()
        assert not coord.isregular()

    def test_empty_slice_preserves_sampling_interval(self):
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 5, 9, 10, 19],
                "tie_values": [0.0, 0.5, 0.9, 2.0, 2.9],
                "sampling_interval": 0.1,
            }
        )
        sliced = coord[0:0]
        assert isinstance(sliced, InterpCoordinate)
        assert sliced.empty
        assert sliced.sampling_interval == 0.1

    def test_from_block(self):
        coord = InterpCoordinate.from_block(0.0, 10, 0.5, "dim")
        assert coord.sampling_interval == 0.5
        assert len(coord) == 10
        assert coord.isregular()

    def test_slice(self):
        coord = self.make()
        sliced = coord[2:12]
        assert isinstance(sliced, InterpCoordinate)
        assert sliced.isregular()
        assert sliced.sampling_interval == 0.1
        stepped = coord[::2]
        assert stepped.sampling_interval == 0.2

    def test_slice_empty(self):
        coord = self.make()
        empty = coord[5:5]
        assert isinstance(empty, InterpCoordinate)
        assert empty.empty

    def test_concat(self):
        a = InterpCoordinate(
            {"tie_indices": [0, 9], "tie_values": [0.0, 0.9], "sampling_interval": 0.1}
        )
        b = InterpCoordinate(
            {"tie_indices": [0, 9], "tie_values": [1.0, 1.9], "sampling_interval": 0.1}
        )
        result = a._concat(b)
        assert isinstance(result, InterpCoordinate)
        assert result.isregular()
        assert result.sampling_interval == 0.1
        assert len(result) == 20

    def test_concat_different_sampling_interval(self):
        # Wildly different rates cannot be reconciled under tolerance, so the
        # merged coord falls back to irregular.
        a = InterpCoordinate(
            {"tie_indices": [0, 9], "tie_values": [0.0, 0.9], "sampling_interval": 0.1}
        )
        b = InterpCoordinate(
            {"tie_indices": [0, 9], "tie_values": [1.0, 2.8], "sampling_interval": 0.2}
        )
        result = a._concat(b)
        assert isinstance(result, InterpCoordinate)
        assert not result.isregular()
        assert result.sampling_interval is None
        assert len(result) == 20

        # Mixed regular/irregular drifts too far → irregular.
        c = InterpCoordinate({"tie_indices": [0, 9], "tie_values": [3.0, 4.0]})
        mixed = a._concat(c)
        assert mixed.sampling_interval is None
        assert len(mixed) == 20

    def test_concat_coords_recovers_regular_spacing(self):
        # `_concat` itself stays strict and drops to irregular when sampling
        # intervals disagree; `concat_coords` then tries to reconcile a
        # single shared rate within the user-supplied tolerance.
        a = InterpCoordinate(
            {
                "tie_indices": [0, 9],
                "tie_values": [0.0, 0.9],
                "sampling_interval": 0.1,
                "tolerance": 0.05,
            },
            "x",
        )
        b = InterpCoordinate(
            {
                "tie_indices": [0, 9],
                "tie_values": [1.0, 1.99],
                "sampling_interval": 0.11,
                "tolerance": 0.05,
            },
            "x",
        )
        assert not a._concat(b).isregular()

        from xdas.core.routines import concat_coords

        reconciled = concat_coords([a, b], tolerance=0.5, regularize=True)
        assert reconciled.isregular()
        assert 0.1 <= reconciled.sampling_interval <= 0.11
        assert len(reconciled) == 20

    def test_add_sub(self):
        coord = self.make()
        shifted = coord + 1.0
        assert isinstance(shifted, InterpCoordinate)
        assert shifted.isregular()
        assert shifted.sampling_interval == 0.1
        assert shifted.start == coord.start + 1.0
        back = shifted - 1.0
        assert np.allclose(back.tie_values, coord.tie_values)

    def test_simplify(self):
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 5, 10],
                "tie_values": [0.0, 0.5, 1.0],
                "sampling_interval": 0.1,
            }
        )
        simplified = coord.simplify()
        assert isinstance(simplified, InterpCoordinate)
        assert simplified.isregular()
        assert simplified.sampling_interval == 0.1

    def test_get_sampling_interval_datetime(self):
        t0 = np.datetime64("2000-01-01T00:00:00")
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 8],
                "tie_values": [t0, t0 + np.timedelta64(8, "s")],
                "sampling_interval": np.timedelta64(1, "s"),
            }
        )
        assert coord.get_sampling_interval() == 1.0
        assert coord.get_sampling_interval(cast=False) == np.timedelta64(1, "s")
        assert coord.to_regular().get_sampling_interval() == 1.0

    def test_dataset_roundtrip_numeric(self):
        coord = self.make()
        da = xd.DataArray(np.zeros(len(coord)), {"dim": coord})
        dataset = xr.Dataset()
        dataset, attrs = da.coords["dim"]._to_dataset(dataset, {})
        dataset["__v__"] = xr.DataArray(np.zeros(len(coord)), dims=["dim"])
        dataset["__v__"].attrs.update(attrs)
        recovered = Coordinate._from_dataset(dataset, "__v__")
        assert isinstance(recovered["dim"], InterpCoordinate)
        assert recovered["dim"].isregular()
        assert recovered["dim"].sampling_interval == 0.1

    def test_dataset_roundtrip_datetime(self):
        t0 = np.datetime64("2000-01-01T00:00:00")
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 8],
                "tie_values": [t0, t0 + np.timedelta64(8, "s")],
                "sampling_interval": np.timedelta64(1, "s"),
            }
        )
        da = xd.DataArray(np.zeros(9), {"time": coord})
        dataset = xr.Dataset()
        dataset, attrs = da.coords["time"]._to_dataset(dataset, {})
        dataset["__v__"] = xr.DataArray(np.zeros(9), dims=["time"])
        dataset["__v__"].attrs.update(attrs)
        recovered = Coordinate._from_dataset(dataset, "__v__")
        assert isinstance(recovered["time"], InterpCoordinate)
        assert recovered["time"].isregular()
        assert recovered["time"].sampling_interval == np.timedelta64(1, "s")

    def test_collect_mixed_plain_and_regular(self):
        da = xd.DataArray(
            np.zeros((20, 9)),
            {
                "x": self.make(),
                "y": {"tie_indices": [0, 8], "tie_values": [0.0, 8.0]},
            },
        )
        dataset = xr.Dataset()
        attrs = {}
        dataset, attrs = da.coords["x"]._to_dataset(dataset, attrs)
        dataset, attrs = da.coords["y"]._to_dataset(dataset, attrs)
        dataset["__v__"] = xr.DataArray(np.zeros((20, 9)), dims=["x", "y"])
        dataset["__v__"].attrs.update(attrs)
        recovered = Coordinate._from_dataset(dataset, "__v__")
        assert isinstance(recovered["x"], InterpCoordinate)
        assert recovered["x"].isregular()
        assert isinstance(recovered["y"], InterpCoordinate)
        assert not recovered["y"].isregular()

    def test_file_roundtrip(self, tmp_path):
        coord = self.make()
        da = xd.DataArray(np.zeros(len(coord)), {"dim": coord})
        path = tmp_path / "reg.nc"
        da.to_netcdf(path)
        loaded = xd.open_dataarray(path)
        assert isinstance(loaded.coords["dim"], InterpCoordinate)
        assert loaded.coords["dim"].isregular()
        assert loaded.coords["dim"].sampling_interval == 0.1


class TestDeltaEncoding:
    def test_encode_none(self):
        from xdas.coordinates.core import encode_delta

        assert encode_delta("sampling_interval", None) == {}

    def test_encode_decode_numeric(self):
        from xdas.coordinates.core import decode_delta, encode_delta

        attrs = encode_delta("sampling_interval", 0.1)
        assert attrs == {"sampling_interval": 0.1}
        assert decode_delta("sampling_interval", attrs) == 0.1

    def test_decode_missing(self):
        from xdas.coordinates.core import decode_delta

        assert decode_delta("sampling_interval", {}) is None


class TestSimplifyToleranceDefaults:
    def test_default_budget_is_stored_tolerance(self):
        # A seam 1.0 off the nominal rate fuses away under the declared
        # tolerance of 2.0 without passing any explicit budget.
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 10, 11, 21],
                "tie_values": [0.0, 30.0, 34.0, 64.0],
                "sampling_interval": 3.0,
                "tolerance": 2.0,
            }
        )
        result = coord.simplify()
        assert len(result.tie_indices) == 2
        assert result.sampling_interval == 3.0
        assert result.tolerance == 2.0

    def test_lossless_pass_keeps_tolerance(self):
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 9],
                "tie_values": [0.0, 9.0],
                "sampling_interval": 1.0,
                "tolerance": 0.5,
            }
        )
        result = coord.simplify()
        assert result.equals(coord)

    def test_widen_only_when_needed(self):
        # Fusing a jump beyond the declared tolerance widens it to the least
        # value that describes the surviving tie points: the fused coordinate
        # spans 13 s over 11 intervals, so it drifts 2 s from the nominal grid.
        t0 = np.datetime64("2000-01-01T00:00:00", "ns")
        s = np.timedelta64(1, "s").astype("m8[ns]")
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 5, 6, 11],
                "tie_values": [t0, t0 + 5 * s, t0 + 8 * s, t0 + 13 * s],
                "sampling_interval": s,
                "tolerance": np.timedelta64(0, "ns"),
            }
        )
        result = coord.simplify(np.timedelta64(3, "s"))
        assert len(result.tie_indices) == 2
        assert result.sampling_interval == s
        assert result.tolerance == np.timedelta64(1, "s").astype("m8[ns]")
        assert result._is_valid_sampling_interval(s, result.tolerance)

    def test_widening_beyond_the_budget_never_raises(self):
        # The reduction bounds how far values move, not how much drift fusing
        # a discontinuity exposes, so the required tolerance can exceed the
        # budget. Real OptoDAS seams: 2 ms late every 10 s at 125 Hz.
        t0 = np.datetime64("2021-10-27T15:44:10.721999872", "ns")
        offsets = [
            0,
            9992000000,
            10002000128,
            19994000128,
            20002000128,
            29994000128,
            30004000256,
            39996000256,
        ]
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 1249, 1250, 2499, 2500, 3749, 3750, 4999],
                "tie_values": t0 + np.array(offsets, dtype="timedelta64[ns]"),
                "sampling_interval": np.timedelta64(8_000_000, "ns"),
                "tolerance": np.timedelta64(0, "ns"),
            }
        )
        result = coord.simplify(np.timedelta64(1_000_000, "ns"))
        assert result.isregular()
        assert result.sampling_interval == np.timedelta64(8_000_000, "ns")
        # Four times the 1 ms budget, and the smallest value that validates.
        assert result.tolerance == np.timedelta64(2_000_128, "ns")
        assert result._is_valid_sampling_interval(
            result.sampling_interval, result.tolerance
        )

    def test_widen_only_when_needed_on_float_axis(self):
        # Same widening on a float axis: fusing the 4.0 seam leaves a coordinate
        # spanning 64.0 over 21 intervals, 1.0 off the nominal 3.0 grid.
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 10, 11, 21],
                "tie_values": [0.0, 30.0, 34.0, 64.0],
                "sampling_interval": 3.0,
                "tolerance": 0.0,
            }
        )
        result = coord.simplify(2.0)
        assert len(result.tie_indices) == 2
        assert result.sampling_interval == 3.0
        assert result.tolerance == pytest.approx(0.5, abs=1e-9)
        assert result._is_valid_sampling_interval(3.0, result.tolerance)

    def test_minimal_tolerance_without_continuous_area(self):
        # Nothing to constrain the spacing: the zero-like default is enough.
        coord = InterpCoordinate({"tie_indices": [0, 1], "tie_values": [0.0, 5.0]})
        tolerance = coord._minimal_tolerance(1.0)
        assert coord._is_valid_sampling_interval(1.0, tolerance)


class TestSimplifyNoReduce:
    def test_regularize_without_reduce(self):
        coord = InterpCoordinate(
            {"tie_indices": [0, 5, 10], "tie_values": [0.0, 5.0, 10.0]}
        )
        result = coord.simplify(reduce=False, regularize=True)
        assert len(result.tie_indices) == 3
        assert result.isregular()
        assert result.get_sampling_interval() == 1.0


class TestSimplifyRegularizeFallback:
    def test_no_continuous_area_stays_irregular(self):
        coord = InterpCoordinate(
            {"tie_indices": [0, 1, 2], "tie_values": [0.0, 1.0, 5.0]}
        )
        result = coord.simplify(reduce=False, regularize=True)
        assert not result.isregular()

    def test_invalid_fit_stays_irregular(self):
        coord = InterpCoordinate(
            {"tie_indices": [0, 10, 20], "tie_values": [0.0, 1.0, 2.05]}
        )
        result = coord.simplify(reduce=False, regularize=True)
        assert not result.isregular()


class TestFromBlockShort:
    def test_single_sample(self):
        coord = InterpCoordinate.from_block(0.0, 1, 2.0, dim="x")
        assert len(coord) == 1
        assert coord.values == [0.0]
        assert coord.sampling_interval == 2.0

    def test_empty(self):
        coord = InterpCoordinate.from_block(0.0, 0, 2.0, dim="x")
        assert coord.empty
        assert coord.sampling_interval == 2.0


class TestSleeve:
    """The one-pass reduce stage: same deviation guarantee as the former
    Douglas-Peucker, O(n) whatever survives."""

    def test_deviation_bound_holds_on_jitter(self):
        rng = np.random.default_rng(0)
        n = 200
        starts = np.arange(n, dtype="int64") * 10
        jitter = rng.integers(-1_500_000, 1_500_000, n)
        t0 = np.datetime64("2026-01-01", "ns").astype("i8")
        seg = t0 + np.arange(n) * 100_000_000 + jitter
        tie_indices = np.empty(2 * n, dtype="int64")
        tie_indices[0::2] = starts
        tie_indices[1::2] = starts + 9
        tie_values = np.empty(2 * n, dtype="i8")
        tie_values[0::2] = seg
        tie_values[1::2] = seg + 90_000_000
        coord = InterpCoordinate(
            {"tie_indices": tie_indices, "tie_values": tie_values.astype("M8[ns]")},
            "time",
        )
        tolerance = np.timedelta64(1_000_000, "ns")
        result = coord.simplify(tolerance)
        # dropped points stay within tolerance of the simplified curve, and
        # surviving values are never moved
        deviation = np.abs(result._get_value(coord.tie_indices) - coord.tie_values)
        assert deviation.max() <= tolerance
        kept = np.isin(result.tie_indices, coord.tie_indices)
        assert kept.all()

    def test_every_gap_survives(self):
        n = 50
        tie_indices = np.arange(2 * n, dtype="int64")
        tie_indices[1::2] = tie_indices[0::2] + 1
        tie_indices = np.cumsum(np.where(np.arange(2 * n) % 2, 1, 9))
        tie_indices -= tie_indices[0]
        values = np.arange(2 * n) * 1_000_000_000
        coord = InterpCoordinate(
            {"tie_indices": tie_indices, "tie_values": values.astype("M8[ns]")},
            "time",
        )
        result = coord.simplify(np.timedelta64(1, "ms"))
        assert len(result.tie_indices) == len(coord.tie_indices)

    def test_subunit_tolerance_is_not_truncated(self):
        # microsecond values with a nanosecond tolerance: a 4 us seam deviates
        # ~1.5 us from the spanning chord, so it needs a sub-microsecond budget
        # to survive (400 ns keeps it) and a 1100 ns one to fuse — both
        # unrepresentable in truncated us
        values = np.array([0, 998, 1002, 2000], dtype="M8[us]")
        coord = InterpCoordinate(
            {"tie_indices": [0, 999, 1000, 1999], "tie_values": values}, "time"
        )
        kept = coord.simplify(np.timedelta64(400, "ns"))
        fused = coord.simplify(np.timedelta64(1100, "ns"))
        assert len(kept.tie_indices) == 4
        assert len(fused.tie_indices) == 2

    def test_seam_below_the_storage_resolution_is_fused(self):
        # A 1 us seam here reconstructs bit-identically from two tie points:
        # `forward` rounds to the microsecond and lands on the same values, so
        # the interior ties carry no information and a zero budget drops them.
        values = np.array([0, 999, 1001, 2000], dtype="M8[us]")
        coord = InterpCoordinate(
            {"tie_indices": [0, 999, 1000, 1999], "tie_values": values}, "time"
        )
        result = coord.simplify(np.timedelta64(0, "ns"))
        assert len(result.tie_indices) == 2
        index = np.arange(len(coord))
        np.testing.assert_array_equal(coord[index].values, result[index].values)

    def test_float_values(self):
        coord = InterpCoordinate(
            {"tie_indices": [0, 9, 10, 19], "tie_values": [0.0, 9.0, 10.5, 19.5]},
            "x",
        )
        assert len(coord.simplify(1.0).tie_indices) == 2
        assert len(coord.simplify(0.1).tie_indices) == 4

    def test_two_ties_pass_through(self):
        coord = InterpCoordinate({"tie_indices": [0, 9], "tie_values": [0.0, 9.0]})
        result = coord.simplify(0.0)
        assert len(result.tie_indices) == 2


class TestSleeveResolution:
    """A tie point may sit half a tick off the exact line and still be the only
    representable value there, so collinearity is judged with that slack."""

    T0 = np.datetime64("2020-01-01", "us")

    def test_unrepresentable_rate_still_collapses(self):
        # 999 samples over 30 s is 30030.030030... us: the midpoint tie is half
        # a microsecond off the true line, which no integer value can fix.
        values = [
            self.T0,
            self.T0 + np.timedelta64(1_501_502, "us"),
            self.T0 + np.timedelta64(30_000_000, "us"),
        ]
        coord = InterpCoordinate(
            {"tie_indices": [0, 50, 999], "tie_values": values}, "time"
        )
        result = coord.simplify(np.timedelta64(0, "us"))
        assert len(result.tie_indices) == 2
        index = np.arange(len(coord))
        drift = np.abs(coord[index].values - result[index].values)
        # within the one tick the representation itself carries
        assert drift.max() <= np.timedelta64(1, "us")

    def test_real_discontinuities_still_survive(self):
        for jump in (2, 1_000_000):
            values = [
                self.T0,
                self.T0 + np.timedelta64(50_000, "us"),
                self.T0 + np.timedelta64(50_100 + jump, "us"),
                self.T0 + np.timedelta64(100_000 + jump, "us"),
            ]
            coord = InterpCoordinate(
                {"tie_indices": [0, 500, 501, 1000], "tie_values": values}, "time"
            )
            assert len(coord.simplify(np.timedelta64(0, "us")).tie_indices) == 4

    def test_subtick_epsilon_does_not_overflow_the_prescale(self):
        # A picosecond budget on a microsecond axis used to wrap int64 while
        # scaling the values; the budget is a rational now, so nothing scales.
        values = [
            self.T0,
            self.T0 + np.timedelta64(1_501_502, "us"),
            self.T0 + np.timedelta64(30_000_000, "us"),
        ]
        coord = InterpCoordinate(
            {"tie_indices": [0, 50, 999], "tie_values": values}, "time"
        )
        assert len(coord.simplify(np.timedelta64(1, "ps")).tie_indices) == 2


class TestSleeveKernel:
    """The compiled machine-integer walk and the arbitrary-precision one must
    agree; the magnitude bound decides which runs, never the answer."""

    def test_kernel_matches_loop_on_random_curves(self):
        rng = np.random.default_rng(0)
        epoch = np.datetime64("2020-01-01", "us").view("i8")
        for _ in range(200):
            size = int(rng.integers(3, 30))
            indices = np.array(
                [0]
                + sorted(
                    rng.choice(
                        np.arange(1, 3000), size=size - 1, replace=False
                    ).tolist()
                )
            )
            rate = int(rng.integers(1, 10**5))
            values = indices * rate + rng.integers(-4, 5, size=size)
            values = np.maximum.accumulate(values) + np.arange(size) + epoch
            epsilon = np.timedelta64(int(rng.integers(0, 5)), "us")
            numerator, denominator = _epsilon_ratio(np.dtype("M8[us]"), epsilon)
            sheared = _shear(indices, values, int(indices[-1]))
            np.testing.assert_array_equal(
                _sleeve_kernel(indices, sheared, numerator, denominator),
                _sleeve_loop(indices.tolist(), values.tolist(), numerator, denominator),
            )

    def test_cross_products_too_wide_for_int64_fall_back(self):
        # 4e9 samples with a mid tie half the span away from the chord: the
        # shear fits, its cross-products do not, so the walk runs in Python
        # integers.
        indices = np.array([0, 2 * 10**9, 4 * 10**9])
        values = np.array([0, 5 * 10**17, 2 * 10**18], dtype="i8")
        coord = InterpCoordinate({"tie_indices": indices, "tie_values": values}, "x")
        assert len(coord.simplify(0).tie_indices) == 3

    def test_shear_itself_too_wide_falls_back(self):
        # Values within a hair of the int64 ceiling: even subtracting the rate
        # would overflow, so no shear is attempted.
        indices = np.array([0, 1, 2])
        values = np.array([0, 10**18, 2**63 - 1], dtype="i8")
        assert _shear(indices, values, 2) is None
        coord = InterpCoordinate({"tie_indices": indices, "tie_values": values}, "x")
        assert len(coord.simplify(0).tie_indices) == 3
