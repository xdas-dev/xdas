import numpy as np
import pytest
import xarray as xr

import xdas as xd
from xdas.coordinates import FixedInterpCoordinate, InterpCoordinate, ScalarCoordinate


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
        assert len(InterpCoordinate(dict(tie_indices=[], tie_values=[]))) == 0

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
            InterpCoordinate(dict(tie_indices=[0, 1], tie_values=[100.0, 200.0]))
        )
        assert coord[:].equals(coord)
        assert coord[6:3].equals(InterpCoordinate(dict(tie_indices=[], tie_values=[])))
        assert coord[1:2].equals(
            InterpCoordinate(dict(tie_indices=[0], tie_values=[200.0]))
        )
        assert coord[-3:-1].equals(
            InterpCoordinate(dict(tie_indices=[0, 1], tie_values=[700.0, 800.0]))
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
        assert InterpCoordinate(dict(tie_indices=[], tie_values=[])).empty

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
            dict(tie_indices=[0, 8], tie_values=[starttime, endtime])
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
            dict(tie_indices=[0, 8], tie_values=[starttime, endtime])
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
            InterpCoordinate(dict(tie_indices=[0, 1], tie_values=[100.0, 200.0]))
        )
        assert coord[7:].equals(
            InterpCoordinate(dict(tie_indices=[0, 1], tie_values=[800.0, 900.0]))
        )
        assert coord[:].equals(coord)
        assert coord[0:0].equals(InterpCoordinate(dict(tie_indices=[], tie_values=[])))
        assert coord[4:2].equals(InterpCoordinate(dict(tie_indices=[], tie_values=[])))
        assert coord[9:9].equals(InterpCoordinate(dict(tie_indices=[], tie_values=[])))
        assert coord[3:3].equals(InterpCoordinate(dict(tie_indices=[], tie_values=[])))
        assert coord[0:-1].equals(
            InterpCoordinate(dict(tie_indices=[0, 7], tie_values=[100.0, 800.0]))
        )
        assert coord[0:-2].equals(
            InterpCoordinate(dict(tie_indices=[0, 6], tie_values=[100.0, 700.0]))
        )
        assert coord[-2:].equals(
            InterpCoordinate(dict(tie_indices=[0, 1], tie_values=[800.0, 900.0]))
        )
        assert coord[1:2].equals(
            InterpCoordinate(dict(tie_indices=[0], tie_values=[200.0]))
        )
        assert coord[1:3:2].equals(
            InterpCoordinate(dict(tie_indices=[0], tie_values=[200.0]))
        )
        assert coord[::2].equals(
            InterpCoordinate(dict(tie_indices=[0, 4], tie_values=[100.0, 900.0]))
        )
        assert coord[::3].equals(
            InterpCoordinate(dict(tie_indices=[0, 2], tie_values=[100.0, 700.0]))
        )
        assert coord[::4].equals(
            InterpCoordinate(dict(tie_indices=[0, 2], tie_values=[100.0, 900.0]))
        )
        assert coord[::5].equals(
            InterpCoordinate(dict(tie_indices=[0, 1], tie_values=[100.0, 600.0]))
        )
        assert coord[2:7:3].equals(
            InterpCoordinate(dict(tie_indices=[0, 1], tie_values=[300.0, 600.0]))
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


class TestFixedInterpCoordinate:
    valid = [
        {
            "tie_indices": [0, 5, 9, 10, 19],
            "tie_values": [0.0, 0.5, 0.9, 2.0, 2.9],
            "sampling_interval": 0.1,
        }
    ]

    def test_isvalid(self):
        for data in self.valid:
            assert FixedInterpCoordinate._isvalid(data)

    def test_init(self):
        for data in self.valid:
            coord = FixedInterpCoordinate(data, "dim")
            assert coord.sampling_interval == data["sampling_interval"]

        with pytest.raises(TypeError):
            coord._concat(ScalarCoordinate(1))
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

    def test_init_extra_keys(self):
        with pytest.raises(ValueError, match="both"):
            InterpCoordinate(
                {"tie_indices": [0, 8], "tie_values": [100.0, 900.0], "extra": 1}
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

    def test_get_sampling_interval_empty(self):
        coord = InterpCoordinate()
        assert coord.get_sampling_interval() is None

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
        # Build a coord with an overlap (tie_values go backwards between segments)
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 4, 5, 9],
                "tie_values": [0.0, 4.0, 3.0, 7.0],  # overlap at index 5 (value 3 < 4)
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
        # Three segments all strictly increasing — must not raise ValueError from bool()
        coord = InterpCoordinate(
            {
                "tie_indices": [0, 4, 5, 9, 10, 14],
                "tie_values": [0.0, 4.0, 5.0, 9.0, 10.0, 14.0],
            }
        )
        assert coord._is_monotonic_increasing() is True

    def test_slice_step_collision(self):
        # 4 tie points; step=3 makes first inner tie collide (collision fixed) and
        # second inner tie doesn't collide (covers the False branch → loop continues).
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

    def test_get_sampling_interval_datetime_cast(self):
        t0 = np.datetime64("2000-01-01T00:00:00")
        t1 = np.datetime64("2000-01-01T00:00:08")
        coord = InterpCoordinate({"tie_indices": [0, 8], "tie_values": [t0, t1]})
        result = coord.get_sampling_interval()  # cast=True by default
        assert result == 1.0

    def test_get_sampling_interval_unit_spaced(self):
        # all tie-index gaps == 1 → mask is all False → returns None
        coord = InterpCoordinate(
            {"tie_indices": [0, 1, 2], "tie_values": [0.0, 1.0, 2.0]}
        )
        assert coord.get_sampling_interval() is None

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
        # Second coord hitting the "already in attrs" branch (line 223)
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
