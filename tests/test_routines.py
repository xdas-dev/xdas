import numpy as np
import numpy.testing as npt
import obspy
import pytest

import xdas as xd
from xdas.core.routines import Bag, CompatibilityError


class TestBag:
    def test_bag_initialization(self):
        bag = Bag(dim="time")
        assert bag.dim == "time"
        assert bag.objs == []

    def test_bag_append_initializes(self):
        da = xd.DataArray(
            np.random.rand(10, 5),
            {
                "time": {
                    "tie_indices": [0, 9],
                    "tie_values": [0.0, 9.0],
                    "sampling_interval": 1.0,
                },
                "space": np.arange(5),
            },
        )
        bag = Bag(dim="time")
        bag.append(da)
        assert len(bag.objs) == 1
        assert bag.objs[0] is da
        assert bag.subcoords.equals(xd.Coordinates({"space": np.arange(5)}))
        assert bag.subshape == (5,)
        assert bag.dims == ("time", "space")
        assert bag.delta

    def test_bag_append_compatible(self):
        da1 = xd.DataArray(np.random.rand(10, 5), dims=("time", "space"))
        da2 = xd.DataArray(np.random.rand(10, 5), dims=("time", "space"))
        bag = Bag(dim="time")
        bag.append(da1)
        bag.append(da2)
        assert len(bag.objs) == 2
        assert bag.objs[1] is da2
        da1 = xd.DataArray(
            np.random.rand(10, 5), {"time": np.arange(10), "space": np.arange(5)}
        )
        da2 = xd.DataArray(
            np.random.rand(10, 5), {"time": np.arange(10, 20), "space": np.arange(5)}
        )
        bag = Bag(dim="time")
        bag.append(da1)
        bag.append(da2)
        assert len(bag.objs) == 2
        assert bag.objs[1] is da2

    def test_bag_append_incompatible_dims(self):
        da1 = xd.DataArray(np.random.rand(10, 5), dims=("time", "space"))
        da2 = xd.DataArray(np.random.rand(10, 5), dims=("space", "time"))
        bag = Bag(dim="time")
        bag.append(da1)
        with pytest.raises(CompatibilityError):
            bag.append(da2)

    def test_bag_append_incompatible_shape(self):
        da1 = xd.DataArray(np.random.rand(10, 5), dims=("time", "space"))
        da2 = xd.DataArray(np.random.rand(10, 6), dims=("time", "space"))
        bag = Bag(dim="time")
        bag.append(da1)
        with pytest.raises(CompatibilityError):
            bag.append(da2)

    def test_bag_append_incompatible_dtype(self):
        da1 = xd.DataArray(np.random.rand(10, 5), dims=("time", "space"))
        da2 = xd.DataArray(
            np.random.randint(0, 10, size=(10, 5)), dims=("time", "space")
        )
        bag = Bag(dim="time")
        bag.append(da1)
        with pytest.raises(CompatibilityError):
            bag.append(da2)

    def test_bag_append_incompatible_coords(self):
        da1 = xd.DataArray(
            np.random.rand(10, 5),
            dims=("time", "space"),
            coords={"space": np.arange(5)},
        )
        da2 = xd.DataArray(
            np.random.rand(10, 5),
            dims=("time", "space"),
            coords={"space": np.arange(5) + 1},
        )
        bag = Bag(dim="time")
        bag.append(da1)
        with pytest.raises(CompatibilityError):
            bag.append(da2)

    def test_bag_append_incompatible_sampling_interval(self):
        da1 = xd.DataArray(
            np.random.rand(10, 5),
            dims=("time", "space"),
            coords={
                "time": {
                    "tie_indices": [0, 9],
                    "tie_values": [0.0, 9.0],
                    "sampling_interval": 1.0,
                }
            },
        )
        da2 = xd.DataArray(
            np.random.rand(10, 5),
            dims=("time", "space"),
            coords={
                "time": {
                    "tie_indices": [0, 9],
                    "tie_values": [0.0, 18.0],
                    "sampling_interval": 2.0,
                }
            },
        )
        bag = Bag(dim="time")
        bag.append(da1)
        with pytest.raises(CompatibilityError):
            bag.append(da2)


class TestCombineByCoords:
    def test_basic(self):
        # without coords
        da1 = xd.DataArray(np.random.rand(10, 5), dims=("time", "space"))
        da2 = xd.DataArray(np.random.rand(10, 5), dims=("time", "space"))
        combined = xd.combine_by_coords([da1, da2], dim="time", squeeze=True)
        assert combined.shape == (20, 5)

        # with coords
        da1, da2 = xd.split(xd.testing.dummy(dims=("time", "space"), shape=(20, 5)), 2)
        combined = xd.combine_by_coords([da1, da2], dim="time", squeeze=True)
        assert combined.shape == (20, 5)

    def test_incompatible_shape(self):
        da1 = xd.DataArray(np.random.rand(10, 5), dims=("time", "space"))
        da2 = xd.DataArray(np.random.rand(10, 6), dims=("time", "space"))
        dc = xd.combine_by_coords([da1, da2], dim="time")
        assert len(dc) == 2
        assert dc[0].equals(da1)
        assert dc[1].equals(da2)

    def test_incompatible_dims(self):
        da1 = xd.DataArray(np.random.rand(10, 5), dims=("time", "space"))
        da2 = xd.DataArray(np.random.rand(10, 5), dims=("space", "time"))
        dc = xd.combine_by_coords([da1, da2], dim="time")
        assert len(dc) == 2
        assert dc[0].equals(da1)
        assert dc[1].equals(da2)

    def test_incompatible_dtype(self):
        da1 = xd.DataArray(np.random.rand(10, 5), dims=("time", "space"))
        da2 = xd.DataArray(
            np.random.randint(0, 10, size=(10, 5)), dims=("time", "space")
        )
        dc = xd.combine_by_coords([da1, da2], dim="time")
        assert len(dc) == 2
        assert dc[0].equals(da1)
        assert dc[1].equals(da2)

    def test_incompatible_coords(self):
        da1 = xd.DataArray(
            np.random.rand(10, 5),
            dims=("time", "space"),
            coords={"space": np.arange(5)},
        )
        da2 = xd.DataArray(
            np.random.rand(10, 5),
            dims=("time", "space"),
            coords={"space": np.arange(5) + 1},
        )
        dc = xd.combine_by_coords([da1, da2], dim="time")
        assert len(dc) == 2
        assert dc[0].equals(da1)
        assert dc[1].equals(da2)

    def test_incompatible_sampling_interval(self):
        da1 = xd.DataArray(
            np.random.rand(10, 5),
            dims=("time", "space"),
            coords={
                "time": {
                    "tie_indices": [0, 9],
                    "tie_values": [0.0, 9.0],
                    "sampling_interval": 1.0,
                }
            },
        )
        da2 = xd.DataArray(
            np.random.rand(10, 5),
            dims=("time", "space"),
            coords={
                "time": {
                    "tie_indices": [0, 9],
                    "tie_values": [0.0, 18.0],
                    "sampling_interval": 2.0,
                }
            },
        )
        dc = xd.combine_by_coords([da1, da2], dim="time")
        assert len(dc) == 2
        assert dc[0].equals(da1)
        assert dc[1].equals(da2)

    def test_expand_scalar_coordinate(self):
        da1 = xd.DataArray(
            np.random.rand(10),
            dims=("time",),
            coords={"time": np.arange(10), "space": 0},
        )
        da2 = xd.DataArray(
            np.random.rand(10),
            dims=("time",),
            coords={"time": np.arange(10), "space": 1},
        )
        dc = xd.combine_by_coords([da1, da2], dim="space", squeeze=True)
        assert dc.shape == (2, 10)
        assert dc.dims == ("space", "time")
        assert dc.coords["space"].values.tolist() == [0, 1]


class TestOpenMFDataArray:
    def test_warn_on_corrupted_files(self, tmp_path):
        expected = xd.testing.dummy(dims=("time", "space"), shape=(10, 5))
        for index, chunk in enumerate(xd.split(expected, 3, "time"), start=1):
            chunk.to_netcdf(tmp_path / f"chunk_{index}.nc")
        result = xd.open_mfdataarray(tmp_path / "*.nc")
        assert result.equals(expected)
        with (tmp_path / "corrupted.nc").open("wb") as f:
            f.write(b"corrupted")

        # single worker
        with pytest.warns(RuntimeWarning):
            result = xd.open_mfdataarray(tmp_path / "*.nc", parallel=False)
        assert result.equals(expected)

        # multiple workers
        with pytest.warns(RuntimeWarning):
            result = xd.open_mfdataarray(tmp_path / "*.nc", parallel=2)
        assert result.equals(expected)

    def test_verbose_single_worker(self, tmp_path):
        expected = xd.testing.dummy(dims=("time", "space"), shape=(10, 5))
        for index, chunk in enumerate(xd.split(expected, 3, "time"), start=1):
            chunk.to_netcdf(tmp_path / f"chunk_{index}.nc")
        result = xd.open_mfdataarray(tmp_path / "*.nc", verbose=True, parallel=1)
        assert result.equals(expected)

    def test_verbose_multiple_workers(self, tmp_path):
        expected = xd.testing.dummy(dims=("time", "space"), shape=(10, 5))
        for index, chunk in enumerate(xd.split(expected, 3, "time"), start=1):
            chunk.to_netcdf(tmp_path / f"chunk_{index}.nc")
        result = xd.open_mfdataarray(tmp_path / "*.nc", verbose=True, parallel=2)
        assert result.equals(expected)


class TestOpen:  # TODO: those tests are weirdly slow...
    def test_open_single_dataarray(self, tmp_path):
        expected = xd.testing.dummy(dims=("time", "space"), shape=(10, 5))

        path = tmp_path / "dataarray.nc"
        expected.to_netcdf(path)

        result = xd.open(path)
        assert result.equals(expected)

    def test_open_multiple_file_dataarray(self, tmp_path):
        expected = xd.testing.dummy(dims=("time", "space"), shape=(10, 5))

        file_paths = []
        for index, chunk in enumerate(xd.split(expected, 3, "time"), start=1):
            file_path = tmp_path / f"chunk_{index}.nc"
            chunk.to_netcdf(file_path)
            file_paths.append(file_path)

        # glob patterns
        result = xd.open(tmp_path / "*.nc")
        assert result.equals(expected)
        result = xd.open(tmp_path / "chunk_[1-3].nc")
        assert result.equals(expected)
        result = xd.open(tmp_path / "chunk_?.nc")
        assert result.equals(expected)

        # list of paths
        result = xd.open(file_paths)
        assert result.equals(expected)

    def test_open_multiple_file_tree(self, tmp_path):
        expected = xd.DataCollection(
            {
                "DAS01": xd.DataCollection(
                    [xd.testing.dummy(dims=("time", "space"), shape=(10, 5))],
                    name="acquisition",
                ),
                "DAS02": xd.DataCollection(
                    [xd.testing.dummy(dims=("time", "space"), shape=(7, 3))],
                    name="acquisition",
                ),
            },
            name="station",
        )

        for station in expected:
            dirpath = tmp_path / station
            dirpath.mkdir()
            for index, chunk in enumerate(
                xd.split(expected[station][0], 3, "time"), start=1
            ):
                chunk.to_netcdf(dirpath / f"chunk_{index}.nc")

        result = xd.open(tmp_path / "{station}" / "[acquisition].nc")
        assert result.equals(expected)

    def test_open_single_datacollection(self, tmp_path):
        expected = xd.DataCollection(
            [xd.testing.dummy(dims=("time", "space"), shape=(10, 5))]
        )

        expected.to_netcdf(tmp_path / "collection.nc")

        result = xd.open(tmp_path / "collection.nc")
        assert result.equals(expected)

    def test_open_multiple_datacollection_with_glob(self, tmp_path):
        expected = xd.DataCollection(
            {
                "DAS01": xd.DataCollection(
                    [xd.testing.dummy(dims=("time", "space"), shape=(10, 5))],
                    name="acquisition",
                ),
                "DAS02": xd.DataCollection(
                    [xd.testing.dummy(dims=("time", "space"), shape=(7, 3))],
                    name="acquisition",
                ),
            },
            name="station",
        )

        expected.isel(time=slice(None, 3)).to_netcdf(tmp_path / "datacollection_1.nc")
        expected.isel(time=slice(3, None)).to_netcdf(tmp_path / "datacollection_2.nc")

        # glob patterns
        result = xd.open(tmp_path / "datacollection_*.nc")
        assert result.equals(expected)
        result = xd.open(tmp_path / "datacollection_[1-2].nc")
        assert result.equals(expected)
        result = xd.open(tmp_path / "datacollection_?.nc")
        assert result.equals(expected)

        # list of paths
        file_paths = [
            tmp_path / "datacollection_1.nc",
            tmp_path / "datacollection_2.nc",
        ]
        result = xd.open(file_paths)
        assert result.equals(expected)

    def test_raise_if_all_files_corrupted(self, tmp_path):
        with (tmp_path / "corrupted1.nc").open("wb") as f:
            f.write(b"corrupted")
        with (tmp_path / "corrupted2.nc").open("wb") as f:
            f.write(b"corrupted")
        with pytest.warns(RuntimeWarning), pytest.raises(RuntimeError):
            xd.open_mfdataarray(str(tmp_path / "*.nc"))


class TestSplit:
    @pytest.fixture
    def dataarray(self, dtype, ctype):
        starts = np.array(
            [
                0,  # 0 - initial block
                10,  # 10 - continuous
                18,  # 20 - 2 overlap
                30,  # 30 - 2 gap
                48,  # 40 - 8 gap
                50,  # 50 - 8 overlap
            ],
            dtype,
        )
        size = 10
        step = np.array(
            1, "timedelta64" if np.issubdtype(dtype, np.datetime64) else dtype
        )
        coord = xd.concat_coords(
            [
                xd.Coordinate[ctype].from_block(start, size, step, "dim")
                for start in starts
            ],
            tolerance=False,
        )
        return xd.DataArray(np.random.randn(len(coord)), {"dim": coord})

    # kind, tolerance, split_indices
    CASES = [
        ("discontinuities", False, [10, 20, 30, 40, 50]),
        ("discontinuities", None, [20, 30, 40, 50]),
        ("discontinuities", 1, [20, 30, 40, 50]),
        ("discontinuities", 2, [40, 50]),
        ("discontinuities", 4, [40, 50]),
        ("discontinuities", 8, []),
        ("discontinuities", 20, []),
        ("gaps", False, [10, 30, 40]),
        ("gaps", None, [30, 40]),  # continuity is a gaps
        ("gaps", 1, [30, 40]),
        ("gaps", 2, [40]),
        ("gaps", 4, [40]),
        ("gaps", 8, []),
        ("gaps", 20, []),
        ("overlaps", False, [20, 50]),
        ("overlaps", None, [20, 50]),  # continuity is not an overlaps
        ("overlaps", 1, [20, 50]),
        ("overlaps", 2, [50]),
        ("overlaps", 4, [50]),
        ("overlaps", 8, []),
        ("overlaps", 20, []),
    ]

    @pytest.mark.parametrize("ctype", ["interpolated", "sampled"])
    @pytest.mark.parametrize("dtype", [int, float, "datetime64[s]"])
    def test_from_integer(self, dataarray):
        chunks = xd.split(dataarray, 4)
        assert len(chunks) == 4
        result = xd.concat(chunks, tolerance=None)
        np.testing.assert_array_equal(
            result["dim"].values, dataarray["dim"].values, strict=True
        )
        np.testing.assert_array_equal(result.values, dataarray.values, strict=True)

    @pytest.mark.parametrize("ctype", ["interpolated", "sampled"])
    @pytest.mark.parametrize("dtype", [int, float, "datetime64[s]"])
    def test_from_coord(self, dataarray):
        for kind, tolerance, expected_split_indices in self.CASES:
            chunks = xd.split(dataarray, kind, "dim", tolerance)
            assert len(chunks) == len(expected_split_indices) + 1
            result = xd.concat(chunks, "dim", tolerance=False)
            np.testing.assert_array_equal(
                result["dim"].values, dataarray["dim"].values, strict=True
            )
            np.testing.assert_array_equal(result.values, dataarray.values, strict=True)

    @pytest.mark.parametrize("ctype", ["interpolated", "sampled"])
    @pytest.mark.parametrize("dtype", [int, float, "datetime64[s]"])
    def test_from_indices(self, dataarray):
        split_indices = [11, 22, 33, 44, 55]
        chunks = xd.split(dataarray, split_indices)
        assert len(chunks) == len(split_indices) + 1
        result = xd.concat(chunks, "dim", tolerance=False)
        np.testing.assert_array_equal(
            result["dim"].values, dataarray["dim"].values, strict=True
        )
        np.testing.assert_array_equal(result.values, dataarray.values, strict=True)

    def test_raise_tolerance_not_used(self):
        da = xd.DataArray()
        with pytest.raises(ValueError):
            xd.split(da, 3, tolerance=1)
        with pytest.raises(ValueError):
            xd.split(da, [10], tolerance=1)


class TestOpenEdgeCases:
    def test_invalid_paths_type_raises(self):
        with pytest.raises(Exception, match="paths"):
            xd.open(123)

    def test_engine_instance(self, tmp_path):
        from xdas.io import Engine

        da = xd.testing.dummy(shape=(10, 5))
        path = str(tmp_path / "test.nc")
        da.to_netcdf(path)
        result = xd.open_dataarray(path, engine=Engine["xdas"]())
        assert result.equals(da)

    def test_engine_instance_rejects_extra_config(self, tmp_path):
        from xdas.io import Engine

        da = xd.testing.dummy(shape=(10, 5))
        path = str(tmp_path / "test.nc")
        da.to_netcdf(path)
        engine = Engine["xdas"]()
        with pytest.raises(ValueError, match="configured engine instance"):
            xd.open_dataarray(path, engine=engine, vtype="tiles")
        with pytest.raises(ValueError, match="configured engine instance"):
            xd.open_dataarray(path, engine=engine, group="somegroup")

    def test_unknown_engine_kwarg_raises(self, tmp_path):
        da = xd.testing.dummy(shape=(10, 5))
        path = str(tmp_path / "test.nc")
        da.to_netcdf(path)
        with pytest.raises(TypeError, match="overlpas"):
            xd.open_dataarray(path, engine="febus", overlpas=(1, 1))
        # auto-detection accepts no format-specific parameters
        with pytest.raises(TypeError, match="overlaps"):
            xd.open_dataarray(path, overlaps=(1, 1))

    def test_invalid_engine_type_raises(self, tmp_path):
        da = xd.testing.dummy(shape=(10, 5))
        path = str(tmp_path / "test.nc")
        da.to_netcdf(path)
        with pytest.raises(TypeError, match="engine must be"):
            xd.open_dataarray(path, engine=42)


class TestOpenMFDatacollectionEdgeCases:
    def test_nonexistent_path_in_list_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            xd.open_mfdatacollection([str(tmp_path / "nonexistent.nc")])

    def test_empty_glob_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            xd.open_mfdatacollection(str(tmp_path / "*.nc"))

    def test_verbose_single_worker(self, tmp_path):
        da = xd.testing.dummy(shape=(10, 5))
        dc = xd.DataCollection([da, da])
        path1 = str(tmp_path / "dc1.nc")
        path2 = str(tmp_path / "dc2.nc")
        dc.to_netcdf(path1)
        dc.to_netcdf(path2)
        result = xd.open_mfdatacollection(
            str(tmp_path / "dc*.nc"), verbose=True, parallel=1
        )
        assert isinstance(result, xd.DataCollection)

    def test_verbose_multiple_worker(self, tmp_path):
        da = xd.testing.dummy(shape=(10, 5))
        dc = xd.DataCollection([da, da])
        path1 = str(tmp_path / "dc1.nc")
        path2 = str(tmp_path / "dc2.nc")
        dc.to_netcdf(path1)
        dc.to_netcdf(path2)
        result = xd.open_mfdatacollection(
            str(tmp_path / "dc*.nc"), verbose=True, parallel=2
        )
        assert isinstance(result, xd.DataCollection)

    def test_invalid_path(self):
        with pytest.raises(ValueError, match="`paths` must be"):
            xd.open_mfdatacollection(42)


class TestOpenMFDataArrayEdgeCases:
    def test_invalid_paths_type_raises(self):
        with pytest.raises(ValueError, match="paths"):
            xd.open_mfdataarray(123)

    def test_parallel_path(self, tmp_path):
        expected = xd.testing.dummy(dims=("time", "space"), shape=(10, 5))
        for i, chunk in enumerate(xd.split(expected, 3, "time"), 1):
            chunk.to_netcdf(tmp_path / f"chunk_{i}.nc")
        result = xd.open_mfdataarray(tmp_path / "*.nc", parallel=2)
        assert result.equals(expected)

    def test_no_files_no_failures_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            xd.open_mfdataarray(str(tmp_path / "*.nc"))


class TestOpenMFDatacollectionParallel:
    def test_parallel_path(self, tmp_path):
        da = xd.testing.dummy(shape=(10, 5))
        dc = xd.DataCollection([da, da])
        path1 = str(tmp_path / "dc1.nc")
        path2 = str(tmp_path / "dc2.nc")
        dc.to_netcdf(path1)
        dc.to_netcdf(path2)
        result = xd.open_mfdatacollection(str(tmp_path / "dc*.nc"), parallel=2)
        assert isinstance(result, xd.DataCollection)


class TestOpenMFDataTree:
    def test_one_level_depth(self, tmp_path):
        keys = ["LOC01", "LOC02"]
        dirnames = [tmp_path / key for key in keys]
        for dirname in dirnames:
            dirname.mkdir()
            for idx, da in enumerate(xd.split(xd.testing.dummy(), 3), start=1):
                da.to_netcdf(dirname / f"{idx:03d}.nc")
        da = xd.testing.dummy()
        dc = xd.open_mfdatatree(tmp_path / "{node}" / "00[acquisition].nc")
        assert list(dc.keys()) == keys
        for key in keys:
            assert dc[key][0].load().equals(da)

    def test_two_level_depth(self, tmp_path):
        dc = xd.DataCollection(
            {
                "NET01": {
                    "STA01": xd.split(xd.testing.dummy(), 1),
                },
                "NET02": {
                    "STA02": xd.split(xd.testing.dummy(), 2),
                    "STA03": xd.split(xd.testing.dummy(), 3),
                },
            }
        )
        dc.name = "network"
        for network in dc:
            dc[network].name = "station"
            for station in dc[network]:
                for idx, da in enumerate(dc[network][station]):
                    path = tmp_path / network / station / f"{idx:03d}.nc"
                    da.to_netcdf(path, create_dirs=True)
                    dc[network][station] = xd.combine_by_coords(dc[network][station])
                    dc[network][station].name = "acquisition"
        result = xd.open_mfdatatree(
            tmp_path / "{network}" / "{station}" / "00[acquisition].nc"
        )
        assert result.equals(dc)


class TestAsdataarray:
    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Cannot convert"):
            xd.asdataarray("not_an_array")

    def test_already_dataarray(self):
        da = xd.DataArray([1, 2, 3], dims="x")
        result = xd.asdataarray(da)
        assert result.equals(da)


class TestCombineByCoordsDimLast:
    def test_dim_last(self):
        da1 = xd.DataArray(
            np.random.rand(5, 3),
            coords={"time": np.arange(5), "space": np.arange(3)},
        )
        da2 = xd.DataArray(
            np.random.rand(5, 3),
            coords={"time": np.arange(5), "space": np.arange(3, 6)},
        )
        result = xd.combine_by_coords([da1, da2], dim="last", squeeze=True)
        assert isinstance(result, xd.DataArray)


class TestConcatEdgeCases:
    def test_empty_list_returns_dataarray(self):
        result = xd.concat([])
        assert isinstance(result, xd.DataArray)
        assert result.empty

    def test_all_empty_elements_returns_empty_dataarray(self):
        da = xd.DataArray(np.zeros((0, 10)), dims=("time", "distance"))
        result = xd.concat([da, da])
        assert isinstance(result, xd.DataArray)
        assert result.empty
        assert result.dims == ("time", "distance")

    def test_mixed_empty_and_nonempty_uses_nonempty(self):
        t_empty = np.array([], dtype="datetime64[ns]")
        da_empty = xd.DataArray(np.zeros((0,)), {"time": t_empty})
        t = np.array(
            ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04", "2000-01-05"],
            dtype="datetime64[ns]",
        )
        da = xd.DataArray(np.ones((5,)), {"time": t})
        result = xd.concat([da_empty, da])
        assert result.equals(da)


class TestConcatNewDim:
    def trace(self, channel, station="CH001", values=None):
        time = {"tie_indices": [0, 4], "tie_values": [0.0, 4.0]}
        if values is None:
            values = np.arange(5.0)
        return xd.DataArray(
            values,
            {
                "network": (None, "DX"),
                "station": (None, station),
                "channel": (None, channel),
                "time": time,
            },
        )

    def test_varying_scalar_is_promoted(self):
        objs = [self.trace(channel) for channel in ("HHZ", "HHN", "HHE")]
        da = xd.concat(objs, "channel")
        assert da.dims == ("channel", "time")
        assert da.shape == (3, 5)
        # `channel` is the concat coordinate: `expand_dims` promotes it and
        # `concat_coords` sorts it
        assert sorted(da["channel"].values.tolist()) == ["HHE", "HHN", "HHZ"]
        # the constant scalars stay scalar
        assert da["network"].dim is None
        assert da["station"].dim is None

    def test_other_varying_scalar_is_promoted_along_the_new_dim(self):
        objs = [self.trace("HHZ", station=f"CH{idx:03d}") for idx in (1, 2, 3)]
        da = xd.concat(objs, "component")
        assert da.dims == ("component", "time")
        assert da["station"].dim == "component"
        assert da["station"].values.tolist() == ["CH001", "CH002", "CH003"]
        assert da["network"].dim is None

    def test_promotion_follows_the_concat_order(self):
        # `concat` sorts by the concat coordinate; a promoted scalar must be
        # gathered in that same order, not in input order
        objs = [
            self.trace(channel, station=station)
            for channel, station in [("HHZ", "C"), ("HHE", "A"), ("HHN", "B")]
        ]
        da = xd.concat(objs, "channel")
        assert da["channel"].values.tolist() == ["HHE", "HHN", "HHZ"]
        assert da["station"].values.tolist() == ["A", "B", "C"]

    def test_unequal_non_scalar_coord_raises(self):
        da1 = self.trace("HHZ")
        da2 = self.trace("HHN")
        da2["time"] = {"tie_indices": [0, 4], "tie_values": [10.0, 14.0]}
        with pytest.raises(ValueError, match="'time' differs"):
            xd.concat([da1, da2], "channel")

    def test_missing_coord_raises(self):
        da1 = self.trace("HHZ")
        da2 = self.trace("HHN").drop_coords("network")
        with pytest.raises(ValueError, match="must share their coordinates"):
            xd.concat([da1, da2], "channel")

    def test_concat_along_existing_dim_is_unchanged(self):
        # the promotion machinery only runs when a new dimension is opened
        da1 = self.trace("HHZ")
        da2 = self.trace("HHN")
        da2["time"] = {"tie_indices": [0, 4], "tie_values": [5.0, 9.0]}
        da = xd.concat([da1, da2], "time")
        assert da.dims == ("time",)
        assert da.shape == (10,)
        assert da["channel"].values == "HHZ"


class TestConcatCoordsEdgeCases:
    def test_tolerance_with_dense_coord_is_noop(self):
        # Dense coordinates now implement a (degenerate) `simplify`, so passing a
        # tolerance no longer raises; it simply has no effect.
        da1 = xd.DataArray(
            np.random.rand(5), {"x": np.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        )
        da2 = xd.DataArray(
            np.random.rand(5), {"x": np.array([5.0, 6.0, 7.0, 8.0, 9.0])}
        )
        from xdas.core.routines import concat_coords

        result = concat_coords([da1["x"], da2["x"]], tolerance=1.0)
        expected = concat_coords([da1["x"], da2["x"]])
        assert result.equals(expected)

    def test_tolerance_with_scalar_coord_raises(self):
        from xdas.core.routines import concat_coords

        scalar = xd.Coordinate("SRN")
        with pytest.raises(TypeError, match="tolerance"):
            concat_coords([scalar], tolerance=1.0)

    def test_default_tolerance_with_scalar_coord_passes(self):
        from xdas.core.routines import concat_coords

        scalar = xd.Coordinate("SRN")
        assert concat_coords([scalar]).equals(scalar)


class TestSplitEdgeCases:
    def test_n_zero_raises(self):
        da = xd.testing.dummy(dims=("time",), shape=(10,), step=0.01)
        with pytest.raises(ValueError, match="`n` must be larger than 0"):
            xd.split(da, 0)

    def test_n_too_large_raises(self):
        da = xd.testing.dummy(dims=("time",), shape=(10,), step=0.01)
        with pytest.raises(ValueError, match="`n` must be smaller"):
            xd.split(da, 10)


class TestBroadcastCoordsScalar:
    def test_scalar_coord_skipped(self):
        da1 = xd.DataArray(
            np.random.rand(5, 3),
            {"time": np.arange(5), "space": np.arange(3), "network": "NET"},
        )
        da2 = xd.DataArray(
            np.random.rand(5, 3),
            {"time": np.arange(5), "space": np.arange(3)},
        )
        result = xd.broadcast_coords(da1, da2)
        assert "network" not in result


class TestPlotAvailability:
    def test_dataarray_plot(self):
        da = xd.testing.dummy(dims=("time",), shape=(100,), step=0.01)
        fig = xd.plot_availability(da)
        assert fig is not None

    def test_datassequence_plot(self):
        da = xd.testing.dummy(dims=("time",), shape=(100,), step=0.01)
        dc = xd.DataCollection([da, da])
        fig = xd.plot_availability(dc)
        assert fig is not None

    def test_datamapping_plot(self):
        da = xd.testing.dummy(dims=("time",), shape=(100,), step=0.01)
        dm = xd.DataCollection({"a": da, "b": da})
        fig = xd.plot_availability(dm)
        assert fig is not None

    def test_invalid_type_raises(self):
        from xdas.core.routines import _get_timeline_dataframe

        with pytest.raises(TypeError, match="DataCollection"):
            _get_timeline_dataframe("not_valid")


class TestTrimOverlaps:
    delta = 0.01

    def segments(self, spans, dtype=float):
        """Build ``(obspy.Stream, DataArray)`` from ``(start_second, values)`` pairs."""
        st = obspy.Stream()
        objs = []
        for start, values in spans:
            values = np.asarray(values, dtype=dtype)
            st.append(
                obspy.Trace(
                    values.copy(),
                    {
                        "delta": self.delta,
                        "starttime": obspy.UTCDateTime(start),
                        "network": "DX",
                        "station": "CH001",
                        "location": "00",
                        "channel": "HHZ",
                    },
                )
            )
            t0 = np.datetime64(round(start * 1e9), "ns")
            dt = np.timedelta64(round(self.delta * 1e9), "ns")
            objs.append(
                xd.DataArray(
                    values,
                    {
                        "time": {
                            "tie_indices": [0, len(values) - 1],
                            "tie_values": [t0, t0 + (len(values) - 1) * dt],
                        }
                    },
                )
            )
        return st, xd.concat(objs, "time", tolerance=False)

    def test_keep_last_matches_obspy_merge(self):
        for spans in [
            [(0.0, np.arange(5.0)), (0.03, np.arange(100.0, 105.0))],
            [(0.0, np.arange(5.0)), (0.04, np.arange(100.0, 105.0))],
            [
                (0.0, np.arange(5.0)),
                (0.03, np.arange(100.0, 105.0)),
                (0.06, np.arange(200.0, 205.0)),
            ],
        ]:
            st, da = self.segments(spans)
            st.merge(method=1, interpolation_samples=0)
            result = xd.trim_overlaps(da)
            npt.assert_array_equal(result.values, np.asarray(st[0].data))
            assert result["time"][0].values == np.datetime64(
                str(st[0].stats.starttime.datetime), "ns"
            )

    def test_keep_first_is_the_mirror(self):
        # the later segment's head goes instead of the earlier one's tail
        _, da = self.segments(
            [(0.0, np.arange(5.0)), (0.03, np.arange(100.0, 105.0))]
        )
        npt.assert_array_equal(
            xd.trim_overlaps(da, keep="first").values,
            [0.0, 1.0, 2.0, 3.0, 4.0, 102.0, 103.0, 104.0],
        )
        npt.assert_array_equal(
            xd.trim_overlaps(da, keep="last").values,
            [0.0, 1.0, 2.0, 100.0, 101.0, 102.0, 103.0, 104.0],
        )

    def test_replaces_ignore_last_sample(self):
        # the old flag dropped the last sample of every segment; the shared
        # sample only, and only where it is genuinely shared, is enough
        _, da = self.segments(
            [(0.0, np.arange(5.0)), (0.04, np.arange(100.0, 105.0))]
        )
        result = xd.trim_overlaps(da)
        npt.assert_array_equal(
            result.values, [0.0, 1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0, 104.0]
        )
        # a clean seam is left untouched
        _, clean = self.segments(
            [(0.0, np.arange(5.0)), (0.05, np.arange(100.0, 105.0))]
        )
        assert xd.trim_overlaps(clean).equals(clean)

    def test_no_overlap_is_a_noop(self):
        da = xd.testing.dummy(dims=("time",), shape=(10,), step=0.01)
        assert xd.trim_overlaps(da).equals(da)

    def test_wholly_covered_part_is_dropped(self):
        # the middle segment is entirely covered by the last one; the first
        # must still be trimmed against that last one, not against the middle
        _, da = self.segments(
            [
                (0.0, np.arange(10.0)),
                (0.05, np.arange(100.0, 103.0)),
                (0.04, np.arange(200.0, 210.0)),
            ]
        )
        result = xd.trim_overlaps(da)
        # sorted by start: [0.00-0.09], [0.04-0.13], [0.05-0.07]; keeping the
        # last, the 0.04-0.13 segment survives only outside 0.05-0.07
        npt.assert_array_equal(
            result.values,
            # 0.00-0.03 from the first, 0.04 from the third, 0.05-0.07 from the
            # second, 0.08-0.13 from the third again
            [0.0, 1.0, 2.0, 3.0, 200.0, 100.0, 101.0, 102.0]
            + [204.0, 205.0, 206.0, 207.0, 208.0, 209.0],
        )
        assert result["time"].get_split_indices("overlaps").size == 0

    def test_enveloped_part_keeps_both_sides(self):
        # a short high-precedence segment inside a long one: the long one must
        # keep a run on each side of it, not lose everything past the overlap
        _, da = self.segments(
            [(0.0, np.arange(20.0)), (0.05, np.arange(100.0, 103.0))]
        )
        result = xd.trim_overlaps(da)
        expected = np.concatenate(
            [np.arange(5.0), np.arange(100.0, 103.0), np.arange(8.0, 20.0)]
        )
        npt.assert_array_equal(result.values, expected)
        assert result.sizes["time"] == 20

    def test_chain_of_three_mutual_overlaps(self):
        _, da = self.segments(
            [
                (0.0, np.arange(10.0)),
                (0.05, np.arange(100.0, 110.0)),
                (0.10, np.arange(200.0, 210.0)),
            ]
        )
        result = xd.trim_overlaps(da)
        npt.assert_array_equal(
            result.values,
            np.concatenate(
                [np.arange(5.0), np.arange(100.0, 105.0), np.arange(200.0, 210.0)]
            ),
        )
        assert result["time"].get_split_indices("overlaps").size == 0

    def test_sub_tolerance_jitter_is_not_trimmed(self):
        t0 = np.datetime64("2024-01-01T00:00:00.000000000")
        dt = np.timedelta64(10_000_000, "ns")
        # the second segment starts one microsecond early: jitter, not an overlap
        coord = {
            "tie_indices": [0, 4, 5, 9],
            "tie_values": [
                t0,
                t0 + 4 * dt,
                t0 + 5 * dt - np.timedelta64(1000, "ns"),
                t0 + 9 * dt - np.timedelta64(1000, "ns"),
            ],
        }
        da = xd.DataArray(np.arange(10.0), {"time": coord})
        result = xd.trim_overlaps(da, tolerance=0.001)
        npt.assert_array_equal(result.values, np.arange(10.0))

    def test_stays_lazy(self, tmp_path):
        from xdas.virtual import TileArray

        objs = []
        for index, start in enumerate([0, 8]):
            da = xd.testing.dummy(dims=("time",), shape=(10,), step=0.01)
            da["time"] = da["time"] + np.timedelta64(start * 10_000_000, "ns")
            path = tmp_path / f"chunk_{index}.nc"
            da.to_netcdf(path)
            objs.append(xd.open_dataarray(path, engine="xdas", vtype="tiles"))
        da = xd.concat(objs, "time", tolerance=False)
        result = xd.trim_overlaps(da)
        assert isinstance(result.data, TileArray)
        assert result.sizes["time"] == 18
        assert result["time"].get_split_indices("overlaps").size == 0

    def test_recurses_over_a_collection(self):
        _, da = self.segments(
            [(0.0, np.arange(5.0)), (0.03, np.arange(100.0, 105.0))]
        )
        dc = xd.DataCollection(
            {"CH001": xd.DataCollection([da, da], "acquisition")}, "station"
        )
        result = xd.trim_overlaps(dc)
        assert result.fields == ("station", "acquisition")
        assert list(result) == ["CH001"]
        assert len(result["CH001"]) == 2
        for element in result["CH001"]:
            npt.assert_array_equal(
                element.values, [0.0, 1.0, 2.0, 100.0, 101.0, 102.0, 103.0, 104.0]
            )

    def test_invalid_keep_raises(self):
        da = xd.testing.dummy(dims=("time",), shape=(10,), step=0.01)
        with pytest.raises(ValueError, match="`keep` must be"):
            xd.trim_overlaps(da, keep="both")


class TestSortby:
    def make_archive(self, tmp_path, vtype, pairs=((0, 2), (1, 3))):
        """Save 4 time chunks and fuse them losslessly as two runs.

        `concat` sorts whatever it is given, so tile-level disorder is
        built the way streamed scans produce it: runs that are internally
        ordered but interleave each other. The default pairing yields the
        tile order 0, 2, 1, 3.
        """
        expected = xd.testing.dummy(dims=("time", "space"), shape=(20, 5))
        chunks = xd.split(expected, 4, "time")
        parts = []
        for index, chunk in enumerate(chunks):
            path = tmp_path / f"chunk_{index}.nc"
            chunk.to_netcdf(path)
            parts.append(xd.open_dataarray(path, engine="xdas", vtype=vtype))
        runs = [
            xd.concat([parts[i] for i in pair], "time", tolerance=False)
            for pair in pairs
        ]
        return expected, xd.concat(runs, "time", tolerance=False)

    def test_sorts_tiles_lazily(self, tmp_path):
        from xdas.virtual import TileArray

        expected, shuffled = self.make_archive(tmp_path, "tiles")
        result = xd.sortby(shuffled, "time")
        assert isinstance(result.data, TileArray)
        assert result.equals(expected)
        assert result["time"].equals(expected["time"])

    def test_sorts_virtual_stack(self, tmp_path):
        from xdas.virtual import VirtualStack

        expected, shuffled = self.make_archive(tmp_path, "hdf5")
        result = xd.sortby(shuffled, "time")
        assert isinstance(result.data, VirtualStack)
        assert result.equals(expected)

    def test_already_sorted_fast_path(self, tmp_path):
        expected, arranged = self.make_archive(
            tmp_path, "tiles", pairs=((0, 1), (2, 3))
        )
        result = xd.sortby(arranged, "time")
        assert result.equals(expected)
        # a second sort is a no-op even though the coordinate is simplified
        assert xd.sortby(result, "time").equals(expected)

    def test_tolerance_false_skips_simplification(self, tmp_path):
        _, shuffled = self.make_archive(tmp_path, "tiles")
        result = xd.sortby(shuffled, "time", tolerance=False)
        # sorted but not simplified: one tie pair per chunk remains
        assert len(result["time"].tie_indices) == 8
        assert bool(np.all(np.diff(result["time"].tie_values.astype("i8")) > 0))

    def test_eager_data_raises(self):
        da = xd.testing.dummy(dims=("time", "space"), shape=(10, 5))
        with pytest.raises(NotImplementedError, match="TileArray or a VirtualStack"):
            xd.sortby(da, "time")

    def test_dense_coordinate_raises(self, tmp_path):
        _, shuffled = self.make_archive(tmp_path, "tiles")
        shuffled["time"] = shuffled["time"].values
        with pytest.raises(NotImplementedError, match="interpolated"):
            xd.sortby(shuffled, "time")

    def test_simplified_unsorted_raises(self, tmp_path):
        # a coordinate whose ties span the first two tiles as one segment
        # (the state a prior simplification leaves) while the tile order
        # still needs fixing: the exact blockwise gather is impossible
        from xdas.coordinates import InterpCoordinate

        _, misaligned = self.make_archive(tmp_path, "tiles")
        coord = misaligned["time"]
        misaligned["time"] = InterpCoordinate(
            {
                "tie_indices": np.array([0, 9, 10, 14, 15, 19]),
                "tie_values": coord.tie_values[[0, 3, 4, 5, 6, 7]],
            },
            "time",
        )
        with pytest.raises(NotImplementedError, match="align"):
            xd.sortby(misaligned, "time")


class TestStreamingCombine:
    def save_shuffled(self, tmp_path, nchunk=6):
        """Save chunks under names whose lexicographic order shuffles time."""
        expected = xd.testing.dummy(dims=("time", "space"), shape=(30, 5))
        names = ["e", "b", "f", "a", "d", "c"][:nchunk]
        for chunk, name in zip(xd.split(expected, nchunk, "time"), names):
            chunk.to_netcdf(tmp_path / f"{name}.nc")
        return expected

    def test_matches_monolithic(self, tmp_path, monkeypatch):
        from xdas.core import routines

        expected = self.save_shuffled(tmp_path)
        mono = xd.open_mfdataarray(
            tmp_path / "*.nc", engine="xdas", vtype="tiles", parallel=False
        )
        monkeypatch.setattr(routines, "MAX_OPEN_FILES", 2)
        streamed = xd.open_mfdataarray(
            tmp_path / "*.nc", engine="xdas", vtype="tiles", parallel=False
        )
        assert streamed.equals(expected)
        assert streamed["time"].equals(mono["time"])
        np.testing.assert_array_equal(np.asarray(streamed.data), np.asarray(mono.data))

    def test_non_consolidating_vtype_raises_instead_of_streaming(
        self, tmp_path, monkeypatch
    ):
        from xdas.core import routines

        # the batch size is the ceiling, so a vtype that cannot consolidate
        # never reaches the streaming path: it raises at the first batch
        self.save_shuffled(tmp_path)
        monkeypatch.setattr(routines, "MAX_OPEN_FILES", 2)
        with pytest.raises(NotImplementedError, match="cannot be consolidated"):
            xd.open_mfdataarray(
                tmp_path / "*.nc", engine="xdas", vtype="hdf5", parallel=False
            )

    def test_warns_and_recovers_on_corrupted_file(self, tmp_path, monkeypatch):
        from xdas.core import routines

        expected = self.save_shuffled(tmp_path)
        with (tmp_path / "ba.nc").open("wb") as file:
            file.write(b"corrupted")
        monkeypatch.setattr(routines, "MAX_OPEN_FILES", 2)
        with pytest.warns(RuntimeWarning):
            streamed = xd.open_mfdataarray(
                tmp_path / "*.nc", engine="xdas", vtype="tiles", parallel=False
            )
        assert streamed.equals(expected)

    def test_groups_interleaved_acquisitions_by_signature(self, tmp_path, monkeypatch):
        from xdas.core import routines

        # acquisition A (5 channels) at t0 and t2, B (3 channels) at t1:
        # signature grouping fuses A whole where the monolithic time-ordered
        # walk would split it around B
        wide = xd.testing.dummy(dims=("time", "space"), shape=(20, 5))
        chunks = xd.split(wide, 2, "time")
        narrow = xd.testing.dummy(dims=("time", "space"), shape=(10, 3))
        narrow["time"] = narrow["time"] + (
            chunks[1]["time"][0].values - narrow["time"][0].values
        )
        chunks[0].to_netcdf(tmp_path / "a.nc")
        narrow.to_netcdf(tmp_path / "b.nc")
        chunks[1].to_netcdf(tmp_path / "c.nc")
        monkeypatch.setattr(routines, "MAX_OPEN_FILES", 2)
        streamed = xd.open_mfdataarray(
            tmp_path / "*.nc", engine="xdas", vtype="tiles", parallel=False
        )
        assert isinstance(streamed, xd.DataCollection)
        assert len(streamed) == 2

    def test_single_run_squeezes(self, tmp_path, monkeypatch):
        from xdas.core import routines

        expected = self.save_shuffled(tmp_path)
        monkeypatch.setattr(routines, "MAX_OPEN_FILES", 2)
        collection = xd.open_mfdataarray(
            tmp_path / "*.nc",
            engine="xdas",
            vtype="tiles",
            parallel=False,
            squeeze=False,
        )
        assert isinstance(collection, xd.DataCollection)
        assert len(collection) == 1
        assert collection[0].equals(expected)


class TestStreamingCombineFallbacks:
    def test_dim_last_and_plain_name(self, tmp_path, monkeypatch):
        from xdas.core import routines

        expected = xd.testing.dummy(dims=("time",), shape=(30,), step=0.01)
        names = ["c", "a", "b"]
        for chunk, name in zip(xd.split(expected, 3, "time"), names):
            chunk.to_netcdf(tmp_path / f"{name}.nc")
        monkeypatch.setattr(routines, "MAX_OPEN_FILES", 2)
        for dim in ("last", "time"):
            result = xd.open_mfdataarray(
                tmp_path / "*.nc",
                dim=dim,
                engine="xdas",
                vtype="tiles",
                parallel=False,
            )
            assert result.equals(expected)

    def test_unsortable_group_falls_back_to_plain_concat(self, tmp_path, monkeypatch):
        from xdas.core import routines

        # dense time coordinates: sortby cannot permute them, the group is
        # concatenated with the tolerance directly (runs sorted by start)
        expected = xd.testing.dummy(
            dims=("time", "space"), shape=(30, 5), ctype="dense"
        )
        for index, chunk in enumerate(xd.split(expected, 3, "time")):
            chunk.to_netcdf(tmp_path / f"chunk_{index}.nc")
        monkeypatch.setattr(routines, "MAX_OPEN_FILES", 2)
        result = xd.open_mfdataarray(
            tmp_path / "*.nc", engine="xdas", vtype="tiles", parallel=False
        )
        assert result.equals(expected)

    def test_no_dim_coordinate(self, tmp_path, monkeypatch):
        from xdas.core import routines

        da = xd.DataArray(
            np.arange(30.0 * 5).reshape(30, 5),
            coords={"space": {"tie_indices": [0, 4], "tie_values": [0.0, 40.0]}},
            dims=("time", "space"),
        )
        for index in range(3):
            da[10 * index : 10 * (index + 1)].to_netcdf(tmp_path / f"chunk_{index}.nc")
        monkeypatch.setattr(routines, "MAX_OPEN_FILES", 2)
        result = xd.open_mfdataarray(
            tmp_path / "*.nc", engine="xdas", vtype="tiles", parallel=False
        )
        assert result.shape == (30, 5)


class TestSortbyMetadataFree:
    def test_permutes_without_declared_sampling_interval(self, tmp_path):
        from xdas.coordinates import InterpCoordinate

        helper = TestSortby()
        expected, shuffled = helper.make_archive(tmp_path, "tiles")
        coord = shuffled["time"]
        shuffled["time"] = InterpCoordinate(
            {"tie_indices": coord.tie_indices, "tie_values": coord.tie_values},
            "time",
        )
        result = xd.sortby(shuffled, "time")
        assert np.array_equal(result["time"].values, expected["time"].values)
        np.testing.assert_array_equal(np.asarray(result.data), expected.values)
