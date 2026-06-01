import numpy as np
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
            np.random.rand(10, 5), {"time": np.arange(10), "space": np.arange(5)}
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
            coords={"time": np.arange(10)},
        )
        da2 = xd.DataArray(
            np.random.rand(10, 5),
            dims=("time", "space"),
            coords={"time": np.arange(10) * 2},
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
        da1 = xd.DataArray(
            np.random.rand(10, 5),
            coords={"time": np.arange(10), "space": np.arange(5)},
        )
        da2 = xd.DataArray(
            np.random.rand(10, 5),
            coords={"time": np.arange(10, 20), "space": np.arange(5)},
        )
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
            coords={"time": np.arange(10)},
        )
        da2 = xd.DataArray(
            np.random.rand(10, 5),
            dims=("time", "space"),
            coords={"time": np.arange(10) * 2},
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
        expected = xd.DataArray(
            np.random.rand(10, 5),
            coords={
                "time": np.arange(10),
                "space": np.arange(5),
            },  # TODO: should work without coords
        )
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
        expected = xd.DataArray(
            np.random.rand(10, 5),
            coords={
                "time": np.arange(10),
                "space": np.arange(5),
            },  # TODO: should work without coords
        )
        for index, chunk in enumerate(xd.split(expected, 3, "time"), start=1):
            chunk.to_netcdf(tmp_path / f"chunk_{index}.nc")
        result = xd.open_mfdataarray(tmp_path / "*.nc", verbose=True, parallel=1)
        assert result.equals(expected)

    def test_verbose_multiple_workers(self, tmp_path):
        expected = xd.DataArray(
            np.random.rand(10, 5),
            coords={
                "time": np.arange(10),
                "space": np.arange(5),
            },  # TODO: should work without coords
        )
        for index, chunk in enumerate(xd.split(expected, 3, "time"), start=1):
            chunk.to_netcdf(tmp_path / f"chunk_{index}.nc")
        result = xd.open_mfdataarray(tmp_path / "*.nc", verbose=True, parallel=2)
        assert result.equals(expected)


class TestOpen:  # TODO: those tests are weirdly slow...
    def test_open_single_dataarray(self, tmp_path):
        expected = xd.DataArray(
            np.random.rand(10, 5),
            coords={
                "time": np.arange(10),
                "space": np.arange(5),
            },
        )

        path = tmp_path / "dataarray.nc"
        expected.to_netcdf(path)

        result = xd.open(path)
        assert result.equals(expected)

    def test_open_multiple_file_dataarray(self, tmp_path):
        expected = xd.DataArray(
            np.random.rand(10, 5),
            coords={
                "time": np.arange(10),
                "space": np.arange(5),
            },
        )

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
                    [
                        xd.DataArray(
                            np.random.rand(10, 5),
                            coords={
                                "time": np.arange(10),
                                "space": np.arange(5),
                            },
                        )
                    ],
                    name="acquisition",
                ),
                "DAS02": xd.DataCollection(
                    [
                        xd.DataArray(
                            np.random.rand(7, 3),
                            coords={
                                "time": np.arange(7),
                                "space": np.arange(3),
                            },
                        )
                    ],
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
            [
                xd.DataArray(
                    np.random.rand(10, 5),
                    coords={
                        "time": np.arange(10),
                        "space": np.arange(5),
                    },
                )
            ]
        )

        expected.to_netcdf(tmp_path / "collection.nc")

        result = xd.open(tmp_path / "collection.nc")
        assert result.equals(expected)

    def test_open_multiple_datacollection_with_glob(self, tmp_path):
        expected = xd.DataCollection(
            {
                "DAS01": xd.DataCollection(
                    [
                        xd.DataArray(
                            np.random.rand(10, 5),
                            coords={
                                "time": np.arange(10),
                                "space": np.arange(5),
                            },
                        )
                    ],
                    name="acquisition",
                ),
                "DAS02": xd.DataCollection(
                    [
                        xd.DataArray(
                            np.random.rand(7, 3),
                            coords={
                                "time": np.arange(7),
                                "space": np.arange(3),
                            },
                        )
                    ],
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
        with pytest.warns(RuntimeWarning):
            with pytest.raises(RuntimeError):
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
            ]
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

    def test_callable_engine(self, tmp_path):
        da = xd.DataArray(np.random.rand(10, 5), dims=("time", "distance"))
        path = str(tmp_path / "test.nc")
        da.to_netcdf(path)

        def my_engine(fname, **kwargs):
            return xd.open_dataarray(fname)

        result = xd.open_dataarray(path, engine=my_engine)
        assert result.equals(da)

    def test_invalid_engine_type_raises(self, tmp_path):
        da = xd.DataArray(np.random.rand(10, 5), dims=("time", "distance"))
        path = str(tmp_path / "test.nc")
        da.to_netcdf(path)
        with pytest.raises(ValueError, match="engine"):
            xd.open_dataarray(path, engine=42)


class TestOpenMFDatacollectionEdgeCases:
    def test_nonexistent_path_in_list_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            xd.open_mfdatacollection([str(tmp_path / "nonexistent.nc")])

    def test_empty_glob_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            xd.open_mfdatacollection(str(tmp_path / "*.nc"))

    def test_verbose_single_worker(self, tmp_path):
        da = xd.DataArray(np.random.rand(10, 5), dims=("time", "distance"))
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
        da = xd.DataArray(np.random.rand(10, 5), dims=("time", "distance"))
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
        expected = xd.DataArray(
            np.random.rand(10, 5),
            coords={"time": np.arange(10), "space": np.arange(5)},
        )
        for i, chunk in enumerate(xd.split(expected, 3, "time"), 1):
            chunk.to_netcdf(tmp_path / f"chunk_{i}.nc")
        result = xd.open_mfdataarray(tmp_path / "*.nc", parallel=2)
        assert result.equals(expected)

    def test_no_files_no_failures_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            xd.open_mfdataarray(str(tmp_path / "*.nc"))


class TestOpenMFDatacollectionParallel:
    def test_parallel_path(self, tmp_path):
        da = xd.DataArray(np.random.rand(10, 5), dims=("time", "distance"))
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
            for idx, da in enumerate(
                xd.synthetics.wavelet_wavefronts(nchunk=3), start=1
            ):
                da.to_netcdf(dirname / f"{idx:03d}.nc")
        da = xd.synthetics.wavelet_wavefronts()
        dc = xd.open_mfdatatree(tmp_path / "{node}" / "00[acquisition].nc")
        assert list(dc.keys()) == keys
        for key in keys:
            assert dc[key][0].load().equals(da)

    def test_two_level_depth(self, tmp_path):
        dc = xd.DataCollection(
            {
                "NET01": {
                    "STA01": xd.synthetics.wavelet_wavefronts(nchunk=1),
                },
                "NET02": {
                    "STA02": xd.synthetics.wavelet_wavefronts(nchunk=2),
                    "STA03": xd.synthetics.wavelet_wavefronts(nchunk=3),
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


class TestConcatCoordsEdgeCases:
    def test_tolerance_with_dense_coord_raises(self):
        da1 = xd.DataArray(
            np.random.rand(5), {"x": np.array([0.0, 1.0, 2.0, 3.0, 4.0])}
        )
        da2 = xd.DataArray(
            np.random.rand(5), {"x": np.array([5.0, 6.0, 7.0, 8.0, 9.0])}
        )
        from xdas.core.routines import concat_coords

        with pytest.raises(TypeError, match="tolerance"):
            concat_coords([da1["x"], da2["x"]], tolerance=1.0)


class TestSplitEdgeCases:
    def test_n_zero_raises(self):
        da = xd.DataArray(np.random.rand(10), dims=("time",))
        with pytest.raises(ValueError, match="`n` must be larger than 0"):
            xd.split(da, 0)

    def test_n_too_large_raises(self):
        da = xd.DataArray(np.random.rand(10), dims=("time",))
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
        da = xd.DataArray(
            np.random.rand(100),
            {
                "time": {
                    "tie_indices": [0, 99],
                    "tie_values": [
                        np.datetime64("2020-01-01"),
                        np.datetime64("2020-01-01T00:00:09.900"),
                    ],
                }
            },
        )
        fig = xd.plot_availability(da)
        assert fig is not None

    def test_datassequence_plot(self):
        da = xd.DataArray(
            np.random.rand(100),
            {
                "time": {
                    "tie_indices": [0, 99],
                    "tie_values": [
                        np.datetime64("2020-01-01"),
                        np.datetime64("2020-01-01T00:00:09.900"),
                    ],
                }
            },
        )
        dc = xd.DataCollection([da, da])
        fig = xd.plot_availability(dc)
        assert fig is not None

    def test_datamapping_plot(self):
        da = xd.DataArray(
            np.random.rand(100),
            {
                "time": {
                    "tie_indices": [0, 99],
                    "tie_values": [
                        np.datetime64("2020-01-01"),
                        np.datetime64("2020-01-01T00:00:09.900"),
                    ],
                }
            },
        )
        dm = xd.DataCollection({"a": da, "b": da})
        fig = xd.plot_availability(dm)
        assert fig is not None

    def test_invalid_type_raises(self):
        from xdas.core.routines import _get_timeline_dataframe

        with pytest.raises(TypeError, match="DataCollection"):
            _get_timeline_dataframe("not_valid")
