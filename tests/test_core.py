import numpy as np
import pytest

import xdas as xd
from xdas.coordinates import InterpCoordinate
from xdas.synthetics import wavelet_wavefronts
from xdas.virtual import VirtualStack


class TestCore:
    def test_open_mfdataarray(self, tmp_path):
        wavelet_wavefronts().to_netcdf(tmp_path / "sample.nc")
        for idx, da in enumerate(wavelet_wavefronts(nchunk=3), start=1):
            da.to_netcdf(tmp_path / f"{idx:03}.nc")
        da_monolithic = xd.open_dataarray(tmp_path / "sample.nc")
        da_chunked = xd.open_mfdataarray(tmp_path / "00*.nc")
        assert da_monolithic.equals(da_chunked)
        da_chunked = xd.open_mfdataarray(
            [tmp_path / fname for fname in ["001.nc", "002.nc", "003.nc"]]
        )
        assert da_monolithic.equals(da_chunked)

    with pytest.raises(FileNotFoundError):
        xd.open_mfdataarray("not_existing_files_*.nc")
    with pytest.raises(FileNotFoundError):
        xd.open_mfdataarray(["not_existing_file.nc"])

    def test_open_mfdataarray_file_limit(self, tmp_path, monkeypatch):
        from xdas.core import routines

        for idx, da in enumerate(wavelet_wavefronts(nchunk=3), start=1):
            da.to_netcdf(tmp_path / f"{idx:03}.nc")
        monkeypatch.setattr(routines, "MAX_OPEN_FILES", 2)
        with pytest.raises(NotImplementedError, match="the limit is 2"):
            xd.open_mfdataarray(tmp_path / "00*.nc")

    def test_open_mfdataarray_no_file_limit_for_tiles(self, tmp_path, monkeypatch):
        from xdas.core import routines

        for idx, da in enumerate(wavelet_wavefronts(nchunk=3), start=1):
            da.to_netcdf(tmp_path / f"{idx:03}.nc")
        monkeypatch.setattr(routines, "MAX_OPEN_FILES", 2)
        da = xd.open_mfdataarray(tmp_path / "00*.nc", engine="xdas", vtype="tiles")
        assert da.shape == wavelet_wavefronts().shape

    def test_open_mfdataarray_file_limit_engine_instance(self, tmp_path, monkeypatch):
        from xdas.core import routines
        from xdas.io.xdas import XdasEngine

        for idx, da in enumerate(wavelet_wavefronts(nchunk=3), start=1):
            da.to_netcdf(tmp_path / f"{idx:03}.nc")
        monkeypatch.setattr(routines, "MAX_OPEN_FILES", 2)
        # the configured instance carries the vtype the limit is keyed on
        with pytest.raises(NotImplementedError, match="the limit is 2"):
            xd.open_mfdataarray(tmp_path / "00*.nc", engine=XdasEngine(vtype="hdf5"))

    def test_open_mfdataarray_grouping(self, tmp_path):
        acqs = [
            {
                "starttime": "2023-01-01T00:00:00",
                "resolution": (np.timedelta64(20, "ms"), 20.0),
                "nchunk": 3,
            },
            {
                "starttime": "2023-01-01T06:00:00",
                "resolution": (np.timedelta64(10, "ms"), 20.0),
                "nchunk": 3,
            },
            {
                "starttime": "2023-01-01T12:00:00",
                "resolution": (np.timedelta64(10, "ms"), 10.0),
                "nchunk": 3,
            },
        ]
        count = 1
        for acq in acqs:
            for da in wavelet_wavefronts(**acq):
                da.to_netcdf(tmp_path / f"{count:03d}.nc")
                count += 1
        dc = xd.open_mfdataarray(tmp_path / "*.nc")
        assert len(dc) == 3
        for da, acq in zip(dc, acqs):
            acq |= {"nchunk": None}
            assert da.equals(wavelet_wavefronts(**acq))

    def test_concatenate(self, tmp_path):
        # concatenate two data arrays
        da1 = wavelet_wavefronts(starttime="2023-01-01T00:00:00")
        da2 = wavelet_wavefronts(starttime="2023-01-01T00:00:06")
        data = np.concatenate([da1.data, da2.data])
        coords = {
            "time": {
                "tie_indices": [0, da1.sizes["time"] + da2.sizes["time"] - 1],
                "tie_values": [da1["time"][0].values, da2["time"][-1].values],
                "sampling_interval": da1.coords["time"].sampling_interval,
            },
            "distance": da1["distance"],
        }
        expected = xd.DataArray(data, coords)
        result = xd.concat([da1, da2])
        assert result.equals(expected)
        # concatenate an empty data array
        result = xd.concat([da1, da2.isel(time=slice(0, 0))])
        assert result.equals(da1)
        # concat of sources and stacks
        da1.to_netcdf(tmp_path / "da1.nc")
        da2.to_netcdf(tmp_path / "da2.nc")
        da1 = xd.open_dataarray(tmp_path / "da1.nc")
        da2 = xd.open_dataarray(tmp_path / "da2.nc")
        result = xd.concat([da1, da2])
        assert isinstance(result.data, VirtualStack)
        assert result.equals(expected)
        da1.data = VirtualStack([da1.data])
        da2.data = VirtualStack([da2.data])
        result = xd.concat([da1, da2])
        assert isinstance(result.data, VirtualStack)
        assert result.equals(expected)
        # concat of 3D data arrays with unsorted coords:
        da1 = xd.DataArray(
            data=np.zeros((5, 4, 3)),
            coords={
                "phase": ["A", "B", "C"],
                "time": {"tie_indices": [0, 4], "tie_values": [0, 4]},
                "distance": [0.0, 1.0, 2.0, 3.0],
            },
            dims=("time", "distance", "phase"),
        )
        da2 = xd.DataArray(
            data=np.ones((7, 4, 3)),
            coords={
                "phase": ["A", "B", "C"],
                "time": {"tie_indices": [0, 6], "tie_values": [5, 11]},
                "distance": [0.0, 1.0, 2.0, 3.0],
            },
            dims=("time", "distance", "phase"),
        )
        expected = xd.DataArray(
            data=np.concatenate((np.zeros((5, 4, 3)), np.ones((7, 4, 3))), axis=0),
            coords={
                "time": {"tie_indices": [0, 11], "tie_values": [0, 11]},
                "distance": [0.0, 1.0, 2.0, 3.0],
                "phase": ["A", "B", "C"],
            },
        )
        assert xd.concat((da1, da2), dim="time").equals(expected)
        # concat dense coordinates
        da1 = xd.DataArray(
            data=np.zeros((5, 4, 3)),
            coords={
                "phase": ["A", "B", "C"],
                "time": [0, 1, 2, 3, 4],
                "distance": [0.0, 1.0, 2.0, 3.0],
            },
            dims=("time", "distance", "phase"),
        )
        da2 = xd.DataArray(
            data=np.ones((7, 4, 3)),
            coords={
                "phase": ["A", "B", "C"],
                "time": [5, 6, 7, 8, 9, 10, 11],
                "distance": [0.0, 1.0, 2.0, 3.0],
            },
            dims=("time", "distance", "phase"),
        )
        expected = xd.DataArray(
            data=np.concatenate((np.zeros((5, 4, 3)), np.ones((7, 4, 3))), axis=0),
            coords={
                "phase": ["A", "B", "C"],
                "time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                "distance": [0.0, 1.0, 2.0, 3.0],
            },
            dims=("time", "distance", "phase"),
        )
        assert xd.concat((da1, da2), dim="time").equals(expected)
        # stack
        da = wavelet_wavefronts()
        objs = list(da)
        result = xd.concat(objs, dim="time")
        time_values = result["time"].values
        result["time"] = InterpCoordinate(
            {
                "tie_indices": np.arange(len(time_values)),
                "tie_values": time_values,
                "sampling_interval": da.coords["time"].sampling_interval,
            },
            "time",
        ).simplify()
        assert result.equals(da)
        objs = [obj.drop_coords("time") for obj in da]
        result = xd.concat(objs, dim="time")
        assert result.equals(da.drop_coords("time"))

    def test_open_dataarray(self):
        with pytest.raises(FileNotFoundError):
            xd.open_dataarray("not_existing_file.nc")

    def test_open_datacollection(self):
        with pytest.raises(FileNotFoundError):
            xd.open_datacollection("not_existing_file.nc")

    def test_asdataarray(self):
        da = xd.testing.dummy(shape=(300, 100), datetime=False)
        out = xd.asdataarray(da.to_xarray())
        assert np.array_equal(out.data, da.data)
        for dim in da.dims:
            assert np.array_equal(out[dim].values, da[dim].values)

    def test_align(self):
        da1 = xd.DataArray(np.arange(2), {"x": [0, 1]})
        da2 = xd.DataArray(np.arange(3), {"y": [2, 3, 4]})
        da1, da2 = xd.align(da1, da2)
        assert da1.sizes == {"x": 2, "y": 1}
        assert da2.sizes == {"x": 1, "y": 3}
        da3 = xd.DataArray(np.arange(4).reshape(2, 2), {"x": [0, 1], "y": [2, 3]})
        with pytest.raises(ValueError, match="incompatible sizes"):
            xd.align(da1, da2, da3)
        da3 = xd.DataArray(np.arange(6).reshape(2, 3), {"x": [1, 2], "y": [2, 3, 4]})
        with pytest.raises(ValueError, match="differs from one data array to another"):
            xd.align(da1, da2, da3)
