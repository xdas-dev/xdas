import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest

import xdas
from xdas.synthetics import wavelet_wavefronts
from xdas.virtual import VirtualStack


class TestCore:
    def generate(self, datetime):
        shape = (300, 100)
        if datetime:
            t = {
                "tie_indices": [0, shape[0] - 1],
                "tie_values": [np.datetime64(0, "ms"), np.datetime64(2990, "ms")],
            }
        else:
            t = {"tie_indices": [0, shape[0] - 1], "tie_values": [0, 3.0 - 1 / 100]}
        s = {"tie_indices": [0, shape[1] - 1], "tie_values": [0, 990.0]}
        return xdas.DataArray(
            data=np.random.randn(*shape),
            coords={
                "time": t,
                "distance": s,
            },
        )

    def test_open_mfdatacollection(self): ...  # TODO

    def test_open_mfdatatree(self):
        with TemporaryDirectory() as dirpath:
            keys = ["LOC01", "LOC02"]
            dirnames = [os.path.join(dirpath, key) for key in keys]
            for dirname in dirnames:
                os.mkdir(dirname)
                for idx, da in enumerate(wavelet_wavefronts(nchunk=3), start=1):
                    da.to_netcdf(os.path.join(dirname, f"{idx:03d}.nc"))
            da = wavelet_wavefronts()
            dc = xdas.open_mfdatatree(
                os.path.join(dirpath, "{node}", "00[acquisition].nc")
            )
            assert list(dc.keys()) == keys
            for key in keys:
                assert dc[key][0].load().equals(da)

    def test_open_mfdataarray(self):
        with TemporaryDirectory() as dirpath:
            wavelet_wavefronts().to_netcdf(os.path.join(dirpath, "sample.nc"))
            for idx, da in enumerate(wavelet_wavefronts(nchunk=3), start=1):
                da.to_netcdf(os.path.join(dirpath, f"{idx:03}.nc"))
            da_monolithic = xdas.open_dataarray(os.path.join(dirpath, "sample.nc"))
            da_chunked = xdas.open_mfdataarray(os.path.join(dirpath, "00*.nc"))
            assert da_monolithic.equals(da_chunked)
            da_chunked = xdas.open_mfdataarray(
                [
                    os.path.join(dirpath, fname)
                    for fname in ["001.nc", "002.nc", "003.nc"]
                ]
            )
            assert da_monolithic.equals(da_chunked)
        with pytest.raises(FileNotFoundError):
            xdas.open_mfdataarray("not_existing_files_*.nc")
        with pytest.raises(FileNotFoundError):
            xdas.open_mfdataarray(["not_existing_file.nc"])

    def test_open_mfdataarray_grouping(self):
        with TemporaryDirectory() as dirpath:
            acqs = [
                {
                    "starttime": "2023-01-01T00:00:00",
                    "resolution": (np.timedelta64(20, "ms"), 20.0),
                    "nchunk": 10,
                },
                {
                    "starttime": "2023-01-01T06:00:00",
                    "resolution": (np.timedelta64(10, "ms"), 20.0),
                    "nchunk": 10,
                },
                {
                    "starttime": "2023-01-01T12:00:00",
                    "resolution": (np.timedelta64(10, "ms"), 10.0),
                    "nchunk": 10,
                },
            ]
            count = 1
            for acq in acqs:
                for da in wavelet_wavefronts(**acq):
                    da.to_netcdf(os.path.join(dirpath, f"{count:03d}.nc"))
                    count += 1
            dc = xdas.open_mfdataarray(os.path.join(dirpath, "*.nc"))
            assert len(dc) == 3
            for da, acq in zip(dc, acqs):
                acq |= {"nchunk": None}
                assert da.equals(wavelet_wavefronts(**acq))

    def test_concatenate(self):
        # concatenate two data arrays
        da1 = wavelet_wavefronts(starttime="2023-01-01T00:00:00")
        da2 = wavelet_wavefronts(starttime="2023-01-01T00:00:06")
        data = np.concatenate([da1.data, da2.data])
        coords = {
            "time": {
                "tie_indices": [0, da1.sizes["time"] + da2.sizes["time"] - 1],
                "tie_values": [da1["time"][0].values, da2["time"][-1].values],
            },
            "distance": da1["distance"],
        }
        expected = xdas.DataArray(data, coords)
        result = xdas.concatenate([da1, da2])
        assert result.equals(expected)
        # concatenate an empty data array
        result = xdas.concatenate([da1, da2.isel(time=slice(0, 0))])
        assert result.equals(da1)
        # concat of sources and stacks
        with TemporaryDirectory() as tmp_path:
            da1.to_netcdf(os.path.join(tmp_path, "da1.nc"))
            da2.to_netcdf(os.path.join(tmp_path, "da2.nc"))
            da1 = xdas.open_dataarray(os.path.join(tmp_path, "da1.nc"))
            da2 = xdas.open_dataarray(os.path.join(tmp_path, "da2.nc"))
            result = xdas.concatenate([da1, da2])
            assert isinstance(result.data, VirtualStack)
            assert result.equals(expected)
            da1.data = VirtualStack([da1.data])
            da2.data = VirtualStack([da2.data])
            result = xdas.concatenate([da1, da2])
            assert isinstance(result.data, VirtualStack)
            assert result.equals(expected)
        # concat of 3D data arrays with unsorted coords:
        da1 = xdas.DataArray(
            data=np.zeros((5, 4, 3)),
            coords={
                "phase": ["A", "B", "C"],
                "time": {"tie_indices": [0, 4], "tie_values": [0, 4]},
                "distance": [0.0, 1.0, 2.0, 3.0],
            },
            dims=("time", "distance", "phase"),
        )
        da2 = xdas.DataArray(
            data=np.ones((7, 4, 3)),
            coords={
                "phase": ["A", "B", "C"],
                "time": {"tie_indices": [0, 6], "tie_values": [5, 11]},
                "distance": [0.0, 1.0, 2.0, 3.0],
            },
            dims=("time", "distance", "phase"),
        )
        expected = xdas.DataArray(
            data=np.concatenate((np.zeros((5, 4, 3)), np.ones((7, 4, 3))), axis=0),
            coords={
                "time": {"tie_indices": [0, 11], "tie_values": [0, 11]},
                "distance": [0.0, 1.0, 2.0, 3.0],
                "phase": ["A", "B", "C"],
            },
        )
        assert xdas.concatenate((da1, da2), dim="time").equals(expected)

    def test_open_dataarray(self):
        with pytest.raises(FileNotFoundError):
            xdas.open_dataarray("not_existing_file.nc")

    def test_open_datacollection(self):
        with pytest.raises(FileNotFoundError):
            xdas.open_datacollection("not_existing_file.nc")

    def test_asdataarray(self):
        da = self.generate(False)
        out = xdas.asdataarray(da.to_xarray())
        assert np.array_equal(out.data, da.data)
        for dim in da.dims:
            assert np.array_equal(out[dim].values, da[dim].values)

    def test_split(self):
        da = xdas.DataArray(
            np.ones(30),
            {
                "time": {
                    "tie_indices": [0, 9, 10, 19, 20, 29],
                    "tie_values": [0.0, 9.0, 20.0, 29.0, 40.0, 49.0],
                },
            },
        )
        assert xdas.concatenate(xdas.split(da)).equals(da)
        assert xdas.split(da, tolerance=20.0)[0].equals(da)

    def test_chunk(self):
        da = wavelet_wavefronts()
        assert xdas.concatenate(xdas.split(da, 3)).equals(da)

    def test_align(self):
        da1 = xdas.DataArray(np.arange(2), {"x": [0, 1]})
        da2 = xdas.DataArray(np.arange(3), {"y": [2, 3, 4]})
        da1, da2 = xdas.align(da1, da2)
        assert da1.sizes == {"x": 2, "y": 1}
        assert da2.sizes == {"x": 1, "y": 3}
        da3 = xdas.DataArray(np.arange(4).reshape(2, 2), {"x": [0, 1], "y": [2, 3]})
        with pytest.raises(ValueError, match="incompatible sizes"):
            xdas.align(da1, da2, da3)
        da3 = xdas.DataArray(np.arange(6).reshape(2, 3), {"x": [1, 2], "y": [2, 3, 4]})
        with pytest.raises(ValueError, match="differs from one data array to another"):
            xdas.align(da1, da2, da3)
