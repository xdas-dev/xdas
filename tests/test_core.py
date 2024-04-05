import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest

import xdas
from xdas.synthetics import generate
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

    def test_open_treedatacollection(self):
        with TemporaryDirectory() as dirpath:
            keys = ["LOC01", "LOC02"]
            dirnames = [os.path.join(dirpath, key) for key in keys]
            for dirname in dirnames:
                os.mkdir(dirname)
                for idx, da in enumerate(generate(nchunk=3), start=1):
                    da.to_netcdf(os.path.join(dirname, f"{idx:03d}.nc"))
            da = generate()
            dc = xdas.open_treedatacollection(
                os.path.join(dirpath, "{node}", "00[acquisition].nc")
            )
            assert list(dc.keys()) == keys
            for key in keys:
                assert dc[key][0].load().equals(da)

    def test_open_mfdataarray(self):
        with TemporaryDirectory() as dirpath:
            generate().to_netcdf(os.path.join(dirpath, "sample.nc"))
            for idx, da in enumerate(generate(nchunk=3), start=1):
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
                for da in generate(**acq):
                    da.to_netcdf(os.path.join(dirpath, f"{count:03d}.nc"))
                    count += 1
            dc = xdas.open_mfdataarray(os.path.join(dirpath, "*.nc"))
            assert len(dc) == 3
            for da, acq in zip(dc, acqs):
                acq |= {"nchunk": None}
                assert da.equals(generate(**acq))

    def test_concatenate(self):
        # concatenate two dataarrays
        da1 = generate(starttime="2023-01-01T00:00:00")
        da2 = generate(starttime="2023-01-01T00:00:06")
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
        # concatenate an empty databse
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
        da = generate()
        assert xdas.concatenate(xdas.chunk(da, 3)).equals(da)
