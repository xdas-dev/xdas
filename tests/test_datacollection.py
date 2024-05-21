import os
from tempfile import TemporaryDirectory

import h5py
import pytest

import xdas
import xdas.signal as xs
from xdas.core.datacollection import get_depth
from xdas.synthetics import wavelet_wavefronts


class TestDataCollection:
    def nest(self, da):
        return xdas.DataCollection(
            {
                "das1": xdas.DataCollection([da, da], "acquisition"),
                "das2": xdas.DataCollection([da, da, da], "acquisition"),
            },
            "instrument",
        )

    def test_init(self):
        da = wavelet_wavefronts()
        dc = self.nest(da)
        data = (
            "instrument",
            {
                "das1": ("acquisition", [da, da]),
                "das2": ("acquisition", [da, da, da]),
            },
        )
        result = xdas.DataCollection(data)
        assert result.equals(dc)

    def test_io(self):
        da = wavelet_wavefronts()
        dc = xdas.DataCollection(
            {
                "das1": da,
                "das2": da,
            },
            "instrument",
        )
        with TemporaryDirectory() as dirpath:
            path = os.path.join(dirpath, "tmp.nc")
            dc.to_netcdf(path)
            result = xdas.DataCollection.from_netcdf(path)
            assert result.equals(dc)
        dc = xdas.DataCollection([da, da], "instrument")
        with TemporaryDirectory() as dirpath:
            path = os.path.join(dirpath, "tmp.nc")
            dc.to_netcdf(path)
            result = xdas.DataCollection.from_netcdf(path)
            assert result.equals(dc)
        dc = xdas.DataCollection(
            {
                "das1": xdas.DataCollection([da, da], "acquisition"),
                "das2": xdas.DataCollection([da, da, da], "acquisition"),
            },
            "instrument",
        )
        with TemporaryDirectory() as dirpath:
            path = os.path.join(dirpath, "tmp.nc")
            dc.to_netcdf(path)
            result = xdas.DataCollection.from_netcdf(path)
            assert result.equals(dc)
            result = xdas.open_datacollection(path)
            assert result.equals(dc)

    def test_depth_counter(self):
        da = wavelet_wavefronts()
        da.name = "da"
        dc = self.nest(da)
        with TemporaryDirectory() as dirpath:
            path = os.path.join(dirpath, "tmp.nc")
            dc.to_netcdf(path)
            with h5py.File(path) as file:
                assert get_depth(file) > 0
                assert get_depth(file["instrument"]) > 0
                assert get_depth(file["instrument/das1"]) > 0
                assert get_depth(file["instrument/das1/acquisition"]) > 0
                assert get_depth(file["instrument/das1/acquisition/0"]) == 0
                with pytest.raises(ValueError):
                    get_depth(file["instrument/das1/acquisition/0/da"]) == 0

    def test_isel(self):
        da = wavelet_wavefronts()
        dc = self.nest(da)
        da_isel = da.isel(distance=slice(100, 200))
        dc_isel = dc.isel(distance=slice(100, 200))
        assert self.nest(da_isel).equals(dc_isel)
        dc_isel = dc.isel(distance=slice(2000, 3000))
        assert dc_isel["das1"].empty
        assert dc_isel["das2"].empty

    def test_sel(self):
        da = wavelet_wavefronts()
        dc = self.nest(da)
        da_sel = da.sel(distance=slice(1000, 2000))
        dc_sel = dc.sel(distance=slice(1000, 2000))
        assert self.nest(da_sel).equals(dc_sel)
        dc_sel = dc.sel(distance=slice(20000, 30000))
        assert dc_sel["das1"].empty
        assert dc_sel["das2"].empty

    def test_query(self):
        da = wavelet_wavefronts()
        dc = self.nest(da)
        result = dc.query(instrument="das1", acquisition=0)
        expected = xdas.DataCollection(
            {
                "das1": xdas.DataCollection([da], "acquisition"),
            },
            "instrument",
        )
        assert result.equals(expected)
        result = dc.query(instrument="das*")
        assert result.equals(dc)
        result = dc.query(acquisition=slice(0, 2))
        assert result.equals(dc)

    def test_fields(self):
        da = wavelet_wavefronts()
        dc = self.nest(da)
        assert dc.fields == ("instrument", "acquisition")

    def test_map(self):
        da = wavelet_wavefronts()
        dc = self.nest(da)
        atom = xs.decimate(..., 2, ftype="fir")
        result = dc.map(atom)
        expected = self.nest(atom(da))
        assert result.equals(expected)
