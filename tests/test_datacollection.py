import os
from tempfile import TemporaryDirectory

import h5py
import pytest

import xdas
from xdas.core.datacollection import get_depth
from xdas.synthetics import generate


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
        da = generate()
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
        da = generate()
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
        da = generate()
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

    def test_sel(self):
        da = generate()
        dc = self.nest(da)
        da_sel = da.sel(distance=slice(1000, 2000))
        dc_sel = dc.sel(distance=slice(1000, 2000))
        assert dc_sel["das1"][0].equals(da_sel)
        dc_sel = dc.sel(distance=slice(20000, 30000))
        assert dc_sel["das1"].empty
        assert dc_sel["das2"].empty

    def test_query(self):
        da = generate()
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

    def test_fiels(self):
        da = generate()
        dc = self.nest(da)
        assert dc.fields == ("instrument", "acquisition")
