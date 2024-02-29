import os
from tempfile import TemporaryDirectory

import h5py
import pytest

import xdas
from xdas.datacollection import get_depth
from xdas.synthetics import generate


class TestDataCollection:
    def nest(self, db):
        return xdas.DataCollection(
            {
                "das1": xdas.DataCollection([db, db], "acquisition"),
                "das2": xdas.DataCollection([db, db, db], "acquisition"),
            },
            "instrument",
        )

    def test_init(self):
        db = generate()
        dc = self.nest(db)
        data = (
            "instrument",
            {
                "das1": ("acquisition", [db, db]),
                "das2": ("acquisition", [db, db, db]),
            },
        )
        result = xdas.DataCollection(data)
        assert result.equals(dc)

    def test_io(self):
        db = generate()
        dc = xdas.DataCollection(
            {
                "das1": db,
                "das2": db,
            },
            "instrument",
        )
        with TemporaryDirectory() as dirpath:
            path = os.path.join(dirpath, "tmp.nc")
            dc.to_netcdf(path)
            result = xdas.DataCollection.from_netcdf(path)
            assert result.equals(dc)
        dc = xdas.DataCollection([db, db], "instrument")
        with TemporaryDirectory() as dirpath:
            path = os.path.join(dirpath, "tmp.nc")
            dc.to_netcdf(path)
            result = xdas.DataCollection.from_netcdf(path)
            assert result.equals(dc)
        dc = xdas.DataCollection(
            {
                "das1": xdas.DataCollection([db, db], "acquisition"),
                "das2": xdas.DataCollection([db, db, db], "acquisition"),
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
        db = generate()
        db.name = "db"
        dc = self.nest(db)
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
                    get_depth(file["instrument/das1/acquisition/0/db"]) == 0

    def test_sel(self):
        db = generate()
        dc = self.nest(db)
        db_sel = db.sel(distance=slice(1000, 2000))
        dc_sel = dc.sel(distance=slice(1000, 2000))
        assert dc_sel["das1"][0].equals(db_sel)
        dc_sel = dc.sel(distance=slice(20000, 30000))
        assert dc_sel["das1"].empty
        assert dc_sel["das2"].empty

    def test_query(self):
        db = generate()
        dc = self.nest(db)
        result = dc.query(instrument="das1", acquisition=0)
        expected = xdas.DataCollection(
            {
                "das1": xdas.DataCollection([db], "acquisition"),
            },
            "instrument",
        )
        assert result.equals(expected)
        result = dc.query(instrument="das*")
        assert result.equals(dc)
        result = dc.query(acquisition=slice(0, 2))
        assert result.equals(dc)

    def test_fiels(self):
        db = generate()
        dc = self.nest(db)
        assert dc.fields == ("instrument", "acquisition")
