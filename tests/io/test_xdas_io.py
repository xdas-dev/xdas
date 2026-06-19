"""Tests for xdas/io/xdas.py covering Engine delegates and edge cases."""

import os

import h5netcdf
import numpy as np
import pytest

import xdas as xd
from xdas.core.datacollection import DataMapping
from xdas.io.core import Engine
from xdas.io.xdas import (
    open_dataarray,
    open_datacollection,
    open_datasequence,
    save_dataarray,
    save_datacollection,
    save_datamapping,
    save_datasequence,
)


def make_da():
    return xd.DataArray(
        np.zeros((10, 5), dtype=np.float32),
        {
            "time": {
                "tie_indices": [0, 9],
                "tie_values": [
                    np.datetime64("2020-01-01T00:00:00.000000000"),
                    np.datetime64("2020-01-01T00:00:09.000000000"),
                ],
            },
            "distance": {"tie_indices": [0, 4], "tie_values": [0.0, 40.0]},
        },
    )


class TestXdasEngineDelegates:
    def test_save_and_open_dataarray_with_str(self, tmp_path):
        da = make_da()
        path = str(tmp_path / "da.nc")
        engine = Engine["xdas"]()
        engine.save_dataarray(da, path)
        result = engine.open_dataarray(path)
        assert result.equals(da)

    def test_save_and_open_datasequence_with_str(self, tmp_path):
        da = make_da()
        dc = xd.DataCollection([da, da])
        path = str(tmp_path / "dc.nc")
        engine = Engine["xdas"]()
        engine.save_datacollection(dc, path)
        result = engine.open_datacollection(path)
        assert result.equals(dc)

    def test_save_and_open_datamapping_with_str(self, tmp_path):
        da = make_da()
        dc = xd.DataCollection({"A": da, "B": da})
        engine = Engine["xdas"]()
        path = str(tmp_path / "dc.nc")
        engine.save_datacollection(dc, path)
        result = engine.open_datacollection(path)
        assert result.equals(dc)

    def test_save_and_open_dataarray_with_path(self, tmp_path):
        da = make_da()
        path = tmp_path / "da.nc"
        engine = Engine["xdas"]()
        engine.save_dataarray(da, path)
        result = engine.open_dataarray(path)
        assert result.equals(da)

    def test_save_and_open_datasequence_with_path(self, tmp_path):
        da = make_da()
        dc = xd.DataCollection([da, da])
        path = tmp_path / "dc.nc"
        engine = Engine["xdas"]()
        engine.save_datacollection(dc, path)
        result = engine.open_datacollection(path)
        assert result.equals(dc)

    def test_save_and_open_datamapping_with_path(self, tmp_path):
        da = make_da()
        dc = xd.DataCollection({"A": da, "B": da})
        engine = Engine["xdas"]()
        path = tmp_path / "dc.nc"
        engine.save_datacollection(dc, path)
        result = engine.open_datacollection(path)
        assert result.equals(dc)


class TestSaveDataarrayEdgeCases:
    def test_encoding_with_virtual_raises(self, tmp_path):
        da = make_da()
        path = str(tmp_path / "da.nc")
        # First save as virtual, then try to re-save with encoding
        da.to_netcdf(path)
        da_virtual = xd.open_dataarray(path)
        path2 = tmp_path / "test2.nc"
        with pytest.raises(ValueError, match="encoding"):
            save_dataarray(da_virtual, path2, virtual=True, encoding={"chunks": (2, 2)})

    def test_virtual_true_with_plain_array_raises(self, tmp_path):
        da = make_da()
        path = str(tmp_path / "da.nc")
        with pytest.raises(ValueError, match="virtual array"):
            save_dataarray(da, path, virtual=True)

    def test_create_dirs_with_no_dirname(self, tmp_path):
        da = make_da()
        orig = os.getcwd()
        os.chdir(tmp_path)
        try:
            save_dataarray(da, "bare_file.nc", create_dirs=True)
            assert (tmp_path / "bare_file.nc").exists()
        finally:
            os.chdir(orig)


class TestOpenDatacollection:
    def test_integer_keys_become_sequence(self, tmp_path):
        da = make_da()
        dc = xd.DataCollection([da, da])
        path = tmp_path / "seq.nc"
        dc.to_netcdf(path)
        result = open_datacollection(path)
        assert result.equals(dc)

    def test_non_integer_keys_stay_mapping(self, tmp_path):
        da = make_da()
        dc = xd.DataCollection({"a": da, "b": da})
        path = tmp_path / "map.nc"
        dc.to_netcdf(path)
        result = open_datacollection(path)
        assert result.equals(dc)


class TestSaveDatacollection:
    def test_save_sequence(self, tmp_path):
        da = make_da()
        dc = xd.DataCollection([da, da])
        path = str(tmp_path / "seq.nc")
        save_datacollection(dc, path)
        result = xd.DataCollection.from_netcdf(path)
        assert result.equals(dc)

    def test_save_mapping(self, tmp_path):
        da = make_da()
        dc = xd.DataCollection({"a": da, "b": da})
        path = str(tmp_path / "map.nc")
        save_datacollection(dc, path)
        result = xd.DataCollection.from_netcdf(path)
        assert result.equals(dc)

    def test_invalid_type_raises(self, tmp_path):
        path = str(tmp_path / "bad.nc")
        with pytest.raises(ValueError, match="DataCollection"):
            save_datacollection("not_a_collection", path)


class TestSaveDatamappingOverwrite:
    def test_overwrite_existing_file(self, tmp_path):
        da = make_da()
        dc = xd.DataCollection({"x": da})
        path = str(tmp_path / "overwrite.nc")
        dc.to_netcdf(path)
        assert os.path.exists(path)
        dc.to_netcdf(path)  # overwrite with mode="w"
        result = xd.DataCollection.from_netcdf(path)
        assert result.equals(dc)


class TestSaveDatamappingCreateDirs:
    def test_create_dirs_no_dirname(self, tmp_path):
        da = make_da()
        orig = os.getcwd()
        os.chdir(tmp_path)
        try:
            dm = DataMapping({"x": da})
            save_datamapping(dm, "bare_dc.nc", create_dirs=True)
            assert (tmp_path / "bare_dc.nc").exists()
        finally:
            os.chdir(orig)


class TestOpenSaveDatasequence:
    def test_open_datasequence(self, tmp_path):
        da = make_da()
        dc = xd.DataCollection([da, da])
        path = str(tmp_path / "seq.nc")
        dc.to_netcdf(path)
        result = open_datasequence(path)
        assert result.equals(dc)

    def test_save_datasequence(self, tmp_path):
        da = make_da()
        ds = xd.DataSequence([da, da])
        path = str(tmp_path / "seq.nc")
        save_datasequence(ds, path)
        result = xd.DataCollection.from_netcdf(path)
        assert result.equals(ds)


class TestOpenDataarrayEdgeCases:
    def test_multiple_coordinate_vars_raises(self, tmp_path):
        path = tmp_path / "multi.nc"
        with h5netcdf.File(str(path), "w") as f:
            f.attrs["Conventions"] = "CF-1.9"
            f.dimensions["time"] = 5
            f.dimensions["distance"] = 3
            v1 = f.create_variable("var1", ("time", "distance"), float)
            v1.attrs["coordinate_interpolation"] = "something"
            v2 = f.create_variable("var2", ("time", "distance"), float)
            v2.attrs["coordinate_interpolation"] = "something"
        with pytest.raises(ValueError, match="several possible"):
            open_dataarray(str(path))

    def test_path_object_accepted(self, tmp_path):
        da = make_da()
        path = tmp_path / "da.nc"
        da.to_netcdf(str(path))
        result = open_dataarray(path)  # pass Path, not str
        assert result.equals(da)


class TestOpenDatacollectionPathObject:
    def test_path_object_accepted(self, tmp_path):
        da = make_da()
        dc = xd.DataCollection([da, da])
        path = tmp_path / "dc.nc"
        dc.to_netcdf(str(path))
        result = open_datacollection(path)  # pass Path, not str
        assert result.equals(dc)


class TestSaveDatacollectionPathObject:
    def test_path_object_accepted(self, tmp_path):
        da = make_da()
        dc = xd.DataCollection([da, da])
        path = tmp_path / "dc.nc"
        save_datacollection(dc, path)  # pass Path, not str
        result = xd.DataCollection.from_netcdf(str(path))
        assert result.equals(dc)


class TestOpenDatacollectionNonSequentialKeys:
    def test_non_sequential_integers_stay_mapping(self, tmp_path):
        da = make_da()
        dc = xd.DataCollection({"2": da, "5": da})
        path = str(tmp_path / "nonseq.nc")
        dc.to_netcdf(path)
        result = open_datacollection(path)
        assert result.equals(dc)
