import h5py
import pytest

import xdas as xd
import xdas.signal as xs
from xdas.core.datacollection import get_depth


class TestDataCollection:
    def nest(self, da):
        return xd.DataCollection(
            {
                "das1": xd.DataCollection([da, da], "acquisition"),
                "das2": xd.DataCollection([da, da, da], "acquisition"),
            },
            "instrument",
        )

    def test_init(self):
        da = xd.testing.dummy()
        dc = self.nest(da)
        data = (
            "instrument",
            {
                "das1": ("acquisition", [da, da]),
                "das2": ("acquisition", [da, da, da]),
            },
        )
        result = xd.DataCollection(data)
        assert result.equals(dc)

    def test_io(self, tmp_path):
        da = xd.testing.dummy()
        dc = xd.DataCollection(
            {
                "das1": da,
                "das2": da,
            },
            "instrument",
        )
        path = tmp_path / "tmp1.nc"
        dc.to_netcdf(path)
        result = xd.DataCollection.from_netcdf(path)
        assert result.equals(dc)
        dc = xd.DataCollection([da, da], "instrument")
        path = tmp_path / "tmp2.nc"
        dc.to_netcdf(path)
        result = xd.DataCollection.from_netcdf(path)
        assert result.equals(dc)
        dc = xd.DataCollection(
            {
                "das1": xd.DataCollection([da, da], "acquisition"),
                "das2": xd.DataCollection([da, da, da], "acquisition"),
            },
            "instrument",
        )
        path = tmp_path / "tmp3.nc"
        dc.to_netcdf(path)
        result = xd.DataCollection.from_netcdf(path)
        assert result.equals(dc)
        result = xd.open_datacollection(path)
        assert result.equals(dc)

    def test_io_create_dirs(self, tmp_path):
        da = xd.testing.dummy()
        dc = xd.DataCollection(
            {
                "das1": da,
                "das2": da,
            },
            "instrument",
        )
        path = tmp_path / "subdir" / "tmp.nc"
        with pytest.raises(FileNotFoundError, match="No such file or directory"):
            dc.to_netcdf(path)
        dc.to_netcdf(path, create_dirs=True)
        result = xd.DataCollection.from_netcdf(path)
        assert result.equals(dc)

    def test_depth_counter(self, tmp_path):
        da = xd.testing.dummy()
        da.name = "da"
        dc = self.nest(da)
        path = tmp_path / "tmp.nc"
        dc.to_netcdf(path)
        with h5py.File(path) as file:
            assert get_depth(file) > 0
            assert get_depth(file["instrument"]) > 0
            assert get_depth(file["instrument/das1"]) > 0
            assert get_depth(file["instrument/das1/acquisition"]) > 0
            assert get_depth(file["instrument/das1/acquisition/0"]) == 0
            with pytest.raises(ValueError):
                get_depth(file["instrument/das1/acquisition/0/da"])

    def test_isel(self):
        da = xd.testing.dummy()
        dc = self.nest(da)
        da_isel = da.isel(distance=slice(2, 5))
        dc_isel = dc.isel(distance=slice(2, 5))
        assert self.nest(da_isel).equals(dc_isel)
        dc_isel = dc.isel(distance=slice(20, 30))
        assert dc_isel["das1"].empty
        assert dc_isel["das2"].empty

    def test_sel(self):
        da = xd.testing.dummy()
        dc = self.nest(da)
        da_sel = da.sel(distance=slice(20, 50))
        dc_sel = dc.sel(distance=slice(20, 50))
        assert self.nest(da_sel).equals(dc_sel)
        dc_sel = dc.sel(distance=slice(200, 300))
        assert dc_sel["das1"].empty
        assert dc_sel["das2"].empty

    def test_query(self):
        da = xd.testing.dummy()
        dc = self.nest(da)
        result = dc.query(instrument="das1", acquisition=0)
        expected = xd.DataCollection(
            {
                "das1": xd.DataCollection([da], "acquisition"),
            },
            "instrument",
        )
        assert result.equals(expected)
        result = dc.query(instrument="das*")
        assert result.equals(dc)
        # an indexer applies wherever its level sits, not only at the root:
        # das2 holds three acquisitions and keeps the first two
        result = dc.query(acquisition=slice(0, 2))
        assert [len(result[key]) for key in result] == [2, 2]
        assert result["das1"].equals(dc["das1"])

    def test_fields(self):
        da = xd.testing.dummy()
        dc = self.nest(da)
        assert dc.fields == ("instrument", "acquisition")

    def test_fields_recursive(self):
        da = xd.testing.dummy()
        dc = xd.DataCollection(
            {
                "DX": xd.DataCollection(
                    {
                        "CH001": xd.DataCollection(
                            {
                                "00": xd.DataCollection(
                                    {"HHZ": xd.DataCollection([da], "acquisition")},
                                    "channel",
                                )
                            },
                            "location",
                        )
                    },
                    "station",
                )
            },
            "network",
        )
        assert dc.fields == (
            "network",
            "station",
            "location",
            "channel",
            "acquisition",
        )

    def test_query_is_strict(self):
        da = xd.testing.dummy()
        dc = self.nest(da)
        with pytest.raises(KeyError, match="do not name any level"):
            dc.query(nonexistent="das1")
        # a dimension name is not a level name: `sel` trims inside leaves,
        # `query` chooses leaves
        with pytest.raises(KeyError, match="do not name any level"):
            dc.query(time=slice(0, 5))

    def test_select_is_query(self):
        da = xd.testing.dummy()
        dc = self.nest(da)
        assert dc.select(instrument="das1").equals(dc.query(instrument="das1"))
        assert dc.select({"instrument": "das1"}).equals(dc.query(instrument="das1"))

    def test_query_does_not_mutate_indexers(self):
        da = xd.testing.dummy()
        dc = self.nest(da)
        indexers = {"instrument": "das1"}
        dc.query(indexers, acquisition=0)
        assert indexers == {"instrument": "das1"}

    def test_map(self):
        da = xd.testing.dummy()
        dc = self.nest(da)
        atom = xs.decimate(..., 2, ftype="fir")
        result = dc.map(atom)
        expected = self.nest(atom(da))
        assert result.equals(expected)

    def test_flat_map(self):
        # DataMapping with DataArrays as direct values
        da = xd.testing.dummy()
        dc = xd.DataCollection({"a": da, "b": da}, "flat")
        atom = xs.decimate(..., 2, ftype="fir")
        result = dc.map(atom)
        assert result["a"].equals(atom(da))

    def test_flat_sequence_map(self):
        # DataSequence with DataArrays as direct values
        da = xd.testing.dummy()
        dc = xd.DataCollection([da, da], "seq")
        atom = xs.decimate(..., 2, ftype="fir")
        result = dc.map(atom)
        assert result[0].equals(atom(da))

    def test_datacollection_from_dataarray(self):
        da = xd.testing.dummy()
        # When DataArray is passed, rename and return it
        result = xd.DataCollection(da, "myname")
        assert isinstance(result, xd.DataArray)
        assert result.name == "myname"

    def test_datacollection_from_raw_data(self):
        import numpy as np

        data = np.ones((3, 4))
        result = xd.DataCollection(data, "raw")
        assert isinstance(result, xd.DataArray)

    def test_empty_mapping_repr(self):
        from xdas.core.datacollection import DataMapping

        dm = DataMapping({}, "empty")
        assert repr(dm) == "Empty"

    def test_mapping_reduce(self):
        import pickle

        da = xd.testing.dummy()
        dc = xd.DataCollection({"a": da}, "test")
        pickled = pickle.dumps(dc)
        restored = pickle.loads(pickled)
        assert restored.equals(dc)

    def test_sequence_reduce(self):
        import pickle

        da = xd.testing.dummy()
        dc = xd.DataCollection([da, da], "test")
        pickled = pickle.dumps(dc)
        restored = pickle.loads(pickled)
        assert restored.equals(dc)

    def test_sequence_fields(self):
        da = xd.testing.dummy()
        dc = xd.DataCollection([da, da], "seq")
        assert "seq" in dc.fields

    def test_mapping_equals_false_different_type(self):
        da = xd.testing.dummy()
        dm = xd.DataCollection({"a": da}, "test")
        assert not dm.equals(xd.DataCollection([da], "test"))

    def test_mapping_equals_false_different_name(self):
        da = xd.testing.dummy()
        dm1 = xd.DataCollection({"a": da}, "name1")
        dm2 = xd.DataCollection({"a": da}, "name2")
        assert not dm1.equals(dm2)

    def test_mapping_equals_false_different_keys(self):
        da = xd.testing.dummy()
        dm1 = xd.DataCollection({"a": da}, "test")
        dm2 = xd.DataCollection({"b": da}, "test")
        assert not dm1.equals(dm2)

    def test_mapping_equals_false_different_values(self):
        da = xd.testing.dummy()
        da2 = xd.testing.dummy()
        da2.data[:] = 0
        dm1 = xd.DataCollection({"a": da}, "test")
        dm2 = xd.DataCollection({"a": da2}, "test")
        assert not dm1.equals(dm2)

    def test_sequence_equals_false(self):
        da = xd.testing.dummy()
        ds1 = xd.DataCollection([da, da], "seq")
        ds2 = xd.DataCollection([da, da], "other")
        assert not ds1.equals(ds2)

    def test_sequence_equals_false_wrong_type(self):
        da = xd.testing.dummy()
        ds = xd.DataCollection([da], "seq")
        dm = xd.DataCollection({"a": da}, "seq")
        assert not ds.equals(dm)

    def test_sequence_load(self):
        da = xd.testing.dummy()
        dc = xd.DataCollection([da, da], "seq")
        loaded = dc.load()
        assert isinstance(loaded, type(dc))

    def test_mapping_load(self):
        da = xd.testing.dummy()
        dc = xd.DataCollection({"a": da, "b": da}, "test")
        loaded = dc.load()
        assert isinstance(loaded, type(dc))

    def test_sequence_copy(self):
        da = xd.testing.dummy()
        dc = xd.DataCollection([da, da], "seq")
        copy = dc.copy()
        assert copy.equals(dc)

    def test_sequence_isel(self):
        da = xd.testing.dummy()
        dc = xd.DataCollection([da, da], "seq")
        result = dc.isel(distance=slice(0, 100))
        assert len(result) == 2

    def test_sequence_sel(self):
        da = xd.testing.dummy()
        dc = xd.DataCollection([da, da], "seq")
        result = dc.sel(distance=slice(0, 5000))
        assert len(result) == 2

    def test_sequence_from_netcdf(self, tmp_path):
        da = xd.testing.dummy()
        dc = xd.DataCollection([da, da], "seq")
        path = tmp_path / "seq.nc"
        dc.to_netcdf(path)
        result = xd.DataCollection.from_netcdf(path)
        assert result.equals(dc)

    def test_query_invalid_key_in_sequence(self):
        da = xd.testing.dummy()
        dc = xd.DataCollection([da, da], "seq")
        with pytest.raises(ValueError, match="query must be an integer or a slice"):
            dc.query(seq="bad_string_key")

    def test_query_invalid_key_in_mapping(self):
        da = xd.testing.dummy()
        dc = xd.DataCollection({"a": da}, "test")
        with pytest.raises(ValueError, match="query must be a string"):
            dc.query(test=123)

    def test_from_netcdf_non_sequential_int_keys(self, tmp_path):
        from xdas.core.datacollection import DataMapping

        da = xd.testing.dummy()
        # Create a mapping with non-sequential int keys (gaps)
        dm = DataMapping({0: da, 2: da}, "test")
        path = tmp_path / "non_seq.nc"
        dm.to_netcdf(path)
        result = xd.DataCollection.from_netcdf(path)
        # Keys 0 and 2 are not a sequential range → returns as-is DataMapping
        assert isinstance(result, xd.DataCollection)

    def test_zero_padded_keys_survive_every_read_path(self, tmp_path):
        # a SEED location such as "00" is a mapping key, not a position: every
        # way in must compare the canonical decimal spelling, not parse ints
        da = xd.testing.dummy()
        dc = xd.DataCollection({"00": da, "01": da}, "location")
        path = tmp_path / "padded.nc"
        dc.to_netcdf(path)
        for result in (
            xd.open_datacollection(path),
            xd.DataCollection.from_netcdf(path),
            xd.open_datacollection(path, engine="xdas"),
        ):
            assert list(result) == ["00", "01"]

    def test_sequence_from_netcdf_direct(self, tmp_path):
        from xdas.core.datacollection import DataSequence

        da = xd.testing.dummy()
        dc = DataSequence([da, da], "seq")
        path = tmp_path / "seq_direct.nc"
        dc.to_netcdf(path)
        result = DataSequence.from_netcdf(str(path))
        assert result.equals(dc)

    def test_sequence_query_slice(self):
        da = xd.testing.dummy()
        dc = xd.DataCollection([da, da], "seq")
        result = dc.query(seq=slice(0, 1))
        assert len(result) == 1

    def test_mapping_repr_nonempty(self):
        da = xd.testing.dummy()
        dm = xd.DataCollection({"a": da}, "test")
        s = repr(dm)
        assert "test" in s.lower() or "Test" in s

    def test_mapping_repr_nested(self):
        # nested DataMapping → triggers the non-DataArray branch in __repr__
        da = xd.testing.dummy()
        dm = self.nest(da)
        s = repr(dm)
        assert "das1" in s

    def test_mapping_repr_int_keys(self):
        from xdas.core.datacollection import DataMapping

        da = xd.testing.dummy()
        dm = DataMapping({0: da, 1: da}, "seq")
        s = repr(dm)
        assert "0" in s

    def test_sequence_repr(self):
        da = xd.testing.dummy()
        dc = xd.DataCollection([da, da], "seq")
        s = repr(dc)
        assert "seq" in s.lower() or "Seq" in s

    def test_mapping_copy(self):
        da = xd.testing.dummy()
        dc = xd.DataCollection({"a": da}, "test")
        copy = dc.copy()
        assert copy.equals(dc)

    def test_sequence_equals_false_different_length(self):
        da = xd.testing.dummy()
        ds1 = xd.DataCollection([da, da], "seq")
        ds2 = xd.DataCollection([da], "seq")
        assert not ds1.equals(ds2)

    def test_sequence_equals_false_different_values(self):
        da = xd.testing.dummy()
        da2 = xd.testing.dummy()
        da2.data[:] = 0
        ds1 = xd.DataCollection([da], "seq")
        ds2 = xd.DataCollection([da2], "seq")
        assert not ds1.equals(ds2)

    def test_nested_sequence_map(self):
        da = xd.testing.dummy()
        inner = xd.DataCollection([da, da], "inner")
        dc = xd.DataCollection([inner, inner], "outer")
        atom = xs.decimate(..., 2, ftype="fir")
        result = dc.map(atom)
        assert len(result) == 2

    def test_parse_tuple_with_name_given(self):
        from xdas.core.datacollection import DataMapping

        da = xd.testing.dummy()
        # When data is a tuple and name is already provided, unpack the tuple ignoring its name
        dm = DataMapping(("inner_name", {"a": da}), "outer_name")
        assert dm.name == "outer_name"

    def test_parse_datacollection_propagates_name(self):
        da = xd.testing.dummy()
        dm = xd.DataCollection({"a": da}, "original_name")
        # just verify parse propagates name
        from xdas.core.datacollection import parse

        _data, name = parse(dm, None)  # should propagate dm.name
        assert name == "original_name"

    def test_mapping_map_invalid_item(self):
        from xdas.core.datacollection import DataMapping

        da = xd.testing.dummy()
        dm = DataMapping({"good": da}, "test")
        # bypass validation to inject an invalid item
        dict.__setitem__(dm, "bad", "not_a_dataarray")
        atom = xs.decimate(..., 2, ftype="fir")
        with pytest.raises(TypeError, match="encountered in the collection"):
            dm.map(atom)

    def test_sequence_map_invalid_item(self):
        from xdas.core.datacollection import DataSequence

        da = xd.testing.dummy()
        ds = DataSequence([da], "test")
        # bypass validation to inject an invalid item
        list.append(ds, "not_a_dataarray")
        atom = xs.decimate(..., 2, ftype="fir")
        with pytest.raises(TypeError, match="encountered in the collection"):
            ds.map(atom)

    def test_mapping_sel_one_element_becomes_empty(self):
        da = xd.testing.dummy()
        da_near = da.sel(distance=slice(0, 45))
        da_far = da.sel(distance=slice(50, 90))
        dc = xd.DataCollection({"near": da_near, "far": da_far}, "instrument")
        result = dc.sel(distance=slice(0, 20))
        assert set(result.keys()) == {"near"}
        assert not result["near"].empty

    def test_mapping_sel_all_elements_become_empty(self):
        da = xd.testing.dummy()
        da_near = da.sel(distance=slice(0, 45))
        da_far = da.sel(distance=slice(50, 90))
        dc = xd.DataCollection({"near": da_near, "far": da_far}, "instrument")
        result = dc.sel(distance=slice(-100, -1))
        assert len(result) == 0

    def test_sequence_sel_one_element_becomes_empty(self):
        da = xd.testing.dummy()
        da_near = da.sel(distance=slice(0, 45))
        da_far = da.sel(distance=slice(50, 90))
        dc = xd.DataCollection([da_near, da_far], "instrument")
        result = dc.sel(distance=slice(0, 20))
        assert len(result) == 1
        assert not result[0].empty

    def test_sequence_sel_all_elements_become_empty(self):
        da = xd.testing.dummy()
        da_near = da.sel(distance=slice(0, 45))
        da_far = da.sel(distance=slice(50, 90))
        dc = xd.DataCollection([da_near, da_far], "instrument")
        result = dc.sel(distance=slice(-100, -1))
        assert len(result) == 0
