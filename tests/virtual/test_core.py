import h5py
import numpy as np
import pytest

from xdas.virtual import (
    TileArray,
    VirtualArray,
    VirtualBackend,
    VirtualSource,
    VirtualStack,
)


class TestVirtualBackend:
    def test_lookup(self):
        assert VirtualBackend["hdf5"] is VirtualArray
        assert VirtualBackend["tiles"] is TileArray

    def test_unknown_vtype_raises_key_error(self):
        with pytest.raises(KeyError, match="no virtual backend registered"):
            VirtualBackend["netcdf"]

    def test_registry_holds_only_named_backends(self):
        assert set(VirtualBackend._registry) == {"hdf5", "tiles"}

    def test_subclasses_inherit_vtype_without_reregistering(self):
        assert VirtualSource.vtype == "hdf5"
        assert VirtualStack.vtype == "hdf5"
        assert VirtualBackend["hdf5"] is VirtualArray

    def test_consolidates(self):
        assert TileArray.consolidates
        assert not VirtualArray.consolidates
        assert not VirtualBackend.consolidates

    def test_isinstance_covers_both_backends(self):
        source = VirtualSource("path.h5", "data", (2, 3), "f8")
        assert isinstance(source, VirtualBackend)
        assert isinstance(VirtualStack([source]), VirtualBackend)

    def test_from_variable_dispatches_to_each_backend(self, tmp_path):
        data = np.arange(6.0).reshape(2, 3)
        path = tmp_path / "source.h5"
        with h5py.File(path, "w") as file:
            file.create_dataset("data", data=data)
        with h5py.File(path) as file:
            source = VirtualBackend["hdf5"].from_variable(file["data"])
            tiled = VirtualBackend["tiles"].from_variable(file["data"])
        assert isinstance(source, VirtualSource)
        assert isinstance(tiled, TileArray)
        np.testing.assert_array_equal(np.asarray(source), data)
        np.testing.assert_array_equal(np.asarray(tiled), data)
