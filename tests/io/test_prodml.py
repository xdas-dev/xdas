import h5py
import numpy as np

import xdas as xd


def make_prodml_file(path, swapped=False):
    nt, nd = 10, 5
    dx = 2.0
    data = np.zeros((nd, nt) if swapped else (nt, nd), dtype=np.float32)
    with h5py.File(path, "w") as f:
        acq = f.create_group("Acquisition")
        acq.attrs["SpatialSamplingInterval"] = dx
        acq.attrs["StartLocusIndex"] = 0
        raw = acq.create_group("Raw[0]")
        ds = raw.create_dataset("RawData", data=data)
        ds.attrs["PartStartTime"] = np.bytes_(b"2020-01-01T00:00:00.000+00:00")
        ds.attrs["PartEndTime"] = np.bytes_(b"2020-01-01T00:00:00.900+00:00")


class TestProdMLEngine:
    def test_open_swapped_dims(self, tmp_path):
        path = tmp_path / "prodml_swapped.h5"
        make_prodml_file(path, swapped=True)
        da = xd.open(str(path), engine="prodml", swapped_dims=True)
        assert isinstance(da, xd.DataArray)
        assert da.dims == ("distance", "time")

    def test_open_normal(self, tmp_path):
        path = tmp_path / "prodml.h5"
        make_prodml_file(path)
        da = xd.open(str(path), engine="prodml")
        assert isinstance(da, xd.DataArray)
        assert da.dims == ("time", "distance")
