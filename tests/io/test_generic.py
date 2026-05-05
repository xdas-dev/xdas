import dascore as dc
import numpy as np
from dascore.utils.downloader import fetch

import xdas as xd


class TestGenericIO:
    TEST_FILES = {
        "apsensing": ["ap_sensing_1.hdf5"],
        "asn": ["opto_das_1.hdf5"],
        "febus": ["febus_1.h5", "febus_2.h5"],
        "optasense": ["opta_sense_quantx_v2.h5"],
        "silixa": ["sample_tdms_file_v4713.tdms"],
        "prodml": ["prodml_2.0.h5", "prodml_2.1.h5"],
        "terra15": ["terra15_v5_test_file.hdf5", "terra15_v6_test_file.hdf5"],
    }

    SKIP_DISTANCE_COMPARISON = [
        "ap_sensing_1.hdf5",  # NOTE: we disagree with dascore
        "sample_tdms_file_v4713.tdms",  # NOTE: dascore does not really know what it does
    ]

    def test_auto_open_files(self):
        for engine, fnames in self.TEST_FILES.items():
            for fname in fnames:
                path = fetch(fname)
                da = xd.open(path, engine=engine)
                da_auto = xd.open(path)
                assert da.equals(da_auto)

    def test_compare_with_dascore(self):
        for engine, fnames in self.TEST_FILES.items():
            for fname in fnames:
                path = fetch(fname)
                da = xd.open(path, engine=engine)
                spool = dc.read(path)
                patch = spool[0]
                assert isinstance(da, xd.DataArray)
                assert da.dtype == patch.dtype
                assert da.shape == patch.shape
                assert np.array_equal(da.data, patch.data, equal_nan=True)
                assert da.dims == patch.dims
                for dim in da.dims:
                    if dim == "distance" and fname in self.SKIP_DISTANCE_COMPARISON:
                        continue
                    expected = patch.coords.get_array(dim)
                    result = da[dim].values
                    # assert result.dtype == expected.dtype
                    assert result.shape == expected.shape
                    if np.issubdtype(da[dim].dtype, np.datetime64):
                        result = (
                            (result - np.datetime64(0, "s")) / np.timedelta64(1, "s"),
                        )
                        expected = (
                            (expected - np.datetime64(0, "s")) / np.timedelta64(1, "s"),
                        )
                    assert np.allclose(result, expected)
