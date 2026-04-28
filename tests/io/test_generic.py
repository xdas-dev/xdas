import dascore as dc
import numpy as np
from dascore.utils.downloader import fetch

import xdas as xd

TEST_FILES = {
    # "apsensing": ["ap_sensing_1.hdf5"],
    "asn": ["opto_das_1.hdf5"],
    "febus": [
        "febus_1.h5",
        # "febus_2.h5",
    ],  # "valencia_febus_example.h5"
    # "optasense": ["opta_sense_quantx_v2.h5"],  # dimensions are swapped
    # "silixa": ["silixa_h5_1.hdf5"],
    # "sintela": ["sintela_binary_v3_test_1.raw"],
    "terra15": ["terra15_v5_test_file.hdf5", "terra15_v6_test_file.hdf5"],
}

for engine, fnames in TEST_FILES.items():
    for fname in fnames:
        print(engine)
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
            expected = patch.coords.get_array(dim)
            result = da[dim].values
            assert result.dtype == expected.dtype
            assert result.shape == expected.shape
            if np.issubdtype(da[dim].dtype, np.datetime64):
                result = ((result - np.datetime64(0, "s")) / np.timedelta64(1, "s"),)
                expected = (
                    (expected - np.datetime64(0, "s")) / np.timedelta64(1, "s"),
                )
            assert np.allclose(result, expected)
