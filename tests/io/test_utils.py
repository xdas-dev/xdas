import dascore
import h5py
import hdf5plugin
import numpy as np
import pytest

from xdas.io.utils import compress


class TestCompression:

    TEST_FILES = [
        # "ap_sensing_1.hdf5",
        ("opto_das_1.hdf5", "data"),
        # "febus_1.h5",
        # "febus_2.h5",
        # "opta_sense_quantx_v2.h5",
        # "prodml_2.0.h5",
        # "prodml_2.1.h5",
        # "terra15_v5_test_file.hdf5",
        # "terra15_v6_test_file.hdf5",
    ]

    @pytest.mark.parametrize("test_file, dataset_location", TEST_FILES)
    def test_compression(self, tmp_path, test_file, dataset_location):
        src_path = dascore.utils.downloader.fetch(test_file)
        dst_path = tmp_path / f"{test_file}_compressed.hdf5"
        compress(
            src_path=src_path,
            dst_path=dst_path,
            dataset_location=dataset_location,
            encoding={
                "compression": hdf5plugin.Bitshuffle(),
                "chunks": (5, 5),
            },
        )

        with (
            h5py.File(src_path, "r") as original_file,
            h5py.File(dst_path, "r") as compressed_file,
        ):

            orig_keys, comp_keys = [], []
            original_file.visit(orig_keys.append)
            compressed_file.visit(comp_keys.append)
            assert list(orig_keys) == list(comp_keys)
            assert dict(original_file.attrs) == dict(compressed_file.attrs)

            for name in orig_keys:
                orig_obj = original_file[name]
                comp_obj = compressed_file[name]

                assert dict(orig_obj.attrs) == dict(comp_obj.attrs)

                if isinstance(orig_obj, h5py.Dataset):
                    assert orig_obj.shape == comp_obj.shape
                    assert orig_obj.dtype == comp_obj.dtype

                    if name != dataset_location.lstrip("/"):
                        np.testing.assert_array_equal(orig_obj[()], comp_obj[()])
                        assert orig_obj.compression == comp_obj.compression
                        assert orig_obj.chunks == comp_obj.chunks
