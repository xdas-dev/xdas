import dascore
import h5py
import hdf5plugin
import numpy as np
import pytest

from xdas.io.utils import compress


class TestCompression:

    TEST_FILES = [
        # ("ap_sensing_1.hdf5", "DAS"),  # TODO: not working for some reason...
        ("opto_das_1.hdf5", "data"),
        ("febus_1.h5", "fa1-21060063/Source1/Zone1/StrainRate"),
        ("febus_2.h5", "fa1-24090193/Source1/Zone1/PSD [dB re 1 nStrain|sqrt(Hz)]"),
        ("opta_sense_quantx_v2.h5", "Acquisition/Raw[0]/RawData"),
        ("prodml_2.0.h5", "Acquisition/Raw[0]/RawData"),
        ("prodml_2.1.h5", "Acquisition/Raw[0]/RawData"),
        ("terra15_v5_test_file.hdf5", "data_product/data"),
        ("terra15_v6_test_file.hdf5", "data_product/data"),
    ]

    @staticmethod
    def assert_attrs_equal(actual, desired, strict=False):
        if set(actual.keys()) != set(desired.keys()):
            raise AssertionError("keys mismatch")
        for key in actual:
            np.testing.assert_array_equal(actual[key], desired[key], strict)

    @pytest.mark.slow
    @pytest.mark.parametrize("test_file, dataset_location", TEST_FILES)
    def test_compression(self, tmp_path, test_file, dataset_location):
        src_path = dascore.utils.downloader.fetch(test_file)
        dst_path = tmp_path / "compressed.hdf5"
        compress(
            src_path=src_path,
            dst_path=dst_path,
            dataset_location=dataset_location,
            encoding={
                "compression": hdf5plugin.Bitshuffle(),
                "chunks": (10, 10) if "febus" not in test_file else (1, 10, 10),
            },
        )

        with (
            h5py.File(src_path, "r") as src_file,
            h5py.File(dst_path, "r") as dst_file,
        ):
            src_keys, dst_keys = [], []
            src_file.visit(src_keys.append)
            dst_file.visit(dst_keys.append)
            assert list(src_keys) == list(dst_keys)
            self.assert_attrs_equal(src_file.attrs, dst_file.attrs, strict=True)

            for name in src_keys:
                src_obj = src_file[name]
                dst_obj = dst_file[name]

                self.assert_attrs_equal(src_obj.attrs, dst_obj.attrs, strict=True)

                if isinstance(src_obj, h5py.Dataset):
                    np.testing.assert_array_equal(src_obj[()], dst_obj[()], strict=True)
                    if name != dataset_location.lstrip("/"):
                        assert src_obj.compression == dst_obj.compression
                        assert src_obj.chunks == dst_obj.chunks
