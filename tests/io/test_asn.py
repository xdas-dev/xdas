import numpy as np

from xdas.io.asn import build_database

fnames = [
    "/home/trabatto/data/abyss/20211106/proc/035622.hdf5",
    "/home/trabatto/data/abyss/20211106/proc/103752.hdf5",
    "/home/trabatto/data/abyss/20211106/proc/161052.hdf5",
]

class TestASN:
    def test_build_database(self):
        build_database(fnames, np.timedelta64(30, "ms"))