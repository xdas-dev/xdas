import numpy as np

from xdas.core import Database
from xdas.io.asn import build_database

# paths = [
#     "/home/trabatto/data/abyss/20211106/proc/035622.hdf5",
#     "/home/trabatto/data/abyss/20211106/proc/103752.hdf5",
#     "/home/trabatto/data/abyss/20211106/proc/161052.hdf5",
# ]


# class TestASN:
#     def test_build_database(self):
#         fname = "/data/results/trabatto/data/database_test.h5"
#         build_database(fname, paths, np.timedelta64(30, "ms"))
#         Database.from_hdf(fname)
