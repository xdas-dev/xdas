import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest

import xdas
from xdas.synthetics import generate


class TestCore:
    def generate(self, datetime):
        shape = (300, 100)
        if datetime:
            t = {
                "tie_indices": [0, shape[0] - 1],
                "tie_values": [np.datetime64(0, "ms"), np.datetime64(2990, "ms")],
            }
        else:
            t = {"tie_indices": [0, shape[0] - 1], "tie_values": [0, 3.0 - 1 / 100]}
        s = {"tie_indices": [0, shape[1] - 1], "tie_values": [0, 990.0]}
        return xdas.Database(
            data=np.random.randn(*shape),
            coords={
                "time": t,
                "distance": s,
            },
        )

    def test_open_mfdatacollection(self):
        with TemporaryDirectory() as dirpath:
            keys = ["LOC01", "LOC02"]
            dirnames = [os.path.join(dirpath, key) for key in keys]
            [os.mkdir(dirname) for dirname in dirnames]
            [generate(dirname) for dirname in dirnames]
            db = generate()
            dc = xdas.open_mfdatacollection(
                os.path.join(dirpath, "{location}", "00{}.nc")
            )
            assert list(dc.keys()) == keys
            for key in keys:
                assert dc[key].load().equals(db)

    def test_open_mfdatabase(self):
        with TemporaryDirectory() as dirpath:
            generate(dirpath)
            db_monolithic = xdas.open_database(os.path.join(dirpath, "sample.nc"))
            db_chunked = xdas.open_mfdatabase(os.path.join(dirpath, "00*.nc"))
            assert db_monolithic.equals(db_chunked)
            db_chunked = xdas.open_mfdatabase(
                [
                    os.path.join(dirpath, fname)
                    for fname in ["001.nc", "002.nc", "003.nc"]
                ]
            )
            assert db_monolithic.equals(db_chunked)
        with pytest.raises(FileNotFoundError):
            xdas.open_mfdatabase("not_existing_files_*.nc")
        with pytest.raises(FileNotFoundError):
            xdas.open_mfdatabase(["not_existing_file.nc"])

    def test_concatenate(self):
        for datetime in [False, True]:
            db = self.generate(datetime)
            dbs = [db[100 * k : 100 * (k + 1)] for k in range(3)]
            _db = xdas.concatenate(dbs)
            assert np.array_equal(_db.data, db.data)
            assert _db["time"].equals(db["time"])
            dbs = [db[:, 20 * k : 20 * (k + 1)] for k in range(5)]
            _db = xdas.concatenate(dbs, "distance")
            assert np.array_equal(_db.data, db.data)
            assert _db["distance"].equals(db["distance"])

    def test_open_database(self):
        with pytest.raises(FileNotFoundError):
            xdas.open_database("not_existing_file.nc")

    def test_open_datacollection(self):
        with pytest.raises(FileNotFoundError):
            xdas.open_datacollection("not_existing_file.nc")

    def test_asdatabase(self):
        db = self.generate(False)
        out = xdas.asdatabase(db.to_xarray())
        assert np.array_equal(out.data, db.data)
        for dim in db.dims:
            assert np.array_equal(out[dim].values, db[dim].values)
