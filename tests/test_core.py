import numpy as np
import pytest

import xdas


class TestCore:
    def generate(self, datetime):
        shape = (300, 100)
        if datetime:
            t = xdas.InterpolatedCoordinate(
                [0, shape[0] - 1],
                [np.datetime64(0, "ms"), np.datetime64(2990, "ms")],
            )
        else:
            t = xdas.InterpolatedCoordinate([0, shape[0] - 1], [0, 3.0 - 1 / 100])
        s = xdas.InterpolatedCoordinate([0, shape[1] - 1], [0, 990.0])
        return xdas.Database(
            data=np.random.randn(*shape),
            coords={
                "time": t,
                "distance": s,
            },
        )

    def test_open_mfdatabase(self):
        with pytest.raises(FileNotFoundError):
            xdas.open_mfdatabase("not_existing_files_*.nc")

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
            assert out[dim].equals(db[dim])
