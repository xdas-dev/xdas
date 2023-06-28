import xdas
import numpy as np


class TestCore:
    def test_core(self):
        shape = (300, 101)
        db = xdas.Database(
            data=np.random.randn(*shape),
            coords={
                "time": xdas.Coordinate([0, shape[0] - 1], [0, 3.0 - 1 / 100]),
                "distance": xdas.Coordinate([0, shape[1] - 1], [0, 1000.0]),
            },
        )
        dbs = [db[100 * k : 100 * (k + 1)] for k in range(3)]
        _db = xdas.concatenate(dbs)
        assert np.array_equal(_db.data, db.data)
        assert _db["time"] == _db["time"]
        shape = (300, 101)
        db = xdas.Database(
            data=np.random.randn(*shape),
            coords={
                "time": xdas.Coordinate(
                    [0, shape[0] - 1],
                    [np.datetime64(0, "ms"), np.datetime64(2990, "ms")],
                ),
                "distance": xdas.Coordinate([0, shape[1] - 1], [0, 1000.0]),
            },
        )
        dbs = [db[100 * k : 100 * (k + 1)] for k in range(3)]
        _db = xdas.concatenate(dbs)
        assert np.array_equal(_db.data, db.data)
        assert _db["time"] == _db["time"]
