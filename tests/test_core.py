import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest

import xdas
from xdas.core import collects, splits
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
        ...  # TODO

    def test_open_treedatacollection(self):
        with TemporaryDirectory() as dirpath:
            keys = ["LOC01", "LOC02"]
            dirnames = [os.path.join(dirpath, key) for key in keys]
            for dirname in dirnames:
                os.mkdir(dirname)
                for idx, db in enumerate(generate(nchunk=3), start=1):
                    db.to_netcdf(os.path.join(dirname, f"{idx:03d}.nc"))
            db = generate()
            dc = xdas.open_treedatacollection(
                os.path.join(dirpath, "{node}", "00[acquisition].nc")
            )
            assert list(dc.keys()) == keys
            for key in keys:
                assert dc[key][0].load().equals(db)

    def test_open_mfdatabase(self):
        with TemporaryDirectory() as dirpath:
            generate().to_netcdf(os.path.join(dirpath, "sample.nc"))
            for idx, db in enumerate(generate(nchunk=3), start=1):
                db.to_netcdf(os.path.join(dirpath, f"{idx:03}.nc"))
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

    def test_open_mfdatabase_grouping(self):
        with TemporaryDirectory() as dirpath:
            acqs = [
                {
                    "starttime": "2023-01-01T00:00:00",
                    "resolution": (np.timedelta64(20, "ms"), 20.0),
                    "nchunk": 10,
                },
                {
                    "starttime": "2023-01-01T06:00:00",
                    "resolution": (np.timedelta64(10, "ms"), 20.0),
                    "nchunk": 10,
                },
                {
                    "starttime": "2023-01-01T12:00:00",
                    "resolution": (np.timedelta64(10, "ms"), 10.0),
                    "nchunk": 10,
                },
            ]
            count = 1
            for acq in acqs:
                for db in generate(**acq):
                    db.to_netcdf(os.path.join(dirpath, f"{count:03d}.nc"))
                    count += 1
            dc = xdas.open_mfdatabase(os.path.join(dirpath, "*.nc"))
            assert len(dc) == 3
            for db, acq in zip(dc, acqs):
                acq |= {"nchunk": None}
                assert db.equals(generate(**acq))

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

    def test_split(self):
        db = xdas.Database(
            np.ones(30),
            {
                "time": {
                    "tie_indices": [0, 9, 10, 19, 20, 29],
                    "tie_values": [0.0, 9.0, 20.0, 29.0, 40.0, 49.0],
                },
            },
        )
        assert xdas.concatenate(xdas.split(db)).equals(db)
        assert xdas.split(db, tolerance=20.0)[0].equals(db)

    def test_chunk(self):
        db = generate()
        assert xdas.concatenate(xdas.chunk(db, 3)).equals(db)

    def test_collects(self):
        @collects
        def double(db):
            return db * 2

        db = generate()
        dc = xdas.DataCollection(("node", {"DAS": ("acquisition", [db, db])}))
        expected = xdas.DataCollection(
            ("node", {"DAS": ("acquisition", [db * 2, db * 2])})
        )
        result = double(dc)
        assert result.equals(expected)

    def test_splits(self):
        @splits
        @collects
        def diff(db, dim="last"):
            return db.diff(dim=dim)

        db1 = generate(starttime="2023-01-01T00:00:00")
        db2 = generate(starttime="2023-01-01T00:00:10")
        db = xdas.concatenate([db1, db2])
        result = diff(db, dim="time")
        naive = db.diff("time")
        expected = xdas.concatenate([db1.diff("time"), db2.diff("time")])
        assert not result.equals(naive)
        assert result.equals(expected)
