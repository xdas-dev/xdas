import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest

import xdas
from xdas.core import collects, splits
from xdas.synthetics import generate
from xdas.virtual import DataStack


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

    def test_open_mfdatacollection(self): ...  # TODO

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
        # concatenate two databases
        db1 = generate(starttime="2023-01-01T00:00:00")
        db2 = generate(starttime="2023-01-01T00:00:06")
        data = np.concatenate([db1.data, db2.data])
        coords = {
            "time": {
                "tie_indices": [0, db1.sizes["time"] + db2.sizes["time"] - 1],
                "tie_values": [db1["time"][0].values, db2["time"][-1].values],
            },
            "distance": db1["distance"],
        }
        expected = xdas.Database(data, coords)
        result = xdas.concatenate([db1, db2])
        assert result.equals(expected)
        # concatenate an empty databse
        result = xdas.concatenate([db1, db2.isel(time=slice(0, 0))])
        assert result.equals(db1)
        # concat of sources and stacks
        with TemporaryDirectory() as tmp_path:
            db1.to_netcdf(os.path.join(tmp_path, "db1.nc"))
            db2.to_netcdf(os.path.join(tmp_path, "db2.nc"))
            db1 = xdas.open_database(os.path.join(tmp_path, "db1.nc"))
            db2 = xdas.open_database(os.path.join(tmp_path, "db2.nc"))
            result = xdas.concatenate([db1, db2])
            assert isinstance(result.data, DataStack)
            assert result.equals(expected)
            db1.data = DataStack([db1.data])
            db2.data = DataStack([db2.data])
            result = xdas.concatenate([db1, db2])
            assert isinstance(result.data, DataStack)
            assert result.equals(expected)

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
        def roll(db, shift, dim="last"):
            axis = db.get_axis_num(dim)
            data = np.roll(db.values, shift, axis)
            return db.copy(data=data)

        roll_decorated = splits(collects(roll))

        db1 = generate(starttime="2023-01-01T00:00:00")
        db2 = generate(starttime="2023-01-01T00:00:10") + 1
        db = xdas.concatenate([db1, db2], dim="time")
        naive = roll(db, 1, dim="time")
        result = roll_decorated(db, 1, dim="time")
        expected = xdas.concatenate(
            [roll(db1, 1, dim="time"), roll(db2, 1, dim="time")]
        )
        assert not expected.equals(naive)
        assert not result.equals(naive)
        assert result.equals(expected)
