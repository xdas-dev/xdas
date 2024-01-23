import tempfile

import numpy as np
import pytest
import xarray as xr

import xdas


class TestAll:
    def test_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # generation
            shape = (6000, 1000)
            resolution = (np.timedelta64(10, "ms"), 5.0)
            starttimes = {
                "001": np.datetime64("2023-01-01T00:00:00"),
                "002": np.datetime64("2023-01-01T00:01:00"),
                "003": np.datetime64("2023-01-01T00:02:00"),
            }
            for name, starttime in starttimes.items():
                db = xdas.Database(
                    data=np.random.randn(*shape).astype("float32"),
                    coords={
                        "time": {
                            "tie_indices": [0, shape[0] - 1],
                            "tie_values": [
                                starttime,
                                starttime + resolution[0] * (shape[0] - 1),
                            ],
                        },
                        "distance": {
                            "tie_indices": [0, shape[1] - 1],
                            "tie_values": [0.0, resolution[1] * (shape[1] - 1)],
                        },
                    },
                )
                db.to_netcdf(f"{tmpdir}/{name}.nc")

            # testing
            _db = xdas.open_database(f"{tmpdir}/002.nc")
            _db_loaded = _db.load()
            _da = _db.to_xarray()
            datasource = _db.data
            assert np.allclose(np.asarray(datasource[0]), _da.values[0])
            assert np.allclose(np.asarray(datasource[0][1]), _da.values[0][1])
            assert np.allclose(np.asarray(datasource[:, 0][1]), _da.values[:, 0][1])
            assert np.allclose(np.asarray(datasource[:, 0][1]), _da.values[:, 0][1])
            assert np.allclose(np.asarray(datasource[10:][1]), _da.values[10:][1])
            with pytest.raises(IndexError):
                datasource[1, 2, 3]
            assert np.allclose(np.asarray(datasource[10:][1]), _da.values[10:][1])
            assert np.array_equal(_da.data, _db_loaded.data)
            db = _db.sel(
                time=slice("2023-01-01T00:01:20", None),
                distance=slice(1000, None),
            )
            db = db.isel(
                time=slice(None, 1000),
                distance=slice(None, 200),
            )
            da1 = db.to_xarray()
            da = _da.sel(
                time=slice("2023-01-01T00:01:20", None),
                distance=slice(1000, None),
            )
            da = da.isel(
                time=slice(None, 1000),
                distance=slice(None, 200),
            )
            da2 = da
            xr.testing.assert_equal(da1, da2)
