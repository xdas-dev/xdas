import numpy as np
import xarray as xr

import xdas

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
            "time": xdas.Coordinate(
                tie_indices=[0, shape[0] - 1],
                tie_values=[starttime, starttime + resolution[0] * (shape[0] - 1)],
            ),
            "distance": xdas.Coordinate(
                tie_indices=[0, shape[1] - 1],
                tie_values=[0.0, resolution[1] * (shape[1] - 1)],
            ),
        },
    )
    db.to_netcdf(f"{name}.nc")


class TestAll:
    def test_all(self):
        _db = xdas.open_database("002.nc")
        _da = _db.to_xarray()
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
