import numpy as np
import xarray as xr

import xdas
import xdas.signal as xp


class TestSignal:
    def test_get_sample_spacing(self):
        s = (5.0 / 2) + 5.0 * np.arange(100)
        dt = np.timedelta64(8, "ms")
        t = np.datetime64(0, "s") + dt * np.arange(1000)
        da = xr.DataArray(np.ones((len(t), len(s))), {"time": t, "distance": s})
        assert xp.get_sample_spacing(da, "time") == 0.008
        assert xp.get_sample_spacing(da, "distance") == 5.0
        shape = (6000, 1000)
        resolution = (np.timedelta64(8, "ms"), 5.0)
        starttime = np.datetime64("2023-01-01T00:00:00")
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
        assert xp.get_sample_spacing(db, "time") == 0.008
        assert xp.get_sample_spacing(db, "distance") == 5.0

    def test_integrate(self):
        n = 100
        d = 5.0
        s = (d / 2) + d * np.arange(n)
        da = xr.DataArray(np.ones(n), {"distance": s})
        da = xp.integrate(da, midpoints=True)
        assert np.allclose(da, da["distance"])

    def test_segment_mean_removal(self):
        n = 100
        d = 5.0
        s = (d / 2) + d * np.arange(n)
        limits = [0, 0.3 * n * d, n * d]
        s = np.linspace(0, 1000, n)
        data = np.zeros(n)
        da = xr.DataArray(data, {"distance": s})
        da.loc[{"distance": slice(limits[0], limits[1])}] = 1.0
        da.loc[{"distance": slice(limits[1], limits[2])}] = 2.0
        da = xp.segment_mean_removal(da, limits)
        assert np.allclose(da, 0)

    def test_sliding_window_removal(self):
        n = 100
        d = 5.0
        s = (d / 2) + d * np.arange(n)
        s = np.linspace(0, 1000, n)
        data = np.ones(n)
        da = xr.DataArray(data, {"distance": s})
        da = xp.sliding_mean_removal(da, 0.1 * n * d)
        assert np.allclose(da, 0)
