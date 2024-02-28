import numpy as np
import xarray as xr

import xdas
import xdas.signal as xp
from xdas.synthetics import generate

class TestSignal:
    def test_get_sample_spacing(self):
        s = (5.0 / 2) + 5.0 * np.arange(100)
        dt = np.timedelta64(8, "ms")
        t = np.datetime64(0, "s") + dt * np.arange(1000)
        da = xr.DataArray(np.ones((len(t), len(s))), {"time": t, "distance": s})
        assert xp.get_sampling_interval(da, "time") == 0.008
        assert xp.get_sampling_interval(da, "distance") == 5.0
        shape = (6000, 1000)
        resolution = (np.timedelta64(8, "ms"), 5.0)
        starttime = np.datetime64("2023-01-01T00:00:00")
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
        assert xp.get_sampling_interval(db, "time") == 0.008
        assert xp.get_sampling_interval(db, "distance") == 5.0

    def test_deterend(self):
        n = 100
        d = 5.0
        s = d * np.arange(n)
        da = xr.DataArray(np.arange(n), {"time": s})
        db = xdas.Database.from_xarray(da)
        da = xp.detrend(da)
        assert np.allclose(da, np.zeros(n))
        db = xp.detrend(db)
        assert np.allclose(db.values, np.zeros(n))

    def test_differentiate(self):
        n = 100
        d = 5.0
        s = (d / 2) + d * np.arange(n)
        da = xr.DataArray(np.ones(n), {"distance": s})
        db = xdas.Database.from_xarray(da)
        da = xp.differentiate(da, midpoints=True)
        assert np.allclose(da, np.zeros(n - 1))
        db = xp.differentiate(db, midpoints=True)
        assert np.allclose(db.values, np.zeros(n - 1))

    def test_integrate(self):
        n = 100
        d = 5.0
        s = (d / 2) + d * np.arange(n)
        da = xr.DataArray(np.ones(n), {"distance": s})
        db = xdas.Database.from_xarray(da)
        da = xp.integrate(da, midpoints=True)
        assert np.allclose(da, da["distance"])
        db = xp.integrate(db, midpoints=True)
        assert np.allclose(db.values, db["distance"].values)

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
        db = xdas.Database.from_xarray(da)
        da = xp.segment_mean_removal(da, limits)
        assert np.allclose(da, 0)
        db = xp.segment_mean_removal(db, limits)
        assert np.allclose(db.values, 0)

    def test_sliding_window_removal(self):
        n = 100
        d = 5.0
        s = (d / 2) + d * np.arange(n)
        s = np.linspace(0, 1000, n)
        data = np.ones(n)
        da = xr.DataArray(data, {"distance": s})
        db = xdas.Database.from_xarray(da)
        da = xp.sliding_mean_removal(da, 0.1 * n * d)
        assert np.allclose(da, 0)
        db = xp.sliding_mean_removal(db, 0.1 * n * d)
        assert np.allclose(db.values, 0)


    def test_medfilt(self):
        da = generate()
        dimensions = np.array(["distance"])
        kernel_length = [3]
        dictionary = dict(zip(dimensions, kernel_length))
        dimensions = np.array(["distance", "time"])
        kernel_length = [3, 1]
        dictionary_inv = dict(zip(dimensions, kernel_length))
        da_fil = xp.medfilt(da, dictionary)
        da_fil_inv = xp.medfilt(da, dictionary_inv)
        assert np.allclose(da_fil.values, da_fil_inv.values)

    def test_multithreaded_concatenate(self):
        arrays = [np.random.rand(100, 20) for _ in range(100)]
        expected = np.concatenate(arrays)
        result = xp.multithreaded_concatenate(arrays)
        assert np.array_equal(expected, result)
        expected = np.concatenate(arrays, axis=1)
        result = xp.multithreaded_concatenate(arrays, axis=1)
        assert np.array_equal(expected, result)
