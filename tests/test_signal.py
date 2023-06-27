import numpy as np
import xarray as xr

import xdas.signal


class TestSignal:
    def test_integrate(self):
        n = 100
        d = 5.0
        s = (d / 2) + d * np.arange(n)
        da = xr.DataArray(np.ones(n), {"distance": s})
        da = xdas.signal.integrate(da, midpoints=True)
        assert np.allclose(da, da["distance"])

    def test_segment_mean(self):
        n = 100
        d = 5.0
        s = (d / 2) + d * np.arange(n)
        limits = [0, 0.3 * n * d, n * d]
        s = np.linspace(0, 1000, n)
        data = np.zeros(n)
        da = xr.DataArray(data, {"distance": s})
        da.loc[{"distance": slice(limits[0], limits[1])}] = 1.0
        da.loc[{"distance": slice(limits[1], limits[2])}] = 2.0
        da = xdas.signal.segment_mean(da, limits)
        assert np.allclose(da, 0)
