import numpy as np
import xarray as xr

import xdas.signal


class TestSigal:
    def test_integrate(self):
        n = 100
        d = 5.0
        s = (d / 2) + d * np.arange(n)
        da = xr.DataArray(data=np.ones(n), coords={"distance": s})
        da = xdas.signal.integrate(da, midpoints=True)
        assert np.allclose(da, da["distance"])
