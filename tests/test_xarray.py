import numpy as np

import xdas as xd
import xdas.core.methods as xm
from xdas.synthetics import wavelet_wavefronts


class TestXarray:
    def test_returns_dataarray(self):
        da = wavelet_wavefronts()
        for name, func in xm.HANDLED_METHODS.items():
            if callable(func):
                if name in [
                    "percentile",
                    "quantile",
                ]:
                    result = func(da, 0.5)
                    assert isinstance(result, xd.DataArray)
                    result = getattr(da, name)(0.5)
                    assert isinstance(result, xd.DataArray)
                elif name == "diff":
                    result = func(da, "time")
                    assert isinstance(result, xd.DataArray)
                    result = getattr(da, name)("time")
                    assert isinstance(result, xd.DataArray)
                else:
                    result = func(da)
                    assert isinstance(result, xd.DataArray)
                    result = getattr(da, name)()
                    assert isinstance(result, xd.DataArray)

    def test_mean(self):
        da = wavelet_wavefronts()
        result = xm.mean(da, "time")
        result_method = da.mean("time")
        expected = np.mean(da, 0)
        assert result.equals(expected)
        assert result_method.equals(expected)
