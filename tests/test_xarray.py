import numpy as np

import xdas.core.methods as xp
from xdas.core.dataarray import DataArray
from xdas.synthetics import wavelet_wavefronts


class TestXarray:
    def test_returns_dataarray(self):
        da = wavelet_wavefronts()
        for name, func in xp.HANDLED_METHODS.items():
            if callable(func):
                if name in [
                    "percentile",
                    "quantile",
                ]:
                    result = func(da, 0.5)
                    assert isinstance(result, DataArray)
                    result = getattr(da, name)(0.5)
                    assert isinstance(result, DataArray)
                elif name == "diff":
                    result = func(da, "time")
                    assert isinstance(result, DataArray)
                    result = getattr(da, name)("time")
                    assert isinstance(result, DataArray)
                else:
                    result = func(da)
                    assert isinstance(result, DataArray)
                    result = getattr(da, name)()
                    assert isinstance(result, DataArray)

    def test_mean(self):
        da = wavelet_wavefronts()
        result = xp.mean(da, "time")
        result_method = da.mean("time")
        expected = np.mean(da, 0)
        assert result.equals(expected)
        assert result_method.equals(expected)
