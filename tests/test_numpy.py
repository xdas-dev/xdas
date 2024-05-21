import numpy as np
import pytest

from xdas.core.dataarray import HANDLED_NUMPY_FUNCTIONS, DataArray
from xdas.synthetics import wavelet_wavefronts


class TestUfuncs:
    def test_unitary_operators(self):
        da = wavelet_wavefronts()
        result = np.abs(da)
        expected = da.copy(data=np.abs(da.data))
        da_out = da.copy()
        np.abs(da, out=da_out)
        da_where = da.copy()
        np.abs(da, out=da_where, where=da.copy(data=da.data > 0))
        assert result.equals(expected)
        assert da_out.equals(expected)
        assert da_where.equals(da)

    def test_binary_operators(self):
        da1 = wavelet_wavefronts()
        da2 = wavelet_wavefronts()
        result = np.add(da1, da2)
        expected = da1.copy(data=da1.data + da2.data)
        da_out = da1.copy()
        np.add(da1, da2, out=da_out)
        da_where = da1.copy()
        np.abs(da1, out=da_where, where=da1.copy(data=np.zeros(da1.shape, "bool")))
        assert result.equals(expected)
        assert da_out.equals(expected)
        assert da_where.equals(da1)
        with pytest.raises(ValueError):
            np.add(da1, da2[1:])

    def test_multiple_outputs(self):
        da = wavelet_wavefronts()
        result1, result2 = np.divmod(da, da)
        expected1 = da.copy(data=np.ones(da.shape))
        expected2 = da.copy(data=np.zeros(da.shape))
        assert result1.equals(expected1)
        assert result2.equals(expected2)
        with pytest.raises(ValueError):
            np.add(da, da[1:])


class TestFunc:
    def test_returns_dataarray(self):
        da = wavelet_wavefronts()
        for numpy_function in HANDLED_NUMPY_FUNCTIONS:
            if numpy_function == np.clip:
                result = numpy_function(da, -1, 1)
                assert isinstance(result, DataArray)
            elif numpy_function in [np.diff, np.ediff1d, np.trapz]:
                result = numpy_function(da)
                assert isinstance(result, np.ndarray)
            elif numpy_function in [
                np.percentile,
                np.nanpercentile,
                np.quantile,
                np.nanquantile,
            ]:
                result = numpy_function(da, 0.5)
                assert isinstance(result, DataArray)
            else:
                result = numpy_function(da)
                assert isinstance(result, DataArray)

    def test_reduce(self):
        da = wavelet_wavefronts()
        result = np.sum(da)
        assert result.shape == ()
        result = np.sum(da, axis=0)
        assert result.dims == ("distance",)
        assert result.coords["distance"].equals(da.coords["distance"])
        result = np.sum(da, axis=1)
        assert result.dims == ("time",)
        assert result.coords["time"].equals(da.coords["time"])
        with pytest.raises(np.AxisError):
            np.sum(da, axis=2)

    def test_out(self):
        da = wavelet_wavefronts()
        out = da.copy()
        np.cumsum(da, axis=-1, out=out)
        assert not out.equals(da)
