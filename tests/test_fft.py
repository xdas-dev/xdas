import numpy as np

import xdas as xd
import xdas.fft as xfft


class TestRFFT:
    def test_with_non_dimensional(self):
        da = xd.synthetics.wavelet_wavefronts()
        da["latitude"] = ("distance", np.arange(da.sizes["distance"]))
        xfft.rfft(da)


class TestInverseTransforms:
    def test_standard(self):
        expected = xd.synthetics.wavelet_wavefronts()
        result = xfft.ifft(
            xfft.fft(expected, dim={"time": "frequency"}),
            dim={"frequency": "time"},
        )
        assert np.allclose(np.real(result).values, expected.values)
        assert np.allclose(np.imag(result).values, 0)
        for name in result.coords:
            if name == "time":
                ref = expected["time"].values
                ref = (ref - ref[0]) / np.timedelta64(1, "s")
                ref += result["time"][0].values
                assert np.allclose(result["time"].values, ref)
            else:
                assert result[name].equals(expected[name])

    def test_real(self):
        expected = xd.synthetics.wavelet_wavefronts()
        result = xfft.irfft(
            xfft.rfft(expected, dim={"time": "frequency"}),
            expected.sizes["time"],
            dim={"frequency": "time"},
        )
        assert np.allclose(result.values, expected.values)
        for name in result.coords:
            if name == "time":
                ref = expected["time"].values
                ref = (ref - ref[0]) / np.timedelta64(1, "s")
                ref += result["time"][0].values
                assert np.allclose(result["time"].values, ref)
            else:
                assert result[name].equals(expected[name])

    def test_real_default_n(self):
        expected = xd.synthetics.wavelet_wavefronts()
        expected = expected.isel(time=slice(0, expected.sizes["time"] // 2 * 2))
        result = xfft.irfft(
            xfft.rfft(expected, dim={"time": "frequency"}),
            dim={"frequency": "time"},
        )
        assert np.allclose(result.values, expected.values)
        for name in result.coords:
            if name == "time":
                ref = expected["time"].values
                ref = (ref - ref[0]) / np.timedelta64(1, "s")
                ref += result["time"][0].values
                assert np.allclose(result["time"].values, ref)
            else:
                assert result[name].equals(expected[name])
