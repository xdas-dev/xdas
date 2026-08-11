import numpy as np
import pytest

import xdas as xd
import xdas.fft as xfft


class TestRFFT:
    def test_with_non_dimensional(self):
        da = xd.testing.dummy()
        da["latitude"] = ("distance", np.arange(da.sizes["distance"]))
        xfft.rfft(da)


class TestInverseTransforms:
    def test_standard(self):
        expected = xd.testing.dummy()
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
        expected = xd.testing.dummy()
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
        expected = xd.testing.dummy()
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


class TestDefaultDim:
    """The default `dim` maps the last dimension; equivalent to naming it."""

    def test_fft(self):
        da = xd.testing.dummy()
        assert xfft.fft(da).equals(xfft.fft(da, dim={"distance": "spectrum"}))

    def test_rfft(self):
        da = xd.testing.dummy()
        assert xfft.rfft(da).equals(xfft.rfft(da, dim={"distance": "spectrum"}))

    def test_ifft(self):
        da = xd.testing.dummy()
        assert xfft.ifft(da).equals(xfft.ifft(da, dim={"distance": "signal"}))

    def test_irfft(self):
        da = xd.testing.dummy()
        assert xfft.irfft(da).equals(xfft.irfft(da, dim={"distance": "signal"}))


class TestChunkGuard:
    """FFTs need the whole record along their dimension: they refuse chunked
    execution along it but stay usable in pipelines chunked along another."""

    def test_chunked_along_transform_dim_raises(self):
        da = xd.testing.dummy()
        chunk = da.isel(time=slice(0, 50))
        for func in (xfft.fft, xfft.rfft, xfft.ifft, xfft.irfft):
            atom = func(..., dim={"time": "frequency"})
            with pytest.raises(ValueError, match="whole record"):
                atom(chunk, chunk_dim="time")

    def test_default_dim_is_conservative(self):
        da = xd.testing.dummy()
        atom = xfft.fft(...)
        with pytest.raises(ValueError, match="whole record"):
            atom(da.isel(time=slice(0, 50)), chunk_dim="time")

    def test_chunked_along_other_dim_commutes(self):
        da = xd.testing.dummy()
        atom = xfft.rfft(..., dim={"distance": "wavenumber"})
        chunks = [atom(chunk, chunk_dim="time") for chunk in xd.split(da, 4, "time")]
        result = xd.concat(chunks, "time")
        expected = xfft.rfft(da, dim={"distance": "wavenumber"})
        assert result.equals(expected)
