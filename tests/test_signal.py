import numpy as np
import pytest
import scipy.signal as sp
import xarray as xr

import xdas as xd
import xdas.signal as xs


class TestSignal:
    def test_get_sample_spacing(self):
        da = xd.testing.dummy(shape=(6000, 1000), step=(0.008, 5.0), dtype="float32")
        assert da.coords["time"].get_sampling_interval() == 0.008
        assert da.coords["distance"].get_sampling_interval() == 5.0

    def test_deterend(self):
        # dummy data is a linear ramp, so detrending must flatten it to zero
        da = xd.testing.dummy(dims=("time",), shape=(100,), step=5.0, datetime=False)
        da = xs.detrend(da)
        assert np.allclose(da.values, np.zeros(100))

    def test_differentiate(self):
        n = 100
        d = 5.0
        s = (d / 2) + d * np.arange(n)
        da = xr.DataArray(np.ones(n), {"distance": s})
        da = xd.DataArray.from_xarray(da)
        da["distance"] = da["distance"].to_regular()
        da = xs.differentiate(da, midpoints=True)
        assert np.allclose(da.values, np.zeros(n - 1))

    def test_integrate(self):
        n = 100
        d = 5.0
        s = (d / 2) + d * np.arange(n)
        da = xr.DataArray(np.ones(n), {"distance": s})
        da = xd.DataArray.from_xarray(da)
        da["distance"] = da["distance"].to_regular()
        da = xs.integrate(da, midpoints=True)
        assert np.allclose(da.values, da["distance"].values)

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
        da = xd.DataArray.from_xarray(da)
        da = xs.segment_mean_removal(da, limits)
        assert np.allclose(da.values, 0)

    def test_sliding_window_removal(self):
        n = 100
        d = 5.0
        s = (d / 2) + d * np.arange(n)
        s = np.linspace(0, 1000, n)
        data = np.ones(n)
        da = xr.DataArray(data, {"distance": s})
        da = xd.DataArray.from_xarray(da)
        da["distance"] = da["distance"].to_regular()
        da = xs.sliding_mean_removal(da, 0.1 * n * d)
        assert np.allclose(da.values, 0)

    def test_medfilt(self):
        da = xd.testing.dummy()
        result1 = xs.medfilt(da, {"distance": 3})
        result2 = xs.medfilt(da, {"time": 1, "distance": 3})
        assert result1.equals(result2)
        da.data = np.zeros(da.shape)
        assert da.equals(xs.medfilt(da, {"time": 7, "distance": 3}))

    def test_hilbert(self):
        da = xd.testing.dummy()
        result = xs.hilbert(da, dim="time")
        assert np.allclose(da.values, np.real(result.values))

    def test_resample(self):
        da = xd.testing.dummy()
        result = xs.resample(da, 50, dim="time", window="hamming", domain="time")
        assert result.sizes["time"] == 50

    def test_resample_poly(self):
        da = xd.testing.dummy()
        result = xs.resample_poly(da, 2, 5, dim="time")
        assert result.sizes["time"] == 40

    def test_lfilter(self):
        da = xd.testing.dummy()
        b, a = sp.iirfilter(4, 0.5, btype="low")
        result1 = xs.lfilter(b, a, da, "time")
        result2, _zf = xs.lfilter(b, a, da, "time", zi=...)
        assert result1.equals(result2)

    def test_filtfilt(self):
        da = xd.testing.dummy()
        b, a = sp.iirfilter(2, 0.5, btype="low")
        xs.filtfilt(b, a, da, "time", padtype=None)

    def test_sosfilter(self):
        da = xd.testing.dummy()
        sos = sp.iirfilter(4, 0.5, btype="low", output="sos")
        result1 = xs.sosfilt(sos, da, "time")
        result2, _zf = xs.sosfilt(sos, da, "time", zi=...)
        assert result1.equals(result2)

    def test_sosfiltfilt(self):
        da = xd.testing.dummy()
        sos = sp.iirfilter(2, 0.5, btype="low", output="sos")
        xs.sosfiltfilt(sos, da, "time", padtype=None)

    def test_filter(self):
        da = xd.testing.dummy()
        axis = da.get_axis_num("time")
        fs = 1 / xd.get_sampling_interval(da, "time")
        sos = sp.butter(
            4,
            [5, 10],
            "band",
            output="sos",
            fs=fs,
        )
        data = sp.sosfilt(sos, da.values, axis=axis)
        expected = da.copy(data=data)
        result = xs.filter(
            da,
            [5, 10],
            btype="band",
            corners=4,
            zerophase=False,
            dim="time",
            parallel=False,
        )
        assert result.equals(expected)
        data = sp.sosfiltfilt(sos, da.values, axis=axis)
        expected = da.copy(data=data)
        result = xs.filter(
            da,
            [5, 10],
            btype="band",
            corners=4,
            zerophase=True,
            dim="time",
            parallel=False,
        )
        assert result.equals(expected)

    def test_decimate_virtual_stack(self, tmp_path):
        da = xd.testing.dummy()
        expected = xs.decimate(da, 5, dim="time")
        chunks = xd.split(da, 5, "time")
        for i, chunk in enumerate(chunks):
            chunk_path = tmp_path / f"chunk_{i}.nc"
            chunk.to_netcdf(chunk_path)
        da_virtual = xd.open(tmp_path / "chunk_*.nc")
        result = xs.decimate(da_virtual, 5, dim="time")
        assert result.equals(expected)


class TestSTFT:
    def test_compare_with_scipy(self):
        da = xd.testing.dummy(shape=(10000, 11), step=(0.01, 0.1))
        for scaling in ["spectrum", "psd"]:
            for return_onesided in [True, False]:
                for nfft in [None, 128]:
                    result = xs.stft(
                        da,
                        window="hamming",
                        nperseg=100,
                        noverlap=50,
                        nfft=nfft,
                        return_onesided=return_onesided,
                        dim={"time": "frequency"},
                        scaling=scaling,
                    )
                    f, t, Zxx = sp.stft(
                        da.values,
                        fs=1 / da.coords["time"].get_sampling_interval(),
                        window="hamming",
                        nperseg=100,
                        noverlap=50,
                        nfft=nfft,
                        return_onesided=return_onesided,
                        boundary=None,
                        padded=False,
                        axis=0,
                        scaling=scaling,
                    )
                    if return_onesided:
                        assert np.allclose(result.values, np.transpose(Zxx, (2, 1, 0)))
                    else:
                        assert np.allclose(
                            result.values,
                            np.fft.fftshift(np.transpose(Zxx, (2, 1, 0)), axes=-1),
                        )
                    assert np.allclose(result["frequency"].values, np.sort(f))
                    assert np.allclose(
                        (result["time"].values - da["time"][0].values)
                        / np.timedelta64(1, "s"),
                        t,
                    )
                    assert result["distance"].equals(da["distance"])

    def test_retrieve_frequency_peak(self):
        fs = 10e3
        N = 1e5
        fc = 3e3
        amp = 2 * np.sqrt(2)
        da = xd.testing.dummy(
            dims=("time",), shape=(int(N),), step=1 / fs, datetime=False
        )
        da.data = amp * np.sin(2 * np.pi * fc * da["time"].values)
        result = xs.stft(
            da, nperseg=1000, noverlap=500, window="hann", dim={"time": "frequency"}
        )
        idx = int(np.abs(np.square(result)).mean("time").argmax("frequency").values)
        assert result["frequency"][idx].values == fc

    def test_parrallel(self):
        da = xd.testing.dummy(shape=(10000, 11), step=(0.01, 0.1))
        serial = xs.stft(
            da,
            nperseg=100,
            noverlap=50,
            window="hamming",
            dim={"time": "frequency"},
            parallel=False,
        )
        parallel = xs.stft(
            da,
            nperseg=100,
            noverlap=50,
            window="hamming",
            dim={"time": "frequency"},
            parallel=True,
        )
        assert serial.equals(parallel)

    def test_last_dimension_with_non_dimensional_coordinates(self):
        da = xd.testing.dummy(shape=(100, 1001))
        da["channel"] = ("distance", np.arange(1001))
        result = xs.stft(
            da,
            nperseg=100,
            noverlap=50,
            window="hamming",
            dim={"distance": "wavenumber"},
        )
        f, t, Zxx = sp.stft(
            da.values,
            fs=1 / da.coords["distance"].get_sampling_interval(),
            window="hamming",
            nperseg=100,
            noverlap=50,
            boundary=None,
            padded=False,
            axis=1,
        )
        assert np.allclose(result.values, np.transpose(Zxx, (0, 2, 1)))
        assert result["time"].equals(da["time"])
        assert np.allclose(result["distance"].values, t)
        assert np.allclose(result["wavenumber"].values, np.sort(f))
        assert "channel" not in result.coords  # TODO: keep non-dimensional coordinates


class TestSignalMissingBranches:
    def test_integrate_no_midpoints(self):
        da = xd.testing.dummy()
        result = xs.integrate(da, midpoints=False)
        assert result.shape == da.shape

    def test_differentiate_no_midpoints(self):
        da = xd.testing.dummy()
        result = xs.differentiate(da, midpoints=False)
        assert result.sizes["distance"] == da.sizes["distance"] - 1

    def test_sliding_mean_removal_even_window(self):
        # When wlen/d gives an even n, sliding_mean_removal increments n by 1.
        da = xd.testing.dummy()
        d = da.coords["time"].get_sampling_interval()
        # Make wlen exactly twice d so n=2 (even) → becomes 3
        result = xs.sliding_mean_removal(da, wlen=2 * d)
        assert result.shape == da.shape

    def test_medfilt_invalid_dim(self):
        da = xd.testing.dummy()
        with pytest.raises(ValueError, match="dims provided not in dataarray"):
            xs.medfilt(da, {"nonexistent_dim": 3})

    def test_stft_default_noverlap(self):
        da = xd.testing.dummy()
        result = xs.stft(da, nperseg=16, dim={"time": "frequency"})
        assert "frequency" in result.dims

    def test_stft_invalid_scaling(self):
        da = xd.testing.dummy()
        with pytest.raises(ValueError, match="Scaling must be"):
            xs.stft(da, nperseg=16, scaling="invalid", dim={"time": "frequency"})

    def test_stft_nperseg_one(self):
        # nperseg=1, noverlap=0 triggers the stride_tricks bypass branch
        da = xd.testing.dummy()
        # nfft=2 avoids single-element frequency axis (which would make tie_indices=[0,0])
        result = xs.stft(da, nperseg=1, noverlap=0, nfft=2, dim={"time": "frequency"})
        assert "frequency" in result.dims

    def test_stft_default_dim(self):
        # the default maps the last dimension; "first"/"last" aliases must resolve
        da = xd.testing.dummy()
        expected = xs.stft(da, nperseg=8, dim={"distance": "sprectrum"})
        assert xs.stft(da, nperseg=8).equals(expected)
        assert xs.stft(da, nperseg=8, dim={"last": "sprectrum"}).equals(expected)


class TestFftMissingBranches:
    def test_fft_explicit_n(self):
        import xdas.fft as xfft

        da = xd.testing.dummy().isel(distance=0)
        n = da.sizes["time"] // 2
        result = xfft.fft(da, n=n, dim={"time": "frequency"})
        assert result.sizes["frequency"] == n

    def test_rfft_explicit_n(self):
        import xdas.fft as xfft

        da = xd.testing.dummy().isel(distance=0)
        n = da.sizes["time"]
        result = xfft.rfft(da, n=n, dim={"time": "frequency"})
        assert "frequency" in result.dims

    def test_rfft_single_frequency(self):
        import xdas.fft as xfft

        da = xd.testing.dummy().isel(distance=0)
        result = xfft.rfft(da, n=1, dim={"time": "frequency"})
        assert result.sizes["frequency"] == 1

    def test_ifft_explicit_n(self):
        import xdas.fft as xfft

        da = xd.testing.dummy().isel(distance=0)
        spectrum = xfft.fft(da, dim={"time": "frequency"})
        n = da.sizes["time"]
        result = xfft.ifft(spectrum, n=n, dim={"frequency": "time"})
        assert result.sizes["time"] == n


class TestResampleTolerance:
    """The resamplers derive a new rate; the declared jitter must survive it."""

    @staticmethod
    def regular():
        da = xd.testing.dummy(shape=(120, 3))
        da["time"] = da["time"].to_regular(
            da["time"].sampling_interval, np.timedelta64(1, "s")
        )
        return da

    def test_resample_poly_carries_declared_tolerance(self):
        da = self.regular()
        result = xs.resample_poly(da, 1, 2, dim="time")
        assert result["time"].isregular()
        assert result["time"].tolerance >= da["time"].tolerance

    def test_resample_poly_declares_representation_error(self):
        # 1/3 of a 10 ms step is not representable in whole nanoseconds, so the
        # truncation must be declared as jitter on top of the inherited bound.
        da = self.regular()
        delta = da["time"].sampling_interval
        result = xs.resample_poly(da, 3, 1, dim="time")
        step = result["time"].sampling_interval
        assert result["time"].tolerance == da["time"].tolerance + np.abs(
            delta - step * 3
        )

    def test_resample_carries_declared_tolerance(self):
        da = self.regular()
        result = xs.resample(da, da.sizes["time"] // 2, dim="time")
        assert result["time"].isregular()
        assert result["time"].tolerance == da["time"].tolerance

    def test_resample_poly_on_irregular_coordinate(self):
        da = xd.testing.dummy(shape=(120, 3))
        da["time"] = xd.Coordinate(
            {
                "tie_indices": da["time"].tie_indices,
                "tie_values": da["time"].tie_values,
            },
            "time",
        )
        result = xs.resample_poly(da, 1, 2, dim="time")
        assert result.sizes["time"] == 60

    def test_resample_on_irregular_coordinate(self):
        da = xd.testing.dummy(shape=(120, 3))
        da["time"] = xd.Coordinate(
            {
                "tie_indices": da["time"].tie_indices,
                "tie_values": da["time"].tie_values,
            },
            "time",
        )
        result = xs.resample(da, 60, dim="time")
        assert result.sizes["time"] == 60

    @pytest.mark.parametrize("ctype", ["sampled", "dense"])
    def test_resamplers_on_non_interpolated_coordinates(self, ctype):
        # Only interpolated coordinates declare a tolerance; the others must
        # still resample without one.
        da = xd.testing.dummy(shape=(120, 3), ctype=ctype)
        assert xs.resample_poly(da, 1, 2, dim="time").sizes["time"] == 60
        assert xs.resample(da, 60, dim="time").sizes["time"] == 60
