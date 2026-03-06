import os
import tempfile

import numpy as np
import scipy.signal as sp
import xarray as xr

import xdas as xd
import xdas.signal as xs
from xdas.synthetics import wavelet_wavefronts


class TestSignal:
    def test_get_sample_spacing(self):
        shape = (6000, 1000)
        resolution = (np.timedelta64(8, "ms"), 5.0)
        starttime = np.datetime64("2023-01-01T00:00:00")
        da = xd.DataArray(
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
        assert xs.get_sampling_interval(da, "time") == 0.008
        assert xs.get_sampling_interval(da, "distance") == 5.0

    def test_deterend(self):
        n = 100
        d = 5.0
        s = d * np.arange(n)
        da = xr.DataArray(np.arange(n), {"time": s})
        da = xd.DataArray.from_xarray(da)
        da = xs.detrend(da)
        assert np.allclose(da.values, np.zeros(n))

    def test_differentiate(self):
        n = 100
        d = 5.0
        s = (d / 2) + d * np.arange(n)
        da = xr.DataArray(np.ones(n), {"distance": s})
        da = xd.DataArray.from_xarray(da)
        da = xs.differentiate(da, midpoints=True)
        assert np.allclose(da.values, np.zeros(n - 1))

    def test_integrate(self):
        n = 100
        d = 5.0
        s = (d / 2) + d * np.arange(n)
        da = xr.DataArray(np.ones(n), {"distance": s})
        da = xd.DataArray.from_xarray(da)
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
        da = xs.sliding_mean_removal(da, 0.1 * n * d)
        assert np.allclose(da.values, 0)

    def test_medfilt(self):
        da = wavelet_wavefronts()
        result1 = xs.medfilt(da, {"distance": 3})
        result2 = xs.medfilt(da, {"time": 1, "distance": 3})
        assert result1.equals(result2)
        da.data = np.zeros(da.shape)
        assert da.equals(xs.medfilt(da, {"time": 7, "distance": 3}))

    def test_hilbert(self):
        da = wavelet_wavefronts()
        result = xs.hilbert(da, dim="time")
        assert np.allclose(da.values, np.real(result.values))

    def test_resample(self):
        da = wavelet_wavefronts()
        result = xs.resample(da, 100, dim="time", window="hamming", domain="time")
        assert result.sizes["time"] == 100

    def test_resample_poly(self):
        da = wavelet_wavefronts()
        result = xs.resample_poly(da, 2, 5, dim="time")
        assert result.sizes["time"] == 120

    def test_lfilter(self):
        da = wavelet_wavefronts()
        b, a = sp.iirfilter(4, 0.5, btype="low")
        result1 = xs.lfilter(b, a, da, "time")
        result2, zf = xs.lfilter(b, a, da, "time", zi=...)
        assert result1.equals(result2)

    def test_filtfilt(self):
        da = wavelet_wavefronts()
        b, a = sp.iirfilter(2, 0.5, btype="low")
        xs.filtfilt(b, a, da, "time", padtype=None)

    def test_sosfilter(self):
        da = wavelet_wavefronts()
        sos = sp.iirfilter(4, 0.5, btype="low", output="sos")
        result1 = xs.sosfilt(sos, da, "time")
        result2, zf = xs.sosfilt(sos, da, "time", zi=...)
        assert result1.equals(result2)

    def test_sosfiltfilt(self):
        da = wavelet_wavefronts()
        sos = sp.iirfilter(2, 0.5, btype="low", output="sos")
        xs.sosfiltfilt(sos, da, "time", padtype=None)

    def test_filter(self):
        da = wavelet_wavefronts()
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

    def test_decimate_virtual_stack(self):
        da = wavelet_wavefronts()
        expected = xs.decimate(da, 5, dim="time")
        chunks = xd.split(da, 5, "time")
        with tempfile.TemporaryDirectory() as tmpdirname:
            for i, chunk in enumerate(chunks):
                chunk_path = os.path.join(tmpdirname, f"chunk_{i}.nc")
                chunk.to_netcdf(chunk_path)
            da_virtual = xd.open_mfdataarray(os.path.join(tmpdirname, "chunk_*.nc"))
            result = xs.decimate(da_virtual, 5, dim="time")
        assert result.equals(expected)


class TestSTFT:
    def test_compare_with_scipy(self):
        starttime = np.datetime64("2023-01-01T00:00:00")
        endtime = starttime + 9999 * np.timedelta64(10, "ms")
        da = xd.DataArray(
            data=np.random.rand(10000, 11),
            coords={
                "time": {"tie_indices": [0, 9999], "tie_values": [starttime, endtime]},
                "distance": {"tie_indices": [0, 10], "tie_values": [0.0, 1.0]},
            },
        )
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
                        fs=1 / xs.get_sampling_interval(da, "time"),
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
        time = np.arange(N) / float(fs)
        data = amp * np.sin(2 * np.pi * fc * time)
        da = xd.DataArray(
            data=data,
            coords={"time": time},
        )
        result = xs.stft(
            da, nperseg=1000, noverlap=500, window="hann", dim={"time": "frequency"}
        )
        idx = int(np.abs(np.square(result)).mean("time").argmax("frequency").values)
        assert result["frequency"][idx].values == fc

    def test_parrallel(self):
        starttime = np.datetime64("2023-01-01T00:00:00")
        endtime = starttime + 9999 * np.timedelta64(10, "ms")
        da = xd.DataArray(
            data=np.random.rand(10000, 11),
            coords={
                "time": {"tie_indices": [0, 9999], "tie_values": [starttime, endtime]},
                "distance": {"tie_indices": [0, 10], "tie_values": [0.0, 1.0]},
            },
        )
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
        starttime = np.datetime64("2023-01-01T00:00:00")
        endtime = starttime + 99 * np.timedelta64(10, "ms")
        da = xd.DataArray(
            data=np.random.rand(100, 1001),
            coords={
                "time": {"tie_indices": [0, 99], "tie_values": [starttime, endtime]},
                "distance": {"tie_indices": [0, 1000], "tie_values": [0.0, 10_000.0]},
                "channel": ("distance", np.arange(1001)),
            },
        )
        result = xs.stft(
            da,
            nperseg=100,
            noverlap=50,
            window="hamming",
            dim={"distance": "wavenumber"},
        )
        f, t, Zxx = sp.stft(
            da.values,
            fs=1 / xs.get_sampling_interval(da, "distance"),
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
