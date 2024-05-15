import numpy as np
import scipy.signal as sp
import xarray as xr

import xdas
import xdas.signal as xp
from xdas.synthetics import wavelet_wavefronts


class TestSignal:
    def test_get_sample_spacing(self):
        shape = (6000, 1000)
        resolution = (np.timedelta64(8, "ms"), 5.0)
        starttime = np.datetime64("2023-01-01T00:00:00")
        da = xdas.DataArray(
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
        assert xp.get_sampling_interval(da, "time") == 0.008
        assert xp.get_sampling_interval(da, "distance") == 5.0

    def test_deterend(self):
        n = 100
        d = 5.0
        s = d * np.arange(n)
        da = xr.DataArray(np.arange(n), {"time": s})
        da = xdas.DataArray.from_xarray(da)
        da = xp.detrend(da)
        assert np.allclose(da.values, np.zeros(n))

    def test_differentiate(self):
        n = 100
        d = 5.0
        s = (d / 2) + d * np.arange(n)
        da = xr.DataArray(np.ones(n), {"distance": s})
        da = xdas.DataArray.from_xarray(da)
        da = xp.differentiate(da, midpoints=True)
        assert np.allclose(da.values, np.zeros(n - 1))

    def test_integrate(self):
        n = 100
        d = 5.0
        s = (d / 2) + d * np.arange(n)
        da = xr.DataArray(np.ones(n), {"distance": s})
        da = xdas.DataArray.from_xarray(da)
        da = xp.integrate(da, midpoints=True)
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
        da = xdas.DataArray.from_xarray(da)
        da = xp.segment_mean_removal(da, limits)
        assert np.allclose(da.values, 0)

    def test_sliding_window_removal(self):
        n = 100
        d = 5.0
        s = (d / 2) + d * np.arange(n)
        s = np.linspace(0, 1000, n)
        data = np.ones(n)
        da = xr.DataArray(data, {"distance": s})
        da = xdas.DataArray.from_xarray(da)
        da = xp.sliding_mean_removal(da, 0.1 * n * d)
        assert np.allclose(da.values, 0)

    def test_medfilt(self):
        da = wavelet_wavefronts()
        result1 = xp.medfilt(da, {"distance": 3})
        result2 = xp.medfilt(da, {"time": 1, "distance": 3})
        assert result1.equals(result2)
        da.data = np.zeros(da.shape)
        assert da.equals(xp.medfilt(da, {"time": 7, "distance": 3}))

    def test_hilbert(self):
        da = wavelet_wavefronts()
        result = xp.hilbert(da, dim="time")
        assert np.allclose(da.values, np.real(result.values))

    def test_resample(self):
        da = wavelet_wavefronts()
        result = xp.resample(da, 100, dim="time", window="hamming", domain="time")
        assert result.sizes["time"] == 100

    def test_resample_poly(self):
        da = wavelet_wavefronts()
        result = xp.resample_poly(da, 2, 5, dim="time")
        assert result.sizes["time"] == 120

    def test_lfilter(self):
        da = wavelet_wavefronts()
        b, a = sp.iirfilter(4, 0.5, btype="low")
        result1 = xp.lfilter(b, a, da, "time")
        result2, zf = xp.lfilter(b, a, da, "time", zi=...)
        assert result1.equals(result2)

    def test_filtfilt(self):
        da = wavelet_wavefronts()
        b, a = sp.iirfilter(2, 0.5, btype="low")
        xp.filtfilt(b, a, da, "time", padtype=None)

    def test_sosfilter(self):
        da = wavelet_wavefronts()
        sos = sp.iirfilter(4, 0.5, btype="low", output="sos")
        result1 = xp.sosfilt(sos, da, "time")
        result2, zf = xp.sosfilt(sos, da, "time", zi=...)
        assert result1.equals(result2)

    def test_sosfiltfilt(self):
        da = wavelet_wavefronts()
        sos = sp.iirfilter(2, 0.5, btype="low", output="sos")
        xp.sosfiltfilt(sos, da, "time", padtype=None)
