import numpy as np

import xdas as xd
import xdas.fft as xfft


class TestRFFT:
    def test_with_non_dimensional(self):
        da = xd.synthetics.wavelet_wavefronts()
        da["latitude"] = ("distance", np.arange(da.sizes["distance"]))
        xfft.rfft(da)
