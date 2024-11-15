import numpy as np
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.signal import get_window

from . import DataArray, get_sampling_interval


def stft(
    da,
    window="hann",
    nperseg=256,
    noverlap=None,
    nfft=None,
    return_onesided=True,
    dim={"last": "sprectrum"},
    scaling="spectrum",
    # parallel=None,
):
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = nperseg
    win = get_window(window, nperseg)
    input_dim, output_dim = next(iter(dim.items()))
    axis = da.get_axis_num(input_dim)
    dt = get_sampling_interval(da, input_dim)
    if scaling == "density":
        scale = 1.0 / ((win * win).sum() / dt)
    elif scaling == "spectrum":
        scale = 1.0 / win.sum() ** 2
    else:
        raise ValueError("Scaling must be 'density' or 'spectrum'")
    scale = np.sqrt(scale)
    if return_onesided:
        freqs = rfftfreq(nfft, dt)
    else:
        freqs = fftfreq(nfft, dt)
    freqs = {"tie_indices": [0, nfft - 1], "tie_values": [freqs[0], freqs[-1]]}

    def func(x):
        if nperseg == 1 and noverlap == 0:
            result = x[..., np.newaxis]
        else:
            step = nperseg - noverlap
            result = np.lib.stride_tricks.sliding_window_view(
                x, window_shape=nperseg, axis=axis, writeable=True
            )
            slc = [slice(None)] * result.ndim
            slc[axis] = slice(None, None, step)
            result = result[tuple(slc)]
        result = win * result
        if return_onesided:
            result = rfft(result, n=nfft)
        else:
            result = fft(result, n=nfft)
        result *= scale
        return result

    data = func(da.values)

    dt = get_sampling_interval(da, input_dim, cast=False)
    t0 = da.coords[input_dim].values[0]
    starttime = t0 + (nperseg / 2) * dt
    endtime = t0 + (da.shape[-1] - nperseg / 2) * dt
    time = {"tie_indices": [0, da.shape[-1] - 1], "tie_values": [starttime, endtime]}

    coords = da.coords.copy()
    coords[input_dim] = time
    coords[output_dim] = freqs

    result = DataArray(data, coords)

    dims = dim + (output_dim,)
    return result.transpose(*dims)
