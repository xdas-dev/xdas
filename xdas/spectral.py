import numpy as np
from scipy.fft import fft, fftfreq, fftshift, rfft, rfftfreq
from scipy.signal import get_window

from .core.coordinates import get_sampling_interval
from .core.dataarray import DataArray
from .parallel import parallelize


def stft(
    da,
    window="hann",
    nperseg=256,
    noverlap=None,
    nfft=None,
    return_onesided=True,
    dim={"last": "sprectrum"},
    scaling="spectrum",
    parallel=None,
):
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = nperseg
    win = get_window(window, nperseg)
    input_dim, output_dim = next(iter(dim.items()))
    axis = da.get_axis_num(input_dim)
    dt = get_sampling_interval(da, input_dim)
    if scaling == "spectrum":
        scale = 1.0 / win.sum() ** 2
    elif scaling == "psd":
        scale = 1.0 / ((win * win).sum() / dt)
    else:
        raise ValueError("Scaling must be 'spectrum' or 'psd'")
    scale = np.sqrt(scale)
    if return_onesided:
        freqs = rfftfreq(nfft, dt)
    else:
        freqs = fftshift(fftfreq(nfft, dt))
    freqs = {"tie_indices": [0, len(freqs) - 1], "tie_values": [freqs[0], freqs[-1]]}

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
            result = fftshift(fft(result, n=nfft), axes=-1)
        result *= scale
        return result

    across = int(axis == 0)
    func = parallelize(across, across, parallel)(func)
    data = func(da.values)

    dt = get_sampling_interval(da, input_dim, cast=False)
    t0 = da.coords[input_dim].values[0]
    starttime = t0 + (nperseg / 2) * dt
    endtime = starttime + (data.shape[axis] - 1) * (nperseg - noverlap) * dt
    time = {
        "tie_indices": [0, data.shape[axis] - 1],
        "tie_values": [starttime, endtime],
    }

    coords = {}
    for name in da.coords:
        if name == input_dim:
            coords[input_dim] = time
        elif name == output_dim:
            coords[output_dim] = freqs
        elif da[name].dim != input_dim:  # TODO: keep non-dimensional coordinates
            coords[name] = da.coords[name]
    coords[output_dim] = freqs

    dims = da.dims + (output_dim,)

    return DataArray(data, coords, dims)
