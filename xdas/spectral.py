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
    """
    Compute the Short-Time Fourier Transform (STFT) of a data array.

    Parameters
    ----------
    da : DataArray
        Input data array.
    window : str or tuple or array_like, optional
        Desired window to use. If a string or tuple, it is passed to
        `scipy.signal.get_window` to generate the window values, which are
        DFT-even by default. See `scipy.signal.get_window` for a list of
        windows and required parameters. If an array, it will be used
        directly as the window and its length must be `nperseg`.
    nperseg : int, optional
        Length of each segment. Defaults to 256.
    noverlap : int, optional
        Number of points to overlap between segments. If None, `noverlap`
        defaults to `nperseg // 2`. Defaults to None.
    nfft : int, optional
        Length of the FFT used, if a zero padded FFT is desired. If None,
        the FFT length is `nperseg`. Defaults to None.
    return_onesided : bool, optional
        If True, return a one-sided spectrum for real data. If False return
        a two-sided spectrum. Defaults to True.
    dim : dict, optional
        Dictionary specifying the input and output dimensions. Defaults to
        {"last": "spectrum"}.
    scaling : {'spectrum', 'psd'}, optional
        Selects between computing the power spectral density ('psd') where
        `scale` is 1 / (sum of window squared) and computing the spectrum
        ('spectrum') where `scale` is 1 / (sum of window). Defaults to
        'spectrum'.
    parallel : optional
        Parallelization option. Defaults to None.

    Returns
    -------
    DataArray
        STFT of `da`.

    Notes
    -----
    The STFT represents a signal in the time-frequency domain by computing
    discrete Fourier transforms (DFT) over short overlapping segments of
    the signal.

    See Also
    --------
    scipy.signal.stft : Compute the Short-Time Fourier Transform (STFT).

    """
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
        elif da[name].dim != input_dim:  # TODO: keep non-dimensional coordinates
            coords[name] = da.coords[name]
    coords[output_dim] = freqs

    dims = da.dims + (output_dim,)

    return DataArray(data, coords, dims)
