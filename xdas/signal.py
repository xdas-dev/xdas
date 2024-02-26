import numpy as np
import scipy.signal as sp
import xarray as xr


def get_sampling_interval(da, dim):
    """
    Returns the sample spacing along a given dimension.

    Parameters
    ----------
    da : DataArray or Database
        The data from which extract the sample spacing.
    dim : str
        The dimension along which get the sample spacing.

    Returns
    -------
    float
        The sample spacing.
    """
    d = (da[dim][-1] - da[dim][0]) / (len(da[dim]) - 1)
    d = np.asarray(d)
    if np.issubdtype(d.dtype, np.timedelta64):
        d = d / np.timedelta64(1, "s")
    d = d.item()
    return d


def detrend(da, type, dim):
    """
    Detrend data along given dimension

    Parameters
    ----------
    da : DataArray
        The data to detrend.
    type : str
        Either "linear" or "constant".
    dim : str
        The dimension along which to detrend the data.

    Returns
    -------
    DataArray
        The detrended data.
    """
    axis = da.get_axis_num(dim)
    data = sp.detrend(da.values, axis, type)
    return da.copy(data=data)


def taper(da, window="hann", fftbins=False, dim="time"):
    """
    Apply a tapering window along the given dimension

    Parameters
    ----------
    da : DataArray
        The data to taper.
    window : str or tuple, optional
        The window to use, by default "hann"
    fftbins : bool, optional
        Weather to use a periodic windowing, by default False
    dim : str, optional
        Dimension along the which to taper, by default "time"

    Returns
    -------
    DataArray
        The tapered data.
    """
    axis = da.get_axis_num(dim)
    w = sp.get_window(window, da.shape[axis], fftbins=fftbins)
    shape = [-1 if ax == axis else 1 for ax in range(da.ndim)]
    w = w.reshape(shape)
    data = w * da.values
    return da.copy(data=data)


def iirfilter(da, freq, btype, corners=4, zerophase=False, dim="time"):
    """
    SOS IIR filtering along given dimension.

    da: DataArray
        Traces to filter.
    freq: float or list
        Cuttoff frequency or band corners [Hz].
    fs: float
        Sampling frequency [Hz].
    btype: {'bandpass', 'lowpass', 'highpass', 'bandstop'}
        The type of the filter.
    corners: int
        The order of the filter.
    zerophase: bool
        If True, apply filter once forwards and once backwards.
        This results in twice the filter order but zero phase shift in
        the resulting filtered trace.
    dim: str, optional
        The dimension along which to filter.
    """
    axis = da.get_axis_num(dim)
    fs = 1.0 / get_sampling_interval(da, dim)
    sos = sp.iirfilter(corners, freq, btype=btype, ftype="butter", output="sos", fs=fs)
    if zerophase:
        data = sp.sosfiltfilt(sos, da.values, axis=axis)
    else:
        data = sp.sosfilt(sos, da.values, axis=axis)
    return da.copy(data=data)


def integrate(da, midpoints=False, dim="distance"):
    """
    Integrate along a given dimension.

    Parameters
    ----------
    da : DataArray
        The data to integrate.
    midpoints : bool, optional
        Whether to move the coordinates by half a step, by default False.
    dim : str, optional
        The dimension along which to integrate, by default "distance".

    Returns
    -------
    DataArray
        The integrated data.
    """
    d = get_sampling_interval(da, dim)
    out = da.cumsum(dim) * d
    if midpoints:
        out[dim] = out[dim] + d / 2
    return out


def differentiate(da, midpoints=False, dim="distance"):
    """
    Differentiate along a given dimension.

    Parameters
    ----------
    da : DataArray
        The data to integrate.
    midpoints : bool, optional
        Whether to move the coordinates by half a step, by default False.
    dim : str, optional
        The dimension along which to integrate, by default "distance".

    Returns
    -------
    DataArray
        The integrated data.
    """
    d = get_sampling_interval(da, dim)
    out = da.diff(dim) / d
    if midpoints:
        out[dim] = out[dim] + d / 2
    return out


def segment_mean_removal(da, limits, window="hann", dim="distance"):
    """
    Piecewise mean removal.

    Parameters
    ----------
    da : DataArray
        The data that segment mean should be removed.
    limits : list of float
        The segments limits.
    window : str, optional
        The tapering windows to apply at each window, by default "hann".
    dim : str, optional
        The axis along which remove the segment means, by default "distance".

    Returns
    -------
    DataArray
        The data with segment means removed.
    """
    out = da.copy()
    for sstart, send in zip(limits[:-1], limits[1:]):
        key = {dim: slice(sstart, np.nextafter(send, -np.inf))}
        subset = out.loc[key]
        win = xr.DataArray(
            sp.get_window(window, subset.sizes[dim]),
            {dim: subset[dim]},
        )
        ref = (subset * win).sum(dim) / win.sum(dim)
        out.loc[key] -= ref
    return out


def sliding_mean_removal(da, wlen, window="hann", pad_mode="reflect", dim="distance"):
    """
    Sliding mean removal.

    Parameters
    ----------
    da : DataArray
        The data that sliding mean should be removed.
    wlen : float
        Length of the sliding mean.
    window : str, optional
        Tapering window used, by default "hann"
    pad_mode : str, optional
        Padding mode used, by default "reflect"
    dim : str, optional
        The dimension along which remove the sliding mean, by default "distance"

    Returns
    -------
    DataArray
        The data with sliding mean removed.
    """
    d = get_sampling_interval(da, dim)
    n = round(wlen / d)
    if n % 2 == 0:
        n += 1
    win = sp.get_window(window, n)
    win /= np.sum(win)
    shape = tuple(-1 if d == dim else 1 for d in da.dims)
    win = np.reshape(win, shape)
    data = da.data
    pad_width = tuple((n // 2, n // 2) if d == dim else (0, 0) for d in da.dims)
    mean = sp.fftconvolve(np.pad(data, pad_width, mode=pad_mode), win, mode="valid")
    data = data - mean
    return da.copy(data=data)


def medfilt(da, dims):
    """
    Median filter data along given dimensions

    Parameters
    ----------
    da : DataArray
        The data to detrend.
    dims : dict
        Dictionary containing the dimensions over which to apply a median filtering.
        The related values are the size of the kernel along that direction.
        If not all dims are provided, missing dimensions are associated to 1,
        i.e. no median filtering along that direction.
        At least one dimension must be passed.

    Returns
    -------
    DataArray
        The median filtered data.

    Examples
    --------
    This example is made to apply median filtering at a randomly generated dataarray
    by selecting a size of 7 for the median filtering along the time dimension
    and a size of 3 for the median filtering along the space dimension.
    The database is synthetic data.
    >>> from xdas.synthetics import generate
    >>> da = generate()
    >>> dimensions = np.array([coord for coord in da.coords])
    >>> kernel_length = [7, 3]
    >>> dims = dict(zip(dimensions, kernel_length))
    >>> filtered_da = medfilt(da, dims)
    """
    coordinates = np.array([coord for coord in da.coords])
    kernel = np.ones(len(da.coords), dtype=int)
    kernel_dict = dict(zip(coordinates, kernel))

    for dim in dims.keys():
        kernel_dict[dim] = dims[dim]

    kernel_size = np.array([kernel_dict[element] for element in kernel_dict.keys()])

    data = sp.medfilt(da.values, kernel_size)
    
    return da.copy(data=data)