import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.signal as sp

from . import config
from .database import Database


def parse_dim(db, dim):
    if dim == "first":
        return db.dims[0]
    elif dim == "last":
        return db.dims[-1]
    else:
        if dim in db.dims:
            return dim
        else:
            ValueError(f"{dim} not in db.dims")


def get_sampling_interval(db, dim):
    """
    Returns the sample spacing along a given dimension.

    Parameters
    ----------
    db : Database or DataArray or Database
        The data from which extract the sample spacing.
    dim : str
        The dimension along which get the sample spacing.

    Returns
    -------
    float
        The sample spacing.
    """
    d = (db[dim][-1].values - db[dim][0].values) / (len(db[dim]) - 1)
    d = np.asarray(d)
    if np.issubdtype(d.dtype, np.timedelta64):
        d = d / np.timedelta64(1, "s")
    d = d.item()
    return d


def parallelize(func, axis, parallel):
    n_workers = get_workers_count(parallel)
    if n_workers == 1:
        return func
    else:
        return multithread_along_axis(func, int(axis == 0), n_workers)


def get_workers_count(parallel):
    if parallel is None:
        return config.get("n_workers")
    elif isinstance(parallel, bool):
        if parallel:
            return os.cpu_count()
        else:
            return 1
    elif isinstance(parallel, int):
        return parallel
    else:
        raise TypeError("`parallel` must be either bool or int.")


def multithread_along_axis(func, axis, n_workers):
    def wrapper(x, *args, **kwargs):
        def fn(x):
            return func(x, *args, **kwargs)

        xs = np.array_split(x, n_workers, axis)
        with ThreadPoolExecutor(n_workers) as executor:
            ys = list(executor.map(fn, xs))
        return multithreaded_concatenate(ys, axis, n_workers=n_workers)

    return wrapper


def multithreaded_concatenate(arrays, axis=0, out=None, dtype=None, n_workers=None):
    arrays = [np.asarray(array, dtype) for array in arrays]

    ndim = set(array.ndim for array in arrays)
    if len(ndim) == 1:
        (ndim,) = ndim
    else:
        raise ValueError("arrays must have the same number of dimensions.")

    dtype = set(array.dtype for array in arrays)
    if len(dtype) == 1:
        (dtype,) = dtype
    else:
        raise ValueError("arrays must have the same dtype.")

    shapes = [list(array.shape) for array in arrays]
    section_sizes = [shape.pop(axis) for shape in shapes]
    subshape = set([tuple(shape) for shape in shapes])
    if len(subshape) == 1:
        (subshape,) = subshape
    else:
        raise ValueError("arrays must have the same shape on axes other than `axis`.")
    shape = list(subshape)
    shape.insert(axis, sum(section_sizes))
    shape = tuple(shape)

    if out is None:
        out = np.empty(shape, dtype=dtype)
    else:
        if not (out.ndim == ndim and out.dtype == dtype, out.shape == shape):
            raise ValueError("`out` does not match with provided arrays.")

    div_points = np.cumsum([0] + section_sizes, dtype=int)

    with ThreadPoolExecutor(n_workers) as executor:
        for idx, array in enumerate(arrays):
            start = div_points[idx]
            end = div_points[idx + 1]
            slices = tuple(
                slice(start, end) if n == axis else slice(None) for n in range(ndim)
            )
            executor.submit(out.__setitem__, slices, array)

    return out


def detrend(db, type="linear", dim="last"):
    """
    Detrend data along given dimension

    Parameters
    ----------
    db : Database or DataArray
        The data to detrend.
    type : str
        Either "linear" or "constant".
    dim : str
        The dimension along which to detrend the data.

    Returns
    -------
    Database or DataArray
        The detrended data.
    """
    dim = parse_dim(db, dim)
    axis = db.get_axis_num(dim)
    data = sp.detrend(db.values, axis, type)
    return db.copy(data=data)


def taper(db, window="hann", fftbins=False, dim="last"):
    """
    Apply a tapering window along the given dimension

    Parameters
    ----------
    db : Database or DataArray
        The data to taper.
    window : str or tuple, optional
        The window to use, by default "hann"
    fftbins : bool, optional
        Weather to use a periodic windowing, by default False
    dim : str, optional
        Dimension along the which to taper, by default "time"

    Returns
    -------
    Database or DataArray
        The tapered data.
    """
    dim = parse_dim(db, dim)
    axis = db.get_axis_num(dim)
    w = sp.get_window(window, db.shape[axis], fftbins=fftbins)
    shape = [-1 if ax == axis else 1 for ax in range(db.ndim)]
    w = w.reshape(shape)
    data = w * db.values
    return db.copy(data=data)


def filter(db, freq, btype, corners=4, zerophase=False, dim="last", parallel=None):
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
    dim = parse_dim(db, dim)
    axis = db.get_axis_num(dim)
    fs = 1.0 / get_sampling_interval(db, dim)
    sos = sp.iirfilter(corners, freq, btype=btype, ftype="butter", output="sos", fs=fs)
    if zerophase:
        func = parallelize(
            lambda x, sos, axis: sp.sosfiltfilt(sos, x, axis), axis, parallel
        )
    else:
        func = parallelize(
            lambda x, sos, axis: sp.sosfilt(sos, x, axis), axis, parallel
        )
    data = func(db.values, sos, axis=axis)
    return db.copy(data=data)


def resample(db, num, dim='last', window=None, domain='time'):
    """
    Original function: scipy.signal.resample
    Resample db to num samples using Fourier method along the given axis.
    The resampled signal starts at the same value as x but is sampled with a spacing of len(x) / num * (spacing of x). 
    Because a Fourier method is used, the signal is assumed to be periodic.
    For more informations see: `Link text https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html`_

    Original Scipy Parameters
    -------------------------
    num: int
        The number of samples in the resampled signal.
    window: array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier domain.
    domain: string, optional
        A string indicating the domain of the input x: time Consider the input x as time-domain (Default), freq Consider the input x as frequency-domain.

    New Xdas Parameters
    -------------------
    db: Database or DataArray
        The data to be resampled.
    dim: str, optional
        The dimension along which to resample.

    Returns
    -------
    Database or DataArray
        The resampled data.
        resampled_x or (resampled_x, resampled_t)
        Either the resampled array, or, if t was given, a tuple containing the resampled array and the corresponding resampled positions.

    Examples
    --------
    This example is made to resample the input database in the time domain at 100 samples 
    with an original shape of 300 in time. The choosed window is a 'hamming' window.
    The database is synthetic data.
    >>> from xdas.synthetics import generate
    >>> db = generate()
    >>> resampled_db = xp.resample(db, 100, dim='time', window='hamming', domain='time')
    """
    dim = parse_dim(db, dim)
    axis = db.get_axis_num(dim)
    (out, t) = sp.resample(db.values, num, db[dim].values, axis, window, domain)
    coords = {}
    for name in db.coords:
        if name == dim:
            coords[dim] = dict(tie_indices=[0, num - 1], tie_values=[t[0], t[-1]])
        else:
            coords[name] = db.coords[name]
    return Database(out, coords, db.dims, db.name, db.attrs)


#def resample_poly(db, num, dim='last', window=None, domain='time'):
#def resample_poly(x, up, down, axis=0, window=('kaiser', 5.0), padtype='constant', cval=None):
def resample_poly(db, up, down, dim='last', window=('kaiser', 5.0), padtype='constant', cval=None):
    """
    Original function: scipy.signal.resample_poly
    Resample db along the given axis using polyphase filtering.
    The signal db is upsampled by the factor up, a zero-phase low-pass FIR filter is applied, 
    and then it is downsampled by the factor down. 
    The resulting sample rate is up / down times the original sample rate. 
    By default, values beyond the boundary of the signal are assumed to be zero during the filtering step.
    For more informations see: `Link text https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html`_

    Original Scipy Parameters
    -------------------------
    up: int
        The upsampling factor.
    down: int
        The downsampling factor.
    window: string, tuple, or array_like, optional
        Desired window to use to design the low-pass filter, or the FIR filter coefficients to employ.
    padtype: string, optional
        constant, line, mean, median, maximum, minimum or any of the other signal extension modes supported by scipy.signal.upfirdn. 
        Changes assumptions on values beyond the boundary. If constant, assumed to be cval (default zero). 
        If line assumed to continue a linear trend defined by the first and last points. 
        mean, median, maximum and minimum work as in np.pad and assume that the values beyond the boundary 
        are the mean, median, maximum or minimum respectively of the array along the axis.
    cval: float, optional
        Value to use if padtype=’constant’. Default is zero.

    New Xdas Parameters
    -------------------
    db: Database or DataArray
        The data to be resampled.
    dim: str, optional
        The dimension along which to resample.

    Returns
    -------
    Database or DataArray
        The resampled data.
        resampled_x or (resampled_x, resampled_t)
        Either the resampled array, or, if t was given, a tuple containing the resampled array and the corresponding resampled positions.

    Examples
    --------
    This example is made to resample the input database in the time domain at 100 samples 
    with an original shape of 300 in time. The choosed window is a 'hamming' window.
    The database is synthetic data.
    >>> from xdas.synthetics import generate
    >>> db = generate()
    >>> resampled_db = xp.resample(db, 100, dim='time', window='hamming', domain='time')
    """
    dim = parse_dim(db, dim)
    axis = db.get_axis_num(dim)
    (out, t) = sp.resample(db.values, num, db[dim].values, axis, window, domain)
    coords = {}
    for name in db.coords:
        if name == dim:
            coords[dim] = dict(tie_indices=[0, num - 1], tie_values=[t[0], t[-1]])
        else:
            coords[name] = db.coords[name]
    return Database(out, coords, db.dims, db.name, db.attrs)


def decimate(db, q, n=None, ftype=None, zero_phase=None, dim="last", parallel=None):
    """
    Downsample the signal after applying an anti-aliasing filter.

    By default, an order 8 Chebyshev type I filter is used. A 30 point FIR
    filter with Hamming window is used if `ftype` is 'fir'.

    Parameters
    ----------
    db : Database or DataArray
        The signal to be downsampled, as an N-dimensional dataarray.
    q : int
        The downsampling factor. When using IIR downsampling, it is recommended
        to call `decimate` multiple times for downsampling factors higher than
        13.
    n : int, optional
        The order of the filter (1 less than the length for 'fir'). Defaults to
        8 for 'iir' and 20 times the downsampling factor for 'fir'.
    ftype : str {'iir', 'fir'} or ``dlti`` instance, optional
        If 'iir' or 'fir', specifies the type of lowpass filter. If an instance
        of an `dlti` object, uses that object to filter before downsampling.
    dim : str, optional
        The dimension along which to decimate.
    zero_phase : bool, optional
        Prevent phase shift by filtering with `filtfilt` instead of `lfilter`
        when using an IIR filter, and shifting the outputs back by the filter's
        group delay when using an FIR filter. The default value of ``True`` is
        recommended, since a phase shift is generally not desired.

    Returns
    -------
    Database or DataArray
        The down-sampled signal.
    """
    dim = parse_dim(db, dim)
    axis = db.get_axis_num(dim)
    func = parallelize(sp.decimate, axis, parallel)
    data = func(db.values, q, n, ftype, axis, zero_phase)
    return db[{dim: slice(None, None, q)}].copy(data=data)


def integrate(db, midpoints=False, dim="last"):
    """
    Integrate along a given dimension.

    Parameters
    ----------
    db : Database or DataArray
        The data to integrate.
    midpoints : bool, optional
        Whether to move the coordinates by half a step, by default False.
    dim : str, optional
        The dimension along which to integrate, by default "distance".

    Returns
    -------
    Database or DataArray
        The integrated data.
    """
    dim = parse_dim(db, dim)
    axis = db.get_axis_num(dim)
    d = get_sampling_interval(db, dim)
    data = np.cumsum(db.values, axis=axis) * d
    out = db.copy(data=data)
    if midpoints:
        out[dim] = out[dim] + d / 2
    return out


def differentiate(db, midpoints=False, dim="last"):
    """
    Differentiate along a given dimension.

    Parameters
    ----------
    db : Database or DataArray
        The data to integrate.
    midpoints : bool, optional
        Whether to move the coordinates by half a step, by default False.
    dim : str, optional
        The dimension along which to integrate, by default "distance".

    Returns
    -------
    Database or DataArray
        The integrated data.
    """
    dim = parse_dim(db, dim)
    axis = db.get_axis_num(dim)
    d = get_sampling_interval(db, dim)
    data = np.diff(db.values, axis=axis) / d
    out = db.isel({dim: slice(None, -1)}).copy(data=data)
    if midpoints:
        out[dim] = out[dim] + d / 2
    return out


def segment_mean_removal(db, limits, window="hann", dim="last"):
    """
    Piecewise mean removal.

    Parameters
    ----------
    db : Database or DataArray
        The data that segment mean should be removed.
    limits : list of float
        The segments limits.
    window : str, optional
        The tapering windows to apply at each window, by default "hann".
    dim : str, optional
        The axis along which remove the segment means, by default "distance".

    Returns
    -------
    Database or DataArray
        The data with segment means removed.
    """
    dim = parse_dim(db, dim)
    out = db.copy()
    axis = db.get_axis_num(dim)
    for sstart, send in zip(limits[:-1], limits[1:]):
        key = {dim: slice(sstart, np.nextafter(send, -np.inf))}
        data = out.loc[key].values
        win = sp.get_window(window, data.shape[axis])
        shape = tuple(-1 if a == axis else 1 for a in range(data.ndim))
        win = np.reshape(win, shape)
        ref = np.sum(data * win, axis=axis) / np.sum(win)
        out.loc[key] = out.loc[key].values - ref  # TODO: Add Database Arithmetics.
    return out


def sliding_mean_removal(db, wlen, window="hann", pad_mode="reflect", dim="last"):
    """
    Sliding mean removal.

    Parameters
    ----------
    db : Database or DataArray
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
    Database or DataArray
        The data with sliding mean removed.
    """
    dim = parse_dim(db, dim)
    d = get_sampling_interval(db, dim)
    n = round(wlen / d)
    if n % 2 == 0:
        n += 1
    win = sp.get_window(window, n)
    win /= np.sum(win)
    shape = tuple(-1 if d == dim else 1 for d in db.dims)
    win = np.reshape(win, shape)
    data = db.data
    pad_width = tuple((n // 2, n // 2) if d == dim else (0, 0) for d in db.dims)
    mean = sp.fftconvolve(np.pad(data, pad_width, mode=pad_mode), win, mode="valid")
    data = data - mean
    return db.copy(data=data)


def medfilt(db, kernel_dim):
    """
    Perform a median filter along given dimensions

    Apply a median filter to the input using a local window-size given by kernel_size.
    The array will automatically be zero-padded.

    Parameters
    ----------
    db : Database
        A database to filter.
    kernel_dim : dict
        A dictionary which keys are the dimensions over which to apply a median
        filtering and which values are the related kernel size in that direction.
        All values must be odd. If not all dims are provided, missing dimensions
        are associated to 1, i.e. no median filtering along that direction.
        At least one dimension must be passed.

    Returns
    -------
    Database
        The median filtered data.

    Examples
    --------
    A median filter is applied to some synthetic database with a median window size
    of 7 along the time dimension and 5 along the space dimension.
    >>> import xdas.signal as xp
    >>> from xdas.synthetics import generate
    >>> db = generate()
    >>> xp.medfilt(db, {"time": 7, "distance": 5})
    <xdas.Database (time: 300, distance: 401)>
    array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]])
    Coordinates:
      * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.980
      * distance (distance): 0.000 to 10000.000
    """
    if not all(dim in db.dims for dim in kernel_dim.keys()):
        raise ValueError("dims provided not in database")
    kernel_size = tuple(kernel_dim[dim] if dim in kernel_dim else 1 for dim in db.dims)
    data = sp.medfilt(db.values, kernel_size)
    return db.copy(data=data)


def fft(db, n=None, dim={"last": "frequency"}, norm=None):
    ((olddim, newdim),) = dim.items()
    olddim = parse_dim(db, olddim)
    if n is None:
        n = db.sizes[olddim]
    axis = db.get_axis_num(olddim)
    d = get_sampling_interval(db, olddim)
    f = np.fft.fftshift(np.fft.fftfreq(n, d))
    data = np.fft.fftshift(np.fft.fft(db.values, n, axis, norm), axis)
    coords = {
        newdim if name == olddim else name: f if name == olddim else db.coords[name]
        for name in db.coords
    }
    dims = tuple(newdim if dim == olddim else dim for dim in db.dims)
    return Database(data, coords, dims, db.name, db.attrs)


def rfft(db, n=None, dim={"last": "frequency"}, norm=None):
    ((olddim, newdim),) = dim.items()
    olddim = parse_dim(db, olddim)
    if n is None:
        n = db.sizes[olddim]
    axis = db.get_axis_num(olddim)
    d = get_sampling_interval(db, olddim)
    f = np.fft.rfftfreq(n, d)
    data = np.fft.rfft(db.values, n, axis, norm)
    coords = {
        newdim if name == olddim else name: f if name == olddim else db.coords[name]
        for name in db.coords
    }
    dims = tuple(newdim if dim == olddim else dim for dim in db.dims)
    return Database(data, coords, dims, db.name, db.attrs)
