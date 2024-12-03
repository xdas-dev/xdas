import numpy as np
import scipy.signal as sp

from .atoms.core import atomized
from .core.coordinates import Coordinate, get_sampling_interval
from .core.dataarray import DataArray
from .parallel import parallelize
from .spectral import stft


@atomized
def detrend(da, type="linear", dim="last", parallel=None):
    """
    Detrend data along given dimension

    Parameters
    ----------
    da : DataArray or DataArray
        The data to detrend.
    type : str
        Either "linear" or "constant".
    dim : str
        The dimension along which to detrend the data.

    Returns
    -------
    DataArray or DataArray
        The detrended data.

    Notes
    -----
    Splits on data discontinuities along `dim`.

    """
    axis = da.get_axis_num(dim)
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(sp.detrend)
    data = func(da.values, axis, type)
    return da.copy(data=data)


@atomized
def taper(da, window="hann", fftbins=False, dim="last", parallel=None):
    """
    Apply a tapering window along the given dimension

    Parameters
    ----------
    da : DataArray or DataArray
        The data to taper.
    window : str or tuple, optional
        The window to use, by default "hann"
    fftbins : bool, optional
        Weather to use a periodic windowing, by default False
    dim : str, optional
        Dimension along the which to taper, by default "time"

    Returns
    -------
    DataArray or DataArray
        The tapered data.
    """
    axis = da.get_axis_num(dim)
    w = sp.get_window(window, da.shape[axis], fftbins=fftbins)
    shape = [-1 if ax == axis else 1 for ax in range(da.ndim)]
    w = w.reshape(shape)
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(np.multiply)
    data = func(da.values, w)
    return da.copy(data=data)


@atomized
def filter(da, freq, btype, corners=4, zerophase=False, dim="last", parallel=None):
    """
    SOS IIR filtering along given dimension.

    Parameters
    ----------
    da: DataArray
        Traces to filter.
    freq: float or list
        Cuttoff frequency or band corners [Hz].
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

    Returns
    -------
    DataArray
        Filtered traces.

    """
    axis = da.get_axis_num(dim)
    across = int(axis == 0)
    fs = 1.0 / get_sampling_interval(da, dim)
    sos = sp.iirfilter(corners, freq, btype=btype, ftype="butter", output="sos", fs=fs)
    if zerophase:
        func = parallelize((None, across), across, parallel)(sp.sosfiltfilt)
    else:
        func = parallelize((None, across), across, parallel)(sp.sosfilt)
    data = func(sos, da.values, axis=axis)
    return da.copy(data=data)


@atomized
def hilbert(da, N=None, dim="last", parallel=None):
    """
    Compute the analytic signal, using the Hilbert transform.

    The transformation is done along the last axis by default.

    Parameters
    ----------
    da : DataArray
        Signal data. Must be real.
    N: int, optional
        Number of Fourier components. Default: `da.sizes[dim]`.
    dim: str, optional
        The dimension along which to transform. Default: last.
    parallel: bool or int, optional
        Whether to parallelize the function, if True all cores are used,
        if False single core, if int: number of cores.

    Returns
    -------
    DataArray
        Analytic signal of `da` along dim.

    Notes
    -----
    Splits on data discontinuities along `dim`.

    Examples
    --------
    In this example we use the Hilbert transform to determine the analytic signal.

    >>> import xdas.signal as xs
    >>> from xdas.synthetics import wavelet_wavefronts

    >>> da = wavelet_wavefronts()
    >>> xs.hilbert(da, dim="time")
    <xdas.DataArray (time: 300, distance: 401)>
    [[ 0.0497+0.1632j -0.0635+0.0125j ...  0.1352-0.3107j -0.2832-0.0126j]
     [-0.1096-0.0335j  0.124 +0.0257j ... -0.0444+0.2409j  0.1378-0.2702j]
     ...
     [-0.1977+0.0545j -0.0533-0.1947j ...  0.3722+0.125j  -0.0127+0.1723j]
     [ 0.1221-0.1808j  0.1888+0.0368j ... -0.4517+0.1581j  0.0411+0.1512j]]
    Coordinates:
      * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.980
      * distance (distance): 0.000 to 10000.000

    """
    axis = da.get_axis_num(dim)
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(sp.hilbert)
    data = func(da.values, N, axis)
    return da.copy(data=data)


@atomized
def resample(da, num, dim="last", window=None, domain="time", parallel=None):
    """
    Resample da to num samples using Fourier method along the given dimension.

    The resampled signal starts at the same value as da but is sampled with a spacing
    of len(da) / num * (spacing of da). Because a Fourier method is used, the signal is
    assumed to be periodic.

    Parameters
    ----------
    da: DataArray
        The data to be resampled.
    num: int
        The number of samples in the resampled signal.
    dim: str, optional
        The dimension along which to resample. Default is last.
    window: array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier domain. See below for
        details.
    domain: string, optional
        A string indicating the domain of the input x: `time` Consider the input da as
        time-domain (Default), `freq` Consider the input da as frequency-domain.

    Returns
    -------
    DataArray
        The resampled dataarray.

    Notes
    -----
    Splits on data discontinuities along `dim`.

    Examples
    --------
    A synthetic dataarray is resample from 300 to 100 samples along the time dimension.
    The 'hamming' window is used.

    >>> import xdas.signal as xs
    >>> from xdas.synthetics import wavelet_wavefronts

    >>> da = wavelet_wavefronts()
    >>> xs.resample(da, 100, dim='time', window='hamming', domain='time')
    <xdas.DataArray (time: 100, distance: 401)>
    [[ 0.039988  0.04855  -0.08251  ...  0.02539  -0.055219 -0.006693]
     [-0.032913 -0.016732  0.033743 ...  0.028534 -0.037685  0.032918]
     [ 0.01215   0.064107 -0.048831 ...  0.009131  0.053133  0.019843]
     ...
     [-0.036508  0.050059  0.015494 ... -0.012022 -0.064922  0.034198]
     [ 0.054003 -0.013902 -0.084095 ...  0.008979  0.080804 -0.063866]
     [-0.042741 -0.03524   0.122637 ... -0.013453 -0.075183  0.093055]]
    Coordinates:
      * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.940
      * distance (distance): 0.000 to 10000.000
    """
    axis = da.get_axis_num(dim)
    dim = da.dims[axis]
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(sp.resample)
    data, t = func(da.values, num, da[dim].values, axis, window, domain)
    new_coord = {"tie_indices": [0, num - 1], "tie_values": [t[0], t[-1]]}
    coords = {
        name: new_coord if name == dim else coord
        for name, coord in da.coords.items()
        if not (coord.dim == dim and not name == dim)  # don't handle non-dimensional
    }
    return DataArray(data, coords, da.dims, da.name, da.attrs)


@atomized
def resample_poly(
    da,
    up,
    down,
    dim="last",
    window=("kaiser", 5.0),
    padtype="constant",
    cval=None,
    parallel=None,
):
    """
    Resample da along the given dimension using polyphase filtering.

    The signal in `da` is upsampled by the factor `up`, a zero-phase low-pass
    FIR filter is applied, and then it is downsampled by the factor `down`.
    The resulting sample rate is ``up / down`` times the original sample
    rate. By default, values beyond the boundary of the signal are assumed
    to be zero during the filtering step.

    Parameters
    ----------
    da : DataArray
        The data to be resampled.
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.
    dim : int, optional
        The dimension of `da` that is resampled. Default is last.
    window : string, tuple, or array_like, optional
        Desired window to use to design the low-pass filter, or the FIR filter
        coefficients to employ. See below for details.
    padtype : string, optional
        `constant`, `line`, `mean`, `median`, `maximum`, `minimum` or any of
        the other signal extension modes supported by `scipy.signal.upfirdn`.
        Changes assumptions on values beyond the boundary. If `constant`,
        assumed to be `cval` (default zero). If `line` assumed to continue a
        linear trend defined by the first and last points. `mean`, `median`,
        `maximum` and `minimum` work as in `np.pad` and assume that the values
        beyond the boundary are the mean, median, maximum or minimum
        respectively of the array along the dimension.
    cval : float, optional
        Value to use if `padtype='constant'`. Default is zero.

    Notes
    -----
    Splits on data discontinuities along `dim`.

    Returns
    -------
    DataArray
        The resampled data.

    Examples
    --------
    This example is made to resample the input dataarray in the time domain at 100 samples
    with an original shape of 300 in time. The choosed window is a 'hamming' window.
    The dataarray is synthetic data.

    >>> import xdas.signal as xs
    >>> from xdas.synthetics import wavelet_wavefronts

    >>> da = wavelet_wavefronts()
    >>> xs.resample_poly(da, 2, 5, dim='time')
    <xdas.DataArray (time: 120, distance: 401)>
    [[-0.006378  0.012767 -0.002068 ... -0.033461  0.002603 -0.027478]
     [ 0.008851 -0.037799  0.009595 ...  0.053291 -0.0396    0.026909]
     [-0.034468  0.085153 -0.038036 ... -0.015803  0.030245  0.047028]
     ...
     [ 0.02834   0.053455 -0.155873 ...  0.033726 -0.036478 -0.016146]
     [ 0.015454 -0.062852  0.049064 ... -0.018409  0.113782 -0.072631]
     [-0.026921 -0.01264   0.087272 ...  0.001695 -0.147191  0.177587]]
    Coordinates:
      * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.950
      * distance (distance): 0.000 to 10000.000

    """
    axis = da.get_axis_num(dim)
    dim = da.dims[axis]
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(sp.resample_poly)
    data = func(da.values, up, down, axis, window, padtype, cval)
    start = da[dim][0].values
    d = da[dim][-1].values - da[dim][-2].values
    end = da[dim][-1].values + d
    new_coord = Coordinate(
        {
            "tie_indices": [0, data.shape[axis]],
            "tie_values": [start, end],
        },
        dim,
    )
    new_coord = new_coord[:-1]
    coords = {
        name: new_coord if name == dim else coord
        for name, coord in da.coords.items()
        if not (coord.dim == dim and not name == dim)  # don't handle non-dimensional
    }
    return DataArray(data, coords, da.dims, da.name, da.attrs)


@atomized
def lfilter(b, a, da, dim="last", zi=None, parallel=None):
    """
    Filter data along one-dimension with an IIR or FIR filter.

    Filter a data sequence, `da`, using a digital filter. The filter is a direct
    form II transposed implementation of the standard difference equation.

    Parameters
    ----------
    b : array_like
        The numerator coefficient vector in a 1-D sequence.
    a : array_like
        The denominator coefficient vector in a 1-D sequence.  If ``a[0]``
        is not 1, then both `a` and `b` are normalized by ``a[0]``.
    da : DataArray
        An N-dimensional input dataarray.
    dim : str, optional
        The dimension of the input data array along which to apply the
        linear filter. Default is last.
    zi : array_like or str, optional
        Initial conditions for the filter delays. If `zi` is None or ... then
        initial rest is assumed.
    parallel: bool or int, optional
        Whether to parallelize the function, if true: all cores are used, if false:
        single core, if int: n cores are used.
    Returns
    -------
    da : DataArray
        The output of the digital filter.
    zf : array, optional
        If `zi` is None, this is not returned. If `zi` is given or ... then `zf`
        holds the final filter delay values.

    Notes
    -----
    Splits on data discontinuities along `dim`.

    Examples
    --------
    >>> import scipy.signal as sp
    >>> import xdas.signal as xs
    >>> from xdas.synthetics import wavelet_wavefronts

    >>> da = wavelet_wavefronts()
    >>> b, a = sp.iirfilter(4, 0.5, btype="low")
    >>> xs.lfilter(b, a, da, dim='time')
    <xdas.DataArray (time: 300, distance: 401)>
    [[ 0.004668 -0.005968  0.007386 ... -0.0138    0.01271  -0.026618]
     [ 0.008372 -0.01222   0.022552 ... -0.041387  0.046667 -0.093521]
     [-0.008928  0.002764  0.012621 ... -0.032496  0.039645 -0.076117]
     ...
     [ 0.012576 -0.1661    0.196026 ...  0.048191 -0.014532 -0.033122]
     [-0.06294  -0.092234  0.316862 ...  0.045337 -0.139729  0.094086]
     [-0.035233  0.036613  0.044002 ... -0.053585 -0.121344  0.241415]]
    Coordinates:
      * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.980
      * distance (distance): 0.000 to 10000.000

    """
    axis = da.get_axis_num(dim)
    across = int(axis == 0)
    if zi is ...:
        n_sections = max(len(a), len(b)) - 1
        shape = tuple(
            n_sections if _axis == axis else _size
            for _axis, _size in enumerate(da.shape)
        )
        zi = np.zeros(shape)
    if zi is None:
        func = parallelize((None, None, across), across, parallel)(sp.lfilter)
        data = func(b, a, da.values, axis, zi)
        return da.copy(data=data)
    else:
        func = parallelize(
            (None, None, across, None, across), (across, across), parallel
        )(sp.lfilter)
        data, zf = func(b, a, da.values, axis, zi)
        return da.copy(data=data), zf


@atomized
def filtfilt(
    b,
    a,
    da,
    dim="last",
    padtype="odd",
    padlen=None,
    method="pad",
    irlen=None,
    parallel=None,
):
    """
    Apply a digital filter forward and backward to a signal.

    This function applies a linear digital filter twice, once forward and
    once backwards.  The combined filter has zero phase and a filter order
    twice that of the original.

    The function provides options for handling the edges of the signal.

    Parameters
    ----------
    b : (N,) array_like
        The numerator coefficient vector of the filter.
    a : (N,) array_like
        The denominator coefficient vector of the filter.  If ``a[0]``
        is not 1, then both `a` and `b` are normalized by ``a[0]``.
    da : DataArray
        The array of data to be filtered.
    dim : srt, optional
        The dimension of `da` to which the filter is applied.
        Default is last.
    padtype : str or None, optional
        Must be 'odd', 'even', 'constant', or None.  This determines the
        type of extension to use for the padded signal to which the filter
        is applied.  If `padtype` is None, no padding is used.  The default
        is 'odd'.
    padlen : int or None, optional
        The number of elements by which to extend `da` at both ends of
        `dim` before applying the filter.  This value must be less than
        ``da.sizes[dim] - 1``.  ``padlen=0`` implies no padding.
        The default value is ``3 * max(len(a), len(b))``.
    method : str, optional
        Determines the method for handling the edges of the signal, either
        "pad" or "gust".  When `method` is "pad", the signal is padded; the
        type of padding is determined by `padtype` and `padlen`, and `irlen`
        is ignored.  When `method` is "gust", Gustafsson's method is used,
        and `padtype` and `padlen` are ignored.
    irlen : int or None, optional
        When `method` is "gust", `irlen` specifies the length of the
        impulse response of the filter.  If `irlen` is None, no part
        of the impulse response is ignored.  For a long signal, specifying
        `irlen` can significantly improve the performance of the filter.

    Notes
    -----
    Splits on data discontinuities along `dim`.

    Returns
    -------
    DataArray
        The filtered output with the same coordinates as `da`.


    Examples
    --------
    >>> import scipy.signal as sp
    >>> import xdas.signal as xs
    >>> from xdas.synthetics import wavelet_wavefronts

    >>> da = wavelet_wavefronts()
    >>> b, a = sp.iirfilter(4, 0.5, btype="low")
    >>> xs.lfilter(b, a, da, dim='time')
    <xdas.DataArray (time: 300, distance: 401)>
    [[ 0.004668 -0.005968  0.007386 ... -0.0138    0.01271  -0.026618]
     [ 0.008372 -0.01222   0.022552 ... -0.041387  0.046667 -0.093521]
     [-0.008928  0.002764  0.012621 ... -0.032496  0.039645 -0.076117]
     ...
     [ 0.012576 -0.1661    0.196026 ...  0.048191 -0.014532 -0.033122]
     [-0.06294  -0.092234  0.316862 ...  0.045337 -0.139729  0.094086]
     [-0.035233  0.036613  0.044002 ... -0.053585 -0.121344  0.241415]]
    Coordinates:
      * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.980
      * distance (distance): 0.000 to 10000.000

    """
    axis = da.get_axis_num(dim)
    across = int(axis == 0)
    func = parallelize((None, None, across), across, parallel)(sp.filtfilt)
    data = func(b, a, da.values, axis, padtype, padlen, method, irlen)
    return da.copy(data=data)


@atomized
def sosfilt(sos, da, dim="last", zi=None, parallel=None):
    """
    Filter data along one dimension using cascaded second-order sections.

    Filter a data sequence, `da`, using a digital IIR filter defined by
    `sos`.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    da : DataArray
        An N-dimensional input dataarray.
    dim : str, optional
        The dimension of the input dataarray  along which to apply the
        linear filter. Default is -1.
    zi : array_like or str, optional
        Initial conditions for the cascaded filter delays.  It is a (at
        least 2D) vector of shape ``(n_sections, ..., 2, ...)``, where
        ``..., 2, ...`` denotes the shape of `da`, but with ``da.sizes[dim]``
        replaced by 2.  If `zi` is None,... , or is not given then initial rest
        (i.e. all zeros) is assumed.

    Returns
    -------
    y : DataArray
        The output of the digital filter.
    zi : ndarray, optional
        If `zi` is None, this is not returned. If `zi` is given or is ... then `zf`
        holds the final filter delay values.

    Notes
    -----
    Splits on data discontinuities along `dim`.

    Examples
    --------
    >>> import scipy.signal as sp
    >>> import xdas.signal as xs
    >>> from xdas.synthetics import wavelet_wavefronts

    >>> da = wavelet_wavefronts()
    >>> sos = sp.iirfilter(4, 0.5, btype="low", output="sos")
    >>> xs.sosfilt(sos, da, dim='time')
    <xdas.DataArray (time: 300, distance: 401)>
    [[ 0.004668 -0.005968  0.007386 ... -0.0138    0.01271  -0.026618]
     [ 0.008372 -0.01222   0.022552 ... -0.041387  0.046667 -0.093521]
     [-0.008928  0.002764  0.012621 ... -0.032496  0.039645 -0.076117]
     ...
     [ 0.012576 -0.1661    0.196026 ...  0.048191 -0.014532 -0.033122]
     [-0.06294  -0.092234  0.316862 ...  0.045337 -0.139729  0.094086]
     [-0.035233  0.036613  0.044002 ... -0.053585 -0.121344  0.241415]]
    Coordinates:
      * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.980
      * distance (distance): 0.000 to 10000.000

    """
    axis = da.get_axis_num(dim)
    across = int(axis == 0)
    if zi is ...:
        n_sections = sos.shape[0]
        shape = (n_sections,) + tuple(
            2 if index == axis else element for index, element in enumerate(da.shape)
        )
        zi = np.zeros(shape)
    if zi is None:
        func = parallelize((None, across), across, parallel)(sp.sosfilt)
        data = func(sos, da.values, axis, zi)
        return da.copy(data=data)
    else:
        func = parallelize(
            (None, across, None, across + 1), (across, across + 1), parallel
        )(sp.sosfilt)
        data, zf = func(sos, da.values, axis, zi)
        return da.copy(data=data), zf


@atomized
def sosfiltfilt(sos, da, dim="last", padtype="odd", padlen=None, parallel=None):
    """
    A forward-backward digital filter using cascaded second-order sections.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    da : DataArray
        The data to be filtered.
    dim : str, optional
        The dimension of `da` to which the filter is applied.
        Default is last.
    padtype : str or None, optional
        Must be 'odd', 'even', 'constant', or None.  This determines the
        type of extension to use for the padded signal to which the filter
        is applied.  If `padtype` is None, no padding is used.  The default
        is 'odd'.
    padlen : int or None, optional
        The number of elements by which to extend `da` at both ends of
        `dim` before applying the filter.  This value must be less than
        ``da.sizes[do,] - 1``.  ``padlen=0`` implies no padding.
        The default value is::

            3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(),
                                        (sos[:, 5] == 0).sum()))

        The extra subtraction at the end attempts to compensate for poles
        and zeros at the origin (e.g. for odd-order filters) to yield
        equivalent estimates of `padlen` to those of `filtfilt` for
        second-order section filters built with `scipy.signal` functions.

    Returns
    -------
    DataArray
        The filtered output with the same coordinates as `da`.

    Notes
    -----
    Splits on data discontinuities along `dim`.

    Examples
    --------
    >>> import scipy.signal as sp
    >>> import xdas.signal as xs
    >>> from xdas.synthetics import wavelet_wavefronts

    >>> da = wavelet_wavefronts()
    >>> sos = sp.iirfilter(4, 0.5, btype="low", output="sos")
    >>> xs.sosfiltfilt(sos, da, dim='time')
    <xdas.DataArray (time: 300, distance: 401)>
    [[ 0.04968  -0.063651  0.078731 ... -0.146869  0.135149 -0.283111]
     [-0.01724   0.018588 -0.037267 ...  0.025092 -0.107095  0.127912]
     [-0.004291 -0.002956 -0.032369 ...  0.078337 -0.150316  0.155965]
     ...
     [-0.068308 -0.06057   0.165391 ... -0.115742 -0.005265  0.19878 ]
     [-0.014123  0.039773 -0.063194 ... -0.090854 -0.053728  0.206149]
     [ 0.122203  0.188674 -0.231442 ...  0.304675 -0.452161  0.041323]]
    Coordinates:
      * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.980
      * distance (distance): 0.000 to 10000.000

    """
    axis = da.get_axis_num(dim)
    across = int(axis == 0)
    func = parallelize((None, across), across, parallel)(sp.sosfiltfilt)
    data = func(sos, da.values, axis, padtype, padlen)
    return da.copy(data=data)


@atomized
def decimate(da, q, n=None, ftype="iir", zero_phase=True, dim="last", parallel=None):
    """
    Downsample the signal after applying an anti-aliasing filter.

    By default, an order 8 Chebyshev type I filter is used. A 30 point FIR
    filter with Hamming window is used if `ftype` is 'fir'.

    Parameters
    ----------
    da : DataArray or DataArray
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
    DataArray or DataArray
        The down-sampled signal.

    Notes
    -----
    Splits on data discontinuities along `dim`.

    """
    axis = da.get_axis_num(dim)
    dim = da.dims[axis]  # TODO: this fist last thing is a bad idea...
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(sp.decimate)
    data = func(da.values, q, n, ftype, axis, zero_phase)
    coords = da.coords.copy()
    for name in coords:
        if coords[name].dim == dim:
            coords[name] = coords[name][::q]
    return DataArray(data, coords, da.dims, da.name, da.attrs)


@atomized
def integrate(da, midpoints=False, dim="last", parallel=None):
    """
    Integrate along a given dimension.

    Parameters
    ----------
    da : DataArray or DataArray
        The data to integrate.
    midpoints : bool, optional
        Whether to move the coordinates by half a step, by default False.
    dim : str, optional
        The dimension along which to integrate, by default "distance".

    Returns
    -------
    DataArray or DataArray
        The integrated data.

    Notes
    -----
    Splits on data discontinuities along `dim`.

    """
    axis = da.get_axis_num(dim)
    d = get_sampling_interval(da, dim)
    func = lambda x: np.cumsum(x, axis=axis) * d
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(func)
    data = func(da.values)
    out = da.copy(data=data)
    if midpoints:
        out[dim] = out[dim] + d / 2
    return out


@atomized
def differentiate(da, midpoints=False, dim="last", parallel=None):
    """
    Differentiate along a given dimension.

    Parameters
    ----------
    da : DataArray or DataArray
        The data to integrate.
    midpoints : bool, optional
        Whether to move the coordinates by half a step, by default False.
    dim : str, optional
        The dimension along which to integrate, by default "distance".

    Returns
    -------
    DataArray or DataArray
        The integrated data.

    Notes
    -----
    Splits on data discontinuities along `dim`.

    """
    axis = da.get_axis_num(dim)
    d = get_sampling_interval(da, dim)
    func = lambda x: np.diff(x, axis=axis) / d
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(func)
    data = func(da.values)
    out = da.isel({dim: slice(None, -1)}).copy(data=data)
    if midpoints:
        out[dim] = out[dim] + d / 2
    return out


@atomized
def segment_mean_removal(da, limits, window="hann", dim="last"):  # TODO: parallelize
    """
    Piecewise mean removal.

    Parameters
    ----------
    da : DataArray or DataArray
        The data that segment mean should be removed.
    limits : list of float
        The segments limits.
    window : str, optional
        The tapering windows to apply at each window, by default "hann".
    dim : str, optional
        The axis along which remove the segment means, by default "distance".

    Returns
    -------
    DataArray or DataArray
        The data with segment means removed.
    """
    out = da.copy()
    axis = da.get_axis_num(dim)
    for sstart, send in zip(limits[:-1], limits[1:]):
        key = {dim: slice(sstart, np.nextafter(send, -np.inf))}
        data = out.loc[key].values
        win = sp.get_window(window, data.shape[axis])
        shape = tuple(-1 if a == axis else 1 for a in range(data.ndim))
        win = np.reshape(win, shape)
        ref = np.sum(data * win, axis=axis) / np.sum(win)
        out.loc[key] = out.loc[key].values - ref  # TODO: Add DataArray Arithmetics.
    return out


@atomized
def sliding_mean_removal(
    da, wlen, window="hann", pad_mode="reflect", dim="last", parallel=None
):
    """
    Sliding mean removal.

    Parameters
    ----------
    da : DataArray or DataArray
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
    DataArray or DataArray
        The data with sliding mean removed.

    Notes
    -----
    Splits on data discontinuities along `dim`.

    """
    axis = da.get_axis_num(dim)
    d = get_sampling_interval(da, dim)
    n = round(wlen / d)
    if n % 2 == 0:
        n += 1
    win = sp.get_window(window, n)
    win /= np.sum(win)
    shape = tuple(-1 if a == axis else 1 for a in range(da.ndim))
    win = np.reshape(win, shape)
    pad_width = tuple((n // 2, n // 2) if a == axis else (0, 0) for a in range(da.ndim))
    func = lambda x: x - sp.fftconvolve(
        np.pad(x, pad_width, mode=pad_mode), win, mode="valid"
    )
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(func)
    data = func(da.values)
    return da.copy(data=data)


@atomized
def medfilt(da, kernel_dim):  # TODO: parallelize
    """
    Perform a median filter along given dimensions

    Apply a median filter to the input using a local window-size given by kernel_size.
    The array will automatically be zero-padded.

    Parameters
    ----------
    da : DataArray
        A dataarray to filter.
    kernel_dim : dict
        A dictionary which keys are the dimensions over which to apply a median
        filtering and which values are the related kernel size in that direction.
        All values must be odd. If not all dims are provided, missing dimensions
        are associated to 1, i.e. no median filtering along that direction.
        At least one dimension must be passed.

    Returns
    -------
    DataArray
        The median filtered data.

    Examples
    --------
    A median filter is applied to some synthetic dataarray with a median window size
    of 7 along the time dimension and 5 along the space dimension.

    >>> import xdas.signal as xs
    >>> from xdas.synthetics import wavelet_wavefronts

    >>> da = wavelet_wavefronts()
    >>> xs.medfilt(da, {"time": 7, "distance": 5})
    <xdas.DataArray (time: 300, distance: 401)>
    [[ 0.        0.        0.       ...  0.        0.        0.      ]
     [ 0.        0.        0.       ...  0.        0.        0.      ]
     [ 0.        0.        0.       ...  0.        0.        0.      ]
     ...
     [ 0.        0.        0.       ... -0.000402  0.        0.      ]
     [ 0.        0.        0.       ...  0.        0.        0.      ]
     [ 0.        0.        0.       ...  0.        0.        0.      ]]
    Coordinates:
      * time (time): 2023-01-01T00:00:00.000 to 2023-01-01T00:00:05.980
      * distance (distance): 0.000 to 10000.000

    """
    if not all(dim in da.dims for dim in kernel_dim.keys()):
        raise ValueError("dims provided not in dataarray")
    kernel_size = tuple(kernel_dim[dim] if dim in kernel_dim else 1 for dim in da.dims)
    data = sp.medfilt(da.values, kernel_size)
    return da.copy(data=data)
