import numpy as np

from .atoms.core import atomized
from .coordinates.core import get_sampling_interval
from .core.dataarray import DataArray
from .parallel import parallelize


@atomized
def fft(da, n=None, dim={"last": "spectrum"}, norm=None, parallel=None):
    """
    Compute the discrete Fourier Transform along a given dimension.

    This function computes the one-dimensional n-point discrete Fourier Transform (DFT)
    with the efficient Fast Fourier Transform (FFT) algorithm.

    Parameters
    ----------
    da: DataArray
        The data array to process, can be complex.
    n: int, optional
        Length of transformed dimension of the output. If `n` is smaller than the
        length of the input, the input is cropped. If it is larger, the input is
        padded with zeros. If `n` is not given, the length of the input along the
        dimension specified by `dim` is used.
    dim: {str: str}, optional
        A mapping indicating as a key the dimension along which to compute the FFT, and
        as value the new name of the dimension. Default to {"last": "spectrum"}.
    norm: {“backward”, “ortho”, “forward”}, optional
        Normalization mode (see `numpy.fft`). Default is "backward". Indicates which
        direction of the forward/backward pair of transforms is scaled and with what
        normalization factor.

    Returns
    -------
    DataArray:
        The transformed input with an updated dimension name and values.

    Notes
    -----
    To perform a multidimensional fourrier operations, repeat this function on the
    desired dimensions.

    Examples
    --------
    >>> import xdas as xd
    >>> import xdas.fft as xfft
    >>> signal = xd.DataArray([0., 1., 0., -1.], coords={"time": [0, 1, 2, 3]})
    >>> xfft.fft(signal, dim={"time": "frequency"})
    <xdas.DataArray (frequency: 4)>
    [0.+0.j 0.+2.j 0.+0.j 0.-2.j]
    Coordinates:
    * frequency (frequency): [-0.5  ...  0.25]

    """
    ((olddim, newdim),) = dim.items()
    olddim = da.dims[da.get_axis_num(olddim)]
    if n is None:
        n = da.sizes[olddim]
    axis = da.get_axis_num(olddim)
    d = get_sampling_interval(da, olddim)
    f = np.fft.fftshift(np.fft.fftfreq(n, d))
    func = lambda x: np.fft.fftshift(np.fft.fft(x, n, axis, norm), axis)
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(func)
    data = func(da.values)
    coords = {
        newdim if name == olddim else name: f if name == olddim else da.coords[name]
        for name in da.coords
        if (da[name].dim != olddim or name == olddim)
    }
    dims = tuple(newdim if dim == olddim else dim for dim in da.dims)
    return DataArray(data, coords, dims, da.name, da.attrs)


@atomized
def rfft(da, n=None, dim={"last": "spectrum"}, norm=None, parallel=None):
    """
    Compute the discrete Fourier Transform  for real inputs along a given dimension.

    This function computes the one-dimensional n-point discrete Fourier Transform (DFT)
    or real-valued inputs with the efficient Fast Fourier Transform (FFT) algorithm.

    Parameters
    ----------
    da: DataArray
        The data array to process, can be complex.
    n: int, optional
        Length of transformed dimension of the output. If `n` is smaller than the
        length of the input, the input is cropped. If it is larger, the input is
        padded with zeros. If `n` is not given, the length of the input along the
        dimension specified by `dim` is used.
    dim: {str: str}, optional
        A mapping indicating as a key the dimension along which to compute the FFT, and
        as value the new name of the dimension. Default to {"last": "spectrum"}.
    norm: {“backward”, “ortho”, “forward”}, optional
        Normalization mode (see `numpy.fft`). Default is "backward". Indicates which
        direction of the forward/backward pair of transforms is scaled and with what
        normalization factor.

    Returns
    -------
    DataArray:
        The transformed input with an updated dimension name and values. The length of
        the transformed dimension is (n/2)+1 if n is even or (n+1)/2 if n is odd.

    Notes
    -----
    To perform a multidimensional fourrier operations, repeat this function on the
    desired dimensions.

    Examples
    --------
    >>> import xdas as xd
    >>> import xdas.fft as xfft
    >>> signal = xd.DataArray([0., 1., 0., -1.], coords={"time": [0, 1, 2, 3]})
    >>> xfft.rfft(signal, dim={"time": "frequency"})
    <xdas.DataArray (frequency: 3)>
    [0.+0.j 0.-2.j 0.+0.j]
    Coordinates:
    * frequency (frequency): [0.  ... 0.5]

    """
    ((olddim, newdim),) = dim.items()
    olddim = da.dims[da.get_axis_num(olddim)]
    if n is None:
        n = da.sizes[olddim]
    axis = da.get_axis_num(olddim)
    d = get_sampling_interval(da, olddim)
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(np.fft.rfft)
    f = np.fft.rfftfreq(n, d)
    data = func(da.values, n, axis, norm)
    coords = {
        newdim if name == olddim else name: f if name == olddim else da.coords[name]
        for name in da.coords
        if (da[name].dim != olddim or name == olddim)
    }
    dims = tuple(newdim if dim == olddim else dim for dim in da.dims)
    return DataArray(data, coords, dims, da.name, da.attrs)


@atomized
def ifft(da, n=None, dim={"last": "signal"}, norm=None, parallel=None):
    """
    Compute the inverse of `fft`.

    Parameters
    ----------
    da: DataArray
        The data array to process, should be complex.
    n: int, optional
        Length of transformed dimension of the output. If `n` is smaller than the
        length of the input, the input is cropped. If it is larger, the input is
        padded with zeros. If `n` is not given, the length of the input along the
        dimension specified by `dim` is used.
    dim: {str: str}, optional
        A mapping indicating as a key the dimension along which to compute the IFFT, and
        as value the new name of the dimension. Default to {"last": "time"}.
    norm: {“backward”, “ortho”, “forward”}, optional
        Normalization mode (see `numpy.fft`). Default is "backward". Indicates which
        direction of the forward/backward pair of transforms is scaled and with what
        normalization factor.

    Returns
    -------
    DataArray:
        The transformed input with an updated dimension name and values.

    Notes
    -----
    To perform a multidimensional inverse fourrier operations, repeat this function on
    the desired dimensions.

    Examples
    --------
    >>> import xdas as xd
    >>> import xdas.fft as xfft
    >>> signal = xd.DataArray([0., 1., 0., -1.], coords={"time": [0, 1, 2, 3]})
    >>> spectrum = xfft.fft(signal, dim={"time": "frequency"})
    >>> result = xfft.ifft(spectrum, dim={"frequency": "time"})
    >>> result["time"] = signal["time"]  # to match time coordinates
    >>> assert np.real(result).equals(signal)

    """
    ((olddim, newdim),) = dim.items()
    olddim = da.dims[da.get_axis_num(olddim)]
    if n is None:
        n = da.sizes[olddim]
    axis = da.get_axis_num(olddim)
    d = get_sampling_interval(da, olddim)
    f = np.fft.ifftshift(np.fft.fftfreq(n, d))
    func = lambda x: np.fft.ifft(np.fft.ifftshift(x, axis), n, axis, norm)
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(func)
    data = func(da.values)
    coords = {
        newdim if name == olddim else name: f if name == olddim else da.coords[name]
        for name in da.coords
        if (da[name].dim != olddim or name == olddim)
    }
    dims = tuple(newdim if dim == olddim else dim for dim in da.dims)
    return DataArray(data, coords, dims, da.name, da.attrs)


@atomized
def irfft(da, n=None, dim={"last": "signal"}, norm=None, parallel=None):
    """
    Computes the inverse of `rfft`.

    Parameters
    ----------
    da: DataArray
        The data array to process, can be complex.
    n : int, optional
        Length of the transformed dimension of the output. For `n` output points,
        ``n//2+1`` input points are necessary. If the input is longer than this, it is
        cropped. If it is shorter than this, it is padded with zeros. If `n` is not
        given, it is taken to be ``2*(m-1)`` where ``m`` is the length of the input
        along the dimension specified by `dim`.
    dim: {str: str}, optional
        A mapping indicating as a key the dimension along which to compute the FFT, and
        as value the new name of the dimension. Default to {"last": "time"}.
    norm: {“backward”, “ortho”, “forward”}, optional
        Normalization mode (see `numpy.fft`). Default is "backward". Indicates which
        direction of the forward/backward pair of transforms is scaled and with what
        normalization factor.

    Returns
    -------
    DataArray:
        The truncated or zero-padded input, transformed along the dimension indicated
        by `dim`, or the last one if `dim` is not specified. The length of the
        transformed dimension is `n`, or, if `n` is not given, ``2*(m-1)`` where ``m``
        is the length of the transformed dimension of the input. To get an odd number
        of output points, `n` must be specified.

    Notes
    -----
    To perform a multidimensional fourrier operations, repeat this function on the
    desired dimensions.

    Examples
    --------
    >>> import xdas as xd
    >>> import xdas.fft as xfft
    >>> signal = xd.DataArray([0., 1., 0., -1.], coords={"time": [0, 1, 2, 3]})
    >>> spectrum = xfft.rfft(signal, dim={"time": "frequency"})
    >>> result = xfft.irfft(
    ...    spectrum,
    ...    n=signal.sizes["time"],  # ensure correct output if n is odd
    ...    dim={"frequency": "time"},
    ... )
    >>> result["time"] = signal["time"]  # to match time coordinates
    >>> assert np.real(result).equals(signal)

    """
    ((olddim, newdim),) = dim.items()
    olddim = da.dims[da.get_axis_num(olddim)]
    if n is None:
        n = (da.sizes[olddim] - 1) * 2
    axis = da.get_axis_num(olddim)
    d = get_sampling_interval(da, olddim)
    across = int(axis == 0)
    func = parallelize(across, across, parallel)(np.fft.irfft)
    f = np.fft.fftshift(np.fft.fftfreq(n, d))
    data = func(da.values, n, axis, norm)
    coords = {
        newdim if name == olddim else name: f if name == olddim else da.coords[name]
        for name in da.coords
        if (da[name].dim != olddim or name == olddim)
    }
    dims = tuple(newdim if dim == olddim else dim for dim in da.dims)
    return DataArray(data, coords, dims, da.name, da.attrs)
