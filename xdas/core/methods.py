import numpy as np

from ..atoms.core import atomized
from .dataarray import HANDLED_METHODS


def implements(name=None):
    def decorator(func):
        key = name if name is not None else func.__name__
        HANDLED_METHODS[key] = func
        return func

    return decorator


@atomized
@implements()
def cumprod(da, dim="last", *, skipna=True, **kwargs):
    """
    Return the cumulative product of elements along a given dimension.

    Parameters
    ----------
    da : DataArray
        Input data.
    dim : str, optional
        Dimension along which the cumulative product is computed.
        Default is "last".
    skipna : bool, optional
        Whether to exclude NaN values (True) or include them (False) in the
        computation. Default is True.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    DataArray
        Cumulative product of the input data.
    """
    axis = da.get_axis_num(dim)
    if skipna and np.issubdtype(da.dtype, np.floating):
        return np.nancumprod(da, axis, **kwargs)
    else:
        return np.cumprod(da, axis, **kwargs)


@atomized
@implements()
def cumsum(da, dim="last", *, skipna=True, **kwargs):
    """
    Return the cumulative sum of elements along a given dimension.

    Parameters
    ----------
    da : DataArray
        Input data.
    dim : str, optional
        Dimension along which the cumulative sum is computed.
        Default is "last".
    skipna : bool, optional
        Whether to exclude NaN values (True) or include them (False) in the
        computation. Default is True.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    DataArray
        Cumulative sum of the input data.
    """
    axis = da.get_axis_num(dim)
    if skipna and np.issubdtype(da.dtype, np.floating):
        return np.nancumsum(da, axis, **kwargs)
    else:
        return np.cumsum(da, axis, **kwargs)


@atomized
@implements()
def all(da, dim=None, **kwargs):
    """
    Test whether all elements along a given dimension evaluate to True.

    Parameters
    ----------
    da : DataArray
        Input data.
    dim : str, optional
        Dimension along which the test is performed. If None, the
        test is applied to the flattened array. Default is None.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    bool or DataArray
        True if all elements along the specified dimension evaluate to True, False
        otherwise. If `dim` is None, a single boolean value is returned; if
        `dim` is specified, the result has the same shape as the input data.
    """
    if dim is not None:
        axis = da.get_axis_num(dim)
    else:
        axis = None
    return np.all(da, axis, **kwargs)


@atomized
@implements()
def any(da, dim=None, **kwargs):
    """
    Test whether any element along a given dimension evaluates to True.

    Parameters
    ----------
    da : DataArray
        Input data.
    dim : str, optional
        Dimension along which the test is performed. If None, the
        test is applied to the flattened array. Default is None.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    bool or DataArray
        True if any element along the specified dimension evaluates to True, False
        otherwise. If `dim` is None, a single boolean value is returned; if
        `dim` is specified, the result has the same shape as the input data.
    """
    if dim is not None:
        axis = da.get_axis_num(dim)
    else:
        axis = None
    return np.any(da, axis, **kwargs)


@atomized
@implements()
def max(da, dim=None, *, skipna=True, **kwargs):
    """
    Compute the maximum of an array or maximum along an dimension.

    Parameters
    ----------
    da : DataArray
        Input data.
    dim : str, optional
        Dimension along which the maximum value is computed. If None,
        the maximum value of the flattened array is returned. Default is None.
    skipna : bool, optional
        Whether to exclude NaN values (True) or include them (False) in the
        computation. Default is True.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    scalar or DataArray
        Maximum value(s) along the specified dimension. If `dim` is None, a single
        scalar value is returned; if `dim` is specified, the result has the
        same shape as the input data.
    """
    if dim is not None:
        axis = da.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(da.dtype, np.floating):
        return np.max(da, axis, **kwargs)
    else:
        return np.nanmax(da, axis, **kwargs)


@atomized
@implements()
def min(da, dim=None, *, skipna=True, **kwargs):
    """
    Compute the minimum of an array or minimum along an dimension.

    Parameters
    ----------
    da : DataArray
        Input data.
    dim : str, optional
        Dimension along which the minimum value is computed. If None,
        the minimum value of the flattened array is returned. Default is None.
    skipna : bool, optional
        Whether to exclude NaN values (True) or include them (False) in the
        computation. Default is True.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    scalar or DataArray
        Minimum value(s) along the specified dimension. If `dim` is None, a single
        scalar value is returned; if `dim` is specified, the result has the
        same shape as the input data.
    """
    if dim is not None:
        axis = da.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(da.dtype, np.floating):
        return np.min(da, axis, **kwargs)
    else:
        return np.nanmin(da, axis, **kwargs)


@atomized
@implements()
def argmax(da, dim=None, *, skipna=True, **kwargs):
    """
    Return the indices of the maximum values along an dimension.

    Parameters
    ----------
    da : DataArray
        Input data.
    dim : str, optional
        Dimension along which the argmax operation is performed. If
        None, the argmax operation is applied to the flattened array.
        Default is None.
    skipna : bool, optional
        Whether to exclude NaN values (True) or include them (False) in the
        computation. Default is True.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    int or DataArray
        Indices of the maximum values along the specified dimension. If `dim` is
        None, a single integer index is returned; if `dim` is specified, the
        result has the same shape as the input data.
    """
    if dim is not None:
        axis = da.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(da.dtype, np.floating):
        return np.argmax(da, axis, **kwargs)
    else:
        return np.nanargmax(da, axis, **kwargs)


@atomized
@implements()
def argmin(da, dim=None, *, skipna=True, **kwargs):
    """
    Return the indices of the minimum values along an dimension.

    Parameters
    ----------
    da : DataArray
        Input data.
    dim : str, optional
        Dimension along which the argmin operation is performed. If
        None, the argmin operation is applied to the flattened array.
        Default is None.
    skipna : bool, optional
        Whether to exclude NaN values (True) or include them (False) in the
        computation. Default is True.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    int or DataArray
        Indices of the minimum values along the specified dimension. If `dim` is
        None, a single integer index is returned; if `dim` is specified, the
        result has the same shape as the input data.
    """
    if dim is not None:
        axis = da.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(da.dtype, np.floating):
        return np.argmin(da, axis, **kwargs)
    else:
        return np.nanargmin(da, axis, **kwargs)


@atomized
@implements()
def median(da, dim=None, *, skipna=True, **kwargs):
    """
    Compute the median along the specified dimension.

    Parameters
    ----------
    da : DataArray
        Input data.
    dim : str, optional
        Dimension along which the median is computed. If None, the
        median of the flattened array is returned. Default is None.
    skipna : bool, optional
        Whether to exclude NaN values (True) or include them (False) in the
        computation. Default is True.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    scalar or DataArray
        Median value(s) along the specified dimension. If `dim` is None, a single
        scalar value is returned; if `dim` is specified, the result has the
        same shape as the input data.
    """
    if dim is not None:
        axis = da.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(da.dtype, np.floating):
        return np.median(da, axis, **kwargs)
    else:
        return np.nanmedian(da, axis, **kwargs)


@atomized
@implements()
def ptp(da, dim=None, **kwargs):
    """
    Compute the range of values along the specified dimension.

    Parameters
    ----------
    da : DataArray
        Input data.
    dim : str, optional
        Dimension along which the range is computed. If None, the
        range of the flattened array is returned. Default is None.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    scalar or DataArray
        Range of values along the specified dimension (maximum - minimum). If `dim`
        is None, a single scalar value is returned; if `dim` is specified, the
        result has the same shape as the input data.
    """
    if dim is not None:
        axis = da.get_axis_num(dim)
    else:
        axis = None
    return np.ptp(da, axis, **kwargs)


@atomized
@implements()
def mean(da, dim=None, *, skipna=True, **kwargs):
    """
    Compute the arithmetic mean along the specified dimension.

    Parameters
    ----------
    da : DataArray
        Input data.
    dim : str, optional
        Dimension along which the mean is computed. If None, the mean
        of the flattened array is returned. Default is None.
    skipna : bool, optional
        Whether to exclude NaN values (True) or include them (False) in the
        computation. Default is True.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    scalar or DataArray
        Arithmetic mean value(s) along the specified dimension. If `dim` is None,
        a single scalar value is returned; if `dim` is specified, the result
        has the same shape as the input data.
    """
    if dim is not None:
        axis = da.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(da.dtype, np.floating):
        return np.mean(da, axis, **kwargs)
    else:
        return np.nanmean(da, axis, **kwargs)


@atomized
@implements()
def prod(da, dim=None, *, skipna=True, **kwargs):
    """
    Compute the product of array elements along the specified dimension.

    Parameters
    ----------
    da : DataArray
        Input data.
    dim : str, optional
        Dimension along which the product is computed. If None, the
        product of the flattened array is returned. Default is None.
    skipna : bool, optional
        Whether to exclude NaN values (True) or include them (False) in the
        computation. Default is True.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    scalar or DataArray
        Product of array elements along the specified dimension. If `dim` is None,
        a single scalar value is returned; if `dim` is specified, the result
        has the same shape as the input data.
    """
    if dim is not None:
        axis = da.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(da.dtype, np.floating):
        return np.prod(da, axis, **kwargs)
    else:
        return np.nanprod(da, axis, **kwargs)


@atomized
@implements()
def std(da, dim=None, *, skipna=True, **kwargs):
    """
    Compute the standard deviation along the specified dimension.

    Parameters
    ----------
    da : DataArray
        Input data.
    dim : str, optional
        Dimension along which the standard deviation is computed. If
        None, the standard deviation of the flattened array is returned.
        Default is None.
    skipna : bool, optional
        Whether to exclude NaN values (True) or include them (False) in the
        computation. Default is True.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    scalar or DataArray
        Standard deviation value(s) along the specified dimension. If `dim` is None,
        a single scalar value is returned; if `dim` is specified, the result
        has the same shape as the input data.
    """
    if dim is not None:
        axis = da.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(da.dtype, np.floating):
        return np.std(da, axis, **kwargs)
    else:
        return np.nanstd(da, axis, **kwargs)


@atomized
@implements()
def sum(da, dim=None, *, skipna=True, **kwargs):
    """
    Compute the sum of array elements along the specified dimension.

    Parameters
    ----------
    da : DataArray
        Input data.
    dim : str, optional
        Dimension along which the sum is computed. If None, the sum of
        the flattened array is returned. Default is None.
    skipna : bool, optional
        Whether to exclude NaN values (True) or include them (False) in the
        computation. Default is True.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    scalar or DataArray
        Sum of array elements along the specified dimension. If `dim` is None, a
        single scalar value is returned; if `dim` is specified, the result has
        the same shape as the input data.
    """
    if dim is not None:
        axis = da.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(da.dtype, np.floating):
        return np.sum(da, axis, **kwargs)
    else:
        return np.nansum(da, axis, **kwargs)


@atomized
@implements()
def var(da, dim=None, *, skipna=True, **kwargs):
    """
    Compute the variance along the specified dimension.

    Parameters
    ----------
    da : DataArray
        Input data.
    dim : str, optional
        Dimension along which the variance is computed. If None, the
        variance of the flattened array is returned. Default is None.
    skipna : bool, optional
        Whether to exclude NaN values (True) or include them (False) in the
        computation. Default is True.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    scalar or DataArray
        Variance value(s) along the specified dimension. If `dim` is None, a single
        scalar value is returned; if `dim` is specified, the result has the
        same shape as the input data.
    """
    if dim is not None:
        axis = da.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(da.dtype, np.floating):
        return np.var(da, axis, **kwargs)
    else:
        return np.nanvar(da, axis, **kwargs)


@atomized
@implements()
def percentile(da, q, dim=None, *, skipna=True, **kwargs):
    """
    Compute the q-th percentile of the data along the specified dimension.

    Parameters
    ----------
    da : DataArray
        Input data.
    q : float or array-like of floats
        Percentile(s) to compute, between 0 and 100 inclusive.
    dim : str, optional
        Dimension along which the percentile is computed. If None, the
        percentile of the flattened array is returned. Default is None.
    skipna : bool, optional
        Whether to exclude NaN values (True) or include them (False) in the
        computation. Default is True.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    scalar or DataArray
        q-th percentile(s) of the data along the specified dimension. If `dim` is
        None, a single scalar value is returned; if `dim` is specified, the
        result has the same shape as the input data.
    """
    if dim is not None:
        axis = da.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(da.dtype, np.floating):
        return np.percentile(da, q, axis, **kwargs)
    else:
        return np.nanpercentile(da, q, axis, **kwargs)


@atomized
@implements()
def quantile(da, q, dim=None, *, skipna=True, **kwargs):
    """
    Compute the q-th quantile of the data along the specified dimension.

    Parameters
    ----------
    da : DataArray
        Input data.
    q : float or array-like of floats
        Quantile(s) to compute, between 0 and 1 inclusive.
    dim : str, optional
        Dimension along which the quantile is computed. If None, the
        quantile of the flattened array is returned. Default is None.
    skipna : bool, optional
        Whether to exclude NaN values (True) or include them (False) in the
        computation. Default is True.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    scalar or DataArray
        q-th quantile(s) of the data along the specified dimension. If `dim` is
        None, a single scalar value is returned; if `dim` is specified, the
        result has the same shape as the input data.
    """
    if dim is not None:
        axis = da.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(da.dtype, np.floating):
        return np.quantile(da, q, axis, **kwargs)
    else:
        return np.nanquantile(da, q, axis, **kwargs)


@atomized
@implements()
def average(da, dim=None, weights=None, **kwargs):
    """
    Compute the weighted average along the specified dimension.

    Parameters
    ----------
    da : DataArray
        Input data.
    dim : str, optional
        Dimension along which the weighted average is computed. If None,
        the weighted average of the flattened array is returned. Default is None.
    weights : array-like, optional
        An array of weights associated with the values in `da`. If None, all
        values are assumed to have equal weight. Default is None.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    scalar or DataArray
        Weighted average value(s) along the specified dimension. If `dim` is None,
        a single scalar value is returned; if `dim` is specified, the result
        has the same shape as the input data.
    """
    if dim is not None:
        axis = da.get_axis_num(dim)
    else:
        axis = None
    return np.average(da, axis, weights, **kwargs)


@atomized
@implements()
def count_nonzero(da, dim=None, **kwargs):
    """
    Count the number of non-zero values along the specified dimension.

    Parameters
    ----------
    da : DataArray
        Input data.
    dim : str, optional
        Dimension along which the count is computed. If None, the count
        of non-zero values in the flattened array is returned. Default is None.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    int or DataArray
        Number of non-zero values along the specified dimension. If `dim` is None,
        a single integer value is returned; if `dim` is specified, the result
        has the same shape as the input data.
    """
    if dim is not None:
        axis = da.get_axis_num(dim)
    else:
        axis = None
    return np.count_nonzero(da, axis, **kwargs)


@atomized
@implements()
def diff(da, dim, n=1, *, label="upper"):
    """
    Calculate the n-th order discrete difference along given axis.

    Parameters
    ----------
    da : DataArray
        Input data.
    dim : str, optional
        Dimension over which to calculate the finite difference.
    n : int, default: 1
        The number of times values are differentiated.
    label : {"upper", "lower"}, default: "upper"
        The new coordinate in dimension ``dim`` will have the
        values of either the minuend's or subtrahend's coordinate
        for values 'upper' and 'lower', respectively.

    Returns
    -------
    difference : DataArray
        The n-th order finite difference of this object.

    """
    if dim is not None:
        axis = da.get_axis_num(dim)
    else:
        axis = None
    data = np.diff(da, n, axis)
    if label == "upper":
        coords = {
            name: coord[1:] if name == dim else coord
            for name, coord in da.coords.items()
        }
    elif label == "lower":
        coords = {
            name: coord[:-1] if name == dim else coord
            for name, coord in da.coords.items()
        }
    else:
        raise ValueError("`label` must be either 'upper' or 'lower'")
    cls = da.__class__
    return cls(data, coords, da.dims, da.attrs, da.name)
