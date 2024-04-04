import numpy as np

from ..atoms.core import atomized
from .database import HANDLED_METHODS


def implements(name=None):
    def decorator(func):
        key = name if name is not None else func.__name__
        HANDLED_METHODS[key] = func
        return func

    return decorator


@atomized
@implements()
def cumprod(db, dim="last", *, skipna=True, **kwargs):
    """
    Return the cumulative product of elements along a given dimension.

    Parameters
    ----------
    db : Database
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
    Database
        Cumulative product of the input data.
    """
    axis = db.get_axis_num(dim)
    if skipna and np.issubdtype(db.dtype, np.floating):
        return np.nancumprod(db, axis, **kwargs)
    else:
        return np.cumprod(db, axis, **kwargs)


@atomized
@implements()
def cumsum(db, dim="last", *, skipna=True, **kwargs):
    """
    Return the cumulative sum of elements along a given dimension.

    Parameters
    ----------
    db : Database
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
    Database
        Cumulative sum of the input data.
    """
    axis = db.get_axis_num(dim)
    if skipna and np.issubdtype(db.dtype, np.floating):
        return np.nancumsum(db, axis, **kwargs)
    else:
        return np.cumsum(db, axis, **kwargs)


@atomized
@implements()
def all(db, dim=None, **kwargs):
    """
    Test whether all elements along a given dimension evaluate to True.

    Parameters
    ----------
    db : Database
        Input data.
    dim : str, optional
        Dimension along which the test is performed. If None, the
        test is applied to the flattened array. Default is None.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    bool or Database
        True if all elements along the specified dimension evaluate to True, False
        otherwise. If `dim` is None, a single boolean value is returned; if
        `dim` is specified, the result has the same shape as the input data.
    """
    if dim is not None:
        axis = db.get_axis_num(dim)
    else:
        axis = None
    return np.all(db, axis, **kwargs)


@atomized
@implements()
def any(db, dim=None, **kwargs):
    """
    Test whether any element along a given dimension evaluates to True.

    Parameters
    ----------
    db : Database
        Input data.
    dim : str, optional
        Dimension along which the test is performed. If None, the
        test is applied to the flattened array. Default is None.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    bool or Database
        True if any element along the specified dimension evaluates to True, False
        otherwise. If `dim` is None, a single boolean value is returned; if
        `dim` is specified, the result has the same shape as the input data.
    """
    if dim is not None:
        axis = db.get_axis_num(dim)
    else:
        axis = None
    return np.any(db, axis, **kwargs)


@atomized
@implements()
def max(db, dim=None, *, skipna=True, **kwargs):
    """
    Compute the maximum of an array or maximum along an dimension.

    Parameters
    ----------
    db : Database
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
    scalar or Database
        Maximum value(s) along the specified dimension. If `dim` is None, a single
        scalar value is returned; if `dim` is specified, the result has the
        same shape as the input data.
    """
    if dim is not None:
        axis = db.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(db.dtype, np.floating):
        return np.max(db, axis, **kwargs)
    else:
        return np.nanmax(db, axis, **kwargs)


@atomized
@implements()
def min(db, dim=None, *, skipna=True, **kwargs):
    """
    Compute the minimum of an array or minimum along an dimension.

    Parameters
    ----------
    db : Database
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
    scalar or Database
        Minimum value(s) along the specified dimension. If `dim` is None, a single
        scalar value is returned; if `dim` is specified, the result has the
        same shape as the input data.
    """
    if dim is not None:
        axis = db.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(db.dtype, np.floating):
        return np.min(db, axis, **kwargs)
    else:
        return np.nanmin(db, axis, **kwargs)


@atomized
@implements()
def argmax(db, dim=None, *, skipna=True, **kwargs):
    """
    Return the indices of the maximum values along an dimension.

    Parameters
    ----------
    db : Database
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
    int or Database
        Indices of the maximum values along the specified dimension. If `dim` is
        None, a single integer index is returned; if `dim` is specified, the
        result has the same shape as the input data.
    """
    if dim is not None:
        axis = db.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(db.dtype, np.floating):
        return np.argmax(db, axis, **kwargs)
    else:
        return np.nanargmax(db, axis, **kwargs)


@atomized
@implements()
def argmin(db, dim=None, *, skipna=True, **kwargs):
    """
    Return the indices of the minimum values along an dimension.

    Parameters
    ----------
    db : Database
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
    int or Database
        Indices of the minimum values along the specified dimension. If `dim` is
        None, a single integer index is returned; if `dim` is specified, the
        result has the same shape as the input data.
    """
    if dim is not None:
        axis = db.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(db.dtype, np.floating):
        return np.argmin(db, axis, **kwargs)
    else:
        return np.nanargmin(db, axis, **kwargs)


@atomized
@implements()
def median(db, dim=None, *, skipna=True, **kwargs):
    """
    Compute the median along the specified dimension.

    Parameters
    ----------
    db : Database
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
    scalar or Database
        Median value(s) along the specified dimension. If `dim` is None, a single
        scalar value is returned; if `dim` is specified, the result has the
        same shape as the input data.
    """
    if dim is not None:
        axis = db.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(db.dtype, np.floating):
        return np.median(db, axis, **kwargs)
    else:
        return np.nanmedian(db, axis, **kwargs)


@atomized
@implements()
def ptp(db, dim=None, **kwargs):
    """
    Compute the range of values along the specified dimension.

    Parameters
    ----------
    db : Database
        Input data.
    dim : str, optional
        Dimension along which the range is computed. If None, the
        range of the flattened array is returned. Default is None.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    scalar or Database
        Range of values along the specified dimension (maximum - minimum). If `dim`
        is None, a single scalar value is returned; if `dim` is specified, the
        result has the same shape as the input data.
    """
    if dim is not None:
        axis = db.get_axis_num(dim)
    else:
        axis = None
    return np.ptp(db, axis, **kwargs)


@atomized
@implements()
def mean(db, dim=None, *, skipna=True, **kwargs):
    """
    Compute the arithmetic mean along the specified dimension.

    Parameters
    ----------
    db : Database
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
    scalar or Database
        Arithmetic mean value(s) along the specified dimension. If `dim` is None,
        a single scalar value is returned; if `dim` is specified, the result
        has the same shape as the input data.
    """
    if dim is not None:
        axis = db.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(db.dtype, np.floating):
        return np.mean(db, axis, **kwargs)
    else:
        return np.nanmean(db, axis, **kwargs)


@atomized
@implements()
def prod(db, dim=None, *, skipna=True, **kwargs):
    """
    Compute the product of array elements along the specified dimension.

    Parameters
    ----------
    db : Database
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
    scalar or Database
        Product of array elements along the specified dimension. If `dim` is None,
        a single scalar value is returned; if `dim` is specified, the result
        has the same shape as the input data.
    """
    if dim is not None:
        axis = db.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(db.dtype, np.floating):
        return np.prod(db, axis, **kwargs)
    else:
        return np.nanprod(db, axis, **kwargs)


@atomized
@implements()
def std(db, dim=None, *, skipna=True, **kwargs):
    """
    Compute the standard deviation along the specified dimension.

    Parameters
    ----------
    db : Database
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
    scalar or Database
        Standard deviation value(s) along the specified dimension. If `dim` is None,
        a single scalar value is returned; if `dim` is specified, the result
        has the same shape as the input data.
    """
    if dim is not None:
        axis = db.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(db.dtype, np.floating):
        return np.std(db, axis, **kwargs)
    else:
        return np.nanstd(db, axis, **kwargs)


@atomized
@implements()
def sum(db, dim=None, *, skipna=True, **kwargs):
    """
    Compute the sum of array elements along the specified dimension.

    Parameters
    ----------
    db : Database
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
    scalar or Database
        Sum of array elements along the specified dimension. If `dim` is None, a
        single scalar value is returned; if `dim` is specified, the result has
        the same shape as the input data.
    """
    if dim is not None:
        axis = db.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(db.dtype, np.floating):
        return np.sum(db, axis, **kwargs)
    else:
        return np.nansum(db, axis, **kwargs)


@atomized
@implements()
def var(db, dim=None, *, skipna=True, **kwargs):
    """
    Compute the variance along the specified dimension.

    Parameters
    ----------
    db : Database
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
    scalar or Database
        Variance value(s) along the specified dimension. If `dim` is None, a single
        scalar value is returned; if `dim` is specified, the result has the
        same shape as the input data.
    """
    if dim is not None:
        axis = db.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(db.dtype, np.floating):
        return np.var(db, axis, **kwargs)
    else:
        return np.nanvar(db, axis, **kwargs)


@atomized
@implements()
def percentile(db, q, dim=None, *, skipna=True, **kwargs):
    """
    Compute the q-th percentile of the data along the specified dimension.

    Parameters
    ----------
    db : Database
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
    scalar or Database
        q-th percentile(s) of the data along the specified dimension. If `dim` is
        None, a single scalar value is returned; if `dim` is specified, the
        result has the same shape as the input data.
    """
    if dim is not None:
        axis = db.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(db.dtype, np.floating):
        return np.percentile(db, q, axis, **kwargs)
    else:
        return np.nanpercentile(db, q, axis, **kwargs)


@atomized
@implements()
def quantile(db, q, dim=None, *, skipna=True, **kwargs):
    """
    Compute the q-th quantile of the data along the specified dimension.

    Parameters
    ----------
    db : Database
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
    scalar or Database
        q-th quantile(s) of the data along the specified dimension. If `dim` is
        None, a single scalar value is returned; if `dim` is specified, the
        result has the same shape as the input data.
    """
    if dim is not None:
        axis = db.get_axis_num(dim)
    else:
        axis = None
    if skipna and np.issubdtype(db.dtype, np.floating):
        return np.quantile(db, q, axis, **kwargs)
    else:
        return np.nanquantile(db, q, axis, **kwargs)


@atomized
@implements()
def average(db, dim=None, weights=None, **kwargs):
    """
    Compute the weighted average along the specified dimension.

    Parameters
    ----------
    db : Database
        Input data.
    dim : str, optional
        Dimension along which the weighted average is computed. If None,
        the weighted average of the flattened array is returned. Default is None.
    weights : array-like, optional
        An array of weights associated with the values in `db`. If None, all
        values are assumed to have equal weight. Default is None.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    scalar or Database
        Weighted average value(s) along the specified dimension. If `dim` is None,
        a single scalar value is returned; if `dim` is specified, the result
        has the same shape as the input data.
    """
    if dim is not None:
        axis = db.get_axis_num(dim)
    else:
        axis = None
    return np.average(db, axis, weights, **kwargs)


@atomized
@implements()
def count_nonzero(db, dim=None, **kwargs):
    """
    Count the number of non-zero values along the specified dimension.

    Parameters
    ----------
    db : Database
        Input data.
    dim : str, optional
        Dimension along which the count is computed. If None, the count
        of non-zero values in the flattened array is returned. Default is None.
    **kwargs
        Additional keyword arguments passed to the NumPy function.

    Returns
    -------
    int or Database
        Number of non-zero values along the specified dimension. If `dim` is None,
        a single integer value is returned; if `dim` is specified, the result
        has the same shape as the input data.
    """
    if dim is not None:
        axis = db.get_axis_num(dim)
    else:
        axis = None
    return np.count_nonzero(db, axis, **kwargs)


@atomized
@implements()
def diff(db, dim, n=1, *, label="upper"):
    """
    Calculate the n-th order discrete difference along given axis.

    Parameters
    ----------
    db : Database
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
    difference : Database
        The n-th order finite difference of this object.

    """
    if dim is not None:
        axis = db.get_axis_num(dim)
    else:
        axis = None
    data = np.diff(db, n, axis)
    if label == "upper":
        coords = {
            name: coord[1:] if name == dim else coord
            for name, coord in db.coords.items()
        }
    elif label == "lower":
        coords = {
            name: coord[:-1] if name == dim else coord
            for name, coord in db.coords.items()
        }
    else:
        raise ValueError("`label` must be either 'upper' or 'lower'")
    cls = db.__class__
    return cls(data, coords, db.dims, db.attrs, db.name)
