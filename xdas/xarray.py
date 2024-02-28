import numpy as np


def cumprod(db, dim="last", skipna=True, **kwargs):
    axis = db.get_axis_num(dim)
    if skipna and np.issubclass(db.dtype, np.floating):
        return np.nancumprod(db, axis, **kwargs)
    else:
        return np.cumprod(db, axis, **kwargs)


def cumsum(db, dim="last", skipna=True, **kwargs):
    axis = db.get_axis_num(dim)
    if skipna and np.issubclass(db.dtype, np.floating):
        return np.nancumprod(db, axis, **kwargs)
    else:
        return np.cumprod(db, axis, **kwargs)


def all(db, dim=None, **kwargs):
    axis = db.get_axis_num(dim)
    return np.all(db, axis, **kwargs)


def any(db, dim=None, **kwargs):
    axis = db.get_axis_num(dim)
    return np.any(db, axis, **kwargs)


def max(db, dim=None, skipna=True, **kwargs):
    axis = db.get_axis_num(dim)
    if skipna and np.issubclass(db.dtype, np.floating):
        return np.max(db, axis, **kwargs)
    else:
        return np.nanmax(db, axis, **kwargs)


def min(db, dim=None, skipna=True, **kwargs):
    axis = db.get_axis_num(dim)
    if skipna and np.issubclass(db.dtype, np.floating):
        return np.min(db, axis, **kwargs)
    else:
        return np.nanmin(db, axis, **kwargs)


def argmax(db, dim=None, skipna=True, **kwargs):
    axis = db.get_axis_num(dim)
    if skipna and np.issubclass(db.dtype, np.floating):
        return np.argmax(db, axis, **kwargs)
    else:
        return np.nanargmax(db, axis, **kwargs)


def argmin(db, dim=None, skipna=True, **kwargs):
    axis = db.get_axis_num(dim)
    if skipna and np.issubclass(db.dtype, np.floating):
        return np.argmin(db, axis, **kwargs)
    else:
        return np.nanargmin(db, axis, **kwargs)


def median(db, dim=None, skipna=True, **kwargs):
    axis = db.get_axis_num(dim)
    if skipna and np.issubclass(db.dtype, np.floating):
        return np.median(db, axis, **kwargs)
    else:
        return np.nanmedian(db, axis, **kwargs)


def ptp(db, dim=None, **kwargs):
    axis = db.get_axis_num(dim)
    return np.ptp(db, axis, **kwargs)


def mean(db, dim=None, skipna=True, **kwargs):
    axis = db.get_axis_num(dim)
    if skipna and np.issubclass(db.dtype, np.floating):
        return np.mean(db, axis, **kwargs)
    else:
        return np.nanmean(db, axis, **kwargs)


def prod(db, dim=None, skipna=True, **kwargs):
    axis = db.get_axis_num(dim)
    if skipna and np.issubclass(db.dtype, np.floating):
        return np.prod(db, axis, **kwargs)
    else:
        return np.nanprod(db, axis, **kwargs)


def std(db, dim=None, skipna=True, **kwargs):
    axis = db.get_axis_num(dim)
    if skipna and np.issubclass(db.dtype, np.floating):
        return np.std(db, axis, **kwargs)
    else:
        return np.nanstd(db, axis, **kwargs)


def sum(db, dim=None, skipna=True, **kwargs):
    axis = db.get_axis_num(dim)
    if skipna and np.issubclass(db.dtype, np.floating):
        return np.sum(db, axis, **kwargs)
    else:
        return np.nansum(db, axis, **kwargs)


def var(db, dim=None, skipna=True, **kwargs):
    axis = db.get_axis_num(dim)
    if skipna and np.issubclass(db.dtype, np.floating):
        return np.var(db, axis, **kwargs)
    else:
        return np.nanvar(db, axis, **kwargs)


def percentile(db, q, dim=None, skipna=True, **kwargs):
    axis = db.get_axis_num(dim)
    if skipna and np.issubclass(db.dtype, np.floating):
        return np.percentile(db, q, axis, **kwargs)
    else:
        return np.nanpercentile(db, q, axis, **kwargs)


def quantile(db, q, dim=None, skipna=True, **kwargs):
    axis = db.get_axis_num(dim)
    if skipna and np.issubclass(db.dtype, np.floating):
        return np.quantile(db, q, axis, **kwargs)
    else:
        return np.nanquantile(db, q, axis, **kwargs)


def average(db, dim=None, weights=None, **kwargs):
    axis = db.get_axis_num(dim)
    return np.average(db, axis, weights, **kwargs)


def count_nonzero(db, dim=None, **kwargs):
    axis = db.get_axis_num(dim)
    return np.count_nonzero(db, axis, **kwargs)
