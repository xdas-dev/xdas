import numpy as np

NUMPY_HANDLED_FUNCTIONS = {}


def implements(numpy_function):

    def decorator(func):
        NUMPY_HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


def apply_ufunc(db, ufunc, method, *inputs, **kwargs):
    if not method == "__call__":
        return NotImplemented
    inputs = tuple(
        value.data if isinstance(value, db.__class__) else value for value in inputs
    )
    if "out" in kwargs:
        kwargs["out"] = tuple(
            value.data if isinstance(value, db.__class__) else value
            for value in kwargs["out"]
        )
    if "where" in kwargs:
        kwargs["where"] = tuple(
            value.data if isinstance(value, db.__class__) else value
            for value in kwargs["where"]
        )
    data = getattr(ufunc, method)(*inputs, **kwargs)
    if isinstance(data, tuple):
        return tuple(db.copy(data=d) for d in data)
    else:
        return db.copy(data=data)


def dispatch(dbpos=0, axispos=None, outpos=None, reduce=False):
    def decorator(func):
        @implements(func)
        def wrapper(*args, **kwargs):
            nargs = len(args)
            db = args[dbpos]
            cls = db.__class__
            args = tuple(db.data if idx == dbpos else args[idx] for idx in range(nargs))
            if axispos is not None:
                if nargs > axispos:
                    axis = args[axispos]
                elif "axis" in kwargs:
                    axis = kwargs["axis"]
                else:
                    axis = None
            if outpos is not None:
                if nargs > outpos:
                    out = args[outpos]
                    if isinstance(out, cls):
                        args = tuple(
                            out.data if idx == outpos else args[idx]
                            for idx in range(nargs)
                        )
                elif "out" in kwargs:
                    out = kwargs["out"]
                    if isinstance(out, cls):
                        kwargs = {
                            key: out.data if key == "out" else value
                            for key, value in kwargs.items()
                        }
                else:
                    out = None
            data = func(*args, **kwargs)
            if reduce:
                if axis is None:
                    coords = {
                        name: coord
                        for name, coord in db.coords.items()
                        if coord.dim is None
                    }
                    dims = ()
                else:
                    coords = {
                        name: coord
                        for name, coord in db.coords.items()
                        if not coord.dim == db.dims[axis]
                    }
                    dims = tuple(dim for dim in db.dims if not dim == db.dims[axis])
            else:
                coords = db.coords
                dims = db.dims
            return cls(data, coords, dims, db.name, db.attrs)

        return wrapper

    return decorator


elementwise = dispatch(outpos=1)
elementwise(np.fix)

elementwise_one_arg = dispatch(outpos=2)
elementwise_one_arg(np.around)
elementwise_one_arg(np.round)

elementwise_two_args = dispatch(outpos=3)
elementwise_two_args(np.clip)

elementwise_no_out = dispatch()
elementwise_no_out(np.angle)
elementwise_no_out(np.i0)
elementwise_no_out(np.imag)
elementwise_no_out(np.nan_to_num)
elementwise_no_out(np.nonzero)
elementwise_no_out(np.real_if_close)
elementwise_no_out(np.real)
elementwise_no_out(np.sinc)

along = dispatch(axispos=1, outpos=3)
along(np.cumprod)
along(np.nancumprod)
along(np.cumsum)
along(np.nancumsum)

reduce = dispatch(axispos=1, outpos=2, reduce=True)
reduce(np.all)
reduce(np.any)
reduce(np.amax)
reduce(np.max)
reduce(np.nanmax)
reduce(np.amin)
reduce(np.min)
reduce(np.nanmin)
reduce(np.argmax)
reduce(np.nanargmax)
reduce(np.argmin)
reduce(np.nanargmin)
reduce(np.median)
reduce(np.nanmedian)
reduce(np.ptp)

reduce_dtype = dispatch(axispos=1, outpos=3, reduce=True)
reduce_dtype(np.mean)
reduce_dtype(np.nanmean)
reduce_dtype(np.prod)
reduce_dtype(np.nanprod)
reduce_dtype(np.std)
reduce_dtype(np.nanstd)
reduce_dtype(np.sum)
reduce_dtype(np.nansum)
reduce_dtype(np.var)
reduce_dtype(np.nanvar)

reduce_one_arg = dispatch(axispos=2, outpos=3, reduce=True)
reduce_one_arg(np.percentile)
reduce_one_arg(np.nanpercentile)
reduce_one_arg(np.quantile)
reduce_one_arg(np.nanquantile)

reduce_no_out = dispatch(axispos=1, reduce=True)
reduce_no_out(np.average)
reduce_no_out(np.count_nonzero)


special = [
    "diff",
    "ediff1d",
    "gradient",
    "trapz",
]
