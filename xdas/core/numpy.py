from inspect import signature

import numpy as np

from .dataarray import HANDLED_NUMPY_FUNCTIONS, DataArray


def implements(numpy_function):
    def decorator(func):
        HANDLED_NUMPY_FUNCTIONS[numpy_function] = func
        return func

    return decorator


def handled(reduce=False, drop_coords=False, **defaults):
    def decorator(func):
        sig = signature(func)

        @implements(func)
        def wrapper(*args, **kwargs):
            ba = sig.bind(*args, **kwargs)
            ba.apply_defaults()
            ba.arguments.update(defaults)
            key = next(iter(ba.arguments))
            da = ba.arguments.get(key)
            axis = ba.arguments.get("axis")
            out = ba.arguments.get("out")
            if isinstance(da, DataArray):
                ba.arguments[key] = da.data
            if isinstance(out, DataArray):
                ba.arguments["out"] = out.data
            data = func(*ba.args, **ba.kwargs)
            if reduce:
                if axis is None:
                    coords = {
                        name: coord
                        for name, coord in da.coords.items()
                        if coord.dim is None
                    }
                    dims = ()
                else:
                    coords = {
                        name: coord
                        for name, coord in da.coords.items()
                        if not coord.dim == da.dims[axis]
                    }
                    dims = tuple(dim for dim in da.dims if not dim == da.dims[axis])
            else:
                coords = da.coords
                dims = da.dims
            if drop_coords:
                return data
            else:
                return DataArray(data, coords, dims, da.name, da.attrs)

        return wrapper

    return decorator


handled()(np.fix)
handled()(np.around)
handled()(np.round)
handled()(np.clip)
handled()(np.angle)
handled()(np.i0)
handled()(np.imag)
handled()(np.nan_to_num)
# handled()(np.nonzero)
handled()(np.real_if_close)
handled()(np.real)
handled()(np.sinc)

handled(axis=-1)(np.cumprod)
handled(axis=-1)(np.nancumprod)
handled(axis=-1)(np.cumsum)
handled(axis=-1)(np.nancumsum)

handled(reduce=True)(np.all)
handled(reduce=True)(np.any)
handled(reduce=True)(np.amax)
handled(reduce=True)(np.max)
handled(reduce=True)(np.nanmax)
handled(reduce=True)(np.amin)
handled(reduce=True)(np.min)
handled(reduce=True)(np.nanmin)
handled(reduce=True)(np.argmax)
handled(reduce=True)(np.nanargmax)
handled(reduce=True)(np.argmin)
handled(reduce=True)(np.nanargmin)
handled(reduce=True)(np.median)
handled(reduce=True)(np.nanmedian)
handled(reduce=True)(np.ptp)
handled(reduce=True)(np.mean)
handled(reduce=True)(np.nanmean)
handled(reduce=True)(np.prod)
handled(reduce=True)(np.nanprod)
handled(reduce=True)(np.std)
handled(reduce=True)(np.nanstd)
handled(reduce=True)(np.sum)
handled(reduce=True)(np.nansum)
handled(reduce=True)(np.var)
handled(reduce=True)(np.nanvar)
handled(reduce=True)(np.percentile)
handled(reduce=True)(np.nanpercentile)
handled(reduce=True)(np.quantile)
handled(reduce=True)(np.nanquantile)
handled(reduce=True)(np.average)
handled(reduce=True)(np.count_nonzero)

handled(drop_coords=True)
handled(drop_coords=True)(np.diff)
handled(drop_coords=True)(np.ediff1d)
handled(drop_coords=True)(np.trapz)

# TODO: gradient
