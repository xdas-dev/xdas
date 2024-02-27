import numpy as np


ufuncs = [name for name, obj in np.__dict__.items() if isinstance(obj, np.ufunc)]
no_changes = ["round"]
reduce = ["sum"]

implemented = {}


def implements(numpy_function):
    def decorator(func):
        implemented[numpy_function] = func
        return func

    return decorator


def wraps_no_changes(wrapped):
    def wrapper(db, *args, **kwargs):
        data = wrapped(db.data, *args, **kwargs)
        return db.copy(data=data)

    return wrapper


def wraps_reduce(wrapped):
    def numpy_wrapper(db, axis):
        dim = db.dims[axis]
        data = wrapped(db.data, axis)
        coords = {
            name: coord for name, coord in db.coords.items() if not coord.dim == dim
        }
        return db.__class__(data, coords, name=db.name, attrs=db.attrs)

    def xdas_wrapper(db, dim):
        axis = db.get_axis_num(dim)
        return numpy_wrapper(db, axis)

    return numpy_wrapper, xdas_wrapper


for name, obj in np.__dict__.items():
    if name in ufuncs:
        globals()[name] = obj
    elif name in no_changes:
        globals()[name] = implements(obj)(wraps_no_changes(obj))
    elif name in reduce:
        numpy_wrapper, xdas_wrapper = wraps_reduce(obj)
        implements(obj)(numpy_wrapper)
        globals()[name] = xdas_wrapper
