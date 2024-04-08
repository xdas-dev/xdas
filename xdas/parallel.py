import os
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

import numpy as np

from . import config


def parallelize(split_axis=0, concat_axis=0, parallel=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            split_axes = split_axis if isinstance(split_axis, tuple) else (split_axis,)
            split_axes += (None,) * (len(args) - len(split_axes))
            inputs = tuple(
                value for value, axis in zip(args, split_axes) if axis is not None
            )
            input_axes = tuple(axis for axis in split_axes if axis is not None)
            args = tuple(value for value, axis in zip(args, split_axes) if axis is None)

            def fn(_inputs, tuplize=True):
                _inputs = iter(_inputs)
                _args = iter(args)
                _args = tuple(
                    next(_inputs) if axis is not None else next(_args)
                    for axis in split_axes
                )
                _outputs = func(*_args, **kwargs)
                if tuplize and not isinstance(_outputs, tuple):
                    return (_outputs,)
                else:
                    return _outputs

            if all(value.ndim <= axis for value, axis in zip(inputs, input_axes)):
                return fn(inputs, tuplize=False)

            n_jobs = inputs[0].shape[input_axes[0]]
            n_cores = get_workers_count(parallel)
            n_workers = min(n_jobs, n_cores)
            if n_workers == 1:
                return fn(inputs, tuplize=False)

            if not all(
                value.shape[axis] == inputs[0].shape[input_axes[0]]
                for value, axis in zip(inputs, input_axes)
            ):
                raise ValueError(
                    "mismatch in size along parallelization axis between inputs"
                )
            inputs = list(
                zip(
                    *tuple(
                        np.array_split(value, n_workers, axis)
                        for axis, value in zip(input_axes, inputs)
                    )
                )
            )
            with ThreadPoolExecutor(n_workers) as executor:
                outputs = tuple(zip(*list(executor.map(fn, inputs))))
            concat_axes = (
                concat_axis if isinstance(concat_axis, tuple) else (concat_axis,)
            )
            concat_axes += (None,) * (len(outputs) - len(concat_axes))
            output = tuple(
                (
                    concatenate(value, axis, n_workers=n_workers)
                    if axis is not None
                    else value[0]
                )
                for axis, value in zip(concat_axes, outputs)
            )
            if len(output) == 1:
                return output[0]
            else:
                return output

        return wrapper

    return decorator


def concatenate(arrays, axis=0, out=None, dtype=None, n_workers=None):
    """
    Multithreaded version of numpy.concatenate.

    Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    arrays: sequence of array_like
        The arrays must have the same shape, except in the dimension corresponding to
        `axis` (the first, by default).
    axis: int, optional
        The axis along which the arrays will be joined. Default is 0.
    out: ndarray, optional
        If provided, the destination to place the result. The shape must be correct,
        matching that of what concatenate would have returned if no out argument were
        specified.
    dtype: str or dtype
        If provided, the destination array will have this dtype. Cannot be provided
        together with out.

    Returns
    -------
    ndarray:
        The concatenated array.

    """
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


def get_workers_count(parallel):
    """
    Get the number of cores to use for multithreaded operations.

    Parameters
    ----------
    parallel: int or bool, optional
        if `parallel` is an integer, that number of cores will be used. if `parallel`
        is a bool either single threading (False) will be used or all cores (True). If
        `parallel` is not given (None) the default value taken from the global xdas
        configuration will be used. You can see and update this value with
        `xdas.config.get("n_workers")` and `xdas.config.set("n_workers", <your_value>)`

    Returns
    -------
    n_workers: int
        The number of cores to use.

    """
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
        raise TypeError("`parallel` must be either None, bool or int.")
