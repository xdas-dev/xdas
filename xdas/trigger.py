import stat

import numpy as np
import pandas as pd
from numba import njit
from sympy import Le


def find_picks(cft, thresh, dim="last"):
    """
    Find picks in a data array along a given axis based on a given threshold.

    The pick findings use a triggering mechanism where triggers are turned on and off
    based on the threshold crossings. The trigger off threshold is half of the trigger
    on threshold. Picks are determined by finding the maximum value on each triggered
    region.

    Parameters
    ----------
    cft : DataArray
        The DataArray object.
    thresh : float
        The threshold value for picking.
    dim : str, optional
        The dimension along which to find picks. Defaults to "last".

    Returns
    -------
    DataFrame
        A DataFrame containing the pick coordinates and their corresponding values.

    Notes
    -----
    In the trigger does not turn off at the end of the array, the last pick will not be
    found. This can be fixed by appending a zero to the end of the array.

    Examples
    --------
    >>> import numpy as np
    >>> import xdas as xd
    >>> from xdas.trigger import find_picks

    >>> cft = xd.DataArray(
    ...     data=[[0., 0.1, 0.9, 0.8, 0.2, 0.1, 0.6, 0.7, 0.3, 0.2]],
    ...     coords={"space": [0.0], "time": np.linspace(0, 0.9, 10)},
    ... )

    >>> find_picks(cft, thresh=0.5, dim="time")
       space  time  value
    0    0.0   0.2    0.9
    1    0.0   0.7    0.7

    """
    data = cft.values
    indices, values = _find_picks_numeric(data, thresh, axis=cft.get_axis_num(dim))
    picks = {
        dim: cft.coords[dim][indices[axis]].values for axis, dim in enumerate(cft.dims)
    }
    picks["value"] = values
    return pd.DataFrame(picks)


def _find_picks_numeric(cft, thresh, axis=-1, state=None):
    """
    Find picks in a N-dimensional array along a given axis based on a given threshold.

    The pick findings use a triggering mechanism where triggers are turned on and off
    based on the threshold crossings. The trigger off threshold is half of the trigger
    on threshold. Picks are determined by finding the maximum value on each triggered
    region.

    Parameters
    ----------
    cft : ndarray
        The array to search for picks.
    thresh : float
        The threshold value for picking.
    axis : int, optional
        The axis along which to find picks. Defaults to -1.
    state : Ellipsis or dict, optional
        The state dictionary containing the previous state of the trigger. If not
        provided or if '...' is passed, a new state will be initialized. The state
        dictionary should have the following keys:

        - "status": A boolean array indicating the trigger status for each lane.
        - "index": An integer array indicating the index of the last triggered value
          for each lane.
        - "value": A float array indicating the value of the last triggered value for
          each lane.

        Each array must have the same shape as the input array without the axis along
        which the picks are being found.

        The state dictionary can be used to continue the trigger from the last state.

    Returns
    -------
    coords : tuple of 1d ndarray
        A tuple containing the coordinates of the picks.
    values : 1d ndarray
        The values of the picks.
    state : dict, optional
        The updated state dictionary containing the current state of the trigger.
        Indices are given as negative values relative to the end of the array to
        distinguish picks from the last state.
        This can be used as the `state` argument in subsequent calls to
        `_find_picks_numeric` to continue the trigger from the last state.

    Notes
    -----
    If the trigger does not turn off at the end of the array, the last pick will not be found.
    This can be fixed by appending a zero to the end of the array.

    Examples
    --------
    >>> import numpy as np
    >>> import xdas as xd
    >>> from xdas.trigger import _find_picks_numeric

    >>> cft = np.array([[0., 0.1, 0.9, 0.8, 0.2, 0.1, 0.6, 0.7, 0.3, 0.2]])

    >>> _find_picks_numeric(cft, thresh=0.5, axis=-1)
    ((array([0, 0]), array([2, 7])), array([0.9, 0.7]))

    """
    cft = np.asarray(cft, dtype=float)
    thresh = float(thresh)
    axis = cft.ndim + int(axis) if axis < 0 else int(axis)
    return_state = state is not None

    # move axis to last
    cft = np.moveaxis(cft, axis, -1)
    shape = cft.shape[:-1]
    length = cft.shape[-1]

    # if needed initialize state variables, copy if provided
    if state is None or state is ...:
        state = {
            "status": np.zeros(shape, dtype=bool),
            "index": np.zeros(shape, dtype=int),
            "value": np.zeros(shape, dtype=float),
        }
    else:
        state = {
            "status": np.copy(state["status"]).astype(bool),
            "index": np.copy(state["index"]).astype(int),
            "value": np.copy(state["value"]).astype(float),
        }

    # group additional axes into a unique lanes axis
    cft = np.reshape(cft, (-1, cft.shape[-1]))
    state = {key: np.reshape(val, (-1,)) for key, val in state.items()}

    # find picks
    lanes, indices, values = _trigger(
        cft, thresh, thresh / 2.0, state["status"], state["index"], state["value"]
    )

    # unravel lanes indices
    if shape:
        coords = np.unravel_index(lanes, shape)
    else:
        coords = ()

    # insert found indices into the original axis position
    coords = coords[:axis] + (indices,) + coords[axis:]

    # set state indices to negative relative to the end of the array
    state["index"] -= length

    # reshape state back to original shape
    state = {key: np.reshape(val, shape) for key, val in state.items()}

    # return picks
    if return_state:
        return coords, values, state
    else:
        return coords, values


@njit("Tuple((i8[:], i8[:], f8[:]))(f8[:, :], f8, f8, b1[:], i8[:], f8[:])")
def _trigger(cft, thresh_on, thresh_off, state_status, state_index, state_value):
    """
    Perform trigger detection on the input data.

    Parameters
    ----------
    cft : ndarray
        2D array of shape (n, m) representing the input data. Each row is a lane. Each
        column is the signal onto perform trigger detection.
    thresh_on : float
        Threshold value for turning on the trigger.
    thresh_off : float
        Threshold value for turning off the trigger.
    state_status : ndarray
        Boolean buffer of shape (n,) holding the trigger status for each lane.
    state_index : ndarray
        Integer buffer of shape (n,) holding the index of the last found pick for each
        lane.
    state_value : ndarray
        Float buffer of shape (n,) holding the value of the last found pick for each
        lane.

    Returns
    -------
    tuple of ndarray
        A tuple containing three arrays of shape (k,) where k is the number of picks
        found. The arrays are:

        - lanes : lanes indices (along first axis) of the picks.
        - indices : signal indices (along last axis) of the picks.
        - values : values of the picks.

    """
    lanes = []
    indices = []
    values = []
    for (lane, index), value in np.ndenumerate(cft):
        if state_status[lane]:
            if value > state_value[lane]:
                state_index[lane] = index
                state_value[lane] = value
            if value < thresh_off:
                state_status[lane] = False
                lanes.append(lane)
                indices.append(state_index[lane])
                values.append(state_value[lane])
        else:
            if value > thresh_on:
                state_status[lane] = True
                state_index[lane] = index
                state_value[lane] = value
    return np.array(lanes), np.array(indices), np.array(values)
