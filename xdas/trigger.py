import numpy as np
import pandas as pd
from numba import njit

from .atoms.core import Atom, State, atomized
from .core.coordinates import Coordinate


class Trigger(Atom):
    """
    Find picks in a data array along a given axis based on a given threshold.

    The pick findings use a triggering mechanism where triggers are turned on and off
    based on the threshold crossings. The trigger off threshold is half of the trigger
    on threshold. Picks are determined by finding the maximum value on each triggered
    region.

    Parameters
    ----------
    thresh : float
        The threshold value for picking.
    dim : str, optional
        The dimension along which to find picks. Defaults to "last".

    Notes
    -----
    For more details see the documentation of the `initialize` and `call` methods.

    Examples
    --------
    >>> import numpy as np
    >>> import xdas as xd
    >>> from xdas.atoms import Trigger

    Use case:

    >>> cft = xd.DataArray(
    ...     data=[[0.0, 0.1, 0.9, 0.8, 0.2, 0.1, 0.6, 0.7, 0.3, 0.2]],
    ...     coords={
    ...         "space": [0.0],
    ...         "time": {"tie_indices": [0, 9], "tie_values": [0.0, 9.0]},
    ...     },
    ... )

    Chunked processing using atomic processing:

    >>> atom = Trigger(thresh=0.5, dim="time")
    >>> chunks = xd.split(cft, 3, dim="time")
    >>> result = []
    >>> for chunk in chunks:
    ...     picks = atom(chunk, chunk_dim="time")
    ...     result.append(picks)
    >>> result = pd.concat(result, ignore_index=True)
    >>> result
       space  time  value
    0    0.0   2.0    0.9
    1    0.0   7.0    0.7

    """

    def __init__(self, thresh, dim="last"):
        super().__init__()

        # parameters
        self.thresh_on = float(thresh)
        self.thresh_off = float(thresh) / 2.0
        self.dim = str(dim)

        # states
        self.axis = State(...)
        self.shape = State(...)
        self.status = State(...)
        self.index = State(...)
        self.value = State(...)
        self.offset = State(...)
        self.coord = State(...)

    def initialize(self, cft, **flags):
        """
        Initialize the trigger with the following states:

        - "axis": An integer indicating the axis number of the dimension along which to
          find picks.
        - "shape": A tuple indicating the unravel shape of the lanes along wigh the
          the picks will be found.
        - "status": A boolean array indicating the trigger status for each lane.
        - "index": An integer array indicating the index of the last triggered value
          for each lane.
        - "value": A float array indicating the value of the last triggered value for
          each lane.
        - "offset": An integer indicating the offset of the chunk.
        - "coord": An InterpCoordinate containing coordinate information along 'dim' up
          to the last processed chunk.


        Parameters
        ----------
        cft : DataArray
            The characteristic function where picks must be found.
        **flags
            Optional flags.

        """
        self.axis = State(cft.get_axis_num(self.dim))
        self.shape = State(cft.shape[: self.axis] + cft.shape[self.axis + 1 :])
        self.status = State(np.zeros(self.shape, dtype=bool))
        self.index = State(np.zeros(self.shape, dtype=int))
        self.value = State(np.zeros(self.shape, dtype=float))
        self.offset = State(0)
        self.coord = State(Coordinate({"tie_indices": [], "tie_values": []}, self.dim))

    def call(self, cft, **flags):
        """
        Call the trigger.

        Parameters
        ----------
        cft : DataArray
            The characteristic function where picks must be found.
        **flags
            Optional flags.

        Returns
        -------
        picks: DataFrame
            A DataFrame containing the pick coordinates and their corresponding values.

        Notes
        -----
        In the trigger does not turn off at the end of the array, the last pick will
        not be found. This can be fixed by appending a zero to the end of the array.

        """
        data = np.asarray(cft.values, dtype=float)
        values, coords = self._call_numeric(data)
        self.coord = _concat([self.coord, cft.coords[self.dim]])

        picks = {}
        for axis, dim in enumerate(cft.dims):
            if dim == self.dim:
                picks[dim] = self.coord[coords[axis]].values
            else:
                picks[dim] = cft.coords[dim][coords[axis]].values
        picks["value"] = values
        return pd.DataFrame(picks)

    def _call_numeric(self, data):
        """
        Find picks in a N-dimensional array along a given axis based on a given threshold.

        The pick findings use a triggering mechanism where triggers are turned on and off
        based on the threshold crossings. The trigger off threshold is half of the trigger
        on threshold. Picks are determined by finding the maximum value on each triggered
        region.

        Parameters
        ----------
        data : DataArray
            The characteristic function where picks must be found.

        Returns
        -------
        coords : tuple of 1d ndarray
            A tuple containing the coordinates of the picks.
        values : 1d ndarray
            The values of the picks.

        Notes
        -----
        If the trigger does not turn off at the end of the array, the last pick will \
        not be found. This can be fixed by appending a zero to the end of the array.

        """
        data = np.moveaxis(data, self.axis, -1)
        length = data.shape[-1]

        # ravel additional axes into a unique lanes axis
        data = np.reshape(data, (-1, data.shape[-1]))
        status_view = np.reshape(self.status, (-1,))
        index_view = np.reshape(self.index, (-1,))
        value_view = np.reshape(self.value, (-1,))

        lanes, indices, values = _trigger(
            data,
            self.thresh_on,
            self.thresh_off,
            status_view,
            index_view,
            value_view,
            self.offset,
        )
        self.offset += length

        # unravel lanes indices
        if self.shape:
            coords = np.unravel_index(lanes, self.shape)
        else:
            coords = ()

        # insert found indices into the original axis position
        coords = coords[: self.axis] + (indices,) + coords[self.axis :]
        return values, coords


@atomized
def find_picks(cft, thresh, dim="last", state_dict=None):  # TODO: state_dict => state
    """
    Find picks in a data array along a given axis based on a given threshold.

    The pick findings use a triggering mechanism where triggers are turned on and off
    based on the threshold crossings. The trigger off threshold is half of the trigger
    on threshold. Picks are determined by finding the maximum value on each triggered
    region.

    Parameters
    ----------
    cft : DataArray
        The characteristic function where picks must be found.
    thresh : float
        The threshold value for picking.
    dim : str, optional
        The dimension along which to find picks. Defaults to "last".
    state_dict : dict, optional
        The state dictionary. If not provided or if '...' is passed, a new buffer will
        be initialized. The state dictionary should have the following keys:

        - "buffer": A dictionary containing the previous buffer of the trigger.
        - "offset": An integer indicating the offset of the chunk.
        - "coord": An InterpCoordinate containing coordinate information along 'dim' up
          to the last processed chunk.

        Defaults to None.

    Returns
    -------
    picks: DataFrame
        A DataFrame containing the pick coordinates and their corresponding values.
    state_dict: dict, optional
        The updated state dictionary containing the current buffer of the trigger. Can
        be used as the `state_dict` argument in the subsequent call to ensure
        continuity. If not provided (or ...), the state dictionary will not be returned.

    Notes
    -----
    In the trigger does not turn off at the end of the array, the last pick will not be
    found. This can be fixed by appending a zero to the end of the array.

    Examples
    --------
    >>> import numpy as np
    >>> import xdas as xd
    >>> from xdas.trigger import find_picks

    Use case:

    >>> cft = xd.DataArray(
    ...     data=[[0.0, 0.1, 0.9, 0.8, 0.2, 0.1, 0.6, 0.7, 0.3, 0.2]],
    ...     coords={
    ...         "space": [0.0],
    ...         "time": {"tie_indices": [0, 9], "tie_values": [0.0, 9.0]},
    ...     },
    ... )

    Monolithic processing:

    >>> find_picks(cft, thresh=0.5, dim="time")
       space  time  value
    0    0.0   2.0    0.9
    1    0.0   7.0    0.7

    Chunked processing using passing state:

    >>> chunks = xd.split(cft, 3, dim="time")
    >>> state_dict = ...
    >>> result = []
    >>> for chunk in chunks:
    ...     picks, state_dict = find_picks(
    ...         chunk, thresh=0.5, dim="time", state_dict=state_dict
    ...     )
    ...     result.append(picks)
    >>> result = pd.concat(result, ignore_index=True)
    >>> result
       space  time  value
    0    0.0   2.0    0.9
    1    0.0   7.0    0.7

    Chunked processing using atomic processing:

    >>> atom = find_picks(..., thresh=0.5, dim="time", state_dict=...)
    >>> result = []
    >>> for chunk in chunks:
    ...     picks = atom(chunk, chunk_dim="time")
    ...     result.append(picks)
    >>> result = pd.concat(result, ignore_index=True)
    >>> result
       space  time  value
    0    0.0   2.0    0.9
    1    0.0   7.0    0.7

    """
    axis = cft.get_axis_num(dim)
    data = cft.values
    return_state = state_dict is not None

    # initialize state if not provided
    if state_dict is None:
        state_dict = {"buffer": None, "offset": None, "coord": None}
    elif state_dict is ...:
        state_dict = {
            "buffer": ...,
            "offset": ...,
            "coord": Coordinate({"tie_indices": [], "tie_values": []}, dim),
        }

    # find pick indices and update state
    if return_state:
        (
            indices,
            values,
            state_dict["buffer"],
            state_dict["offset"],
        ) = _find_picks_numeric(
            data,
            thresh,
            axis,
            buffer=state_dict["buffer"],
            offset=state_dict["offset"],
        )
        state_dict["coord"] = _concat([state_dict["coord"], cft.coords[dim]])
    else:
        indices, values = _find_picks_numeric(data, thresh, axis)
        state_dict["coord"] = cft.coords[dim]

    # get picks coordinates from indices and pack it into a dataframe
    picks = {}
    for a, d in enumerate(cft.dims):
        if d == dim:
            picks[d] = state_dict["coord"][indices[a]].values
        else:
            picks[d] = cft.coords[d][indices[a]].values
    picks["value"] = values
    picks = pd.DataFrame(picks)

    # return state if requested
    if return_state:
        return picks, state_dict
    else:
        return picks


def _concat(list_of_coord):  # TODO: make it a public function/method
    """
    Concatenates a list of interpolated coordinates.

    Parameters
    ----------
    list_of_coord : list
        A list of InterpCoordinate objects to be concatenated.

    Returns
    -------
    InterpCoordinate
        The concatenated interpolated coordinate.

    Examples
    --------
    >>> import xdas as xd

    >>> coord1 = xd.Coordinate(
    ...     {"tie_indices": [0, 2], "tie_values": [10, 30]},
    ...      dim="dim",
    ... )
    >>> coord2 = xd.Coordinate(
    ...     {"tie_indices": [0, 3], "tie_values": [40, 70]},
    ...      dim="dim",
    ... )

    >>> concatenated = _concat([coord1, coord2])

    >>> concatenated.tie_indices
    array([0, 6])
    >>> concatenated.tie_values
    array([10, 70])
    >>> concatenated.dim
    'dim'

    """
    tie_indices = []
    tie_values = []
    idx = 0
    dim = list_of_coord[0].dim
    for coord in list_of_coord:
        if not coord.isinterp:
            raise ValueError("Only interpolated coordinates can be concatenated.")
        if not coord.dim == dim:
            raise ValueError("All coordinates must have the same dimension.")
        tie_indices.extend(idx + coord.tie_indices)
        tie_values.extend(coord.tie_values)
        idx += len(coord)
    coord = Coordinate({"tie_indices": tie_indices, "tie_values": tie_values}, dim)
    return coord.simplify()


def _find_picks_numeric(cft, thresh, axis=-1, buffer=None, offset=None):
    """
    Find picks in a N-dimensional array along a given axis based on a given threshold.

    The pick findings use a triggering mechanism where triggers are turned on and off
    based on the threshold crossings. The trigger off threshold is half of the trigger
    on threshold. Picks are determined by finding the maximum value on each triggered
    region.

    This function support chunked processing through the buffer and offset arguments.

    Parameters
    ----------
    cft : ndarray
        The N-dimensional characteristic function where picks must be found.
    thresh : float
        The threshold value above which picks are looked for.
    axis : int, optional
        The axis along which to find picks. Defaults to -1.
    buffer : Ellipsis or dict, optional
        The buffer dictionary containing the previous buffer of the trigger. If not
        provided or if '...' is passed, a new buffer will be initialized. The buffer
        dictionary should have the following keys:

        - "status": A boolean array indicating the trigger status for each lane.
        - "index": An integer array indicating the index of the last triggered value
          for each lane.
        - "value": A float array indicating the value of the last triggered value for
          each lane.

        Each array must have the same shape as the input array without the axis along
        which the picks are being found. Defaults to None.
    offset : Ellipsis or int, optional
        The relative position of the processed chunk within the whole array. Used to
        get absolute indices when performing chunked computations. If not provided the
        indices will be given relative the the actual given array, meaning that picks
        that occurred before the provided chunk will have negative indices.
        Defaults to None.

    Returns
    -------
    coords : tuple of 1d ndarray
        A tuple containing the coordinates of the picks.
    values : 1d ndarray
        The values of the picks.
    buffer : dict, optional
        The updated buffer dictionary containing the current buffer of the trigger.
        Can be used as the `buffer` argument in the subsequent call to ensure
        continuity.
    offset : int, optional
        The updated offset value. Can be used as the `offset` argument in the
        subsequent call to ensure correct absolute indices.

    Notes
    -----
    If the trigger does not turn off at the end of the array, the last pick will not be
    found. This can be fixed by appending a zero to the end of the array.

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
    return_buffer = buffer is not None
    return_offset = offset is not None

    # move axis to last
    cft = np.moveaxis(cft, axis, -1)
    shape = cft.shape[:-1]
    length = cft.shape[-1]

    # if not provided initialize buffer variables, copy if provided
    if buffer is None or buffer is ...:
        buffer = {
            "status": np.zeros(shape, dtype=bool),
            "index": np.zeros(shape, dtype=int),
            "value": np.zeros(shape, dtype=float),
        }
    else:
        buffer = {
            "status": np.copy(buffer["status"]).astype(bool),
            "index": np.copy(buffer["index"]).astype(int),
            "value": np.copy(buffer["value"]).astype(float),
        }

    # if not provided initialize offset
    if offset is None or offset is ...:
        offset = 0
    else:
        offset = int(offset)

    # group additional axes into a unique lanes axis
    cft = np.reshape(cft, (-1, cft.shape[-1]))
    buffer = {key: np.reshape(val, (-1,)) for key, val in buffer.items()}

    # find picks
    lanes, indices, values = _trigger(
        cft,
        thresh,
        thresh / 2.0,
        buffer["status"],
        buffer["index"],
        buffer["value"],
        offset,
    )
    # handle offsetting
    if return_offset:
        offset += length
    else:
        buffer["index"] -= length

    # unravel lanes indices
    if shape:
        coords = np.unravel_index(lanes, shape)
    else:
        coords = ()

    # insert found indices into the original axis position
    coords = coords[:axis] + (indices,) + coords[axis:]

    # reshape buffer back to original shape
    buffer = {key: np.reshape(val, shape) for key, val in buffer.items()}

    # return outputs
    out = (coords, values)
    if return_buffer:
        out = out + (buffer,)
    if return_offset:
        out = out + (offset,)
    return out


@njit("Tuple((i8[:], i8[:], f8[:]))(f8[:, :], f8, f8, b1[:], i8[:], f8[:], i8)")
def _trigger(
    cft, thresh_on, thresh_off, buffer_status, buffer_index, buffer_value, offset
):
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
    buffer_status : ndarray
        Boolean buffer of shape (n,) holding the trigger status for each lane.
    buffer_index : ndarray
        Integer buffer of shape (n,) holding the index of the last found pick for each
        lane.
    buffer_value : ndarray
        Float buffer of shape (n,) holding the value of the last found pick for each
        lane.
    offset : int
        The offset to add to the found indices.

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
        index += offset
        if buffer_status[lane]:
            if value > buffer_value[lane]:
                buffer_index[lane] = index
                buffer_value[lane] = value
            if value < thresh_off:
                buffer_status[lane] = False
                lanes.append(lane)
                indices.append(buffer_index[lane])
                values.append(buffer_value[lane])
        else:
            if value > thresh_on:
                buffer_status[lane] = True
                buffer_index[lane] = index
                buffer_value[lane] = value
    return np.array(lanes), np.array(indices), np.array(values)
