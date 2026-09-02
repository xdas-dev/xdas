"""
Detection atoms: turn a characteristic function into picks.

These atoms emit :class:`pandas.DataFrame` objects rather than arrays — the
transducer contract makes that unremarkable, a pick table is just another
chunk type flowing downstream to a CSV sink. :class:`Trigger` is the
threshold detector; its lowercase twin is :func:`xdas.trigger`.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd
from numba import njit

from ..coordinates import Coordinate
from ..core import concat_coords
from .core import Atom, State, atomized

__all__ = ["Trigger", "trigger"]

#: Name of the label dimension a mapping of thresholds keys on. It is the
#: dimension :class:`~xdas.atoms.Annotate` appends to its characteristic
#: function, so the two agree by construction.
PHASE_DIM = "phase"


class Trigger(Atom):
    """
    Find picks in a data array along a given axis based on a given threshold.

    The pick findings use a triggering mechanism where triggers are turned on and off
    based on the threshold crossings. The trigger off threshold is half of the trigger
    on threshold. Picks are determined by finding the maximum value on each triggered
    region.

    Parameters
    ----------
    thresh : float or mapping
        The threshold value for picking. A scalar applies to every lane. A
        mapping keyed on the ``phase`` coordinate gives one threshold per
        label; labels the mapping does not list are **never** triggered, which
        is how a characteristic function keeps carrying its noise class
        without that class ever producing a pick. Keying on the label rather
        than on its position is a correctness requirement: the label order of
        a model is a property of its weight set and flips between them.
    dim : str, optional
        The dimension along which to find picks. Defaults to "time".
    coords : sequence of str, "auto" or None, optional
        The coordinates used to annotate the picks, one column per name. Any
        coordinate of the input can be named, including non-dimensional ones
        (a station identifier attached to the distance dimension, a latitude,
        ...): each is indexed along the dimension it varies on. Scalar (0-d)
        coordinates are emitted as constant columns, which is what lets a pick
        table carry a ``network``/``station``/``location`` identity when
        picking a single array. Defaults to ``"auto"``: the scalar
        coordinates, then the other dimension coordinates, then the picked
        dimension — identity first, measurement last, so the columns do not
        depend on the input's dimension order (see :meth:`_names`); ``None``
        restricts it to the dimension coordinates, in the input's own order.

    Notes
    -----
    For more details see the documentation of the `initialize`, `call` and `flush`
    methods.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import xdas as xd

    Use case:

    >>> cft = xd.DataArray(
    ...     data=[[0.0, 0.1, 0.9, 0.8, 0.2, 0.1, 0.6, 0.7, 0.3, 0.2]],
    ...     coords={
    ...         "space": [0.0],
    ...         "time": {"tie_indices": [0, 9], "tie_values": [0.0, 9.0], "sampling_interval": 1.0},
    ...     },
    ... )

    Eager processing with the functional twin:

    >>> xd.trigger(cft, thresh=0.5)
       space  time  value
    0    0.0   2.0    0.9
    1    0.0   7.0    0.7

    Chunked processing using atomic processing. `flush` closes whatever trigger
    is still open once the stream ends:

    >>> atom = xd.trigger(..., thresh=0.5)
    >>> chunks = xd.split(cft, 3, dim="time")
    >>> result = []
    >>> for chunk in chunks:
    ...     picks = atom(chunk, chunk_dim="time")
    ...     result.append(picks)
    >>> result += atom.flush()
    >>> result = pd.concat(result, ignore_index=True)
    >>> result
       space  time  value
    0    0.0   2.0    0.9
    1    0.0   7.0    0.7

    A trigger that never turns off used to lose its pick; it is now closed at
    the end of the record, as `obspy.trigger_onset` does:

    >>> tail = cft.isel(time=slice(0, 4))
    >>> xd.trigger(tail, thresh=0.5)
       space  time  value
    0    0.0   2.0    0.9

    Annotating the picks with a non-dimensional coordinate instead of the
    dimension it is attached to:

    >>> cft = cft.assign_coords(station=("space", ["ST01"]))
    >>> xd.trigger(cft, thresh=0.5, coords=["time", "station"])
       time station  value
    0   2.0    ST01    0.9
    1   7.0    ST01    0.7

    A characteristic function labelled by phase takes one threshold per label:

    >>> cft = xd.DataArray(
    ...     data=[[0.0, 0.9, 0.0], [0.0, 0.8, 0.0], [0.0, 0.9, 0.0]],
    ...     coords={
    ...         "phase": ["N", "P", "S"],
    ...         "time": {"tie_indices": [0, 2], "tie_values": [0.0, 2.0], "sampling_interval": 1.0},
    ...     },
    ... )
    >>> xd.trigger(cft, thresh={"P": 0.5, "S": 0.5})
      phase  time  value
    0     P   1.0    0.8
    1     S   1.0    0.9

    The noise class is the loudest lane here and carries no entry, so it never
    triggers. Scalar coordinates come along as constant columns, leading:

    >>> cft = cft.assign_coords(station="ST01")
    >>> xd.trigger(cft, thresh={"P": 0.5, "S": 0.5})
      station phase  time  value
    0    ST01     P   1.0    0.8
    1    ST01     S   1.0    0.9

    Picking a whole collection gives one table for the whole network. Each
    leaf's picks are labelled with the tree path they were found under, as
    they are found, and `merge` folds the tables on the way back up. Here the
    ``station`` column has two sources — the tree key and the leaf's own
    scalar coordinate — which agree, so it stays one column, leading:

    >>> dc = xd.DataCollection(
    ...     {
    ...         "DBNFM": cft.assign_coords(station="DBNFM"),
    ...         "LBFI": cft.assign_coords(station="LBFI"),
    ...     },
    ...     "station",
    ... )
    >>> xd.trigger(dc, thresh={"P": 0.5, "S": 0.5})
      station phase  time  value
    0   DBNFM     P   1.0    0.8
    1   DBNFM     S   1.0    0.9
    2    LBFI     P   1.0    0.8
    3    LBFI     S   1.0    0.9

    `merge=False` keeps the tree, each leaf already annotated:

    >>> tree = xd.trigger(..., thresh={"P": 0.5, "S": 0.5})(dc, merge=False)
    >>> tree["DBNFM"]
      station phase  time  value
    0   DBNFM     P   1.0    0.8
    1   DBNFM     S   1.0    0.9

    """

    def __init__(self, thresh, dim="time", coords="auto"):
        super().__init__()

        # parameters
        if isinstance(thresh, Mapping):
            self.thresh = {str(key): float(value) for key, value in thresh.items()}
        else:
            self.thresh = float(thresh)
        self.dim = str(dim)
        if coords is None or isinstance(coords, str):
            if isinstance(coords, str) and coords != "auto":
                raise ValueError(
                    f"`coords` must be 'auto', None or a sequence of "
                    f"coordinate names, got {coords!r}"
                )
            self.coords = coords
        else:
            self.coords = tuple(coords)

        # states
        self.axis = State(...)
        self.shape = State(...)
        self.thresh_on = State(...)
        self.thresh_off = State(...)
        self.status = State(...)
        self.index = State(...)
        self.value = State(...)
        self.offset = State(...)
        self.coord = State(...)
        self.annotations = State(...)

    def initialize(self, cft, **flags):
        """
        Initialize the trigger with the following states.

        - "axis": An integer indicating the axis number of the dimension along which to
          find picks.
        - "shape": A tuple indicating the unravel shape of the lanes along which
          the picks will be found.
        - "thresh_on"/"thresh_off": Float arrays holding the trigger on and off
          thresholds of each lane, raveled like the lanes.
        - "status": A boolean array indicating the trigger status for each lane.
        - "index": An integer array indicating the index of the last triggered value
          for each lane.
        - "value": A float array indicating the value of the last triggered value for
          each lane.
        - "offset": An integer indicating the offset of the chunk.
        - "coord": An InterpCoordinate containing coordinate information along 'dim' up
          to the last processed chunk.
        - "annotations": The resolved pick columns, one ``(name, axis, source)``
          triple each.


        Parameters
        ----------
        cft : DataArray
            The characteristic function where picks must be found.
        **flags
            Optional flags.

        """
        self.axis = State(cft.get_axis_num(self.dim))
        self.shape = State(cft.shape[: self.axis] + cft.shape[self.axis + 1 :])
        thresh_on = self._thresholds(cft)
        self.thresh_on = State(thresh_on)
        self.thresh_off = State(thresh_on / 2.0)
        self.status = State(np.zeros(self.shape, dtype=bool))
        self.index = State(np.zeros(self.shape, dtype=int))
        self.value = State(np.zeros(self.shape, dtype=float))
        self.offset = State(0)
        self.coord = State(Coordinate({"tie_indices": [], "tie_values": []}, self.dim))
        self.annotations = State(self._annotations(cft))
        self.labels = State({})

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
        A trigger that has not turned off by the end of the chunk stays open: its
        pick is emitted by the chunk that closes it, or by `flush` at the end of
        the run.

        Chunked along a dimension other than `dim`, none of that state carries:
        the next chunk holds *other* lanes, so its open triggers, its sample
        offset, its accumulated coordinate and the lane values annotating its
        picks all belong to the chunk that produced them. Such a chunk is a
        whole record on its own, run from a fresh state and closed here, which
        is what makes the cross-dimension exemption of the chunk-semantics gate
        true of this atom.

        """
        chunk_dim = flags.get("chunk_dim")
        independent = chunk_dim is not None and chunk_dim != self.dim
        if independent:
            self.initialize(cft, **flags)
        data = np.asarray(cft.values, dtype=float)
        values, indices = self._call_numeric(data)
        self.coord = concat_coords([self.coord, cft.coords[self.dim]], tolerance=None)
        self.labels = self._accumulate(cft)
        picks = self._picks(indices, values)
        if independent:
            return [picks] + self.flush()
        return picks

    def merge(self, results):
        """
        Fold the pick tables of a collection walk into one.

        A plain concatenation is all this takes: the walk already gave each
        table the columns of the tree path its leaf was reached by, so the
        rows carry their identity and nothing has to be reconstructed here.

        Parameters
        ----------
        results : sequence of DataFrame
            The per-leaf pick tables, in walk order. A leaf that produced no
            pick contributes an empty table, which concatenates away.

        Returns
        -------
        DataFrame
            The concatenated table, reindexed from zero. Leaves disagreeing
            on their columns — one carrying a scalar coordinate another does
            not — union them, the missing cells left empty. A collection
            without a single pick gives an empty table.

        """
        if not results:
            return pd.DataFrame()
        return pd.concat(results, ignore_index=True)

    def flush(self):
        """
        Close the triggers still open at the end of a run.

        `obspy.trigger_onset` closes whatever is on when the array ends, so the
        last pick of a record is not lost. Doing it here rather than at the end
        of each chunk keeps the result chunk-invariant: a run is closed once,
        whether it arrived in one piece or in twenty.

        Returns
        -------
        list of DataFrame
            One frame of the closed picks, or nothing if no trigger was open.

        """
        if not self.initialized:
            return []
        lanes = np.flatnonzero(np.reshape(self.status, (-1,)))
        if lanes.size == 0:
            return []
        values = np.reshape(self.value, (-1,))[lanes]
        indices = np.reshape(self.index, (-1,))[lanes]
        self.status = State(np.zeros(self.shape, dtype=bool))
        return [self._picks(self._unravel(lanes, indices), values)]

    def _thresholds(self, cft):
        """Resolve `thresh` into one trigger-on threshold per lane, raveled."""
        if not isinstance(self.thresh, dict):
            return np.full(self.shape, self.thresh, dtype=float).reshape(-1)
        if self.dim == PHASE_DIM:
            raise ValueError(
                f"cannot key thresholds on {PHASE_DIM!r}: it is the dimension "
                "the picks are found along"
            )
        if PHASE_DIM not in cft.dims or PHASE_DIM not in cft.coords:
            raise ValueError(
                f"a mapping of thresholds is keyed on the {PHASE_DIM!r} "
                f"coordinate, which the data to pick on does not have "
                f"(dimensions: {cft.dims})"
            )
        labels = [
            value.decode() if isinstance(value, bytes) else str(value)
            for value in cft.coords[PHASE_DIM].values
        ]
        unknown = [key for key in self.thresh if key not in labels]
        if unknown:
            raise KeyError(
                f"the threshold mapping names labels that are not in the "
                f"{PHASE_DIM!r} coordinate ({sorted(unknown)}); it holds {labels}"
            )
        # An unlisted label gets an infinite threshold: it can never trigger.
        values = np.array([self.thresh.get(label, np.inf) for label in labels])
        axis = cft.get_axis_num(PHASE_DIM)
        shape = [1] * len(self.shape)
        shape[axis - 1 if axis > self.axis else axis] = values.size
        return np.broadcast_to(values.reshape(shape), self.shape).reshape(-1).copy()

    def _annotations(self, cft):
        """
        Resolve the requested columns into ``(name, axis, source)`` triples.

        *source* is the coordinate the column is read from, or ``None`` when
        it is attached to the picked dimension: those are indexed by absolute
        sample number, so they are read off the coordinates accumulated over
        the run rather than off the chunk in hand.
        """
        annotations = []
        for name in self._names(cft):
            if name == self.dim:
                annotations.append((name, self.axis, None))
                continue
            coord = self._annotation(cft, name)
            if coord.dim is None:
                annotations.append((name, None, coord))
            elif coord.dim == self.dim:
                annotations.append((name, self.axis, None))
            else:
                annotations.append((name, cft.get_axis_num(coord.dim), coord))
        return tuple(annotations)

    def _names(self, cft):
        """
        Return the names of the columns annotating the picks, in order.

        Identity first, measurement last: the scalar coordinates lead, then
        the dimension coordinates that say *which* lane the pick was found in,
        then the picked dimension itself, then the value. So a pick table
        reads the same — ``network station location phase time value`` —
        whether its identity came from scalar coordinates on one array or
        from the tree path of a collection (which the walk puts in the same
        leading position), and whatever order the input's dimensions came in:
        a characteristic function laid out ``(distance, phase, time)`` — as
        :class:`~xdas.atoms.Annotate` emits it — and one laid out
        ``(time, distance, phase)`` give the same columns.
        """
        if self.coords is None:
            # A dimension without a coordinate has no labels to annotate with.
            return tuple(dim for dim in cft.dims if dim in cft.coords)
        if self.coords == "auto":
            scalars = tuple(
                name for name, coord in cft.coords.items() if coord.dim is None
            )
            dims = tuple(
                dim for dim in cft.dims if dim in cft.coords and dim != self.dim
            )
            picked = (self.dim,) if self.dim in cft.coords else ()
            return scalars + dims + picked
        return self.coords

    def _annotation(self, cft, name):
        """Return the coordinate *name* of *cft*, checked as a pick annotation."""
        if name not in cft.coords:
            raise KeyError(
                f"cannot annotate picks with {name!r}: it is not a coordinate "
                f"of the data to pick on (available: {sorted(cft.coords)})"
            )
        return cft.coords[name]

    def _accumulate(self, cft):
        """
        Extend the coordinates of the picked dimension with those of *cft*.

        A trigger reports the sample its onset was found at, which may lie in
        a chunk already gone: the labels of the picked dimension are kept for
        the whole run so that an absolute index still names something. The
        dimension coordinate itself is kept as a coordinate (`coord`), which
        stays compact; the others are plain arrays.
        """
        labels = dict(self.labels)
        for name, axis, source in self.annotations:
            if source is not None or name == self.dim:
                continue
            values = np.asarray(cft.coords[name].values)
            previous = labels.get(name)
            labels[name] = (
                values if previous is None else np.concatenate([previous, values])
            )
        labels[self.dim] = self.coord.values
        return labels

    def _picks(self, indices, values):
        """Build the pick table of the *values* found at *indices*."""
        picks = {}
        for name, axis, source in self.annotations:
            if source is None:
                picks[name] = self.labels[name][indices[axis]]
            elif axis is None:
                picks[name] = np.full(len(values), source.values, dtype=source.dtype)
            else:
                picks[name] = source[indices[axis]].values
        picks["value"] = values
        return pd.DataFrame(picks)

    def _unravel(self, lanes, indices):
        """Turn lane numbers and sample indices into one index array per axis."""
        coords = np.unravel_index(lanes, self.shape) if self.shape else ()
        return coords[: self.axis] + (indices,) + coords[self.axis :]

    def _call_numeric(self, data):
        """
        Find picks in a N-dimensional array along a given axis based on a given threshold.

        The pick findings use a triggering mechanism where triggers are turned on and off
        based on the threshold crossings. The trigger off threshold is half of the trigger
        on threshold. Picks are determined by finding the maximum value on each triggered
        region.

        Parameters
        ----------
        data : ndarray
            The characteristic function where picks must be found.

        Returns
        -------
        values : 1d ndarray
            The values of the picks.
        indices : tuple of 1d ndarray
            One index array per axis, locating each pick.

        Notes
        -----
        A trigger that has not turned off by the end of the array stays open; `flush`
        closes it at the end of the run.

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

        return values, self._unravel(lanes, indices)


@njit(
    "Tuple((i8[:], i8[:], f8[:]))(f8[:, :], f8[:], f8[:], b1[:], i8[:], f8[:], i8)",
    cache=True,
)
def _trigger(  # pragma: no cover
    cft, thresh_on, thresh_off, buffer_status, buffer_index, buffer_value, offset
):
    """
    Perform trigger detection on the input data.

    Parameters
    ----------
    cft : ndarray
        2D array of shape (n, m) representing the input data. Each row is a lane. Each
        column is the signal onto perform trigger detection.
    thresh_on : ndarray
        Float array of shape (n,) holding the threshold value for turning on the
        trigger of each lane. An infinite threshold never triggers.
    thresh_off : ndarray
        Float array of shape (n,) holding the threshold value for turning off the
        trigger of each lane.
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
            if value < thresh_off[lane]:
                buffer_status[lane] = False
                lanes.append(lane)
                indices.append(buffer_index[lane])
                values.append(buffer_value[lane])
        else:
            # `>=` so the onset matches ObsPy's trigger_onset on a sample that
            # sits exactly on the threshold
            if value >= thresh_on[lane]:
                buffer_status[lane] = True
                buffer_index[lane] = index
                buffer_value[lane] = value
    return np.array(lanes), np.array(indices), np.array(values)


trigger = atomized(Trigger)
