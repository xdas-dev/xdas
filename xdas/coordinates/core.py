"""
Core coordinate infrastructure.

Includes the :class:`Coordinates` container, :class:`Coordinate` factory/base
class, :class:`AxisCoordinate` (the axis-mapping ABC), and shared helpers used by
all concrete coordinate types (parsing, interpolation, tolerance handling).
"""

import weakref
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from functools import wraps
from itertools import pairwise

import numpy as np
import pandas as pd

#: Mapping from numpy datetime64/timedelta64 unit codes to CF-style unit names,
#: used to serialise timedelta scalars into dataset attributes.
CODE_TO_UNITS = {
    "h": "hours",
    "m": "minutes",
    "s": "seconds",
    "ms": "milliseconds",
    "us": "microseconds",
    "ns": "nanoseconds",
}
UNITS_TO_CODE = {v: k for k, v in CODE_TO_UNITS.items()}


def wraps_first_last(func):
    """Resolve ``"first"`` and ``"last"`` dim aliases before calling *func*."""

    @wraps(func)
    def wrapper(self, dim, *args, **kwargs):
        """Resolve ``"first"``/``"last"`` aliases then delegate to *func*."""
        if dim == "first":
            dim = self._dims[0]
        if dim == "last":
            dim = self._dims[-1]
        return func(self, dim, *args, **kwargs)

    return wrapper


def wraps_first_last_all(func):
    """Resolve ``"first"`` and ``"last"`` aliases in every positional argument."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        resolved = tuple(
            self._dims[0] if a == "first" else (self._dims[-1] if a == "last" else a)
            for a in args
        )
        return func(self, *resolved, **kwargs)

    return wrapper


class Coordinates(dict):
    """
    Dictionary like container for coordinates.

    Parameters
    ----------
    coords: dict-like, optional
        Mapping from coordinate names to any of the followings:

        - Coordinate objects
        - tuples (dim, coordinate-like) which can be either dimensional (`dim == name`)
          or non-dimensional (`dim != name` or `dim == None`).
        - coordinate-like objects (that are passed to the Coordinate constructor)
          which are assumed to be a dimensional coordinate with `dim` set to the
          related name.

    dims: sequence of str, optional
        An ordered sequence of dimensions. It is meant to match the dimensionality of
        its associated data. If provided, it must at least include all dimensions found
        in `coords` (extras dimensions will be considered as empty coordinates).
        Otherwise, dimensions will be guessed from `coords`.

    Examples
    --------
    >>> import xdas as xd

    >>> coords = {
    ...     "time": {"tie_indices": [0, 999], "tie_values": [0.0, 10.0]},
    ...     "distance": [0, 1, 2],
    ...     "channel": ("distance", ["DAS01", "DAS02", "DAS03"]),
    ...     "interrogator": (None, "SRN"),
    ... }
    >>> xd.Coordinates(coords)
    Coordinates:
      * time (time): 0.000 to 10.000
      * distance (distance): [0 ... 2]
        channel (distance): ['DAS01' ... 'DAS03']
        interrogator: 'SRN'
    """

    def __init__(self, coords=None, dims=None):
        super().__init__()
        if isinstance(coords, Coordinates):
            if dims is None:
                dims = coords.dims
            coords = dict(coords)
        self._dims = () if dims is None else tuple(dims)
        if coords is not None:
            for name in coords:
                self[name] = coords[name]

    @wraps_first_last
    def __getitem__(self, key):
        if key in self.dims and key not in self:
            raise KeyError(f"dimension {key} has no coordinate")
        return super().__getitem__(key)

    @wraps_first_last
    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("dimension names must be of type str")
        coord = Coordinate(value)
        if coord.dim is None and isinstance(coord, AxisCoordinate):
            coord.dim = key
        if self.parent is None:
            if coord.dim is not None and coord.dim not in self.dims:
                self._dims = self.dims + (coord.dim,)
        else:
            if coord.dim is not None:
                if coord.dim not in self.dims:
                    raise KeyError(
                        f"cannot add new dimension {coord.dim} to an existing DataArray"
                    )
                size = self.parent.sizes[coord.dim]
                if not len(coord) == size:
                    raise ValueError(
                        f"conflicting sizes for dimension {coord.dim}: size {len(coord)} "
                        f"in `coords` and size {size} in `data`"
                    )
        coord._assign_parent(self)
        return super().__setitem__(key, coord)

    def __repr__(self):
        lines = ["Coordinates:"]
        for name, coord in self.items():
            if self.isdim(name):
                lines.append(f"  * {name} ({coord.dim}): {coord}")
            else:
                if coord.dim is None:
                    lines.append(f"    {name}: {coord}")
                else:
                    lines.append(f"    {name} ({coord.dim}): {coord}")
        return "\n".join(lines)

    def __reduce__(self):
        return self.__class__, (dict(self), self.dims)

    @property
    def dims(self):
        """Ordered tuple of dimension names for this coordinates container."""
        return self._dims

    @property
    def parent(self):
        """The parent object (usually a :class:`DataArray`) this container is attached to."""
        if hasattr(self, "_parent"):
            return self._parent()
        else:
            return None

    def isdim(self, name):
        """Return ``True`` if *name* is a dimensional coordinate (i.e. its dim equals its name)."""
        return self[name].dim == name

    def _get_query(self, item):
        """
        Format a query from one or multiple indexer.

        Parameters
        ----------
        item: indexer-like or sequence or mapping
            Object to be parsed as a query. If item is indexer-like object, it is
            applied on the first dimension. If item is a sequence, positional indexing
            is performed. If item is a mapping, labeled indexing is performed.

        Returns
        -------
        dict:
            A mapping between each dim and a given indexer. If No indexer was found for
            a given dim, slice(None) will be used.
        """
        query = {dim: slice(None) for dim in self.dims}
        if isinstance(item, dict):
            if "first" in item:
                item[self.dims[0]] = item.pop("first")
            if "last" in item:
                item[self.dims[-1]] = item.pop("last")
            query.update(item)
        elif isinstance(item, tuple):
            for k in range(len(item)):
                query[self.dims[k]] = item[k]
        else:
            query[self.dims[0]] = item
        for dim, item in query.items():
            if isinstance(item, tuple):
                msg = f"cannot use tuple {item} to index dim '{dim}'"
                if len(item) == 2:
                    msg += f". Did you mean: {dim}=slice({item[0]}, {item[1]})?"
                raise TypeError(msg)
        return query

    def to_index(self, item, method=None, endpoint=True):
        """
        Convert an item selector to a dict of per-dimension integer indices.

        Parameters
        ----------
        item : indexer-like, sequence, or mapping
            Passed to :meth:`get_query` to resolve dimension-by-dimension indexers.
        method : str, optional
            Interpolation method forwarded to each coordinate's :meth:`~Coordinate.to_index`.
        endpoint : bool, optional
            Whether to include the stop endpoint of slice selectors. Default ``True``.

        Returns
        -------
        dict
            Mapping from dimension name to integer index or slice.
        """
        query = self._get_query(item)
        return {dim: self[dim].to_index(query[dim], method, endpoint) for dim in query}

    def equals(self, other):
        """Return ``True`` if *other* is a :class:`Coordinates` with identical coordinate values."""
        if not isinstance(other, Coordinates):
            return False
        if set(self) != set(other):
            return False
        for name in self:
            if not self[name].equals(other[name]):
                return False
        return True

    @classmethod
    def _from_dataset(cls, dataset, name):
        """Build a :class:`Coordinates` by delegating to each registered coordinate subclass."""
        return cls(Coordinate._from_dataset(dataset, name))

    def copy(self, deep=True):
        """Return a copy of this :class:`Coordinates` container.

        Parameters
        ----------
        deep : bool, optional
            If ``True`` (default) perform a deep copy of every coordinate.
        """
        return self.__class__(
            {key: value.copy(deep) for key, value in self.items()}, self.dims
        )

    @wraps_first_last_all
    def drop_dims(self, *dims):
        """Return a new :class:`Coordinates` with *dims* and their associated coordinates removed."""
        coords = {key: value for key, value in self.items() if value.dim not in dims}
        dims = tuple(value for value in self.dims if value not in dims)
        return self.__class__(coords, dims)

    @wraps_first_last_all
    def drop_coords(self, *names):
        """Return a new :class:`Coordinates` with the named coordinates removed."""
        coords = {key: value for key, value in self.items() if key not in names}
        return self.__class__(coords, self.dims)

    def _assign_parent(self, parent):
        """Attach this container to its parent, validating dimension counts and sizes."""
        if not len(self.dims) == parent.ndim:
            raise ValueError(
                f"inferred number of dimensions {len(self.dims)} from `coords` does "
                f"not match `data` dimensionality of {parent.ndim}"
            )
        for dim, size in zip(self.dims, parent.shape):
            if (dim in self) and (not len(self[dim]) == size):
                raise ValueError(
                    f"conflicting sizes for dimension {dim}: size {len(self[dim])} "
                    f"in `coords` and size {size} in `data`"
                )
        self._parent = weakref.ref(parent)


class Coordinate(ABC):
    """
    Base class and factory for all coordinate types.

    A coordinate attaches physical meaning to a :class:`DataArray`.  Two kinds
    exist:

    - **Axis coordinates** (:class:`AxisCoordinate` subclasses) map the integer
      positions of one array axis to physical values (e.g. timestamps,
      distances) and support index- and label-based selection.
    - **Scalar coordinates** (:class:`ScalarCoordinate`) carry a single value
      with no associated axis.

    This base class holds only what is genuinely shared between the two: the
    factory/registry machinery, identity/equality, copying, and (de)serialisation
    hooks.  The full axis-mapping contract lives on :class:`AxisCoordinate`.

    **Factory behaviour** — calling ``Coordinate(data)`` directly acts as a
    factory: it inspects *data* and returns an instance of the most suitable
    registered subclass (:class:`SampledCoordinate`, :class:`InterpCoordinate`,
    :class:`DenseCoordinate`, or :class:`ScalarCoordinate`).

    **Subclassing** — register a new subclass by passing ``ctype=`` in the
    class definition::

        class MyCoord(Coordinate, ctype="mycoord"):
            ...

    Parameters
    ----------
    data : array-like or mapping
        The coordinate data.  Interpretation is subclass-specific.
    dim : str, optional
        Name of the dimension this coordinate is associated with.
    dtype : dtype-like, optional
        Desired dtype for the underlying data array.
    """

    # --- class machinery ---

    _registry = {}

    def __init_subclass__(cls, *, ctype=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if ctype is not None:
            Coordinate._registry[ctype] = cls

    def __class_getitem__(cls, item):
        return cls._registry[item]

    def __new__(cls, data=None, dim=None, dtype=None):
        """Instantiate the appropriate Coordinate subclass based on *data*."""
        # class factory if instantiating Coordinate directly
        if cls is Coordinate:
            if data is None:
                raise TypeError("cannot infer coordinate type if no `data` is provided")

            data, dim = parse_data_dim(data, dim)

            for subcls in Coordinate._registry.values():
                if subcls._isvalid(data):
                    cls = subcls
                    break
            else:
                raise TypeError("could not parse `data`")

        # normal allocation
        return super().__new__(cls)

    # --- abstract contract ---

    @abstractmethod
    def __init__(self, data=None, dim=None, dtype=None):
        """Initialise the coordinate from subclass-specific *data*."""

    @property
    @abstractmethod
    def dtype(self):
        """NumPy dtype of the underlying coordinate values."""

    @staticmethod
    @abstractmethod
    def _isvalid(data):
        """Return ``True`` if *data* is a valid input for this coordinate subclass."""

    @property
    @abstractmethod
    def shape(self):
        """Shape tuple of the coordinate (``()`` for scalar, ``(len(self),)`` for axis)."""

    @abstractmethod
    def __array__(self, dtype=None, copy=None):
        """Materialise this coordinate as a numpy array (numpy array protocol)."""

    @abstractmethod
    def _to_dataset(self, dataset, attrs):
        """
        Serialise this coordinate into an xarray *dataset*, updating *attrs* in place.

        Parameters
        ----------
        dataset : xarray.Dataset
            Target dataset to write coordinate data into.
        attrs : dict
            Global attribute mapping to update (e.g. ``coordinate_interpolation``).

        Returns
        -------
        dataset : xarray.Dataset
        attrs : dict
        """

    @classmethod
    @abstractmethod
    def _collect_from_dataset(cls, dataset, name):
        """
        Extract coordinates of this subclass's type from *dataset* variable *name*.

        Parameters
        ----------
        dataset : xarray.Dataset
            Source dataset.
        name : str
            Name of the variable whose coordinates should be extracted.

        Returns
        -------
        dict
            Mapping from coordinate name to coordinate-like data, ready to be
            passed to :class:`Coordinate`.
        """

    # -- properties ---

    #: Name of the dimension this coordinate is associated with, or ``None``.
    dim = None

    @property
    def size(self):
        """Number of elements in this coordinate (``1`` for a scalar)."""
        return int(np.prod(self.shape))

    @property
    def values(self):
        """Materialised numpy array of coordinate values."""
        return self.__array__(copy=False)

    @property
    def parent(self):
        """The parent :class:`Coordinates` container, or ``None`` if unattached."""
        if hasattr(self, "_parent"):
            return self._parent()
        else:
            return None

    @property
    def name(self):
        """The name under which this coordinate is stored in its parent container."""
        if self.parent is None:
            return self.dim
        return next((name for name in self.parent if self.parent[name] is self), None)

    # --- dunders logic ---

    def __reduce__(self):
        return self.__class__, (self.data, self.dim)

    # --- queries ---

    def isdim(self):
        """Return ``True`` if this coordinate is a dimensional coordinate."""
        if self.parent is None or self.name is None:
            return None
        else:
            return self.parent.isdim(self.name)

    def equals(self, other):
        """Return ``True`` if *other* is the same coordinate type with identical dim and data.

        Comparison is strict on dtype. Same type implies same ``data`` structure:
        either a single ``np.ndarray`` or a flat ``dict[str, np.ndarray]`` with
        the same keys.
        """
        if type(self) is not type(other) or self.dim != other.dim:
            return False
        a, b = self.data, other.data
        if isinstance(a, dict):
            pairs = [(a[key], b[key]) for key in a]
        else:
            pairs = [(a, b)]
        for x, y in pairs:
            x, y = np.asarray(x), np.asarray(y)
            if x.dtype != y.dtype or not np.array_equal(x, y, equal_nan=False):
                return False
        return True

    # --- routines ---

    def copy(self, deep=True):
        """
        Return a copy of this coordinate.

        Parameters
        ----------
        deep : bool, optional
            If ``True`` (default) perform a deep copy; otherwise a shallow copy.

        Returns
        -------
        Coordinate
            A new coordinate of the same subclass with copied data and metadata.
        """
        if deep:
            func = deepcopy
        else:
            func = copy
        return self.__class__(func(self.data), func(self.dim), func(self.dtype))

    # --- IO ---

    @classmethod
    def _from_dataset(cls, dataset, name):
        """Read coordinates named *name* from an xarray *dataset* via each registered subclass."""
        coords = {}
        for subcls in cls._registry.values():
            coords |= subcls._collect_from_dataset(dataset, name)
        return coords

    # --- internals ---

    def _assign_parent(self, parent):
        """Attach this coordinate to its parent :class:`Coordinates` container."""
        self._parent = weakref.ref(parent)


class AxisCoordinate(Coordinate, ABC):
    """
    Base class for coordinates that map an array axis to physical values.

    Adds the full axis-mapping contract on top of :class:`Coordinate`: it
    supports two complementary directions of lookup:

    - **Index-based selection** — ``coord[i]`` or ``coord[start:stop]``:
      given integer position(s), return the corresponding physical value(s)
      as a new coordinate.
    - **Label-based selection** — ``coord.to_index(v)``: given a physical
      value (or slice of values), return the integer index (or slice) at
      that label.  An optional *method* argument controls nearest/forward/
      backward matching for values that fall between samples.  The returned
      index can then be passed to ``coord[idx]`` to retrieve the
      coordinate subset, and is also used internally to index into the
      parent data array.

    Concrete subclasses are :class:`DenseCoordinate`, :class:`InterpCoordinate`,
    and :class:`SampledCoordinate`.
    """

    # --- abstract contract ---

    @classmethod
    @abstractmethod
    def from_block(cls, start, size, step, dim=None, dtype=None):
        """
        Construct a coordinate from a start value, element count, and step size.

        Parameters
        ----------
        start : scalar
            Value of the first element.
        size : int
            Number of elements.
        step : scalar
            Spacing between consecutive elements.
        dim : str, optional
            Dimension name.
        dtype : dtype-like, optional
            Desired dtype for the coordinate values.

        Returns
        -------
        Coordinate
            A new coordinate instance of this subclass.
        """

    @abstractmethod
    def __len__(self):
        """Return the number of elements along this coordinate's axis."""

    @abstractmethod
    def _is_monotonic_increasing(self):
        """Return ``True`` if all consecutive differences in this coordinate are positive."""

    @abstractmethod
    def _get_value(self, index):
        """
        Return the coordinate value(s) at integer *index*.

        Parameters
        ----------
        index : int or numpy.ndarray of int
            Non-negative integer index or array of indices.

        Returns
        -------
        scalar or numpy.ndarray
            Coordinate value(s) at the requested position(s).
        """

    @abstractmethod
    def _get_indexer(self, value, method=None):
        """
        Return the integer index for label *value* using the segment structure.

        Parameters
        ----------
        value : scalar, str (ISO datetime), or array-like
            Label(s) to locate.
        method : {None, "nearest", "ffill", "bfill"}, optional
            How to handle values that fall in gaps or between samples.
            ``None`` (default) requires an exact match and raises ``KeyError``
            if the value is not present. ``"nearest"`` returns the index of
            the closest label. ``"ffill"`` (forward-fill) returns the last
            index whose label is less than or equal to *value*. ``"bfill"``
            (backward-fill) returns the first index whose label is greater
            than or equal to *value*.

        Returns
        -------
        int or numpy.ndarray

        Raises
        ------
        KeyError
            If *value* falls in an overlap region or is not found (exact mode).
        """

    @abstractmethod
    def _slice(self, slc):
        """
        Return a new coordinate covering the integer slice *slc*.

        Parameters
        ----------
        slc : slice
            Integer slice (already normalised by the caller).

        Returns
        -------
        Coordinate
            A new coordinate of the same subclass.
        """

    @abstractmethod
    def _concat(self, other):
        """
        Return a new coordinate formed by appending *other* after this one.

        Parameters
        ----------
        other : Coordinate
            Must be the same subclass and have the same ``dim`` and ``dtype``.

        Returns
        -------
        Coordinate
            Concatenated coordinate of the same subclass.
        """

    @abstractmethod
    def get_sampling_interval(self, cast=True):
        """
        Return the nominal sample spacing for this coordinate, or ``None``.

        Parameters
        ----------
        cast : bool, optional
            If ``True`` (default), cast timedelta64 results to seconds (float).

        Returns
        -------
        float or None
            ``None`` if the coordinate has fewer than two elements or has no
            defined sampling interval.
        """

    @abstractmethod
    def _split_candidates(self):
        """
        Return the candidate segment boundaries used by :meth:`get_split_indices`.

        Returns
        -------
        positions : numpy.ndarray
            Integer index of each candidate boundary. Each ``positions[k]``
            marks the start of a new segment (the boundary lies between element
            ``positions[k] - 1`` and ``positions[k]``).
        deltas : numpy.ndarray
            Signed jump at each candidate, i.e. the value step across the
            boundary minus the nominal sampling interval. Positive values are
            gaps, negative values are overlaps, zero means a clean continuation.
        """

    @abstractmethod
    def simplify(self, tolerance=None, *, reduce=True, regularize=False):
        """
        Return the simplest faithful copy of this coordinate within *tolerance*.

        A coordinate carries two independent, orthogonal properties:

        - **Monotonic** — tie values strictly increase, which is what makes
          label-based selection (:meth:`to_index`) work.
        - **Regular** — every continuous segment fits a single nominal
          ``sampling_interval``, which signal-processing routines (FFT,
          filtering, resampling) need for a clean sample rate.

        ``simplify`` spends an accuracy budget *tolerance* across two
        independently toggleable stages:

        - *reduce* drops redundant points whose removal shifts the curve by no
          more than *tolerance* (which also absorbs soft gaps and overlaps,
          helping monotonicity). Surviving values are never moved.
        - *regularize* promotes the coordinate to *regular* when the surviving
          continuous segments admit a single spacing within *tolerance*.

        Parameters
        ----------
        tolerance : float, timedelta, None, or ``False``, optional
            Accuracy budget; maximum allowed deviation from the original
            values.  ``None`` uses zero tolerance (lossless).  ``False``
            returns an unchanged copy regardless of the flags below.
        reduce : bool, optional
            Whether to drop redundant tie points. Default ``True``.
        regularize : bool, optional
            Whether to try to acquire a nominal ``sampling_interval``. Default
            ``False`` (opt-in). A no-op for coordinates that are already regular
            by construction or cannot become regular.

        Returns
        -------
        Coordinate
            A new coordinate of the same subclass.
        """

    # --- properties ---

    @property
    def ndim(self):
        """Number of dimensions (always 1 for dimensional coordinates)."""
        return 1

    @property
    def shape(self):
        """Shape tuple ``(len(self),)``."""
        return (len(self),)

    @property
    def empty(self):
        """``True`` if the coordinate has zero length."""
        return len(self) == 0

    @property
    def indices(self):
        """Integer array ``[0, 1, ..., len(self) - 1]``."""
        return np.arange(len(self))

    @property
    def start(self):
        """Value at index 0 (first element)."""
        return self._get_value(0)

    @property
    def end(self):
        """Value at the last element."""
        return self._get_value(len(self) - 1)

    # --- dunders logic ---

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self._slice(self._format_slice(item))
        else:
            item = self._format_index(item)
            return Coordinate(
                self._get_value(item), None if np.ndim(item) == 0 else self.dim
            )

    def __array__(self, dtype=None, copy=None):
        if self.empty:
            out = np.array([], dtype=self.dtype)
        else:
            out = self._get_value(self.indices)
        if dtype is not None:
            out = out.__array__(dtype)
        return out

    def __repr__(self):
        if self.empty:
            return "empty coordinate"
        elif len(self) == 1:
            return f"{self.tie_values[0]}"
        else:
            if np.issubdtype(self.dtype, np.floating):
                return f"{self.start:.3f} to {self.end:.3f}"
            elif np.issubdtype(self.dtype, np.datetime64):
                start_str = format_datetime(self.start)
                end_str = format_datetime(self.end)
                return f"{start_str} to {end_str}"
            else:
                return f"{self.start} to {self.end}"

    # --- queries ---

    def isregular(self):
        """Return ``True`` if this coordinate has a well-defined nominal sampling interval."""
        return self.get_sampling_interval() is not None

    def get_split_indices(self, kind="discontinuities", tolerance=False):
        """
        Return integer indices where this coordinate should be split.

        Each returned index ``i`` marks the start of a new segment: the
        boundary lies between element ``i - 1`` and element ``i``. The first
        segment always starts at index 0, so 0 is never included in the result.

        Parameters
        ----------
        kind : {"discontinuities", "gaps", "overlaps"}, optional
            Which boundary type to return. ``"gaps"`` returns only boundaries
            where the axis jumps forward by more than one sampling interval;
            ``"overlaps"`` returns only boundaries where the axis jumps
            backward. ``"discontinuities"`` (default) returns both.
        tolerance : float, timedelta, None, or ``False``, optional
            Minimum absolute magnitude of the jump to report. Boundaries
            smaller than *tolerance* are silently dropped. ``None`` removes
            only zero-magnitude jumps (i.e. consecutive equal values).
            ``False`` (default) disables magnitude filtering and returns all
            boundaries of the requested kind.

        Returns
        -------
        numpy.ndarray
            Integer indices of the start of each new segment (excluding the first).
        """
        valid_kinds = {"discontinuities", "gaps", "overlaps"}
        if kind not in valid_kinds:
            raise ValueError(f"`kind` must be one of {valid_kinds}; got {kind!r}")

        positions, deltas = self._split_candidates()

        # Fast path: every candidate boundary is a discontinuity by construction
        if kind == "discontinuities" and tolerance is False:
            return positions

        if tolerance is False:
            zero = np.timedelta64(0) if np.issubdtype(self.dtype, np.datetime64) else 0
            match kind:
                case "gaps":
                    mask = deltas >= zero
                case "overlaps":  # pragma: no branch
                    mask = deltas < zero
        else:
            tolerance = parse_scalar_delta(tolerance, self.dtype, default_zero=True)
            match kind:
                case "discontinuities":
                    mask = np.abs(deltas) > tolerance
                case "gaps":
                    mask = deltas > tolerance
                case "overlaps":  # pragma: no branch
                    mask = deltas < -tolerance

        return positions[mask]

    def get_discontinuities(self, tolerance=None):
        """
        Return a DataFrame containing information about the discontinuities.

        Parameters
        ----------
        tolerance : float, timedelta, or None, optional
            Minimum magnitude of a gap or overlap to include.  ``None``
            (default) reports all discontinuities regardless of size.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with the following columns:

            - start_index : int
                The index where the discontinuity starts.
            - end_index : int
                The index where the discontinuity ends.
            - start_value : float
                The value at the start of the discontinuity.
            - end_value : float
                The value at the end of the discontinuity.
            - delta : float
                The difference between the end_value and start_value.
            - type : str
                The type of the discontinuity, either "gap" or "overlap".

        """
        if self.empty:
            return pd.DataFrame(
                columns=[
                    "start_index",
                    "end_index",
                    "start_value",
                    "end_value",
                    "delta",
                    "type",
                ]
            )
        indices = self.get_split_indices("discontinuities", tolerance)
        records = []
        for index in indices:
            start_index = index
            end_index = index + 1
            start_value = self._get_value(index)
            end_value = self._get_value(index + 1)
            delta = end_value - start_value
            if tolerance is not None and np.abs(delta) < tolerance:
                continue
            record = {
                "start_index": start_index,
                "end_index": end_index,
                "start_value": start_value,
                "end_value": end_value,
                "delta": delta,
                "type": ("gap" if end_value > start_value else "overlap"),
            }
            records.append(record)
        return pd.DataFrame.from_records(records)

    def get_availabilities(self):
        """
        Return a DataFrame containing information about the data availability.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with the following columns:

            - start_index : int
                The index where the discontinuity starts.
            - end_index : int
                The index where the discontinuity ends.
            - start_value : float
                The value at the start of the discontinuity.
            - end_value : float
                The value at the end of the discontinuity.
            - delta : float
                The difference between the end_value and start_value.
            - type : str
                The type of the discontinuity, always "data".

        """
        if self.empty:
            return pd.DataFrame(
                columns=[
                    "start_index",
                    "end_index",
                    "start_value",
                    "end_value",
                    "delta",
                    "type",
                ]
            )
        indices = np.concatenate([[0], self.get_split_indices(), [len(self)]])
        records = []
        for start_index, stop_index in pairwise(indices):
            end_index = stop_index - 1
            start_value = self._get_value(start_index)
            end_value = self._get_value(end_index)
            records.append(
                {
                    "start_index": start_index,
                    "end_index": end_index,
                    "start_value": start_value,
                    "end_value": end_value,
                    "delta": end_value - start_value,
                    "type": "data",
                }
            )
        return pd.DataFrame.from_records(records)

    # --- selection / indexing ---

    def to_index(self, item, method=None, endpoint=True):
        """
        Convert a label-based selector to an integer index or slice.

        Parameters
        ----------
        item : label, slice, or array-like
            Selector to resolve.
        method : {None, "nearest", "ffill", "bfill"}, optional
            How to resolve *item* when it does not match a label exactly.
            ``None`` (default) requires an exact match. ``"nearest"`` selects
            the closest label. ``"ffill"`` selects the last label ≤ *item*;
            ``"bfill"`` selects the first label ≥ *item*. Ignored when *item*
            is a slice.
        endpoint : bool, optional
            Whether to include the stop of a slice. Default ``True``.

        Returns
        -------
        int, array of ints or slice
        """
        if isinstance(item, slice):
            return self._slice_indexer(item.start, item.stop, item.step, endpoint)
        else:
            return self._get_indexer(item, method)

    def _format_index(self, idx, bounds="raise"):
        """
        Normalise integer index *idx*, handling negative indices and optional bounds checking.

        Parameters
        ----------
        idx : int or array-like of int
            Index or indices to normalise.
        bounds : {"raise", "clip"}, optional
            ``"raise"`` (default) raises :exc:`IndexError` for out-of-bounds indices;
            ``"clip"`` clamps them to the valid range.

        Returns
        -------
        numpy.ndarray
            Non-negative integer index array.
        """
        idx = np.asarray(idx)
        if not np.issubdtype(idx.dtype, np.integer):
            raise IndexError("only integer are valid index")
        idx = idx + (idx < 0) * len(self)
        if bounds == "raise":
            if np.any(idx < 0) or np.any(idx >= len(self)):
                raise IndexError("index is out of bounds")
        elif bounds == "clip":
            idx = np.clip(idx, 0, len(self))
        return idx

    def _format_slice(self, slc):
        """
        Normalise *slc*, resolving ``None`` bounds, negative indices, and out-of-bounds.

        Parameters
        ----------
        slc : slice
            Raw slice, as received from user code.

        Returns
        -------
        slice
            Concrete ``slice(start, stop, step)`` with non-negative integer bounds
            clipped to ``[0, len(self)]``.
        """
        start, stop, step = slc.indices(len(self))
        if step < 0:
            raise NotImplementedError("negative slice step is not implemented")
        return slice(start, stop, step)

    def _slice_indexer(self, start=None, stop=None, step=None, endpoint=True):
        """
        Return an integer :class:`slice` corresponding to the label range [*start*, *stop*].

        Parameters
        ----------
        start : label, optional
            First label to include (inclusive, via ``"bfill"`` look-up).
        stop : label, optional
            Last label to include (inclusive by default, via ``"ffill"`` look-up).
        step : not supported
            Reserved; raises :exc:`NotImplementedError` if provided.
        endpoint : bool, optional
            If ``True`` (default), include *stop* in the result.

        Returns
        -------
        slice
        """
        if start is not None:
            try:
                start_index = self._get_indexer(start, method="bfill")
            except KeyError:
                start_index = len(self)
        else:
            start_index = None
        if stop is not None:
            try:
                end_index = self._get_indexer(stop, method="ffill")
                stop_index = end_index + 1
            except KeyError:
                stop_index = 0
        else:
            stop_index = None
        if step is not None:
            raise NotImplementedError("cannot use step yet")
        if (
            (not endpoint)
            and (stop is not None)
            and (self[stop_index - 1].values == stop)
        ):
            stop_index -= 1
        return slice(start_index, stop_index)

    # --- routines ---

    def to_dataarray(self):
        """Convert this coordinate to a :class:`~xdas.DataArray` with a single dimension."""
        from ..core import DataArray  # TODO: avoid deferred import?

        if self.name is None:
            raise ValueError("cannot convert unnamed coordinate to DataArray")

        if self.parent is None:
            return DataArray(
                self.values,
                {self.dim: self},
                dims=[self.dim],
                name=self.name,
            )
        else:
            return DataArray(
                self.values,
                {
                    name: coord
                    for name, coord in self.parent.items()
                    if coord.dim == self.dim
                },
                dims=[self.dim],
                name=self.name,
            )


def parse_data_dim(data, dim=None):
    """
    Normalise *data* / *dim* inputs accepted by coordinate constructors.

    Unpacks ``(dim, data)`` tuples and strips :class:`Coordinate` wrappers so
    that downstream constructors always receive a plain data object and an
    optional dimension string.

    Parameters
    ----------
    data : array-like, Coordinate, or (dim, array-like) tuple
        Raw coordinate input.
    dim : str, optional
        Explicit dimension name; overrides any dimension carried by *data*.

    Returns
    -------
    data : array-like
        Unwrapped data.
    dim : str or None
        Resolved dimension name.
    """
    if isinstance(data, tuple):
        if dim is None:
            dim, data = data
        else:
            _, data = data
    if isinstance(data, Coordinate):
        if dim is None:
            dim = data.dim
        data = data.data
    return data, dim


def parse_scalar_delta(value, dtype, default_zero=False):
    """
    Normalise a scalar *value* to the correct type for *dtype*.

    When ``default_zero`` is ``True``, a ``None`` input is replaced by a
    sensible zero-like default: ``timedelta64(0)`` for datetime64 dtypes,
    a dtype-appropriate epsilon for floating-point dtypes, or plain ``0``
    otherwise.  For datetime64 dtypes a numeric value is interpreted as
    seconds and converted to :class:`numpy.timedelta64`.

    Parameters
    ----------
    value : scalar or None
        Raw scalar value to normalise.
    dtype : numpy.dtype
        Target dtype that determines the output type.
    default_zero : bool, optional
        If ``True``, replace ``None`` with a zero-like default for *dtype*.
        Default is ``False``.

    Returns
    -------
    value : numpy scalar
        Normalised scalar cast to the appropriate numpy scalar type.

    Raises
    ------
    ValueError
        If *value* is not a scalar (i.e. has non-zero ndim), or if *value* is
        ``None`` while *default_zero* is ``False`` (no default is available).
    """
    # check shape
    if not np.ndim(value) == 0:
        raise ValueError("`value` must be a scalar value")

    # default
    if value is None:
        if not default_zero:
            raise ValueError("`value` cannot be None when `default_zero` is False")
        if np.issubdtype(dtype, np.datetime64):
            value = np.timedelta64(0)
        elif dtype == np.float16:
            value = 1e-2
        elif dtype == np.float32:
            value = 1e-5
        elif dtype == np.float64:
            value = 1e-8
        else:
            value = 0

    # ensure numpy scalar
    value = np.asarray(value)[()]

    # check dtype
    if np.issubdtype(dtype, np.datetime64):
        if not np.issubdtype(value.dtype, np.timedelta64):
            value = np.timedelta64(round(value * 1e9), "ns")  # TODO: not `dtype`
    else:
        value = value.astype(dtype)

    return value


def get_sampling_interval(da, dim, cast=True):
    """
    Return the sample spacing along a given dimension.

    Parameters
    ----------
    da : DataArray
        The data from which extract the sample spacing.
    dim : str
        The dimension along which get the sample spacing.
    cast: bool, optional
        Whether to cast datetime64 to seconds, by default True.

    Returns
    -------
    float
        The sample spacing.

    """
    coord = da[dim]
    if not isinstance(coord, AxisCoordinate):
        return None
    if coord.isregular():
        return coord.get_sampling_interval(cast=cast)
    if hasattr(coord, "_to_regular"):
        return coord._to_regular().get_sampling_interval(cast=cast)
    return coord.get_sampling_interval(cast=cast)


def encode_delta(key, value):
    """Serialise a scalar (possibly timedelta64) into a dict of dataset attributes."""
    if value is None:
        return {}
    if np.issubdtype(np.asarray(value).dtype, np.timedelta64):
        code, count = np.datetime_data(value.dtype)
        if code == "generic":  # e.g. timedelta64(0); promote to nanoseconds
            value = value.astype("timedelta64[ns]")
            code, count = np.datetime_data(value.dtype)
        return {
            key: int(count * value.astype(int)),
            f"{key}_dtype": "timedelta64[ns]",
            f"{key}_units": CODE_TO_UNITS[code],
        }
    return {key: value}


def decode_delta(key, attrs):
    """Inverse of :func:`encode_delta`: read a scalar back from dataset attributes."""
    if key not in attrs:
        return None
    value = attrs[key]
    if f"{key}_units" in attrs:
        value = np.timedelta64(value, UNITS_TO_CODE[attrs[f"{key}_units"]]).astype(
            attrs[f"{key}_dtype"]
        )
    return value


def is_monotonic_increasing(x):
    """Return ``True`` if every element of *x* is strictly greater than the previous one."""
    zero = np.timedelta64(0) if np.issubdtype(x.dtype, np.datetime64) else 0
    return np.all(np.diff(x) > zero)


def format_datetime(x):
    """Format a datetime64-like *x* as an ISO string, truncating sub-millisecond digits."""
    string = str(x)
    if "." in string:
        datetime, digits = string.split(".")
        digits = digits[:3]
        return ".".join([datetime, digits])
    else:
        return string
