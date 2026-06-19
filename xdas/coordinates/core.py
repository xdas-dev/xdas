"""
Core coordinate infrastructure.

Includes the :class:`Coordinates` container, :class:`Coordinate` factory/base
class, and shared helpers used by all concrete coordinate types (parsing,
interpolation, tolerance handling).
"""

import weakref
from abc import ABC, abstractmethod
from copy import copy, deepcopy
from functools import wraps
from itertools import pairwise

import numpy as np
import pandas as pd


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
        if coord.dim is None and not coord.isscalar():
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

    def get_query(self, item):
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
        query = self.get_query(item)
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
    def from_dataset(cls, dataset, name):
        """Build a :class:`Coordinates` by delegating to each registered coordinate subclass."""
        return cls(Coordinate.from_dataset(dataset, name))

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

    def assign_parent(self, parent):
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

    When called as ``Coordinate(data)``, acts as a factory and returns the first
    registered subclass who accepts *data*.  When subclassed, use the ``name=`` 
    keyword in the class definition to register the subclass (e.g. 
    ``class MyCoord(Coordinate, name="mycoord")``).

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

    def __init_subclass__(cls, *, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name is not None:
            Coordinate._registry[name] = cls

    def __class_getitem__(cls, item):
        return cls._registry[item]

    def __new__(cls, data=None, dim=None, dtype=None):
        """Instantiate the appropriate Coordinate subclass based on *data*."""
        # class factory if instantiating Coordinate directly
        if cls is Coordinate:
            if data is None:
                raise TypeError("cannot infer coordinate type if no `data` is provided")

            data, dim = parse(data, dim)

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

    @property
    @abstractmethod
    def dtype(self):
        """NumPy dtype of the underlying coordinate values."""

    @staticmethod
    @abstractmethod
    def _isvalid(data):
        """Return ``True`` if *data* is a valid input for this coordinate subclass."""

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
s
        """

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
    def ndim(self):
        """Number of dimensions (always 1 for dimensional coordinates)."""
        return 1

    @property
    def shape(self):
        """Shape tuple ``(len(self),)``."""
        return (len(self),)

    @property
    def size(self):
        """Number of elements along this coordinate's axis."""
        return len(self)

    @property
    def empty(self):
        """``True`` if the coordinate has zero length."""
        return len(self) == 0

    @property
    def indices(self):
        """Integer array ``[0, 1, ..., len(self) - 1]``."""
        return np.arange(len(self))

    @property
    def values(self):
        """Materialised numpy array of coordinate values."""
        return self.__array__(copy=False)

    @property
    def start(self):
        """Value at index 0 (first element)."""
        return self._get_value(0)

    @property
    def end(self):
        """Value at the last element."""
        return self._get_value(len(self) - 1)

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

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self._slice(self.format_slice(item))
        else:
            item = self.format_index(item)
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

    def __reduce__(self):
        return self.__class__, (self.data, self.dim)

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

    # -- queries ------------------------------------------------------------

    def isscalar(self):
        """Return ``True`` if this is a :class:`ScalarCoordinate`."""
        return False

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

    # -- selection / indexing -----------------------------------------------

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
            return self.slice_indexer(item.start, item.stop, item.step, endpoint)
        else:
            return self._get_indexer(item, method)

    def format_index(self, idx, bounds="raise"):
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

    def format_slice(self, slc):
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

    def slice_indexer(self, start=None, stop=None, step=None, endpoint=True):
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

    def to_dataarray(self):
        """Convert this coordinate to a :class:`~xdas.DataArray` with a single dimension."""
        from ..core.dataarray import DataArray  # TODO: avoid defered import?

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

    # --- IO ---

    @classmethod
    def from_dataset(cls, dataset, name):
        """Read coordinates named *name* from an xarray *dataset* via each registered subclass."""
        coords = {}
        for subcls in cls.__subclasses__():
            coords |= subcls._collect_from_dataset(dataset, name)
        return coords

    # --- internals ---

    def _assign_parent(self, parent):
        """Attach this coordinate to its parent :class:`Coordinates` container."""
        self._parent = weakref.ref(parent)


class SampledMixin(ABC):
    """
    Shared behaviour for coordinates that carry sampled values along an axis.

    Mixed into the tie-point coordinate types (:class:`SampledCoordinate`,
    :class:`InterpCoordinate`). Both types describe a piecewise-monotonic axis
    composed of contiguous segments separated by *gaps* (the axis jumps forward
    by more than one sampling interval) or *overlaps* (the axis jumps backward,
    creating doubly-covered regions). This mixin provides the shared logic for
    detecting, cataloguing, and querying those discontinuities.
    """

    @abstractmethod
    def get_sampling_interval(self, cast=True):
        """
        Return the nominal sample spacing for this coordinate.

        Parameters
        ----------
        cast : bool, optional
            If ``True`` (default), cast timedelta64 results to seconds (float).

        Returns
        -------
        float or None
            ``None`` if the coordinate has fewer than two elements.
        """

    @abstractmethod
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

    @abstractmethod
    def simplify(self, tolerance=None):
        """
        Return a simplified copy of this coordinate with redundant tie points removed.

        Tie points whose removal would shift any label by no more than *tolerance*
        are dropped, reducing memory and I/O cost without meaningfully changing
        the represented axis. As a side effect, small gaps or overlaps that fall
        within *tolerance* may be absorbed, merging adjacent segments into one.

        Parameters
        ----------
        tolerance : float, timedelta, None, or ``False``, optional
            Maximum allowed deviation from the original values.  ``None`` uses
            zero tolerance (lossless).  ``False`` returns an unchanged copy.

        Returns
        -------
        Coordinate
            A new coordinate of the same subclass with fewer stored points.
        """

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


def parse(data, dim=None):
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


def parse_tolerance(tolerance, dtype):
    """
    Normalise *tolerance* to the correct type for *dtype*.

    Converts ``None`` to zero, and for datetime64 dtypes converts a
    numeric tolerance (in seconds) to the appropriate :class:`numpy.timedelta64`.

    Parameters
    ----------
    tolerance : float or None
        Raw tolerance value.
    dtype : numpy.dtype
        The dtype of the coordinate values the tolerance will be compared against.

    Returns
    -------
    tolerance : int, float, or numpy.timedelta64
    """
    if np.issubdtype(dtype, np.datetime64):
        if tolerance is None:
            tolerance = np.timedelta64(0)
        elif isinstance(tolerance, (int, float)):
            tolerance = np.timedelta64(round(tolerance * 1e9), "ns")
    else:
        if tolerance is None:
            tolerance = 0
    return tolerance


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
    return da[dim].get_sampling_interval(cast=cast)


def isscalar(data):
    """Return ``True`` if *data* converts to a 0-d non-object numpy array."""
    data = np.asarray(data)
    return (data.dtype != np.dtype(object)) and (data.ndim == 0)


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
