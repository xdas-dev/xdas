"""
Core coordinate infrastructure: :class:`Coordinates` container,
:class:`Coordinate` factory/base class, and shared helpers used by all
concrete coordinate types (parsing, interpolation, tolerance handling).
"""

import weakref
from copy import copy, deepcopy
from functools import wraps
from itertools import pairwise

import numpy as np
import pandas as pd


def wraps_first_last(func):
    """Decorator that resolves ``"first"`` and ``"last"`` dim aliases before calling *func*."""

    @wraps(func)
    def wrapper(self, dim, *args, **kwargs):
        """Resolve ``"first"``/``"last"`` aliases then delegate to *func*."""
        if dim == "first":
            dim = self._dims[0]
        if dim == "last":
            dim = self._dims[-1]
        return func(self, dim, *args, **kwargs)

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

    dims: squence of str, optional
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
        for name in self:
            if not self[name].equals(other[name]):
                return False
        return True

    def to_dict(self):
        """Convert this `Coordinates` object into a pure python dictionnary.

        Examples
        --------

        >>> import xdas as xd

        >>> coords = xd.Coordinates(
        ...     {
        ...         "time": {"tie_indices": [0, 999], "tie_values": [0.0, 10.0]},
        ...         "distance": [0, 1, 2],
        ...         "channel": ("distance", ["DAS01", "DAS02", "DAS03"]),
        ...         "interrogator": (None, "SRN"),
        ...     }
        ... )
        >>> coords.to_dict()
        {'dims': ('time', 'distance'),
         'coords': {'time': {'dim': 'time',
           'data': {'tie_indices': [0, 999], 'tie_values': [0.0, 10.0]},
           'dtype': 'float64'},
          'distance': {'dim': 'distance', 'data': [0, 1, 2], 'dtype': 'int64'},
          'channel': {'dim': 'distance',
           'data': ['DAS01', 'DAS02', 'DAS03'],
           'dtype': '<U5'},
          'interrogator': {'dim': None, 'data': 'SRN', 'dtype': '<U3'}}}

        """
        return {
            "dims": self.dims,
            "coords": {name: self[name].to_dict() for name in self},
        }

    @classmethod
    def from_dict(cls, dct):
        """Reconstruct a :class:`Coordinates` from the dict returned by :meth:`to_dict`."""
        return cls(
            {key: Coordinate.from_dict(value) for key, value in dct["coords"].items()},
            dct["dims"],
        )

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
        return self.__class__({key: value.copy(deep) for key, value in self.items()})

    @wraps_first_last
    def drop_dims(self, *dims):
        """Return a new :class:`Coordinates` with *dims* and their associated coordinates removed."""
        coords = {key: value for key, value in self.items() if value.dim not in dims}
        dims = tuple(value for value in self.dims if value not in dims)
        return self.__class__(coords, dims)

    @wraps_first_last
    def drop_coords(self, *names):
        """Return a new :class:`Coordinates` with the named coordinates removed."""
        coords = {key: value for key, value in self.items() if key not in names}
        return self.__class__(coords, self.dims)

    def _assign_parent(self, parent):
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


class Coordinate:
    """
    Base class and factory for all coordinate types.

    When called as ``Coordinate(data)``, acts as a factory and returns the first
    registered subclass whose :meth:`isvalid` method accepts *data*.  When
    subclassed, use the ``name=`` keyword in the class definition to register
    the subclass (e.g. ``class MyCoord(Coordinate, name="mycoord")``).

    Concrete subclasses must implement :meth:`isvalid`, :meth:`equals`,
    and :meth:`to_dict` at minimum.

    Parameters
    ----------
    data : array-like or mapping
        The coordinate data.  Interpretation is subclass-specific.
    dim : str, optional
        Name of the dimension this coordinate is associated with.
    dtype : dtype-like, optional
        Desired dtype for the underlying data array.
    """

    _registry = {}

    def __init_subclass__(cls, *, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name is not None:
            Coordinate._registry[name] = cls

    def __class_getitem__(cls, item):
        return cls._registry[item]

    def __new__(cls, data=None, dim=None, dtype=None):
        # class factory if instantiating Coordinate directly
        if cls is Coordinate:
            if data is None:
                raise TypeError("cannot infer coordinate type if no `data` is provided")

            data, dim = parse(data, dim)

            for subcls in Coordinate._registry.values():
                if subcls.isvalid(data):
                    cls = subcls
                    break
            else:
                raise TypeError("could not parse `data`")

        # normal allocation
        return super().__new__(cls)

    def __getitem__(self, item):
        data = self.data.__getitem__(item)
        dim = None if isscalar(data) else self.dim
        return Coordinate(data, dim)

    def __len__(self):
        return self.data.__len__()

    def __repr__(self):
        return np.array2string(self.data, threshold=0, edgeitems=1)

    def __reduce__(self):
        return self.__class__, (self.data, self.dim)

    def __add__(self, other):
        return self.__class__(self.data + other, self.dim)

    def __sub__(self, other):
        return self.__class__(self.data - other, self.dim)

    def __array__(self, dtype=None):
        if dtype is None:
            return self.data.__array__()
        else:
            return self.data.__array__(dtype)

    def __array__ufunc__(self, ufunc, method, *inputs, **kwargs):
        return self.data.__array__ufunc__(ufunc, method, *inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        return self.data.__array_function__(func, types, args, kwargs)

    @staticmethod
    def isvalid(data):
        """Return ``True`` if *data* is a valid input for this coordinate subclass."""
        raise NotImplementedError

    @property
    def dtype(self):
        """NumPy dtype of the underlying data array."""
        return self.data.dtype

    @property
    def ndim(self):
        """Number of dimensions of the underlying data array (always 1 for dimensional coords)."""
        return self.data.ndim

    @property
    def shape(self):
        """Shape tuple of the underlying data array."""
        return self.data.shape

    @property
    def values(self):
        """Materialised numpy array of coordinate values."""
        return self.__array__()

    @property
    def empty(self):
        """``True`` if the coordinate has zero length."""
        return len(self) == 0

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

    def _assign_parent(self, parent):
        self._parent = weakref.ref(parent)

    def get_sampling_interval(self, cast=True):
        """
        Return the average sample spacing (end-to-end distance divided by N-1).

        Parameters
        ----------
        cast : bool, optional
            If ``True`` (default), cast timedelta64 results to seconds (float).

        Returns
        -------
        float or None
            ``None`` if the coordinate has fewer than two elements.
        """
        if len(self) < 2:
            return None
        delta = (self[-1].values - self[0].values) / (len(self) - 1)
        delta = np.asarray(delta)  # TODO: why?
        if cast and np.issubdtype(delta.dtype, np.timedelta64):
            delta = delta / np.timedelta64(1, "s")
        return delta

    def is_monotonic_increasing(self):
        """Return ``True`` if all consecutive differences in this coordinate are positive."""
        if np.issubdtype(self.dtype, np.datetime64):
            zero = np.timedelta64(0)
        else:
            zero = 0
        return np.all(np.diff(self.values) > zero)

    def isdim(self):
        """Return ``True`` if this coordinate is a dimensional coordinate in its parent container."""
        if self.parent is None or self.name is None:
            return None
        else:
            return self.parent.isdim(self.name)

    def copy(self, deep=True):
        """
        Return a copy of this coordinate.

        Parameters
        ----------
        deep : bool, optional
            If ``True`` (default) perform a deep copy; otherwise a shallow copy.
        """
        if deep:
            func = deepcopy
        else:
            func = copy
        return self.__class__(func(self.data), func(self.dim), func(self.dtype))

    def equals(self, other):
        """Return ``True`` if *other* represents the same coordinate values. Subclass must implement."""
        raise NotImplementedError

    def to_index(self, item, method=None, endpoint=True):
        """
        Convert a label-based selector to an integer index or slice.

        Parameters
        ----------
        item : label, slice, or array-like
            Selector to resolve.
        method : str, optional
            Look-up method (e.g. ``"ffill"``, ``"bfill"``).
        endpoint : bool, optional
            Whether to include the stop of a slice. Default ``True``.

        Returns
        -------
        int or slice
        """
        if isinstance(item, slice):
            return self.slice_indexer(item.start, item.stop, item.step, endpoint)
        else:
            return self.get_indexer(item, method)

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
                start_index = self.get_indexer(start, method="bfill")
            except KeyError:
                start_index = len(self)
        else:
            start_index = None
        if stop is not None:
            try:
                end_index = self.get_indexer(stop, method="ffill")
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

    def isscalar(self):
        """Return ``True`` if this is a :class:`ScalarCoordinate` (non-dimensional)."""
        return False

    def isdefault(self):
        """Return ``True`` if this is a :class:`DefaultCoordinate` (integer range)."""
        return False

    def isdense(self):
        """Return ``True`` if this is a :class:`DenseCoordinate` (explicit numpy array)."""
        return False

    def isinterp(self):
        """Return ``True`` if this is an :class:`InterpCoordinate` (piecewise-linear)."""
        return False

    def issampled(self):
        """Return ``True`` if this is a :class:`SampledCoordinate` (regularly sampled)."""
        return False

    def concat(self, other):
        """Concatenate *other* coordinate to this one. Subclass must implement."""
        raise NotImplementedError(f"concat is not implemented for {self.__class__}")

    def simplify(self, tolerance=None):
        """Reduce tie-point count within *tolerance*. Subclass must implement."""
        raise NotImplementedError(f"simplify is not implemented for {self.__class__}")

    def get_split_indices(self, kind="discontinuities", tolerance=False):
        """Return integer indices where this coordinate should be split. Subclass must implement."""
        raise NotImplementedError(
            f"get_split_indices is not implemented for {self.__class__}"
        )

    def get_discontinuities(self, tolerance=None):
        """
        Returns a DataFrame containing information about the discontinuities.

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
            start_value = self.get_value(index)
            end_value = self.get_value(index + 1)
            delta = end_value - start_value
            if tolerance is not None and np.abs(delta) < tolerance:
                continue
            record = {
                "start_index": start_index,
                "end_index": end_index,
                "start_value": start_value,
                "end_value": end_value,
                "delta": end_value - start_value,
                "type": ("gap" if end_value > start_value else "overlap"),
            }
            records.append(record)
        return pd.DataFrame.from_records(records)

    def get_availabilities(self):
        """
        Returns a DataFrame containing information about the data availability.

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
            start_value = self.get_value(start_index)
            end_value = self.get_value(end_index)
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

    def to_dict(self):
        """Serialise this coordinate to a plain-dict representation. Subclass must implement."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, dct):
        """Reconstruct a coordinate from the dict returned by :meth:`to_dict`."""
        return cls(**dct)

    def to_dataset(self, dataset, attrs):
        """Write this coordinate into an xarray *dataset*, updating *attrs* in place."""
        dataset = dataset.assign_coords(
            {self.name: (self.dim, self.values) if self.dim else self.values}
        )
        return dataset, attrs

    @classmethod
    def from_dataset(cls, dataset, name):
        """Read coordinates named *name* from an xarray *dataset* via each registered subclass."""
        coords = {}
        for subcls in cls.__subclasses__():
            if hasattr(subcls, "from_dataset"):
                coords |= subcls.from_dataset(dataset, name)
        return coords

    @classmethod
    def from_block(cls, start, size, step, dim=None, dtype=None):
        """Construct a coordinate from a start value, element count, and step size. Subclass must implement."""
        raise NotImplementedError


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
    Returns the sample spacing along a given dimension.

    Parameters
    ----------
    da : DataArray or DataArray or DataArray
        The data from which extract the sample spacing.
    dim : str
        The dimension along which get the sample spacing.
    cast: bool, optional
        Wether to cast datetime64 to seconds, by default True.

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
    if np.issubdtype(x.dtype, np.datetime64):
        return np.all(np.diff(x) > np.timedelta64(0))
    else:
        return np.all(np.diff(x) > 0)


def format_datetime(x):
    """Format a datetime64-like *x* as an ISO string, truncating sub-millisecond digits."""
    string = str(x)
    if "." in string:
        datetime, digits = string.split(".")
        digits = digits[:3]
        return ".".join([datetime, digits])
    else:
        return string
