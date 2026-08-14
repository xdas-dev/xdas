"""
Nested tree structures for grouping multiple :class:`DataArray` objects.

Includes :class:`DataCollection`, :class:`DataSequence`, and
:class:`DataMapping`.
"""

from fnmatch import fnmatch
from pathlib import Path

import h5py
import pandas as pd

from .dataarray import DataArray

#: printed where a branch has no key at all for a depth, as opposed to the
#: blank that repeats the key of the row above
ABSENT = "-"


class DataCollection:
    """
    Nested collection of dataarray.

    Parameters
    ----------
    data: list or dict of DataCollection or DataArray
        The nested data. It can be composed either of sequences or mapping. The
        leaves must be dataarrays.
    name: str
        The name of the current level of nesting.

    Returns
    -------
    DataCollection:
        The nested data as a DataSequence or DataMapping.

    Examples
    --------
    >>> import xdas as xd
    >>> from xdas.synthetics import wavelet_wavefronts
    >>> da = wavelet_wavefronts()
    >>> dc = xd.DataCollection(
    ...     {
    ...         "das1": ("record", [da, da]),
    ...         "das2": ("record", [da, da, da]),
    ...     },
    ...     "instrument",
    ... )
    >>> dc
    <xdas.DataCollection: 5 leaves, 4.6 MB>
    instrument  record
    das1             0  (time: 300, distance: 401)  939.8 KB
                     1  (time: 300, distance: 401)  939.8 KB
    das2             0  (time: 300, distance: 401)  939.8 KB
                     1  (time: 300, distance: 401)  939.8 KB
                     2  (time: 300, distance: 401)  939.8 KB

    """

    def __new__(cls, data, name=None):
        """Dispatch to :class:`DataSequence` or :class:`DataMapping` based on *data* type."""
        data, name = parse(data, name)
        if isinstance(data, list):
            return list.__new__(DataSequence)
        elif isinstance(data, dict):
            return dict.__new__(DataMapping)
        elif isinstance(data, DataArray):
            if name is not None:
                data = data.rename(name)
            return data
        elif isinstance(data, pd.DataFrame):
            # A table is a leaf of its own kind: atoms emitting pick tables
            # walk collections like any other, and coercing their result into
            # a `DataArray` would silently destroy it.
            return data
        else:
            return DataArray(data, name=name)

    @property
    def empty(self):
        """``True`` if the collection contains no elements."""
        return len(self) == 0

    @property
    def fields(self):
        """Ordered, deduplicated tuple of the node names of the whole subtree."""
        values = self.values() if self.ismapping() else self
        out = (self.name,) + tuple(
            name
            for value in values
            if isinstance(value, DataCollection)
            for name in value.fields
        )
        return uniquifiy(out)

    def query(self, indexers=None, **indexers_kwargs):
        """
        Query a given subset from a data collection.

        The data collection is walked through, if any node name corresponds to a key of
        the `indexers`, the corresponding value is used to select a subset of that node.

        Each indexer must name a level of the collection, i.e. be one of `fields`.
        This is what distinguishes querying from `sel`: `query` chooses *which*
        leaves are kept by their position in the hierarchy, while `sel` trims
        *inside* each leaf by coordinate label.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching fields and values given by string or int.
        **indexers_kwargs : dict, optional
            The keyword arguments form of indexers. Overwrite indexers input if both
            are provided.

        Returns
        -------
        DataCollection:
            The queried data.

        Raises
        ------
        KeyError
            If an indexer does not name any level of the collection.

        Examples
        --------
        >>> import xdas as xd
        >>> from xdas.synthetics import wavelet_wavefronts
        >>> da = wavelet_wavefronts()
        >>> dc = xd.DataCollection(
        ...     {
        ...         "das1": ("record", [da, da]),
        ...         "das2": ("record", [da, da, da]),
        ...     },
        ...     "instrument",
        ... )
        >>> dc.query(instrument="das1", record=0)
        <xdas.DataCollection: 1 leaf, 939.8 KB>
        instrument  record
        das1             0  (time: 300, distance: 401)  939.8 KB

        """
        indexers = {} if indexers is None else dict(indexers)
        indexers.update(indexers_kwargs)
        fields = self.fields
        unknown = [key for key in indexers if key not in fields]
        if unknown:
            raise KeyError(
                f"{unknown} do not name any level of the collection; "
                f"available: {list(fields)}"
            )
        return self._query(indexers)

    def select(self, indexers=None, **indexers_kwargs):
        """
        Select a given subset from a data collection.

        Alias of `query`, named after `obspy.Stream.select`. See `query`.
        """
        return self.query(indexers, **indexers_kwargs)

    def _query(self, indexers):
        """Recursive half of `query`, with the indexers already validated.

        Every level is walked, whether or not it is named in *indexers*: an
        indexer applies wherever its level sits in the tree, not only at the
        root.
        """
        key = indexers.get(self.name)
        if self.issequence():
            data = list(self)
            if self.name in indexers:
                if isinstance(key, int):
                    data = [data[key]]
                elif isinstance(key, slice):
                    data = data[key]
                else:
                    raise ValueError(f"{self.name} query must be an integer or a slice")
            data = [
                (value._query(indexers) if isinstance(value, DataCollection) else value)
                for value in data
            ]
        elif self.ismapping():
            data = dict(self)
            if self.name in indexers:
                if isinstance(key, str):
                    data = {
                        name: value
                        for name, value in data.items()
                        if fnmatch(name, key)
                    }
                else:
                    raise ValueError(f"{self.name} query must be a string")
            data = {
                name: (
                    value._query(indexers)
                    if isinstance(value, DataCollection)
                    else value
                )
                for name, value in data.items()
            }
        else:  # pragma: no cover
            raise TypeError("unknown type of data collection")
        return DataCollection(data, self.name)

    def issequence(self):
        """Return ``True`` if this is a :class:`DataSequence`."""
        return isinstance(self, DataSequence)

    def ismapping(self):
        """Return ``True`` if this is a :class:`DataMapping`."""
        return isinstance(self, DataMapping)

    @classmethod
    def from_netcdf(cls, fname, group=None):
        """
        Lazily read a data collection from a NetCDF file.

        Parameters
        ----------
        fname: str
            The path of the file to open.
        group: str, optional
            The location of the data collection within the file. Root by default.

        Returns
        -------
        DataCollection:
            The opened data collection.

        """
        if isinstance(fname, Path):
            fname = str(fname)
        return as_sequence_if_positional(DataMapping.from_netcdf(fname, group))


class DataMapping(DataCollection, dict):
    """
    A Mapping of dataarrays.

    A data mapping is a dictionary whose keys are any user defined identifiers and
    values are dataarray objects.
    """

    def __new__(cls, data, name=None):
        """Allocate a new dict-backed DataMapping instance."""
        return dict.__new__(cls)

    def __init__(self, data, name=None):
        data, name = parse(data, name)
        data = {
            key: (value if isinstance(value, DataCollection) else DataCollection(value))
            for key, value in data.items()
        }
        dict.__init__(self, data)
        self.name = name

    def __repr__(self):
        return format_collection(self)

    def __reduce__(self):
        return self.__class__, (dict(self), self.name)

    def to_netcdf(
        self,
        fname,
        mode="w",
        group=None,
        virtual=None,
        encoding=None,
        create_dirs=False,
    ):
        """Write this :class:`DataMapping` to a NetCDF file (see :func:`~xdas.io.xdas.save_datamapping`)."""
        from ..io.xdas import save_datamapping

        save_datamapping(self, fname, mode, group, virtual, encoding, create_dirs)

    @classmethod
    def from_netcdf(cls, fname, group=None):
        """Lazily read a :class:`DataMapping` from a NetCDF file (see :func:`~xdas.io.xdas.open_datamapping`)."""
        from ..io.xdas import open_datamapping

        return open_datamapping(fname, group)

    def equals(self, other):
        """Return ``True`` if *other* is a :class:`DataMapping` with identical keys and values."""
        if not isinstance(other, self.__class__):
            return False
        if self.name != other.name:
            return False
        if list(self.keys()) != list(other.keys()):
            return False
        return all(self[key].equals(other[key]) for key in self)

    def isel(self, indexers=None, **indexers_kwargs):
        """
        Perform index selection to each data array of the data collection.

        If a selection results in a empty data array, the data array is discarded.

        See `DataArray.isel` for more details.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given by integers, slice
            objects or arrays.
        **indexers_kwargs : dict, optional
            The keyword arguments form of integers. Overwrite indexers input if both
            are provided.

        Returns
        -------
        DataCollection
            The selected data collection.
        """
        data = {
            key: value.isel(indexers, **indexers_kwargs) for key, value in self.items()
        }
        data = {
            key: value
            for key, value in data.items()
            if (isinstance(value, DataCollection) or not value.empty)
        }
        return self.__class__(data, self.name)

    def sel(self, indexers=None, method=None, endpoint=True, **indexers_kwargs):
        """
        Perform labeled selection to each data array of the data collection.

        If a selection results in a empty data array, the data array is discarded.

        See DataArray.sel for more details.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given by scalars, slices or
            arrays of tick labels.
        method : str, optional
            Method to use for inexact matches. None (default) means only exact matches.
        endpoint : bool, optional
            Whether to include the endpoint of a slice. Default is True.
        **indexers_kwargs : dict, optional
            The keyword arguments form of integers. Overwrite indexers input if both
            are provided.

        Returns
        -------
        DataCollection
            The selected data collection.

        """
        data = {
            key: value.sel(indexers, method, endpoint, **indexers_kwargs)
            for key, value in self.items()
        }
        data = {
            key: value
            for key, value in data.items()
            if (isinstance(value, DataCollection) or not value.empty)
        }
        return self.__class__(data, self.name)

    def load(self):
        """
        Load in memory each data array of the data collection.

        See `DataArray.load` for more details

        Returns
        -------
        The loaded data collection.

        """
        data = {key: value.load() for key, value in self.items()}
        return self.__class__(data, self.name)

    def map(self, atom):
        """
        Apply an atom to each data array of the data collection.

        Parameters
        ----------
        atom: Atom or callable
            The atom to apply, i.e, a function that takes a unique data array argument
            and returns a unique data array output.

        Returns
        -------
        DataCollection
            Resulting processed data collection.

        """
        data = {}
        for key, obj in self.items():
            if isinstance(obj, DataArray):
                data[key] = atom(obj)
            elif isinstance(obj, DataCollection):
                data[key] = obj.map(atom)
            else:
                raise TypeError(f"{type(obj)} encountered in the collection")
        return self.__class__(data, self.name)

    def copy(self, deep=True):
        """
        Return a copy of the data collection.

        Parameters
        ----------
        deep: bool, optional
            If True, a deep copy is returned. If False, a shallow copy is returned.

        Returns
        -------
        DataCollection:
            The copied data collection.

        """
        return self.__class__(
            {key: value.copy() for key, value in self.items()}, self.name
        )


class DataSequence(DataCollection, list):
    """
    A collection of dataarrays.

    A data sequence is a list whose values are dataarray objects.
    """

    def __new__(cls, data, name=None):
        """Allocate a new list-backed DataSequence instance."""
        return list.__new__(cls)

    def __init__(self, data, name=None):
        data, name = parse(data, name)
        data = [
            (value if isinstance(value, DataCollection) else DataCollection(value))
            for value in data
        ]
        list.__init__(self, data)
        self.name = name

    def __repr__(self):
        return format_collection(self)

    def __reduce__(self):
        return self.__class__, (list(self), self.name)

    def to_mapping(self):
        """Convert to an integer-keyed :class:`DataMapping`."""
        return DataMapping(dict(enumerate(self)), self.name)

    @classmethod
    def from_mapping(cls, data):
        """Build a :class:`DataSequence` from the values of a :class:`DataMapping`."""
        return cls(data.values(), data.name)

    def to_netcdf(
        self,
        fname,
        mode="w",
        group=None,
        virtual=None,
        encoding=None,
        create_dirs=False,
    ):
        """Write this :class:`DataSequence` to a NetCDF file by converting to a mapping first."""
        self.to_mapping().to_netcdf(
            fname,
            mode=mode,
            group=group,
            virtual=virtual,
            encoding=encoding,
            create_dirs=create_dirs,
        )

    @classmethod
    def from_netcdf(cls, fname, group=None):
        """Lazily read a :class:`DataSequence` from a NetCDF file."""
        return cls.from_mapping(DataMapping.from_netcdf(fname, group))

    def equals(self, other):
        """Return ``True`` if *other* is a :class:`DataSequence` with identical elements."""
        if not isinstance(other, self.__class__):
            return False
        if self.name != other.name:
            return False
        if len(self) != len(other):
            return False
        return all(a.equals(b) for a, b in zip(self, other))

    def isel(self, indexers=None, **indexers_kwargs):
        """
        Perform index selection to each data array of the data collection.

        If a selection results in a empty data array, the data array is discarded.

        See `DataArray.isel` for more details.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given by integers, slice
            objects or arrays.
        **indexers_kwargs : dict, optional
            The keyword arguments form of integers. Overwrite indexers input if both
            are provided.

        Returns
        -------
        DataCollection
            The selected data collection.
        """
        data = [value.isel(indexers, **indexers_kwargs) for value in self]
        data = [
            value
            for value in data
            if (isinstance(value, DataCollection) or not value.empty)
        ]
        return self.__class__(data, self.name)

    def sel(self, indexers=None, method=None, endpoint=True, **indexers_kwargs):
        """
        Perform labeled selection to each data array of the data collection.

        If a selection results in a empty data array, the data array is discarded.

        See DataArray.sel for more details.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given by scalars, slices or
            arrays of tick labels.
        method : str, optional
            Method to use for inexact matches. None (default) means only exact matches.
        endpoint : bool, optional
            Whether to include the endpoint of a slice. Default is True.
        **indexers_kwargs : dict, optional
            The keyword arguments form of integers. Overwrite indexers input if both
            are provided.

        Returns
        -------
        DataCollection
            The selected data collection.

        """
        data = [
            value.sel(indexers, method, endpoint, **indexers_kwargs) for value in self
        ]
        data = [
            value
            for value in data
            if (isinstance(value, DataCollection) or not value.empty)
        ]
        return self.__class__(data, self.name)

    def load(self):
        """
        Load in memory each data array of the data collection.

        See `DataArray.load` for more details

        Returns
        -------
        The loaded data collection.

        """
        data = [value.load() for value in self]
        return self.__class__(data, self.name)

    def map(self, atom):
        """
        Apply an atom to each data array of the data collection.

        Parameters
        ----------
        atom: Atom or callable
            The atom to apply, i.e, a function that takes a unique data array argument
            and returns a unique data array output.

        Returns
        -------
        DataCollection
            Resulting processed data collection.

        """
        data = []
        for obj in self:
            if isinstance(obj, DataArray):
                data.append(atom(obj))
            elif isinstance(obj, DataCollection):
                data.append(obj.map(atom))
            else:
                raise TypeError(f"{type(obj)} encountered in the collection")
        return self.__class__(data, self.name)

    def copy(self, deep=True):
        """
        Return a copy of the data collection.

        Parameters
        ----------
        deep: bool, optional
            If True, a deep copy is returned. If False, a shallow copy is returned.

        Returns
        -------
        DataCollection:
            The copied data collection.

        """
        return self.__class__([value.copy() for value in self], self.name)


def format_collection(dc):
    """
    Render a data collection as a flat table, one row per leaf.

    Each row spells out the keys that address a leaf, then that leaf's shape
    and the memory it takes once loaded. Repeated keys are blanked so that a
    row shows only what changed from the one above.

    Parameters
    ----------
    dc : DataCollection
        The collection to render.

    Returns
    -------
    str
        The representation, starting with a summary line.

    """
    rows = get_leaves(dc)
    header = (
        f"<xdas.DataCollection: {len(rows)} {'leaf' if len(rows) == 1 else 'leaves'}, "
        f"{to_human(sum(get_nbytes(leaf) for _, leaf in rows))}>"
    )
    if not rows:
        return header

    keys = [[key for _, key in path] for path, _ in rows]
    ncolumns = max(len(row) for row in keys)
    # a column is a depth, so a branch that stops short has no key to put in
    # the deeper ones and says so, rather than leaving a blank that would read
    # as "as in the row above"
    grid = [row + [ABSENT] * (ncolumns - len(row)) for row in keys]
    body = [
        [
            (
                ""
                if index
                and key != ABSENT
                and all(grid[index][j] == grid[index - 1][j] for j in range(d + 1))
                else key
            )
            for d, key in enumerate(row)
        ]
        + [get_shape(leaf), to_human(get_nbytes(leaf))]
        for index, (row, (_, leaf)) in enumerate(zip(grid, rows))
    ]

    fields = get_fields(rows)
    heads = (fields if fields else [""] * ncolumns) + ["", ""]
    # positional keys and the memory column are right aligned so that they
    # can be compared down the column
    columns = zip(*[row + cells[-2:] for row, cells in zip(grid, body)])
    right = [
        all(cell.isdigit() or cell == ABSENT or not cell for cell in column)
        for column in columns
    ]
    right[-1] = True
    widths = [
        max(len(heads[d]), *(len(row[d]) for row in body)) for d in range(len(heads))
    ]

    lines = [header]
    if any(heads):
        lines.append(format_row(heads, widths, right))
    lines.extend(format_row(row, widths, right) for row in body)
    return "\n".join(lines)


def format_row(cells, widths, right):
    """Pad *cells* to *widths*, right justifying those flagged in *right*."""
    return "  ".join(
        cell.rjust(width) if flag else cell.ljust(width)
        for cell, width, flag in zip(cells, widths, right)
    ).rstrip()


def get_leaves(dc):
    """Return one ``(path, leaf)`` per leaf, *path* being ``[(field, key), ...]``."""
    leaves = []

    def walk(node, path):
        items = enumerate(node) if node.issequence() else node.items()
        for key, value in items:
            row = path + [(node.name, str(key))]
            if isinstance(value, DataCollection):
                walk(value, row)
            else:
                leaves.append((row, value))

    walk(dc, [])
    return leaves


def get_fields(rows):
    """Return the name of each depth, empty where the branches disagree.

    A field name belongs to a node, not to a depth: two branches can name
    the same depth differently or not at all, and only some of them reach
    the deepest ones. Such a depth is left unheaded rather than headed with
    a name that speaks for one branch only, but the depths that do agree are
    still worth naming.
    """
    ncolumns = max(len(path) for path, _ in rows)
    heads = []
    for depth in range(ncolumns):
        found = {path[depth][0] for path, _ in rows if depth < len(path)}
        name = next(iter(found)) if len(found) == 1 else None
        heads.append(name if name is not None else "")
    return heads


def get_shape(leaf):
    """Describe the extent of *leaf*, naming its axes."""
    if isinstance(leaf, pd.DataFrame):
        return f"(rows: {len(leaf)}, columns: {len(leaf.columns)})"
    sizes = ", ".join(f"{dim}: {size}" for dim, size in leaf.sizes.items())
    return f"({sizes})"


def get_nbytes(leaf):
    """Return what *leaf* occupies in memory once loaded.

    `nbytes` covers every array backing, virtual sources included, and gives
    the loaded size without reading anything. A dataframe has no such
    attribute: it is measured shallowly and without its index, which is O(1)
    and counts data but not labels, as `DataArray.nbytes` does.
    """
    if hasattr(leaf, "nbytes"):
        return leaf.nbytes
    return int(leaf.memory_usage(index=False, deep=False).sum())


def to_human(nbytes):
    """Format a byte count with the largest unit that keeps it above one."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} B"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def parse(data, name=None):
    """
    Normalise *(data, name)* inputs accepted by :class:`DataCollection` constructors.

    Unpacks ``(name, data)`` tuples and propagates the name from an existing
    :class:`DataCollection` when no explicit name is given.
    """
    if isinstance(data, tuple):
        if name is None:
            name, data = data
        else:
            _, data = data
    if isinstance(data, DataCollection) and name is None:
        name = data.name
    return data, name


def as_sequence_if_positional(dm):
    """Return :class:`DataMapping` *dm* as a sequence if its keys are its positions.

    A sequence is written under the canonical decimal spelling of its
    positions, so that is what is compared: parsing the keys as integers
    instead would read a mapping keyed by a zero-padded code — a SEED
    location, say — back as a sequence, losing the keys.
    """
    if list(dm) == [str(index) for index in range(len(dm))]:
        return DataSequence.from_mapping(dm)
    return dm


def get_depth(group):
    """Return the maximum nesting depth of an HDF5 *group* by counting ``"/"`` separators."""
    if not isinstance(group, h5py.Group):
        raise ValueError("not a group")
    depths = []
    group.visit(lambda name: depths.append(name.count("/")))
    return max(depths)


def uniquifiy(seq):
    """Return a deduplicated tuple of *seq* elements in their original order."""
    seen = set()
    return tuple(x for x in seq if x not in seen and not seen.add(x))
