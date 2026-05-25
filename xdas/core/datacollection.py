"""
:class:`DataCollection`, :class:`DataSequence`, and :class:`DataMapping`:
nested tree structures for grouping multiple :class:`DataArray` objects.
"""

from fnmatch import fnmatch
from pathlib import Path

import h5py

from .dataarray import DataArray


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

    Returns:
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
    ...         "das1": ("acquisition", [da, da]),
    ...         "das2": ("acquisition", [da, da, da]),
    ...     },
    ...     "instrument",
    ... )
    >>> dc
    Instrument:
      das1:
        Acquisition:
          0: <xdas.DataArray (time: 300, distance: 401)>
          1: <xdas.DataArray (time: 300, distance: 401)>
      das2:
        Acquisition:
          0: <xdas.DataArray (time: 300, distance: 401)>
          1: <xdas.DataArray (time: 300, distance: 401)>
          2: <xdas.DataArray (time: 300, distance: 401)>

    """

    def __new__(cls, data, name=None):
        data, name = parse(data, name)
        if isinstance(data, list):
            return list.__new__(DataSequence)
        elif isinstance(data, dict):
            return dict.__new__(DataMapping)
        elif isinstance(data, DataArray):
            if name is not None:
                data = data.rename(name)
            return data
        else:
            return DataArray(data, name=name)

    @property
    def empty(self):
        """``True`` if the collection contains no elements."""
        return len(self) == 0

    def query(self, indexers=None, **indexers_kwargs):
        """
        Query a given subset from a data collection.

        The data collection is walked through, if any node name corresponds to a key of
        the `indexers`, the corresponding value is used to select a subset of that node.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching fields and values given by string or int.
        **indexers_kwargs : dict, optional
            The keyword arguments form of indexers. Overwrite indexers input if both
            are provided.

        Returns:
        -------
        DataCollection:
            The queried data.

        Examples
        --------
        >>> import xdas as xd
        >>> from xdas.synthetics import wavelet_wavefronts
        >>> da = wavelet_wavefronts()
        >>> dc = xd.DataCollection(
        ...     {
        ...         "das1": ("acquisition", [da, da]),
        ...         "das2": ("acquisition", [da, da, da]),
        ...     },
        ...     "instrument",
        ... )
        >>> dc.query(instrument="das1", acquisition=0)
        Instrument:
          das1:
            Acquisition:
            0: <xdas.DataArray (time: 300, distance: 401)>

        """
        if indexers is None:
            indexers = {}
        indexers.update(indexers_kwargs)
        if self.name in indexers:
            key = indexers[self.name]
            if self.issequence():
                if isinstance(key, int):
                    data = [self[key]]
                elif isinstance(key, slice):
                    data = self[key]
                else:
                    raise ValueError(f"{self.name} query must be a string")
                data = [
                    (
                        value.query(indexers)
                        if isinstance(value, DataCollection)
                        else value
                    )
                    for value in data
                ]
            elif self.ismapping():
                if isinstance(key, str):
                    data = {
                        name: value
                        for name, value in self.items()
                        if fnmatch(name, key)
                    }
                else:
                    raise ValueError(f"{self.name} query must be a string")
                data = {
                    name: (
                        value.query(indexers)
                        if isinstance(value, DataCollection)
                        else value
                    )
                    for name, value in data.items()
                }
            else:
                raise TypeError("unknown type of data collection")
            return DataCollection(data, self.name)
        else:
            return self

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
        self = DataMapping.from_netcdf(fname, group)
        try:
            keys = [int(key) for key in self.keys()]
            if keys == list(range(len(keys))):
                return DataSequence.from_mapping(self)
            else:
                return self
        except ValueError:
            return self


class DataMapping(DataCollection, dict):
    """
    A Mapping of dataarrays.

    A data mapping is a dictionary whose keys are any user defined identifiers and
    values are dataarray objects.
    """

    def __new__(cls, data, name=None):
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
        if len(self) == 0:
            return "Empty"
        width = max([len(str(key)) for key in self])
        name = self.name if self.name is not None else "collection"
        s = f"{name.capitalize()}:\n"
        for key, value in self.items():
            if isinstance(key, int):
                label = f"  {key:{width}}: "
            else:
                label = f"  {key + ':':{width + 1}} "
            if isinstance(value, DataArray):
                s += label + repr(value).split("\n")[0] + "\n"
            else:
                s += label + "\n"
                s += "\n".join(f"    {e}" for e in repr(value).split("\n")[:-1]) + "\n"
        return s

    def __reduce__(self):
        return self.__class__, (dict(self), self.name)

    @property
    def fields(self):
        """Ordered, deduplicated tuple of node names at this level and its immediate children."""
        out = (self.name,) + tuple(
            value.name for value in self.values() if isinstance(value, DataCollection)
        )
        return uniquifiy(out)

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
        if not self.name == other.name:
            return False
        if not list(self.keys()) == list(other.keys()):
            return False
        if not all(self[key].equals(other[key]) for key in self):
            return False
        return True

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
        Apply an atom to each data array of the data collection

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
        return repr(self.to_mapping())

    def __reduce__(self):
        return self.__class__, (list(self), self.name)

    @property
    def fields(self):
        """Ordered, deduplicated tuple of node names at this level and its immediate children."""
        out = (self.name,) + tuple(
            value.name for value in self if isinstance(value, DataCollection)
        )
        return uniquifiy(out)

    def to_mapping(self):
        """Convert to an integer-keyed :class:`DataMapping`."""
        return DataMapping({key: value for key, value in enumerate(self)}, self.name)

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
        if not self.name == other.name:
            return False
        if not len(self) == len(other):
            return False
        if not all(a.equals(b) for a, b in zip(self, other)):
            return False
        return True

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
        Apply an atom to each data array of the data collection

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
