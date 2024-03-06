import os
from fnmatch import fnmatch

import h5py

from .database import Database


class AbstractDataCollection:
    @property
    def empty(self):
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
        >>> import xdas
        >>> from xdas.synthetics import generate
        >>> db = generate()
        >>> dc = xdas.DataCollection(
        ...     {
        ...         "das1": ("acquisition", [db, db]),
        ...         "das2": ("acquisition", [db, db, db]),
        ...     },
        ...     "instrument",
        ... )
        >>> dc.query(instrument="das1", acquisition=0)
        Instrument:
          das1:
            Acquisition:
            0: <xdas.Database (time: 300, distance: 401)>

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
                        if isinstance(value, AbstractDataCollection)
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
                        if isinstance(value, AbstractDataCollection)
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
        return isinstance(self, DataSequence)

    def ismapping(self):
        return isinstance(self, DataMapping)


class DataCollection:
    def __new__(cls, data, name=None):
        """
        Nested collection of database.

        Parameters
        ----------
        data: list or dict of DataCollection or Database
            The nested data. It can be composed either of sequences or mapping. The
            leaves must be databases.
        name: str
            The name of the current level of nesting.

        Returns:
        -------
        DataCollection:
            The nested data as a DataSequence or DataMapping.

        Examples
        --------
        >>> import xdas
        >>> from xdas.synthetics import generate
        >>> db = generate()
        >>> dc = xdas.DataCollection(
        ...     {
        ...         "das1": ("acquisition", [db, db]),
        ...         "das2": ("acquisition", [db, db, db]),
        ...     },
        ...     "instrument",
        ... )
        >>> dc
        Instrument:
          das1:
            Acquisition:
            0: <xdas.Database (time: 300, distance: 401)>
            1: <xdas.Database (time: 300, distance: 401)>
          das2:
            Acquisition:
            0: <xdas.Database (time: 300, distance: 401)>
            1: <xdas.Database (time: 300, distance: 401)>
            2: <xdas.Database (time: 300, distance: 401)>

        """
        if isinstance(data, tuple) and name is None:
            name, data = data
            return DataCollection(data, name)
        if isinstance(data, list):
            return DataSequence(data, name)
        elif isinstance(data, dict):
            return DataMapping(data, name)
        else:
            raise TypeError("could not parse `data`")

    @classmethod
    def from_netcdf(cls, fname, group=None):
        self = DataMapping.from_netcdf(fname, group)
        try:
            keys = [int(key) for key in self.keys()]
            if keys == list(range(len(keys))):
                return DataSequence.from_mapping(self)
            else:
                return self
        except ValueError:
            return self


class DataMapping(AbstractDataCollection, dict):
    """
    A Mapping of databases.

    A data mapping is a dictionary whose keys are any user defined identifiers and
    values are database objects.
    """

    def __init__(self, data, name=None):
        data = {
            key: (
                value
                if isinstance(value, (Database, AbstractDataCollection))
                else DataCollection(value)
            )
            for key, value in data.items()
        }
        super().__init__(data)
        self.name = name

    def __repr__(self):
        width = max([len(str(key)) for key in self])
        name = self.name if self.name is not None else "sequence"
        s = f"{name.capitalize()}:\n"
        for key, value in self.items():
            if isinstance(key, int):
                label = f"  {key:{width}}: "
            else:
                label = f"  {key + ':':{width + 1}} "
            if isinstance(value, Database):
                s += label + repr(value).split("\n")[0] + "\n"
            else:
                s += label + "\n"
                s += "\n".join(f"    {e}" for e in repr(value).split("\n")[:-1]) + "\n"
        return s

    @property
    def fields(self):
        out = (self.name,) + tuple(
            value.name
            for value in self.values()
            if isinstance(value, AbstractDataCollection)
        )
        return uniquifiy(out)

    def to_netcdf(self, fname, group=None, virtual=False, **kwargs):
        if group is None and os.path.exists(fname):
            os.remove(fname)
        for key in self:
            name = self.name if self.name is not None else "collection"
            location = "/".join([name, str(key)])
            if group is not None:
                location = "/".join([group, location])
            self[key].to_netcdf(fname, location, virtual, mode="a")

    @classmethod
    def from_netcdf(cls, fname, group=None):
        with h5py.File(fname, "r") as file:
            if group is None:
                group = file[list(file.keys())[0]]
            else:
                group = file[group]
            name = group.name.split("/")[-1]
            keys = list(group.keys())
            self = cls({}, name=None if name == "collection" else name)
            for key in keys:
                subgroup = group[key]
                if get_depth(subgroup) == 0:
                    self[key] = Database.from_netcdf(fname, subgroup.name)
                else:
                    subgroup = subgroup[list(subgroup.keys())[0]]
                    self[key] = DataCollection.from_netcdf(fname, subgroup.name)
        return self

    def equals(self, other):
        if not isinstance(other, self.__class__):
            return False
        if not self.name == other.name:
            return False
        if not list(self.keys()) == list(other.keys()):
            return False
        if not all(self[key].equals(other[key]) for key in self):
            return False
        return True

    def sel(self, indexers=None, **indexers_kwargs):
        data = {
            key: value.sel(indexers, **indexers_kwargs) for key, value in self.items()
        }
        data = {
            key: value
            for key, value in data.items()
            if (isinstance(value, AbstractDataCollection) or not value.empty)
        }
        return self.__class__(data, self.name)


class DataSequence(AbstractDataCollection, list):
    """
    A collection of databases.

    A data sequence is a list whose values are database objects.
    """

    def __init__(self, data, name=None):
        data = [
            (
                value
                if isinstance(value, (Database, AbstractDataCollection))
                else DataCollection(value)
            )
            for value in data
        ]
        super().__init__(data)
        self.name = name

    def __repr__(self):
        return repr(self.to_mapping())

    @property
    def fields(self):
        out = (self.name,) + tuple(
            value.name for value in self if isinstance(value, AbstractDataCollection)
        )
        return uniquifiy(out)

    def to_mapping(self):
        return DataMapping({key: value for key, value in enumerate(self)}, self.name)

    @classmethod
    def from_mapping(cls, data):
        return cls(data.values(), data.name)

    def to_netcdf(self, fname, group=None, virtual=False, **kwargs):
        self.to_mapping().to_netcdf(fname, group, virtual, **kwargs)

    @classmethod
    def from_netcdf(cls, fname, group=None):
        return DataMapping.from_netcdf(fname, group).from_mapping()

    def equals(self, other):
        if not isinstance(other, self.__class__):
            return False
        if not self.name == other.name:
            return False
        if not len(self) == len(other):
            return False
        if not all(a.equals(b) for a, b in zip(self, other)):
            return False
        return True

    def sel(self, indexers=None, **indexers_kwargs):
        data = [value.sel(indexers, **indexers_kwargs) for value in self]
        data = [
            value
            for value in data
            if (isinstance(value, AbstractDataCollection) or not value.empty)
        ]
        return self.__class__(data, self.name)


def get_depth(group):
    if not isinstance(group, h5py.Group):
        raise ValueError("not a group")
    depths = []
    group.visit(lambda name: depths.append(name.count("/")))
    return max(depths)


def uniquifiy(seq):
    seen = set()
    return tuple(x for x in seq if x not in seen and not seen.add(x))
