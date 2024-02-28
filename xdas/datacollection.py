import os

import h5py

from .database import Database


class AbstractDataCollection:
    def __init__(self, data, name=None):
        super().__init__(data)
        self.name = name


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
        ...         "das1": xdas.DataCollection([db, db], "acquisition"),
        ...         "das2": xdas.DataCollection([db, db, db], "acquisition"),
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
        if isinstance(data, list):
            return DataSequence(data, name)
        elif isinstance(data, dict):
            return DataMapping(data, name)
        else:
            raise TypeError("could not parse `data`")

    @classmethod
    def from_netcdf(cls, fname):
        self = DataMapping.from_netcdf(fname)
        keys = list(self.keys())
        if keys == list(range(len(keys))):
            return DataSequence.from_mapping(self)
        else:
            return self


class DataMapping(AbstractDataCollection, dict):
    """
    A Mapping of databases.

    A data mapping is a dictionary whose keys are any user defined identifiers and
    values are database objects.
    """

    def __new__(cls, *args, **kwargs):
        return dict.__new__(cls)

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

    def to_netcdf(self, fname, group=None, virtual=False, **kwargs):
        if os.path.exists(fname):
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
            if group is not None:
                file = file[group]
            name = list(file.keys())[0]
            groups = list(file[name].keys())
        self = cls({}, name=None if name == "collection" else name)
        for group in groups:
            location = "/".join([name, group])
            self[group] = Database.from_netcdf(fname, location)
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


class DataSequence(AbstractDataCollection, list):
    """
    A collection of databases.

    A data sequencw is a list whose values are database objects.
    """

    def __new__(cls, *args, **kwargs):
        return list.__new__(cls)

    def __repr__(self):
        return repr(self.to_mapping())

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
