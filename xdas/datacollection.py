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
        >>> dc = DataCollection
        >>> dc = DataCollection(
        ... {
        ...     "das1": DataCollection([db, db], "acquisition"),
        ...     "das2": DataCollection([db, db, db], "acquisition"),
        ... },
        ... "instrument",
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
        keys = list(self.keys)
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

    def to_netcdf(self, fname, virtual=False):
        if os.path.exists(fname):
            os.remove(fname)
        for key in self:
            self[key].to_netcdf(fname, group=key, virtual=virtual, mode="a")

    @classmethod
    def from_netcdf(cls, fname):
        with h5py.File(fname, "r") as file:
            groups = list(file.keys())
        self = cls()
        for group in groups:
            self[group] = Database.from_netcdf(fname, group=group)
        return self


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

    def to_netcdf(self, fname, virtual=False):
        self.to_mapping().to_netcdf(fname, virtual)

    @classmethod
    def from_netcdf(cls, fname):
        return DataMapping.from_netcdf(fname).from_mapping()
