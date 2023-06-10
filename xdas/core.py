import copy
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import xarray as xr
from tqdm import tqdm

from .coordinates import Coordinate, Coordinates


def open_mfdatabase(paths, engine="netcdf", tolerance=np.timedelta64(0, "us")):
    """
    Open a multiple file database.

    Parameters
    ----------
    paths: str
        The path names given using shell=style wildcards.
    engine: str
        The engine to use to read the file.
    tolerance: timedelta64
        The tolerance to consider that the end of a file is continuous with the begging
        of the following

    Returns
    -------
    Database
        The database containing all files data.
    """
    fnames = sorted(glob(paths))
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(open_database, fname, engine=engine) for fname in fnames
        ]
        dbs = [
            future.result()
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Fetching metadata from files",
            )
        ]
    return concatenate(dbs, tolerance=tolerance)


def concatenate(dbs, tolerance=np.timedelta64(0, "us")):
    """
    Concatenate several databases along the time dimension.

    Parameters
    ----------
    dbs : list
        List of databases to concatenate.
    tolerance : timedelta64, optional
        The tolerance to consider that the end of a file is continuous with beginning of
        the following, by default np.timedelta64(0, "us").

    Returns
    -------
    Database
        The concatenated database.
    """

    dbs = sorted(dbs, key=lambda db: db["time"][0])
    shape = (sum([db.shape[0] for db in dbs]), dbs[0].shape[1])
    dtype = dbs[0].dtype
    layout = DataLayout(shape=shape, dtype=dtype)
    idx = 0
    tie_indices = []
    tie_values = []
    for db in dbs:
        layout[idx : idx + db.shape[0]] = db.data
        tie_indices.extend(idx + db["time"].tie_indices)
        tie_values.extend(db["time"].tie_values)
        idx += db.shape[0]
    time = Coordinate(tie_indices, tie_values).simplify(tolerance)
    return Database(layout, {"time": time, "distance": dbs[0]["distance"]})


def open_database(fname, group=None, engine="netcdf", **kwargs):
    """
    Open a database.

    Parameters
    ----------
    fname : str
        The path of the database.
    group : str, optional
        The file group where the database is located, by default None which corresponds
        to the root of the file.
    engine : str, optional
        The file format, by default "netcdf".

    Returns
    -------
    Database
        The opened database.

    Raises
    ------
    ValueError
        If the engine si not recognized.
    """
    if engine == "netcdf":
        return Database.from_netcdf(fname, group=group, **kwargs)
    elif engine == "asn":
        from .io.asn import read

        return read(fname)
    else:
        raise ValueError("engine not recognized")


def open_datacollection(fname, **kwargs):
    """
    Open a DataCollection from a file.

    Parameters
    ----------
    fname : str
        The path of the DataCollection.

    Returns
    -------
    DataCollection
        The opened DataCollection.
    """
    return DataCollection.from_netcdf(fname, **kwargs)


class DataCollection(dict):
    """
    A collection of databases.

    A data collection is a dictionary whose keys are any user defined identifiers and
    values are database objects.
    """

    def to_netcdf(self, fname, virtual=False):
        for key in self:
            self[key].to_netcdf(fname, group=key, virtual=virtual, mode="a")

    @classmethod
    def from_netcdf(cls, fname):
        with h5py.File(fname, "r") as file:
            groups = file.keys()
        self = cls()
        for group in groups:
            self[group] = Database.from_netcdf(fname, group=group)
        return self


class Database:
    """
    N-dimensional array with labeled coordinates and dimensions.

    It is the equivalent of and xarray.DataArray but with custom coordinate objects.
    Most of the Database API follows the DataArray one. Database objects also provide
    virtual dataset capabilities to manipulate huge multi-file NETCDF4 or HDF5 datasets.

    Parameters
    ----------
    data : array_like
        Values of the array. Can be a DataSource or a DataLayout for lazy loading of
        netCDF4/HSF5 files.
    coords : dict of Coordinate
        Coordinates to use for indexing along each dimension.
    dims : sequence of string, optional
        Name(s) of the data dimension(s). If provided, must be equal to the keys of
        `coords`. Used for API compatibility with xarray.
    name : str, optional
        Name of this array.
    attrs : dict_like, optional
        Attributes to assign to the new instance.

    Raises
    ------
    ValueError
        If dims do not match the keys of coords.
    """

    def __init__(self, data, coords, dims=None, name=None, attrs=None):
        # if not (data.shape == tuple(len(coord) for coord in coords.values())):
        # raise ValueError("Shape mismatch between data and coordinates")
        self.data = data
        self.coords = Coordinates(coords)
        self.dims = tuple(coords.keys())
        if dims is not None:
            if not (self.dims == dims):
                raise ValueError("Dimension mismatch between coordinates and dims")
        self.name = name
        self.attrs = attrs

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.coords[key]
        else:
            query = self.coords.get_query(key)
            data = self.data.__getitem__(tuple(query.values()))
            dct = {dim: self.coords[dim][query[dim]] for dim in query}
            coords = Coordinates(dct)
            return self.__class__(data, coords)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.coords[key] = value
        else:
            query = self.coords.get_query(key)
            self.data.__setitem__(tuple(query.values()), value)

    def __repr__(self):
        string = "<xdas.Database ("
        string += ", ".join([f"{dim}: {size}" for dim, size in self.sizes.items()])
        string += ")>\n"
        string += repr(self.data) + "\n" + repr(self.coords)
        return string

    def __array__(self):
        return self.data.__array__()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise NotImplementedError()

    def __array_function__(self, func, types, args, kwargs):
        raise NotImplementedError()

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return len(self.dims)

    @property
    def sizes(self):
        return {dim: len(coord) for dim, coord in self.coords.items()}

    @property
    def values(self):
        return self.__array__()

    @property
    def loc(self):
        return LocIndexer(self)

    def get_axis_num(self, dim):
        """
        Return axis number corresponding to dimension in this array.

        Parameters
        ----------
        dim : str
            Dimension name for which to lookup axis.

        Returns
        -------
        int
            Axis number corresponding to the given dimension
        """
        return self.dims.index(dim)

    def isel(self, indexers=None, **indexers_kwargs):
        """
        Return a new Database whose data is given by selecting indexes along the
        specified dimension(s).

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
        Database
            The selected subset of the Database.
        """
        if indexers is None:
            indexers = {}
        indexers.update(indexers_kwargs)
        return self[indexers]

    def sel(self, indexers=None, **indexers_kwargs):
        """
        Return a new Database whose data is given by selecting index labels along the
        specified dimension(s).

        In contrast to Database.isel, indexers for this method should use labels
        instead of integers.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given by scalars, slices or
            arrays of tick labels.
        **indexers_kwargs : dict, optional
            The keyword arguments form of integers. Overwrite indexers input if both
            are provided.

        Returns
        -------
        Database
            _description_
        """
        if indexers is None:
            indexers = {}
        indexers.update(indexers_kwargs)
        return self.loc[indexers]

    def copy(self, deep=True, data=None):
        """
        Returns a copy of this array

        If deep=True, a deep copy is made of the data array. Otherwise, a shallow copy
        is made, and the returned data array's values are a new view of this data
        array's values.

        Use data to create a new object with the same structure as original but
        entirely new data.

        Parameters
        ----------
        deep : bool, optional
            Whether the data array and its coordinates are loaded into memory and copied
            onto the new object. Default is True.
        data : array_like, optional
            Data to use in the new object. Must have same shape as original. When data
            is used, deep is ignored for all data variables, and only used for coords.

        Returns
        -------
        DataArray
            New object with dimensions, attributes, coordinates, name, encoding, and
            optionally data copied from original.
        """
        if deep:
            copy_fn = copy.deepcopy
        else:
            copy_fn = copy.copy
        if data is None:
            data = copy_fn(self.data)
        return self.__class__(
            data,
            copy_fn(self.coords),
            copy_fn(self.dims),
            copy_fn(self.name),
            copy_fn(self.attrs),
        )

    def to_xarray(self):
        """
        Convert the Database to a DataArray object.

        Coordinates are converted to dense arrays and lazy values are loaded in memory.

        Returns
        -------
        DataArray
            The converted in-memory DataArray.
        """
        return xr.DataArray(
            data=self.__array__(),
            coords={dim: self.coords[dim].__array__() for dim in self.coords},
            dims=self.dims,
            name=self.name,
            attrs=self.attrs,
        )

    def to_netcdf(self, fname, group=None, virtual=False, **kwargs):
        """
        Write Database contents to a netCDF file.

        Parameters
        ----------
        fname : str
            Path to which to save this dataset.
        group : str, optional
            Path to the netCDF4 group in the given file to open.
        virtual : bool, optional
            Weather to write a virtual dataset. The Database data must be a DataSource
            or a DataLayout. Default is False.

        Raises
        ------
        ValueError
            _description_
        """
        data_vars = []
        mapping = ""
        for dim in self.coords.dims:
            mapping += f"{dim}: {dim}_indices {dim}_values "
            interpolation = xr.DataArray(
                name=f"{dim}_interpolation",
                attrs={
                    "interpolation_name": self.coords[dim].kind,
                    "tie_points_mapping": f"{dim}_points: {dim}_indices {dim}_values",
                },
            ).astype("i4")
            indices = xr.DataArray(
                name=f"{dim}_indices",
                data=self.coords[dim].tie_indices,
                dims=f"{dim}_points",
            )
            values = xr.DataArray(
                name=f"{dim}_values",
                data=self.coords[dim].tie_values,
                dims=f"{dim}_points",
            )
            data_vars.extend([interpolation, indices, values])
        dataset = xr.Dataset(
            data_vars={xarr.name: xarr for xarr in data_vars},
            attrs={"Conventions": "CF-1.9"},
        )
        if not virtual:
            xarr = xr.DataArray(
                self.values,
                dims=self.dims,
                name=self.name,
                attrs={"coordinate_interpolation": mapping},
            )
            dataset[xarr.name] = xarr
            dataset.to_netcdf(fname, group=group, **kwargs)
        elif virtual and isinstance(self.data, (DataSource, DataLayout)):
            xarr = xr.DataArray(
                data=np.empty((0, 0)),
                dims=self.dims,
                name="__tmp__",
                attrs={"coordinate_interpolation": mapping},
            )
            dataset[xarr.name] = xarr
            dataset.to_netcdf(fname, group=group, **kwargs)
            with h5py.File(fname, "r+") as file:
                if self.name is None:
                    name = "__values__"
                else:
                    name = self.name
                if group:
                    file = file["group"]
                self.data.to_dataset(file, name)
                for axis, dim in enumerate(self.dims):
                    file[name].dims[axis].attach_scale(file[dim])
                for key in file["__tmp__"].attrs:
                    file[name].attrs[key] = file["__tmp__"].attrs[key]
                del file["__tmp__"]
        else:
            raise ValueError(
                "can only use `virtual=True` with a DataSource or a DataLayout"
            )

    @classmethod
    def from_netcdf(cls, fname, group=None, **kwargs):
        with xr.open_dataset(fname, group=group, **kwargs) as dataset:
            data_vars = [
                var
                for var in dataset.values()
                if "coordinate_interpolation" in var.attrs
            ]
            if len(data_vars) == 1:
                da = data_vars[0]
            else:
                raise ValueError("several possible data arrays detected")
            name = da.name
            coords = Coordinates()
            mapping = da.attrs.pop("coordinate_interpolation")
            matches = re.findall(r"(\w+): (\w+) (\w+)", mapping)
            for match in matches:
                dim, indices, values = match
                coords[dim] = Coordinate(dataset[indices], dataset[values])
        with h5py.File(fname) as file:
            if group:
                file = file[group]
            if name is None:
                name = "__values__"
            data = DataSource(file[name])
        return cls(data, coords)


class LocIndexer:
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, key):
        key = self.obj.coords.to_index(key)
        return self.obj[key]

    def __setitem__(self, key, value):
        key = self.obj.coords.to_index(key)
        self.obj[key] = value


class DataSource(h5py.VirtualSource):
    """
    A lazy array object pointing toward a netCDF4/HDF5 file.
    """

    def __array__(self):
        return self.to_layout().__array__()

    def __repr__(self):
        return f"DataSource: {to_human(self.nbytes)} ({self.dtype})"

    @property
    def nbytes(self):
        return np.prod(self.shape) * self.dtype.itemsize

    def to_layout(self):
        layout = DataLayout(self.shape, self.dtype)
        layout[...] = self
        return layout

    def to_dataset(self, file, name):
        self.to_layout().to_dataset(self, file, name)

    def to_dict(self):
        return {
            "path": self.path,
            "name": self.name,
            "shape": self.shape,
            "dtype": str(self.dtype),
            "maxshape": self.maxshape,
            "sel": self.sel._sel,
        }

    @classmethod
    def from_dict(cls, dtc):
        vsource = cls(
            dtc["path"], dtc["name"], dtc["shape"], dtc["dtype"], dtc["maxshape"]
        )
        vsource.sel._sel = dtc["sel"]
        return vsource


class DataLayout(h5py.VirtualLayout):
    """
    A composite lazy array pointing toward multiple netCDF4/HDF5 files.
    """

    def __array__(self):
        with TemporaryDirectory() as tmpdirname:
            fname = os.path.join(tmpdirname, "vds.h5")
            with h5py.File(fname, "w") as file:
                dataset = file.create_virtual_dataset(
                    "__values__", self, fillvalue=np.nan
                )
            with h5py.File(fname, "r") as file:
                dataset = file["__values__"]
                out = dataset[...]
        return out

    def __repr__(self):
        return f"DataSource: {to_human(self.nbytes)} ({self.dtype})"

    def __getitem__(self, key):
        raise NotImplementedError(
            "Cannot slice DataLayout. Use `self.to_netcdf(fname, virtual=True)` to "
            "write to disk and reopen it with `xdas.open_database(fname)`"
        )

    @property
    def nbytes(self):
        return np.prod(self.shape) * self.dtype.itemsize

    def to_dataset(self, file, name):
        return file.create_virtual_dataset(name, self, fillvalue=np.nan)


def to_human(size):
    unit = {0: "B", 1: "K", 2: "M", 3: "G", 4: "T"}
    n = 0
    while size > 1024:
        size /= 1024
        n += 1
    return f"{size:.1f}{unit[n]}"
