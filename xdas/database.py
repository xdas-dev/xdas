import copy
import os
import re

import dask.array as da
import h5py
import numpy as np
import xarray as xr

from .coordinates import Coordinates, InterpCoordinate
from .virtual import DataLayout, DataSource


class DataCollection(dict):
    """
    A collection of databases.

    A data collection is a dictionary whose keys are any user defined identifiers and
    values are database objects.
    """

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
        coords = Coordinates(coords)
        if dims is None:
            dims = coords.dims
        self.data = data
        self.coords = Coordinates(coords)
        self.dims = dims
        self.name = name
        self.attrs = attrs

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.coords[key]
        else:
            query = self.coords.get_query(key)
            data = self.data.__getitem__(tuple(query.values()))
            dct = {dim: self.coords[dim].__getitem__(query[dim]) for dim in query}
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

    def __add__(self, other):
        return self.copy(data=self.data.__add__(other))

    def __radd__(self, other):
        return self.copy(data=self.data.__radd__(other))

    def __sub__(self, other):
        return self.copy(data=self.data.__sub__(other))

    def __rsub__(self, other):
        return self.copy(data=self.data.__rsub__(other))

    def __mul__(self, other):
        return self.copy(data=self.data.__mul__(other))

    def __rmul__(self, other):
        return self.copy(data=self.data.__rmul__(other))

    def __truediv__(self, other):
        return self.copy(data=self.data.__truediv__(other))

    def __rtruediv__(self, other):
        return self.copy(data=self.data.__rtruediv__(other))

    def __pow__(self, other):
        return self.copy(data=self.data.__pow__(other))

    def __rpow__(self, other):
        return self.copy(data=self.data.__rpow__(other))

    def __array__(self):
        return self.data.__array__()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        _, *args = inputs
        assert _ is self
        data = getattr(ufunc, method)(self.data, *args, **kwargs)
        return self.copy(data=data)

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
        return {dim: len(self.coords[dim]) for dim in self.dims}

    @property
    def values(self):
        return self.__array__()

    @property
    def loc(self):
        return LocIndexer(self)

    def equals(self, other):
        if isinstance(other, self.__class__):
            if not np.array_equal(self.values, other.values):
                return False
            if not self.coords.equals(other.coords):
                return False
            if not self.dims == other.dims:
                return False
            if not self.name == other.name:
                return False
            if not self.attrs == other.attrs:
                return False
            return True
        else:
            return False

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

    def load(self):
        return self.copy(data=self.data.__array__())

    def to_xarray(self, load=True):
        """
        Convert the Database to a DataArray object.

        Coordinates are converted to dense arrays and lazy values are loaded in memory.

        Returns
        -------
        DataArray
            The converted in-memory DataArray.
        """
        if load:
            data = self.__array__()
        else:
            chunks = tuple("auto" if axis == 0 else -1 for axis in range(self.ndim))
            data = da.from_array(self.data, chunks=chunks)
        coords = {dim: self.coords[dim].__array__() for dim in self.coords}
        return xr.DataArray(data, coords, self.dims, self.name, self.attrs)

    @classmethod
    def from_xarray(cls, da, tolerance=None):
        coords = {
            dim: InterpCoordinate.from_array(da[dim].values, tolerance)
            for dim in da.dims
        }
        return cls(da.data, coords, da.dims, da.name, da.attrs)

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
        for dim in self.coords:
            if self.coords[dim].isinterp():
                mapping += f"{dim}: {dim}_indices {dim}_values "
                interpolation = xr.DataArray(
                    name=f"{dim}_interpolation",
                    attrs={
                        "interpolation_name": "linear",
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
        ds = xr.Dataset(
            data_vars={da.name: da for da in data_vars},
            attrs={"Conventions": "CF-1.9"},
        )
        coords = {
            dim: self.coords[dim].__array__()
            for dim in self.coords
            if not self.coords[dim].isinterp()
        }
        if not virtual:
            da = xr.DataArray(
                data=self.values,
                coords=coords,
                dims=self.dims,
                name=self.name,
                attrs={"coordinate_interpolation": mapping},
            )
            ds[da.name] = da
            ds.to_netcdf(fname, group=group, **kwargs)
        elif virtual and isinstance(self.data, (DataSource, DataLayout)):
            da = xr.DataArray(
                data=np.empty((0, 0)),
                coords=coords,
                dims=self.dims,
                name="__tmp__",
                attrs={"coordinate_interpolation": mapping},
            )
            ds[da.name] = da
            ds.to_netcdf(fname, group=group, **kwargs)
            with h5py.File(fname, "r+") as file:
                if self.name is None:
                    name = "__values__"
                else:
                    name = self.name
                if group:
                    file = file[group]
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
        with xr.open_dataset(fname, group=group, **kwargs) as ds:
            if len(ds) == 1:
                name, da = next({"a": "b"}.items())
                coords = {
                    name: (
                        coord.dims[0],
                        coord.values.astype("U")
                        if coord.dtype == np.dtype("O")
                        else coord.values,
                    )
                    for name, coord in da.coords.items()
                }
            else:
                data_vars = [
                    var
                    for var in ds.values()
                    if "coordinate_interpolation" in var.attrs
                ]
                if len(data_vars) == 1:
                    da = data_vars[0]
                else:
                    raise ValueError("several possible data arrays detected")
                coords = {
                    name: (
                        coord.dims[0],
                        coord.values.astype("U")
                        if coord.dtype == np.dtype("O")
                        else coord.values,
                    )
                    for name, coord in da.coords.items()
                }
                mapping = da.attrs.pop("coordinate_interpolation")
                matches = re.findall(r"(\w+): (\w+) (\w+)", mapping)
                for match in matches:
                    dim, indices, values = match
                    coords[dim] = InterpCoordinate(
                        {"tie_indices": ds[indices], "tie_values": ds[values]}
                    )
        with h5py.File(fname) as file:
            if group:
                file = file[group]
            name = "__values__" if da.name is None else da.name
            data = DataSource(file[name])
        return cls(data, coords, da.dims, da.name, None if da.attrs == {} else da.attrs)

    def plot(self, *args, **kwargs):
        if self.ndim == 1:
            self.to_xarray().plot.line(*args, **kwargs)
        elif self.ndim == 2:
            self.to_xarray().plot.imshow(*args, **kwargs)
        else:
            self.to_xarray().plot(*args, **kwargs)


class LocIndexer:
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, key):
        key = self.obj.coords.to_index(key)
        return self.obj.__getitem__(key)

    def __setitem__(self, key, value):
        key = self.obj.coords.to_index(key)
        self.obj.__setitem__(key, value)
