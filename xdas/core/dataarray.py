import copy
import re
import warnings
from functools import partial

import h5py
import numpy as np
import xarray as xr

from ..virtual import VirtualArray, VirtualSource
from .coordinates import Coordinate, Coordinates, get_sampling_interval

HANDLED_NUMPY_FUNCTIONS = {}
HANDLED_METHODS = {}


class DataArray:
    """
    N-dimensional array with labeled coordinates and dimensions.

    It is the equivalent of and xarray.DataArray but with custom coordinate objects.
    Most of the DataArray API follows the DataArray one. DataArray objects also provide
    virtual dataset capabilities to manipulate huge multi-file NETCDF4 or HDF5 datasets.

    Parameters
    ----------
    data : array_like
        Values of the array. Can be a VirtualSource or a VirtualLayout for lazy loading
        of netCDF4/HDF5 files.
    coords : dict of Coordinate
        Coordinates to use for indexing along each dimension.
    dims : sequence of string, optional
        Name(s) of the data dimension(s). If provided, must be equal to the keys of
        `coords`.
    name : str, optional
        Name of this array.
    attrs : dict_like, optional
        Attributes to assign to the new instance.

    Raises
    ------
    ValueError
        If dims do not match the keys of coords.
    """

    def __init__(self, data=None, coords=None, dims=None, name=None, attrs=None):
        if data is None:
            data = np.array(np.nan)
        if not hasattr(data, "__array__"):
            data = np.asarray(data)
        if coords is None and dims is None:
            dims = tuple(f"dim_{index}" for index in range(data.ndim))
        if dims is not None and len(dims) != data.ndim:
            raise ValueError("different number of dimensions on `data` and `dims`")
        coords = Coordinates(coords, dims)
        if not len(coords.dims) == data.ndim:
            raise ValueError(
                "infered dimension number from `coords` does not match "
                "`data` dimensionality`"
            )
        self.data = data
        self.coords = coords
        self.name = name
        self.attrs = attrs

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.coords[key]
        else:
            query = self.coords.get_query(key)
            data = self.data.__getitem__(tuple(query.values()))
            coords = {
                name: (
                    coord.__getitem__(query[coord.dim])
                    if coord.dim is not None
                    else coord
                )
                for name, coord in self.coords.items()
            }
            dims = tuple(dim for dim in self.dims if not coords[dim].isscalar())
            return self.__class__(data, coords, dims, self.name, self.attrs)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.coords[key] = value
        else:
            query = self.coords.get_query(key)
            self.data.__setitem__(tuple(query.values()), value)

    def __repr__(self):
        edgeitems = 3 if not np.issubdtype(self.dtype, np.complexfloating) else 2
        precision = 6 if not np.issubdtype(self.dtype, np.complexfloating) else 4
        if isinstance(self.data, np.ndarray):
            data_repr = np.array2string(
                self.data, precision=precision, threshold=0, edgeitems=edgeitems
            )
        else:
            data_repr = repr(self.data)
        string = "<xdas.DataArray ("
        string += ", ".join([f"{dim}: {size}" for dim, size in self.sizes.items()])
        string += ")>\n"
        string += data_repr + "\n" + repr(self.coords)
        return string

    def __array__(self, dtype=None):
        if dtype is None:
            return self.data.__array__()
        else:
            return self.data.__array__(dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if not method == "__call__":
            return NotImplemented
        inputs = tuple(
            value.data if isinstance(value, self.__class__) else value
            for value in inputs
        )
        if "out" in kwargs:
            kwargs["out"] = tuple(
                value.data if isinstance(value, self.__class__) else value
                for value in kwargs["out"]
            )
        if "where" in kwargs:
            kwargs["where"] = tuple(
                value.data if isinstance(value, self.__class__) else value
                for value in kwargs["where"]
            )
        data = getattr(ufunc, method)(*inputs, **kwargs)
        if isinstance(data, tuple):
            return tuple(self.copy(data=d) for d in data)
        else:
            return self.copy(data=data)

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_NUMPY_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MyArray objects
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_NUMPY_FUNCTIONS[func](*args, **kwargs)

    def __getattr__(self, name):
        if name in HANDLED_METHODS:
            func = HANDLED_METHODS[name]
            method = partial(func, self)
            method.__name__ = name
            method.__doc__ = (
                f"    Method implementation of {name} function.\n\n"
                + "    *Original docstring below. Skip first parameter.*\n"
                + func.__doc__
            )
            return method
        else:
            raise AttributeError(f"'DataArray' object has no attribute '{name}'")

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

    @property
    def dims(self):
        return self.coords.dims

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
        return DimSizer(self)

    @property
    def nbytes(self):
        return self.data.nbytes

    @property
    def values(self):
        return self.__array__()

    @property
    def empty(self):
        return np.prod(self.shape) == 0

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
        if dim == "first":
            return 0
        elif dim == "last":
            return self.ndim - 1
        elif dim in self.dims:
            return self.dims.index(dim)
        else:
            raise ValueError("dim not found")

    def isel(self, indexers=None, **indexers_kwargs):
        """
        Return a new DataArray whose data is given by selecting indexes along the
        specified dimension(s).

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given by integers, slice
            objects or arrays.
        method : {None, "nearest", "ffill", "bfill"}, optional
            Method to use for inexact matches:
            - None (default): only exact matches
            - nearest: use nearest valid index value
            - ffill: propagate last valid index value forward
            - bfill: propagate next valid index value backward
        **indexers_kwargs : dict, optional
            The keyword arguments form of integers. Overwrite indexers input if both
            are provided.

        Returns
        -------
        DataArray
            The selected subset of the DataArray.
        """
        if indexers is None:
            indexers = {}
        indexers.update(indexers_kwargs)
        return self[indexers]

    def sel(self, indexers=None, method=None, endpoint=True, **indexers_kwargs):
        """
        Return a new DataArray whose data is given by selecting index labels along the
        specified dimension(s).

        In contrast to DataArray.isel, indexers for this method should use labels
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
        DataArray
            The selected part of the original data array.
        """
        if indexers is None:
            indexers = {}
        indexers.update(indexers_kwargs)
        key = self.coords.to_index(indexers, method, endpoint)
        return self[key]

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

    def rename(self, name):
        da = self.copy(deep=False)
        da.name = name
        return da

    def load(self):
        return self.copy(data=self.data.__array__())

    def to_xarray(self):
        """
        Convert to the xarray implementation of the DataArray structure.

        Coordinates are converted to dense arrays and lazy values are loaded in memory.

        Returns
        -------
        DataArray
            The converted in-memory DataArray.
        """
        data = self.__array__()
        return xr.DataArray(data, self.coords, self.dims, self.name, self.attrs)

    @classmethod
    def from_xarray(cls, da):
        return cls(da.data, da.coords, da.dims, da.name, da.attrs)

    def to_stream(
        self,
        network="NET",
        station="DAS{:05}",
        location="00",
        channel="{:1}N1",
        dim={"last": "first"},
    ):
        """
        Convert a data array into an obspy stream.

        Parameters
        ----------
        network : str, optional
            The network code, by default "NET".
        station : str, optional
            The station code. Must be a string that can be formatted.
            By default "DAS{:05}"
        location : str, optional
            The location code, by default "00".
        channel : str, optional
            The channel code. If the string can be formatted, the band code will be
            inferred from the sampling rate. By default "{:1}N1"
        dim : dict, optional
            A dict with as key the spatial dimension to split into traces, and as key
            the temporal dimension. By default {"last": "first"}.

        Returns
        -------
        Stream
            the obspy stream version of the data array.

        """
        dimdist, dimtime = dim.copy().popitem()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from obspy import Stream, Trace, UTCDateTime
        except ImportError:
            raise ImportError("obspy is not installed. Please install it.")
        if not self.ndim == 2:
            raise ValueError("the data array must be 2D")
        starttime = UTCDateTime(str(self[dimtime][0].values))
        delta = get_sampling_interval(self, dimtime)
        band_code = get_band_code(1.0 / delta)
        if "{" in channel and "}" in channel:
            channel = channel.format(band_code)
        header = {
            "network": network,
            "location": location,
            "channel": channel,
            "starttime": starttime,
            "delta": delta,
        }
        return Stream(
            [
                Trace(
                    data=self.isel({dimdist: idx}).values,
                    header=header | {"station": station.format(idx + 1)},
                )
                for idx in range(len(self[dimdist]))
            ]
        )

    @classmethod
    def from_stream(cls, st, dims=("channel", "time")):
        """
        Convert an obspy stream into a data array.

        Traces in the stream must have the same length an must be syncronized. Traces
        are stacked along the first axis. The trace ids are used as labels along the
        first dimension.

        Parameters
        ----------
        st: Stream
            The stream to convert.
        dims: (str, str)
            The name of the dimension respectively given to the trace and time
            dimensions.

        Returns
        -------
        DataArray:
            The consolidated data array.
        """
        data = np.stack([tr.data for tr in st])
        channel = [tr.id for tr in st]
        time = {
            "tie_indices": [0, st[0].stats.npts - 1],
            "tie_values": [
                np.datetime64(st[0].stats.starttime.datetime),
                np.datetime64(st[0].stats.endtime.datetime),
            ],
        }
        return cls(data, {dims[0]: channel, dims[1]: time})

    def to_netcdf(self, fname, group=None, virtual=None, **kwargs):
        """
        Write DataArray contents to a netCDF file.

        Parameters
        ----------
        fname : str
            Path to which to save this dataset.
        group : str, optional
            Path to the netCDF4 group in the given file to open.
        virtual : bool, optional
            Weather to write a virtual dataset. The DataArray data must be a VirtualSource
            or a VirtualLayout. Default (None) is to try to write a virtual dataset if
            possible.

        Raises
        ------
        ValueError
            _description_
        """
        if virtual is None:
            virtual = isinstance(self.data, VirtualArray)
        data_vars = []
        mapping = ""
        for dim in self.coords:
            if self.coords[dim].isinterp():
                tie_indices = self.coords[dim].tie_indices
                tie_values = self.coords[dim].tie_values
                if np.issubdtype(tie_values.dtype, np.datetime64):
                    tie_values = tie_values.astype("M8[ns]")
                mapping += f"{dim}: {dim}_indices {dim}_values "
                interpolation = xr.DataArray(
                    name=f"{dim}_interpolation",
                    attrs={
                        "interpolation_name": "linear",
                        "tie_points_mapping": f"{dim}_points: {dim}_indices {dim}_values",
                    },
                )
                indices = xr.DataArray(
                    name=f"{dim}_indices",
                    data=tie_indices,
                    dims=f"{dim}_points",
                )
                values = xr.DataArray(
                    name=f"{dim}_values",
                    data=tie_values,
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
        attrs = {"coordinate_interpolation": mapping} if mapping else None
        if not virtual:
            da = xr.DataArray(
                data=self.values,
                coords=coords,
                dims=self.dims,
                name=self.name,
                attrs=attrs,
            )
            ds[da.name] = da
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds.to_netcdf(fname, group=group, **kwargs)
        elif virtual and isinstance(self.data, VirtualArray):
            da = xr.DataArray(  # TODO: this is dirty
                data=np.empty((0, 0)),
                coords=coords,
                dims=self.dims,
                name="__tmp__",
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
                for key in attrs:
                    file[name].attrs[key] = attrs[key]
        else:
            raise ValueError(
                "can only use `virtual=True` with a VirtualSource or a VirtualLayout"
            )

    @classmethod
    def from_netcdf(cls, fname, group=None):
        """
        Lazily read a data array from a NetCDF file.

        Parameters
        ----------
        fname: str
            The path of the file to open.
        group: str, optional
            The location of the data array within the file. Root by default

        Returns
        -------
        DataArray
            The openend data array.
        """
        with xr.open_dataset(fname, group=group) as ds:
            if not ("Conventions" in ds.attrs and "CF" in ds.attrs["Conventions"]):
                raise TypeError(
                    "file format not recognized. please provide the file format "
                    "with the `engine` keyword argument"
                )
            if len(ds) == 1:
                name, da = next(iter(ds.items()))
                coords = {
                    name: (
                        coord.dims[0],
                        (
                            coord.values.astype("U")
                            if coord.dtype == np.dtype("O")
                            else coord.values
                        ),
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
                        (
                            coord.values.astype("U")
                            if coord.dtype == np.dtype("O")
                            else coord.values
                        ),
                    )
                    for name, coord in da.coords.items()
                }
                mapping = da.attrs.pop("coordinate_interpolation")
                matches = re.findall(r"(\w+): (\w+) (\w+)", mapping)
                for match in matches:
                    dim, indices, values = match
                    data = {"tie_indices": ds[indices], "tie_values": ds[values]}
                    coords[dim] = Coordinate(data, dim)
        with h5py.File(fname) as file:
            if group:
                file = file[group]
            name = "__values__" if da.name is None else da.name
            data = VirtualSource(file[name])
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


class DimSizer(dict):
    def __init__(self, obj):
        super().__init__({dim: len(obj.coords[dim]) for dim in obj.dims})

    def __getitem__(self, key):
        if key == "first":
            key = list(self.keys())[0]
        if key == "last":
            key = list(self.keys())[-1]
        return super().__getitem__(key)


def get_band_code(sampling_rate):
    band_code = ["T", "P", "R", "U", "V", "L", "M", "B", "H", "C", "F"]
    limits = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 80, 250, 1000, 5000]
    index = np.searchsorted(limits, sampling_rate, "right") - 1
    if index < 0 or index >= len(band_code):
        return "X"
    else:
        return band_code[index]
