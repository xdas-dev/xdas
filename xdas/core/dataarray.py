import copy
import json
import re
import warnings
from functools import partial

import h5netcdf
import h5py
import hdf5plugin
import numpy as np
import xarray as xr
from dask.array import Array as DaskArray
from numpy.lib.mixins import NDArrayOperatorsMixin

from ..dask.core import dumps, from_dict, loads, to_dict
from ..virtual import VirtualArray, VirtualSource, _to_human
from .coordinates import Coordinate, Coordinates, get_sampling_interval

HANDLED_NUMPY_FUNCTIONS = {}
HANDLED_METHODS = {}


class DataArray(NDArrayOperatorsMixin):
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
        # data
        if data is None:
            data = np.array(np.nan)
        if not hasattr(data, "__array__"):
            data = np.asarray(data)
        self._data = data

        # coords & dims
        if dims is None:
            if coords is None:
                dims = tuple(f"dim_{index}" for index in range(data.ndim))
            elif isinstance(coords, Coordinates):
                dims = coords.dims
        if dims is not None and len(dims) != data.ndim:
            raise ValueError("different number of dimensions on `data` and `dims`")
        coords = Coordinates(coords, dims)
        coords._assign_parent(self)
        self._coords = coords

        # metadata
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
            dims = tuple(dim for dim in self.dims if not np.isscalar(query[dim]))
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
        elif isinstance(self.data, DaskArray):
            data_repr = f"DaskArray: {_to_human(self.data.nbytes)} ({self.data.dtype})"
        else:
            data_repr = repr(self.data)
        string = "<xdas.DataArray ("
        string += ", ".join([f"{dim}: {size}" for dim, size in self.sizes.items()])
        string += ")>\n"
        string += data_repr
        if self.coords:
            string += "\n" + repr(self.coords)
        dim_without_coords = tuple(dim for dim in self.dims if dim not in self.coords)
        if dim_without_coords:
            string += (
                "\n" + "Dimensions without coordinates: " + ",".join(dim_without_coords)
            )
        return string

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None):
        if dtype is None:
            return self.data.__array__()
        else:
            return self.data.__array__(dtype)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        from .routines import broadcast_coords, broadcast_to  # TODO: circular import

        if not method == "__call__":
            return NotImplemented

        coords = broadcast_coords(
            *tuple(input for input in inputs if isinstance(input, self.__class__))
        )
        inputs = tuple(broadcast_to(input, coords) for input in inputs)

        arrays = tuple(input.values for input in inputs)
        if "out" in kwargs:
            # TODO: check outputs alignements
            kwargs["out"] = tuple(np.asarray(output) for output in kwargs["out"])
        if "where" in kwargs:
            kwargs["where"] = np.asarray(broadcast_to(kwargs["where"], coords))

        outputs = getattr(ufunc, method)(*arrays, **kwargs)
        if isinstance(outputs, tuple):
            return tuple(self.__class__(output, coords) for output in outputs)
        else:
            return self.__class__(outputs, coords)

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

    def conj(self):
        return np.conj(self)

    def conjugate(self):
        return np.conjugate(self)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if not hasattr(value, "__array__"):
            value = np.asarray(value)
        if not value.shape == self.shape:
            raise ValueError(
                f"replacement data must match the same shape. Replacement data "
                f"has shape {value.shape}; original data has shape {self.shape}"
            )
        self._data = value

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, value):
        value = Coordinates(value)
        if not value.dims == self.coords.dims:
            raise ValueError(
                f"replacement coords must have the same dimensions. Replacement coords "
                f"has dims {value.dims}; original coords has dims {self.dims}"
            )
        value._assign_parent(self)
        self._coords = value

    @property
    def dims(self):
        return self.coords.dims

    @dims.setter
    def dims(self, value):
        raise AttributeError(
            "you cannot assign dims on a DataArray, "
            "use .rename(), .transpose() or .swap_dims() instead"
        )

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

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
        return np.prod(self.data.shape) == 0

    @property
    def loc(self):
        return LocIndexer(self)

    def equals(self, other):
        if isinstance(other, self.__class__):
            if not self.dtype == other.dtype:
                return False
            if not np.array_equal(self.values, other.values, equal_nan=True):
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

    def isel(self, indexers=None, drop=False, **indexers_kwargs):
        """
        Return a new DataArray whose data is given by selecting indexes along the
        specified dimension(s).

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given by integers, slice
            objects or arrays.
        drop : bool, optional
            If ``drop=True``, drop coordinates variables in `indexers` instead
            of making them scalar.
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
        da = self[indexers]
        if drop:
            for dim in indexers:
                if da[dim].isscalar():
                    da = da.drop_coords(dim)
        return da

    def sel(
        self, indexers=None, method=None, endpoint=True, drop=False, **indexers_kwargs
    ):
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
        drop : bool, optional
            If ``drop=True``, drop coordinates variables in `indexers` instead
            of making them scalar.
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
        da = self[key]
        if drop:
            for dim in indexers:
                if da[dim].isscalar():
                    da = da.drop_coords(dim)
        return da

    def drop_dims(self, *dims):
        coords = self.coords.drop_dims(*dims)
        return self.__class__(self.data, coords, coords.dims, self.name, self.attrs)

    def drop_coords(self, *names):
        coords = self.coords.drop_coords(*names)
        return self.__class__(self.data, coords, coords.dims, self.name, self.attrs)

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

    def assign_coords(self, coords=None, **coords_kwargs):
        """Assign new coordinates to this object.

        Returns a new object with all the original data in addition to the new
        coordinates.

        Parameters
        ----------
        coords : mapping of dim to coord, optional
            A mapping whose keys are the names of the coordinates and values are the
            coordinates to assign. The mapping will generally be a dict or
            :class:`Coordinates`.

            - If a value is a standard data value that can be parsed by Coordinate —
              the data is simply assigned as a coordinate.
            - A coordinate can also be defined and attached to an existing dimension
              using a tuple with the first element the dimension name and the second
              element the values for this new coordinate.

        **coords_kwargs : optional
            The keyword arguments form of ``coords``.
            One of ``coords`` or ``coords_kwargs`` must be provided.

        Returns
        -------
        assigned : same type as caller
            A new object with the new coordinates in addition to the existing
            data.

        Examples
        --------
        Reset `DataArray` time to start at zero:

        >>> import xdas as xd
        >>> import numpy as np

        >>> da = xd.DataArray(
        ...     data=np.zeros(3),
        ...     coords={"time": np.array([3, 4, 5])},
        ... )
        >>> da
        <xdas.DataArray (time: 3)>
        [0. 0. 0.]
        Coordinates:
          * time (time): [3 ... 5]

        >>> da.assign_coords(time=[0, 1, 2])
        <xdas.DataArray (time: 3)>
        [0. 0. 0.]
        Coordinates:
          * time (time): [0 ... 2]

        The function also accepts dictionary arguments:

        >>> da.assign_coords({"time": [0, 1, 2]})
        <xdas.DataArray (time: 3)>
        [0. 0. 0.]
        Coordinates:
          * time (time): [0 ... 2]

        New coordinate can also be attached to an existing dimension:

        >>> da.assign_coords(relative_time=("time", [0, 1, 2]))
        <xdas.DataArray (time: 3)>
        [0. 0. 0.]
        Coordinates:
          * time (time): [3 ... 5]
            relative_time (time): [0 ... 2]

        """
        da = self.copy(deep=False)
        if coords is None:
            coords = {}
        coords.update(coords_kwargs)
        for name, coord in coords.items():
            da.coords[name] = coord
        return da

    def swap_dims(self, dims_dict=None, **dims_kwargs):
        """
        Returns a new DataArray with swapped dimensions.

        Parameters
        ----------
        dims_dict : dict-like
            Dictionary whose keys are current dimension names and whose values
            are new names.
        **dims_kwargs : {existing_dim: new_dim, ...}, optional
            The keyword arguments form of ``dims_dict``.
            One of dims_dict or dims_kwargs must be provided.

        Returns
        -------
        swapped : DataArray
            DataArray with swapped dimensions.

        Examples
        --------
        >>> import xdas as xd

        >>> da = xd.DataArray(
        ...     data=[0, 1],
        ...     coords={"x": ["a", "b"], "y": ("x", [0, 1])},
        ... )
        >>> da
        <xdas.DataArray (x: 2)>
        [0 1]
        Coordinates:
          * x (x): ['a' 'b']
            y (x): [0 1]

        Make y the dimensional coordinate:

        >>> da.swap_dims({"x": "y"})
        <xdas.DataArray (y: 2)>
        [0 1]
        Coordinates:
            x (y): ['a' 'b']
          * y (y): [0 1]

        Assign a new empy coordinate z as dimensional coordinate.
        Use the **kwargs syntax this time:

        >>> da.swap_dims(x="z")
        <xdas.DataArray (z: 2)>
        [0 1]
        Coordinates:
            x (z): ['a' 'b']
            y (z): [0 1]
        Dimensions without coordinates: z

        """
        if dims_dict is None:
            dims_dict = {}
        dims_dict.update(dims_kwargs)
        for dim in dims_dict:
            if dim not in self.dims:
                raise KeyError(
                    f"dimension {dim} not found in current object with dims {self.dims}"
                )
        dims = tuple(dims_dict[dim] if dim in dims_dict else dim for dim in self.dims)
        coords = {}
        for name, coord in self.coords.copy(deep=False).items():
            if coord.dim in dims_dict:
                coord.dim = dims_dict[coord.dim]
            coords[name] = coord
        return self.__class__(self.data, coords, dims, self.name, self.attrs)

    @property
    def T(self):
        return self.transpose()

    def transpose(self, *dims):
        """
        Return a new DataArray object with transposed dimensions.

        Parameters
        ----------
        *dims : Hashable, optional
            By default, reverse the dimensions. Otherwise, reorder the dimensions to
            this order. The provided `dims` must be a permutation of the original
            dimensions. If `...` is provided, it is replaced by the missing dimensions.

        Returns
        -------
        transposed : DataArray
            The returned DataArray's array is transposed.

        Notes
        -----
        This operation returns a view of this array's data if this later is a
        numpy.ndarray object. Otherwise the data is loaded into memory.

        See Also
        --------
        numpy.transpose

        Examples
        --------
        >>> import xdas as xd
        >>> import numpy as np

        >>> da = xd.DataArray(
        ...     np.arange(2 * 3).reshape(2, 3), {"x": [0, 1], "y": [2, 3, 4]}
        ... )
        >>> da
        <xdas.DataArray (x: 2, y: 3)>
        [[0 1 2]
         [3 4 5]]
        Coordinates:
          * x (x): [0 1]
          * y (y): [2 ... 4]

        >>> da.transpose("y", "x")  # equivalent to not providing any arguments here
        <xdas.DataArray (y: 3, x: 2)>
        [[0 3]
         [1 4]
         [2 5]]
        Coordinates:
          * x (x): [0 1]
          * y (y): [2 ... 4]

        >>> assert da.transpose(..., "x").equals(da.transpose("y", ...))  # equivalent

        """
        if not dims:
            dims = tuple(reversed(self.dims))
        if ... in dims:
            missing_dims = tuple(dim for dim in self.dims if dim not in dims)
            dims = tuple(
                item
                for dim in dims
                for item in (missing_dims if dim is ... else (dim,))
            )
        if not (len(dims) == len(self.dims) and set(dims) == set(self.dims)):
            raise ValueError(f"{dims} must be a permutation of {self.dims}")
        axes = tuple(self.get_axis_num(dim) for dim in dims)
        data = np.transpose(self.data, axes)
        return self.__class__(data, self.coords, dims, self.name, self.attrs)

    def expand_dims(self, dim, axis=0):
        """
        Add an additional dimension at a given axis position.

        Parameters
        ----------
        dim : str
            Dimensions to include on the new variable.
        axis : int
            Axis position where new axis is to be inserted (position(s) on
            the result array).

        Returns
        -------
        expanded : DataArray
            A copy of this object, but with additional dimension.

        Notes
        -----
        This operation returns a view of this array's data if this later is a
        numpy.ndarray object. Otherwise the data is loaded into memory.

        See Also
        --------
        numpy.expand_dims

        Examples
        --------
        >>> import xdas as xd

        >>> da = xd.DataArray([1., 2., 3.], {"x": [0, 1, 2]})
        >>> da
        <xdas.DataArray (x: 3)>
        [1. 2. 3.]
        Coordinates:
          * x (x): [0 ... 2]

        >>> da.expand_dims("y", 0)
        <xdas.DataArray (y: 1, x: 3)>
        [[1. 2. 3.]]
        Coordinates:
          * x (x): [0 ... 2]
        Dimensions without coordinates: y

        """
        if dim in self.dims:
            raise ValueError(f"cannot expand on existing dimension {dim}")
        coords = self.coords.copy()
        if dim in coords:
            if coords[dim].isscalar():
                coords[dim] = [coords[dim].values]
            else:
                raise ValueError(
                    f"cannot expand along {dim} because of existing non-dimensional "
                    f"coordinate {dim}. Consider dropping this coordinate."
                )
        data = np.expand_dims(self.data, axis)
        dims = self.dims[:axis] + (dim,) + self.dims[axis:]
        return self.__class__(data, coords, dims, self.name, self.attrs)

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
        coords = {
            name: (coord.dim if coord.dim else (), coord.values)
            for name, coord in self.coords.items()
        }
        return xr.DataArray(data, coords, self.dims, self.name, self.attrs)

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

    def to_netcdf(self, fname, mode="w", group=None, virtual=None, encoding=None):
        """
        Write DataArray contents to a netCDF file.

        Parameters
        ----------
        fname : str
            Path to which to save this dataset.
        mode : {'w', 'a'}, optional
            Write ('w') or append ('a') mode. If mode='a', the file must already exist.
        group : str, optional
            Path to the netCDF4 group in the given file to open.
        virtual : bool, optional
            Weather to write a virtual dataset. The DataArray data must be a VirtualSource
            or a VirtualLayout. Default (None) is to try to write a virtual dataset if
            possible.
        encoding : dict, optional
            Dictionary of encoding attributes. Because a DataArray contains a unique
            data variable, the encoding dictionary should not contain the variable name.
            For more information on encoding, see the `xarray documentation
            <http://xarray.pydata.org/en/stable/io.html#netcdf>`_. Note that xdas use
            the `h5netcdf` engine to write the data. If you want to use a specific plugin
            for compression, you can use the `hdf5plugin` package. For example, to use the
            ZFP compression, you can use the `hdf5plugin.Zfp` class.

        Examples
        --------
        >>> import os
        >>> import tempfile

        >>> import numpy as np
        >>> import xdas as xd
        >>> import hdf5plugin

        Create some sample data array:

        >>> da = xd.DataArray(np.random.rand(100, 100))

        Save the dataset with ZFP compression:

        >>> encoding = {"chunks": (10, 10), **hdf5plugin.Zfp(accuracy=0.001)}
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     tmpfile = os.path.join(tmpdir, "path.nc")
        ...     da.to_netcdf(tmpfile, encoding=encoding)

        """
        if virtual is None:
            virtual = isinstance(self.data, (VirtualArray, DaskArray))
        ds = xr.Dataset(attrs={"Conventions": "CF-1.9"})
        mappings = []
        for name, coord in self.coords.items():
            if coord.isinterp():
                mappings.append(f"{name}: {name}_indices {name}_values")
                tie_indices = coord.tie_indices
                tie_values = (
                    coord.tie_values.astype("M8[ns]")
                    if np.issubdtype(coord.tie_values.dtype, np.datetime64)
                    else coord.tie_values
                )
                attrs = {
                    "interpolation_name": "linear",
                    "tie_points_mapping": f"{name}_points: {name}_indices {name}_values",
                }
                ds.update(
                    {
                        f"{name}_interpolation": ((), np.nan, attrs),
                        f"{name}_indices": (f"{name}_points", tie_indices),
                        f"{name}_values": (f"{name}_points", tie_values),
                    }
                )
            else:
                ds = ds.assign_coords(
                    {name: (coord.dim, coord.values) if coord.dim else coord.values}
                )
        mapping = " ".join(mappings)
        attrs = {} if self.attrs is None else self.attrs
        attrs |= {"coordinate_interpolation": mapping} if mapping else attrs
        name = "__values__" if self.name is None else self.name
        with h5netcdf.File(fname, mode=mode) as file:
            if group is not None and group not in file:
                file.create_group(group)
            file = file if group is None else file[group]
            file.dimensions.update(self.sizes)
            if not virtual:
                encoding = {} if encoding is None else encoding
                variable = file.create_variable(
                    name,
                    self.dims,
                    self.dtype,
                    data=self.values,
                    **encoding,
                )
            else:
                if encoding is not None:
                    raise ValueError("cannot use `encoding` with in virtual mode")
                if isinstance(self.data, VirtualArray):
                    self.data.to_dataset(file._h5group, name)
                    variable = file._variable_cls(file, name, self.dims)
                    file._variables[name] = variable
                    variable._attach_dim_scales()
                    variable._attach_coords()
                    variable._ensure_dim_id()
                elif isinstance(self.data, DaskArray):
                    variable = file.create_variable(
                        name,
                        self.dims,
                        self.dtype,
                    )
                    variable.attrs.update(
                        {"__dask_array__": np.frombuffer(dumps(self.data), "uint8")}
                    )
                else:
                    raise ValueError(
                        "can only use `virtual=True` with a virtual array as data"
                    )
            if attrs:
                variable.attrs.update(attrs)
        ds.to_netcdf(fname, mode="a", group=group, engine="h5netcdf")

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
        with xr.open_dataset(fname, group=group, engine="h5netcdf") as ds:
            if not ("Conventions" in ds.attrs and "CF" in ds.attrs["Conventions"]):
                raise TypeError(
                    "file format not recognized. please provide the file format "
                    "with the `engine` keyword argument"
                )
            if len(ds) == 1:
                name, da = next(iter(ds.items()))
                coords = {
                    name: (
                        (
                            coord.dims[0],
                            (
                                coord.values.astype("U")
                                if coord.dtype == np.dtype("O")
                                else coord.values
                            ),
                        )
                        if coord.dims
                        else coord.values
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
                        (
                            coord.dims[0],
                            (
                                coord.values.astype("U")
                                if coord.dtype == np.dtype("O")
                                else coord.values
                            ),
                        )
                        if coord.dims
                        else coord.values
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
            variable = file[name]
            if "__dask_array__" in variable.attrs:
                data = loads(da.attrs.pop("__dask_array__"))
            else:
                data = VirtualSource(file[name])
        return cls(data, coords, da.dims, da.name, None if da.attrs == {} else da.attrs)

    def to_dict(self):
        """Convert the DataArray to a dictionary."""
        if isinstance(self.data, VirtualArray):
            raise NotImplementedError("cannot convert a virtual array to a dictionary")
        elif isinstance(self.data, np.ndarray):
            data = self.data.tolist()
        elif isinstance(self.data, DaskArray):
            data = to_dict(self.data)
        return {
            "data": data,
            "coords": self.coords.to_dict()["coords"],
            "dims": self.dims,
            "name": self.name,
            "attrs": self.attrs,
        }

    @classmethod
    def from_dict(cls, dct):
        """Create a DataArray from a dictionary."""
        if isinstance(dct["data"], list):
            data = np.array(dct["data"])
        elif isinstance(dct["data"], dict):
            data = from_dict(dct["data"])
        else:
            raise ValueError("data must be a list or a dictionary")
        coords = Coordinates.from_dict({key: dct[key] for key in ["coords", "dims"]})
        return cls(data, coords, dct["dims"], dct["name"], dct["attrs"])

    def plot(self, *args, **kwargs):
        """
        Plot a DataArray.

        This plot function uses the xarray way of plotting depending on the
        number of dimensions your data has. Please for the args and kwargs
        refer to the corresponding xarray functions.

        For a DataArray with one dimension: refer to `xarray.plot.line <https://docs.xarray.dev/en/stable/generated/xarray.plot.line.html>`_.
        For a DataArray of 2 dimensions or more: refer to `xarray.plot.imshow <https://docs.xarray.dev/en/stable/generated/xarray.plot.imshow.html>`_.
        For other: refer to `xarray.plot.hist <https://docs.xarray.dev/en/latest/generated/xarray.plot.hist.html#xarray.plot.hist>`_.

        Parameters
        ----------
        *args:
            See the corresponding xarray args.
        **kwargs:
            See the corresponding xarray kwargs.

        Returns
        -------
        artist
            The same type of primitive artist that the wrapped Matplotlib function returns.
        """
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
        super().__init__({dim: size for dim, size in zip(obj.dims, obj.shape)})

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
