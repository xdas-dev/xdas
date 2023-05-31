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
            query = get_query(key, self.coords.dims)
            data = self.data.__getitem__(tuple(query.values()))
            dct = {dim: self.coords[dim][query[dim]] for dim in query}
            coords = Coordinates(dct)
            return self.__class__(data, coords)

    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.coords[key] = value
        else:
            query = get_query(key, self.coords.dims)
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


class Coordinates(dict):
    """
    A dictionary whose keys are dimension names and values are Coordinate objects.
    """

    @property
    def dims(self):
        return tuple(self.keys())

    @property
    def ndim(self):
        return len(self)

    def __repr__(self):
        s = "Coordinates:\n"
        for dim, coord in self.items():
            s += f"  * {dim}: "
            s += repr(coord) + "\n"
        return s

    def to_index(self, item):
        query = get_query(item, self.dims)
        return {dim: self[dim].to_index(query[dim]) for dim in query}


class LocIndexer:
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, key):
        key = self.obj.coords.to_index(key)
        return self.obj[key]

    def __setitem__(self, key, value):
        key = self.obj.coords.to_index(key)
        self.obj[key] = value


def get_query(item, dims):
    query = {dim: slice(None) for dim in dims}
    if isinstance(item, dict):
        query.update(item)
    elif isinstance(item, tuple):
        for k in range(len(item)):
            query[dims[k]] = item[k]
    else:
        query[dims[0]] = item
    return query


class Coordinate:
    """
    Array-like object used to represent piecewise evenly spaced coordinates using the
    CF convention.

    The coordinate ticks are describes by the mean of tie points that are interpolated
    when intermediate values are required. For more details see:
    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#compression-by-coordinate-subsampling

    Coordinate objects provides label based selections methods.

    Parameters
    ----------
    tie_indices : sequence of integers
        The indices of the tie points. Must include index 0 and be strictly ncreasing.
    tie_values : sequence of float or datetime64
        The values of the tie points. Must be strictly increasing to enable label-based
        selection. The len of `tie_indices` and `tie_values` sizes must match.
    """

    def __init__(self, tie_indices, tie_values):
        self.tie_indices = np.asarray(tie_indices)
        self.tie_values = np.asarray(tie_values)
        self.kind = "linear"

    def __bool__(self):
        if len(self.tie_indices) == 0 or len(self.tie_values) == 0:
            return False
        else:
            return True

    def __len__(self):
        if self:
            return self.tie_indices[-1] - self.tie_indices[0] + 1
        else:
            return 0

    def __repr__(self):
        if len(self) == 0:
            return "empty coordinate"
        elif len(self) == 1:
            return f"one point at {self.tie_values[0]}"
        else:
            return (
                f"{len(self.tie_indices)} tie points from {self.tie_values[0]} to "
                f"{self.tie_values[-1]}"
            )

    def __eq__(self, other):
        return np.array_equal(self.tie_indices, other.tie_indices) and np.array_equal(
            self.tie_values, other.tie_values
        )

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.slice_index(item)
        else:
            return self.get_value(item)

    def __array__(self):
        return self.values

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise NotImplementedError()

    def __array_function__(self, func, types, args, kwargs):
        raise NotImplementedError()

    @property
    def dtype(self):
        return self.tie_values.dtype

    @property
    def ndim(self):
        return 1

    @property
    def shape(self):
        return (len(self),)

    @property
    def indices(self):
        return np.arange(self.tie_indices[-1] + 1)

    @property
    def values(self):
        return self.get_value(self.indices)

    def format_index(self, idx, bounds="raise"):
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

    def format_index_slice(self, slc):
        start = slc.start
        stop = slc.stop
        step = slc.step
        if start is None:
            start = 0
        if stop is None:
            stop = len(self)
        if step is None:
            step = 1
        start = self.format_index(start, bounds="clip")
        stop = self.format_index(stop, bounds="clip")
        return slice(start, stop, step)

    def get_value(self, index):
        index = self.format_index(index)
        return linear_interpolate(index, self.tie_indices, self.tie_values)

    def get_index(self, value, method=None):
        if isinstance(value, str):
            value = np.datetime64(value)
        else:
            value = np.asarray(value)
        if method is None:
            index = linear_interpolate(value, self.tie_values, self.tie_indices)
            index = np.rint(index).astype("int")
            index_value = self.get_value(index)
            if np.issubdtype(self.dtype, np.datetime64):
                if not np.all(index_value == value):
                    raise KeyError("value not found in index")
                else:
                    return index
            else:
                if not np.allclose(index_value, value):
                    raise KeyError("value not found in index")
                else:
                    return index
        elif method == "nearest":
            index = linear_interpolate(value, self.tie_values, self.tie_indices)
            return np.rint(index).astype("int")
        elif method == "before":
            index = linear_interpolate(
                value, self.tie_values, self.tie_indices, left=np.nan
            )
            if np.any(np.isnan(index)):
                raise KeyError("value not found in index")
            else:
                return np.floor(index).astype("int")
        elif method == "after":
            index = linear_interpolate(
                value, self.tie_values, self.tie_indices, right=np.nan
            )
            if np.any(np.isnan(index)):
                raise KeyError("value not found in index")
            else:
                return np.ceil(index).astype("int")
        else:
            raise ValueError("valid methods are: 'nearest', 'before', 'after'")

    def get_index_slice(self, value_slice):
        if value_slice.start is None:
            start = None
        else:
            try:
                start = self.get_index(value_slice.start, method="after")
            except KeyError:
                start = len(self)
        if value_slice.stop is None:
            stop = None
        else:
            try:
                end = self.get_index(value_slice.stop, method="before")
                stop = end + 1
            except KeyError:
                stop = 0
        return slice(start, stop)

    def slice_index(self, index_slice):
        index_slice = self.format_index_slice(index_slice)
        start_index, stop_index, step_index = (
            index_slice.start,
            index_slice.stop,
            index_slice.step,
        )
        if stop_index - start_index <= 0:
            return Coordinate([], [])
        elif (stop_index - start_index) <= step_index:
            tie_indices = [0]
            tie_values = [self.get_value(start_index)]
            return Coordinate(tie_indices, tie_values)
        else:
            end_index = stop_index - 1
            start_value = self.get_value(start_index)
            end_value = self.get_value(end_index)
            mask = (start_index < self.tie_indices) & (self.tie_indices < end_index)
            tie_indices = np.insert(
                self.tie_indices[mask],
                (0, self.tie_indices[mask].size),
                (start_index, end_index),
            )
            tie_values = np.insert(
                self.tie_values[mask],
                (0, self.tie_values[mask].size),
                (start_value, end_value),
            )
            tie_indices -= tie_indices[0]
            coord = Coordinate(tie_indices, tie_values)
            if step_index != 1:
                coord = coord.decimate(step_index)
            return coord

    def decimate(self, q):
        tie_indices = (self.tie_indices // q) * q
        for k in range(1, len(tie_indices) - 1):
            if tie_indices[k] == tie_indices[k - 1]:
                tie_indices[k] += q
        tie_values = [self.get_value(idx) for idx in tie_indices]
        tie_indices //= q
        return self.__class__(tie_indices, tie_values)

    def to_index(self, item):
        if isinstance(item, slice):
            return self.get_index_slice(item)
        else:
            return self.get_index(item)

    def simplify(self, tolerance):
        tie_indices, tie_values = douglas_peucker(
            self.tie_indices, self.tie_values, tolerance
        )
        return self.__class__(tie_indices, tie_values)


class ScaleOffset:
    def __init__(self, scale, offset):
        self.scale = scale
        self.offset = offset

    def __eq__(self, other):
        try:
            return (self.scale - other.scale == 0) and (self.offset - other.offset == 0)
        except:
            return False

    @classmethod
    def floatize(cls, arr):
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.datetime64):
            unit, count = np.datetime_data(arr.dtype)
            scale = np.timedelta64(count, unit)
            offset = np.min(arr) + (np.max(arr) - np.min(arr)) / 2
        else:
            scale = 1.0
            offset = 0.0
        transform = cls(scale, offset)
        transform.check_resolution(arr)
        return transform

    def direct(self, arr):
        arr = np.asarray(arr)
        self.check_resolution(arr)
        return (arr - self.offset) / self.scale

    def inverse(self, arr):
        arr = np.asarray(arr)
        if np.issubdtype(np.asarray(self.scale).dtype, np.timedelta64):
            arr = np.rint(arr)
        return self.scale * arr + self.offset

    def check_resolution(self, arr):
        arr = np.asarray(arr)
        nmax = 2 ** np.finfo("float").nmant
        if not np.all((arr - self.offset).astype("int") < nmax):
            warnings.warn(
                "float resolution is not sufficient to represent the full integer range"
            )


def linear_interpolate(x, xp, fp, left=None, right=None):
    if not is_strictly_increasing(xp):
        raise ValueError("xp must be strictly increasing")
    x_transform = ScaleOffset.floatize(xp)
    f_transform = ScaleOffset.floatize(fp)
    x = x_transform.direct(x)
    xp = x_transform.direct(xp)
    fp = f_transform.direct(fp)
    f = np.interp(x, xp, fp, left=left, right=right)
    f = f_transform.inverse(f)
    return f


def is_strictly_increasing(x):
    if np.issubdtype(x.dtype, np.datetime64):
        return np.all(np.diff(x) > np.timedelta64(0, "us"))
    else:
        return np.all(np.diff(x) > 0)


def douglas_peucker(x, y, epsilon):
    mask = np.ones(len(x), dtype=bool)
    stack = [(0, len(x))]
    while stack:
        start, stop = stack.pop()
        ysimple = linear_interpolate(
            x[start:stop],
            x[[start, stop - 1]],
            y[[start, stop - 1]],
        )
        d = np.abs(y[start:stop] - ysimple)
        index = np.argmax(d)
        dmax = d[index]
        index += start
        if dmax > epsilon:
            stack.append([start, index + 1])
            stack.append([index, stop])
        else:
            mask[start + 1 : stop - 1] = False
    return x[mask], y[mask]
