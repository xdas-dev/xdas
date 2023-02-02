import copy
import os
import re
import warnings
from tempfile import TemporaryDirectory

import dask.array as da
import h5py
import numpy as np
import xarray as xr


class DataCollection(dict):
    def to_hdf(self, fname, virtual=False):
        with h5py.File(fname, "w") as file:
            for key in self:
                group = file.create_group(key)
                self[key].to_group(group, virtual=virtual)

    @classmethod
    def from_hdf(cls, fname):
        with h5py.File(fname, "r") as file:
            self = cls()
            for key in file.keys():
                self[key] = Database.from_group(file[key])
        return self


class Database:
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
        return repr(self.data) + "\n" + repr(self.coords)

    def __array__(self, dtype=None):
        if isinstance(self.data, h5py.VirtualSource):
            with TemporaryDirectory() as tmpdirname:
                fname = os.path.join(tmpdirname, "vds.h5")
                with h5py.File(fname, "w") as file:
                    layout = h5py.VirtualLayout(self.data.shape, self.data.dtype)
                    layout[...] = self.data
                    dataset = file.create_virtual_dataset("data", layout)
                with h5py.File(fname, "r") as file:
                    dataset = file["data"]
                    out = dataset[...]
            return out
        else:
            return self.data.__array__(dtype)

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

    def get_axis_num(self, dim):
        return self.dims.index(dim)

    @property
    def sizes(self):
        return {dim: len(coord) for dim, coord in self.coords.items()}

    @property
    def values(self):
        return self.__array__()

    @property
    def loc(self):
        return LocIndexer(self)

    def isel(self, **kwargs):
        return self[kwargs]

    def sel(self, **kwargs):
        return self.loc[kwargs]

    def copy(self, deep=True, data=None):
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
        return self.compute()

    def compute(self):
        return self.copy(data=self.data.compute())

    def to_xarray(self):
        return xr.DataArray(
            data=self.__array__(),
            coords=self.coords,
            dims=self.dims,
            name=self.name,
            attrs=self.attrs,
        )

    @classmethod
    def from_netcdf(cls, *args, **kwargs):
        dataset = xr.open_dataset(*args, **kwargs)
        data = [
            var for var in dataset.values() if "coordinate_interpolation" in var.attrs
        ]
        if len(data) == 1:
            data = data[0]
        else:
            ValueError("several possible data arrays detected")
        coords = Coordinates()
        mapping = data.attrs.pop("coordinate_interpolation")
        matches = re.findall(r"(\w+): (\w+) (\w+)", mapping)
        for match in matches:
            dim, indices, values = match
            coords[dim] = Coordinate(dataset[indices], dataset[values])
        return cls(data, coords)

    def to_netcdf(self, *args, **kwargs):
        datas = []
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
                dims=(f"{dim}_points"),
            )
            values = xr.DataArray(
                name=f"{dim}_values",
                data=self.coords[dim].tie_values,
                dims=(f"{dim}_points"),
            )
            datas.extend([interpolation, indices, values])
        data = self.data.copy(deep=False)
        data.attrs["coordinate_interpolation"] = mapping
        datas.append(data)
        dataset = xr.Dataset(
            data_vars={xarr.name: xarr for xarr in datas},
            attrs={"Conventions": "CF-1.9"},
        )
        dataset.to_netcdf(*args, **kwargs)

    @classmethod
    def from_hdf(cls, fname):
        with h5py.File(fname, "r") as file:
            return cls.from_group(file)

    @classmethod
    def from_group(cls, group):
        data = h5py.VirtualSource(group["data"])
        time_tie_indices = np.asarray(group["time_tie_indices"])
        time_tie_values = np.asarray(group["time_tie_values"]).astype("datetime64[us]")
        distance_tie_indices = np.asarray(group["distance_tie_indices"])
        distance_tie_values = np.asarray(group["distance_tie_values"])
        time_coordinate = Coordinate(time_tie_indices, time_tie_values)
        distance_coordinate = Coordinate(distance_tie_indices, distance_tie_values)
        coords = Coordinates(time=time_coordinate, distance=distance_coordinate)
        return cls(data, coords)

    def to_hdf(self, fname, virtual=False):
        with h5py.File(fname, "w") as file:
            self.to_group(file, virtual=virtual)

    def to_group(self, group, virtual=False):
        if not virtual:
            group.create_dataset("data", data=self.values)
        elif virtual and isinstance(self.data, h5py.VirtualSource):
            layout = h5py.VirtualLayout(self.shape, self.dtype)
            layout[...] = self.data
            group.create_virtual_dataset("data", layout, fillvalue=np.nan)
        else:
            raise ValueError("can only use `virtual=True` with a VirtualSource")
        for dim in self.dims:
            tie_indices = self[dim].tie_indices
            tie_values = self[dim].tie_values
            if np.issubdtype(tie_values.dtype, np.datetime64):
                tie_values = tie_values.astype("datetime64[us]").astype("int")
            group.create_dataset(f"{dim}_tie_indices", data=tie_indices)
            group.create_dataset(f"{dim}_tie_values", data=tie_values)


class Coordinates(dict):
    @property
    def dims(self):
        return tuple(self.keys())

    @property
    def ndim(self):
        return len(self)

    def __repr__(self):
        s = "Coordinates:\n"
        for dim, coord in self.items():
            s += f"  * {dim}".ljust(12)
            s += f"({dim}) "
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
                f"{len(self)} points from {self.tie_values[0]} "
                f"to {self.tie_values[-1]}"
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

    def __array__(self, dtype=None):
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
        start = self.format_index(start, bounds="clip")
        stop = self.format_index(stop, bounds="clip")
        return slice(start, stop, step)

    def get_value(self, index):
        index = self.format_index(index)
        return linear_interpolate(index, self.tie_indices, self.tie_values)

    def get_index(self, value, method=None):
        value = np.asarray(value)
        if method is None:
            index = linear_interpolate(value, self.tie_values, self.tie_indices)
            index = np.rint(index).astype("int")
            if not np.allclose(self.get_value(index), value):
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
        start_index, stop_index = index_slice.start, index_slice.stop
        if stop_index - start_index <= 0:
            return Coordinate([], [])
        elif stop_index - start_index == 1:
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
            return Coordinate(tie_indices, tie_values)

    def to_index(self, item):
        if isinstance(item, slice):
            return self.get_index_slice(item)
        else:
            return self.get_index(item)

    def simplify(self, epsilon):
        self.tie_indices, self.tie_values = douglas_peucker(
            self.tie_indices, self.tie_values, epsilon
        )


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
