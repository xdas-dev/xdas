import re

import dask.array as da
import h5py
import numpy as np
import xarray as xr


class Database:
    def __init__(self, data, coords, dims=None, name=None, attrs=None):
        self.data = data
        self.coords = coords
        if dims is None:
            self.dims = tuple(coords.keys())
        else:
            raise NotImplemented()
        self.name = name
        self.attrs = attrs

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.coords[item]
        else:
            query = get_query(item, self.coords.dims)
            data = self.data.__getitem__(tuple(query.values()))
            dct = {dim: self.coords[dim][query[dim]] for dim in query}
            coords = Coordinates(dct)
            return self.__class__(data, coords)

    @property
    def loc(self):
        return LocIndexer(self)

    def isel(self, **kwargs):
        return self[kwargs]

    def sel(self, **kwargs):
        return self.loc[kwargs]

    def __repr__(self):
        return repr(self.data) + "\n" + repr(self.coords)

    def to_xarray(self):
        return xr.DataArray(
            data=self.data,
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
        file = h5py.File(fname, "r")

        data = da.from_array(file["data"], chunks=(-1, -1))

        time_tie_indices = np.asarray(file["time_tie_indices"])
        time_tie_values = np.asarray(file["time_tie_values"]).astype("datetime64[us]")
        time_coordinate = Coordinate(time_tie_indices, time_tie_values)

        distance_tie_indices = np.asarray(file["distance_tie_indices"])
        distance_tie_values = np.asarray(file["distance_tie_values"])
        distance_coordinate = Coordinate(distance_tie_indices, distance_tie_values)

        coords = Coordinates(time=time_coordinate, distance=distance_coordinate)

        return cls(data, coords)


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

    def __getitem__(self, item):
        item = self.obj.coords.to_index(item)
        return self.obj[item]


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

    def __len__(self):
        return len(self.tie_indices)

    def __repr__(self):
        return (
            f"{len(self.tie_indices)} tie points from {self.tie_values[0]} "
            f"to {self.tie_values[-1]}"
        )

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.slice(item)
        else:
            return self.get_value(item)

    def __array__(self, dtype=None):
        return self.values()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise NotImplemented()

    def __array_function__(self, func, types, args, kwargs):
        raise NotImplemented()

    @property
    def dtype(self):
        return self.tie_values.dtype

    @property
    def ndim(self):
        return 1

    @property
    def shape(self):
        return (len(self),)

    def get_value(self, index):
        return _linear_interpolate(index, self.tie_indices, self.tie_values)

    def get_index(self, value, method="nearest"):
        index = _linear_interpolate(value, self.tie_values, self.tie_indices)
        if method == "nearest":
            return np.rint(index).astype("int")
        elif method == "before":
            return np.floor(index).astype("int")
        elif method == "after":
            return np.ceil(index).astype("int")
        else:
            raise ValueError("valid methods are: 'nearest', 'before', 'after'")

    def indices(self):
        return np.arange(self.tie_indices[-1] + 1)

    def values(self):
        return self.get_value(self.indices())

    def get_index_slice(self, value_slice):
        if value_slice.start is None:
            start = None
        else:
            start = self.get_index(value_slice.start, method="after")
        if value_slice.stop is None:
            stop = None
        else:
            end = self.get_index(value_slice.stop, method="before")
            stop = end + 1
        return slice(start, stop)

    def slice(self, index_slice):
        if index_slice.start is None:
            start_index = self.tie_indices[0]
        else:
            start_index = index_slice.start
        if index_slice.stop is None:
            end_index = self.tie_indices[-1]
        else:
            end_index = index_slice.stop - 1
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
        self.tie_indices, self.tie_values = _simplify(
            self.tie_indices, self.tie_values, epsilon
        )


class ScaleOffset:
    def __init__(self, scale, offset):
        self.scale = scale
        self.offset = offset

    @classmethod
    def floatize(cls, arr):
        if np.issubdtype(arr.dtype, np.datetime64):
            scale = np.timedelta64(1, "us")
            offset = arr[0]
        else:
            scale = 1.0
            offset = 0.0
        return cls(scale, offset)

    def direct(self, arr):
        return (arr - self.offset) / self.scale

    def inverse(self, arr):
        if np.issubdtype(np.asarray(self.scale).dtype, np.timedelta64):
            arr = np.rint(arr)
        return self.scale * arr + self.offset


def _linear_interpolate(x, xp, fp):
    if not _is_strictly_increasing(xp):
        raise ValueError("xp must be strictly increasing")
    x_transform = ScaleOffset.floatize(xp)
    f_transform = ScaleOffset.floatize(fp)
    x = x_transform.direct(x)
    xp = x_transform.direct(xp)
    fp = f_transform.direct(fp)
    f = np.interp(x, xp, fp)
    f = f_transform.inverse(f)
    return f


def _is_strictly_increasing(x):
    if np.issubdtype(x.dtype, np.datetime64):
        return np.all(np.diff(x) > np.timedelta64(0, "us"))
    else:
        return np.all(np.diff(x) > 0)


def _simplify(x, y, epsilon):
    mask = np.ones(len(x), dtype=bool)
    stack = [(0, len(x))]
    while stack:
        start, stop = stack.pop()
        ysimple = _linear_interpolate(
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
