import os
from pathlib import Path

import h5netcdf
import h5py
import hdf5plugin
import xarray as xr
from dask.array import Array as DaskArray

from ..coordinates import Coordinates
from ..core.dataarray import DataArray
from ..core.datacollection import DataCollection, DataMapping, DataSequence
from ..dask.core import create_variable, loads
from ..virtual import VirtualArray, VirtualSource
from .core import Engine


class XdasEngine(Engine, name="xdas"):
    # TODO: does not make sense for the XdasEngine...
    _supported_vtypes = ["hdf5"]
    _supported_ctypes = {
        "time": ["interpolated", "sampled", "dense"],
        "distance": ["interpolated", "sampled", "dense"],
    }

    def open_dataarray(self, fname, **kwargs):
        return open_dataarray(fname, **kwargs)

    def save_dataarray(self, da, fname, **kwargs):
        return save_dataarray(da, fname, **kwargs)

    def open_datacollection(self, fname, **kwargs):
        return open_datamapping(fname, **kwargs)

    def save_datacollection(self, dc, fname, **kwargs):
        return save_datamapping(dc, fname, **kwargs)


def open_dataarray(fname, group=None):
    if isinstance(fname, Path):
        fname = str(fname)

    # read metadata
    with xr.open_dataset(
        fname, group=group, engine="h5netcdf", decode_timedelta=False
    ) as dataset:
        # check file format
        if not (
            "Conventions" in dataset.attrs and "CF" in dataset.attrs["Conventions"]
        ):
            raise TypeError(
                "file format not recognized. please provide the file format "
                "with the `engine` keyword argument"
            )

        # identify the "main" data array
        if len(dataset) == 1:
            name = next(iter(dataset.keys()))
        else:
            data_vars = {
                key: var
                for key, var in dataset.items()
                if any("coordinate" in attr for attr in var.attrs)
            }
            if len(data_vars) == 1:
                name = next(iter(data_vars.keys()))
            else:
                raise ValueError("several possible data arrays detected")

        # read coordinates
        coords = Coordinates.from_dataset(dataset, name)

    # read data
    if "__dask_array__" in dataset[name].attrs:
        data = loads(dataset[name].attrs.pop("__dask_array__"))
    else:
        with h5py.File(fname) as file:
            if group:
                file = file[group]
            variable = file["__values__" if name is None else name]
            data = VirtualSource(variable)

    # pack everything
    return DataArray(
        data,
        coords,
        dataset[name].dims,
        name,
        None if dataset[name].attrs == {} else dataset[name].attrs,
    )


def save_dataarray(
    da, fname, mode="w", group=None, virtual=None, encoding=None, create_dirs=False
):
    if isinstance(fname, Path):
        fname = str(fname)

    if virtual is None:
        virtual = isinstance(da.data, (VirtualArray, DaskArray))

    # initialize
    dataset = xr.Dataset(attrs={"Conventions": "CF-1.9"})
    variable_attrs = {} if da.attrs is None else da.attrs
    variable_name = "__values__" if da.name is None else da.name

    # prepare metadata
    for coord in da.coords.values():
        dataset, variable_attrs = coord.to_dataset(dataset, variable_attrs)

    # create parent directories if needed
    if create_dirs:
        dirname = os.path.dirname(fname)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    # write data
    with h5netcdf.File(fname, mode=mode) as file:
        # group
        if group is not None and group not in file:
            file.create_group(group)
        file = file if group is None else file[group]

        # dims
        file.dimensions.update(da.sizes)

        # variable
        if not virtual:
            encoding = {} if encoding is None else encoding
            variable = file.create_variable(
                variable_name, da.dims, da.dtype, data=da.values, **encoding
            )
        else:
            if encoding is not None:
                raise ValueError("cannot use `encoding` with in virtual mode")
            if isinstance(da.data, VirtualArray):
                variable = da.data.create_variable(
                    file, variable_name, da.dims, da.dtype
                )
            elif isinstance(da.data, DaskArray):
                variable = create_variable(
                    da.data, file, variable_name, da.dims, da.dtype
                )
            else:
                raise ValueError(
                    "can only use `virtual=True` with a virtual array as data"
                )

        # attrs
        if variable_attrs:
            variable.attrs.update(variable_attrs)

    # write metadata
    dataset.to_netcdf(fname, mode="a", group=group, engine="h5netcdf")


def open_datacollection(fname, group=None):
    if isinstance(fname, Path):
        fname = str(fname)
    dc = open_datamapping(fname, group)
    try:
        keys = [int(key) for key in dc.keys()]
        if keys == list(range(len(keys))):
            return DataSequence.from_mapping(dc)
        else:
            return dc
    except ValueError:
        return dc


def save_datacollection(
    dc, fname, mode="w", group=None, virtual=None, encoding=None, create_dirs=False
):
    if isinstance(fname, Path):
        fname = str(fname)

    if isinstance(dc, DataSequence):
        save_datasequence(dc, fname, mode, group, virtual, encoding, create_dirs)
    elif isinstance(dc, DataCollection):
        save_datamapping(dc, fname, mode, group, virtual, encoding, create_dirs)
    else:
        raise ValueError("can only save a DataCollection or a DataSequence")


def open_datamapping(fname, group=None):
    if isinstance(fname, Path):
        fname = str(fname)

    with h5py.File(fname, "r") as file:
        if group is None:
            group = file[list(file.keys())[0]]
        else:
            group = file[group]
        name = group.name.split("/")[-1]
        if isinstance(group, h5py.Dataset):
            raise ValueError(
                "it looks like you are trying to open a data array as a data collection."
            )
        else:
            if not isinstance(group, h5py.Group):
                raise RuntimeError(
                    "something went wrong while opening the data collection."
                )
        keys = list(group.keys())
        dm = DataCollection({}, name=None if name == "collection" else name)
        for key in keys:
            subgroup = group[key]
            if _get_depth(subgroup) == 0:
                dm[key] = DataArray.from_netcdf(fname, subgroup.name)
            else:
                subgroup = subgroup[list(subgroup.keys())[0]]
                dm[key] = DataCollection.from_netcdf(fname, subgroup.name)
    return dm


def save_datamapping(
    dm, fname, mode="w", group=None, virtual=None, encoding=None, create_dirs=False
):
    if mode == "w" and group is None and os.path.exists(fname):
        os.remove(fname)
    for key in dm:
        name = dm.name if dm.name is not None else "collection"
        location = "/".join([name, str(key)])
        if group is not None:
            location = "/".join([group, location])
        if create_dirs:
            dirname = os.path.dirname(fname)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
        dm[key].to_netcdf(
            fname, mode="a", group=location, virtual=virtual, encoding=encoding
        )


def open_datasequence(fname, group=None):
    dm = open_datamapping(fname, group)
    return DataSequence.from_mapping(dm)


def save_datasequence(
    ds, fname, mode="w", group=None, virtual=None, encoding=None, create_dirs=False
):
    dm = ds.to_mapping()
    save_datamapping(dm, fname, mode, group, virtual, encoding, create_dirs)


def _get_depth(group):
    if not isinstance(group, h5py.Group):
        raise ValueError("not a group")
    depths = []
    group.visit(lambda name: depths.append(name.count("/")))
    return max(depths)
