import os
from pathlib import Path

import zarr
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


class XdasZarrEngine(Engine, name="zarr"):

    def open_dataarray(self, fname, **kwargs):
        return open_dataarray_zarr(fname, **kwargs)

    def save_dataarray(self, da, fname, **kwargs):
        return save_dataarray_zarr(da, fname, **kwargs)

    def open_datacollection(self, fname, **kwargs):
        return open_datamapping(fname, **kwargs)

    def save_datacollection(self, dc, fname, **kwargs):
        return save_datamapping(dc, fname, **kwargs)

def open_dataarray_zarr(fname, **kwargs):
    xr_da = xr.open_zarr(fname, **kwargs)
    return DataArray(xr_da.data, xr_da.coords, xr_da.dims)

def save_dataarray_zarr(
    da, fname, mode="w", group=None, virtual=None, encoding=None, create_dirs=False
):
    dx = da.to_xarray()
    dx.name = "data"

    ### TODO: da coords != dx coords. Write without xarray?
    # chunks=(1000,1000)
    # codec = codecs.ZFPY(mode=4, tolerance=tolerance)
    # z_write = zarr.array(da.data, chunks=chunks, codecs=[codec], store=fname, overwrite=True)

    xr.DataArray.to_zarr(dx,fname, mode=mode, group=group, encoding=encoding)


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
        dm = DataMapping({}, name=None if name == "collection" else name)
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
        dm[key].to_zarr(
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
