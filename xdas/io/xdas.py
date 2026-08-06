"""
I/O engine for the native xdas HDF5/NetCDF4 format (:class:`XdasEngine`).

Supports :class:`DataArray`, :class:`DataSequence`, and :class:`DataMapping`.

Whatever the shape of the object, the file is opened once in each
direction: reading walks a single :func:`xarray.open_datatree`, writing
lands every group's metadata through a single
:meth:`xarray.DataTree.to_netcdf` and the data variables through one
writable `h5netcdf` handle. One handle matters on write: every writable
`h5netcdf` open walks the whole file to find the next free dimension
id, so per-array opens made saving a collection quadratic in its size.
"""

import json
import os
import warnings
from pathlib import Path
from typing import ClassVar

import h5netcdf
import h5py
import hdf5plugin  # noqa
import xarray as xr
from dask.array import Array as DaskArray

from ..coordinates import Coordinates
from ..core import DataArray, DataCollection, DataMapping, DataSequence
from ..dask import create_variable, loads
from ..virtual import TileArray, VirtualBackend
from ..virtual.tiles import TILES_GROUP
from .core import Engine


class XdasEngine(Engine, name="xdas"):
    """
    Engine for the native xdas HDF5/NetCDF4 format.

    Parameters
    ----------
    vtype : str, optional
        The virtualization type to use. Default to "hdf5".
    ctype : str or dict, optional
        Ignored: the native format stores coordinates as they were written.
    group : str, optional
        The location of the data array within the file. Default to the root group.

    """

    _supported_vtypes: ClassVar[list] = ["hdf5", "tiles"]

    def __init__(self, vtype=None, ctype=None, group=None):
        super().__init__(vtype, ctype)
        self.group = group

    def open_dataarray(self, fname):
        """Delegate to module-level :func:`open_dataarray`."""
        return open_dataarray(fname, group=self.group, vtype=self.vtype)

    def save_dataarray(self, da, fname, **kwargs):
        """Delegate to module-level :func:`save_dataarray`."""
        return save_dataarray(da, fname, **kwargs)

    def open_datacollection(self, fname):
        """Delegate to module-level :func:`open_datacollection`."""
        return open_datacollection(fname, group=self.group)

    def save_datacollection(self, dc, fname, **kwargs):
        """Delegate to module-level :func:`save_datacollection`."""
        return save_datacollection(dc, fname, **kwargs)

    @staticmethod
    def load_tile(path, selection, *, dataset):
        """Read a source selection of a native xdas file.

        The variable is read with h5py, which resolves any HDF5 virtual
        dataset the file may store transparently.

        Parameters
        ----------
        path : str
            Path of the NetCDF4/HDF5 file.
        selection : tuple of slice
            The source selection to read, one possibly strided slice per
            axis.
        dataset : str
            Location of the data variable within the file.
        """
        with h5py.File(path, "r") as file:
            return file[dataset][selection]


def open_dataarray(fname, group=None, vtype=None):
    """
    Read a :class:`DataArray` from a native xdas NetCDF4/HDF5 file.

    Parameters
    ----------
    fname : str or Path
        Path to the file.
    group : str, optional
        HDF5 group path inside the file.
    vtype : str, optional
        Virtualization backing of the returned data: ``"hdf5"`` (default,
        an HDF5 virtual source) or ``"tiles"`` (a lazy
        :class:`~xdas.virtual.TileArray` over the stored variable). Files
        that store a tile manifest reopen as tile arrays regardless.

    Returns
    -------
    DataArray
    """
    if isinstance(fname, Path):
        fname = str(fname)

    # one open covers the data array and any tile manifest beside it.
    # "access" is xarray's own default and silences its warning; "sort"
    # would rescan every group of the file on each open.
    with xr.open_datatree(
        fname,
        group=group,
        engine="h5netcdf",
        decode_timedelta=False,
        phony_dims="access",
    ) as node:
        return _read_dataarray(node, fname, group, vtype)


def _read_dataarray(node, fname, group=None, vtype=None):
    """Build a :class:`DataArray` from the open tree *node* holding it.

    Parameters
    ----------
    node : xarray.DataTree
        The open node holding the data array (and its tile manifest as a
        child, if any).
    fname : str
        Path of the file the node was opened from, reopened by the hdf5
        virtual backend.
    group : str, optional
        Location of *node* within the file, needed by the same backend.
    vtype : str, optional
        Virtualization backing of the returned data (see
        :func:`open_dataarray`).
    """
    dataset = node.dataset

    # check file format
    if not ("Conventions" in dataset.attrs and "CF" in dataset.attrs["Conventions"]):
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
    coords = Coordinates._from_dataset(dataset, name)

    # read data
    if "__tile_array__" in dataset[name].attrs:
        spec = json.loads(dataset[name].attrs.pop("__tile_array__"))
        manifest = node[TILES_GROUP].to_dataset(inherit=False).load()
        # the placeholder variable carries the dtype; the spec only the engine
        data = TileArray(manifest, dataset[name].dtype, spec["engine"])
    elif "__dask_array__" in dataset[name].attrs:
        data = loads(dataset[name].attrs.pop("__dask_array__"))
    else:
        with h5py.File(fname) as file:
            if group:
                file = file[group]
            variable = file["__values__" if name is None else name]
            data = VirtualBackend["hdf5" if vtype is None else vtype].from_variable(
                variable
            )

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
    """
    Write *da* to a native xdas NetCDF4/HDF5 file.

    Parameters
    ----------
    da : DataArray
        Data to write.
    fname : str or Path
        Output file path.
    mode : str, optional
        File open mode (``"w"`` or ``"a"``).
    group : str, optional
        HDF5 group path within the file.
    virtual : bool, optional
        If ``True``, write as a virtual (lazy) dataset.
    encoding : dict, optional
        HDF5/NetCDF4 encoding options.
    create_dirs : bool, optional
        Create parent directories if they do not exist.
    """
    if isinstance(fname, Path):
        fname = str(fname)
    _save_tree({group: da}, fname, mode, virtual, encoding, create_dirs)


def _save_tree(leaves, fname, mode, virtual, encoding, create_dirs):
    """Write *leaves* (``{location or None: DataArray}``) in two passes.

    First every group's metadata — coordinates and tile manifests — as
    one :class:`xarray.DataTree`, then every data variable through a
    single writable `h5netcdf` handle, since xarray cannot write the
    virtual ones. *mode* applies to the first pass; the second appends.
    """
    # prepare metadata: one tree node per group, plus per-leaf variable
    # attributes and virtual-ness for the second pass
    nodes = {}
    entries = []
    for location, da in leaves.items():
        isvirtual = (
            isinstance(da.data, (VirtualBackend, DaskArray))
            if virtual is None
            else virtual
        )
        if isvirtual:
            if encoding is not None:
                raise ValueError("cannot use `encoding` with in virtual mode")
            if not isinstance(da.data, (VirtualBackend, DaskArray)):
                raise ValueError(
                    "can only use `virtual=True` with a virtual array as data"
                )
        dataset = xr.Dataset(attrs={"Conventions": "CF-1.9"})
        attrs = {} if da.attrs is None else dict(da.attrs)
        for coord in da.coords.values():
            dataset, attrs = coord._to_dataset(dataset, attrs)
        nodes["/" if location is None else location] = dataset
        if isvirtual and isinstance(da.data, VirtualBackend):
            for relpath, sibling in da.data.sibling_datasets().items():
                nodes[relpath if location is None else f"{location}/{relpath}"] = (
                    sibling
                )
        entries.append((location, da, isvirtual, attrs))

    # create parent directories if needed
    if create_dirs:
        dirname = os.path.dirname(fname)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    # write metadata, one public-API call for every group
    xr.DataTree.from_dict(nodes).to_netcdf(fname, mode=mode, engine="h5netcdf")

    # write data variables, one writable open for the whole file
    with h5netcdf.File(fname, mode="a") as file:
        for location, da, isvirtual, attrs in entries:
            target = file if location is None else file[location]

            # dims the metadata pass did not create (those carrying no
            # coordinate variable)
            target.dimensions.update(
                {
                    dim: size
                    for dim, size in da.sizes.items()
                    if dim not in target.dimensions
                }
            )

            # variable
            variable_name = "__values__" if da.name is None else da.name
            if not isvirtual:
                variable = target.create_variable(
                    variable_name,
                    da.dims,
                    da.dtype,
                    data=da.values,
                    **({} if encoding is None else encoding),
                )
            elif isinstance(da.data, VirtualBackend):
                variable = da.data.create_variable(
                    target, variable_name, da.dims, da.dtype
                )
            else:
                warnings.warn(
                    "writing dask-backed virtual arrays is deprecated; the "
                    "tile-backed engines (xdas.virtual.tiles) replace them",
                    FutureWarning,
                )
                variable = create_variable(
                    da.data, target, variable_name, da.dims, da.dtype
                )

            # attrs
            if attrs:
                variable.attrs.update(attrs)


def open_datacollection(fname, group=None):
    """Read a :class:`DataCollection` from *fname*, auto-detecting sequence vs. mapping."""
    dc = open_datamapping(fname, group)
    try:
        keys = [int(key) for key in dc]
    except ValueError:
        return dc
    if set(keys) == set(range(len(keys))):
        return DataSequence([dc[str(key)] for key in range(len(keys))], dc.name)
    else:
        return dc


def save_datacollection(
    dc, fname, mode="w", group=None, virtual=None, encoding=None, create_dirs=False
):
    """Write *dc* to *fname*, dispatching to sequence or mapping writer as needed."""
    if isinstance(dc, DataSequence):
        save_datasequence(dc, fname, mode, group, virtual, encoding, create_dirs)
    elif isinstance(dc, DataCollection):
        save_datamapping(dc, fname, mode, group, virtual, encoding, create_dirs)
    else:
        raise ValueError("can only save a DataCollection or a DataSequence")


def open_datamapping(fname, group=None):
    """Read a :class:`DataMapping` from *fname*."""
    if isinstance(fname, Path):
        fname = str(fname)

    # the whole collection in one open — walking the already open tree
    # replaces one targeted reopen per data array
    with xr.open_datatree(
        fname,
        engine="h5netcdf",
        decode_timedelta=False,
        phony_dims="access",
    ) as tree:
        node = tree if group is None else tree[group]
        if group is None and not node.dataset.data_vars:
            node = next(iter(node.children.values()))
        # a collection node holds only groups; finding variables on it (or
        # being handed a variable path) means the file is something else
        if isinstance(node, xr.DataArray) or node.dataset.data_vars:
            raise ValueError(
                "it looks like you are trying to open a data array as a data collection."
            )
        return _read_datamapping(node, fname)


def _read_datamapping(node, fname):
    """Build a :class:`DataMapping` from the open tree *node* holding it.

    A child node holding variables is a data array; one holding only
    groups is a nesting level whose single child is a named collection.
    """
    name = node.name
    dm = DataMapping({}, name=None if name == "collection" else name)
    for key, child in node.children.items():
        if child.dataset.data_vars:
            dm[key] = _read_dataarray(child, fname, group=child.path)
        else:
            subnode = next(iter(child.children.values()))
            dm[key] = _read_datacollection(subnode, fname)
    return dm


def _read_datacollection(node, fname):
    """Read the collection at *node*, auto-detecting sequence vs. mapping."""
    dm = _read_datamapping(node, fname)
    try:
        keys = [int(key) for key in dm]
    except ValueError:
        return dm
    if keys == list(range(len(keys))):
        return DataSequence.from_mapping(dm)
    else:
        return dm


def save_datamapping(
    dm, fname, mode="w", group=None, virtual=None, encoding=None, create_dirs=False
):
    """Write :class:`DataMapping` *dm* to *fname*, writing each key as a separate group."""
    if isinstance(fname, Path):
        fname = str(fname)
    if mode == "w" and group is None and os.path.exists(fname):
        os.remove(fname)
    leaves = _collect_leaves(dm, group)
    if leaves:
        _save_tree(leaves, fname, "a", virtual, encoding, create_dirs)


def _collect_leaves(dc, group):
    """Flatten collection *dc* into ``{location: DataArray}`` under *group*."""
    if isinstance(dc, DataSequence):
        dc = dc.to_mapping()
    name = dc.name if dc.name is not None else "collection"
    leaves = {}
    for key in dc:
        location = "/".join([name, str(key)])
        if group is not None:
            location = f"{group}/{location}"
        if isinstance(dc[key], DataArray):
            leaves[location] = dc[key]
        else:
            leaves.update(_collect_leaves(dc[key], location))
    return leaves


def open_datasequence(fname, group=None):
    """Read a :class:`DataSequence` from *fname* via :func:`open_datamapping`."""
    dm = open_datamapping(fname, group)
    return DataSequence.from_mapping(dm)


def save_datasequence(
    ds, fname, mode="w", group=None, virtual=None, encoding=None, create_dirs=False
):
    """Write :class:`DataSequence` *ds* to *fname* by converting to a mapping first."""
    dm = ds.to_mapping()
    save_datamapping(dm, fname, mode, group, virtual, encoding, create_dirs)
