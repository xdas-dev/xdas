import os
from pathlib import Path

import h5netcdf
import h5py
import hdf5plugin
import xarray as xr
from dask.array import Array as DaskArray

from ..coordinates import Coordinates
from ..core.dataarray import DataArray
from ..dask.core import create_variable, loads
from ..virtual import VirtualArray, VirtualSource


def read(fname, group=None):
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


def write(
    da,
    fname,
    mode="w",
    group=None,
    virtual=None,
    encoding=None,
    create_dirs=False,
):
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
    create_dirs : bool, optional
        Whether to create parent directories if they do not exist. Default is False.

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
                variable_name,
                da.dims,
                da.dtype,
                data=da.values,
                **encoding,
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
