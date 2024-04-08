import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
import xarray as xr
from tqdm import tqdm

from ..virtual import VirtualSource, VirtualStack
from .coordinates import InterpCoordinate, get_sampling_interval
from .dataarray import DataArray
from .datacollection import DataCollection


def open_mfdatacollection(
    paths, dim="first", tolerance=None, squeeze=False, verbose=False
):
    """
    Open a multiple file DataCollection.

    Files matching the wildcarded `paths` string will be opened and combined into one
    data collection. Each opened file must be a DataCollection. The data arrays nested
    inside the data collections are concatenated by their position within the data
    collection hierarchy using `combine_by_field`.

    For exemple, it can be used to combine daily data collections into one master
    data collection.

    Parameters
    ----------
    paths : str or list
        The path names given as a shell-style wildcards string or a list of paths.
    dim : str, optional
        The dimension along which the data arrays are concatenated. Default to "first".
    tolerance : float of timedelta64, optional
        During concatenation, the tolerance to consider that the end of a file is
        continuous with beginning of the following one. Default to zero tolerance.
    squeeze : bool, optional
        Whether to return a DataArray instead of a DataCollection if the combination
        results in a data collection containing a unique data array.
    verbose: bool
        Whether to display a progress bar. Default to False.

    Returns
    -------
    DataCollection
        The combined data collection

    """
    if isinstance(paths, str):
        paths = sorted(glob(paths))
    elif isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"could not find {path}")
    else:
        raise ValueError(
            f"`paths` must be either a string or a list, found {type(paths)}"
        )
    if len(paths) == 0:
        raise FileNotFoundError("no file to open")
    if len(paths) > 100_000:
        raise NotImplementedError(
            "The maximum number of file that can be opened at once is for now limited "
            "to 100 000."
        )
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(open_datacollection, path) for path in paths]
        objs = [
            future.result()
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Fetching metadata from files",
            )
        ]
    return combine_by_field(objs, dim, tolerance, squeeze, True, verbose)


def open_mfdatatree(
    paths, dim="first", tolerance=None, squeeze=False, engine=None, verbose=False
):
    """
    Open a directory tree structure as a data collection.

    The tree structure is descirebed by a path descriptor provided as a string
    containings placeholders. Two flavours of placeholder can be provided:

    - `{field}`: this level of the tree will behave as a dict. It will use the
    directory/file names as keys.
    - `[field]`: this level of the tree will behave as a list. The directory/file
    names are not considered (as if the placeholder was replaced by a `*`) and
    files are gathered and combined as if using `open_mfdataarray`.

    Several dict placeholders with different names can be provided. They must be
    followed by one or more list placeholders that must share a unique name. The
    resulting data collection will be a nesting of dicts down to the lower level
    which will be a list of dataarrays.

    Parameters
    ----------
    paths : str
        The path descriptor.
    dim : str, optional
        The dimension along which the data arrays are concatenated. Default to "first".
    tolerance : float of timedelta64, optional
        During concatenation, the tolerance to consider that the end of a file is
        continuous with beginning of the following one. Default to zero tolerance.
    squeeze : bool, optional
        Whether to return a DataArray instead of a DataCollection if the combination
        results in a data collection containing a unique data array.
    engine: str of callable, optional
        The type of file to open or a read function. Default to xdas netcdf format.
    verbose: bool
        Whether to display a progress bar. Default to False.

    Returns
    -------
    DataCollection
        The collected data.

    Examples
    --------
    >>> import xdas
    >>> paths = "/data/{node}/{cable}/[acquisition]/proc/[acquisition].h5"
    >>> xdas.open_mfdatatree(paths, engine="asn") # doctest: +SKIP
    Node:
      CCN:
        Cable:
          N:
            Acquisition:
              0: <xdas.DataArray (time: ..., distance: ...)>
              1: <xdas.DataArray (time: ..., distance: ...)>
      SER:
        Cable:
          N:
            Acquisition:
              0: <xdas.DataArray (time: ..., distance: ...)>
          S:
            Acquisition:
              0: <xdas.DataArray (time: ..., distance: ...)>
              1: <xdas.DataArray (time: ..., distance: ...)>
              2: <xdas.DataArray (time: ..., distance: ...)>


    """
    placeholders = re.findall(r"[\{\[].*?[\}\]]", paths)

    seen = set()
    fields = tuple(
        placeholder[1:-1]
        for placeholder in placeholders
        if not (placeholder in seen or seen.add(placeholder))
    )

    wildcard = paths
    for placeholder in placeholders:
        wildcard = wildcard.replace(placeholder, "*")
    fnames = sorted(glob(wildcard))

    regex = paths
    regex = regex.replace(".", r"\.")
    for placeholder in placeholders:
        if placeholder.startswith("{") and placeholder.endswith("}"):
            regex = regex.replace(placeholder, f"(?P<{placeholder[1:-1]}>.+)")
        else:
            regex = regex.replace(placeholder, r".*")
    regex = re.compile(regex)

    tree = defaulttree(len(fields))
    for fname in fnames:
        match = regex.match(fname)
        bag = tree
        for field in fields[:-1]:
            bag = bag[match.group(field)]
        bag.append(fname)

    return collect(tree, fields, dim, tolerance, squeeze, engine, verbose)


def collect(
    tree, fields, dim="first", tolerance=None, squeeze=False, engine=None, verbose=False
):
    """
    Collects the data from a tree of paths using `fields` as level names.

    Parameters
    ----------
    tree : nested dict of lists
        The paths grouped in a tree hierarchy.
    fields : tuple of str
        The names of the levels of the tree hierarchy.
    dim : str, optional
        The dimension along which the data arrays are concatenated. Default to "first".
    tolerance : float of timedelta64, optional
        During concatenation, the tolerance to consider that the end of a file is
        continuous with beginning of the following one. Default to zero tolerance.
    squeeze : bool, optional
        Whether to return a DataArray instead of a DataCollection if the combination
        results in a data collection containing a unique data array.
    engine: str of callable, optional
        The type of file to open or a read function. Default to xdas netcdf format.
    verbose: bool
        Whether to display a progress bar. Default to False.


    Returns
    -------
    DataCollection
        The collected data.
    """
    fields = list(fields)
    name = fields.pop(0)
    collection = DataCollection({}, name=name)
    for key, value in tree.items():
        if isinstance(value, list):
            dc = open_mfdataarray(value, dim, tolerance, squeeze, engine, verbose)
            dc.name = fields[0]
            collection[key] = dc
        else:
            collection[key] = collect(
                value, fields, dim, tolerance, squeeze, engine, verbose
            )
    return collection


def defaulttree(depth):
    """Generate a default tree of lists with given depth."""
    if depth == 1:
        return list()
    else:
        return defaultdict(lambda: defaulttree(depth - 1))


def open_mfdataarray(
    paths, dim="first", tolerance=None, squeeze=True, engine=None, verbose=False
):
    """
    Open a multiple file dataarray.

    Parameters
    ----------
    paths : str or list
        The path names given as a shell-style wildcards string or a list of paths.
    dim : str, optional
        The dimension along which the data arrays are concatenated. Default to "first".
    tolerance : float of timedelta64, optional
        During concatenation, the tolerance to consider that the end of a file is
        continuous with beginning of the following one. Default to zero tolerance.
    squeeze : bool, optional
        Whether to return a DataArray instead of a DataCollection if the combination
        results in a data collection containing a unique data array.
    engine: str of callable, optional
        The type of file to open or a read function. Default to xdas netcdf format.
    verbose: bool
        Whether to display a progress bar. Default to False.

    Returns
    -------
    DataArray
        The dataarray containing all files data.

    Raises
    ------
    FileNotFound
        If no file can be found.
    """
    if isinstance(paths, str):
        paths = sorted(glob(paths))
    elif isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"could not find {path}")
    else:
        raise ValueError(
            f"`paths` must be either a string or a list, found {type(paths)}"
        )
    if len(paths) == 0:
        raise FileNotFoundError("no file to open")
    if len(paths) > 100_000:
        raise NotImplementedError(
            "The maximum number of file that can be opened at once is for now limited "
            "to 100 000."
        )
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(open_dataarray, path, engine=engine) for path in paths
        ]
        objs = [
            future.result()
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Fetching metadata from files",
            )
        ]
    return combine_by_coords(objs, dim, tolerance, squeeze, True, verbose)


def open_dataarray(fname, group=None, engine=None, **kwargs):
    """
    Open a dataarray.

    Parameters
    ----------
    fname : str
        The path of the dataarray.
    group : str, optional
        The file group where the dataarray is located, by default None which corresponds
        to the root of the file.
    engine: str of callable, optional
        The type of file to open or a read function. Default to xdas netcdf format.

    Returns
    -------
    DataArray
        The opened dataarray.

    Raises
    ------
    ValueError
        If the engine si not recognized.

    Raises
    ------
    FileNotFound
        If no file can be found.
    """
    if not os.path.exists(fname):
        raise FileNotFoundError("no file to open")
    if engine is None:
        return DataArray.from_netcdf(fname, group=group, **kwargs)
    elif callable(engine):
        return engine(fname)
    elif isinstance(engine, str):
        from .. import io

        module = getattr(io, engine)
        return module.read(fname)
    else:
        raise ValueError("engine not recognized")


def open_datacollection(fname, group=None):
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

    Raises
    ------
    FileNotFound
        If no file can be found.
    """
    if not os.path.exists(fname):
        raise FileNotFoundError("no file to open")
    return DataCollection.from_netcdf(fname, group)


def asdataarray(obj, tolerance=None):
    """
    Try to convert given object to a dataarray.

    Only support DataArray or DataArray as input.

    Parameters
    ----------
    obj : object
        The objected to convert
    tolerance : float or datetime64, optional
        For dense coordinates, tolerance error for interpolation representation, by
        default zero.

    Returns
    -------
    DataArray
        The object converted to a DataArray. Data is not copied.

    Raises
    ------
    ValueError
        _description_
    """
    if isinstance(obj, DataArray):
        return obj
    elif isinstance(obj, xr.DataArray):
        return DataArray.from_xarray(obj)
    else:
        raise ValueError("Cannot convert to dataarray.")


def combine_by_field(
    objs, dim="first", tolerance=None, squeeze=False, virtual=None, verbose=False
):
    """
    Combine data collections by field along a dimension.

    The data arrays nested into each data collections are first grouped by their
    hierachical position. Data sequences are appended to each other such as each group
    consist of a list of data arrays which order is first given by the order of the
    `objs` data collections, and second by the order of the data array within its data
    sequence (if part of any sequence). Each group is eventually combined using
    `combined_by_coords`.

    Parameters
    ----------
    objs : list of DataCollection
        The data collections to combine.
    dim : str, optional
        The dimension along which concatenate. Default to "first".
    tolerance : float of timedelta64, optional
        The tolerance to consider that the end of a file is continuous with beginning of
        the following, zero by default.
    squeeze : bool, optional
        Whether to return a Database instead of a DataCollection if the combinatison
        results in a data collection containing a unique Database.
    virtual : bool, optional
        Whether to create a virtual dataset. It requires that all concatenated
        dataarrays are virtual. By default tries to create a virtual dataset if possible.
    verbose: bool
        Whether to display a progress bar. Default to False.

    Returns
    -------
    DataCollection
        The combined data collection.

    """
    leaves = [dc for dc in objs if isinstance(dc, list)]
    nodes = [dc for dc in objs if isinstance(dc, dict)]
    if leaves and not nodes:
        objs = [da for dc in leaves for da in dc]
        dc = combine_by_coords(objs, dim, tolerance, squeeze, virtual, verbose)
        dc.name = leaves[0].name
        return dc
    elif nodes and not leaves:
        (name,) = set(dc.name for dc in nodes)
        keys = sorted(set.union(*[set(dc.keys()) for dc in nodes]))
        return DataCollection(
            {
                key: combine_by_field([dc[key] for dc in objs if key in dc])
                for key in keys
            },
            name,
        )
    else:
        raise NotImplementedError("cannot combine mixed node/leave levels for now")


def combine_by_coords(
    objs, dim="first", tolerance=None, squeeze=False, virtual=None, verbose=False
):
    """
    Combine several data arrays by coordinates.

    The list `objs` if traversed and data arrays are grouped together as long as they
    share compatible coordinates. If a change is detected a new group is created. Shape
    compatibiliy implies same sampling interval along the combination dimension and
    exact equality along other dimensions. Each group is then concatenated.

    Parameters
    ----------
    objs : list of DataArray
        The data arrays to combine.
    dim : str, optional
        The dimension along which concatenate. Default to "first".
    tolerance : float of timedelta64, optional
        The tolerance to consider that the end of a file is continuous with beginning of
        the following, zero by default.
    squeeze : bool, optional
        Whether to return a Database instead of a DataCollection if the combinatison
        results in a data collection containing a unique Database.
    virtual : bool, optional
        Whether to create a virtual dataset. It requires that all concatenated
        dataarrays are virtual. By default tries to create a virtual dataset if possible.
    verbose: bool
        Whether to display a progress bar. Default to False.

    Returns
    -------
    DataSequence or DataArray
        The combined data arrays.
    """
    objs = sorted(objs, key=lambda da: da[dim][0].values)
    out = []
    bag = []
    for da in objs:
        if not bag:
            bag = [da]
        elif da.coords.drop(dim).equals(bag[-1].coords.drop(dim)) and (
            get_sampling_interval(da, dim) == get_sampling_interval(bag[-1], dim)
        ):
            bag.append(da)
        else:
            out.append(bag)
            bag = [da]
    out.append(bag)
    collection = DataCollection(
        [concatenate(bag, dim, tolerance, virtual, verbose) for bag in out]
    )
    if squeeze and len(collection) == 1:
        return collection[0]
    else:
        return collection


def concatenate(objs, dim="first", tolerance=None, virtual=None, verbose=None):
    """
    Concatenate dataarrays along a given dimension.

    Parameters
    ----------
    objs : list of DataArray
        List of dataarrays to concatenate.
    dim : str
        The dimension along which concatenate.
    tolerance : float of timedelta64, optional
        The tolerance to consider that the end of a file is continuous with beginning of
        the following, zero by default.
    virtual : bool, optional
        Whether to create a virtual dataset. It requires that all concatenated
        dataarrays are virtual. By default tries to create a virtual dataset if possible.
    verbose: bool
        Whether to display a progress bar.

    Returns
    -------
    DataArray
        The concatenated dataarray.
    """
    objs = [da for da in objs if not da.empty]
    if virtual is None:
        virtual = all(isinstance(da.data, (VirtualSource, VirtualStack)) for da in objs)
    if not all(isinstance(da[dim], InterpCoordinate) for da in objs):
        raise NotImplementedError("can only concatenate along interpolated coordinate")
    axis = objs[0].get_axis_num(dim)
    dim = objs[0].dims[axis]
    coords = objs[0].coords.copy()
    objs = sorted(objs, key=lambda da: da[dim][0].values)
    iterator = tqdm(objs, desc="Linking dataarray") if verbose else objs
    data = []
    tie_indices = []
    tie_values = []
    idx = 0
    for da in iterator:
        if isinstance(da.data, VirtualStack):
            for source in da.data.sources:
                data.append(source)
        else:
            data.append(da.data)
        tie_indices.extend(idx + da[dim].tie_indices)
        tie_values.extend(da[dim].tie_values)
        idx += da.shape[axis]
    if virtual:
        data = VirtualStack(data, axis)
    else:
        data = np.concatenate(data, axis)
    coords[dim] = InterpCoordinate(
        {"tie_indices": tie_indices, "tie_values": tie_values}, dim
    ).simplify(tolerance)
    return DataArray(data, coords)


def split(da, indices_or_sections="discontinuities", dim="first", tolerance=None):
    """
    Split a data array along a dimension.

    Splitting can either be performed at each discontinuity (along interpolated
    coordinates), at a given set of indices (give as a list of int) or in order to get
    a given number of equal sized chunks (if a single int is provided).

    Parameters
    ----------
    da : DataArray
        The data array to split
    indices_or_sections : str, int or list of int, optional
        If `indices_or_section` is an interger N, the array will be diveided into N
        almost equal (can differ by one element if the `dim` size is not a multiple of
        N). If `indices_or_section` is a 1-D array of sorted integers, the entries
        indicate where the array is split along `dim`. For example, `[2, 3]` would, for
        `dim="first"`, result in [da[:2], da[2:3], da[3:]]. If `indices_or_section` is
        "discontinuites", the `dim` must be an interpolated coordinate and splitting
        will occurs at locations where they are two consecutive tie_indices with only
        one index of difference and where the tie_values differance is greater than
        `tolerance`. Default to "discontinuities".
    dim : str, optional
        The dimension along which to split, by default "first"
    tolerance : float or timedelta64, optional
        If `indices_or_sections="discontinuities"` split will only occur on gaps and
        overlaps that are bigger thatn `tolerance`. Zero tolerance by default.

    Returns
    -------
    list of DataArray
        The splitted data array.
    """
    if isinstance(indices_or_sections, str) and (
        indices_or_sections == "discontinuities"
    ):
        if isinstance(da[dim], InterpCoordinate):
            coord = da[dim].simplify(tolerance)
            (points,) = np.nonzero(np.diff(coord.tie_indices, prepend=[0]) == 1)
            div_points = [coord.tie_indices[point] for point in points]
            div_points = [0] + div_points + [da.sizes[dim]]
        else:
            raise TypeError(
                "discontinuities can only be found on dimension that have as type "
                "`InterpCoordinate`."
            )
    elif isinstance(indices_or_sections, int):
        nsamples = da.sizes[dim]
        nchunk = indices_or_sections
        if nchunk <= 0:
            raise ValueError("`n` must be larger than 0")
        if nchunk >= nsamples:
            raise ValueError("`n` must be smaller than the number of samples")
        chunk_size, extras = divmod(nsamples, nchunk)
        chunks = extras * [chunk_size + 1] + (nchunk - extras) * [chunk_size]
        div_points = np.cumsum([0] + chunks, dtype=np.int64)
    else:
        div_points = [0] + indices_or_sections + [da.sizes[dim]]
    return DataCollection(
        [
            da.isel({dim: slice(div_points[idx], div_points[idx + 1])})
            for idx in range(len(div_points) - 1)
        ]
    )
