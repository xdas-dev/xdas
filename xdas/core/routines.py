import os
import re
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr
from tqdm import tqdm

from ..virtual import VirtualSource, VirtualStack
from .coordinates import Coordinates, InterpCoordinate, get_sampling_interval
from .dataarray import DataArray
from .datacollection import DataCollection, DataMapping, DataSequence


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
        if verbose:
            iterator = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Fetching metadata from files",
            )
        else:
            iterator = as_completed(futures)
        objs = [future.result() for future in iterator]
    return combine_by_field(objs, dim, tolerance, squeeze, True, verbose)


def open_mfdatatree(
    paths,
    dim="first",
    tolerance=None,
    squeeze=False,
    engine=None,
    verbose=False,
    **kwargs,
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
    **kwargs
        Additional keyword arguments to be passed to the read function.

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
            regex = regex.replace(placeholder, f"(?P<{placeholder[1:-1]}>.+)", 1)
            regex = regex.replace(placeholder, f"(?P={placeholder[1:-1]})")
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

    return collect(tree, fields, dim, tolerance, squeeze, engine, verbose, **kwargs)


def collect(
    tree,
    fields,
    dim="first",
    tolerance=None,
    squeeze=False,
    engine=None,
    verbose=False,
    **kwargs,
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
    **kwargs
        Additional keyword arguments to be passed to the read function.


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
            dc = open_mfdataarray(
                value, dim, tolerance, squeeze, engine, verbose, **kwargs
            )
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
    paths,
    dim="first",
    tolerance=None,
    squeeze=True,
    engine=None,
    verbose=False,
    **kwargs,
):
    """
    Open a multiple file dataset.

    Each file described by `path` will be opened as a data array. The data arrays are
    then combined along the `dim` dimension using `combine_by_coords`. If the
    coordinates of the data arrays are not compatible, the resulting object will be
    split into a sequence of data arrays.

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
    **kwargs
        Additional keyword arguments to be passed to the read function.

    Returns
    -------
    DataArray or DataSequence
        The data array containing all files data. If different acquisitions are found,
        a DataSequence is returned.

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
    max_workers = 1 if engine == "miniseed" else None  # TODO: dirty fix
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures_to_paths = {
            executor.submit(open_dataarray, path, engine=engine, **kwargs): path
            for path in paths
        }
        if verbose:
            iterator = tqdm(
                as_completed(futures_to_paths),
                total=len(futures_to_paths),
                desc="Fetching metadata from files",
            )
        else:
            iterator = as_completed(futures_to_paths)
        objs = []
        for future in iterator:
            try:
                obj = future.result()
            except Exception as e:
                path = futures_to_paths[future]
                warnings.warn(f"could not open {path}: {e}", RuntimeWarning)
            else:
                objs.append(obj)
    return combine_by_coords(objs, dim, tolerance, squeeze, None, verbose)


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
    **kwargs
        Additional keyword arguments to be passed to the read function.

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
        return DataArray.from_netcdf(fname, group=group)
    elif callable(engine):
        return engine(fname, **kwargs)
    elif isinstance(engine, str):
        from .. import io

        module = getattr(io, engine)
        return module.read(fname, **kwargs)
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
                key: combine_by_field(
                    [dc[key] for dc in objs if key in dc],
                    dim,
                    tolerance,
                    squeeze,
                    virtual,
                    verbose,
                )
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
    compatibility implies same sampling interval along the combination dimension, exact
    equality along other dimensions and same dtype. Each group is then concatenated.

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
        Whether to return a Database instead of a DataCollection if the combination
        results in a data collection containing a unique Database.
    virtual : bool, optional
        Whether to create a virtual dataset. It requires that all concatenated
        data arrays are virtual. By default tries to create a virtual dataset if possible.
    verbose: bool
        Whether to display a progress bar. Default to False.

    Returns
    -------
    DataSequence or DataArray
        The combined data arrays.
    """
    # parse dim
    if dim == "first":
        dim = objs[0].dims[0]
    if dim == "last":
        dim = objs[0].dims[-1]

    # sort objs by dim
    if dim in objs[0].coords:
        objs = sorted(
            objs,
            key=lambda da: da[dim].values if da[dim].isscalar() else da[dim][0].values,
        )

    # combine objs
    bags = []
    bag = Bag(dim)
    for da in objs:
        try:
            bag.append(da)
        except CompatibilityError:
            bags.append(bag)
            bag = Bag(dim)
            bag.append(da)
    bags.append(bag)

    # concatenate each bag
    collection = DataCollection(
        [concatenate(bag, dim, tolerance, virtual, verbose) for bag in bags]
    )

    # squeeze if possible
    if squeeze and len(collection) == 1:
        return collection[0]
    else:
        return collection


class CompatibilityError(Exception):
    """Custom exception to signal required splitting."""

    def __init__(self, message):
        super().__init__(message)


class Bag:
    def __init__(self, dim):
        self.objs = []
        self.dim = dim

    def __iter__(self):
        return iter(self.objs)

    def initialize(self, da):
        self.objs = [da]
        self.dims = da.dims
        self.subshape = tuple(
            size for dim, size in da.sizes.items() if not dim == self.dim
        )
        self.subcoords = (
            da.coords.drop_dims(self.dim)
            if self.dim in self.dims
            else da.coords.drop_coords(self.dim)
        )
        try:
            self.delta = get_sampling_interval(da, self.dim)
        except (ValueError, KeyError):
            self.delta = None
        self.dtype = da.dtype

    def append(self, da):
        if not self.objs:
            self.initialize(da)
        else:
            self.check_dims(da)
            self.check_shape(da)
            self.check_coords(da)
            self.check_sampling_interval(da)
            self.check_dtype(da)
            self.objs.append(da)

    def check_dims(self, da):
        if not self.dims == da.dims:
            raise CompatibilityError("dimensions are not compatible")

    def check_shape(self, da):
        subshape = tuple(size for dim, size in da.sizes.items() if not dim == self.dim)
        if not self.subshape == subshape:
            raise CompatibilityError("shapes are not compatible")

    def check_dtype(self, da):
        if not self.dtype == da.dtype:
            raise CompatibilityError("data types are not compatible")

    def check_coords(self, da):
        subcoords = (
            da.coords.drop_dims(self.dim)
            if self.dim in self.dims
            else da.coords.drop_coords(self.dim)
        )
        if not self.subcoords.equals(subcoords):
            raise CompatibilityError("coordinates are not compatible")

    def check_sampling_interval(self, da):
        if self.delta is None:
            pass
        else:
            delta = get_sampling_interval(da, self.dim)
            if not np.isclose(delta, self.delta):
                raise CompatibilityError("sampling intervals are not compatible")


def concatenate(objs, dim="first", tolerance=None, virtual=None, verbose=None):
    """
    Concatenate data arrays along a given dimension.

    Parameters
    ----------
    objs : list of DataArray
        List of data arrays to concatenate.
    dim : str
        The dimension along which concatenate.
    tolerance : float of timedelta64, optional
        The tolerance to consider that the end of a file is continuous with beginning of
        the following, zero by default.
    virtual : bool, optional
        Whether to create a virtual dataset. It requires that all concatenated
        data arrays are virtual. By default tries to create a virtual dataset if possible.
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

    if dim in objs[0].dims + ("first", "last"):
        axis = objs[0].get_axis_num(dim)
        dim = objs[0].dims[axis]  # ensure not "first" or "last"
        dims = objs[0].dims
    else:
        axis = 0
        dims = (dim, *objs[0].dims)
        objs = [da.expand_dims(dim) for da in objs]

    coords = objs[0].coords.copy()
    name = objs[0].name
    attrs = objs[0].attrs

    dim_has_coords = dim in coords

    if dim_has_coords:
        objs = sorted(objs, key=lambda da: da[dim][0].values)
        coord = coords[dim].__class__(data=None, dim=dim, dtype=coords[dim].dtype)

    iterator = tqdm(objs, desc="Linking dataarray") if verbose else objs
    data = []
    for da in iterator:
        if isinstance(da.data, VirtualStack):
            for source in da.data.sources:
                data.append(source)
        else:
            data.append(da.data)

        if dim in coords:
            coord = coord.append(da[dim])

    if virtual:
        data = VirtualStack(data, axis)
    else:
        data = np.concatenate(data, axis)
    if tolerance is not False:
        if dim_has_coords:
            if hasattr(coord, "simplify"):
                coord = coord.simplify(tolerance)
            else:
                if tolerance is not None:
                    raise TypeError(
                        "tolerance can only be used with interpolated coordinates"
                    )
            coords[dim] = coord
        else:
            if tolerance is not None:
                raise TypeError("cannot use tolerance on non-existing coordinates")

    return DataArray(data, coords, dims, name, attrs)


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
        If `indices_or_section` is an integer N, the array will be divided into N
        almost equal (can differ by one element if the `dim` size is not a multiple of
        N). If `indices_or_section` is a 1-D array of sorted integers, the entries
        indicate where the array is split along `dim`. For example, `[2, 3]` would, for
        `dim="first"`, result in [da[:2], da[2:3], da[3:]]. If `indices_or_section` is
        "discontinuities", the `dim` must be an interpolated coordinate and splitting
        will occurs at locations where they are two consecutive tie_indices with only
        one index of difference and where the tie_values difference is greater than
        `tolerance`. Default to "discontinuities".
    dim : str, optional
        The dimension along which to split, by default "first"
    tolerance : float or timedelta64, optional
        If `indices_or_sections="discontinuities"` split will only occur on gaps and
        overlaps that are bigger than `tolerance`. Zero tolerance by default.

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


def align(*objs):
    """
    Given any number of data arrays, returns new objects with aligned dimensions.

    New objects will all share the same dimensions with the same order. This is done by
    expanding missing dimensions and transposing to the same `dims`. The order of
    the resulting `dims` is given by the order in which dimensions are first encountered
    while iterating through each objects `dims`. For each dimensions, the data arrays
    must either share the same coordinate or not having any.

    Array from the aligned objects are suitable as input to mathematical
    operators, as their shapes are compatible in term of broadcasting.

    Parameters
    ----------
    *objects : DataArray
        Data arrays to align.

    Returns
    -------
    aligned : tuple of DataArray
        Tuple of data arrays with aligned coordinates.

    Examples
    --------
    >>> import xdas as xd
    >>> import numpy as np

    >>> da1 = xd.DataArray(np.arange(2), {"x": [0, 1]})
    >>> da2 = xd.DataArray(np.arange(3), {"y": [2, 3, 4]})
    >>> da1, da2 = xd.align(da1, da2)
    >>> da1
    <xdas.DataArray (x: 2, y: 1)>
    [[0]
    [1]]
    Coordinates:
      * x (x): [0 1]
    Dimensions without coordinates: y

    >>> da2
    <xdas.DataArray (x: 1, y: 3)>
    [[0 1 2]]
    Coordinates:
      * y (y): [2 ... 4]
    Dimensions without coordinates: x

    """
    coords = broadcast_coords(*objs)
    return tuple(broadcast_to(obj, coords) for obj in objs)


def broadcast_coords(*objs):
    """
    Broadcasts the coordinates of multiple objects and returns a new Coordinates object.

    Parameters
    ----------
    *objs : Variable number of objects with sizes and coordinates.

    Returns
    -------
    Coordinates
        A new Coordinates object with the broadcasted coordinates.

    Raises
    ------
    ValueError
        If the data arrays have incompatible sizes along any dimension or if the
        coordinates differ between data arrays.

    Examples
    --------
    >>> import xdas as xd
    >>> import numpy as np

    >>> da1 = xd.DataArray(np.arange(2), {"x": [0, 1]})
    >>> da2 = xd.DataArray(np.arange(3), {"y": [2, 3, 4]})
    >>> xd.broadcast_coords(da1, da2)
    Coordinates:
      * x (x): [0 1]
      * y (y): [2 ... 4]

    """
    sizes = {}
    coords = {}
    for obj in objs:
        for dim, size in obj.sizes.items():
            if dim in sizes:
                if sizes[dim] == 1:
                    sizes[dim] = size
                if not (size == 1 or size == sizes[dim]):
                    raise ValueError(
                        f"data arrays to align have incompatible sizes along {dim}"
                    )
            else:
                sizes[dim] = size
        for name, coord in obj.coords.items():
            if coord.isscalar():
                continue
            if name in coords:
                if not coord.equals(coords[name]):
                    raise ValueError(
                        f"coordinate {name} differs from one data array to another"
                    )
            else:
                coords[name] = coord
    dims = tuple(dim for dim in sizes)
    return Coordinates(coords, dims)


def broadcast_to(obj, coords):
    """
    Broadcasts an object to match the dimensions specified by the given coordinates.

    Parameters
    ----------
    obj : DataArray or array-like
        The object to be broadcasted.
    coords : Coordinates
        The coordinates specifying the dimensions to match.

    Returns
    -------
    DataArray
        The broadcasted object.

    Notes
    -----
    - If the input object is not a DataArray, it will be converted to a DataArray using
      the pro.
    - The dimensions of the input object will be expanded to match the dimensions
      specified by the coordinates.
    - The order of dimensions in the output object will be rearranged to match the
      order specified by the coordinates.

    """
    if not isinstance(obj, DataArray):
        _data = np.asarray(obj)
        _dims = coords.dims[len(coords.dims) - _data.ndim :]
        _coords = {
            name: (coord.dim, coord)
            for name, coord in coords.items()
            if coord.dim in _dims
        }
        obj = DataArray(_data, _coords, _dims)
    for dim in coords.dims:
        if dim not in obj.dims:
            obj = obj.expand_dims(dim)
    obj = obj.transpose(*coords.dims)
    return obj


def plot_availability(obj, dim="first", **kwargs):
    """
    Plot the availability of a given dimension in a timeline chart.

    The availability is determined by finding the discontinuities and availabilities
    of the specified dimension in the object. The resulting timeline chart shows
    the start and end values of each availability period, as well as any gaps or
    overlaps in the data. If a data collection is provided, the timeline chart will
    show the availability of each data array in the collection. Note that data arrays
    in the same data sequence will be on the same timeline whereas data arrays in
    data mappings will be on separate timelines.

    This function only works on interpolated coordinates.

    Parameters
    ----------
    obj : DataArray or DataCollection
        The data array containing the dimension to plot.
    dim : str
        The name of the dimension to plot.
    **kwargs
        Additional keyword arguments to be passed to the `px.timeline` function.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The timeline

    Notes
    -----
    This function uses the `px.timeline` function from the `plotly.express` library.

    """
    dataframe = _get_timeline_dataframe(obj, dim, "")
    category_orders = {"type": ["data", "gap", "overlap"]}
    color_discrete_map = {"data": "#00CC96", "gap": "#636EFA", "overlap": "#EF553B"}
    pattern_shape_map = {"data": "", "gap": "/", "overlap": "\\"}
    fig = px.timeline(
        dataframe,
        x_start="start_value",
        x_end="end_value",
        y="name",
        color="type",
        category_orders=category_orders,
        color_discrete_map=color_discrete_map,
        pattern_shape_map=pattern_shape_map,
        **kwargs,
    )
    for elem in fig.data:
        elem["marker"]["line_color"] = color_discrete_map[elem["legendgroup"]]
    fig.update_yaxes(title_text="")
    return fig


def _get_timeline_dataframe(obj, dim="first", name=None):
    if isinstance(obj, DataArray):
        discontinuities = obj[dim].get_discontinuities()
        availabilities = obj[dim].get_availabilities()
        dataframe = pd.concat([availabilities, discontinuities])
        dataframe["name"] = "" if name is None else name
    elif isinstance(obj, DataSequence):
        dataframes = [_get_timeline_dataframe(val, dim, name) for val in obj]
        dataframe = pd.concat(dataframes)
    elif isinstance(obj, DataMapping):
        dataframes = [
            _get_timeline_dataframe(val, dim, f"{name}.{key}" if name else key)
            for key, val in obj.items()
        ]
        dataframe = pd.concat(dataframes)
    else:
        raise TypeError(
            f"`obj` must be a DataArray of a DataCollection, found {type(obj)}"
        )
    return dataframe
