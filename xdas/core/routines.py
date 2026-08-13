"""
Top-level routines for opening, concatenating, aligning, and splitting arrays.

Operates on :class:`DataArray` and :class:`DataCollection` objects; includes
multi-file helpers (``open_mfdataarray``, ``open_mfdatacollection``).
"""

import os
import re
import warnings
from collections import defaultdict
from concurrent.futures import as_completed
from glob import glob
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import xarray as xr
from tqdm import tqdm

from ..coordinates import AxisCoordinate, Coordinates
from ..coordinates.core import parse_scalar_delta
from ..parallel import get_scan_pool, get_scan_workers
from ..virtual import TileArray, VirtualBackend, VirtualSource, VirtualStack
from .dataarray import DataArray
from .datacollection import DataCollection, DataMapping, DataSequence

# How many scan products one call may hold at once: a scan keeps one data array
# per file (~6 KiB) until they are fused. Vtypes that consolidate drain a full
# batch and carry on; for the others this is a hard ceiling.
MAX_OPEN_FILES = 100_000


def open(
    paths,
    dim="first",
    tolerance=None,
    squeeze=None,
    engine=None,
    vtype=None,
    ctype=None,
    parallel=None,
    verbose=False,
    **engine_kwargs,
):
    """
    Open one or several files as a data array or collection.

    Automatically dispatches to the appropriate reader based on the shape of `paths`:

    - **Single file** (plain path string): tries to open as a data collection first,
      falls back to a data array if the file does not contain a data collection.
    - **Multi-file** (wildcarded string with ``*``, ``?``, or ``[…]``, or a list of
      paths): tries to open and combine as a multi-file data collection first, falls
      back to a multi-file data array if the files are not data collections.
    - **Tree-like** (string containing ``{field}`` placeholders):
      opens a directory tree as a nested data collection using
      :func:`open_mfdatatree`.

    Parameters
    ----------
    paths : str or list of str
        The path(s) to open. Can be:

        - A plain file path (single file).
        - A shell-style wildcard string (``*``, ``?``, ``[…]``) matching multiple
          files.
        - A list of explicit file paths.
        - A tree descriptor string containing ``{field}`` (dict level) and
          ``[field]`` (list level) placeholders.
    dim : str, optional
        The dimension along which multiple files are concatenated. Ignored when
        opening a single file. Default is ``"first"``.
    tolerance : float or timedelta64, optional
        Maximum gap or overlap allowed between consecutive files to still be
        considered continuous. For time coordinates, numeric values are interpreted
        as seconds. Ignored when opening a single file. Default is zero tolerance.
    squeeze : bool or None, optional
        Whether to return a DataArray instead of a DataCollection when the result
        contains only one data array. When ``None`` (default), the behaviour depends
        on the dispatch path: ``True`` for multi-file data arrays, ``False``
        otherwise. Ignored when opening a single file.
    engine : str or Engine, optional
        The file format engine to use, given by name or as a configured
        :class:`~xdas.io.Engine` instance. When ``None`` (default), the format is
        auto-detected. Providing an engine skips the automatic DataCollection
        detection.
    vtype : str, optional
        The virtualization type to use. If None, the engine default is used.
        Only valid when `engine` is given by name (or None).
    ctype : str or dict, optional
        The coordinate type(s) to use. If None, the engine defaults are used.
        Only valid when `engine` is given by name (or None).
    parallel: bool or int, optional
        Whether to use multiprocessing to fetch file metadata. If False or 1,
        runs in single-process mode. If an integer, use that many processes.
        If True, use as many processes as available cores. If None, use the
        global xdas configuration. Default to None.
    verbose : bool, optional
        Whether to display a progress bar while reading metadata. Ignored when
        opening a single file. Default is ``False``.
    **engine_kwargs
        Format-specific engine parameters forwarded to the engine constructor
        (e.g. ``overlaps`` for "febus"). Only valid when `engine` is given by name.

    Returns
    -------
    DataArray or DataCollection
        The opened data. The exact type depends on the dispatch path and the
        ``squeeze`` setting.

    Raises
    ------
    ValueError
        If `paths` is neither a string nor a list.
    FileNotFoundError
        If no file matching `paths` can be found.

    See Also
    --------
    open_dataarray : Open a single DataArray file.
    open_datacollection : Open a single DataCollection file.
    open_mfdataarray : Open and combine multiple DataArray files.
    open_mfdatacollection : Open and combine multiple DataCollection files.
    open_mfdatatree : Open a directory tree as a nested DataCollection.

    Examples
    --------
    Open a single file (auto-detects DataCollection vs DataArray):

    >>> import xdas as xd
    >>> da = xd.open("path/to/file.nc")  # doctest: +SKIP

    Open multiple files with a wildcard:

    >>> da = xd.open("path/to/files/*.nc")  # doctest: +SKIP

    Open a list of explicit paths:

    >>> da = xd.open(["file1.nc", "file2.nc"])  # doctest: +SKIP

    Open a directory tree:

    >>> dc = xd.open("/data/{node}/[acq].nc", engine="asn")  # doctest: +SKIP

    """
    paths = _ensure_str_paths(paths)
    if isinstance(paths, str):
        if "{" in paths:
            method = "tree-like"
        elif "*" in paths or "?" in paths or "[" in paths:
            method = "multi-file"
        else:
            method = "single-file"
    elif isinstance(paths, list):
        method = "multi-file"
    else:
        raise ValueError(
            f"`paths` must be either a string or a list, found {type(paths)}"
        )
    match method:
        case "single-file":
            if engine is None:
                try:
                    return open_datacollection(paths)
                except Exception:  # noqa: BLE001, S110 - fall back to dataarray
                    pass
            try:
                dc = _resolve_engine(
                    engine, vtype, ctype, engine_kwargs
                ).open_datacollection(paths)
            except NotImplementedError:
                pass  # the engine describes a file as one array
            else:
                # combine whether one file was opened or many, so the returned
                # shape never depends on the file count
                return combine_by_field(
                    [dc],
                    dim,
                    tolerance,
                    False if squeeze is None else squeeze,
                    None,
                    verbose,
                )
            return open_dataarray(
                paths, engine=engine, vtype=vtype, ctype=ctype, **engine_kwargs
            )
        case "multi-file":
            if engine is None:
                try:
                    return open_mfdatacollection(
                        paths,
                        dim,
                        tolerance,
                        squeeze=False if squeeze is None else squeeze,
                        parallel=parallel,
                        verbose=verbose,
                    )
                except Exception:  # noqa: BLE001, S110 - not native collections
                    pass
            try:
                return open_mfdatacollection(
                    paths,
                    dim,
                    tolerance,
                    squeeze=False if squeeze is None else squeeze,
                    parallel=parallel,
                    verbose=verbose,
                    engine=_resolve_engine(engine, vtype, ctype, engine_kwargs),
                )
            except NotImplementedError:
                pass  # the engine describes a file as one array
            return open_mfdataarray(
                paths,
                dim,
                tolerance,
                squeeze=True if squeeze is None else squeeze,
                engine=engine,
                vtype=vtype,
                ctype=ctype,
                parallel=parallel,
                verbose=verbose,
                **engine_kwargs,
            )
        case "tree-like":  # pragma: no branch
            return open_mfdatatree(
                paths,
                dim,
                tolerance,
                squeeze=False if squeeze is None else squeeze,
                engine=engine,
                vtype=vtype,
                ctype=ctype,
                parallel=parallel,
                verbose=verbose,
                **engine_kwargs,
            )


def open_mfdatacollection(
    paths,
    dim="first",
    tolerance=None,
    squeeze=False,
    verbose=False,
    parallel=None,
    engine=None,
    vtype=None,
    ctype=None,
    **engine_kwargs,
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
    tolerance : float or timedelta64, optional
        During concatenation, the tolerance to consider that the end of a file is
        continuous with beginning of the following one. For time coordinates, numeric
        values are considered as seconds. Default to zero tolerance.
    squeeze : bool, optional
        Whether to return a DataArray instead of a DataCollection if the combination
        results in a data collection containing a unique data array.
    parallel: bool or int, optional
        Whether to use multiprocessing to fetch file metadata. If False or 1,
        runs in single-process mode. If an integer, use that many processes.
        If True, use as many processes as available cores. If None, use the
        global xdas configuration. Default to None.
    verbose: bool
        Whether to display a progress bar. Default to False.
    engine: str or Engine, optional
        The file format engine to use, given by name or as a configured
        :class:`~xdas.io.Engine` instance. Default to the native format.
    vtype : str, optional
        The virtualization type to use. If None, the engine default is used.
        Only valid when `engine` is given by name.
    ctype : str or dict, optional
        The coordinate type(s) to use. If None, the engine defaults are used.
        Only valid when `engine` is given by name.
    **engine_kwargs
        Format-specific engine parameters forwarded to the engine constructor.
        Only valid when `engine` is given by name.

    Returns
    -------
    DataCollection
        The combined data collection

    """
    paths = _ensure_str_paths(paths)
    if engine is not None:
        engine = _resolve_engine(engine, vtype, ctype, engine_kwargs)

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
    if len(paths) > MAX_OPEN_FILES:
        raise NotImplementedError(
            f"cannot open {len(paths)} files at once: the limit is "
            f"{MAX_OPEN_FILES}, because the scan holds one data collection per "
            "file in memory. Open the files in batches and combine the results."
        )
    max_workers = get_scan_workers(parallel, len(paths))
    if max_workers == 1:
        if verbose:
            iterator = tqdm(paths, desc="Fetching metadata from files")
        else:
            iterator = paths
        objs = [open_datacollection(path, engine=engine) for path in iterator]
    else:
        executor = get_scan_pool(max_workers)
        futures = [
            executor.submit(open_datacollection, path, engine=engine) for path in paths
        ]
        if verbose:
            iterator = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Fetching metadata from files",
            )
        else:
            iterator = as_completed(futures)
        objs = [future.result() for future in iterator]
    # the native format stacks hdf5 sources; the engines that describe a file
    # as a collection are tile-backed, and let `concat` pick
    virtual = True if engine is None else None
    return combine_by_field(objs, dim, tolerance, squeeze, virtual, verbose)


def open_mfdatatree(
    paths,
    dim="first",
    tolerance=None,
    squeeze=False,
    engine=None,
    vtype=None,
    ctype=None,
    verbose=False,
    parallel=None,
    **engine_kwargs,
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
    tolerance : float or timedelta64, optional
        During concatenation, the tolerance to consider that the end of a file is
        continuous with beginning of the following one. For time coordinates, numeric
        values are considered as seconds. Default to zero tolerance.
    squeeze : bool, optional
        Whether to return a DataArray instead of a DataCollection if the combination
        results in a data collection containing a unique data array.
    engine: str or Engine, optional
        The file format engine to use, given by name or as a configured
        :class:`~xdas.io.Engine` instance. Default to format auto-detection.
    vtype : str, optional
        The virtualization type to use. If None, the engine default is used.
        Only valid when `engine` is given by name (or None).
    ctype : str or dict, optional
        The coordinate type(s) to use. If None, the engine defaults are used.
        Only valid when `engine` is given by name (or None).
    parallel: bool or int, optional
        Whether to use multiprocessing to fetch file metadata. If False or 1,
        runs in single-process mode. If an integer, use that many processes.
        If True, use as many processes as available cores. If None, use the
        global xdas configuration. Default to None.
    verbose: bool
        Whether to display a progress bar. Default to False.
    **engine_kwargs
        Format-specific engine parameters forwarded to the engine constructor
        (e.g. ``overlaps`` for "febus"). Only valid when `engine` is given by name.

    Returns
    -------
    DataCollection
        The collected data.

    Examples
    --------
    >>> import xdas as xd
    >>> paths = "/data/{node}/{cable}/[acquisition]/proc/[acquisition].h5"
    >>> xd.open_mfdatatree(paths, engine="asn") # doctest: +SKIP
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
    paths = _ensure_str_paths(paths)

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

    return collect(
        tree,
        fields,
        dim,
        tolerance,
        squeeze,
        engine,
        vtype,
        ctype,
        parallel,
        verbose,
        **engine_kwargs,
    )


def collect(
    tree,
    fields,
    dim="first",
    tolerance=None,
    squeeze=False,
    engine=None,
    vtype=None,
    ctype=None,
    parallel=None,
    verbose=False,
    **engine_kwargs,
):
    """
    Collect the data from a tree of paths using `fields` as level names.

    Parameters
    ----------
    tree : nested dict of lists
        The paths grouped in a tree hierarchy.
    fields : tuple of str
        The names of the levels of the tree hierarchy.
    dim : str, optional
        The dimension along which the data arrays are concatenated. Default to "first".
    tolerance : float or timedelta64, optional
        During concatenation, the tolerance to consider that the end of a file is
        continuous with beginning of the following one. For time coordinates, numeric
        values are considered as seconds. Default to zero tolerance.
    squeeze : bool, optional
        Whether to return a DataArray instead of a DataCollection if the combination
        results in a data collection containing a unique data array.
    engine: str or Engine, optional
        The file format engine to use, given by name or as a configured
        :class:`~xdas.io.Engine` instance. Default to format auto-detection.
    vtype : str, optional
        The virtualization type to use. If None, the engine default is used.
        Only valid when `engine` is given by name (or None).
    ctype : str or dict, optional
        The coordinate type(s) to use. If None, the engine defaults are used.
        Only valid when `engine` is given by name (or None).
    parallel: bool or int, optional
        Whether to use multiprocessing to fetch file metadata. If False or 1,
        runs in single-process mode. If an integer, use that many processes.
        If True, use as many processes as available cores. If None, use the
        global xdas configuration. Default to None.
    verbose: bool
        Whether to display a progress bar. Default to False.
    **engine_kwargs
        Format-specific engine parameters forwarded to the engine constructor
        (e.g. ``overlaps`` for "febus"). Only valid when `engine` is given by name.


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
                value,
                dim,
                tolerance,
                squeeze,
                engine,
                vtype,
                ctype,
                parallel,
                verbose,
                **engine_kwargs,
            )
            dc.name = fields[0]
            collection[key] = dc
        else:
            collection[key] = collect(
                value,
                fields,
                dim,
                tolerance,
                squeeze,
                engine,
                vtype,
                ctype,
                parallel,
                verbose,
                **engine_kwargs,
            )
    return collection


def defaulttree(depth):
    """Generate a default tree of lists with given depth."""
    if depth == 1:
        return []
    else:
        return defaultdict(lambda: defaulttree(depth - 1))


def _resolve_engine(engine, vtype, ctype, engine_kwargs):
    """Turn the `engine` argument of the open functions into an Engine instance."""
    from ..io.core import Engine

    if isinstance(engine, Engine):
        if vtype is not None or ctype is not None or engine_kwargs:
            raise ValueError(
                "`vtype`, `ctype` and engine keyword arguments cannot be combined "
                "with an already configured engine instance; configure the instance "
                "instead"
            )
        return engine
    elif engine is None or isinstance(engine, str):
        return Engine[engine](vtype=vtype, ctype=ctype, **engine_kwargs)
    else:
        raise TypeError(
            "engine must be None, a registered engine name or an Engine instance, "
            f"found {type(engine)}"
        )


def open_mfdataarray(
    paths,
    dim="first",
    tolerance=None,
    squeeze=True,
    engine=None,
    vtype=None,
    ctype=None,
    parallel=None,
    verbose=False,
    **engine_kwargs,
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
    tolerance : float or timedelta64, optional
        During concatenation, the tolerance to consider that the end of a file is
        continuous with beginning of the following one. For time coordinates, numeric
        values are considered as seconds. Default to zero tolerance.
    squeeze : bool, optional
        Whether to return a DataArray instead of a DataCollection if the combination
        results in a data collection containing a unique data array.
    engine: str or Engine, optional
        The file format engine to use, given by name or as a configured
        :class:`~xdas.io.Engine` instance. Default to format auto-detection.
    vtype : str, optional
        The virtualization type to use. If None, the engine default is used.
        Only valid when `engine` is given by name (or None).
    ctype : str or dict, optional
        The coordinate type(s) to use. If None, the engine defaults are used.
        Only valid when `engine` is given by name (or None).
    parallel: bool or int, optional
        Whether to use multiprocessing to fetch file metadata. If False or 1,
        runs in single-process mode. If an integer, use that many processes.
        If True, use as many processes as available cores. If None, use the
        global xdas configuration. Default to None.
    verbose: bool
        Whether to display a progress bar. Default to False.
    **engine_kwargs
        Format-specific engine parameters forwarded to the engine constructor
        (e.g. ``overlaps`` for "febus"). Only valid when `engine` is given by name.

    Returns
    -------
    DataArray or DataSequence
        The data array containing all files data. If different acquisitions are found,
        a DataSequence is returned.

    Raises
    ------
    FileNotFound
        If no file can be found.
    NotImplementedError
        If more than `MAX_OPEN_FILES` files are given with a vtype that does not
        consolidate (see `VirtualBackend.consolidates`). A consolidating vtype scans any
        number of files, `MAX_OPEN_FILES` at a time; the others hold every scan
        product until the end, so larger sets must be opened in batches and
        combined with `combine_by_coords`.
    """
    paths = _ensure_str_paths(paths)
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
    engine = _resolve_engine(engine, vtype, ctype, engine_kwargs)
    backend = VirtualBackend._registry.get(engine.vtype)
    if (backend is None or not backend.consolidates) and len(paths) > MAX_OPEN_FILES:
        consolidating = ", ".join(
            repr(vtype)
            for vtype, cls in sorted(VirtualBackend._registry.items())
            if cls.consolidates
        )
        raise NotImplementedError(
            f"cannot open {len(paths)} files at once with vtype "
            f"{engine.vtype!r}: the limit is {MAX_OPEN_FILES}, because its scan "
            "products cannot be consolidated into a compact one. Open the files "
            "in batches and pass the results to `combine_by_coords`, or use a "
            f"vtype that consolidates ({consolidating}), which has no ceiling."
        )
    max_workers = get_scan_workers(parallel, len(paths))
    objs = []  # pending scan products, drained into `runs` every MAX_OPEN_FILES
    runs = []  # per-batch continuous runs (streaming mode only)
    failures = []

    def consume(da):
        # stream the combine: every MAX_OPEN_FILES scan products are fused into
        # compact runs (losslessly: no coordinate simplification) and freed,
        # so memory is bounded by the batch, not the archive
        objs.append(da)
        if len(objs) >= MAX_OPEN_FILES:
            runs.extend(combine_by_coords(objs, dim, False, False))
            objs.clear()

    if max_workers == 1:
        iterator = (
            tqdm(paths, desc="Fetching metadata from files") if verbose else paths
        )
        for path in iterator:
            try:
                consume(open_dataarray(path, engine=engine))
            except Exception as error:  # noqa: BLE001 - collected and warned below
                failures.append((path, error))
                warnings.warn(f"could not open {path}: {error}", RuntimeWarning)
    else:
        executor = get_scan_pool(max_workers)
        futures_to_paths = {
            executor.submit(open_dataarray, path, engine=engine): path for path in paths
        }
        if verbose:
            iterator = tqdm(
                as_completed(futures_to_paths),
                total=len(futures_to_paths),
                desc="Fetching metadata from files",
            )
        else:
            iterator = as_completed(futures_to_paths)
        for future in iterator:
            try:
                obj = future.result()
            except Exception as error:  # noqa: BLE001 - collected and warned below
                path = futures_to_paths[future]
                failures.append((path, error))
                warnings.warn(f"could not open {path}: {error}", RuntimeWarning)
            else:
                consume(obj)
    if not objs and not runs:  # there must be failures
        path, error = failures[0]
        raise RuntimeError(
            f"could not open any file with engine: "
            f"{engine.name or type(engine).__name__}; "
            f"first failure was {path}: {error}"
        ) from error
    if not runs:
        # a single batch: the exact monolithic path
        return combine_by_coords(objs, dim, tolerance, squeeze, None, verbose)
    if objs:
        runs.extend(combine_by_coords(objs, dim, False, False))
        objs.clear()
    return _combine_runs(runs, dim, tolerance, squeeze)


def _combine_runs(runs, dim, tolerance, squeeze):
    """Fuse the compact runs of a streamed scan into the final collection.

    Runs are grouped by compatibility signature (unlike the monolithic
    walk, grouping does not depend on time order, so acquisitions
    interleaved in time still fuse into one array each). Each group is
    concatenated without simplification — losslessly, whatever the arrival
    order — then :func:`sortby` permutes the tiles into coordinate order
    and spends the whole *tolerance* budget once, on sorted segments:
    the same state, and so the same result, as the monolithic combine.
    Groups whose data or coordinate :func:`sortby` cannot permute (eager
    data, non-interpolated coordinates) are concatenated with *tolerance*
    directly, correct whenever batches do not interleave in time.
    """
    if dim == "first":
        dim = runs[0].dims[0]
    if dim == "last":
        dim = runs[0].dims[-1]
    bags = []
    for da in runs:
        for bag in bags:
            try:
                bag.append(da)
                break
            except CompatibilityError:
                continue
        else:
            bag = Bag(dim)
            bag.append(da)
            bags.append(bag)
    results = []
    for bag in bags:
        try:
            fused = sortby(concat(bag, dim, tolerance=False), dim, tolerance)
        except (KeyError, ValueError, NotImplementedError):
            fused = concat(bag, dim, tolerance)
        results.append(fused)
    if all(dim in da.coords for da in results):
        results.sort(
            key=lambda da: (
                da[dim][0].values
                if isinstance(da[dim], AxisCoordinate)
                else da[dim].values
            )
        )
    collection = DataCollection(results, "acquisition")
    if squeeze and len(collection) == 1:
        return collection[0]
    return collection


def open_dataarray(fname, engine=None, vtype=None, ctype=None, **engine_kwargs):
    """
    Open a dataarray.

    Parameters
    ----------
    fname : str
        The path of the dataarray.
    engine: str or Engine, optional
        The file format engine to use, given by name or as a configured
        :class:`~xdas.io.Engine` instance. Default to format auto-detection.
    vtype : str, optional
        The virtualization type to use. If None, the engine default is used.
        Only valid when `engine` is given by name (or None).
    ctype : str or dict, optional
        The coordinate type(s) to use. If None, the engine defaults are used.
        Only valid when `engine` is given by name (or None).
    **engine_kwargs
        Format-specific engine parameters forwarded to the engine constructor
        (e.g. ``overlaps`` for "febus"). Only valid when `engine` is given by name.

    Returns
    -------
    DataArray
        The opened dataarray.

    Raises
    ------
    TypeError
        If `engine` is neither None, an engine name nor an Engine instance, or
        if an engine keyword argument is unknown to the engine.
    ValueError
        If `vtype`, `ctype` or engine keyword arguments are combined with an
        already configured engine instance.
    FileNotFoundError
        If no file can be found.
    """
    # parse & checks
    fname = _ensure_str_paths(fname)
    if not os.path.exists(fname):
        raise FileNotFoundError("no file to open")

    # dispatch & open
    engine = _resolve_engine(engine, vtype, ctype, engine_kwargs)
    return engine.open_dataarray(fname)


def open_datacollection(
    fname, group=None, engine=None, vtype=None, ctype=None, **engine_kwargs
):
    """
    Open a DataCollection from a file.

    Parameters
    ----------
    fname : str
        The path of the DataCollection.
    group : str, optional
        The location of the data collection within the file. Root by default.
        Only meaningful for the native format.
    engine: str or Engine, optional
        The file format engine to use, given by name or as a configured
        :class:`~xdas.io.Engine` instance. Default to the native format.
    vtype : str, optional
        The virtualization type to use. If None, the engine default is used.
        Only valid when `engine` is given by name.
    ctype : str or dict, optional
        The coordinate type(s) to use. If None, the engine defaults are used.
        Only valid when `engine` is given by name.
    **engine_kwargs
        Format-specific engine parameters forwarded to the engine constructor.
        Only valid when `engine` is given by name.

    Returns
    -------
    DataCollection
        The opened DataCollection.

    Raises
    ------
    FileNotFound
        If no file can be found.
    NotImplementedError
        If the engine does not describe a file as a collection.
    """
    fname = _ensure_str_paths(fname)
    if not os.path.exists(fname):
        raise FileNotFoundError("no file to open")
    if engine is None:
        if vtype is not None or ctype is not None or engine_kwargs:
            raise ValueError(
                "`vtype`, `ctype` and engine keyword arguments require naming an "
                "engine; the native format reads a collection as it was written"
            )
        return DataCollection.from_netcdf(fname, group)
    if group is not None:
        raise ValueError(
            "`group` is a native-format parameter; pass it as an engine keyword "
            "argument instead"
        )
    return _resolve_engine(engine, vtype, ctype, engine_kwargs).open_datacollection(
        fname
    )


def asdataarray(obj, tolerance=None):
    """
    Try to convert given object to a dataarray.

    Only supports DataArray or xr.DataArray as input.

    Parameters
    ----------
    obj : object
        The objected to convert
    tolerance : float or datetime64, optional
        For dense coordinates, tolerance error for interpolation representation.
        For time coordinates, numeric values are considered as seconds.
        Zero by default.

    Returns
    -------
    DataArray
        The object converted to a DataArray. Data is not copied.

    Raises
    ------
    ValueError
        If the object cannot be converted to a DataArray.
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
    tolerance : float or timedelta64, optional
        The tolerance to consider that the end of a file is continuous with beginning of
        the following. For time coordinates, numeric  values are considered as seconds.
        Zero by default.
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
        # the level is named for what its elements are, and combining changes
        # that: whatever the inputs held, each output element is one
        # acquisition epoch. `combine_by_coords` names it.
        return combine_by_coords(objs, dim, tolerance, squeeze, virtual, verbose)
    elif nodes and not leaves:
        (name,) = {dc.name for dc in nodes}
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
    tolerance : float or timedelta64, optional
        The tolerance to consider that the end of a file is continuous with beginning of
        the following. For time coordinates, numeric values are considered as seconds.
        Zero by default.
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
            key=lambda da: (
                da[dim][0].values
                if isinstance(da[dim], AxisCoordinate)
                else da[dim].values
            ),
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

    # concatenate each bag. `Bag` splits on sampling rate, dtype and non-concat
    # coordinates, and gaps land inside the coordinate, so every element of the
    # result is one acquisition epoch — which is what the level is named for.
    collection = DataCollection(
        [concatenate(bag, dim, tolerance, virtual, verbose) for bag in bags],
        "acquisition",
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
    """
    Accumulator that collects :class:`DataArray` objects for concatenation along *dim*.

    Compatibility checks (dims, shape, coords, sampling interval, dtype) are run on
    each appended object; incompatible objects raise :exc:`CompatibilityError` so the
    caller can start a new bag.
    """

    def __init__(self, dim):
        self.objs = []
        self.dim = dim

    def __iter__(self):
        return iter(self.objs)

    def initialize(self, da):
        """Set *da* as the first element and record its shape, coords, sampling interval, and dtype."""
        self.objs = [da]
        self.dims = da.dims
        self.subshape = tuple(size for dim, size in da.sizes.items() if dim != self.dim)
        self.subcoords = (
            da.coords.drop_dims(self.dim)
            if self.dim in self.dims
            else da.coords.drop_coords(self.dim)
        )
        self.delta = self._get_delta(da)
        self.dtype = da.dtype

    def _get_delta(self, da):
        """Nominal sampling interval of *da* along *dim*, or ``None`` (irregular or absent)."""
        if self.dim not in da.coords:
            return None
        coord = da.coords[self.dim]
        if not isinstance(coord, AxisCoordinate):
            return None
        return coord.get_sampling_interval()

    def append(self, da):
        """Add *da* after running all compatibility checks; initialises on first call."""
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
        """Raise :exc:`CompatibilityError` if *da* has different dimensions."""
        if not self.dims == da.dims:
            raise CompatibilityError("dimensions are not compatible")

    def check_shape(self, da):
        """Raise :exc:`CompatibilityError` if *da* has a different non-concat shape."""
        subshape = tuple(size for dim, size in da.sizes.items() if dim != self.dim)
        if not self.subshape == subshape:
            raise CompatibilityError("shapes are not compatible")

    def check_dtype(self, da):
        """Raise :exc:`CompatibilityError` if *da* has a different dtype."""
        if not self.dtype == da.dtype:
            raise CompatibilityError("data types are not compatible")

    def check_coords(self, da):
        """Raise :exc:`CompatibilityError` if *da* has incompatible non-concat coordinates."""
        subcoords = (
            da.coords.drop_dims(self.dim)
            if self.dim in self.dims
            else da.coords.drop_coords(self.dim)
        )
        if not self.subcoords.equals(subcoords):
            raise CompatibilityError("coordinates are not compatible")

    def check_sampling_interval(self, da):
        """Raise :exc:`CompatibilityError` if *da* has a different sampling interval."""
        if self.delta is None:
            pass
        else:
            delta = self._get_delta(da)
            if delta is None or not np.isclose(delta, self.delta):
                raise CompatibilityError("sampling intervals are not compatible")


def _get_promoted_coords(objs, dim):
    """Check the non-concat coordinates of *objs* and report the varying scalars.

    Called when *dim* opens a new dimension, where :func:`concat` would
    otherwise keep the first element's coordinates and silently discard the
    others'. Coordinates that every element shares are kept as they are;
    scalar ones that differ are what the new dimension is made of, and are
    returned for promotion to a coordinate along it — the ``channel`` of a
    stack of seismic traces, say. Anything else is a genuine incompatibility.

    Parameters
    ----------
    objs : list of DataArray
        The data arrays about to be concatenated, before ``expand_dims``.
    dim : str
        The name of the new dimension. A coordinate of that name is left
        alone: ``expand_dims`` promotes it.

    Returns
    -------
    dict
        Mapping from coordinate name to the list of per-element scalar values,
        in the order of *objs*.
    """
    names = [name for name in objs[0].coords if name != dim]
    for da in objs[1:]:
        other = [name for name in da.coords if name != dim]
        if set(other) != set(names):
            raise ValueError(
                "objects to concatenate along the new dimension "
                f"{dim!r} must share their coordinates; got {sorted(names)} "
                f"and {sorted(other)}"
            )
    promoted = {}
    for name in names:
        coord = objs[0].coords[name]
        if all(da.coords[name].equals(coord) for da in objs[1:]):
            continue
        if not all(da.coords[name].dim is None for da in objs):
            raise ValueError(
                f"coordinate {name!r} differs across the objects to concatenate "
                f"along the new dimension {dim!r}; only scalar coordinates may "
                "vary, and are then promoted to a coordinate along that dimension"
            )
        promoted[name] = [da.coords[name].values for da in objs]
    return promoted


def concat(
    objs,
    dim="first",
    tolerance=None,
    virtual=None,
    verbose=None,
    *,
    reduce=True,
    regularize=False,
):
    """
    Concatenate data arrays along a given dimension.

    Parameters
    ----------
    objs : list of DataArray
        List of data arrays to concatenate.
    dim : str
        The dimension along which concatenate.
    tolerance : float or timedelta64, optional
        The tolerance to consider that the end of a file is continuous with beginning of
        the following, For time coordinates, numeric values are considered as seconds.
        By default each coordinate spends its own declared tolerance when it
        carries one, else a zero-like default. Pass ``False`` to disable
        simplification entirely.
    virtual : bool, optional
        Whether to create a virtual dataset. It requires that all concatenated
        data arrays are virtual. By default tries to create a virtual dataset if possible.
    verbose: bool
        Whether to display a progress bar.
    reduce : bool, optional
        Whether to drop redundant tie points from the concatenated coordinate.
        Default True.
    regularize : bool, optional
        Whether to promote the concatenated coordinate to a regular one when its
        segments admit a single shared rate within *tolerance*. Default False:
        regular inputs already stay regular through concatenation, so promotion
        only matters for irregular inputs and stays opt-in.

    Returns
    -------
    DataArray
        The concatenated dataarray. Coordinates along axes other than *dim* are
        taken from the first element. When *dim* opens a new dimension, the
        other elements must carry the same non-concat coordinates, except for
        scalar ones, which are promoted to a coordinate along the new dimension
        when they vary.

    Raises
    ------
    ValueError
        If *dim* opens a new dimension and the elements do not agree on their
        non-concat coordinates other than varying scalars.

    """
    objs = list(objs)
    non_empty = [da for da in objs if not da.empty]
    if not non_empty:
        return objs[0] if objs else DataArray([])
    objs = non_empty

    if dim in objs[0].dims + ("first", "last"):
        axis = objs[0].get_axis_num(dim)
        dim = objs[0].dims[axis]  # ensure not "first" or "last"
        dims = objs[0].dims
        promoted = {}
    else:
        axis = 0
        dims = (dim, *objs[0].dims)
        promoted = _get_promoted_coords(objs, dim)
        objs = [da.expand_dims(dim) for da in objs]

    # inferred on what will actually be concatenated: opening a new dimension
    # goes through `expand_dims`, which a `VirtualSource` cannot follow (a
    # stack of sources is a longer axis, never an extra one) and so loads.
    # Inferring beforehand would promise a `VirtualStack` of dense arrays.
    if virtual is None:
        virtual = all(isinstance(da.data, (VirtualSource, VirtualStack)) for da in objs)

    coords = objs[0].coords.drop_dims(dim)
    name = objs[0].name
    attrs = objs[0].attrs
    dim_has_coords = dim in objs[0].coords
    order = list(range(len(objs)))

    if dim_has_coords:
        coord, order = concat_coords(
            [obj[dim] for obj in objs],
            sort=True,
            return_order=True,
            tolerance=tolerance,
            reduce=reduce,
            regularize=regularize,
        )
        objs = [objs[idx] for idx in order]
        coords[dim] = coord

    for coord_name, values in promoted.items():
        coords[coord_name] = (dim, [values[idx] for idx in order])

    iterator = (
        tqdm(objs, desc="Linking dataarray") if verbose else objs
    )  # TODO : remove tqdm?
    data = []
    for da in iterator:
        if isinstance(da.data, VirtualStack):
            data.extend(da.data.sources)
        else:
            data.append(da.data)

    if virtual:
        data = VirtualStack(data, axis)
    else:
        data = np.concatenate(data, axis)

    return DataArray(data, coords, dims, name, attrs)


concatenate = concat  # TODO: deprecate it


#: The alignment strategies :func:`stack` accepts. Deliberately an open
#: enumeration rather than a boolean: SeisBench answers the same mismatch by
#: *splitting the record* into maximal stretches of constant member coverage
#: (``GroupingHelper._get_intervals``), which is the right shape for ragged
#: station deployments and would join this tuple as a further mode rather than
#: replace it.
JOIN_METHODS = (None, "inner", "outer")

#: Default agreement tolerance of :func:`stack`, as a fraction of the nominal
#: sampling interval. One percent of a sample is far above the sub-nanosecond
#: rounding real acquisitions differ by (a reference three-component station
#: was measured 1 ns apart at 40 Hz, i.e. 4e-8 of a sample) and far below any
#: misalignment worth reporting: it takes a hundred times the budget to hide a
#: one-sample shift, and fifty to hide the half-sample one that would already
#: change which sample a value lands on.
SNAP_FRACTION = 1e-2


def stack(dc, level, dim=None, join=None, tolerance=None):
    """
    Collapse a level of a data collection into an array dimension.

    The inverse of :func:`combine_by_coords`, which concatenates *along* an
    existing dimension: here the keys of one collection level become the
    coordinate of a *new* dimension, and everything below that level is merged
    in lock-step. Stacking the ``channel`` level of a seismological collection
    turns each station's three traces into one ``(channel, time)`` array.

    The new dimension is named after the level it collapsed, so nothing is
    renamed behind your back; pass *dim* to choose another name.

    Parameters
    ----------
    dc : DataCollection
        The collection to stack.
    level : str
        The name of the level to collapse. Must name one of ``dc.fields``.
        Only the outermost occurrence of that name on each branch is
        collapsed.
    dim : str, optional
        The name of the new dimension. Defaults to *level*. It must not
        already name a dimension of the leaves.
    join : None or str, optional
        How to reconcile leaves that do not share their coordinates. ``None``
        (default) raises, naming what disagreed. ``"inner"`` keeps the
        coordinate values every leaf has, ``"outer"`` keeps the values any
        leaf has and fills the missing samples with NaN. Aligning materialises
        the joined coordinates and, for ``"outer"``, the data.
    tolerance : scalar, None, or ``False``, optional
        How far apart two leaves may describe the same sampling grid and still
        count as agreeing (see the notes). ``None`` (default) spends
        :data:`SNAP_FRACTION` of the nominal sampling interval, and only on
        coordinates that declare one. ``False`` disables snapping, restoring
        strict equality. A scalar is an absolute budget in coordinate units
        (seconds for a datetime axis) and applies to every axis coordinate,
        declared spacing or not.

    Returns
    -------
    DataCollection or DataArray
        The collection with the level collapsed. When *level* is the outermost
        level and the leaves sit directly below it, the result is a single
        data array.

    Raises
    ------
    KeyError
        If *level* names no level of the collection.
    ValueError
        If the sub-trees below the collapsed level do not agree structurally,
        or if their leaves do not agree on their other coordinates and *join*
        does not resolve it.

    Notes
    -----
    Stacking is a :func:`concat` over the leaves, so it inherits its
    behaviour: the stacked coordinate is sorted, and scalar coordinates that
    vary from leaf to leaf are promoted onto the new dimension.

    **Agreement is judged on the sampling grid, not on tie points.** Two
    acquisitions of one instrument routinely round their start time
    differently by a fraction of a sample; those are the same coordinate, and
    comparing them exactly would raise on data that is perfectly aligned — or,
    worse, send it to ``join="outer"``, which would interleave the two grids
    into an array twice as long. So before any mismatch is reported, leaves
    whose axis coordinates have the same length and stay within *tolerance*
    of each other everywhere are snapped onto **the first leaf's coordinate**,
    in the collection's own key order; the sub-sample offset of the others is
    dropped. Only leaves that then still disagree are reported or joined.

    Snapping is deliberately narrow. It never changes a length, it never moves
    a value by as much as a sample, and by default it only applies to
    coordinates that declare a nominal sampling interval — without one there
    is no grid to snap to, and structurally different descriptions of the same
    values stay a *join*'s business rather than an equality's.

    Tile-backed leaves stay tile-backed, so stacking a collection of virtual
    arrays reads nothing. Alignment is where that can stop: ``"inner"`` slices
    the leaves to a shared span, which stays virtual only while the resulting
    tile geometries still agree, and ``"outer"`` has to write the NaNs and so
    always materialises.

    See Also
    --------
    concat : the primitive this is built on, over one list of arrays.
    combine_by_coords : concatenate along an existing dimension instead.

    Examples
    --------
    >>> import numpy as np
    >>> import xdas as xd

    >>> def trace(channel):
    ...     return xd.DataArray(
    ...         np.arange(4.0),
    ...         {"channel": (None, channel), "time": [0.0, 1.0, 2.0, 3.0]},
    ...     )

    >>> dc = xd.DataCollection(
    ...     {
    ...         "SX01": ("channel", {code: trace(code) for code in ["SHZ", "SHN"]}),
    ...         "SX02": ("channel", {code: trace(code) for code in ["SHZ", "SHN"]}),
    ...     },
    ...     "station",
    ... )
    >>> dc
    Station:
      SX01:
        Channel:
          SHZ: <xdas.DataArray (time: 4)>
          SHN: <xdas.DataArray (time: 4)>
      SX02:
        Channel:
          SHZ: <xdas.DataArray (time: 4)>
          SHN: <xdas.DataArray (time: 4)>

    >>> stacked = xd.stack(dc, "channel")
    >>> stacked
    Station:
      SX01: <xdas.DataArray (channel: 2, time: 4)>
      SX02: <xdas.DataArray (channel: 2, time: 4)>

    >>> stacked["SX01"]["channel"].values
    array(['SHN', 'SHZ'], dtype='<U3')

    """
    if join not in JOIN_METHODS:
        raise ValueError(
            f"unknown join method {join!r}; expected one of "
            + ", ".join(repr(method) for method in JOIN_METHODS)
        )
    if not isinstance(dc, DataCollection):
        raise TypeError(
            f"can only stack a level of a data collection, got {type(dc).__name__}"
        )
    if level not in dc.fields:
        raise KeyError(
            f"{level!r} does not name any level of the collection; "
            f"available: {list(dc.fields)}"
        )
    if dim is None:
        dim = level
    return _stack_level(dc, level, dim, join, tolerance)


def _stack_level(obj, level, dim, join, tolerance):
    """Return *obj* with the outermost node named *level* of each branch collapsed."""
    if isinstance(obj, DataArray):
        return obj
    if obj.name == level:
        keys, values = _entries(obj)
        if not values:
            raise ValueError(f"level {level!r} is empty; there is nothing to stack")
        return _stack_entries(values, keys, level, dim, join, tolerance, ())
    if obj.ismapping():
        data = {
            key: _stack_level(value, level, dim, join, tolerance)
            for key, value in obj.items()
        }
    else:
        data = [_stack_level(value, level, dim, join, tolerance) for value in obj]
    return DataCollection(data, obj.name)


def _entries(node):
    """Return the ``(keys, values)`` of a collection node, keying a sequence by position."""
    if node.ismapping():
        return list(node.keys()), list(node.values())
    return list(range(len(node))), list(node)


def _kind(obj):
    """Return a human-readable description of what *obj* is, as structure only."""
    if isinstance(obj, DataArray):
        return "a data array"
    what = "mapping" if obj.ismapping() else "sequence"
    return f"a {obj.name!r} {what}"


def _at(path):
    """Render a tree *path* as a locating suffix, empty at the root."""
    return f" at {'/'.join(path)}" if path else ""


def _stack_entries(objs, keys, level, dim, join, tolerance, path):
    """Merge the sub-trees of one collapsed node, in lock-step, down to the leaves."""
    kinds = [_kind(obj) for obj in objs]
    if len(set(kinds)) > 1:
        raise ValueError(
            f"the sub-trees{_at(path)} of level {level!r} do not agree: "
            + ", ".join(f"{key!r} is {kind}" for key, kind in zip(keys, kinds))
        )
    if isinstance(objs[0], DataArray):
        return _stack_arrays(objs, keys, level, dim, join, tolerance, path)
    name = objs[0].name
    if name == level:
        raise ValueError(
            f"level {level!r} is nested under itself{_at(path)}; stacking two "
            "levels sharing a name is not supported"
        )
    if objs[0].ismapping():
        subkeys = list(objs[0])
        for key, obj in zip(keys[1:], objs[1:]):
            if set(obj) != set(subkeys):
                raise ValueError(
                    f"the {name!r} level{_at(path)} does not hold the same keys "
                    f"under every {level!r}: {keys[0]!r} has {sorted(subkeys)} "
                    f"and {key!r} has {sorted(obj)}"
                )
        data = {
            subkey: _stack_entries(
                [obj[subkey] for obj in objs],
                keys,
                level,
                dim,
                join,
                tolerance,
                (*path, f"{name}={subkey}"),
            )
            for subkey in subkeys
        }
    else:
        length = len(objs[0])
        for key, obj in zip(keys[1:], objs[1:]):
            if len(obj) != length:
                raise ValueError(
                    f"the {name!r} level{_at(path)} does not hold the same number "
                    f"of elements under every {level!r}: {keys[0]!r} has {length} "
                    f"and {key!r} has {len(obj)}"
                )
        data = [
            _stack_entries(
                [obj[index] for obj in objs],
                keys,
                level,
                dim,
                join,
                tolerance,
                (*path, f"{name}={index}"),
            )
            for index in range(length)
        ]
    return DataCollection(data, name)


def _stack_arrays(objs, keys, level, dim, join, tolerance, path):
    """Concatenate one lock-step group of leaves onto the new dimension *dim*."""
    for key, obj in zip(keys, objs):
        if dim in obj.dims:
            raise ValueError(
                f"cannot stack level {level!r} onto {dim!r}: the leaf {key!r}"
                f"{_at(path)} already has a {dim!r} dimension; pass `dim=` to "
                "name the new dimension otherwise"
            )
    objs = _snap_leaves(objs, tolerance)
    messages, joinable = _leaf_mismatches(objs, keys)
    if messages and join is not None and joinable:
        objs = _join_leaves(objs, keys, joinable, join)
        messages, _ = _leaf_mismatches(objs, keys)
        joinable = []  # already spent: what remains is not a join away
    if messages:
        hint = (
            " (pass join='inner' or join='outer' to align them first)"
            if joinable and join is None
            else ""
        )
        raise ValueError(
            f"the leaves{_at(path)} of level {level!r} do not agree: "
            + "; ".join(messages)
            + hint
        )
    objs = [_with_key(obj, dim, key) for obj, key in zip(objs, keys)]
    return concat(objs, dim)


def _snap_leaves(objs, tolerance):
    """Put the leaves on one representation of every grid they share within *tolerance*.

    The first leaf is the reference; any other whose axis coordinate has the
    same length and stays within *tolerance* of the reference everywhere
    adopts it verbatim, so the strict equality that follows sees one
    coordinate instead of two roundings of it. ``tolerance=False`` disables
    the whole pass.
    """
    if tolerance is False or len(objs) < 2:
        return objs
    reference = objs[0]
    out = [reference]
    for obj in objs[1:]:
        snapped = {
            name: reference.coords[name]
            for name, coord in obj.coords.items()
            if name in reference.coords
            and _same_grid(reference.coords[name], coord, tolerance)
        }
        if snapped:
            obj = obj.copy(deep=False)
            for name, coord in snapped.items():
                obj.coords[name] = coord.copy()
        out.append(obj)
    return out


def _same_grid(reference, coord, tolerance):
    """Whether *coord* describes *reference*'s grid to within *tolerance*.

    Both must be axis coordinates of the same dimension, same length and same
    kind of values; the deviation is then measured exactly. Coordinates are
    piecewise linear in their index, so comparing them at the union of their
    breakpoints bounds their distance everywhere in between.
    """
    if not (
        isinstance(reference, AxisCoordinate)
        and isinstance(coord, AxisCoordinate)
        and reference.dim == coord.dim
        and len(reference) == len(coord)
        and len(coord) > 0
    ):
        return False
    if any(
        not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.datetime64))
        for dtype in (reference.dtype, coord.dtype)
    ):
        return False
    if np.issubdtype(reference.dtype, np.datetime64) != np.issubdtype(
        coord.dtype, np.datetime64
    ):
        return False
    tolerance = _snap_tolerance(reference, coord, tolerance)
    if tolerance is None:
        return False
    indices = np.union1d(_breakpoints(reference), _breakpoints(coord))
    deviation = np.abs(reference._get_value(indices) - coord._get_value(indices))
    return bool(np.all(deviation <= tolerance))


def _snap_tolerance(reference, coord, tolerance):
    """Return the absolute budget to compare two coordinates with, or ``None``.

    An explicit *tolerance* is taken as given, in the coordinate's own units.
    The default is :data:`SNAP_FRACTION` of the nominal sampling interval, and
    ``None`` — do not snap at all — when neither coordinate declares one: a
    fraction of a sample means nothing on an axis that has no samples.
    """
    if tolerance is not None:
        return parse_scalar_delta(tolerance, reference.dtype)
    for candidate in (reference, coord):
        sampling_interval = candidate.get_sampling_interval(cast=False)
        if sampling_interval is not None:
            return np.abs(sampling_interval) * SNAP_FRACTION
    return None


def _breakpoints(coord):
    """Return the indices at which *coord*'s value curve may bend.

    Between two of them the values are affine in the index, which is what lets
    a comparison sampled there bound the deviation everywhere. A coordinate
    that ties values to indices bends at its tie points (plus the end of each
    segment when they carry lengths); one that stores every value is its own
    worst case and bends anywhere.
    """
    indices = getattr(coord, "tie_indices", None)
    if indices is None:
        return coord.indices
    lengths = getattr(coord, "tie_lengths", None)
    if lengths is None:
        return np.asarray(indices)
    return np.union1d(indices, np.asarray(indices) + np.asarray(lengths) - 1)


def _leaf_mismatches(objs, keys):
    """Report what the leaves disagree on, and which dimensions a join could fix.

    Returns a list of human-readable messages — empty when the leaves are
    stackable as they are — and the names of the dimension coordinates that
    differ but could be aligned.
    """
    messages = []
    dims = objs[0].dims
    for key, obj in zip(keys[1:], objs[1:]):
        if obj.dims != dims:
            messages.append(
                f"{keys[0]!r} has dimensions {dims} and {key!r} has {obj.dims}"
            )
    if messages:
        return messages, []
    names = list(objs[0].coords)
    for key, obj in zip(keys[1:], objs[1:]):
        missing = [name for name in names if name not in obj.coords]
        extra = [name for name in obj.coords if name not in names]
        if missing or extra:
            messages.append(
                f"{key!r} lacks the coordinates {missing} of {keys[0]!r} and "
                f"carries {extra} it does not"
            )
    if messages:
        return messages, []
    joinable = []
    for dim in dims:
        sizes = sorted({obj.sizes[dim] for obj in objs})
        if len(sizes) > 1 and dim not in objs[0].coords:
            messages.append(
                f"dimension {dim!r} has sizes {sizes} and no coordinate to align on"
            )
    for name in names:
        coords = [obj.coords[name] for obj in objs]
        if all(coord.equals(coords[0]) for coord in coords[1:]):
            continue
        if all(coord.dim is None for coord in coords):
            continue  # varying scalars are promoted onto the new dimension
        if name in dims and all(isinstance(coord, AxisCoordinate) for coord in coords):
            joinable.append(name)
        messages.append(f"coordinate {name!r} differs from one leaf to another")
    return messages, joinable


def _join_leaves(objs, keys, dims, join):
    """Reindex the leaves onto a shared index along each of *dims*."""
    for dim in dims:
        indices = [pd.Index(obj.coords[dim].values) for obj in objs]
        for key, index in zip(keys, indices):
            if not index.is_unique:
                raise ValueError(
                    f"cannot align on {dim!r}: the leaf {key!r} repeats coordinate "
                    "values; resolve its overlaps first (see `trim_overlaps`)"
                )
        target = indices[0]
        for index in indices[1:]:
            if join == "inner":
                target = target.intersection(index, sort=False)
            else:
                target = target.union(index, sort=None)
        if len(target) == 0:
            raise ValueError(
                f"cannot align on {dim!r}: the leaves share no coordinate value"
            )
        _refuse_interleaving(objs, dim, target)
        objs = [_reindex(obj, dim, target, index) for obj, index in zip(objs, indices)]
        objs = _unify_coord(objs, dim)
    return objs


def _refuse_interleaving(objs, dim, target):
    """Raise when the joined index holds more samples than its span can carry.

    An outer join over leaves that are on the same grid spans it once. Over
    leaves that are a fraction of a sample apart it spans it as many times as
    there are offsets, interleaving grids into an array that looks plausible
    and is mostly missing samples. The finest declared sampling interval says
    how many samples the joined span may hold; anything beyond that is
    interleaving, and silence would be the worst answer.
    """
    intervals = [
        obj.coords[dim].get_sampling_interval(cast=False)
        for obj in objs
        if obj.coords[dim].isregular()
    ]
    if len(intervals) < len(objs):
        return  # an irregular leaf declares no grid to violate
    sampling_interval = min(np.abs(interval) for interval in intervals)
    # rounded, not truncated: a float span is worth a sample either way, while
    # interleaving overshoots by a factor, never by one.
    expected = round((target.max() - target.min()) / sampling_interval) + 1
    if len(target) > expected:
        raise ValueError(
            f"cannot align on {dim!r}: the leaves are not on a common sampling "
            f"grid, and joining them would interleave {len(target)} samples "
            f"where the span holds {expected}; snap them together first with a "
            "larger `tolerance`"
        )


def _reindex(obj, dim, target, index):
    """Return *obj* with its *dim* axis put on *target*, staying lazy when it can."""
    positions = index.get_indexer(target)
    if len(index) == len(target) and np.array_equal(positions, np.arange(len(index))):
        return obj
    if (positions >= 0).all():
        start = int(positions[0])
        # a contiguous run is a slice, and slicing a virtual array reads nothing
        if np.array_equal(positions, np.arange(start, start + len(positions))):
            return obj.isel({dim: slice(start, start + len(positions))})
        return obj.isel({dim: positions})
    return _pad(obj, dim, target, positions)


def _pad(obj, dim, target, positions):
    """Return *obj* on *target*, filling the samples it does not have with NaN."""
    others = [
        name for name, coord in obj.coords.items() if coord.dim == dim and name != dim
    ]
    if others:
        raise ValueError(
            f"cannot pad along {dim!r}: the leaves carry the coordinates {others} "
            "along it, which have no value where the data is missing"
        )
    axis = obj.get_axis_num(dim)
    present = positions >= 0
    dtype = (
        obj.dtype
        if np.issubdtype(obj.dtype, np.inexact)
        else np.result_type(obj.dtype, np.float32)
    )
    shape = list(obj.shape)
    shape[axis] = len(target)
    data = np.full(tuple(shape), np.nan, dtype)
    key = [slice(None)] * obj.ndim
    key[axis] = np.nonzero(present)[0]
    data[tuple(key)] = np.asarray(obj.isel({dim: positions[present]}).data)
    coords = obj.coords.copy()
    coords[dim] = target.values
    return DataArray(data, coords, obj.dims, obj.name, obj.attrs)


def _unify_coord(objs, dim):
    """Give every leaf the same *dim* coordinate object once they agree on its values.

    Reindexing puts every leaf on the same values, but not necessarily on the
    same *representation*: two interpolated coordinates sliced out of
    differently tied inputs describe one grid with different tie points, and
    :meth:`Coordinate.equals` is structural. Normalising here is what lets the
    equality check that follows stay strict.
    """
    reference = objs[0].coords[dim]
    out = [objs[0]]
    for obj in objs[1:]:
        coord = obj.coords[dim]
        if not coord.equals(reference) and np.array_equal(
            coord.values, reference.values
        ):
            obj = obj.copy(deep=False)
            obj.coords[dim] = reference.copy()
        out.append(obj)
    return out


def _with_key(obj, dim, key):
    """Return *obj* carrying its level key as a scalar coordinate named *dim*.

    That is all it takes for :func:`concat` to open the new dimension with the
    keys as its coordinate: ``expand_dims`` promotes the scalar and
    ``concat_coords`` concatenates the promoted length-one coordinates.
    """
    obj = obj.copy(deep=False)
    obj.coords[dim] = key
    return obj


def sortby(da, dim="first", tolerance=None):
    """
    Sort a blocked virtual data array along *dim* by coordinate value, lazily.

    The data blocks (the tiles of a :class:`~xdas.virtual.TileArray`, the
    sources of a :class:`~xdas.virtual.VirtualStack`) are permuted into
    ascending start-value order without reading any of them: the permutation
    is a manifest (or source-list) gather, and the coordinate tie points are
    gathered blockwise the same way. Ties between equal start values keep
    their current order. The reordered coordinate is then simplified with
    *tolerance*, spending the accuracy budget once, on sorted segments —
    exactly as :func:`concat` does on time-ordered inputs.

    Parameters
    ----------
    da : DataArray
        The data array to sort. Its data must be a :class:`TileArray` or a
        :class:`VirtualStack` blocked along *dim*, and its *dim* coordinate
        an interpolated coordinate whose tie points align with the block
        boundaries (the state produced by concatenation without
        simplification, ``tolerance=False``).
    dim : str, optional
        The dimension to sort along. Default to "first".
    tolerance : float or timedelta64, optional
        The tolerance spent by the final coordinate simplification. If None
        (default), each coordinate spends its own declared tolerance. Pass
        ``False`` to skip simplification entirely.

    Returns
    -------
    DataArray
        The sorted data array, as lazy as its input.
    """
    from ..coordinates import InterpCoordinate

    axis = da.get_axis_num(dim)
    dim = da.dims[axis]
    coord = da.coords[dim]
    if not isinstance(coord, InterpCoordinate):
        raise NotImplementedError("can only sort along an interpolated coordinate")
    data = da.data
    if isinstance(data, TileArray):
        sizes = np.asarray(data.chunks[axis])
    elif isinstance(data, VirtualStack) and data.axis == axis:
        sizes = np.asarray([source.shape[axis] for source in data.sources])
    else:
        raise NotImplementedError(
            "can only sort a TileArray or a VirtualStack blocked along `dim`"
        )
    edges = np.concatenate(([0], np.cumsum(sizes)))
    tie_indices = coord.tie_indices
    tie_values = coord.tie_values
    order = np.argsort(coord._get_value(edges[:-1]), kind="stable")
    if np.array_equal(order, np.arange(len(order))):
        sorted_coord = coord
    else:
        # every block must begin and end on a tie point, so that the blockwise
        # gather is exact: the state concatenation without simplification
        # leaves; a simplified coordinate can only be verified, not permuted
        starts = np.searchsorted(tie_indices, edges[:-1])
        ends = np.searchsorted(tie_indices, edges[1:] - 1, side="right")
        if not (
            np.array_equal(tie_indices[starts], edges[:-1])
            and np.array_equal(tie_indices[ends - 1], edges[1:] - 1)
        ):
            raise NotImplementedError(
                "tie points do not align with the block boundaries; sort "
                "before simplifying (or concatenate with `tolerance=False`)"
            )
        if isinstance(data, TileArray):
            data = data._permute_tiles(order, axis)
        else:
            data = VirtualStack([data.sources[i] for i in order], axis)
        # blockwise tie-point gather, fully vectorized: for each block in
        # sorted order, its run of tie points, re-offset to its new position
        counts = (ends - starts)[order]
        offsets = np.cumsum(counts) - counts
        gather = np.arange(counts.sum()) - np.repeat(offsets, counts)
        gather += np.repeat(starts[order], counts)
        new_edges = np.cumsum(sizes[order]) - sizes[order]
        shift = np.repeat(new_edges - edges[:-1][order], counts)
        parts = {
            "tie_indices": tie_indices[gather] + shift,
            "tie_values": tie_values[gather],
        }
        if coord.sampling_interval is not None:
            parts["sampling_interval"] = coord.sampling_interval
            parts["tolerance"] = coord.tolerance
        sorted_coord = InterpCoordinate(parts, dim)
    sorted_coord = sorted_coord.simplify(tolerance)
    coords = da.coords.copy()
    coords[dim] = sorted_coord
    return DataArray(data, coords, da.dims, da.name, da.attrs)


def concat_coords(
    objs,
    *,
    sort=False,
    return_order=False,
    tolerance=None,
    reduce=True,
    regularize=False,
):
    """
    Concatenate coordinate objects.

    Parameters
    ----------
    objs : sequence
        Sequence of coordinate-like objects to concatenate.
    sort : bool, optional
        If True, sort `objs` by the start value before concatenation.
    return_order : bool, optional
        If True, return `(coord, order)` where `order` is the list of
        indices used to sort the input objects.
    tolerance : float or timedelta64, optional
        The tolerance to consider that the end of a coordinate object is continuous
        with beginning of the following, For time coordinates, numeric values are
        considered as seconds. By default the coordinate spends its own declared
        tolerance when it carries one, else a zero-like default. Pass ``False``
        to disable simplification entirely.
    reduce : bool, optional
        Whether to drop redundant tie points after concatenation. Default True.
    regularize : bool, optional
        Whether to promote the result to a regular coordinate when the merged
        segments admit a single shared rate within *tolerance*. Default False:
        regular inputs already stay regular through concatenation.

    Returns
    -------
    coord
        The concatenated coordinate object.
    order : list of int, optional
        The sort order for `objs` when `return_order` is True.

    """
    # sort
    order = list(range(len(objs)))
    if sort:
        order = sorted(order, key=lambda idx: objs[idx][0].values)
        objs = [objs[index] for index in order]
    out = objs[0]

    # concat
    for obj in objs[1:]:
        out = out._concat(obj)

    # simplify
    if tolerance is not False:
        if isinstance(out, AxisCoordinate):
            # `_concat` is strict: same-rate inputs stay regular, mismatched
            # rates drop to irregular. `simplify` then drops redundant tie
            # points (chunk seams within tolerance fuse away) and, with
            # `regularize=True`, recovers a single shared rate when the merged
            # segments admit one within *tolerance*.
            out = out.simplify(tolerance, reduce=reduce, regularize=regularize)
        elif tolerance is not None:
            raise TypeError(
                "`tolerance` can only be used with coordinates "
                "that implements `simplify`"
            )

    if return_order:
        return out, order

    return out


def split(da, indices_or_sections="discontinuities", dim="first", tolerance=None):
    """
    Split a data array along a dimension.

    Splitting can either be performed at each discontinuity , at a given set of indices
    (given as a list of int) or in order to get a given number of equal sized chunks
    (if a single int is provided).

    Parameters
    ----------
    da : DataArray
        The data array to split
    indices_or_sections : str, int or list of int, default="discontinuities"
        Describe how the splitting must be done:
        - If `indices_or_section` is an integer N, the array will be divided into N
        almost equal (can differ by one element if the `dim` size is not a multiple of
        N).
        - If `indices_or_section` is a 1-D array of sorted integers, the entries
        indicate where the array is split along `dim`. For example, `[2, 3]` would, for
        `dim="first"`, result in [da[:2], da[2:3], da[3:]].
        - If `indices_or_section` is one of "discontinuities", "gaps" or "overlaps",
        splitting will occurs at the indices given by `Coordinate.get_split_indices`.
    dim : str, optional
        The dimension along which to split, by default "first"
    tolerance : float or timedelta64, optional
        Passed to `Coordinate.get_split_indices` if `indices_or_section` is
        "discontinuities", "gaps" or "overlaps" to determine what can be considered as
        a discontiuity. For time coordinates, numeric values are considered as seconds.
        Zero tolerance by default.

    Returns
    -------
    list of DataArray
        The splitted data array.
    """
    if isinstance(indices_or_sections, str):
        indices_or_sections = da[dim].get_split_indices(indices_or_sections, tolerance)
    else:
        if tolerance:
            raise ValueError(
                "`tolerance` cannot be used when `indices_or_sections` "
                "is an integer or a list of indices"
            )

    if isinstance(indices_or_sections, int):
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
        div_points = np.concatenate([[0], indices_or_sections, [da.sizes[dim]]])

    return DataCollection(
        [da.isel({dim: slice(start, stop)}) for start, stop in pairwise(div_points)]
    )


def trim_overlaps(obj, keep="last", dim="first", tolerance=None):
    """
    Remove the overlapping samples of a data array, keeping one copy of each.

    An overlap is a place where the coordinate steps backwards: two segments
    describe the same span of time (or distance), typically because two files
    share a sample at their seam, or because an acquisition was restarted
    slightly before it stopped. This routine cuts the data array at its
    overlaps, drops the duplicated samples from all but one segment, and
    concatenates what is left back into a single data array.

    Trimming lands on a sample boundary, never between two: nothing is ever
    resampled, interpolated or filled. Sub-sample misalignment therefore
    survives as a discontinuity of the coordinate. Everything is done at the
    manifest level, so a lazy data array stays lazy and no data is read.

    Parameters
    ----------
    obj : DataArray or DataCollection
        The data to trim. A data collection is trimmed leaf by leaf, its tree
        preserved.
    keep : {"last", "first"}, optional
        Which copy of an overlapping span to keep. ``"last"`` (default) gives
        the later segment precedence, on the assumption that the following
        data carries the more correct time — this is ObsPy's
        ``Stream.merge(method=1, interpolation_samples=0)``. ``"first"`` is
        the mirror image.
    dim : str, optional
        The dimension along which to look for overlaps. Default to "first".
    tolerance : float or timedelta64, optional
        The magnitude below which a backward step is not considered an
        overlap. For time coordinates, numeric values are considered as
        seconds. By default only exactly zero-magnitude steps are ignored.
        Note that jitter is usually better handled upstream, by spending a
        tolerance when combining or by `da[dim] = da[dim].simplify(tolerance)`.

    Returns
    -------
    DataArray or DataCollection
        The data with its overlaps resolved, of the same type as *obj*.

    See Also
    --------
    split : Cut a data array at its overlaps, keeping every copy (1 to N).

    Examples
    --------
    >>> import numpy as np
    >>> import xdas as xd

    Two segments of five samples overlapping by two:

    >>> coord = {"tie_indices": [0, 4, 5, 9], "tie_values": [0.0, 4.0, 3.0, 7.0]}
    >>> da = xd.DataArray(np.arange(10.0), {"time": coord})
    >>> xd.trim_overlaps(da).values
    array([0., 1., 2., 5., 6., 7., 8., 9.])
    >>> xd.trim_overlaps(da, keep="first").values
    array([0., 1., 2., 3., 4., 7., 8., 9.])

    """
    if keep not in ("last", "first"):
        raise ValueError(f"`keep` must be either 'last' or 'first', got {keep!r}")
    if isinstance(obj, DataCollection):
        return obj.map(lambda da: trim_overlaps(da, keep, dim, tolerance))
    axis = obj.get_axis_num(dim)
    dim = obj.dims[axis]
    parts = split(obj, "overlaps", dim, tolerance)
    if len(parts) == 1:
        return obj

    # The parts are walked in order of decreasing precedence, each keeping only
    # what no higher-precedence part already claimed. Claims accumulate as a set
    # of spans rather than a single watermark, because a part may be *enveloped*
    # in a lower-precedence one — the covering part then keeps a run on each
    # side of it, and a watermark, which can only trim an end, would drop the
    # far side along with the overlap. This also resolves a part wholly covered
    # by a neighbour (it contributes nothing, while the part beyond it is still
    # compared against that neighbour) and chains of mutually overlapping parts.
    kept = []
    claimed = []
    for part in reversed(parts) if keep == "last" else parts:
        coord = part[dim]
        for start, stop in _uncovered(coord, claimed):
            kept.append(part.isel({dim: slice(start, stop)}))
        claimed = _claim(claimed, coord[0].values, coord[-1].values)
    return concat(kept, dim, tolerance)


def _uncovered(coord, claimed):
    """Index ranges of *coord* that no span of *claimed* covers.

    *claimed* is a list of disjoint ``(first, last)`` value pairs in ascending
    order, both bounds inclusive. Bounds are resolved through the coordinate's
    own label look-up, so nothing is materialised.
    """
    runs = []
    cursor = 0
    for first, last in claimed:
        # `to_index` clamps: a bound past either end of the coordinate resolves
        # to the full length or to zero rather than raising
        stop = coord.to_index(slice(None, first), endpoint=False).stop
        if stop > cursor:
            runs.append((cursor, stop))
        cursor = max(cursor, coord.to_index(slice(None, last)).stop)
    if cursor < len(coord):
        runs.append((cursor, len(coord)))
    return runs


def _claim(claimed, first, last):
    """Add the span ``(first, last)`` to the disjoint, ascending list *claimed*."""
    out = []
    for start, stop in claimed:
        if stop < first or start > last:
            out.append((start, stop))
        else:
            first, last = min(first, start), max(last, stop)
    out.append((first, last))
    out.sort()
    return out


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
            if not isinstance(coord, AxisCoordinate):
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


def _ensure_str_paths(paths):
    if isinstance(paths, Path):
        paths = str(paths)
    if isinstance(paths, list):
        paths = [str(path) if isinstance(path, Path) else path for path in paths]
    return paths
