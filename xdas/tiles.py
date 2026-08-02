"""
Lazy tile-backed virtual arrays over file archives.

A :class:`TileArray` is a numpy-like duck array whose values live in a
dense rectilinear grid of file-backed *tiles*, described by a plain
:class:`xarray.Dataset` (used as a container only) with one dimension
``tile_k`` per data axis:

- 1-D *geometry* variables place the grid along each axis ``k``:
  ``sizes_k`` (samples contributed by each tile, required), ``starts_k``
  (origin inside the decoded source, default 0) and ``steps_k`` (source
  stride, the per-tile decimation, default 1), reading as
  ``virtual[pos : pos + size] = source[start : start + size * step : step]``
  with ``pos`` the running sum of the previous sizes along the axis;
- N-D *parameter* variables: ``paths`` (the source file of each tile,
  required) plus any format-private per-tile values forwarded to the
  engine as keyword arguments. A parameter carries only the tile
  dimensions along which it actually varies — a constant folds to a 0-d
  variable — and broadcasts over the grid at read time;
- an optional 0-d ``root`` variable: the common directory of the
  sources, split off so the per-tile ``paths`` stay root-relative —
  one shared constant instead of a per-tile repeat. Absent (or empty),
  the paths are used as stored.

What arrays cannot carry — the ``dtype`` and the ``engine``
specification — lives on the array itself, by value, and travels
beside the dataset when the array is persisted (see
:func:`xdas.io.xdas.save_dataarray`). The dtype is not a decode
target: engines decode into the element type of their sources, and
the array records that type at scan time so laziness holds — a
virtual array answers ``dtype`` without touching its sources — and
verifies it against every decoded tile. Casting is an explicit
extra step (:meth:`TileArray.astype`), outside the tiles machinery.
Every dataset attribute is a user attribute.

Geometry loads eagerly at construction (tiny); parameters stay folded —
a constant occupies one element whatever the grid size — and broadcast
over the grid only as tiles are read. Positive-step slicing folds into
the geometry and returns a new :class:`TileArray` (as lazy and
self-described as its input); any other indexing reads the bounding box
of the selection and resolves the rest in memory. ``np.asarray``
materializes: tiles are read one by one by the registered *engine*
(``xdas.io.Engine[name]``), whose ``load_tile`` opens each tile's path
itself and returns the tile's *source selection* (one possibly strided
slice per axis), every part landing directly in the output array.

A tile array is used *raw* as the data of a :class:`xdas.DataArray`
(``DataArray(arr, coords)``), so ``da.data`` returns the inspectable
lazy object. The tiling is the only blocking the array knows:
:attr:`~TileArray.chunks` reports it, and whole-array reductions
stream one tile row at a time.

:meth:`TileArray.concat` fuses arrays along any axis by concatenating
the geometry and the per-tile parameters (O(tiles), the data is never
read). Tile arrays persist inside the native xdas netCDF format: the
wrapped dataset *is* the stored form.

Ported from the 0.3 line (``xdas/virtual/tilearray.py``); the lazy
:meth:`TileArray.expand_dims` is a 0.2 extension supporting the legacy
concat-along-a-new-dimension path.
"""

from __future__ import annotations

import functools
import inspect
import itertools
import json
import math
import os

import numpy as np
import xarray as xr

TILE_PREFIX = "tile_"
"""Prefix of the tile-grid dimensions of a manifest dataset."""

_UNITS = ("B", "kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
"""Decimal byte units, as xarray spells them in its ``Size:`` header."""


class _Unfoldable(Exception):
    """A key that no tile grid can express (private to :meth:`TileArray._fold`).

    Raised for non-foldable entries and for empty selections (a grid
    needs at least one tile). Its own type so a genuine failure inside
    the fold is never mistaken for "fall back to a bounded read".
    """


# reductions that stream tile row by tile row instead of materializing:
# numpy function -> (per-block reduce, pairwise combine, mean-style
# finalizer: None, "count" or "nancount")
_STREAMING_REDUCTIONS = {
    np.sum: (np.sum, np.add, None),
    np.nansum: (np.nansum, np.add, None),
    np.max: (np.max, np.maximum, None),
    np.nanmax: (np.nanmax, np.fmax, None),
    np.min: (np.min, np.minimum, None),
    np.nanmin: (np.nanmin, np.fmin, None),
    np.all: (np.all, np.logical_and, None),
    np.any: (np.any, np.logical_or, None),
    np.mean: (np.sum, np.add, "count"),
    np.nanmean: (np.nansum, np.add, "nancount"),
}


def _fold_param(values, counts, dims):
    """Broadcast *values* over the grid and fold its constant axes away.

    Returns ``(dims, values)`` where the kept dimensions are exactly
    those along which the values actually vary — the manifest-level
    constant folding, expressed as xarray dimensions.
    """
    values = values.reshape(values.shape + (1,) * (len(counts) - values.ndim))
    values = np.broadcast_to(values, counts)
    keep = [
        axis
        for axis in range(values.ndim)
        if values.shape[axis] > 1
        and not bool((values == values.take([0], axis=axis)).all())
    ]
    index = tuple(slice(None) if axis in keep else 0 for axis in range(values.ndim))
    # re-wrap: plain indexing of a fully-reduced object array yields the
    # bare element, which numpy would re-box as a fixed-width string
    return tuple(dims[axis] for axis in keep), np.asarray(
        values[index], dtype=values.dtype
    )


def _split_root(paths):
    """Split absolute *paths* into their common directory and relative rest.

    The root is the deepest directory containing every path (dirnames
    only, so at least a basename always remains in the relative part).
    Falls back to ``("", paths)`` when no common directory exists
    (paths spread over several drives).
    """
    try:
        root = os.path.commonpath([os.path.dirname(path) for path in paths.flat])
    except ValueError:
        return "", paths
    # commonpath ends on a component boundary: a plain strip is exact
    strip = np.frompyfunc(lambda path: path[len(root) :].lstrip(os.sep), 1, 1)
    return root, strip(paths)


def _common_root(roots):
    """Return the deepest directory containing every *root* ("" when none does)."""
    if any(not root for root in roots):
        return ""
    try:
        return os.path.commonpath(roots)
    except ValueError:
        return ""


def _normalize_key(key, ndim):
    """Return *key* as a full-length tuple with ``Ellipsis`` expanded.

    Accepts the plain keys produced by xarray's indexing adapters and
    by dask-style block slicing, plus a defensive unwrap of explicit
    indexer objects carrying a ``tuple`` attribute.
    """
    key = getattr(key, "tuple", key)
    if not isinstance(key, tuple):
        key = (key,)
    if any(entry is Ellipsis for entry in key):
        index = key.index(Ellipsis)
        fill = (slice(None),) * (ndim - len(key) + 1)
        key = key[:index] + fill + key[index + 1 :]
    if len(key) > ndim:
        raise IndexError(f"too many indices: got {len(key)} for {ndim} axes")
    return key + (slice(None),) * (ndim - len(key))


def _bounding_key(key, shape):
    """Split *key* into a positive-step bounding box and a residual key.

    The box is one positive-step slice per axis covering every selected
    index; the residual, applied to the values of the box, produces the
    exact selection. Returns ``(box, residual, empty)`` where *empty*
    flags a selection with no elements on some axis (the caller can
    then skip reading entirely). Raises :class:`NotImplementedError`
    for entries that have no bounded reduction (new axes or boolean
    masks of more than one dimension).
    """
    box, residual = [], []
    empty = False
    for entry, extent in zip(key, shape):
        if isinstance(entry, slice):
            start, stop, step = entry.indices(extent)
            size = len(range(start, stop, step))
            if size == 0:
                box.append(slice(0, 0))
                residual.append(slice(None))
                empty = True
            elif step > 0:
                box.append(slice(start, stop, step))
                residual.append(slice(None))
            else:
                last = start + (size - 1) * step
                box.append(slice(last, start + 1, -step))
                residual.append(slice(None, None, -1))
        elif isinstance(entry, (int, np.integer)):
            index = int(entry)
            if index < 0:
                index += extent
            if not 0 <= index < extent:
                raise IndexError(
                    f"index {entry} is out of bounds for axis of size {extent}"
                )
            box.append(slice(index, index + 1))
            residual.append(0)
        elif isinstance(entry, (list, np.ndarray)):
            indices = np.asarray(entry)
            if indices.dtype == bool:
                if indices.ndim != 1:
                    raise NotImplementedError("multi-dimensional boolean mask")
                (indices,) = np.nonzero(indices)
            if not np.issubdtype(indices.dtype, np.integer):
                raise IndexError(f"invalid index array dtype: {indices.dtype}")
            if indices.size == 0:
                box.append(slice(0, 0))
                residual.append(indices)
                empty = True
                continue
            indices = np.where(indices < 0, indices + extent, indices)
            if indices.min() < 0 or indices.max() >= extent:
                raise IndexError(f"index out of bounds for axis of size {extent}")
            low = int(indices.min())
            box.append(slice(low, int(indices.max()) + 1))
            residual.append(indices - low)
        else:
            raise NotImplementedError(f"unsupported index entry: {entry!r}")
    return tuple(box), tuple(residual), empty


def _materialize(value):
    """Read any :class:`TileArray` in *value*, descending one level."""
    if isinstance(value, TileArray):
        return np.asarray(value)
    if isinstance(value, (list, tuple)) and any(
        isinstance(item, TileArray) for item in value
    ):
        return type(value)(
            np.asarray(item) if isinstance(item, TileArray) else item for item in value
        )
    return value


def _to_si(nbytes):
    """Render a byte count the way xarray renders its ``Size:`` header.

    Decimal units, no decimals: the repr sits right under that header,
    so a base-1024 count would read as a different number.
    """
    dividend = float(nbytes)
    index = 0
    while dividend >= 1000.0 and index < len(_UNITS) - 1:
        dividend /= 1000.0
        index += 1
    return f"{dividend:.0f}{_UNITS[index]}"


def _row_ranges(edges, shape):
    """Return the streaming blocks: one tile row along axis 0, whole elsewhere.

    The tiling is the only blocking a tile array has, and a whole row
    bounds the memory a streaming pass holds at once.
    """
    rows = [slice(int(lo), int(hi)) for lo, hi in itertools.pairwise(edges)]
    return [rows] + [[slice(0, extent)] for extent in shape[1:]]


class TileArray(np.lib.mixins.NDArrayOperatorsMixin):
    """A dense rectilinear grid of file-backed tiles as one virtual array.

    Numpy-like duck array over the *manifest dataset* described in the
    module docstring: construction loads the 1-D geometry (tile sizes
    and the optional per-tile source origins and strides) and validates
    it; the N-D per-tile parameters are left untouched until a read.
    The array is immutable by convention: every tiling-changing
    operation returns a new instance over a new dataset, and the first
    full read is cached.

    The tiled box is *anonymous*: a tile array carries no dimension
    names and no variable name (the ``tile_k`` dimensions are internal).
    Those are labeled-array identity, supplied when a
    :class:`xdas.DataArray` is emitted around it.

    The constructor wraps a manifest dataset directly — hand-built, or
    reopened from a native xdas file; :meth:`from_tiles` builds the
    manifest from per-tile descriptions at scan time.

    Parameters
    ----------
    dataset : xarray.Dataset
        The manifest dataset: the 1-D geometry and the N-D per-tile
        parameter variables described in the module docstring. Every
        dataset attribute is a user attribute.
    dtype : str or numpy.dtype
        Element type of the sources as the engine decodes them
        (little-endian or single-byte), recorded so the lazy array can
        report it without reading. Verified against every decoded
        tile, never used to cast.
    engine : str or dict
        The engine specification, stored by value with the array: the
        key ``"name"`` selects a registered engine
        (``xdas.io.Engine[name]``); the remaining keys are passed to
        its ``load_tile`` as keyword parameters. A plain string is
        shorthand for ``{"name": engine}``. Never an
        :class:`~xdas.io.Engine` instance: the specification must
        reproduce the decode with no instance alive, so open-time
        settings (``vtype``, ``ctype``) do not belong in it.
    """

    def __init__(self, dataset, dtype, engine):
        self.dataset = dataset
        self._cache = None
        if isinstance(engine, str):
            engine = {"name": engine}
        # the json round trip deep-copies and normalizes (tuples become
        # lists), so equality survives a store round trip
        engine = json.loads(json.dumps(engine))
        if not isinstance(engine, dict) or "name" not in engine:
            raise ValueError("the engine specification must have a `name` key")
        # imported here: xdas.io imports this module at package init
        from .io.core import Engine

        Engine[engine["name"]]  # fail fast on unregistered engines
        self._engine = engine
        self.dtype = np.dtype(dtype)
        if self.dtype.byteorder == ">":
            raise ValueError("only little-endian or single-byte dtypes are supported")
        ndim = 0
        while f"sizes_{ndim}" in dataset:
            ndim += 1
        if ndim == 0:
            raise ValueError("a tile array needs a `sizes_0` geometry variable")
        self.ndim = ndim
        self.dims = dims = tuple(f"{TILE_PREFIX}{k}" for k in range(ndim))
        self._sizes = self._geometry("sizes", None)
        self._starts = self._geometry("starts", 0)
        self._steps = self._geometry("steps", 1)
        for kind, arrays, bound in (
            ("sizes", self._sizes, 1),
            ("starts", self._starts, 0),
            ("steps", self._steps, 1),
        ):
            for k, values in enumerate(arrays):
                if np.any(values < bound):
                    kind_bound = "non-negative" if bound == 0 else "strictly positive"
                    raise ValueError(f"`{kind}_{k}` must be {kind_bound}")
        self._edges = tuple(
            np.concatenate(([0], np.cumsum(sizes))) for sizes in self._sizes
        )
        self.shape = tuple(int(edges[-1]) for edges in self._edges)
        if "paths" not in dataset:
            raise ValueError("a tile array needs a `paths` variable")
        if "root" in dataset:
            if tuple(dataset["root"].dims) != ():
                raise ValueError("`root` must be a 0-d variable")
            self.root = str(dataset["root"].values[()])
        else:
            self.root = ""
        geometry = {
            f"{kind}_{k}" for kind in ("sizes", "starts", "steps") for k in range(ndim)
        }
        self._params = tuple(
            sorted(
                name
                for name in map(str, dataset.data_vars)
                if name not in geometry and name not in ("paths", "root")
            )
        )
        for name in ("paths", *self._params):
            vdims = tuple(map(str, dataset[name].dims))
            if tuple(dim for dim in dims if dim in vdims) != vdims:
                raise ValueError(
                    f"`{name}` dimensions must be an ordered subset of {dims}"
                )

    @classmethod
    def from_tiles(cls, paths, sizes, dtype, engine, *, attrs=None, **params):
        """Build a tile array from per-tile descriptions of fresh sources.

        The scan-time encoder: every tile is read from the origin of
        its decoded source, without decimation. Trimmed (``starts_k``)
        and decimated (``steps_k``) geometry is view state — it arises
        by slicing, or comes from a stored manifest through the class
        constructor.

        Parameters
        ----------
        paths : str or array-like
            Source file of each tile. A scalar describes a
            one-tile-per-axis grid; an array is padded with trailing
            length-1 axes up to the rank. A path may appear in several
            tiles. Relative paths are made absolute at construction —
            the working directory cannot be trusted later, as reads are
            lazy and stored views outlive the session. The common
            directory of the absolute paths is then split off into the
            0-d ``root`` variable, the stored per-tile paths staying
            root-relative.
        sizes : sequence of int or 1-D array-like
            One entry per axis (this defines the rank): the samples each
            tile contributes along that axis. An int is uniform across
            the axis' tiles; an array gives the per-tile extents (its
            length is the number of tiles along the axis).
        dtype : str or numpy.dtype
            Element type of the sources as the engine decodes them
            (little-endian or single-byte) — recorded, not a cast
            target; the scanner reads it off the file it describes.
        engine : str or dict
            The engine specification, stored by value with the array:
            the key ``"name"`` selects a registered engine
            (``xdas.io.Engine[name]``); the remaining keys are passed
            to its ``load_tile`` as keyword parameters. A plain string
            is shorthand for ``{"name": engine}``.
        attrs : dict, optional
            User attributes of the virtual array.
        **params : array-like
            Per-tile engine parameters, broadcast over the grid: each
            read passes the tile's value to the engine as a keyword
            argument (shadowing a same-named specification constant).

        Returns
        -------
        TileArray
        """
        ndim = len(sizes)
        if ndim == 0:
            raise ValueError("a tile array needs at least one axis")
        dims = tuple(f"{TILE_PREFIX}{k}" for k in range(ndim))
        paths = np.asarray(paths, dtype=object)
        if paths.ndim > ndim:
            raise ValueError("`paths` has more axes than `sizes` entries")
        paths = paths.reshape(paths.shape + (1,) * (ndim - paths.ndim))
        # reads are lazy and stored views outlive the session: anchor the
        # paths now, while the scan's working directory still applies
        paths = np.frompyfunc(os.path.abspath, 1, 1)(paths)
        # the common directory is one shared constant, not a per-tile
        # repeat: split it off, the stored paths stay root-relative
        root, paths = _split_root(paths)
        data = {}
        counts = []
        for k, entry in enumerate(sizes):
            values = np.atleast_1d(np.asarray(entry, dtype=np.int64))
            if values.size == 1 and paths.shape[k] > 1:
                values = np.full(paths.shape[k], values[0], dtype=np.int64)
            counts.append(len(values))
            data[f"sizes_{k}"] = (dims[k], values)
        counts = tuple(counts)
        if any(have not in (1, count) for have, count in zip(paths.shape, counts)):
            raise ValueError(
                f"`paths` shape {paths.shape} does not match the grid {counts}"
            )
        data["paths"] = _fold_param(paths, counts, dims)
        if root:
            data["root"] = ((), np.asarray(root, dtype=object))
        reserved = (
            set(data)
            | {"root"}
            | {f"{kind}_{k}" for kind in ("starts", "steps") for k in range(ndim)}
        )
        for name, values in params.items():
            if name in reserved:
                raise ValueError(f"parameter name {name!r} is reserved")
            data[name] = _fold_param(np.asarray(values), counts, dims)
        dataset = xr.Dataset(data, attrs=dict(attrs or {}))
        return cls(dataset, dtype, engine)

    def to_dataset(self):
        """Encode this tile array as its manifest dataset.

        The stored form — a copy of the wrapped dataset carrying the
        user attributes. What the dataset cannot hold, the by-value
        ``dtype`` and ``engine``, the caller reads off the properties
        and stores beside it. Source paths are stored exactly as the
        array holds them: relative to the 0-d ``root`` variable
        carrying their common directory.

        Returns
        -------
        xarray.Dataset
            The manifest dataset.
        """
        return xr.Dataset(self.dataset.data_vars, attrs=self.attrs)

    def _geometry(self, kind, default):
        """Load the eager 1-D ``{kind}_k`` arrays (*default* where absent)."""
        arrays = []
        for k, dim in enumerate(self.dims):
            name = f"{kind}_{k}"
            if name in self.dataset:
                if tuple(self.dataset[name].dims) != (dim,):
                    raise ValueError(f"`{name}` must have dimensions ({dim!r},)")
                arrays.append(np.asarray(self.dataset[name].values, dtype=np.int64))
            else:
                count = int(self.dataset.sizes[dim])
                arrays.append(np.full(count, default, dtype=np.int64))
        return tuple(arrays)

    @property
    def engine(self):
        """dict: the engine specification (``"name"`` plus its parameters)."""
        return self._engine

    @property
    def attrs(self):
        """dict: the user attributes of the virtual array."""
        return dict(self.dataset.attrs)

    @property
    def chunks(self):
        """Tuple of tuple of int: the tiling, as per-axis tile extents.

        Not a hint — the tiling *is* the only blocking the array has.
        """
        return tuple(tuple(int(size) for size in sizes) for sizes in self._sizes)

    @property
    def ntiles(self):
        """int: total number of tiles in the grid."""
        return math.prod(len(sizes) for sizes in self._sizes)

    @property
    def size(self):
        """int: total number of elements."""
        return math.prod(self.shape)

    def __getitem__(self, key):
        """Index the array, staying virtual whenever possible.

        Positive-step slices fold into the geometry and return a new
        :class:`TileArray` without touching the sources:
        ``np.asarray(arr[key])`` equals ``np.asarray(arr)[key]``. Per
        axis, the overlapping tiles are located by binary search on the
        running tile sizes and their geometry is trimmed — and, for
        stepped slices, decimated — to the selection (steps multiply,
        origins compose, one tile stays one tile); tiles the selection
        strides over entirely are dropped. The parameters are sliced
        through the wrapped dataset, so a lazy array stays lazy.

        Every other key (integers, index arrays, boolean masks,
        reversed slices, empty selections) reads the bounding box of
        the selection and applies the remainder in memory, returning a
        numpy array.
        """
        key = _normalize_key(key, self.ndim)
        try:
            return self._fold(key)
        except _Unfoldable:
            pass
        try:
            box, residual, empty = _bounding_key(key, self.shape)
        except NotImplementedError:
            return np.asarray(self)[key]
        if empty:
            # zero-strided: the result is empty, so no value is ever read
            # and the full shape is never allocated
            return np.broadcast_to(np.zeros((), self.dtype), self.shape)[key].copy()
        return np.asarray(self._fold(box))[residual]

    def _fold(self, key):
        """Fold a full-length tuple of positive-step slices into a new array.

        Raises :class:`_Unfoldable` for non-foldable entries and for
        empty selections (a grid needs at least one tile);
        :meth:`__getitem__` then falls back to a bounded read.
        """
        indexers = {}
        assign = {}
        for axis, (entry, extent) in enumerate(zip(key, self.shape)):
            if not isinstance(entry, slice) or (entry.step or 1) < 1:
                raise _Unfoldable(
                    "only positive-step slices can be folded into the tile grid"
                )
            lo, hi, s = entry.indices(extent)
            if len(range(lo, hi, s)) == 0:
                raise _Unfoldable(f"empty selection along axis {axis}")
            if (lo, hi, s) == (0, extent, 1):
                continue
            edges = self._edges[axis]
            i0 = int(np.searchsorted(edges, lo, "right")) - 1
            i1 = int(np.searchsorted(edges, hi, "left"))
            pos = edges[i0:i1]
            size = self._sizes[axis][i0:i1]
            start = self._starts[axis][i0:i1]
            step = self._steps[axis][i0:i1]
            # selected positions are lo, lo + s, ...; j0/j1 index the first
            # and last of them falling inside each tile
            j0 = np.maximum(0, -((lo - pos) // s))
            j1 = (np.minimum(pos + size, hi) - 1 - lo) // s
            keep = j1 >= j0
            dim = self.dims[axis]
            indexers[dim] = slice(i0, i1) if keep.all() else i0 + np.flatnonzero(keep)
            assign[f"sizes_{axis}"] = (dim, (j1 - j0 + 1)[keep])
            assign[f"starts_{axis}"] = (dim, (start + (lo + j0 * s - pos) * step)[keep])
            assign[f"steps_{axis}"] = (dim, (step * s)[keep])
        # all-default starts/steps columns fold away (they stay derivable)
        drop = [
            name
            for name, (_, values) in assign.items()
            if (name.startswith("starts_") and not values.any())
            or (name.startswith("steps_") and not (values != 1).any())
        ]
        assign = {name: entry for name, entry in assign.items() if name not in drop}
        dataset = self.dataset.isel(indexers).assign(assign)
        dataset = dataset.drop_vars([name for name in drop if name in dataset])
        return type(self)(dataset, self.dtype, self.engine)

    @classmethod
    def concat(cls, arrays, dim=0):
        """Concatenate tile arrays along axis *dim* into a new array.

        Requires equal engines and dtype, and equal geometry on every
        *other* axis — nothing else: differently trimmed or decimated
        subviews of the same sources concatenate, and repeated sources
        are legitimate. The geometry is chained; parameters stay folded
        when every input agrees and are broadcast out and concatenated
        otherwise (the tile tables load, the data does not). Arrays
        rooted in different directories fuse under the deepest
        directory containing every root, their per-tile paths rebased
        (absolute when no common directory exists).

        Parameters
        ----------
        arrays : list of TileArray
            The tile arrays to concatenate, in order along *dim*.
        dim : int, optional
            The axis along which to concatenate. Default 0. (Dimension
            *names* are mapped to axes by the callers — the tile array
            mirrors the positional numpy API.)

        Returns
        -------
        TileArray
        """
        first = arrays[0]
        ndim = first.ndim
        axis = int(dim)
        if not 0 <= axis < ndim:
            raise ValueError(f"no axis {dim} in a {ndim}-dimensional tile array")
        dims = first.dims
        for other in arrays[1:]:
            if (
                other.ndim != ndim
                or other.dtype != first.dtype
                or other.engine != first.engine
                or other._params != first._params
                or any(
                    k != axis
                    and not (
                        np.array_equal(other._sizes[k], first._sizes[k])
                        and np.array_equal(other._starts[k], first._starts[k])
                        and np.array_equal(other._steps[k], first._steps[k])
                    )
                    for k in range(ndim)
                )
            ):
                raise ValueError("can only concatenate compatible tile arrays")
        data = {}
        for kind, per_axis, default in (
            ("sizes", [array._sizes for array in arrays], None),
            ("starts", [array._starts for array in arrays], 0),
            ("steps", [array._steps for array in arrays], 1),
        ):
            for k in range(ndim):
                if k == axis:
                    values = np.concatenate([entries[k] for entries in per_axis])
                else:
                    values = per_axis[0][k]
                if default is not None and bool((values == default).all()):
                    continue
                data[f"{kind}_{k}"] = (dims[k], values)
        root = _common_root([array.root for array in arrays])
        if root:
            data["root"] = ((), np.asarray(root, dtype=object))
        for name in ("paths", *first._params):
            if name == "paths":
                variables = [array._rebased_paths(root) for array in arrays]
            else:
                variables = [array.dataset[name].variable for array in arrays]
            vdims = variables[0].dims
            if dims[axis] not in vdims and all(v.dims == vdims for v in variables):
                values = variables[0].values
                if all(np.array_equal(v.values, values) for v in variables[1:]):
                    data[name] = (vdims, values)
                    continue
            union = tuple(
                d
                for d in dims
                if d == dims[axis] or any(d in v.dims for v in variables)
            )
            parts = [
                _expand(variable, union, array)
                for variable, array in zip(variables, arrays)
            ]
            axis_pos = union.index(dims[axis])
            values = np.concatenate([part.values for part in parts], axis=axis_pos)
            data[name] = xr.Variable(union, values)
        dataset = xr.Dataset(data, attrs=first.attrs)
        return cls(dataset, first.dtype, first.engine)

    def _full_paths(self):
        """Return the full source path of every tile, root joined, over the grid."""
        paths = self._grid_values("paths")
        if not self.root:
            return paths
        return np.frompyfunc(lambda path: os.path.join(self.root, path), 1, 1)(paths)

    def _rebased_paths(self, root):
        """Return the ``paths`` variable of this array, rebased on directory *root*.

        Folded dimensions are preserved: the rebase rewrites the stored
        values under the new root, it never broadcasts.
        """
        variable = self.dataset["paths"].variable
        if root == self.root:
            return variable

        def rebase(path):
            full = os.path.join(self.root, path)
            return os.path.relpath(full, root) if root else full

        values = np.frompyfunc(rebase, 1, 1)(np.asarray(variable.values, dtype=object))
        return xr.Variable(variable.dims, values)

    def _grid_values(self, name):
        """Load parameter *name* and broadcast it over the full tile grid."""
        variable = self.dataset[name].variable
        values = np.asarray(variable.values)
        counts = tuple(len(sizes) for sizes in self._sizes)
        shape = tuple(
            count if dim in variable.dims else 1
            for dim, count in zip(self.dims, counts)
        )
        return np.broadcast_to(values.reshape(shape), counts)

    @functools.cached_property
    def _engine_impl(self):
        """The ``(load_tile, spec)`` of the engine specification."""
        from .io.core import Engine

        spec = dict(self.engine)
        name = spec.pop("name")
        return Engine[name].load_tile, spec

    def __array__(self, dtype=None, copy=None):
        """Read every tile and return the values as a numpy array.

        The value-materialization primitive: tiles are read one by one,
        each exactly once, its part landing directly in the output
        array. Read a subset by slicing first: ``np.asarray(arr[key])``.
        The first full read is cached.
        """
        if self._cache is None:
            self._cache = self._read()
        values = self._cache
        if dtype is not None and np.dtype(dtype) != values.dtype:
            return values.astype(dtype)
        if copy:
            return values.copy()
        return values

    def _read(self):
        """Read every tile into a fresh output array, one engine call each."""
        out = np.empty(self.shape, dtype=self.dtype)
        read, spec = self._engine_impl
        counts = tuple(len(sizes) for sizes in self._sizes)
        paths = self._grid_values("paths")
        params = {name: self._grid_values(name) for name in self._params}
        for index in np.ndindex(counts):
            selection, dest = [], []
            for k, i in enumerate(index):
                first = int(self._starts[k][i])
                size = int(self._sizes[k][i])
                step = int(self._steps[k][i])
                selection.append(slice(first, first + (size - 1) * step + 1, step))
                dest.append(slice(int(self._edges[k][i]), int(self._edges[k][i + 1])))
            selection, dest = tuple(selection), tuple(dest)
            kwargs = dict(spec)
            for name, values in params.items():
                value = values[index]
                kwargs[name] = value.item() if isinstance(value, np.generic) else value
            path = os.path.join(self.root, str(paths[index]))
            part = np.asarray(read(path, selection, **kwargs))
            widths = tuple(entry.stop - entry.start for entry in dest)
            if part.shape != widths or part.dtype != self.dtype:
                raise ValueError(
                    f"engine {self.engine['name']!r} produced a {part.dtype} "
                    f"part of shape {part.shape} where the array records "
                    f"{self.dtype} parts and the selection has shape {widths}"
                )
            out[dest] = part
        return out

    def equals(self, other):
        """Whether *other* describes the same tiling (not elementwise).

        Compares the engine, dtype, geometry, parameters and user
        attributes; ``==`` stays elementwise, as on any numpy-like
        array. Paths compare joined: two arrays naming the same source
        files are equal however each splits its ``root``.
        """
        if not isinstance(other, TileArray):
            return False
        if (
            self.engine != other.engine
            or self.dtype != other.dtype
            or self.shape != other.shape
            or self.attrs != other.attrs
            or self._params != other._params
        ):
            return False
        for k in range(self.ndim):
            if not (
                np.array_equal(self._sizes[k], other._sizes[k])
                and np.array_equal(self._starts[k], other._starts[k])
                and np.array_equal(self._steps[k], other._steps[k])
            ):
                return False
        for name in self._params:
            mine, theirs = self._grid_values(name), other._grid_values(name)
            if not np.array_equal(mine, theirs):
                return False
        if self.root == other.root:
            mine, theirs = self._grid_values("paths"), other._grid_values("paths")
        else:
            mine, theirs = self._full_paths(), other._full_paths()
        return bool(np.array_equal(mine, theirs))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Materialize tile-backed inputs and apply the ufunc."""
        if any(isinstance(value, TileArray) for value in kwargs.get("out", ())):
            return NotImplemented
        inputs = tuple(
            np.asarray(value) if isinstance(value, TileArray) else value
            for value in inputs
        )
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        """Dispatch numpy functions, keeping select operations lazy.

        ``numpy.concatenate`` of compatible tile-backed arrays fuses
        the tilings and stays virtual; the streaming reductions
        (``sum``, ``mean``, ``min``, ``max``, their nan variants,
        ``any`` and ``all``) accumulate one tile row at a time with
        bounded memory. Everything else materializes and delegates to
        numpy.
        """
        if func is np.result_type:
            args = tuple(
                value.dtype if isinstance(value, TileArray) else value for value in args
            )
            return np.result_type(*args)
        if func is np.concatenate:
            result = _concatenate_virtual(args, kwargs)
            if result is not NotImplemented:
                return result
        if func is np.expand_dims:
            result = self._expand_virtual(args, kwargs)
            if result is not NotImplemented:
                return result
        if func in _STREAMING_REDUCTIONS:
            result = self._reduce_streaming(func, args, kwargs)
            if result is not NotImplemented:
                return result
        args = tuple(_materialize(value) for value in args)
        kwargs = {name: _materialize(value) for name, value in kwargs.items()}
        return func(*args, **kwargs)

    def _reduce_streaming(self, func, args, kwargs):
        """Accumulate a reduction one tile row at a time, bounded memory.

        Arguments are rebound against the reduction's own signature: the
        DataArray wrapper calls with every parameter bound positionally
        (no-value sentinels included), where plain numpy calls pass
        keywords.
        """
        try:
            bound = inspect.signature(func).bind(*args, **kwargs)
        except TypeError:
            return NotImplemented
        arguments = {
            name: value
            for name, value in bound.arguments.items()
            if value is not np._NoValue
        }
        if arguments.pop("a", None) is not self:
            return NotImplemented
        axis = arguments.pop("axis", None)
        keepdims = arguments.pop("keepdims", False)
        dtype = arguments.pop("dtype", None)
        if arguments.pop("out", None) is not None:
            return NotImplemented
        if any(value is not None for value in arguments.values()):
            return NotImplemented
        block_reduce, combine, counting = _STREAMING_REDUCTIONS[func]
        if axis is None:
            axes = tuple(range(self.ndim))
        else:
            axis = axis if isinstance(axis, tuple) else (axis,)
            axes = tuple(a + self.ndim if a < 0 else a for a in axis)
        kept = tuple(a for a in range(self.ndim) if a not in axes)
        out_shape = tuple(self.shape[a] for a in kept)
        acc = None
        filled = np.zeros(out_shape, dtype=bool)
        counts = np.zeros(out_shape) if counting else None
        for box in itertools.product(*_row_ranges(self._edges[0], self.shape)):
            block = np.asarray(self[box])
            partial = np.asarray(block_reduce(block, axis=axes, keepdims=True))
            partial = partial.reshape(tuple(block.shape[a] for a in kept))
            target = tuple(box[a] for a in kept)
            if acc is None:
                acc = np.zeros(out_shape, dtype=partial.dtype)
            acc[target] = np.where(
                filled[target], combine(acc[target], partial), partial
            )
            filled[target] = True
            if counting == "count":
                counts[target] += np.prod([block.shape[a] for a in axes])
            elif counting == "nancount":
                counts[target] += np.sum(~np.isnan(block), axis=axes).reshape(
                    partial.shape
                )
        result = acc / counts if counting else acc
        if dtype is None and counting:
            dtype = func(np.zeros(1, self.dtype)).dtype
        if dtype is not None:
            result = np.asarray(result).astype(dtype)
        if keepdims:
            full = tuple(1 if a in axes else self.shape[a] for a in range(self.ndim))
            result = np.asarray(result).reshape(full)
        elif not kept:
            result = np.asarray(result).reshape(())[()]
        return result

    def expand_dims(self, axis=0):
        """Insert a unit leading axis, staying virtual (0.2 extension).

        The legacy concat-along-a-new-dimension path
        (:meth:`xdas.DataArray.expand_dims` then :func:`xdas.concat`)
        expands the data with :func:`numpy.expand_dims`; this keeps
        that path lazy instead of materializing. Only the leading
        position is supported: the new axis holds one tile of size
        one, and the engine ``load_tile`` receives one extra leading
        ``slice(0, 1)`` per expanded axis, padding its output rank
        accordingly (see the silixa and miniseed engines).

        Parameters
        ----------
        axis : int, optional
            Position of the new axis; only ``0`` (equivalently
            ``-ndim - 1``) is supported.

        Returns
        -------
        TileArray
        """
        axis = int(axis)
        if axis == -self.ndim - 1:
            axis = 0
        if axis != 0:
            raise ValueError("only a leading axis can be virtually expanded")
        rename = {}
        for k in range(self.ndim - 1, -1, -1):
            rename[f"{TILE_PREFIX}{k}"] = f"{TILE_PREFIX}{k + 1}"
            for kind in ("sizes", "starts", "steps"):
                if f"{kind}_{k}" in self.dataset:
                    rename[f"{kind}_{k}"] = f"{kind}_{k + 1}"
        dataset = self.dataset.rename(rename)
        dataset = dataset.assign(sizes_0=(f"{TILE_PREFIX}0", np.ones(1, np.int64)))
        return type(self)(dataset, self.dtype, self.engine)

    def _expand_virtual(self, args, kwargs):
        """Dispatch ``numpy.expand_dims``, delegating to :meth:`expand_dims`."""
        kwargs = dict(kwargs)
        axis = kwargs.pop("axis", args[1] if len(args) > 1 else None)
        if kwargs or len(args) > 2 or args[0] is not self:
            return NotImplemented
        if not isinstance(axis, (int, np.integer)):
            return NotImplemented
        if int(axis) not in (0, -self.ndim - 1):
            return NotImplemented
        return self.expand_dims(0)

    def transpose(self, order):
        """Materialize and transpose to the given axis *order*."""
        return np.transpose(np.asarray(self), order)

    def astype(self, dtype, **kwargs):
        """Materialize and cast the values to *dtype*."""
        return np.asarray(self).astype(dtype, **kwargs)

    def __deepcopy__(self, memo):
        """Copy without the read cache; the dataset is immutable."""
        return type(self)(self.dataset, self.dtype, self.engine)

    def __repr__(self):
        """Summarize the array on one line, as the data of a data array.

        The shape is left out: the labeled array prints it one line
        above. What remains is what only the tiling knows — the volume
        it stands for, and how many tiles it took.
        """
        return (
            f"TileArray[{self.engine['name']}] "
            f"{_to_si(self.size * self.dtype.itemsize)} ({self.dtype}) "
            f"{self.ntiles} {'tile' if self.ntiles == 1 else 'tiles'}"
        )

    def _repr_inline_(self, max_width):
        """Return the one-line summary used by xarray inline reprs.

        Shorter than :meth:`__repr__`: an inline row already prints the
        dtype and the size, so only the tiling is left to report.
        """
        summary = (
            f"TileArray[{self.engine['name']}] "
            f"({self.ntiles} {'tile' if self.ntiles == 1 else 'tiles'})"
        )
        return summary if len(summary) <= max_width else "TileArray"


def _expand(variable, union, array):
    """Broadcast *variable* over the *union* tile dims of *array*."""
    if variable.dims == union:
        return variable
    counts = {dim: len(sizes) for dim, sizes in zip(array.dims, array._sizes)}
    values = np.asarray(variable.values)
    shape = tuple(counts[dim] if dim in variable.dims else 1 for dim in union)
    full = tuple(counts[dim] for dim in union)
    return xr.Variable(union, np.broadcast_to(values.reshape(shape), full))


def _concatenate_virtual(args, kwargs):
    """Fuse tile arrays for ``numpy.concatenate`` when possible."""
    if not args:
        return NotImplemented
    arrays, *rest = args
    if len(rest) > 1 or set(kwargs) - {"axis"}:
        return NotImplemented
    axis = kwargs.get("axis", rest[0] if rest else 0)
    try:
        arrays = list(arrays)
    except TypeError:
        return NotImplemented
    if axis is None or not all(isinstance(array, TileArray) for array in arrays):
        return NotImplemented
    try:
        return TileArray.concat(arrays, dim=axis)
    except (ValueError, IndexError):
        return NotImplemented


__all__ = [
    "TileArray",
]
