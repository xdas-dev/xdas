import os
from copy import copy, deepcopy
from tempfile import TemporaryDirectory

import h5py
import numpy as np


class VirtualArray:
    def __repr__(self):
        return f"{self.__class__.__name__}: {_to_human(self.nbytes)} ({self.dtype})"

    def __getitem__(self, key):
        NotImplemented

    def __array__(self, dtype=None):
        NotImplemented

    @property
    def shape(self):
        NotImplemented

    @property
    def dtype(self):
        NotImplemented

    def to_dataset(self, file_or_group, name):
        NotImplemented

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        if self.shape:
            return np.prod(self.shape)
        else:
            return 0

    @property
    def empty(self):
        return self.size == 0

    @property
    def nbytes(self):
        if self.shape:
            return self.size * self.dtype.itemsize
        else:
            return 0


class VirtualStack(VirtualArray):
    def __init__(self, sources=[], axis=0):
        self._sources = list()
        self._axis = axis
        self._shape = ()
        self.extend(sources)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            indexers = [
                key,
            ]
        else:
            indexers = list(key)
        if self._axis in range(len(indexers)):
            div_points = np.cumsum(
                [0] + [source.shape[self._axis] for source in self._sources]
            )
            indexer = indexers[self._axis]
            if isinstance(indexer, int):
                idx = indexer if indexer >= 0 else indexer + self.shape[self._axis]
                if idx < 0 or idx >= self.shape[self._axis]:
                    raise IndexError(
                        f"index {idx} is out of bounds for axis {self._axis} "
                        f"with size {self.shape[self._axis]}"
                    )
                isrc = np.searchsorted(div_points, idx, side="right") - 1
                irel = idx - div_points[isrc]
                indexers[self._axis] = irel
                sources = [self._sources[isrc][tuple(indexers)]]
            if isinstance(indexer, slice):
                start, stop, step = indexer.indices(self.shape[self._axis])
                if not step == 1:
                    raise NotImplementedError(
                        "cannot make stepped slicing along stacked dimension."
                    )
                isrc_start, isrc_stop = (
                    np.searchsorted(div_points[:-1], [start, stop], side="right") - 1
                )
                irel_start, irel_stop = (
                    np.array([start, stop]) - div_points[[isrc_start, isrc_stop]]
                )
                sources = []
                if isrc_start == isrc_stop:
                    indexers[self._axis] = slice(irel_start, irel_stop)
                    sources = [self._sources[isrc_start][tuple(indexers)]]
                else:
                    sources = []
                    indexers[self._axis] = slice(irel_start, None)
                    sources.append(self._sources[isrc_start][tuple(indexers)])
                    indexers[self._axis] = slice(None)
                    sources.extend(
                        [
                            source[tuple(indexers)]
                            for source in self._sources[isrc_start + 1 : isrc_stop]
                        ]
                    )
                    indexers[self._axis] = slice(None, irel_stop)
                    sources.append(self._sources[isrc_stop][tuple(indexers)])
        else:
            sources = [source[tuple(indexers)] for source in self._sources]
        return VirtualStack(sources, self._axis)

    def __array__(self, dtype=None):
        if not self._sources:
            raise ValueError("no sources in stack")
        return self._to_layout().__array__(dtype)

    @property
    def sources(self):
        return self._sources

    @property
    def axis(self):
        return self._axis

    @property
    def shape(self):
        return tuple(
            (
                sum(source.shape[self._axis] for source in self._sources)
                if axis == self._axis
                else size
            )
            for axis, size in enumerate(self._shape)
        )

    @property
    def dtype(self):
        if not hasattr(self, "_dtype"):
            raise AttributeError("empty stack has no dtype")
        return self._dtype

    def append(self, source):
        if not self._sources:
            self._initialize(source)
        self._check(source)
        self._sources.append(source)

    def extend(self, sources):
        if not isinstance(sources, list):
            raise TypeError("`sources` must be a list")
        for source in sources:
            self.append(source)

    def to_dataset(self, file_or_group, name):
        self._to_layout().to_dataset(file_or_group, name)

    def _initialize(self, source):
        self._shape = tuple(
            None if axis == self._axis else size
            for axis, size in enumerate(source.shape)
        )
        self._dtype = source.dtype

    def _check(self, source):
        if not isinstance(source, VirtualSource):
            raise TypeError("only `VirtualSource` object can be provided")
        if not (source.dtype == self.dtype):
            raise ValueError("all sources must share the same dtype")
        if not all(
            True if size_self is None else size_self == size_other
            for size_self, size_other in zip(self._shape, source.shape)
        ):
            raise ValueError(
                "all sources must share the same shape "
                "except along the concatenation axis"
            )

    def _to_layout(self):
        layout = VirtualLayout(self.shape, self.dtype)
        ndim = self.ndim
        index = 0
        for source in self._sources:
            slc = tuple(
                (
                    slice(index, index := index + source.shape[self._axis])
                    if axis == self._axis
                    else slice(None)
                )
                for axis in range(ndim)
            )
            layout[slc] = source
        return layout


class VirtualLayout(VirtualArray):
    """
    A lazy array layout pointing toward multiple netCDF4/HDF5 files.

    Instantiate this class with the final shape of the virtual array. Then Fill it up
    by assigning to slices of it VirtualSource objects.

    Once the data assignement is completed, the layout can be virually written into a
    `h5py.File` or `h5py.Group` object using the `to_dataset` method.

    The VirtualLayout can be sliced and the selected data accessed without writting the
    layout to disk. To that end, use `numpy.asarray` or the `__array__` special method.
    Note that for now, sliced VirtualLayout cannot be written to disk.

    Parameters
    ----------
    shape: tuple of int
        The shape of the layout.
    dtype: str of dtype
        The dtype of the layout.
    maxshape: tuple of int or None, optional
        The layout is resizable up to this shape. Use None for axes you want to
        be unlimited.
    filename: str, optional
        The name of the destination file, if known in advance. Mappings from
        data in the same file will be stored with filename '.', allowing the
        file to be renamed later.

    Attributes
    ----------
    shape: tuple of int or
        The shape of the layout.
    dtype: dtype
        The dtype of the layout.
    ndim: int
        The number of dimensions of the layout.
    nbytes: int
        The number of bytes virtually linked into the layout.

    Methods
    -------
    to_dataset(file_or_group, name)
        Writes virtually the layout into the specified HDF5 file of group with the
        given name.

    """

    def __init__(self, shape, dtype, maxshape=None, filename=None):
        self._layout = h5py.VirtualLayout(shape, dtype, maxshape, filename)
        self._sel = Selection(self._layout.shape)

    def __array__(self, dtype=None):
        with TemporaryDirectory() as tmpdirname:
            fname = os.path.join(tmpdirname, "vds.h5")
            with h5py.File(fname, "w") as file:
                dataset = self.to_dataset(file, "__values__")
            with h5py.File(fname, "r") as file:
                dataset = file["__values__"]
                out = np.asarray(dataset[self._sel.get_indexer()])
            if dtype is not None:
                out = out.astype(dtype)
        return out

    def __getitem__(self, key):
        self = copy(self)
        self._sel = self._sel.__getitem__(key)
        return self

    def __setitem__(self, key, value):
        if not self._sel._whole:
            raise NotImplementedError(
                "cannot link VirtualSources to a sliced VirtualLayout"
            )
        if isinstance(value, VirtualSource):
            value = value.vsource
        self._layout.__setitem__(key, value)

    @property
    def shape(self):
        return self._sel.shape

    @property
    def dtype(self):
        return self._layout.dtype

    def to_dataset(self, file_or_group, name):
        if np.issubdtype(self.dtype, np.integer):
            fillvalue = np.iinfo(self.dtype).min
        elif np.issubdtype(self.dtype, np.floating):
            fillvalue = np.nan
        elif np.issubdtype(self.dtype, np.complexfloating):
            fillvalue = np.nan + 1j * np.nan
        else:
            fillvalue = None
        return file_or_group.create_virtual_dataset(
            name, self._layout, fillvalue=fillvalue
        )


class VirtualSource(VirtualArray):
    """
    A lazy array object pointing toward a netCDF4/HDF5 file.

    At creation the array corresponds to an entire file dataset. It can then be
    sliced to indicate which regions should be used. Sliced VirtualSource eventually can
    be assigned to a VirtualLayout to

    Best practive is to pass it a `h5py.Dataset` obtain destructuring a `h5py.File`.
    Otherwise the exact filename, dataset name, shape and dtype must be passed.

    The data can be accessed using `numpy.asarray` or the `__array__` special method.

    Parameters
    ----------
    path_or_dataset: str or h5py.Dataset
        The path to a file, or an h5py dataset. If a dataset is given,
        no other parameters are allowed, as the relevant values are taken from
        the dataset instead.
    name: str, optional
        The name of the source dataset within the file.
    shape: tuple of int, optional
        A tuple giving the shape of the dataset.
    dtype: dtype or str, optional
        Numpy dtype or string.
    maxshape: tuple or int or None, optional
        The source dataset is resizable up to this shape. Use `None` for
        axes you want to be unlimited.

    Attributes
    ----------
    vsource: h5py.VirtualSource
        The underlying sliced virtual source
    shape: tuple of int or
        The shape of the source.
    dtype: dtype
        The dtype of the source.
    ndim: int
        The number of dimensions of the source.
    nbytes: int
        The number of bytes virtually linked into the source.

    Methods
    -------
    to_dataset(file_or_group, name)
        Puts the source into a layout and writes it virtually into the specified HDF5
        file of group with the given name.

    Examples
    --------
    >>> import os
    >>> from tempfile import TemporaryDirectory

    >>> import h5py
    >>> import numpy as np

    >>> from xdas.virtual import VirtualSource

    >>> with TemporaryDirectory() as tmpdir: # doctest:+ELLIPSIS
    ...     shape = (2, 3, 5)
    ...     data = np.arange(np.prod(shape)).reshape(*shape)
    ...     with h5py.File(os.path.join(tmpdir, "source.h5"), "w") as file:
    ...         file.create_dataset("data", data.shape, data.dtype, data)
    ...         source = VirtualSource(file["data"])  # we both write and get source here
    ...     source = source[1:-1]  # the source can be sliced
    ...     result = np.asarray(source)
    ...     assert np.array_equal(result, data[1:-1])
    <...>
    """

    def __init__(
        self, path_or_dataset, name=None, shape=None, dtype=None, maxshape=None
    ):
        self._vsource = h5py.VirtualSource(
            path_or_dataset, name, shape, dtype, maxshape
        )
        self._sel = Selection(self._vsource.shape)

    def __getitem__(self, key):
        self = copy(self)
        self._sel = self._sel.__getitem__(key)
        return self

    def __array__(self, dtype=None):
        return self._to_layout().__array__(dtype)

    @property
    def vsource(self):
        return self._vsource.__getitem__(self._sel.get_indexer())

    @property
    def shape(self):
        return self.vsource.shape

    @property
    def dtype(self):
        return self.vsource.dtype

    def to_dataset(self, file_or_group, name):
        self._to_layout().to_dataset(file_or_group, name)

    def _to_layout(self):
        layout = VirtualLayout(self.shape, self.dtype)
        layout[...] = self
        return layout


class Selection:
    """
    Used to perform lazy selection.

    It is usefull when dealing with lazy array to avoid loading unneccessary data.
    It must be initialized with the shape of the underlying array. It allows to track
    the succesive slice or single element selections made along the different
    dimensions of the array. Once the overall selection must be aaplied, the
    `get_indexer` method can be called to retreive a tuple of slice and/or int that
    can be applied to the array.

    Parameters
    ----------
    shape: tuple of int
        The shape of the array to slice lazily.

    Attributes
    ----------
    shape: tuple of int
        The shape of the array after selection is applied.
    ndim: int
        The dimensionality of the array after selection is applied.

    Methods
    -------
    get_indexer()
        Retrun a tuple of slice or int that can be applied to the array to effectively
        perform the selection

    Examples
    --------
    >>> import numpy as np
    >>> from xdas.virtual import Selection

    `Selection` is meant to be used on lazy arrays but here we show an example on a
    in-memory array for simplicity:

    >>> arr = np.arange(2 * 3 * 5).reshape(2, 3, 5)

    Successive indexing can be computed on the flight:

    >>> expected = arr[0][:, 1:-1][::2]

    Or it can be delayed up to the point where the user is done with indexing:

    >>> sel = Selection(arr.shape)
    >>> sel = sel[0][:, 1:-1][::2]  # Successive selections
    >>> slc = sel.get_indexer()
    >>> assert np.array_equal(arr[slc], expected)

    The shape of the the resulting array can be known in advance:

    >>> assert sel.shape == expected.shape

    """

    def __init__(self, shape):
        self._shape = shape
        self._selectors = Selectors([SliceSelector(size) for size in shape])
        self._whole = True

    def __getitem__(self, key):
        sel = deepcopy(self)
        sel._whole = False
        if not isinstance(key, tuple):
            key = (key,)
        dim = 0
        for k in key:
            while isinstance(sel._selectors[dim], SingleSelector):
                dim += 1
            sel._selectors[dim] = sel._selectors[dim][k]
            dim += 1
        return sel

    @property
    def shape(self):
        return tuple(
            len(selector)
            for selector in self._selectors
            if not isinstance(selector, SingleSelector)
        )

    @property
    def ndim(self):
        return len(self.shape)

    def get_indexer(self):
        return tuple(selector.get_indexer() for selector in self._selectors)


class Selectors(list):
    """Selector list that raises informative error on out of bound indexing."""

    def __getitem__(self, key):
        if key >= len(self):
            raise IndexError(
                f"too many indices for selection: selection is {len(self)}-dimensional"
            )
        return super().__getitem__(key)


class SingleSelector:
    """
    A lazy single element selector.

    Parameters
    ----------
    index: int
        The index to select.

    """

    def __init__(self, index):
        self._index = index

    def get_indexer(self):
        return self._index


class SliceSelector:
    """
    A lazy slice selection.

    The actual selection is stored as a `range` object that is transformed to a `slice`
    object when calling `get_indexer`.

    Parameters
    ----------
    size: int
        The size of the axis of the data to slice lazily.

    """

    def __init__(self, size):
        self._range = range(size)

    def __getitem__(self, key):
        try:
            item = self._range[key]
        except IndexError:
            raise IndexError(
                f"index {key} is out of bounds for axis with size {len(self)}"
            )
        if isinstance(item, int):
            return SingleSelector(item)
        else:
            sel = copy(self)
            sel._range = item
            return sel

    def __len__(self):
        return len(self._range)

    def get_indexer(self):
        if len(self) == 0:
            return slice(0)
        elif self._range.stop < 0:
            return slice(self._range.start, None, self._range.step)
        else:
            return slice(self._range.start, self._range.stop, self._range.step)


def _to_human(size):
    """Convert raw byte numbers into a human readable ones with units."""
    unit = {0: "B", 1: "KB", 2: "MB", 3: "GB", 4: "TB"}
    n = 0
    while size > 1024:
        size /= 1024
        n += 1
    return f"{size:.1f}{unit[n]}"
