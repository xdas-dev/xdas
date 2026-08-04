"""Registry base class :class:`VirtualBackend` for the virtual array backends."""

import math
from typing import ClassVar


class VirtualBackend:
    """
    Base class and registry for the virtual array backends.

    A virtual backend is a lazy, numpy-like duck array whose values
    stay on disk. The contract, declared here:

    - :attr:`shape` and :attr:`dtype` answer without reading, with
      :attr:`ndim`, :attr:`size`, :attr:`nbytes` and :attr:`empty`
      derived from them;
    - ``__getitem__`` selects lazily and ``__array__`` materializes;
    - :meth:`from_variable` wraps one stored HDF5 variable, so open
      paths dispatch ``VirtualBackend[vtype].from_variable(...)``;
    - the :meth:`create_variable`/:meth:`finalize_save` pair writes
      the backend's stored form, so save paths need no per-backend
      branch.

    Beyond the contract the backends share no implementation and
    differ freely (blocking, concatenation). Subclasses register by
    passing ``vtype=`` in the class definition and are retrieved with
    the ``VirtualBackend[vtype]`` syntax — the same registry fashion as
    :class:`~xdas.io.Engine` (by name) and
    :class:`~xdas.coordinates.Coordinate` (by ctype).

    Attributes
    ----------
    vtype : str or None
        The registered name of the backend, inherited by its subclasses
        (``VirtualSource.vtype`` is ``"hdf5"``). ``None`` on this base.
    consolidates : bool
        Whether concatenating scan products of this backend fuses them
        into one compact object, so that multi-file scans can drain
        batches and keep memory bounded. Default ``False``: an HDF5
        stack keeps one virtual mapping per source, so batching would
        free nothing.

    Examples
    --------
    >>> from xdas.virtual import TileArray, VirtualArray, VirtualBackend

    >>> VirtualBackend["tiles"] is TileArray
    True
    >>> VirtualBackend["hdf5"] is VirtualArray
    True

    >>> VirtualBackend["netcdf"]
    Traceback (most recent call last):
    KeyError: "no virtual backend registered under 'netcdf'; available: ['hdf5', 'tiles']"
    """

    _registry: ClassVar[dict] = {}
    vtype: ClassVar[str | None] = None
    consolidates: ClassVar[bool] = False

    def __init_subclass__(cls, *, vtype=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if vtype is not None:
            cls.vtype = vtype
            VirtualBackend._registry[vtype] = cls

    def __class_getitem__(cls, item):
        if item in cls._registry:
            return cls._registry[item]
        raise KeyError(
            f"no virtual backend registered under {item!r}; "
            f"available: {sorted(cls._registry)}"
        )

    # --- the contract, implemented by every backend ---

    @property
    def shape(self):
        """Tuple of array dimensions (abstract — must be overridden)."""
        raise NotImplementedError

    @property
    def dtype(self):
        """NumPy dtype of the array elements (abstract — must be overridden)."""
        raise NotImplementedError

    def __getitem__(self, key):
        """Select lazily, returning an array of the same kind (abstract)."""
        raise NotImplementedError

    def __array__(self, dtype=None, copy=None):
        """Materialize as a numpy array (abstract — must be overridden)."""
        raise NotImplementedError

    @classmethod
    def from_variable(cls, variable):
        """
        Expose one stored HDF5 variable as this backend's lazy array.

        The open half of the dispatch (abstract — each backend wraps
        its own way): a virtual source pointing at the variable for the
        hdf5 backend, a single tile covering it for the tiles backend.

        Parameters
        ----------
        variable : h5py.Dataset
            The open variable to wrap.
        """
        raise NotImplementedError

    def create_variable(self, file, name, dims=None, dtype=None):
        """
        Write this array as variable *name* of an open h5netcdf *file*.

        The first half of the persistence contract (abstract — each
        backend writes its own stored form): an HDF5 virtual dataset
        for the hdf5 backend, a placeholder variable carrying the
        engine specification for the tiles backend.
        """
        raise NotImplementedError

    def finalize_save(self, fname, group=None):
        """
        Append what of the stored form outlives the variable.

        The second half of the persistence contract, called once the
        file handle of :meth:`create_variable` is closed. Default: the
        variable is the whole stored form, nothing to append — the
        tiles backend appends its manifest as a sibling group.
        """

    # --- derived from the contract, shared by every backend ---

    @property
    def ndim(self):
        """Number of dimensions."""
        return len(self.shape)

    @property
    def size(self):
        """Total number of elements."""
        if self.shape:
            return math.prod(self.shape)
        else:
            return 0

    @property
    def nbytes(self):
        """Total number of bytes occupied by the array elements."""
        if self.shape:
            return self.size * self.dtype.itemsize
        else:
            return 0

    @property
    def empty(self):
        """``True`` if the array contains no elements."""
        return self.size == 0
