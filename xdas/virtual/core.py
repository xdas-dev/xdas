"""Registry base class :class:`VirtualBackend` for the virtual array backends."""

from typing import ClassVar


class VirtualBackend:
    """
    Marker base and registry for the virtual array backends.

    A virtual backend is a lazy, numpy-like duck array whose values stay
    on disk: it reports ``shape`` and ``dtype`` without reading, slices
    lazily through ``__getitem__``, and materializes through
    ``__array__``; a ``from_variable`` classmethod wraps one stored
    HDF5 variable, so open paths dispatch
    ``VirtualBackend[vtype].from_variable(...)``. That contract is
    informal — the backends share no implementation and differ beyond
    it (blocking, persistence, concatenation), so this base only names
    them: subclasses register by
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
