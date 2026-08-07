"""
Plugin base class :class:`Engine` for file format handlers.

Also provides :class:`AutoEngine` for format auto-detection and
:func:`get_free_port`.
"""

import socket
from typing import ClassVar

from ..virtual import VirtualBackend


class Engine:
    """
    Base class for file format handlers in xdas.

    The Engine class provides a plugin architecture for reading and writing various
    file formats. Each Engine subclass corresponds to a specific file format (e.g.,
    "xdas", "asn", "obspy") and implements methods to open and save DataArray or
    DataCollection objects.

    Engines are registered in a class-level registry using the `__init_subclass__` hook,
    allowing them to be accessed by name using the `Engine[name]` syntax. Aliases can
    also be defined for backwards compatibility or convenience.

    Parameters
    ----------
    vtype : str, optional
        The virtualization type to use. If vtype is None, the first supported type is
        used.
    ctype : str or dict, optional
        The coordinate type(s) to use. Can be:
        - None: uses the first supported ctype for each component
        - str: uses the same ctype for all components
        - dict: maps component names to their specific ctypes
        If None or incomplete, missing ctypes default to the first supported option.

    Attributes
    ----------
    vtype : str
        The version type for this engine instance.
    ctype : str or dict
        The component type(s) for this engine instance.

    Notes
    -----
    Subclasses should define class attributes:
    - `_supported_vtypes` (list): List of supported virtualization types, each
    the name of a registered :class:`~xdas.virtual.VirtualBackend`
    - `_supported_ctypes` (dict): Maps component names to lists of supported coordinate
    types

    Engines with format-specific parameters define their own `__init__` taking those
    parameters after `vtype` and `ctype` and calling `super().__init__(vtype, ctype)`.
    They are then reachable from the open functions either by configuring an instance
    or as extra keyword arguments next to the engine name.

    Examples
    --------
    Subclass registration (automatic via `__init_subclass__`):

    >>> class MyFormatEngine(Engine, name="myformat", aliases=["my"]):
    ...     _supported_vtypes = ["hdf5"]
    ...     _supported_ctypes = {
    ...         "time": ["sampled", "dense"], "distance": ["sampled", "dense"]
    ...     }
    ...     def open_dataarray(self, fname):
    ...         raise NotImplementedError

    Access registered engines:

    >>> engine = Engine["myformat"](vtype="hdf5")
    >>> engine = Engine["my"](ctype="dense")  # Using alias
    """

    _registry: ClassVar[dict] = {}
    _aliases: ClassVar[dict] = {}
    _supported_vtypes = None
    _supported_ctypes = None
    name = None

    def __init__(self, vtype=None, ctype=None):
        self.vtype = self._parse_vtype(vtype)
        self.ctype = self._parse_ctype(ctype)

    def __init_subclass__(cls, *, name=None, aliases=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name is not None:
            cls.name = name
            Engine._registry[name] = cls
        if aliases is not None:
            for alias in aliases:
                Engine._aliases[alias] = name

    def __class_getitem__(cls, item):
        if item is None:
            return AutoEngine
        elif item in cls._registry:
            return cls._registry[item]
        elif item in cls._aliases:
            return cls._registry[cls._aliases[item]]
        else:
            raise KeyError(
                f"no engine registered under {item!r}; "
                f"available: {sorted([*cls._registry, *cls._aliases])}"
            )

    def open_dataarray(self, fname):
        """Open *fname* and return a :class:`DataArray` (abstract)."""
        raise NotImplementedError

    def save_dataarray(self, da, fname, **kwargs):
        """Write *da* to *fname* (abstract)."""
        raise NotImplementedError

    def open_datacollection(self, fname):
        """Open *fname* and return a :class:`DataCollection` (abstract)."""
        raise NotImplementedError

    def save_datacollection(self, dc, fname, **kwargs):
        """Write *dc* to *fname* (abstract)."""
        raise NotImplementedError

    @staticmethod
    def load_tile(path, selection, **kwargs):
        """Read the selected sub-box of one tile of *path* (abstract).

        The decode half of the tiles machinery: called on the class by
        :class:`~xdas.virtual.TileArray` once per tile touched, with
        exactly one source-local, possibly strided :class:`slice` per
        source axis, in source order — whatever virtual arrangement
        (transposes, inserted axes) the tile array presents — and the
        manifest's engine specification (merged with the per-tile
        variables) as keyword arguments. It must return exactly the
        selected sub-box of the decoded source as a numpy array, and must
        depend only on its arguments — never on engine instance state —
        so that stored manifests decode identically everywhere.
        """
        raise NotImplementedError

    def _parse_vtype(self, vtype):
        if vtype is not None:
            if not isinstance(vtype, str):
                raise ValueError("vtype must be None or a string")
            VirtualBackend[vtype]  # fail fast on unregistered vtypes
        if self._supported_vtypes is None:
            return vtype
        if vtype is None:
            vtype = self._supported_vtypes[0]
        if vtype not in self._supported_vtypes:
            raise NotImplementedError(
                f"vtype '{vtype}' is not supported by {self.__class__.__name__}"
            )
        return vtype

    def _parse_ctype(self, ctype):
        if self._supported_ctypes is None:
            return ctype
        if ctype is None:
            ctype = {
                key: self._supported_ctypes[key][0] for key in self._supported_ctypes
            }
        elif isinstance(ctype, str):
            ctype = dict.fromkeys(self._supported_ctypes, ctype)
        elif isinstance(ctype, dict):
            ctype = {
                key: ctype.get(key, self._supported_ctypes[key][0])
                for key in self._supported_ctypes
            }
            for key in ctype:
                if ctype[key] is None:
                    ctype[key] = self._supported_ctypes[key][0]
        else:
            raise ValueError(
                "ctype must be None, str, or dict with the supported dimensions"
            )
        for key in ctype:
            if ctype[key] not in self._supported_ctypes[key]:
                raise NotImplementedError(
                    f"ctype '{ctype[key]}' for '{key}' is not supported by {self.__class__.__name__}"
                )
        return ctype


class AutoEngine(Engine):
    """
    Automatic engine dispatcher for file format detection.

    AutoEngine attempts to open a file using all registered engines in a smart order,
    making it possible to open files without explicitly specifying the file format.
    This is the default behavior when no engine is specified in `xdas.open_dataarray()`.

    The engine selection strategy is optimized for performance:
    - The last successfully used engine is tried first
    - All other registered engines are tried in their registration order
    - The first engine that successfully opens the file is used
    - If all engines fail, an informative error message is raised

    Registration order therefore settles which engine wins when several read
    the same file: `"obspy"` is registered before the legacy `"miniseed"`, and
    both after the format-specific engines.

    Parameters
    ----------
    vtype : str, optional
        The virtualization type to use. Passed to all engines during auto-detection.
        If None, each engine uses its default vtype.
    ctype : str or dict, optional
        The coordinate type(s) to use. Passed to all engines during auto-detection.
        Can be a string, dict, or None (each engine uses its default).
        Format-specific engine parameters cannot be used with auto-detection:
        they require naming a concrete engine.

    Attributes
    ----------
    vtype : str, optional
        The virtualization type for engine attempts.
    ctype : str or dict, optional
        The coordinate type(s) for engine attempts.

    Notes
    -----
    All exceptions raised by individual engines are silently caught; only if all
      engines fail is an error raised to the user.

    Examples
    --------
    >>> from xdas.io import AutoEngine
    >>> engine = AutoEngine(ctype="dense")
    >>> da = engine.open_dataarray("data.hdf5")  # doctest: +SKIP

    """

    _last_successful_engine = "xdas"

    def open_dataarray(self, fname):
        """Try each registered engine in order and return the first successful result."""
        for engine in self._ordered_engines():
            try:
                out = Engine[engine](vtype=self.vtype, ctype=self.ctype).open_dataarray(
                    fname
                )
                AutoEngine._last_successful_engine = engine
                return out
            except Exception:  # noqa: BLE001, S112 - try the next engine
                continue
        raise ValueError(self._failure_message(fname))

    def open_datacollection(self, fname):
        """Try each registered engine in order and return the first collection.

        Raises :exc:`NotImplementedError` when no engine describes *fname* as a
        collection, so that callers fall back to opening it as a data array the
        same way they do for a named engine.
        """
        for engine in self._ordered_engines():
            try:
                out = Engine[engine](
                    vtype=self.vtype, ctype=self.ctype
                ).open_datacollection(fname)
                AutoEngine._last_successful_engine = engine
                return out
            except Exception:  # noqa: BLE001, S112 - try the next engine
                continue
        raise NotImplementedError(
            self._failure_message(fname) + " as a data collection"
        )

    def _failure_message(self, fname):
        message = f"no engine could open the file '{fname}'"
        if self.ctype is not None:
            message += f" with ctype '{self.ctype}'"
        if self.vtype is not None:
            message += f" with vtype '{self.vtype}'"
        return message

    def _ordered_engines(self):
        return [self._last_successful_engine] + [
            e for e in Engine._registry if e != self._last_successful_engine
        ]


def get_free_port():
    """
    Find and return a free port on the host machine.

    This function creates a temporary socket, binds it to an available port
    provided by the host, retrieves the port number, and then closes the socket.
    This is useful for finding an available port for network communication.

    Returns
    -------
    int:
        A free port number on the host machine.

    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
