import socket


class Engine:
    """
    Base class for file format handlers in xdas.

    The Engine class provides a plugin architecture for reading and writing various
    file formats. Each Engine subclass corresponds to a specific file format (e.g.,
    "xdas", "asn", "miniseed") and implements methods to open and save DataArray or
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
    - `_supported_vtypes` (list): List of supported virtualization types
    - `_supported_ctypes` (dict): Maps component names to lists of supported coordinate
    types

    Examples
    --------
    Subclass registration (automatic via `__init_subclass__`):

    >>> class NetCDFEngine(Engine, name="netcdf", aliases=["nc"]):
    ...     _supported_vtypes = ["hdf5"]
    ...     _supported_ctypes = {
    ...         "time": ["sampled", "dense"], "distance": ["sampled", "dense"]
    ...     }
    ...     def open_dataarray(self, fname, **kwargs):
    ...         ...

    Access registered engines:

    >>> engine = Engine["netcdf"](vtype="hdf5")
    >>> engine = Engine["nc"](ctype="dense")  # Using alias
    """

    _registry = {}
    _aliases = {}
    _supported_vtypes = None
    _supported_ctypes = None

    def __init__(self, vtype=None, ctype=None):
        self.vtype = self._parse_vtype(vtype)
        self.ctype = self._parse_ctype(ctype)

    def __init_subclass__(cls, *, name=None, aliases=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name is not None:
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
            raise KeyError(f"Item '{item}' not found in registry or aliases")

    def open_dataarray(self, fname, **kwargs):
        raise NotImplementedError

    def save_dataarray(self, da, fname, **kwargs):
        raise NotImplementedError

    def open_datacollection(self, fname, **kwargs):
        raise NotImplementedError

    def save_datacollection(self, dc, fname, **kwargs):
        raise NotImplementedError

    def _parse_vtype(self, vtype):
        if self._supported_vtypes is None:
            return vtype
        if vtype is None:
            vtype = self._supported_vtypes[0]
        elif isinstance(vtype, str):
            pass
        else:
            raise ValueError("vtype must be None or a string")
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
            ctype = {key: ctype for key in self._supported_ctypes}
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

    Parameters
    ----------
    vtype : str, optional
        The virtualization type to use. Passed to all engines during auto-detection.
        If None, each engine uses its default vtype.
    ctype : str or dict, optional
        The coordinate type(s) to use. Passed to all engines during auto-detection.
        Can be a string, dict, or None (each engine uses its default).

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

    def open_dataarray(self, fname, **kwargs):
        for engine in self._ordered_engines():
            try:
                out = Engine[engine](vtype=self.vtype, ctype=self.ctype).open_dataarray(
                    fname, **kwargs
                )
                AutoEngine._last_successful_engine = engine
                return out
            except Exception:
                continue
        message = f"no engine could open the file '{fname}'"
        if self.ctype is not None:
            message += f" with ctype '{self.ctype}'"
        if self.vtype is not None:
            message += f" with vtype '{self.vtype}'"
        raise ValueError(message)

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
