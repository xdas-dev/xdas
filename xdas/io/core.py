import socket


class Engine:
    _registry = {}
    _supported_vtypes = None
    _supported_ctypes = None

    def __init__(self, vtype=None, ctype=None):
        self.vtype = self._parse_vtype(vtype)
        self.ctype = self._parse_ctype(ctype)

    def __init_subclass__(cls, *, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name is not None:
            Engine._registry[name] = cls

    def __class_getitem__(cls, item):
        return cls._registry[item]

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
    _last_successful_engine = "xdas"

    def __init__(self, vtype=None, ctype=None):
        self.vtype = vtype
        self.ctype = ctype

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
