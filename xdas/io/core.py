import socket


class Engine:
    _registry = {}

    def __init_subclass__(cls, *, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name is not None:
            Engine._registry[name] = cls

    def __class_getitem__(cls, item):
        return cls._registry[item]

    @staticmethod
    def open_dataarray(fname, **kwargs):
        raise NotImplementedError

    @staticmethod
    def save_dataarray(da, fname, **kwargs):
        raise NotImplementedError


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


def parse_ctype(ctype):
    if ctype is None:
        ctype = {
            "time": "interpolated",
            "distance": "interpolated",
        }
    elif isinstance(ctype, str):
        ctype = {
            "time": ctype,
            "distance": ctype,
        }
    elif isinstance(ctype, dict):
        ctype = {
            "time": ctype.get("time", "interpolated"),
            "distance": ctype.get("distance", "interpolated"),
        }
    else:
        raise ValueError(
            "ctype must be None, str, or dict with 'time' and/or 'distance' keys"
        )
    return ctype
