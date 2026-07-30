"""Engine lookup of the tiles machinery.

The 0.2 line keeps a single per-format plugin socket: :class:`xdas.io.Engine`.
Tile decoding is its ``load_tile`` half; :func:`get_engine` resolves the
``name`` key of the engine specifications stored in tile manifests against
that registry. The 0.3 line hosts the same lookup over its own registry,
keeping :mod:`xdas.tiles.tilearray` identical in both lines.
"""


def get_engine(name):
    """Return the engine class registered under *name*.

    Parameters
    ----------
    name : str
        The ``name`` key of a tile manifest's engine specification.

    Returns
    -------
    type
        The :class:`xdas.io.Engine` subclass registered under *name*; its
        ``load_tile`` static method decodes tiles of that format.

    Raises
    ------
    KeyError
        If no engine is registered under *name*.
    """
    from ..io.core import Engine

    try:
        return Engine[name]
    except KeyError:
        raise KeyError(
            f"no engine registered under {name!r}; "
            f"available: {sorted(Engine._registry)}"
        ) from None


__all__ = [
    "get_engine",
]
