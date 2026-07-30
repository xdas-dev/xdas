"""Tile engine registry — the format plugin socket of :mod:`xdas.tiles`.

A tile engine is a subclass of :class:`Engine`, one per format,
registered by subclassing with a ``name``. It carries a ``load`` half
that reads one tile of a source file; the :class:`~xdas.tiles.TileArray`
read path looks it up by the ``name`` key of its engine specification.
This registry is distinct from :class:`xdas.io.Engine`, which handles
whole-file opening and saving of labeled arrays.

Ported from the 0.3 line (``xdas/virtual/registry.py``).
"""

ENGINES = {}


class Engine:
    """Base class of the tile format engines; subclassing registers.

    ``class MyEngine(Engine, name="myformat")`` registers the subclass
    in :data:`ENGINES` under *name* (omit it for unregistered
    intermediate bases). An engine implements one or both halves as
    static methods — a format only referenced by stored manifests needs
    only ``load``:

    - ``open(path, **kwargs)``: read only the metadata of one file and
      return a lazy tile-backed array. Unused by the 0.2 line, where
      the :class:`xdas.io.Engine` subclasses do the opening; kept for
      forward compatibility with the 0.3 stack.
    - ``load(path, selection, **params)``: read one tile — open the
      source itself (h5py, obspy, ...) and return exactly the selected
      sub-box of the decoded source as a numpy array, *selection* being
      one source-local, possibly strided :class:`slice` per source
      axis. The keyword parameters are the manifest's engine
      specification merged with the per-tile manifest variables (a
      per-tile value shadows a same-named spec constant).
    """

    name = None

    def __init_subclass__(cls, /, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name is not None:
            cls.name = name
            ENGINES[name] = cls

    @classmethod
    def open(cls, path, **kwargs):
        """Scan *path* lazily; overridden by engines that open files."""
        raise NotImplementedError(
            f"engine {cls.name!r} cannot open files (no `open` method)"
        )

    @classmethod
    def load(cls, path, selection, **params):
        """Read one tile of *path*; overridden by engines that load data."""
        raise NotImplementedError(
            f"engine {cls.name!r} cannot load tile data (no `load` method)"
        )


__all__ = [
    "ENGINES",
    "Engine",
]
