```{eval-rst}
.. currentmodule:: xdas.virtual.tiles
```

# xdas.virtual.tiles

Lazy tile-backed virtual arrays: the only backend of the formats that
HDF5 virtual datasets cannot serve (Silixa TDMS, MiniSEED), the default
one for Febus, and available on request from every other engine
(`vtype="tiles"`).

## TileArray

A dense rectilinear grid of file-backed tiles exposed as one lazy
numpy-like array.

Attributes

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   TileArray.shape
   TileArray.dtype
   TileArray.ndim
   TileArray.size
   TileArray.chunks
   TileArray.ntiles
   TileArray.engine
   TileArray.root
```

Methods

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   TileArray.from_tiles
   TileArray.from_variable
   TileArray.to_dataset
   TileArray.create_variable
   TileArray.sibling_datasets
   TileArray.concat
   TileArray.expand_dims
   TileArray.squeeze
   TileArray.transpose
   TileArray.astype
   TileArray.equals
```

## Engine lookup

Tiles are decoded by the ``load_tile`` half of the
{class}`xdas.io.Engine` format plugins; the engine names stored in tile
manifests resolve on that registry (``Engine[name]``).

```{eval-rst}
.. currentmodule:: xdas.io

.. autosummary::
   :toctree: ../_autosummary

   Engine.load_tile
```
