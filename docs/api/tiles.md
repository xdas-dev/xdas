```{eval-rst}
.. currentmodule:: xdas.tiles
```

# xdas.tiles

Lazy tile-backed virtual arrays, the backend of the formats that HDF5
virtual datasets cannot serve (Silixa TDMS, MiniSEED).

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
   TileArray.attrs
```

Methods

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   TileArray.from_dataset
   TileArray.to_dataset
   TileArray.concat
   TileArray.expand_dims
   TileArray.equals
```

## Engine registry

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   Engine
   Engine.open
   Engine.load
   extract_array
```
