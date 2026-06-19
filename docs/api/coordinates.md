```{eval-rst}
.. currentmodule:: xdas.coordinates
```
# xdas.coordinates

## Coordinates

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   Coordinates
```

Methods

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   Coordinates.isdim
   Coordinates.get_query
   Coordinates.to_index
   Coordinates.equals
   Coordinates.copy
   Coordinates.drop_dims
   Coordinates.drop_coords
```

## Coordinate

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   Coordinate
```

Attributes

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   Coordinate.dtype
   Coordinate.ndim
   Coordinate.shape
   Coordinate.size
   Coordinate.empty
   Coordinate.dim
   Coordinate.indices
   Coordinate.values
   Coordinate.start
   Coordinate.end
   Coordinate.name
```

Methods

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   Coordinate.isscalar
   Coordinate.isdim
   Coordinate.equals
   Coordinate.to_index
   Coordinate.format_index
   Coordinate.slice_indexer
   Coordinate.copy
   Coordinate.to_dataarray
```

## ScalarCoordinate

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   ScalarCoordinate
```

## DenseCoordinate

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   DenseCoordinate
```

Attributes

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   DenseCoordinate.index
```

Methods

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   DenseCoordinate.from_block
   DenseCoordinate.get_sampling_interval
   DenseCoordinate.get_div_points
```

## InterpCoordinate

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   InterpCoordinate
```

Attributes

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   InterpCoordinate.tie_indices
   InterpCoordinate.tie_values
```

Methods

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   InterpCoordinate.from_block
   InterpCoordinate.get_sampling_interval
   InterpCoordinate.get_split_indices
   InterpCoordinate.get_discontinuities
   InterpCoordinate.get_availabilities
   InterpCoordinate.simplify
```

## SampledCoordinate

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   SampledCoordinate
```

Attributes

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   SampledCoordinate.tie_values
   SampledCoordinate.tie_lengths
   SampledCoordinate.tie_indices
   SampledCoordinate.sampling_interval
```

Methods

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   SampledCoordinate.from_block
   SampledCoordinate.get_sampling_interval
   SampledCoordinate.get_split_indices
   SampledCoordinate.get_discontinuities
   SampledCoordinate.get_availabilities
   SampledCoordinate.simplify
```
