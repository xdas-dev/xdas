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
   Coordinate.shape
   Coordinate.size
   Coordinate.dim
   Coordinate.values
   Coordinate.name
```

Methods

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   Coordinate.isdim
   Coordinate.isregular
   Coordinate.equals
   Coordinate.copy
```

## AxisCoordinate

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   AxisCoordinate
```

Attributes

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   AxisCoordinate.ndim
   AxisCoordinate.empty
   AxisCoordinate.indices
   AxisCoordinate.start
   AxisCoordinate.end
```

Methods

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   AxisCoordinate.isregular
   AxisCoordinate.get_sampling_interval
   AxisCoordinate.to_regular
   AxisCoordinate.get_split_indices
   AxisCoordinate.get_discontinuities
   AxisCoordinate.get_availabilities
   AxisCoordinate.simplify
   AxisCoordinate.to_index
   AxisCoordinate.to_dataarray
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
   DenseCoordinate.to_regular
   DenseCoordinate.simplify
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
   InterpCoordinate.sampling_interval
   InterpCoordinate.tolerance
```

Methods

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   InterpCoordinate.from_block
   InterpCoordinate.to_regular
   InterpCoordinate.get_sampling_interval
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
   SampledCoordinate.to_regular
   SampledCoordinate.simplify
```

## Functions

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   get_sampling_interval
```
