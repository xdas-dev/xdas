```{eval-rst}
.. currentmodule:: xdas.coordinates
```
# xdas.coordinates

## Coordinates

Constructor


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
   Coordinates.to_dict
   Coordinates.copy
   Coordinates.drop_dims 
   Coordinates.drop_coords 
```

### Coordinate

Constructor

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
   Coordinate.values
```

Methods

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   Coordinate.to_index
   Coordinate.isscalar
   Coordinate.isdense
   Coordinate.isinterp
```


### ScalarCoordinate

Constructor

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   ScalarCoordinate
```

Methods

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   ScalarCoordinate.isvalid
   ScalarCoordinate.equals
   ScalarCoordinate.to_index
   ScalarCoordinate.to_dict
```

### DenseCoordinate

Constructor

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   DenseCoordinate
```

Methods

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   DenseCoordinate.isvalid
   DenseCoordinate.index
   DenseCoordinate.get_indexer
   DenseCoordinate.slice_indexer
   DenseCoordinate.to_dict
```

### InterpCoordinate

Constructor

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
   InterpCoordinate.empty
   InterpCoordinate.dtype
   InterpCoordinate.ndim
   InterpCoordinate.shape
   InterpCoordinate.indices
   InterpCoordinate.values
```

Methods

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   InterpCoordinate.isvalid
   InterpCoordinate.equals
   InterpCoordinate.get_value
   InterpCoordinate.format_index
   InterpCoordinate.slice_index
   InterpCoordinate.get_indexer
   InterpCoordinate.slice_indexer
   InterpCoordinate.decimate
   InterpCoordinate.simplify
   InterpCoordinate.get_discontinuities
   InterpCoordinate.from_array
   InterpCoordinate.to_dict
```


### SampledCoordinate

Constructor

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
   SampledCoordinate.empty
   SampledCoordinate.dtype
   SampledCoordinate.ndim
   SampledCoordinate.shape
   SampledCoordinate.indices
   SampledCoordinate.values
```

Methods

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   SampledCoordinate.concat
   SampledCoordinate.decimate
   SampledCoordinate.equals
   SampledCoordinate.from_array
   SampledCoordinate.from_block
   SampledCoordinate.get_indexer
   SampledCoordinate.get_sampling_interval
   SampledCoordinate.get_split_indices
   SampledCoordinate.get_value
   SampledCoordinate.isvalid
   SampledCoordinate.simplify
   SampledCoordinate.slice_index
   SampledCoordinate.slice_indexer
   SampledCoordinate.to_dict
```