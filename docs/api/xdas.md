```{eval-rst}
.. currentmodule:: xdas
```

# *xdas*

## Opening files

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   open_dataarray
   open_mfdataarray
   open_mfdatatree
   open_datacollection
   open_mfdatacollection
```

## Data manipulation

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   align
   asdataarray
   combine_by_coords
   combine_by_field
   concatenate
   concatenate
   split
```

## Mathematical and statistical functions

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   cumprod
   cumsum
   all
   any
   max
   min
   argmax
   argmin
   median
   ptp
   mean
   prod
   std
   sum
   var
   percentile
   quantile
   average
   count_nonzero
   diff
```

## Data Structures

### DataArray

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   DataArray
```

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   DataArray.equals
   DataArray.get_axis_num
   DataArray.isel
   DataArray.sel
   DataArray.copy
   DataArray.rename
   DataArray.load
   DataArray.to_xarray
   DataArray.from_xarray
   DataArray.to_stream
   DataArray.from_stream
   DataArray.to_netcdf
   DataArray.from_netcdf
   DataArray.plot
```

### DataCollection

Constructor

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary
   
   DataCollection
```

Methods

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   DataCollection.query
   DataCollection.issequence
   DataCollection.ismapping
   DataCollection.from_netcdf
```

### DataMapping

Constructor

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary
   
   DataMapping
```

Methods

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   DataMapping.to_netcdf
   DataMapping.from_netcdf
   DataMapping.equals
   DataMapping.isel
   DataMapping.sel
   DataMapping.load
   DataMapping.map
```

### DataSequence

Constructor

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary
   
   DataSequence
```

Methods

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   DataSequence.to_mapping
   DataSequence.from_mapping
   DataSequence.to_netcdf
   DataSequence.from_netcdf
   DataSequence.equals
   DataSequence.isel
   DataSequence.sel
   DataSequence.load
   DataSequence.map
```

### Coordinates

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
   InterpCoordinate.format_index_slice
   InterpCoordinate.get_indexer
   InterpCoordinate.slice_indexer
   InterpCoordinate.decimate
   InterpCoordinate.simplify
   InterpCoordinate.get_discontinuities
   InterpCoordinate.from_array
   InterpCoordinate.to_dict
```