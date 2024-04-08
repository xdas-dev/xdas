```{eval-rst}
.. currentmodule:: xdas
```

# *xdas*

## Top-level functions

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   open_dataarray
   open_mfdataarray
   open_datacollection
   open_mfdatacollection
   concatenate
```

## Manipulation routines

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   open_dataarray
   open_mfdataarray
   open_mfdatatree
   open_datacollection
   open_mfdatacollection
   asdataarray
   combine_by_field
   combine_by_coords
   concatenate
   split
```

## Mathematical and statistical functions

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

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
   :toctree: _autosummary

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

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   DataCollection.query
   DataCollection.issequence
   DataCollection.ismapping
   DataCollection.from_netcdf
```

### DataMapping

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   DataMapping.to_netcdf
   DataMapping.from_netcdf
   DataMapping.equals
   DataMapping.isel
   DataMapping.sel
   DataMapping.load
   DataMapping.map
```

### DataSequence

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

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

### Coordinate

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   Coordinate.dtype
   Coordinate.ndim
   Coordinate.shape
   Coordinate.values
   Coordinate.equals
   Coordinate.to_index
   Coordinate.isscalar
   Coordinate.isdense
   Coordinate.isinterp
```

### Coordinates

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   Coordinates.isdim
   Coordinates.get_query
   Coordinates.to_index
   Coordinates.equals
   Coordinates.to_dict
   Coordinates.copy
   Coordinates.drop 
```
### ScalarCoordinate

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   ScalarCoordinate.isvlid
   ScalarCoordinate.equals
   ScalarCoordinate.to_index
   ScalarCoordinate.to_dict
```

### DenseCoordinate

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   DenseCoordinate.isvlid
   DenseCoordinate.index
   DenseCoordinate.get_indexer
   DenseCoordinate.slice_indexer
   DenseCoordinate.to_dict
```

### InterpCoordinate

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   InterpCoordinate.isvlid
   InterpCoordinate.tie_indices
   InterpCoordinate.tie_values
   InterpCoordinate.empty
   InterpCoordinate.dtype
   InterpCoordinate.ndim
   InterpCoordinate.shape
   InterpCoordinate.indices
   InterpCoordinate.values
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