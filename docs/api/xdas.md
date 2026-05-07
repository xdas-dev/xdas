```{eval-rst}
.. currentmodule:: xdas
```

# *xdas*

## Opening files

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   open
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

