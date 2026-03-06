---
file_format: mystnb
kernelspec:
  name: python3
---

# Coordinates


{py:class}`~xdas.DataArray` is the base class in *xdas*. It is mainly composed of a N-dimensional array and a set of {py:class}`~xdas.Coordinate` objects that are gathered in a {py:class}`~xdas.Coordinates` dict-like object that can be accessed by the `DataArray.coords` attribute. Xdas comes with several flavors of {py:class}`~xdas.Coordinate` objects.

| Type | Description | `name` | `data` |
|:---|:---|:---:|:---:|
| {py:class}`~xdas.coordinates.ScalarCoordinate` | Used to label 0D dimensions | `scalar` | `{"value": any}` |
| {py:class}`~xdas.coordinates.DefaultCoordinate` | Each value is equal to its index | `default` | `{"size": int}` |
| {py:class}`~xdas.coordinates.DenseCoordinate` | Each index is mapped to a given value | `dense` | `array-like[any]` |
| {py:class}`~xdas.coordinates.InterpCoordinate` | Values are interpolated linearly between tie points | `interpolated` | `{"tie_indices": array-like[int], "tie_values": array-like[any]}` |
| {py:class}`~xdas.coordinates.SampledCoordinate` | Values are given as a multiple of a fixed sampling interval and several start values | `sampled` | `{"tie_values": array-like[any], "tie_indices": array-like[int], "sampling_interval": any}` |

In the current state of the documentation, most coordinate information can be found on the [Interpolated Coordinates](interpolated-coordinates) page.

## Per type information

```{toctree}
:maxdepth: 1

interpolated-coordinates
sampled-coordinates
```

<!-- ```{code-cell}
import xdas as xd
da = xd.DataArray([0, 1, -1, 0, -1, 2], coords={"time": [.0, .1, .2, .3, .4, .5]})
da
```

In *xdas* a {py:class}`~xdas.Coordinate` is an object that maps indices to values. Inversely `~xdas.Coordinate` object allow to retreive the index of a given value allowing labeled base selection. 

```{code-cell}
# index-based selection updates the time coordinate object
da.isel(time=slice(1,3))
```

```{code-cell}
# label-based selection find the corresponding indices to keep
da.sel(time=slice(0.1, 0.3))
``` -->