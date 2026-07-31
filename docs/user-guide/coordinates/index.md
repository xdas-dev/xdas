---
file_format: mystnb
kernelspec:
  name: python3
---

# Coordinates

{py:class}`~xdas.DataArray` combines an N-dimensional array with a set of
{py:class}`~xdas.coordinates.Coordinate` objects gathered in a
{py:class}`~xdas.coordinates.Coordinates` dict-like container, accessible via
`DataArray.coords`.  Each coordinate labels one axis (or attaches scalar
metadata) and supports both integer-index access and label-based selection.

*xdas* ships four concrete coordinate types:

| Type | Description | `name` | `data` |
|:---|:---|:---:|:---|
| {py:class}`~xdas.coordinates.ScalarCoordinate` | Scalar metadata, not tied to any axis | `scalar` | scalar-like |
| {py:class}`~xdas.coordinates.DenseCoordinate` | One stored value per element | `dense` | `array-like` |
| {py:class}`~xdas.coordinates.InterpCoordinate` | Piecewise-linear from tie points | `interpolated` | `{"tie_indices": array-like[int], "tie_values": array-like}` plus optional `"sampling_interval"` and `"tolerance"` scalars |
| {py:class}`~xdas.coordinates.SampledCoordinate` | Uniform grid with optional gaps | `sampled` | `{"tie_values": array-like, "tie_lengths": array-like[int], "sampling_interval": scalar}` |

The three axis-mapping types (`DenseCoordinate`, `InterpCoordinate`,
`SampledCoordinate`) share the {py:class}`~xdas.coordinates.AxisCoordinate`
base, which defines the index/label selection contract. `ScalarCoordinate`
carries a single value with no axis and implements only the thin
{py:class}`~xdas.coordinates.Coordinate` interface. Use
`isinstance(coord, AxisCoordinate)` to test whether a coordinate labels an axis.

## Creating coordinates

{py:class}`~xdas.coordinates.Coordinate` acts as a factory: it inspects the
shape and structure of `data` and returns the correct subclass automatically.

```{code-cell}
import numpy as np
import xdas as xd

# DenseCoordinate — one stored value per index
xd.Coordinate([0.0, 500.0, 1000.0, 1500.0])
```

```{code-cell}
# InterpCoordinate — piecewise-linear between tie points
xd.Coordinate({"tie_indices": [0, 999], "tie_values": [0.0, 5000.0]})
```

```{code-cell}
# SampledCoordinate — uniform sampling with a fixed interval
xd.Coordinate(
    {"tie_values": [0.0], "tie_lengths": [1000], "sampling_interval": 5.0}
)
```

```{code-cell}
# ScalarCoordinate — a single metadata value (not an axis)
xd.Coordinate(42.0)
```

Subclasses can also be instantiated directly when you need the specific type
explicitly:

```{code-cell}
from xdas.coordinates import SampledCoordinate

SampledCoordinate(
    {"tie_values": [0.0, 600.0], "tie_lengths": [100, 100], "sampling_interval": 5.0}
)
```

## Coordinates in a DataArray

Coordinates are attached to a {py:class}`~xdas.DataArray` through the `coords`
argument.  A dimensional coordinate shares its name with the dimension it
labels; a non-dimensional coordinate (or a `ScalarCoordinate`) can use a
different name.

```{code-cell}
da = xd.DataArray(
    data=np.zeros((1000, 500)),
    coords={
        "time": {
            "tie_values": [np.datetime64("2024-01-01T00:00:00", "ms")],
            "tie_lengths": [1000],
            "sampling_interval": np.timedelta64(4, "ms"),
        },
        "distance": {"tie_indices": [0, 499], "tie_values": [0.0, 9980.0]},
        "network": (None, "DAS-NET"),
    },
)
da
```

## Per-type details

```{toctree}
:maxdepth: 1

interpolated-coordinates
sampled-coordinates
```
