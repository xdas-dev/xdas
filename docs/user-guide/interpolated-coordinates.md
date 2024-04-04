---
file_format: mystnb
kernelspec:
  name: python3
---

# Interpolated Coordinates

## Coordinate

Because data is generally sampled with a constant sampling rate/resolution keeping the 
corresponding value for each index as a dense array is inefficient. In `xdas`
coordinates are saved using the [CF convention][CF] through the 
{py:class}`xdas.Coordinate` object. Only a few tie points are kept and intermediate 
values are retrieved by linear interpolation. Discontinuities are marked by two 
consecutive tie points.

![](/_static/coordinate.svg)

## Creating a Coordinate

The {py:class}`xdas.Coordinate` constructor takes `tie_indices` and `tie_values` inputs. 
Below the code corresponding to the example showed above. 

```{code-cell}
import xdas as xd

coord = xd.Coordinate(
    {
        "tie_indices": [0, 9, 19, 20, 29],
        "tie_values": [0.0, 90.0, 190.0, 400.0, 490.0]
    }
)
coord 
```

The resulting object acts as an {py:class}`numpy.ndarray` object. Indexing and 
selecting works out of the box. Note that using a stepping, the tie points can be moved
a little bit.

```{code-cell}
coord = coord[1:-3:2]
coord
```

The main advantage of coordinates is that the enable labeled based selection. To 
retrieve the index of a value the {py:meth}`get_index` method can be used:

```{code-cell}
coord.to_index(430.0)
```

```{warning}
To be able to do labeled based selection, `tie_values` must be strictly increasing.
In other words they must not be any overlap. To deal with small overlaps, a solution
is to `simplify` the coordinate increasing the tolerance to the points the overlaps 
disappear. 
```

# Gaps and Overlaps

Gaps and Overlaps can be easily extracted:

```{code-cell}
coord.get_discontinuities()
```

While gaps represents missing data and are not problematic, overlaps usually arise from
labeling errors and should be taken care of.

Using the {py:meth}`simplify` method, the coordinate can be simplified with controlled 
accuracy using the [Ramer–Douglas–Peucker algorithm][RDP]. In this example, the second 
tie point do not provide useful information and is safely discarded. 

```{code-cell}
coord = coord.simplify(tolerance=0.0)
coord
```

# Temporal Coordinates

The main use of coordinates in *xdas* is to deal with long time series. By default 
*xdas* uses `"datetime64[us]"` dtype. Microseconds are used because to perform 
interpolation *xdas* convert `datetime64` to POSIX `float` which cannot safely 
represent timestamps with better accuracies.

```{code-cell}
import numpy as np

coord = xd.Coordinate(
    {
        "tie_indices": [0, 3600 * 100],
        "tie_values": [
            np.datetime64("2023-01-01T00:00:00"), 
            np.datetime64("2023-01-01T01:00:00"),
        ],
    }
)
coord.to_index(slice("2023-01-01T00:10:00", "2023-01-01T00:20:00"))
```

[CF]: <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#compression-by-coordinate-subsampling>
[RDP]: <https://en.wikipedia.org/wiki/Ramer–Douglas–Peucker_algorithm>