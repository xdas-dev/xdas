---
file_format: mystnb
kernelspec:
  name: python3
---

# Interpolated Coordinates

## Coordinate

Because DAS data are generally sampled with a constant sampling rate/resolution, keeping the 
corresponding value for each index as a dense array is inefficient. *xdas* stores the
coordinates using the [CF convention][CF] through the 
{py:class}`xdas.Coordinate` object. With this method, only a few tie points are kept and intermediate 
values are retrieved by linear interpolation. Discontinuities are marked by two 
consecutive tie points, as illustrated below:

![](/_static/coordinate.svg)

The resulting coordinate vector is sparse but contains all the information
necessary to exactly recover the original, dense coordinate vector.

## Creating a Coordinate

The {py:class}`xdas.Coordinate` constructor takes `tie_indices` and `tie_values` as inputs. 
The code below corresponds with the example illustrated in the figure above: 

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
selecting works out of the box. Note that when specifying an increment step greater than 1, the tie points can be displaced a little bit.

```{code-cell}
coord = coord[1:-3:2]
coord
```

A major advantage of {py:class}`xdas.Coordinate` is that it enables label-based selection. 
For instance, to retrieve the index of a value the {py:meth}`get_index` method can be used:

```{code-cell}
coord.to_index(430.0)
```

```{warning}
To be able to do label-based selection, `tie_values` must be strictly increasing.
In other words there must not be any overlap. To deal with small overlaps, a solution
is to `simplify` the coordinates, increasing the tolerance such that the overlapping points
disappear. 
```

# Gaps and Overlaps

Gaps and Overlaps can be easily identified based on the tie point positions, and extracted with:

```{code-cell}
coord.get_discontinuities()
```

While gaps represents missing data and are not problematic, overlaps usually arise from
labeling errors and should be taken care of.

Using the {py:meth}`simplify` method, the coordinate can be simplified with controlled 
accuracy using the [Ramer–Douglas–Peucker algorithm][RDP]. In this example, the second 
tie point does not provide useful information and is safely discarded. 

```{code-cell}
coord = coord.simplify(tolerance=0.0)
coord
```

# Temporal Coordinates

```{warning}
Please rewrite the second sentence about the POSIX format: it is not clear to me
what you mean.
```

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