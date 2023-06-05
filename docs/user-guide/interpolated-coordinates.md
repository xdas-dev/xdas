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
import xdas

coord = xdas.Coordinate(
    tie_indices=[0, 9, 19, 20, 29],
    tie_values=[0.0, 90.0, 190.0, 400.0, 490.0]
)
coord 
```

In this example, the second tie point do not provide useful information. Using the
{py:meth}`simplify` method, the coordinate can be simplified without loss of accuracy:

```{code-cell}
coord = coord.simplify(tolerance=0.0)
coord
```

The resulting object acts as an {py:class}`numpy.ndarray` object. Indexing and 
selecting works out of the box:

```{code-cell}
coord = coord[1:-3:2]
coord
```

To retrieve the index of a value the {py:meth}`get_index` method can be used.

```{code-cell}
coord.get_index(430.0)
```

[CF]: <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#compression-by-coordinate-subsampling>