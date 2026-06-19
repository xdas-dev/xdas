# Frequently Asked Questions

## Why not using Xarray and Dask?

Originally, Xdas was meant to be a simple add-on to Xarray, taking advantage of its [Dask integration](https://docs.xarray.dev/en/stable/user-guide/dask.html). But two main limitations forced us to create a parallel project:

- Coordinates have to be loaded into memory as NumPy arrays. This is prohibitive for very long time series, where storing the time coordinate as a dense array with a value for each time sample leads to metadata that in some extreme cases cannot fit in memory.
- Dask arrays become sluggish when dealing with a very large number of files. Dask is a pure Python package, and processing graphs of millions of tasks can take several seconds or more. Also, Dask does not provide a way to serialise a graph for later reuse.

Because of this, and the fact that the Xarray object was not designed to be subclassed, we decided to go our own way. Hopefully, if the progress of Xarray allows it, we could imagine merging the two projects. Xdas tries to follow the Xarray API as much as possible.

## Which coordinate type should I use for my time axis?

Use {py:class}`~xdas.coordinates.SampledCoordinate` when your acquisition has a constant
sampling rate (even if there are gaps between files). It is the most compact representation
and maps directly to the block time model used in miniSEED / SEED.

Use {py:class}`~xdas.coordinates.InterpCoordinate` when the sampling rate itself varies
within a single acquisition, or when the data has been GPS-corrected and the timestamps
are not strictly uniform.

See the [](coordinates/sampled-coordinates.md) and
[](coordinates/interpolated-coordinates.md) pages for details.

## My virtual dataset returns NaN values. What is going on?

NaN values in a virtual dataset almost always mean one of two things:

1. **Files have moved or been deleted.** The virtual dataset only stores pointers. If the
   pointed-to files are no longer at the recorded path, HDF5 silently returns NaN.
2. **Too many files are open simultaneously.** The HDF5 C library has a [known
   limit](https://forum.hdfgroup.org/t/virtual-datasets-and-open-file-limit/6757) on the
   number of concurrently open files. Raise the system limit with `ulimit -n <large number>`
   or load smaller slices of data.

## How do I fix gaps and overlaps between files?

Small timing errors (e.g. NTP drift) often create sub-sample overlaps between consecutive
files. Use the `simplify` method on the time coordinate to merge nearly-contiguous segments
within a given tolerance:

```python
import numpy as np
tolerance = np.timedelta64(30, "ms")  # typically enough for NTP-synced experiments
da["time"] = da["time"].simplify(tolerance)
```

Larger overlaps or gaps require manual inspection. See [](coordinates/interpolated-coordinates.md)
for the `get_discontinuities` method.

## What is the difference between `xd.open`, `xd.open_dataarray`, and `xd.open_mfdataarray`?

- {py:func}`xdas.open` — the recommended entry point. It auto-detects the file format and
  dispatches to the appropriate lower-level function based on the path pattern (single
  file, glob, or field template).
- {py:func}`xdas.open_dataarray` — opens a single file (or a previously saved virtual
  dataset file) and returns a {py:class}`~xdas.DataArray`.
- {py:func}`xdas.open_mfdataarray` — opens multiple files matching a pattern and
  concatenates them along the time axis into a single {py:class}`~xdas.DataArray`.

In practice you almost never need to call `open_dataarray` or `open_mfdataarray` directly.

## My filter produces different results when applied chunk by chunk. Why?

Recursive (IIR) filters are stateful: each output sample depends on previous input and
output samples. When you split data into chunks and apply the filter independently to
each chunk, the state is re-initialised at every boundary and the transient response
distorts the result near each chunk edge.

Use the stateful atom equivalents from {py:mod}`xdas.atoms` (e.g.
{py:class}`~xdas.atoms.IIRFilter`, {py:class}`~xdas.atoms.LFilter`) inside a
{py:class}`~xdas.atoms.Sequential` pipeline. These atoms carry the filter state across
chunk boundaries automatically when used with {py:func}`~xdas.processing.process`.

## Can I use xdas with seismic data that is not DAS?

Yes. The data model is generic: a {py:class}`~xdas.DataArray` can represent any
labeled N-dimensional array. The [](io/miniseed.md) page shows a complete example with
a large-N seismic array stored as miniSEED files. All signal processing routines in
{py:mod}`xdas.signal` and {py:mod}`xdas.fft` work on any DataArray regardless of the
physical quantity it represents.

## How do I convert a xdas DataArray to/from xarray?

```python
# xdas → xarray
xr_da = da.to_xarray()

# xarray → xdas
xd_da = xd.DataArray.from_xarray(xr_da)
```

Note that the coordinate representation is simplified during the round-trip: *xarray*
always uses dense coordinate arrays, so a `SampledCoordinate` or `InterpCoordinate` will
be converted to a {py:class}`~xdas.coordinates.DenseCoordinate` when going through
*xarray*.
