# Frequently Asked Questions

## Why not using Xarray and Dask?

Originally, Xdas was meant to be a simple add-on to Xarray, taking advantage of its [Dask integration] (https://docs.xarray.dev/en/stable/user-guide/dask.html). But two main limitations forced us to create a parallel project:

- Coordinates have to be loaded into memory as NumPy arrays. This is prohibitive for very long time series, where storing the time coordinate as a dense array with a value for each time sample leads to metadata that in some extreme cases cannot fit in memory.
- Dask arrays become sluggish when dealing with a very large number of files. Dask is a pure Python package, and processing graphs of millions of tasks can take several seconds or more. Also, Dask does not provide a way to serialise a graph for later reuse.

Because of this, and the fact that the Xarray object was not designed to be subclassed, we decided to go our own way. Hopefully, if the progress of Xarray allows it, we could imagine merging the two projects. Xdas tries to follow the Xarray API as much as possible. 
