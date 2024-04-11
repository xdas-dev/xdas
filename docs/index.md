# Xdas documentation

*Xdas* is an open source Python library for managing, processing and visualizing **Distributed Acoustic Sensing (DAS)** data. It transforms any DAS format into a self-contained data that encapsulates both the data and coordinate metadata.

**Main features**:

- Composability of operations: build your own data processing pipeline with a mix of xdas, NumPy/SciPy, and custom routines.
- Numerous multi-threaded signal processing routines and tools to manage complex datasets, including I/O latency hiding.
- Convenient data selection across multiple files, with automatic interpolation of small data gaps
- Parsers for common file formats (NetCDF4,HDF5) and DAS manufacturers (ASN, Febus, OptaSense, Sintela).
- Extendability to other dense and heavy N-dimenional arrays such as large-N seismic arrays.

It uses the
[xarray](https://xarray.dev) API to work with labelled multi-dimensional arrays. 


````{grid} 1 2 2 2
:gutter: 4
:padding: 2 2 0 0

```{grid-item-card} Getting Started
:link: getting-started
:link-type: doc

New to *Xdas*? Check out the Getting Started Guide. It contains a guide to installing *Xdas* and a tutorial on using the most important basic functions. 
```

```{grid-item-card} User Guide
:link: user-guide/index
:link-type: doc
The User Guide provides information on all the features of *Xdas* with explanations and examples.

```

```{grid-item-card} API Reference
:link: api/index
:link-type: doc

The reference guide contains a detailed description of the *Xdas* API.
The reference describes how the methods work and which parameters can
be used.
```

```{grid-item-card} Contributing to xdas
:link: contribute
:link-type: doc

Found a bug, or you want to propose new features?
Follow these instructions to merge your contributions with the xdas code base.
```

````

**Citing xdas**

If you use *Xdas* for your DAS data processing, please consider 
[citing the project](cite).


```{toctree}
  :maxdepth: 2
  :hidden:

getting-started
user-guide/index
api/index
contribute
cite
```
