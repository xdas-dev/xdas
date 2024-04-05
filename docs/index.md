# Xdas documentation


*Xdas* is an open source Python library for managing, processing and visualizing **Distributed Acoustic Sensing (DAS)** data. It transforms any DAS format into a self-contained data that encapsulates both the data and coordinate metadata.
Its functionnality can be extended for other dense and heavy N-dimenional arrays such as large-N seismic arrays. It provides parsers for common 
file format (NetCDF4,HDF5), multithreaded signal processing routines and tools to manage complex datasets. *Xdas* makes a point of offering the user extensibility, allowing users to take advantage of the basic functions optimized for DAS to build their own parallelized pipelines. 
  

It uses the
[xarray](https://xarray.dev) API to work with labelled multi-dimensional arrays. 




````{grid} 1 1 2 3

```{card} Getting Started
:link: getting-started
:link-type: doc

New to *Xdas*? Check out the Getting Started Guide. It contains a guide to installing *Xdas* and a tutorial on using the most important basic functions. 
```

```{card} User Guide
:link: user-guide/index
:link-type: doc
The User Guide provides information on all the features of *Xdas* with explanations and examples.

```

```{card} API Reference
:link: api
:link-type: doc

The reference guide contains a detailed description of the *Xdas* API.
The reference describes how the methods work and which parameters can
be used.
```

```{card} Contributing to xdas
:link: contribute
:link-type: doc

Found a bug, or you want to propose new features?
Follow these instructions to merge your contributions with the xdas code base.
```

```{card} Citing xdas
:link: cite
:link-type: doc

If you use *Xdas* for your DAS data processing, please consider citing the project.
```

````



```{toctree}
  :maxdepth: 2
  :hidden:

getting-started
user-guide/index
api
contribute
cite
```
