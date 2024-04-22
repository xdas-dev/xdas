# Xdas: a Python Framework for Distributed Acoustic Sensing

*Xdas* is an python library for managing, processing and visualizing **Distributed Acoustic Sensing (DAS)** data. It reads any DAS format into self-described python abstractions that encapsulates both the data and the metadata (coordinates and attributes). Xdas reuses concepts of labeled N-dimensional arrays developed by the [Xarray](https://xarray.dev) library. It takes inspiration from [Dask](https://www.dask.org/) in term of lazy computing.

**Useful links:** [Code Repository](https://github.com/xdas-dev/xdas) | [Issues](https://github.com/xdas-dev/xdas/issues) | [Discussions](https://github.com/xdas-dev/xdas/discussions) | [Newsletter](https://groups.google.com/g/xdas)

## Key Features

- Seamless manipulation of large multi-file datasets in their native DAS-specific format.
- Signal Processing: Multi-threaded implementations of common routines.
- Extensibility: build your pipeline with a mix of Xdas, NumPy/SciPy, and/or your own custom routines. 
- Larger-than-memory processing: apply your pipelines with optimized I/O latencies.

Xdas can also be used in other context than DAS, for example in the context of other dense and heavy N-dimensional arrays such as large-N seismic arrays.


````{grid} 1 2 2 2
:gutter: 4
:padding: 2 2 0 0

```{grid-item-card} Getting Started
:link: getting-started
:link-type: doc
New to *Xdas*? Check out the Getting Started Guide. It explains how install *Xdas* and how to use its most important features. 
```

```{grid-item-card} User Guide
:link: user-guide/index
:link-type: doc
The User Guide provides information on all the features of *Xdas* with explanations and examples.
```

```{grid-item-card} API Reference
:link: api/index
:link-type: doc
The reference guide contains a detailed description of the *Xdas* API. The reference describes how the methods work and which parameters can be used.
```

```{grid-item-card} Contributing to xdas
:link: contribute
:link-type: doc
Found a bug, or you want to propose new features? Follow these instructions to learn the process of improving Xdas.
```
````

## Citing xdas

If you use *Xdas* for your DAS data processing, please consider [citing the project](cite).

```{toctree}
  :maxdepth: 1
  :hidden:

getting-started
user-guide/index
api/index
contribute
cite
```
