# User Guide

````{grid} 1 2 2 2
:gutter: 4
:padding: 2 2 0 0

```{grid-item-card} Data Structures
:link: data-structures/index
:link-type: doc
The two core objects: {py:class}`~xdas.DataArray` for a single acquisition and
{py:class}`~xdas.DataCollection` for grouping multiple acquisitions together.
```

```{grid-item-card} Coordinates
:link: coordinates/index
:link-type: doc
How array axes are mapped to physical values — interpolated, sampled, dense, and scalar
coordinate types, including handling of gaps and overlaps.
```

```{grid-item-card} I/O
:link: io/index
:link-type: doc
Supported DAS file formats, how to add a custom engine, and how virtual datasets let you
access large multi-file acquisitions as a single seamless array.
```

```{grid-item-card} Pipeline Processing
:link: pipeline/index
:link-type: doc
Building processing sequences with {py:class}`~xdas.atoms.Atom` objects and applying
them chunk-by-chunk on datasets larger than memory, including real-time streaming.
```

```{grid-item-card} How-To Guides
:link: how-to/index
:link-type: doc
Short, task-focused recipes for common domain-specific workflows such as converting
strain-rate to displacement.
```

```{grid-item-card} FAQ
:link: faq
:link-type: doc
Answers to frequently asked questions, including why *xdas* exists alongside *xarray*
and *dask*.
```
````

```{toctree}
:maxdepth: 2
:hidden:

data-structures/index
coordinates/index
io/index
pipeline/index
how-to/index
faq
```
