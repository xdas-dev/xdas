# Data Structure

```{warning}
The whole section is rather technical, and these details are not very relevant
for a casual user. I propose to split the User Guide in a *Beginner Guide* and
an *Expert Guide*.
```

```{warning}
Does `xarray.DataArray` need to be discussed here? It doesn't seem very relevant...
```

*xdas* leverages two kinds of data structures. The workhorse data structures are the 
{py:class}`xarray.DataArray` and {py:class}`xdas.DataArray` objects, which are 
N-dimensional labeled array object. The second, more abstract data structures are the 
{py:class}`xarray.Dataset` and {py:class}`xdas.DataCollection` objects that are 
mappings of DataArray and DataArray objects respectively. `DataArray`s are intended for
single contiguous datasets, whereas `DataCollection`s combine multiple `DataArray`s (or
nested `DataCollection`s) to facilitate operations across contiguous blocks.

The *xdas* objects extend their equivalent *xarray* counterpart in term of huge multi-file datasets handling, but 
at the cost of a limited subset of the *xarray* API. When manipulating data that are 
small enough to be stored in-memory, the full capabilities of *xarray* objects are available, and is 
generally preferred. In a near future, we hope that the extra functionalities provided 
by the *xdas* objects will be part of the *xarray* library to enjoy the best of both 
words.

The *xarray* objects structure description can be found in the 
[xarray documentation](https://docs.xarray.dev/en/stable/user-guide/data-structures.html). 
Here, we focus on the *xdas* objects. They follow the same philosophy as
*xarray* objects, so reading the *xarray* documentation is a good start to use *xdas*.

```{toctree}
:maxdepth: 1

dataarray
datacollection
```