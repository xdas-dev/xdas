```{eval-rst}
.. currentmodule:: xdas.virtual
```

# xdas.virtual

## VirtualArray

Base class for all virtual array types.

Attributes

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   VirtualArray.shape
   VirtualArray.dtype
   VirtualArray.ndim
   VirtualArray.size
   VirtualArray.nbytes
   VirtualArray.empty
```

Methods

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   VirtualArray.to_dataset
```

## VirtualSource

A lazy pointer to a single dataset inside an HDF5/NetCDF4 file.

Constructor

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   VirtualSource
```

Attributes

```{eval-rst}
.. autosummary::

   VirtualSource.vsource
   VirtualSource.shape
   VirtualSource.dtype
   VirtualSource.ndim
   VirtualSource.size
   VirtualSource.nbytes
   VirtualSource.empty
```

Methods

```{eval-rst}
.. autosummary::

   VirtualSource.to_dataset
```

## VirtualStack

A lazy concatenation of multiple {py:class}`VirtualSource` objects along one axis.

Constructor

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   VirtualStack
```

Attributes

```{eval-rst}
.. autosummary::

   VirtualStack.sources
   VirtualStack.axis
   VirtualStack.shape
   VirtualStack.dtype
   VirtualStack.ndim
   VirtualStack.size
   VirtualStack.nbytes
   VirtualStack.empty
```

Methods

```{eval-rst}
.. autosummary::

   VirtualStack.append
   VirtualStack.extend
   VirtualStack.to_dataset
```

## VirtualLayout

Internal HDF5 virtual dataset layout object.

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   VirtualLayout
```

Attributes

```{eval-rst}
.. autosummary::

   VirtualLayout.shape
   VirtualLayout.dtype
```

Methods

```{eval-rst}
.. autosummary::

   VirtualLayout.to_dataset
```

## Selection

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   Selection
```

Attributes

```{eval-rst}
.. autosummary::

   Selection.shape
   Selection.ndim
```

Methods

```{eval-rst}
.. autosummary::

   Selection.get_indexer
```
