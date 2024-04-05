---
file_format: mystnb
kernelspec:
  name: python3
---

```{code-cell}
:tags: [remove-cell]

import os
import xdas as xd
os.chdir("../_data")
```

# Composing a processing sequence

```{warning}
The API of this part of xdas is still experimental.
```

The xdas library provides various routines from NumPy, SciPy, and ObsPy that have been optimized for DAS DataArray objects, and which can be incorporated in a processing pipeline. See [Process big dataarrays](processing) for an explanation of the xdas processing workflows, e.g. for bigger-than-RAM datasets. Higher-level operations (FK-filters, STA/LTA detector, etc.) can be constructed from a sequence of the elementary operations implemented in xdas. To facilitate this and other user-defined operations, xdas offers a convenient framework to create, manage, and execute a sequence of atomic operations, which we refer to as compositions.

By using compositions, built-in and user-defined processing tasks mesh seamlessly with the optimization and IO-infrastructure that xdas offers, improving the robustness and reproducibility of complex processing pipelines.

## Elementary operations: Atoms and StateAtoms

The building blocks of a processing sequence are atomic operations: small routines that feature a single operation, which can be reused in a range of scenarios. For example, `abs`, `square`, `fft`, `sum`, etc. Each atomic operation is wrapped by an `Atom` class that handles things like passing arguments, bookkeeping, and integration. For operations that rely on a state that is updated over time (i.e., they are *stateful*), a special class exists called `StateAtom`, which inherits from the `Atom` class and adds state-specific manipulations.

An `Atom` is initiated with a function `func` as its first argument, and additional keyword arguments that are passed to `func`. The `StateAtom` class requires the initial state and name of the keyword that hold the state (e.g., for `scipy.signal.sosfilt`, this argument is called `zi`). Finally, user-defined functions can be passed in the same way (assuming that the function is compatible with the xdas framework; see **page describing extension**).

To keep track of the various operations in a sequence, each atom can be assigned a name with the `name` argument. If no name is provided, it is derived from the function name that the `Atom` class wraps.

Example usage:

```{code-cell} 
import numpy as np
import xdas.signal as xp
from xdas.atoms import Partial
op1 = Partial(xp.taper, dim="time")

sos = np.zeros((6, 2))
op2 = Partial(xp.sosfilt, sos, ..., zi=..., dim="time")

my_func = lambda x: x
op3 = Partial(my_func, name="my special function")
```

## Composing a sequence

- Basic construction of `Sequence`
- Ordering operations
- Loading pre-defined sequences and manipulating

## Executing a sequence

- Data in memory
- Data in chunks
