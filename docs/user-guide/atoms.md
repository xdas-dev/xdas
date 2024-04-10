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

The xdas library provides various routines from NumPy, SciPy, and ObsPy that have been optimized for DAS DataArray objects, and which can be incorporated in a processing pipeline. See [](processing) for an explanation of the xdas processing workflows, e.g. for bigger-than-RAM datasets. Higher-level operations (FK-filters, STA/LTA detector, etc.) can be constructed from a sequence of the elementary operations implemented in xdas. To facilitate this and other user-defined operations, xdas offers a convenient framework to create and execute a (nested) sequences of atomic operations. By using sequences, built-in and user-defined processing tasks mesh seamlessly with the optimization and IO-infrastructure that xdas offers, improving the robustness and reproducibility of complex processing pipelines.

## Chaining elementary operations (atoms)

There are three "flavours" declaring the atoms that can be used to compose a sequence, illustrated by the following example:

```{code-cell} 
import numpy as np
import xdas
import xdas.signal as xp
from xdas.atoms import Partial, Sequential, IIRFilter

sequence = Sequential(
    [
      xp.taper(..., dim="time"),
      Partial(np.square),
      IIRFilter(order=4, cutoff=1.5, btype="highpass", dim="time"),
    ]
)
sequence
```

In the snippet above, we define our `sequence` as an instance of the `Sequential` class, which contains three operations. The first operation applies a Tukey taper along the time dimension, encoded by the xdas implementation of the SciPy library routines (`xdas.signal`). Since this functions takes a data array as the first argument, we use `...` as a placeholder. 

The second operation in this sequence is defined by the `square` operation built into NumPy. Since this function is not imported directly from xdas, using `...` as a placeholder won't work. This is where `Partial` comes in: wrapping `Partial` around `np.square` would be equivalent to `np.square(...)`, effectively converting an arbitrary routine into an xdas routine and inserting a placeholder as the first argument (to be substituted with a data array later).

The last operation, `IIRFilter`, instantiates a specific class dedicated to chunked execution. It inherits from the `Atom` class, which handles the logic of initialising and passing around state objects (like the filter state). This allows us to process our data one chunk at a time, without explicitly having to handle state updates and transfer.

## Executing a sequence

Once the processing sequence has been defined, it can operate on data in memory by simply calling the sequence with the data array as the argument:

```{code-cell} 
from xdas.synthetics import generate

da = generate()
result = sequence(da)
result.plot(yincrease=False)
```

The same sequence can be re-used, so it only needs to be defined once.

For executing a sequence on chunked data (e.g., larger-than-memory data sets), see the next section: [](processing.md).

## Defining custom atoms

The `Partial` method is a convenient wrapper for simple functions that take an xdas DataArray as the first argument, which covers a lot of cases. However, more complex routines, particularly those that rely on a state, will require a more explicit treatment. Such operations can be subclassed from the `Atom` base class, and adhere to the following structure:

```{code-cell} 
from xdas.atoms import Atom, State

class MyStatefulRoutine(Atom):

  def __init__(self, a, b, c=10):
    super().__init__()
    # Set class-specific parameters
    self.a = a
    self.b = b
    self.c = c
    # Define the state variable (if needed)
    self.state = State(...)

  def initialize(self, da, **kwargs):
    # Initialize state based on DataArray ``da``
    ...
  
  def call(self, da, **kwargs):
    # Apply routine to DataArray ``da``
    ...
```

