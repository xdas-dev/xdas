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

# Process big dataarrays

```{warning}
The API of this part of xdas is still experimental.
```

## Sequential processing

In most cases, computation must be computed chunk by chunk, the result of a previous 
chunk being required to start the computation of the following one. This prevent full
parallelization of the processing. Yet, some degree of parallelization can be achieved 
when several filters are applied. As soon that a chunk is processed by a filter, the 
next chunks can be processed by the same filter but also the next filter can be applied 
to the same chunk. In addition the reading and writing operation this allows sufficient 
parallelization in most cases (the processing speed reaching the i/o speed).

![](/_static/processing.svg)

```{code-cell} 
da = xd.open_dataarray("sample.nc")
da.plot(yincrease=False, vmin=-0.5, vmax=0.5);
```