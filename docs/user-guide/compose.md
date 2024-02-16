---
file_format: mystnb
kernelspec:
  name: python3
---

```{code-cell}
:tags: [remove-cell]

import os
import xdas
os.chdir("../_data")
```

# Composing a processing sequence

```{warning}
The API of this part of xdas is still experimental.
```

## Sequential processing

Flexibility while allowing for optimization

```{code-cell} 
db = xdas.open_database("sample.nc")
db.to_xarray().plot.imshow(yincrease=False, vmin=-0.5, vmax=0.5);
```

```{code-cell} 
import scipy.signal as sp
from xdas.processing import ProcessingChain, SOSFilter

sos = sp.iirfilter(4, 0.5, btype="lowpass", output="sos")
sosfilter = SOSFilter(sos, "time")
chain = ProcessingChain([sosfilter])
out = chain.process(db, "time", 100, parallel=False)
out.to_xarray().plot.imshow(yincrease=False, vmin=-0.5, vmax=0.5);
```