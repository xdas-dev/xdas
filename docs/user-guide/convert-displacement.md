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

# Convert to displacement

Most DAS interrogators provides strain-rate. For sufficiently rectilinear deployments 
it is more natural to convert strain-rate to displacement. Read this [article][REF] for 
more details.

First open some data and convert it to xarray format.

```{code-cell} 
db = xdas.open_database("sample.nc")
strain_rate = db.to_xarray()
strain_rate.plot.imshow(yincrease=False, vmin=-0.5, vmax=0.5);
```

Then convert strain rate to deformation and then to displacement.

```{code-cell} 
import xdas.signal as xp

strain = xp.integrate(strain_rate, dim="time")
deformation = xp.integrate(strain, dim="distance")
displacement = xp.sliding_mean_removal(deformation, wlen=2000.0)
displacement.plot.imshow(yincrease=False, vmin=-0.5, vmax=0.5);
```

[REF]: <https://doi.org/10.31223/X5ZD3C>