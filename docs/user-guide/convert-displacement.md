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

# Convert to displacement

Most DAS interrogators provides strain-rate. For sufficiently rectilinear deployments 
it is more natural to convert strain-rate to displacement. Read this [article][REF] for 
more details.

First open some data and convert it to xarray format.

```{code-cell} 
strain_rate = xd.open_dataarray("sample.nc")
strain_rate.plot(yincrease=False, vmin=-0.5, vmax=0.5);
```

Then convert strain rate to deformation and then to displacement.

```{code-cell} 
import xdas.signal as xs

strain = xs.integrate(strain_rate, dim="time")
deformation = xs.integrate(strain, dim="distance")
displacement = xs.sliding_mean_removal(deformation, wlen=2000.0, dim="distance")
displacement.plot(yincrease=False, vmin=-0.5, vmax=0.5);
```

[REF]: <https://doi.org/10.31223/X5ZD3C>