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

# Recover displacement using deformation

Most DAS interrogators provides strain-rate. For sufficiently rectilinear deployments 
it is more natural to convert strain-rate to displacement. Read this [article][REF] for 
more details.

First open some data and convert it to xarray format.

```{code-cell} 
db = xdas.open_database("sample.nc")
strain_rate = db.to_xarray()
```

Then convert strain rate to deformation.

```{code-cell} 
import xdas.signal as xp

strain = xp.integrate(strain_rate, dim="time")
deformation = xp.integrate(strain, dim="time")
displacement = xp.sliding_mean_removal(deformation, wlen=250.0)
```

And finally convert deformation to displacement.

```{code-cell} 
displacement = xp.sliding_mean_removal(deformation, wlen=250.0)
```

[REF]: <https://doi.org/10.31223/X5ZD3C>