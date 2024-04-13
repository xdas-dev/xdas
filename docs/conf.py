# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# -- Project information -----------------------------------------------------

project = "xdas"
copyright = "2024, Alister Trabattoni"
author = "Alister Trabattoni"

# The full version, including alpha/beta/rc tags
release = "0.1rc"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_sidebars = {"getting-started": [], "contribute": [], "cite": []}

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "logo": {
        "image_light": "_static/logo-light.png",
        "image_dark": "_static/logo-dark.png",
    },
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/xdas-dev/xdas",
            "icon": "fa-brands fa-square-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/xdas",
            "icon": "fa-custom fa-pypi",
        },
    ],
}


# Configuration for intersphinx.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
}

# -- Generate dummy data -----------------------------------------------------
import os

import h5py
import numpy as np

import xdas as xd
from xdas.synthetics import generate

dirpath = os.path.join(os.path.split(__file__)[0], "_data")
if not os.path.exists(dirpath):
    os.makedirs(dirpath)

da = generate()
chunks = xd.split(da, 3)
da.to_netcdf(os.path.join(dirpath, "sample.h5"))
da.to_netcdf(os.path.join(dirpath, "sample.nc"))
for index, chunk in enumerate(chunks, start=1):
    if index == 2:
        chunk["time"] += np.timedelta64(3, "ms")
    chunk.to_netcdf(os.path.join(dirpath, f"00{index}.h5"))
    chunk.to_netcdf(os.path.join(dirpath, f"00{index}.nc"))


data = np.random.rand(20, 10)
with h5py.File(os.path.join(dirpath, "other_format.hdf5"), "w") as f:
    dset = f.create_dataset("dataset", data.shape, data=data, dtype="float32")
    f["dataset"].attrs["t0"] = "2024-01-01T14:00:00.000"
    f["dataset"].attrs["dt"] = 1 / 100
    f["dataset"].attrs["dx"] = 10
