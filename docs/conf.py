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

json_url = "https://xdas.readthedocs.io/en/latest/_static/switcher.json"

version_match = os.environ.get("READTHEDOCS_VERSION")
# If READTHEDOCS_VERSION doesn't exist, we're not on RTD
# If it is an integer, we're in a PR build and the version isn't correct.
# If it's "latest" → change to "dev" (that's what we want the switcher to call it)
if not version_match or version_match.isdigit() or version_match == "latest":
    # For local development, infer the version to match from the package.
    if "dev" in release or "rc" in release:
        version_match = "dev"
        # We want to keep the relative reference if we are in dev mode
        # but we want the whole url if we are effectively in a released version
        json_url = "_static/switcher.json"
    else:
        version_match = f"v{release}"
elif version_match == "stable":
    version_match = f"v{release}"

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
    "navbar_start": ["navbar-logo", "version-switcher"],
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
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

dirpath = "_data"
if not os.path.exists(dirpath):
    os.makedirs(dirpath)

da = generate()
chunks = xd.split(da, 3)
dirname = os.path.split(__file__)[0]
da.to_netcdf(os.path.join(dirname, "_data/sample.h5"))
for index, chunk in enumerate(chunks, start=1):
    if index == 2:
        chunk["time"] += np.timedelta64(3, "ms")
    chunk.to_netcdf(os.path.join(dirname, f"_data/00{index}.h5"))


data = np.random.rand(20, 10)
with h5py.File(os.path.join(dirname, "_data/other_format.hdf5"), "w") as f:
    dset = f.create_dataset("dataset", data.shape, data=data, dtype="float32")
    f["dataset"].attrs["t0"] = "2024-01-01T14:00:00.000"
    f["dataset"].attrs["dt"] = 1 / 100
    f["dataset"].attrs["dx"] = 10
