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

sys.path.insert(0, os.path.abspath("../xdas"))


# -- Project information -----------------------------------------------------

project = "xdas"
copyright = "2023, Alister Trabattoni"
author = "Alister Trabattoni"

# The full version, including alpha/beta/rc tags
release = "0.1a1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
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
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = dict(
    use_repository_button=True,
    repository_url="https://github.com/xdas-dev/xdas",
)

# -- Generate dummy data -----------------------------------------------------
import os

import numpy as np
import xdas

dirpath = "_data"
if not os.path.exists(dirpath):
    os.makedirs(dirpath)

shape = (6000, 1000)
resolution = (np.timedelta64(10, "ms"), 5.0)
starttime = np.datetime64("2023-01-01T00:00:00")


db = xdas.Database(
    data=np.random.randn(*shape),
    coords={
        "time": xdas.Coordinate(
            tie_indices=[0, shape[0] - 1],
            tie_values=[starttime, starttime + resolution[0] * (shape[0] - 1)],
        ),
        "distance": xdas.Coordinate(
            tie_indices=[0, shape[1] - 1],
            tie_values=[0.0, resolution[1] * (shape[1] - 1)],
        ),
    },
)
db.to_netcdf("_data/sample.nc")
