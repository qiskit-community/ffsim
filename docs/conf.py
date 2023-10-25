# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Sphinx configuration."""

import importlib.metadata

project = "ffsim"
copyright = "2023, IBM"
author = "Kevin J. Sung"
release = importlib.metadata.version("ffsim")

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "jupyter_sphinx",
    "sphinx_autodoc_typehints",
    "nbsphinx",
]

exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# HTML output options
html_theme = "alabaster"

# nbsphinx options (for tutorials)
nbsphinx_timeout = 500
nbsphinx_execute = "always"
