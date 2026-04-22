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
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "nbsphinx",
    "qiskit_sphinx_theme",
]

exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# HTML output options
html_theme = "qiskit-ecosystem"
html_title = f"{project} {release}"
html_theme_options = {
    "source_repository": "https://github.com/qiskit-community/ffsim/",
    "source_branch": "main",
    "source_directory": "docs/",
    "sidebar_qiskit_ecosystem_member": True,
    "light_logo": "logo.svg",
    "dark_logo": "logo-dark.svg",
}

# HTML static files
html_static_path = ["_static"]
html_favicon = "_static/logo.svg"

# autosummary options
autosummary_generate = True
templates_path = ["_templates"]

# intersphinx options
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "qiskit": ("https://quantum.cloud.ibm.com/docs/api/qiskit", None),
}

# nbsphinx options (for tutorials)
nbsphinx_timeout = 300
nbsphinx_execute = "always"
