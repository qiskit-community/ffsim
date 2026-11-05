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
import inspect
import os
import sys
from pathlib import Path

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
    "sphinx.ext.linkcode",
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

# Source code links


def determine_github_ref() -> str:
    """Determine the Git reference to use for GitHub source links."""
    if sha := os.environ.get("GITHUB_SHA"):
        return sha
    return os.environ.get("GITHUB_REF_NAME", "main")


GITHUB_REF = determine_github_ref()


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """Determine the URL corresponding to Python source code."""
    if domain != "py":
        return None

    module_name = info.get("module")
    fullname = info.get("fullname")
    if not module_name or not fullname:
        return None

    module = sys.modules.get(module_name)
    if module is None:
        return None

    obj = module
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    while hasattr(obj, "__wrapped__"):
        obj = obj.__wrapped__

    if isinstance(obj, property) and obj.fget is not None:
        obj = obj.fget

    if isinstance(obj, (staticmethod, classmethod)):
        obj = obj.__func__

    try:
        source_file = inspect.getsourcefile(obj)
    except (TypeError, OSError):
        return None

    if source_file is None:
        return None

    ffsim_mod = sys.modules.get("ffsim")
    if ffsim_mod is None:
        return None

    pkg_file = getattr(ffsim_mod, "__file__", None)
    if not isinstance(pkg_file, (str, os.PathLike)):
        return None

    try:
        source_path = Path(source_file).resolve()
        pkg_dir = Path(pkg_file).resolve().parent
    except (OSError, RuntimeError):
        return None

    if not source_path.is_relative_to(pkg_dir):
        return None

    rel_path = source_path.relative_to(pkg_dir).as_posix()

    if inspect.ismodule(obj):
        linespec = ""
    else:
        try:
            source, lineno = inspect.getsourcelines(obj)
            linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
        except (OSError, TypeError):
            linespec = ""

    return f"https://github.com/qiskit-community/ffsim/blob/{GITHUB_REF}/python/ffsim/{rel_path}{linespec}"
