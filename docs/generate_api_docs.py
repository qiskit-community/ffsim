# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Generate API docs for the ffsim package.

Run this script after adding or removing symbols from the ffsim package:

    python docs/generate_api_docs.py

This generates:
- docs/api/index.rst (API index page)
- docs/api/ffsim.rst from python/ffsim/__init__.py
- docs/api/ffsim.<submodule>.rst for each public submodule

The section headings in ffsim.rst are read from the first line of each
submodule's docstring, and the remainder of the docstring is included as a
paragraph. To change a heading or description, edit the corresponding module
docstring and re-run this script.
"""

import ast
import pkgutil
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
FFSIM_PKG = REPO_ROOT / "python" / "ffsim"
DOCS_API_DIR = REPO_ROOT / "docs" / "api"


def module_path(submodule: str) -> Path:
    """Return the source file path for ffsim.<submodule>."""
    pkg_path = FFSIM_PKG / submodule / "__init__.py"
    mod_path = FFSIM_PKG / f"{submodule}.py"
    if pkg_path.exists():
        return pkg_path
    if mod_path.exists():
        return mod_path
    raise FileNotFoundError(f"Cannot find module file for ffsim.{submodule}")


def read_symbols(path: Path) -> list[str]:
    """Return the list of names defined in __all__ at the given path."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    return [
                        elt.value
                        for elt in node.value.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ]
    return []


def get_module_docstring(submodule: str) -> tuple[str, str]:
    """Return (heading, body) from a submodule's docstring.

    heading is the first line with trailing period stripped.
    body is the remainder of the docstring after the first line (stripped).
    """
    tree = ast.parse(module_path(submodule).read_text())
    docstring = ast.get_docstring(tree)
    if not docstring:
        raise ValueError(f"Module ffsim.{submodule} has no docstring")
    lines = docstring.splitlines()
    heading = lines[0].rstrip(".")
    body = "\n".join(lines[1:]).strip()
    return heading, body


def autosummary_section(heading: str, body: str, symbols: list[str]) -> str:
    underline = "-" * len(heading)
    entries = "\n".join(f"   {sym}" for sym in symbols)
    paragraph = f"\n{body}\n" if body else ""
    return f"{heading}\n{underline}\n{paragraph}\n.. autosummary::\n   :toctree: stubs/\n\n{entries}\n"  # noqa: E501


def generate_ffsim_rst(sections: list[tuple[str, list[str]]]) -> None:
    """Generate docs/api/ffsim.rst."""
    title = "ffsim"
    underline = "=" * len(title)
    body = "\n".join(
        autosummary_section(*get_module_docstring(submodule), symbols)
        for submodule, symbols in sections
    )
    output_path = DOCS_API_DIR / "ffsim.rst"
    output_path.write_text(
        f"{title}\n{underline}\n\n.. currentmodule:: ffsim\n\n{body}"
    )
    print(f"Written {output_path}")


def generate_submodule_rst(submodule: str) -> None:
    """Generate docs/api/ffsim.<submodule>.rst."""
    heading, _ = get_module_docstring(submodule)
    symbols = read_symbols(module_path(submodule))

    title = f"ffsim.{submodule}"
    underline = "=" * len(title)
    entries = "\n".join(f"   {sym}" for sym in symbols)

    output_path = DOCS_API_DIR / f"ffsim.{submodule}.rst"
    output_path.write_text(
        f"{title}\n"
        f"{underline}\n"
        f"\n"
        f"{heading}\n"
        f"\n"
        f".. currentmodule:: ffsim.{submodule}\n"
        f"\n"
        f".. autosummary::\n"
        f"   :toctree: stubs/\n"
        f"\n"
        f"{entries}\n"
    )
    print(f"Written {output_path}")


def generate_index_rst(submodules: list[str]) -> None:
    """Generate docs/api/index.rst."""
    toctree_entries = "\n".join(f"   ffsim.{sub}" for sub in submodules)
    rows = [
        "   * - :doc:`ffsim`\n"
        "     - Top-level module where ffsim's main functions and classes are exposed"
    ]
    rows += [
        f"   * - :doc:`ffsim.{sub}`\n     - {get_module_docstring(sub)[0]}"
        for sub in submodules
    ]
    table_rows = "\n".join(rows)

    output_path = DOCS_API_DIR / "index.rst"
    output_path.write_text(
        f"API reference\n"
        f"=============\n"
        f"\n"
        f".. toctree::\n"
        f"   :maxdepth: 1\n"
        f"   :hidden:\n"
        f"\n"
        f"   ffsim\n"
        f"{toctree_entries}\n"
        f"\n"
        f".. list-table::\n"
        f"   :widths: auto\n"
        f"\n"
        f"{table_rows}\n"
    )
    print(f"Written {output_path}")


def main() -> None:
    tree = ast.parse((FFSIM_PKG / "__init__.py").read_text())

    # Collect (submodule, symbols) groups in source order.
    # Only process "from ffsim.<submodule> import ..." nodes.
    sections: list[tuple[str, list[str]]] = []
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if not module.startswith("ffsim."):
            continue
        submodule = module[len("ffsim.") :]
        symbols = [alias.name for alias in node.names]
        sections.append((submodule, symbols))

    # Detect separately-documented submodules: public submodules of the ffsim
    # package that are not re-exported in ffsim's __init__.py.
    exported_submodules = {submodule for submodule, _ in sections}
    standalone_submodules = sorted(
        mod.name
        for mod in pkgutil.iter_modules([str(FFSIM_PKG)])
        if not mod.name.startswith("_") and mod.name not in exported_submodules
    )

    # Generate API docs
    DOCS_API_DIR.mkdir(parents=True, exist_ok=True)
    generate_ffsim_rst(sections)
    for submodule in standalone_submodules:
        generate_submodule_rst(submodule)
    generate_index_rst(standalone_submodules)


if __name__ == "__main__":
    main()
