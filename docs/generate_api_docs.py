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
- docs/api/ffsim.<submodule>.rst for each public submodule, and for each
  nested submodule, such as ffsim.qiskit.nonconserving, that its parent package
  lists in __all__

The section headings in ffsim.rst are read from the first line of each
submodule's docstring, and the remainder of the docstring is included as a
paragraph. To change a heading or description, edit the corresponding module
docstring and re-run this script.
"""

import ast
import pkgutil
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
FFSIM_PKG = REPO_ROOT / "python" / "ffsim"
DOCS_API_DIR = REPO_ROOT / "docs" / "api"


def module_path(submodule: str) -> Path:
    """Return the source file path for ffsim.<submodule>.

    submodule may be dotted, for example "qiskit.nonconserving".
    """
    base = FFSIM_PKG.joinpath(*submodule.split("."))
    pkg_path = base / "__init__.py"
    mod_path = base.with_suffix(".py")
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
    heading, body = get_module_docstring(submodule)
    # Nested submodules get their own pages, so they are not listed as symbols
    child_modules = set(exported_child_modules(submodule))
    symbols = [
        sym for sym in read_symbols(module_path(submodule)) if sym not in child_modules
    ]

    title = f"ffsim.{submodule}"
    underline = "=" * len(title)
    entries = "\n".join(f"   {sym}" for sym in symbols)
    paragraph = f"\n{body}\n" if body else ""

    output_path = DOCS_API_DIR / f"ffsim.{submodule}.rst"
    output_path.write_text(
        f"{title}\n"
        f"{underline}\n"
        f"\n"
        f"{heading}\n"
        f"{paragraph}\n"
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


def exported_child_modules(package: str) -> list[str]:
    """Return the submodules that ffsim.<package> imports and lists in __all__.

    A submodule counts only if it is imported as a module, as in
    "from ffsim.<package> import <submodule>", and not if a symbol with the same
    name is imported from it. package is a dotted name relative to ffsim.
    """
    path = module_path(package)
    if path.name != "__init__.py":
        return []
    tree = ast.parse(path.read_text())
    symbols = set(read_symbols(path))
    children = {mod.name for mod in pkgutil.iter_modules([str(path.parent)])}
    return sorted(
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == f"ffsim.{package}"
        for alias in node.names
        if alias.name in children and (alias.asname or alias.name) in symbols
    )


def nested_submodules(package: str) -> list[str]:
    """Return the exported submodules of ffsim.<package>, recursively."""
    result = []
    for name in exported_child_modules(package):
        submodule = f"{package}.{name}"
        result.append(submodule)
        result.extend(nested_submodules(submodule))
    return result


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
    # package that are not re-exported in ffsim's __init__.py, along with nested
    # submodules that their parent packages list in __all__.
    exported_submodules = {submodule for submodule, _ in sections}
    submodules = []
    for mod in sorted(pkgutil.iter_modules([str(FFSIM_PKG)]), key=lambda m: m.name):
        if mod.name.startswith("_") or mod.name in exported_submodules:
            continue
        submodules.append(mod.name)
        submodules.extend(nested_submodules(mod.name))

    # Generate API docs. Remove the existing directory first so that autosummary
    # stubs for removed symbols don't linger and break the build.
    shutil.rmtree(DOCS_API_DIR, ignore_errors=True)
    DOCS_API_DIR.mkdir(parents=True)
    generate_ffsim_rst(sections)
    for submodule in submodules:
        generate_submodule_rst(submodule)
    generate_index_rst(submodules)


if __name__ == "__main__":
    main()
