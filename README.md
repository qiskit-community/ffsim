# ffsim

Faster simulations of fermionic quantum circuits.

## What is ffsim?

ffsim is a software library for simulating fermionic quantum circuits that preserve particle number
and the Z component of spin. This category includes many quantum circuits used for quantum chemistry simulations.
By exploiting the symmetries and using specialized algorithms, ffsim can simulate these circuits much faster
than a generic quantum circuit simulator.

**Experimental disclaimer**: ffsim is currently an experimental release. Breaking changes may be introduced without warning.

## Features

- Fast simulation of fermionic quantum circuits that preserve particle number and the Z component of spin.
  ffsim supports the simulation of basic fermionic gates and includes specialized routines for simulation
  of molecular Hamiltonians in the "double-factorized" representation.
- Compatible with PySCF. State vectors use the same indexing convention as PySCF's `fci` module.

## Limitations

- There is no support for operations that do not preserve particle number and the Z component of spin.

## Installation

### From PyPI

ffsim is available on [PyPI](https://pypi.org/project/ffsim/). It can be installed by running the command

```bash
pip install ffsim
```

### From source

Installing ffsim from source requires a Rust compiler to be present on the system.
A Rust compiler can be installed by following the instructions [here](https://www.rust-lang.org/tools/install).
Once the Rust compiler is installed, ffsim can be installed by running the command

```bash
pip install .
```

from the root directory of the code repository.

Installing from source has only been tested in a Linux environment.

#### Known issues

- It may be a bit tricky to get OpenBLAS to link successfully. If you have issues, try the following:
  - Run `pip install patchelf`.
  - Run `pip install -e .` twice in a row.

## Development

To set up ffsim for development, install it from source in editable mode along with the development requirements:

    pip install -e ".[dev]"

If you add or modify any Rust modules, rebuild them by running the command

    maturin develop --release

Tests and other code checks are managed using [tox](https://tox.wiki/en/latest/).
To run the default tox environments, simply run

    tox

To run a specific environment, for example, to run the lint checks, do

    tox run -e lint

You can also use `pytest` to run the tests directly. For example,

    pytest tests/
